#!/usr/bin/env python3
"""
gallery_server.py  —  unified image gallery
--------------------------------------------
Combines two sources:
  • Label Studio  → qwen3_age_results.json   (images served from S3)
  • Grafana       → data/disagree_pool.json  (images served from /img/ locally)

Usage:
    python3 gallery_server.py          # port 7823
    python3 gallery_server.py --port 7825
"""

import json
import os
import re
import math
import sqlite3
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

try:
    import lightgbm as _lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

BASE_DIR    = Path(__file__).resolve().parent
DB_PATH     = BASE_DIR / "gallery.db"
LS_FILE     = BASE_DIR / "qwen3_age_results.json"
POOL_FILE   = BASE_DIR / "data" / "disagree_pool.json"
IMAGES_DIR  = BASE_DIR / "data" / "disagree_images"
PORT        = 7823


def _db_connect():
    # DB_PATH is on a mounted filesystem — WAL is unreliable there.
    # We work on a /tmp copy and sync back on commit via _db_flush().
    import shutil
    tmp = Path('/tmp/gallery_work.db')
    if not tmp.exists() or tmp.stat().st_mtime < DB_PATH.stat().st_mtime:
        shutil.copy2(DB_PATH, tmp)
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def _db_flush():
    """Copy /tmp working copy back to the mounted path atomically, with dated backup."""
    import shutil
    from datetime import datetime
    tmp = Path('/tmp/gallery_work.db')
    if not tmp.exists():
        return
    # Rotate backup: keep last 3 dated copies
    backup_dir = BASE_DIR / 'backups'
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(tmp, backup_dir / f'gallery_{stamp}.db')
    # Keep only last 3 backups
    backups = sorted(backup_dir.glob('gallery_*.db'))
    for old in backups[:-3]:
        old.unlink(missing_ok=True)
    shutil.copy2(tmp, DB_PATH)

# ── eval data files ───────────────────────────────────────────────────────────
# Active versions in gallery:
#   v6   — legacy baseline, scored from pre-computed JSON eval files
#   v8   — V8pas80-v2 (current production in d2911d10bb), scored INLINE from siglip2_details
#   v11  — V11s80 (BCI + hard-neg + slim 80, candidate next gen), scored INLINE
#
# V4 / V5 / V10 were removed 2026-05-22 as obsolete.
EVAL_FILES = {
    "v6": [
        BASE_DIR / "data" / "eval_v6_db_all.json",          # full DB coverage (V6 LGBM)
        BASE_DIR / "data" / "eval_v6_g1g2_results.json",
        BASE_DIR / "data" / "eval_v6_g3456_results.json",
    ],
}

LGBM_THRESHOLD   = 0.30   # V6 default (was 0.80 legacy; aligned with V8/V11 for direct comparison)
V6_LGBM_THRESHOLD  = 0.30
V8_LGBM_THRESHOLD  = 0.30  # V8pas80-v2 production threshold
V11_LGBM_THRESHOLD = 0.30  # V11s80 production threshold
MINOR_THRESHOLD  = 0.72

# ── V5/V6 LGBM (lazy-loaded from JS files) ───────────────────────────────────
_V5_LGBM_CACHE: dict | None = None
_V6_LGBM_CACHE: dict | None = None

def _load_v5_lgbm() -> dict | None:
    global _V5_LGBM_CACHE
    if _V5_LGBM_CACHE is not None:
        return _V5_LGBM_CACHE
    js_path = BASE_DIR / "data" / "lgbm_evaluate_v5_piper.js"
    if not js_path.exists():
        return None
    try:
        js = js_path.read_text(encoding="utf-8")
        features = json.loads(re.search(r'const LGBM_FEATURES = (\[.*?\]);', js, re.DOTALL).group(1))
        trees    = json.loads(re.search(r'const LGBM_TREES = (\[.*?\]);',    js, re.DOTALL).group(1))
        feat_idx = {f: i for i, f in enumerate(features)}
        _V5_LGBM_CACHE = {"features": features, "trees": trees, "feat_idx": feat_idx}
        print(f"  V5 LGBM loaded: {len(features)} features, {len(trees)} trees")
    except Exception as e:
        print(f"  V5 LGBM load error: {e}")
        return None
    return _V5_LGBM_CACHE

def _v5_lgbm_score(siglip2_details: dict) -> float | None:
    """Apply V5-new LGBM to siglip2_details dict. Returns score or None."""
    model = _load_v5_lgbm()
    if model is None:
        return None
    und = siglip2_details.get("underage", {})
    if not und:
        return None
    labels        = und.get("labels", {})
    underage_lbl  = labels.get("underage", {})
    adult_lbl     = labels.get("adult",    {})
    feat_idx      = model["feat_idx"]
    vec           = [0.0] * len(model["features"])
    for k, v in underage_lbl.items():
        if k in feat_idx:
            vec[feat_idx[k]] = v
    for k, v in adult_lbl.items():
        fname = "adult__" + k
        if fname in feat_idx:
            vec[feat_idx[fname]] = v
    score = 0.0
    for t in model["trees"]:
        node = t["r"]
        while node >= 0:
            fi, thr = t["s"][node]
            l, r    = t["c"][node]
            node    = l if vec[fi] <= thr else r
        score += t["l"][-(node + 1)]
    return 1.0 / (1.0 + math.exp(-score))

def _load_v6_lgbm() -> dict | None:
    global _V6_LGBM_CACHE
    if _V6_LGBM_CACHE is not None:
        return _V6_LGBM_CACHE
    js_path = BASE_DIR / "data" / "lgbm_evaluate_v6.js"
    if not js_path.exists():
        return None
    try:
        js = js_path.read_text(encoding="utf-8")
        features = json.loads(re.search(r'const LGBM_FEATURES = (\[.*?\]);', js, re.DOTALL).group(1))
        trees    = json.loads(re.search(r'const LGBM_TREES = (\[.*?\]);',    js, re.DOTALL).group(1))
        feat_idx = {f: i for i, f in enumerate(features)}
        _V6_LGBM_CACHE = {"features": features, "trees": trees, "feat_idx": feat_idx}
        print(f"  V6 LGBM loaded: {len(features)} features, {len(trees)} trees")
    except Exception as e:
        print(f"  V6 LGBM load error: {e}")
        return None
    return _V6_LGBM_CACHE

def _v6_lgbm_score(siglip2_details: dict) -> float | None:
    """Apply V6 LGBM to siglip2_details dict. Returns score or None."""
    model = _load_v6_lgbm()
    if model is None:
        return None
    und = siglip2_details.get("underage", {})
    if not und:
        return None
    labels        = und.get("labels", {})
    underage_lbl  = labels.get("underage", {})
    adult_lbl     = labels.get("adult",    {})
    feat_idx      = model["feat_idx"]
    vec           = [0.0] * len(model["features"])
    for k, v in underage_lbl.items():
        if k in feat_idx:
            vec[feat_idx[k]] = v
    for k, v in adult_lbl.items():
        fname = "adult__" + k
        if fname in feat_idx:
            vec[feat_idx[fname]] = v
    score = 0.0
    for t in model["trees"]:
        node = t["r"]
        while node >= 0:
            fi, thr = t["s"][node]
            l, r    = t["c"][node]
            node    = l if vec[fi] <= thr else r
        score += t["l"][-(node + 1)]
    return 1.0 / (1.0 + math.exp(-score))


# ── V8pas80-v2 / V11s80 LGBM (Python LightGBM — BCI-aware) ───────────────────
# Both models share the same scoring pattern (BCI aggregates + sanitized features).
# V8 = current production (d2911d10bb). V11 = candidate next-gen (more no_underage_).
import re as _re_mult, sys as _sys
import importlib.util as _iutil

_X20_RE = _re_mult.compile(r':x(\d+(?:\.\d+)?)$')


def _unmult(key, val):
    """Undo :xN multiplier stored in the SigLIP key suffix."""
    m = _X20_RE.search(key)
    if not m:
        return float(val)
    mult = float(m.group(1))
    if val >= 0.999:
        return 0.999 / mult
    return float(val) / mult


def _strip_mult(k):
    return _X20_RE.sub('', k)


def _sanitize_model(name: str) -> str:
    return name.replace(':', '_x').replace('"', '').replace("'", '').replace('{', '').replace('}', '')


# Lazy-load BCI taxonomy from scripts/bci_taxonomy.py
_BCI_CACHE = None
def _load_bci():
    global _BCI_CACHE
    if _BCI_CACHE is not None:
        return _BCI_CACHE
    bci_path = BASE_DIR / 'scripts' / 'bci_taxonomy.py'
    if not bci_path.exists():
        _BCI_CACHE = (frozenset(), frozenset(), frozenset())
        return _BCI_CACHE
    try:
        spec = _iutil.spec_from_file_location('bci_taxonomy', bci_path)
        mod  = _iutil.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BCI_CACHE = (mod.BODY_LABELS, mod.CONTEXT_LABELS, mod.INTERACTION_LABELS)
        print(f"  BCI taxonomy loaded: body={len(_BCI_CACHE[0])} ctx={len(_BCI_CACHE[1])} int={len(_BCI_CACHE[2])}")
    except Exception as e:
        print(f"  BCI taxonomy load error: {e}")
        _BCI_CACHE = (frozenset(), frozenset(), frozenset())
    return _BCI_CACHE


def _noisy_or(d, key_set):
    p = 1.0
    for k, v in d.items():
        if k in key_set:
            p *= 1.0 - float(v)
    return 1.0 - p


# Generic loader — same pattern for V8 and V11
_LGBM_MODEL_CACHE: dict = {}    # tag → (booster, features)

def _load_lgbm_py(tag: str, model_file: str, feat_file: str):
    if tag in _LGBM_MODEL_CACHE:
        return _LGBM_MODEL_CACHE[tag]
    if not _LGB_AVAILABLE:
        _LGBM_MODEL_CACHE[tag] = (None, None)
        return None, None
    model_path = BASE_DIR / 'data' / model_file
    feat_path  = BASE_DIR / 'data' / feat_file
    if not model_path.exists() or not feat_path.exists():
        print(f"  {tag} LGBM not found: {model_path.name} / {feat_path.name}")
        _LGBM_MODEL_CACHE[tag] = (None, None)
        return None, None
    try:
        booster = _lgb.Booster(model_file=str(model_path))
        feats   = json.loads(feat_path.read_text())
        _LGBM_MODEL_CACHE[tag] = (booster, feats)
        print(f"  {tag} LGBM loaded: {len(feats)} features")
        return booster, feats
    except Exception as e:
        print(f"  {tag} LGBM load error: {e}")
        _LGBM_MODEL_CACHE[tag] = (None, None)
        return None, None


def _v8_lgbm_score(underage_labels: dict, adult_labels: dict) -> float | None:
    """V8pas80-v2 — slim 80 features, BCI-aware, no no_underage_."""
    booster, feats = _load_lgbm_py('V8pas80-v2',
                                    'lgbm_underage_v8pas80_v2.txt',
                                    'lgbm_v8pas80_v2_features.json')
    if booster is None:
        return None
    return _bci_score(booster, feats, underage_labels, adult_labels)


def _v11_lgbm_score(underage_labels: dict, adult_labels: dict,
                    no_underage_labels: dict | None = None) -> float | None:
    """V11s80 — slim 80 features, BCI-aware.
    CRITICAL: V11 was trained with no_underage_labels merged into adult_labels
    (train_v11.py:69-70). Skipping the merge inflates adult FPR by ~6pp.
    """
    booster, feats = _load_lgbm_py('V11s80',
                                    'lgbm_underage_v11s80.txt',
                                    'lgbm_v11s80_features.json')
    if booster is None:
        return None
    if no_underage_labels:
        a = dict(adult_labels or {})
        for k, v in no_underage_labels.items():
            a[k] = max(a.get(k, 0.0), float(v))
        adult_labels = a
    return _bci_score(booster, feats, underage_labels, adult_labels)


def _bci_score(booster, feats, underage_labels: dict, adult_labels: dict) -> float | None:
    """Build feature vector with BCI aggregates and run inference."""
    body_set, ctx_set, inter_set = _load_bci()
    feat_idx = {f: i for i, f in enumerate(feats)}
    vec = [0.0] * len(feats)
    # Unmultiply :x suffixes and dedupe to plain key with max value
    u_plain = {}
    for k, v in (underage_labels or {}).items():
        raw = _unmult(k, v); k2 = _strip_mult(k)
        if k2 not in u_plain or u_plain[k2] < raw:
            u_plain[k2] = raw
    for k, v in u_plain.items():
        fk = _sanitize_model(k)
        if fk in feat_idx:
            vec[feat_idx[fk]] = float(v)
    for k, v in (adult_labels or {}).items():
        fk = 'adult__' + _sanitize_model(k)
        if fk in feat_idx:
            vec[feat_idx[fk]] = float(v)
    # BCI aggregates
    body  = _noisy_or(u_plain, body_set)
    ctx   = _noisy_or(u_plain, ctx_set)
    inter = _noisy_or(u_plain, inter_set)
    bc = body + ctx
    if '_child_body' in feat_idx:        vec[feat_idx['_child_body']] = body
    if '_child_context' in feat_idx:     vec[feat_idx['_child_context']] = ctx
    if '_child_interaction' in feat_idx: vec[feat_idx['_child_interaction']] = inter
    if '_body_vs_context' in feat_idx:   vec[feat_idx['_body_vs_context']] = (body / bc) if bc > 0 else 0.0
    try:
        return float(booster.predict([vec])[0])
    except Exception:
        return None


def _eval_id_to_gallery_id(eid: str) -> str:
    """Convert eval ID to gallery item ID.
    qwen_12345  → ls_12345
    ls_12345    → ls_12345
    dg_XXXX     → XXXX  (strip dg_ prefix, pool keys are bare UUIDs)
    """
    if eid.startswith("qwen_"):
        return "ls_" + eid[5:]
    if eid.startswith("dg_"):
        return eid[3:]
    return eid

def _make_eval_entry(lgbm, minor, human_label=None, variant=None, pipe_blocked=None, version='v6'):
    """Per-version blocked rule:
      v6  → lgbm >= 0.80 or pipe_blocked (legacy semantics with pipeline OR)
      v8  → lgbm >= 0.30 (LGBM-only, production V8pas80-v2)
      v11 → lgbm >= 0.30 (LGBM-only, candidate V11s80)
    """
    if version == 'v8':
        thr = V8_LGBM_THRESHOLD
        blocked = lgbm >= thr
    elif version == 'v11':
        thr = V11_LGBM_THRESHOLD
        blocked = lgbm >= thr
    else:
        thr = V6_LGBM_THRESHOLD
        if pipe_blocked is not None:
            blocked = pipe_blocked
        else:
            blocked = lgbm >= thr
    which = "lgbm" if lgbm >= thr else ("pipeline" if blocked else None)
    return {
        "lgbm":        round(lgbm,  4),
        "minor":       round(minor, 4),
        "blocked":     blocked,
        "which":       which,
        "human_label": human_label,
        "variant":     variant,
    }

def load_eval_data() -> dict:
    """Return dict: gallery_id → {v4: {lgbm, minor, blocked, which, human_label}, v5: ..., v6: ...}"""
    result: dict = {}

    # NOTE: precomputed EVAL_FILES (eval_v6_*.json) are DEPRECATED — they contain
    # stale V6 scores from an older model/pipeline revision. Inline scoring below
    # (via _v6_lgbm_score from lgbm_evaluate_v6.js) uses the current model and matches
    # the sweep tables. The files are kept on disk for archival/debugging only.
    # (Earlier code path: for version, files in EVAL_FILES.items(): ...)

    # Fill V6 gaps from DB only (V4/V5 inline blocks removed 2026-05-22)
    if DB_PATH.exists():
        # ── Fill V6 gaps from DB ──────────────────────────────────────────────
        try:
            conn3 = _db_connect()
        except Exception:
            conn3 = None

        if conn3 is not None:
            # grafana_pool
            for row in conn3.execute("SELECT id, label, variant, piper_result FROM grafana_pool"):
                pid = row["id"]
                if pid in result and "v6" in result[pid]:
                    continue
                pr  = json.loads(row["piper_result"]) if row["piper_result"] else {}
                det = pr.get("siglip2_details") or {}
                if not det:
                    continue
                v6  = _v6_lgbm_score(det)
                if v6 is None:
                    continue
                und   = det.get("underage", {})
                minor = und.get("minor", 0.0)
                entry = _make_eval_entry(lgbm=v6, minor=minor, human_label=row["label"], variant=row["variant"])
                result.setdefault(pid, {})["v6"] = entry

            # ls_images — from DB (may hit corruption on large DBs; JSON fallback below)
            for offset in range(0, 5000, 200):
                try:
                    rows = conn3.execute("""
                        SELECT task_id, age_from, variant, siglip2_details
                        FROM ls_images WHERE siglip2_details IS NOT NULL
                        LIMIT 200 OFFSET ?
                    """, (offset,)).fetchall()
                    if not rows:
                        break
                    for row in rows:
                        gid = f"ls_{row['task_id']}"
                        if gid in result and "v6" in result[gid]:
                            continue
                        det   = json.loads(row["siglip2_details"])
                        v6    = _v6_lgbm_score(det)
                        if v6 is None:
                            continue
                        und   = det.get("underage", {})
                        minor = und.get("minor", 0.0)
                        af    = row["age_from"]
                        entry = _make_eval_entry(lgbm=v6, minor=minor,
                                                 human_label=ls_cat(af) if af is not None else None,
                                                 variant=row["variant"])
                        result.setdefault(gid, {})["v6"] = entry
                except Exception:
                    break  # DB page corruption — stop, fall through to JSON
            conn3.close()

        # ── V6 JSON fallback for LS items (qwen3_age_results.json) ───────────
        # Covers items that DB couldn't provide due to page corruption or missing data.
        ls_json = BASE_DIR / "qwen3_age_results.json"
        if ls_json.exists():
            try:
                raw = ls_json.read_bytes().rstrip(b"\x00").decode("utf-8")
                ls_data = json.loads(raw)
                for v in ls_data.values():
                    gid = f"ls_{v.get('task_id', '')}"
                    if gid in result and "v6" in result[gid]:
                        continue
                    det = (v.get("siglip2_details") or {})
                    if not det:
                        continue
                    v6 = _v6_lgbm_score(det)
                    if v6 is None:
                        continue
                    und   = det.get("underage", {})
                    minor = und.get("minor", 0.0)
                    af    = (v.get("age") or {}).get("ageFrom")
                    entry = _make_eval_entry(lgbm=v6, minor=minor,
                                             human_label=ls_cat(af) if af is not None else None)
                    result.setdefault(gid, {})["v6"] = entry
            except Exception:
                pass

    # ── V8pas80-v2 + V11s80 inline scoring ───────────────────────────────────
    ls_rescored_map = {}
    ls_rescored_path = BASE_DIR / "data" / "ls_holdout_rescored.json"
    if ls_rescored_path.exists():
        try:
            for r in json.loads(ls_rescored_path.read_text()):
                if r.get('done'):
                    ls_rescored_map[f"ls_{r['task_id']}"] = r
        except Exception:
            pass

    v317_map = {}
    v317_path = BASE_DIR / "data" / "v9_317_scores.json"
    if v317_path.exists():
        try:
            for r in json.loads(v317_path.read_text()):
                if r.get('done'):
                    v317_map[r['id']] = r
        except Exception:
            pass

    # Tom's K=30 rescore: gives Tom k30 lgbm score + 317-tag native input for V11
    tom_map = {}
    tom_path = BASE_DIR / "data" / "tom_scores.json"
    if tom_path.exists():
        try:
            for r in json.loads(tom_path.read_text()):
                if r.get('done'):
                    tom_map[r['id']] = r
        except Exception:
            pass

    # V11 NATIVE scores — produced by scripts/rescore_via_v11.py through V11's own
    # Piper pipeline ce79f7e299. Highest priority for V11 input: uses lgbm_score
    # straight from the pipeline (so taxonomy/feature alignment is guaranteed).
    # Records with no_face=True are items the V11 pipeline rejected at the
    # face_detect stage — V11 does not score them. We materialise them with
    # lgbm=None so the stat block skips them (lgbm==null guard) and the card
    # shows "—" instead of a misleading fallback star.
    v11_native_map = {}
    v11_native_path = BASE_DIR / "data" / "v11_native_scores.json"
    if v11_native_path.exists():
        try:
            for r in json.loads(v11_native_path.read_text()):
                if r.get('done'):
                    v11_native_map[r['id']] = r
        except Exception:
            pass

    # ── V8pas80-v2 + V11s80 inline scoring ───────────────────────────────────
    # CRITICAL: V8 and V11 were trained on DIFFERENT taxonomies — they MUST be
    # fed scores from the source they were trained on:
    #   V8 ← 180-tag legacy (qwen3_age_results.json / piper_result.siglip2_details — no :x20)
    #   V11 ← 317-tag rescored (ls_holdout_rescored.json / v9_317_scores.json — with :x20)
    # Mixing them inflates scores wildly (e.g. V8 adult FPR jumps 6% → 53%).
    if _LGB_AVAILABLE:
        try:
            conn_inline = _db_connect()

            # ───── V8 scoring path: 180-tag taxonomy ─────
            # Grafana: read from piper_result.siglip2_details (always 180-tag in d2911d10bb)
            for row in conn_inline.execute("SELECT id, label, variant, piper_result FROM grafana_pool"):
                pid = row["id"]
                if "v8" in result.get(pid, {}):
                    continue
                try:
                    pr  = json.loads(row["piper_result"]) if row["piper_result"] else {}
                    det = (pr.get('siglip2_details') or {}).get('underage', {})
                    lbl_data = det.get('labels', {})
                    u = lbl_data.get('underage', {}); a = lbl_data.get('adult', {})
                    und_minor = det.get('minor', 0.0)
                except Exception:
                    continue
                if not u and not a:
                    continue
                s8 = _v8_lgbm_score(u, a)
                if s8 is not None:
                    result.setdefault(pid, {})["v8"] = _make_eval_entry(
                        lgbm=s8, minor=und_minor,
                        human_label=row["label"], variant=row["variant"], version='v8')

            # LS V8: from qwen3_age_results.json (180-tag taxonomy)
            ls_json_inline = BASE_DIR / "qwen3_age_results.json"
            if ls_json_inline.exists():
                try:
                    import ast as _ast
                    raw_inline = ls_json_inline.read_bytes().rstrip(b"\x00").decode("utf-8")
                    ls_data_inline = json.loads(raw_inline)
                    for v in ls_data_inline.values():
                        gid = f"ls_{v.get('task_id', '')}"
                        if "v8" in result.get(gid, {}):
                            continue
                        siglip_raw = v.get("siglip2_details")
                        if not siglip_raw:
                            continue
                        det = ((_ast.literal_eval(siglip_raw) if isinstance(siglip_raw, str) else siglip_raw) or {}).get("underage", {})
                        lbl_data = det.get("labels", {})
                        u = lbl_data.get("underage", {}); a = lbl_data.get("adult", {})
                        af = (v.get("age") or {}).get("ageFrom")
                        cat = ls_cat(af) if af is not None else None
                        s8 = _v8_lgbm_score(u, a)
                        if s8 is not None:
                            result.setdefault(gid, {})["v8"] = _make_eval_entry(
                                lgbm=s8, minor=det.get("minor", 0.0), human_label=cat, version='v8')
                except Exception:
                    pass

            # ───── V11 scoring path: V11 NATIVE pipeline (ce79f7e299) — HIGHEST PRIORITY ─────
            # Uses the lgbm_score computed by V11's own Piper pipeline, so taxonomy
            # alignment is guaranteed. Items with no_face=True (face_detect rejected
            # them) are recorded with lgbm=None → excluded from stats, card shows "—".
            # NB: for no_face records we DO NOT pre-populate V11 entry. _make_eval_entry
            # cannot accept lgbm=None (it does lgbm >= thr inside), so the previous
            # implementation crashed mid-iteration and dropped LS V11, K30 scoring and
            # Tom integration. Now no_face items simply fall through to the legacy
            # fallback chain (ls_holdout_rescored / v9_317 / 180-tag fallback with `*`).
            n_native = n_nf_skip = 0
            for pid, rec in v11_native_map.items():
                if "v11" in result.get(pid, {}):
                    continue
                if rec.get('no_face'):
                    n_nf_skip += 1
                    continue  # let fallback chain handle these
                ls_score = rec.get('lgbm_score')
                if ls_score is None:
                    continue
                lbl = rec.get('label')
                var = default_variant(lbl) if lbl else None
                try:
                    entry = _make_eval_entry(
                        lgbm=float(ls_score), minor=rec.get('minor', 0.0) or 0.0,
                        human_label=lbl, variant=var, version='v11')
                    entry['v11_native'] = True
                    result.setdefault(pid, {})["v11"] = entry
                    n_native += 1
                except Exception:
                    # per-record safety so one bad record never kills the whole block
                    continue
            if v11_native_map:
                print(f"  V11 native pipeline: scored={n_native}  no_face_skipped={n_nf_skip}  "
                      f"(out of {len(v11_native_map)} records)")

            # ───── V11 scoring path: 317-tag rescored taxonomy (legacy fallback chain) ─────
            # Grafana: from v9_317_scores.json (covers ~317 hard-curated items)
            for pid, rec in v317_map.items():
                if "v11" in result.get(pid, {}):
                    continue
                u = rec.get('underage_labels', {})
                a = rec.get('adult_labels', {})
                nu = rec.get('no_underage_labels', {})
                if not u and not a:
                    continue
                s11 = _v11_lgbm_score(u, a, nu)
                if s11 is not None:
                    result.setdefault(pid, {})["v11"] = _make_eval_entry(
                        lgbm=s11, minor=rec.get('minor', 0.0),
                        human_label=rec.get('label'),
                        variant=default_variant(rec.get('label')) if rec.get('label') else None,
                        version='v11')

            # LS V11: from ls_holdout_rescored.json (317-tag, :x20-aware)
            for gid, rec in ls_rescored_map.items():
                if "v11" in result.get(gid, {}):
                    continue
                u = rec.get('underage_labels', {})
                a = rec.get('adult_labels', {})
                nu = rec.get('no_underage_labels', {})
                if not u and not a:
                    continue
                lbl = rec.get('label')
                s11 = _v11_lgbm_score(u, a, nu)
                if s11 is not None:
                    result.setdefault(gid, {})["v11"] = _make_eval_entry(
                        lgbm=s11, minor=rec.get('minor', 0.0), human_label=lbl,
                        variant=default_variant(lbl) if lbl else None, version='v11')

            # ── V11 fallback for items NOT in rescored ────────────────────────
            # Scores V11 on 180-tag input (qwen3 for LS, siglip2_details for grafana).
            # NOTE: V11 was trained on 317-tag :x20 input — fallback scores are degraded
            # but populated so every card has a number to show.

            # Grafana V11 fallback (piper_result siglip2_details — 180-tag)
            for row in conn_inline.execute("SELECT id, label, variant, piper_result FROM grafana_pool"):
                pid = row["id"]
                if "v11" in result.get(pid, {}):
                    continue
                try:
                    pr  = json.loads(row["piper_result"]) if row["piper_result"] else {}
                    det = (pr.get('siglip2_details') or {}).get('underage', {})
                    lbl_data = det.get('labels', {})
                    u = lbl_data.get('underage', {}); a = lbl_data.get('adult', {})
                    und_minor = det.get('minor', 0.0)
                except Exception:
                    continue
                if not u and not a:
                    continue
                s11 = _v11_lgbm_score(u, a, None)
                if s11 is not None:
                    entry = _make_eval_entry(
                        lgbm=s11, minor=und_minor,
                        human_label=row["label"], variant=row["variant"], version='v11')
                    entry['fallback_taxonomy'] = True  # mark — taxonomy mismatch
                    result.setdefault(pid, {})["v11"] = entry

            # LS V11 fallback (qwen3_age_results.json — 180-tag)
            if ls_json_inline.exists():
                try:
                    raw_inline = ls_json_inline.read_bytes().rstrip(b"\x00").decode("utf-8")
                    ls_data_inline = json.loads(raw_inline)
                    for v in ls_data_inline.values():
                        gid = f"ls_{v.get('task_id', '')}"
                        if "v11" in result.get(gid, {}):
                            continue
                        siglip_raw = v.get("siglip2_details")
                        if not siglip_raw:
                            continue
                        det = ((_ast.literal_eval(siglip_raw) if isinstance(siglip_raw, str) else siglip_raw) or {}).get("underage", {})
                        lbl_data = det.get("labels", {})
                        u = lbl_data.get("underage", {}); a = lbl_data.get("adult", {})
                        if not u and not a:
                            continue
                        af = (v.get("age") or {}).get("ageFrom")
                        cat = ls_cat(af) if af is not None else None
                        s11 = _v11_lgbm_score(u, a, None)
                        if s11 is not None:
                            entry = _make_eval_entry(
                                lgbm=s11, minor=det.get("minor", 0.0),
                                human_label=cat, version='v11')
                            entry['fallback_taxonomy'] = True
                            result.setdefault(gid, {})["v11"] = entry
                except Exception:
                    pass

            # ── K30 scoring ───────────────────────────────────────────────────
            # Prefer data/k30_rescored.json (full 180-tag, freshly rescored via Piper d2911d10bb)
            # over the sparse top-5 in k30_pool.piper_result. Items not yet rescored fall
            # back to the stored top-5 (will be marked sparse).
            k30_rescored = {}
            k30_rescored_path = BASE_DIR / "data" / "k30_rescored.json"
            if k30_rescored_path.exists():
                try:
                    for r in json.loads(k30_rescored_path.read_text()):
                        if r.get('done'):
                            k30_rescored[r['id']] = r
                except Exception:
                    pass
            try:
                for row in conn_inline.execute("""SELECT id, label, label_confirmed, label_source,
                                                          variant, piper_result, qwen3_result
                                                   FROM k30_pool
                                                   WHERE deleted IS NULL OR deleted = 0"""):
                    pid = row["id"]
                    if pid in result and 'v6' in result[pid] and 'v8' in result[pid] and 'v11' in result[pid]:
                        continue
                    # Prefer freshly rescored full-180-tag data over stored top-5
                    rescored = k30_rescored.get(pid)
                    sparse_input = False
                    if rescored is not None:
                        u = rescored.get('underage_labels') or {}
                        a = rescored.get('adult_labels') or {}
                        und_minor = rescored.get('minor', 0.0)
                    else:
                        sparse_input = True
                        try:
                            pr = json.loads(row["piper_result"]) if row["piper_result"] else {}
                            det = (pr.get('siglip2_details') or {}).get('underage', {})
                            lbl_data = det.get('labels', {})
                            u = lbl_data.get('underage', {})
                            a = lbl_data.get('adult', {})
                            und_minor = det.get('minor', 0.0)
                        except Exception:
                            continue
                    if not u and not a:
                        continue
                    # V6
                    if 'v6' not in result.get(pid, {}):
                        s6 = _v6_lgbm_score({'underage': {'labels': {'underage': u, 'adult': a}, 'minor': und_minor}})
                        if s6 is not None:
                            result.setdefault(pid, {})['v6'] = _make_eval_entry(
                                lgbm=s6, minor=und_minor, human_label=row["label"],
                                variant=row["variant"], version='v6')
                    # V8 — native if rescored, sparse otherwise
                    if 'v8' not in result.get(pid, {}):
                        s8 = _v8_lgbm_score(u, a)
                        if s8 is not None:
                            e8 = _make_eval_entry(lgbm=s8, minor=und_minor, human_label=row["label"],
                                                  variant=row["variant"], version='v8')
                            if sparse_input:
                                e8['k30_sparse'] = True
                            result.setdefault(pid, {})['v8'] = e8
                    # V11 — taxonomy mismatch (V11 trained on 317-tag, d2911d10bb is 180-tag)
                    # so always mark fallback_taxonomy. Even after rescore the format differs.
                    # no_underage_labels also not available from K30 rescore path.
                    if 'v11' not in result.get(pid, {}):
                        s11 = _v11_lgbm_score(u, a, None)
                        if s11 is not None:
                            e11 = _make_eval_entry(lgbm=s11, minor=und_minor, human_label=row["label"],
                                                   variant=row["variant"], version='v11')
                            e11['fallback_taxonomy'] = True
                            if sparse_input:
                                e11['k30_sparse'] = True
                            result.setdefault(pid, {})['v11'] = e11
                    # Tom's k30 model score — stored in piper_result.k30_models.k30_score
                    if 'k30tom' not in result.get(pid, {}):
                        try:
                            pr_full = json.loads(row["piper_result"]) if row["piper_result"] else {}
                            k30s = (pr_full.get('k30_models') or {}).get('k30_score')
                            if k30s is not None:
                                ek = _make_eval_entry(lgbm=float(k30s), minor=und_minor,
                                                     human_label=row["label"],
                                                     variant=row["variant"], version='k30tom')
                                result.setdefault(pid, {})['k30tom'] = ek
                        except Exception:
                            pass
            except Exception as e:
                print(f"  K30 inline-score error: {e}")

            # ── Tom rescore integration ────────────────────────────────────────
            # For any item in tom_map: (a) inject k30tom score, (b) if V11 is in
            # fallback, REPLACE it with native 317-tag scoring using Tom's labels
            # (V11 was trained on this exact taxonomy).
            # Tom integration: inject k30tom score only.
            # IMPORTANT: We do NOT use Tom data to score V11 anymore. V11 was trained on V9
            # pipeline (9cd1798843, now removed) which output a denser 317-tag distribution
            # than Tom's a4aa9dbd9c. Force-feeding Tom labels into V11 distorts its scores
            # (V11 has 35 adult features, Tom outputs only ~5-24 per image — 6% overlap).
            # V11 keeps its original input source: ls_holdout_rescored (LS native), v9_317_scores
            # (grafana subset), or fallback for K30. K30 V11 stays marked as fallback (*).
            for pid, t in tom_map.items():
                if 'k30tom' not in result.get(pid, {}):
                    ks = t.get('k30_score')
                    if ks is not None:
                        result.setdefault(pid, {})['k30tom'] = _make_eval_entry(
                            lgbm=float(ks), minor=t.get('minor', 0.0),
                            human_label=t.get('label'),
                            variant=default_variant(t.get('label')) if t.get('label') else None,
                            version='k30tom')
            if tom_map:
                print(f"  Tom integration: k30tom scores injected for {len(tom_map)} items "
                      f"(V11 stays on native ls_holdout_rescored/fallback sources)")

            conn_inline.close()
        except Exception as e:
            print(f"  V8/V11 inline-score error: {e}")

    return result

# ── helpers ──────────────────────────────────────────────────────────────────

def ls_cat(age_from):
    """child: 1–14, teen: 15–17, adult: 18+  (by ageFrom / minimum bound)."""
    if age_from is None: return None
    if age_from <= 14:   return "child"   # 1–14
    if age_from <= 17:   return "teen"    # 15–17
    return "adult"                        # 18+

def default_variant(label):
    """child/teen → positive (should be blocked), adult → negative (should pass)."""
    if label in ("child", "teen"): return "positive"
    if label == "adult":           return "negative"
    return None

def migrate_variants():
    """One-time: set variant from label for all rows in both tables."""
    if not DB_PATH.exists(): return
    conn = _db_connect()
    # grafana_pool
    n_gp = 0
    for row in conn.execute("SELECT id, label FROM grafana_pool"):
        expected = default_variant(row["label"])
        if expected is not None and row["label"]:
            conn.execute("UPDATE grafana_pool SET variant=? WHERE id=? AND (variant IS NULL OR variant != ?)",
                         (expected, row["id"], expected))
            n_gp += conn.execute("SELECT changes()").fetchone()[0]
    # ls_images — derive from age_from
    n_ls = 0
    for row in conn.execute("SELECT task_id, age_from FROM ls_images"):
        lbl = ls_cat(row["age_from"])
        expected = default_variant(lbl)
        if expected is not None:
            conn.execute("UPDATE ls_images SET variant=? WHERE task_id=? AND (variant IS NULL OR variant != ?)",
                         (expected, row["task_id"], expected))
            n_ls += conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    if n_gp or n_ls:
        print(f"  migrate_variants: fixed {n_gp} grafana + {n_ls} LS rows")

def pipe_status(entry):
    res = entry.get("piper_result") or {}
    if not res or res.get("error"): return "unprocessed"
    labels = res.get("siglip2_labels") or []
    if "underage" in labels:  return "underage"
    if not res.get("siglip2_passed"): return "other"
    return "passed"

def load_ls():
    if not DB_PATH.exists(): return []
    conn = _db_connect()
    rows = conn.execute("""
        SELECT task_id, media, variant, age_from, age_to,
               siglip2_labels, siglip2_passed, siglip2_details, face_detect
        FROM ls_images
        WHERE age_from IS NOT NULL
    """).fetchall()
    conn.close()
    result = []
    for row in rows:
        af = row["age_from"]
        pipe_res = None
        if row["siglip2_labels"] is not None:
            pipe_res = {
                "siglip2_labels":     json.loads(row["siglip2_labels"]) if row["siglip2_labels"] else None,
                "siglip2_passed":     bool(row["siglip2_passed"]),
                "siglip2_details":    json.loads(row["siglip2_details"]) if row["siglip2_details"] else None,
                "face_detect_result": json.loads(row["face_detect"]) if row["face_detect"] else None,
            }
        result.append({
            "id":           f"ls_{row['task_id']}",
            "_ls_id":       row["task_id"],
            "source":       "labelstudio",
            "session":      "labelstudio",
            "_serve_url":   row["media"] or "",
            "label":        ls_cat(af),
            "labeled_at":   None,
            "ageFrom":      af,
            "ageTo":        row["age_to"],
            "variant":      default_variant(ls_cat(af)) or "positive",
            "prompt":       None,
            "piper_result": pipe_res,
            "export_batch": None,
        })
    return result

def load_grafana():
    if not DB_PATH.exists(): return []
    conn = _db_connect()
    rows = conn.execute("""
        SELECT id, thumb_url, local_path, prompt, label, label_source,
               label_confirmed, labeled_at, variant, export_batch, piper_result, qwen3_result
        FROM grafana_pool
        WHERE deleted IS NULL OR deleted = 0
    """).fetchall()
    conn.close()
    result = []
    for v in rows:
        local = v["local_path"]
        serve = ("/img/" + Path(local).name) if local and (BASE_DIR / local).exists() else (v["thumb_url"] or "")
        lbl = v["label"]
        result.append({
            "id":              v["id"],
            "_ls_id":          None,
            "source":          "grafana",
            "session":         v["export_batch"] or "unknown",
            "_serve_url":      serve,
            "thumb_url":       v["thumb_url"] or "",
            "label":           lbl,
            "label_source":    v["label_source"],
            "label_confirmed": bool(v["label_confirmed"]),
            "labeled_at":      v["labeled_at"],
            "ageFrom":         None,
            "ageTo":           None,
            "variant":         default_variant(lbl),
            "prompt":          v["prompt"] or "",
            "piper_result":    json.loads(v["piper_result"]) if v["piper_result"] else None,
            "export_batch":    v["export_batch"],
            "qwen3_result":    json.loads(v["qwen3_result"]) if v["qwen3_result"] else None,
        })
    return result

def load_k30():
    """K30 dataset (Tom Renneberg). Images served from S3 thumb_url directly."""
    if not DB_PATH.exists(): return []
    try:
        conn = _db_connect()
        rows = conn.execute("""
            SELECT id, thumb_url, local_path, prompt, label, label_source,
                   label_confirmed, labeled_at, variant, piper_result, qwen3_result
            FROM k30_pool
            WHERE deleted IS NULL OR deleted = 0
        """).fetchall()
        conn.close()
    except Exception as e:
        print(f"  load_k30 error (table may not exist yet): {e}")
        return []
    result = []
    for v in rows:
        local = v["local_path"]
        serve = ("/img/" + Path(local).name) if local and (BASE_DIR / local).exists() else (v["thumb_url"] or "")
        lbl = v["label"]
        result.append({
            "id":              v["id"],
            "_ls_id":          None,
            "source":          "k30",
            "session":         "k30",
            "_serve_url":      serve,
            "thumb_url":       v["thumb_url"] or "",
            "label":           lbl,
            "label_source":    v["label_source"],
            "label_confirmed": bool(v["label_confirmed"]),
            "labeled_at":      v["labeled_at"],
            "ageFrom":         None,
            "ageTo":           None,
            "variant":         v["variant"] or default_variant(lbl),
            "prompt":          v["prompt"] or "",
            "piper_result":    json.loads(v["piper_result"]) if v["piper_result"] else None,
            "export_batch":    "k30",
            "qwen3_result":    json.loads(v["qwen3_result"]) if v["qwen3_result"] else None,
        })
    return result


def combined_data():
    ls      = load_ls()
    grafana = load_grafana()
    k30     = load_k30()
    # Newest grafana first, then LS, then K30 at the end
    grafana.sort(key=lambda r: r.get("export_batch") or "", reverse=True)
    return grafana + ls + k30

def grafana_sessions():
    if not DB_PATH.exists(): return []
    conn = _db_connect()
    rows = conn.execute("SELECT DISTINCT export_batch FROM grafana_pool WHERE deleted IS NULL OR deleted = 0").fetchall()
    conn.close()
    seen = sorted({r["export_batch"] or "unknown" for r in rows}, reverse=True)
    return seen

# ── JSON sync helpers (keep JSON files in sync for external scripts) ───────────

def _sync_ls_json():
    """Regenerate qwen3_age_results.json from DB — called after label saves."""
    if not DB_PATH.exists(): return
    conn = _db_connect()
    rows = conn.execute("""
        SELECT task_id, media, variant, category, age_from, age_to,
               launch_id, siglip2_labels, siglip2_passed, siglip2_details,
               face_detect, error, processed_at, extra
        FROM ls_images
    """).fetchall()
    conn.close()
    data = {}
    for row in rows:
        tid = row["task_id"]
        item = {
            "task_id":            tid,
            "media":              row["media"],
            "variant":            row["variant"],
            "category":           row["category"],
            "age":                {"ageFrom": row["age_from"], "ageTo": row["age_to"]} if row["age_from"] is not None else None,
            "launch_id":          row["launch_id"],
            "siglip2_labels":     json.loads(row["siglip2_labels"]) if row["siglip2_labels"] else None,
            "siglip2_passed":     bool(row["siglip2_passed"]) if row["siglip2_passed"] is not None else None,
            "siglip2_details":    json.loads(row["siglip2_details"]) if row["siglip2_details"] else None,
            "face_detect_result": json.loads(row["face_detect"]) if row["face_detect"] else None,
            "error":              row["error"],
            "piper_processed_at": row["processed_at"],
        }
        if row["extra"]:
            item.update(json.loads(row["extra"]))
        data[str(tid)] = item
    tmp = str(LS_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, LS_FILE)

def _sync_pool_json():
    """Regenerate disagree_pool.json from DB — called after label saves."""
    if not DB_PATH.exists(): return
    conn = _db_connect()
    rows = conn.execute("""
        SELECT id, thumb_url, local_path, prompt, label, label_source,
               label_confirmed, labeled_at, variant, export_batch, exported_at,
               piper_result, qwen3_result, extra, deleted, deleted_at
        FROM grafana_pool
    """).fetchall()
    conn.close()
    data = {}
    for row in rows:
        item = {
            "id":             row["id"],
            "thumb_url":      row["thumb_url"],
            "local_path":     row["local_path"],
            "prompt":         row["prompt"],
            "label":          row["label"],
            "label_source":   row["label_source"],
            "label_confirmed": bool(row["label_confirmed"]),
            "labeled_at":     row["labeled_at"],
            "variant":        row["variant"],
            "export_batch":   row["export_batch"],
            "exported_at":    row["exported_at"],
            "piper_result":   json.loads(row["piper_result"]) if row["piper_result"] else None,
            "qwen3_result":   json.loads(row["qwen3_result"]) if row["qwen3_result"] else None,
        }
        if row["deleted"]:
            item["deleted"] = True
            item["deleted_at"] = row["deleted_at"]
        if row["extra"]:
            item.update(json.loads(row["extra"]))
        data[row["id"]] = item
    tmp = str(POOL_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, POOL_FILE)

# ── save helpers ──────────────────────────────────────────────────────────────

def save_ls(updates: dict, to_delete: list):
    if not DB_PATH.exists(): return 0, 0
    conn = _db_connect()
    saved = 0
    for item_id, upd in updates.items():
        tid = int(item_id.replace("ls_", ""))
        lbl = upd.get("label")
        if lbl not in ("child", "teen", "adult"):
            continue
        var = default_variant(lbl)
        # Age data is NOT updated — it stays as originally set (informational only)
        conn.execute("UPDATE ls_images SET variant=? WHERE task_id=?",
                     (var, tid))
        saved += 1
    deleted = 0
    for item_id in to_delete:
        tid = int(item_id.replace("ls_", ""))
        conn.execute("DELETE FROM ls_images WHERE task_id=?", (tid,))
        deleted += 1
    conn.commit()
    conn.close()
    _db_flush()
    _sync_ls_json()
    return saved, deleted

def save_k30(updates: dict, to_delete: list, to_confirm: list = None):
    """Save user labels for K30 items (mirror of save_grafana on k30_pool)."""
    if not DB_PATH.exists(): return 0, 0
    conn = _db_connect()
    now = datetime.now(timezone.utc).isoformat()
    saved = 0
    for item_id, upd in (updates or {}).items():
        lbl = upd.get("label")
        if lbl not in ("child", "teen", "adult"):
            continue
        if not conn.execute("SELECT 1 FROM k30_pool WHERE id=?", (item_id,)).fetchone():
            continue
        var = default_variant(lbl)
        conn.execute("""UPDATE k30_pool SET label=?, labeled_at=?, label_source='human',
                        label_confirmed=1, variant=? WHERE id=?""", (lbl, now, var, item_id))
        saved += 1
    for item_id in (to_confirm or []):
        row = conn.execute("SELECT label FROM k30_pool WHERE id=?", (item_id,)).fetchone()
        if row and row["label"]:
            lbl = row["label"]; var = default_variant(lbl)
            conn.execute("""UPDATE k30_pool SET label_confirmed=1, label_source='human',
                            labeled_at=COALESCE(labeled_at, ?), variant=? WHERE id=?""",
                         (now, var, item_id))
            saved += 1
    deleted = 0
    for item_id in (to_delete or []):
        conn.execute("UPDATE k30_pool SET deleted=1, deleted_at=? WHERE id=?", (now, item_id))
        deleted += 1
    conn.commit(); conn.close(); _db_flush()
    return saved, deleted


def save_grafana(updates: dict, to_delete: list, to_confirm: list = None):
    if not DB_PATH.exists(): return 0, 0
    conn = _db_connect()
    now = datetime.now(timezone.utc).isoformat()
    saved = 0
    for item_id, upd in updates.items():
        lbl = upd.get("label")
        if lbl not in ("child", "teen", "adult"):
            continue
        row = conn.execute("SELECT 1 FROM grafana_pool WHERE id=?", (item_id,)).fetchone()
        if not row:
            continue
        var = default_variant(lbl)
        conn.execute("""
            UPDATE grafana_pool SET label=?, labeled_at=?, label_source='human',
            label_confirmed=1, variant=? WHERE id=?
        """, (lbl, now, var, item_id))
        saved += 1
    for item_id in (to_confirm or []):
        row = conn.execute("SELECT label, labeled_at FROM grafana_pool WHERE id=?", (item_id,)).fetchone()
        if row and row["label"]:
            lbl = row["label"]
            var = default_variant(lbl)
            conn.execute("""
                UPDATE grafana_pool SET label_confirmed=1, label_source='human',
                labeled_at=COALESCE(labeled_at, ?), variant=? WHERE id=?
            """, (now, var, item_id))
            saved += 1
    deleted = 0
    for item_id in to_delete:
        conn.execute(
            "UPDATE grafana_pool SET deleted=1, deleted_at=? WHERE id=?",
            (now, item_id)
        )
        deleted += 1
    conn.commit()
    conn.close()
    _db_flush()
    _sync_pool_json()
    return saved, deleted


# ── HTML ──────────────────────────────────────────────────────────────────────

GALLERY_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Gallery</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; font-size: 12px; background: #0f0f0f; color: #e0e0e0; }
header {
  padding: 8px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  position: sticky; top: 0; z-index: 100;
}
header h1 { font-size: 13px; font-weight: normal; color: #aaa; white-space: nowrap; }
select, input[type=text] {
  background: #222; border: 1px solid #444; color: #e0e0e0;
  padding: 3px 7px; border-radius: 3px; font-size: 11px; font-family: monospace;
}
#confirm-btn {
  background: #1a1a2e; border: 1px solid #3a3a8a; color: #8888da;
  padding: 4px 14px; border-radius: 4px; cursor: pointer;
  font-size: 12px; font-family: monospace;
}
#confirm-btn:hover { background: #22223a; }
#confirm-btn.has-pending { background: #1a2e3a; border-color: #3a8aaa; color: #60c0e0; font-weight: bold; }
#save-btn {
  background: #1a2e1a; border: 1px solid #3a6a3a; color: #8fda8f;
  padding: 4px 14px; border-radius: 4px; cursor: pointer;
  font-size: 12px; font-family: monospace; margin-left: auto;
}
#save-btn:hover { background: #22401e; }
#save-btn.dirty { background: #3a1a1a; border-color: #cc4444; color: #ff8888; }
#status { font-size: 11px; color: #666; white-space: nowrap; }
.stats-bar {
  font-size: 11px; color: #555; display: flex; gap: 16px;
  padding: 5px 16px; background: #111; border-bottom: 1px solid #1f1f1f; flex-wrap: wrap;
}
.stats-bar b { color: #888; }
.s-ls   { color: #7ab0dd; }
.s-gr   { color: #c792ea; }
.s-k30  { color: #ff9050; }
.s-ch   { color: #ff7070; }
.s-tn   { color: #ffd040; }
.s-ad   { color: #6fda72; }
.s-un   { color: #444; }
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 7px; padding: 10px;
}
.card {
  background: #181818; border: 1px solid #2a2a2a; border-radius: 5px;
  overflow: hidden; display: flex; flex-direction: column; transition: border-color .12s;
}
.card:hover { border-color: #555; }
.card.modified { border-color: #4a7a4a; }
.card.lbl-child { border-color: #883333; }
.card.lbl-teen  { border-color: #887733; }
.card.lbl-adult { border-color: #337744; }
.card.src-ls    { border-left: 3px solid #2a5a8a; }
.card.src-grafana { border-left: 3px solid #5a2a8a; }

/* image */
.img-wrap {
  width: 100%; aspect-ratio: 1; overflow: hidden; background: #111;
  display: flex; align-items: center; justify-content: center;
  cursor: zoom-in; position: relative;
}
.img-wrap img { width:100%; height:100%; object-fit:contain; transition:transform .18s; }
.img-wrap:hover img { transform: scale(1.04); }
/* hide button */
.hide-btn {
  position: absolute; top: 5px; right: 5px; width: 20px; height: 20px;
  border-radius: 50%; background: rgba(0,0,0,.55); border: 1px solid #555;
  color: #aaa; font-size: 13px; line-height: 18px; text-align: center;
  cursor: pointer; opacity: 0; transition: opacity .15s, background .15s; z-index: 2;
}
.card:hover .hide-btn { opacity: 1; }
.hide-btn:hover { background: rgba(180,40,40,.85) !important; border-color: #cc4444; color: #fff; }
.card.hidden-pending { opacity: .35; border-color: #663333 !important; }
/* Mark-as-favourite star button on cards */
.mark-btn {
  position: absolute; top: 4px; left: 50%; transform: translateX(-50%);
  width: 24px; height: 24px;
  display: flex; align-items: center; justify-content: center;
  background: rgba(0,0,0,.55); border-radius: 50%;
  font-size: 14px; color: #ffd060; cursor: pointer;
  opacity: 0; transition: opacity 0.1s, background 0.1s;
  z-index: 3; user-select: none;
}
.card:hover .mark-btn { opacity: .9; }
.mark-btn:hover { background: rgba(60,40,0,.85); }
.card.marked .mark-btn { opacity: 1; background: rgba(80,55,0,.85); color: #ffd040; }
/* Full-perimeter golden frame using outline (not clipped by overflow:hidden of img-wrap) */
.card.marked { outline: 2px solid #d0a040; outline-offset: -2px; }

/* Marked-only highlight on thumb in lb-thumbs */
.lb-thumb.marked { border-color: #ffd040 !important; box-shadow: 0 0 4px rgba(255,208,64,.6); }

/* Export button (toolbar) */
#export-btn {
  background:#2a221a;border:1px solid #6a5030;color:#ffd040;
  padding:5px 12px;border-radius:3px;cursor:pointer;font-size:12px;
}
#export-btn:hover { background:#3a2e1a; border-color:#aa8030; }
#export-btn:disabled { opacity:.4; cursor:default; }

/* Marked stats span in header */
.s-mk { color: #ffd040; }
/* Viewed-in-lightbox marker on the card (top-left eye) */
.viewed-badge {
  position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,.65); border-radius: 50%;
  width: 18px; height: 18px;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; color: #7ec8e3;
  pointer-events: none; z-index: 3;
}
/* Lightbox thumb that has been marked for deletion → dim like the card */
.lb-thumb.hidden-pending { opacity: .35; border-color: #803030 !important; }
.lb-thumb.hidden-pending img { opacity: .45 !important; }
/* Disagreement value colour in lb-info */
#lb-info .lb-v.disagree-hi  { color: #ff7878; }
#lb-info .lb-v.disagree-mid { color: #ffd040; }
#lb-info .lb-v.disagree-lo  { color: #50c860; }
.card.hidden-pending .hide-btn { opacity: 1; background: rgba(180,40,40,.7); color: #fff; }

/* source badge top-left */
.src-badge {
  position: absolute; top: 4px; left: 4px; font-size: 9px; padding: 1px 4px;
  border-radius: 2px; pointer-events: none; font-weight: bold; letter-spacing: .3px;
}
.src-badge.ls    { background: rgba(30,70,120,.85); color: #7ab0dd; }
.src-badge.gr    { background: rgba(60,20,100,.85); color: #c792ea; }
.src-badge.k30   { background: rgba(20,40,100,.85); color: #88aaff; }
.disagree-badge {
  position: absolute; top: 4px; right: 4px;
  background: rgba(180,80,20,.92); color: #ffd0a0;
  border: 1px solid #aa6620; border-radius: 3px;
  font-size: 10px; padding: 1px 4px; font-family: monospace;
  pointer-events: none; z-index: 4;
}
#lgbm-disagree-lbl.active { background: #3a2010; border-color: #aa6620; color: #ffc080; }
/* pipeline badge bottom-left */
.pipe-badge {
  position: absolute; bottom: 4px; left: 4px; font-size: 9px; padding: 1px 5px;
  border-radius: 2px; pointer-events: none;
}
.pipe-badge.underage { background: rgba(120,30,30,.85); color: #ffa0a0; }
.pipe-badge.other    { background: rgba(80,55,10,.85);  color: #ffd080; }
.pipe-badge.passed   { background: rgba(15,60,20,.85);  color: #80e080; }
/* confirmation badge bottom-right */
.confirm-badge {
  position: absolute; bottom: 4px; right: 4px; font-size: 9px; padding: 1px 5px;
  border-radius: 2px; pointer-events: none; letter-spacing: .2px;
}
.confirm-badge.auto  { background: rgba(80,50,0,.85);  color: #ffa040; }
.confirm-badge.human { background: rgba(10,60,10,.85); color: #60d060; }

/* info row */
.info {
  padding: 4px 7px 2px; border-bottom: 1px solid #1f1f1f;
  display: flex; justify-content: space-between; align-items: center; gap: 4px;
}
.info .iid { color: #777; font-size: 10px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:100px; }
.info .iid a { color: #c792ea; text-decoration: none; }
.info .iid a:hover { text-decoration: underline; }
.info .imeta { color: #555; font-size: 10px; white-space: nowrap; }

/* prompt row */
.prompt-row {
  padding: 2px 7px 3px; font-size: 10px; color: #3a3a3a;
  white-space: nowrap; overflow:hidden; text-overflow:ellipsis;
  border-bottom: 1px solid #1a1a1a; cursor: pointer;
}
.prompt-row:hover { color: #999; background: #141414; }
.prompt-row.has-prompt { color: #555; }
.prompt-row.has-prompt:hover { color: #aaa; }

/* prompt modal */
#pm {
  display: none; position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,.75); align-items: center; justify-content: center;
}
#pm.open { display: flex; }
#pm-box {
  background: #1a1a1a; border: 1px solid #444; border-radius: 6px;
  padding: 18px 22px; max-width: 640px; width: 90vw; max-height: 70vh;
  overflow-y: auto; position: relative;
}
#pm-box h3 {
  font-size: 11px; color: #666; font-weight: normal; margin-bottom: 10px;
  text-transform: uppercase; letter-spacing: .5px;
}
#pm-text {
  font-size: 12px; color: #ccc; line-height: 1.7; white-space: pre-wrap;
  word-break: break-word; user-select: text;
}
#pm-close {
  position: absolute; top: 10px; right: 12px; background: none; border: none;
  color: #555; font-size: 16px; cursor: pointer; line-height: 1;
}
#pm-close:hover { color: #ccc; }

/* label radios */
.radios { display: flex; padding: 4px 7px 5px; }
.radios label {
  flex:1; text-align:center; padding: 4px 1px; cursor:pointer;
  border: 1px solid #2a2a2a; font-size: 10px; color: #666;
  transition: background .1s;
}
.radios label:first-child { border-radius: 4px 0 0 4px; }
.radios label:last-child  { border-radius: 0 4px 4px 0; }
.radios label:not(:first-child) { border-left: none; }
.radios input { display: none; }
.radios label.cat-child:has(input:checked) { background:#3a1010; border-color:#883333; color:#ff8080; font-weight:bold; }
.radios label.cat-teen:has(input:checked)  { background:#3a2e00; border-color:#887733; color:#ffd040; font-weight:bold; }
.radios label.cat-adult:has(input:checked) { background:#0f2210; border-color:#337744; color:#6fda72; font-weight:bold; }


/* lightbox */
#lb { display:none; position:fixed; inset:0; background:rgba(0,0,0,.9); z-index:999; align-items:center; justify-content:center; cursor:zoom-out; flex-direction:column; gap:8px; }
#lb.open { display:flex; }
#lb img  { max-width:90vw; max-height:80vh; object-fit:contain; border-radius:3px; }
#lb-prompt { max-width:80vw; font-size:11px; color:#777; text-align:center; line-height:1.5; }
#lb-wrap { position: relative; display: flex; align-items: center; justify-content: center; }
#lb-counter {
  position: absolute; top: 10px; left: 10px;
  background: rgba(0,0,0,.78); padding: 4px 9px; border-radius: 3px;
  font-family: monospace; font-size: 11px; color: #aaa; pointer-events: none;
}
/* Lightbox: image + info sidebar side-by-side */
#lb-main { display: flex; flex-direction: row; gap: 14px; align-items: stretch;
           max-width: 98vw; max-height: 82vh; justify-content: center; }
#lb-wrap { flex: 0 1 auto; min-width: 0; position: relative;
           display: flex; align-items: center; justify-content: center; }
#lb-wrap img { max-width: 100%; max-height: 82vh; object-fit: contain;
               display: block; border-radius: 3px; }
#lb-info {
  flex: 0 0 290px; align-self: stretch; overflow-y: auto;
  background: rgba(20,20,20,.96); border: 1px solid #2a2a2a;
  padding: 10px 12px; border-radius: 4px;
  font-family: monospace; font-size: 11px; color: #ccc;
  text-align: left; pointer-events: auto;
}
#lb-info .lb-row { display: flex; justify-content: space-between; gap: 10px; margin: 2px 0; line-height: 1.5; }
#lb-info .lb-k { color: #666; text-transform: uppercase; font-size: 9px; align-self: center; letter-spacing: .5px; }
#lb-info .lb-v { color: #ddd; font-weight: bold; }
#lb-info .lb-v.lo { color: #555; font-weight: normal; }
#lb-info .lb-v.score-blocked { color: #ff8060; }    /* lgbm >= thr — would block */
#lb-info .lb-v.score-passed  { color: #50c860; }    /* lgbm <  thr — passes */
#lb-info .lb-lbl-child { color: #ff5c5c; }
#lb-info .lb-lbl-teen  { color: #ffd040; }
#lb-info .lb-lbl-adult { color: #6fda72; }
#lb-info .lb-section   { border-top: 1px solid #2a2a2a; margin-top: 6px; padding-top: 5px; }
#lb-info .lb-badge-del  { color: #ff6060; font-weight: bold; }
#lb-info .lb-badge-conf { color: #6fda72; }
/* Verdict badge — at top of sidebar */
#lb-info .lb-verdict {
  text-align: center; padding: 6px 8px; margin: -2px -4px 8px; border-radius: 3px;
  font-size: 12px; font-weight: bold; letter-spacing: 1px;
}
#lb-info .lb-verdict.underage { background: #4a1818; color: #ff7878; border: 1px solid #802020; }
#lb-info .lb-verdict.ok       { background: #14361b; color: #6fda72; border: 1px solid #2d6c3a; }
#lb-info .lb-verdict.unknown  { background: #2a2a2a; color: #888;   border: 1px solid #444; }
/* Narrow viewport → fallback overlay over image */
@media (max-width: 920px) {
  #lb-main { flex-direction: column; max-height: 92vh; align-items: center; }
  #lb-info { flex: 0 0 auto; max-height: 30vh; width: 280px; }
}

/* Image column = picture + thumbs strip below */
#lb-img-col { display: flex; flex-direction: column; gap: 8px;
              flex: 0 1 auto; min-width: 0; align-items: center; }
#lb-img-col #lb-wrap { flex: 0 1 auto; min-height: 0; }
#lb-img-col #lb-wrap img { max-height: calc(82vh - 76px); }

/* Thumbs strip */
#lb-thumbs {
  flex: 0 0 68px;
  width: min(70vw, 1100px);
  overflow: hidden;
  position: relative;
  background: rgba(20,20,20,.55);
  border: 1px solid #2a2a2a;
  border-radius: 4px;
  mask-image: linear-gradient(to right, transparent 0, #000 8%, #000 92%, transparent 100%);
  -webkit-mask-image: linear-gradient(to right, transparent 0, #000 8%, #000 92%, transparent 100%);
}
#lb-thumbs-track {
  display: flex; gap: 4px; padding: 6px 4px;
  height: 100%; position: absolute; left: 50%; top: 0;
  transition: transform 0.22s cubic-bezier(.25,.46,.45,.94);
  will-change: transform;
}
.lb-thumb {
  flex: 0 0 56px; height: 56px; cursor: pointer;
  border: 2px solid transparent; border-radius: 2px; overflow: hidden;
  background: #111; position: relative;
  transition: border-color 0.15s, transform 0.15s;
}
.lb-thumb img { width: 100%; height: 100%; object-fit: cover; display: block;
                pointer-events: none; opacity: .55; transition: opacity 0.15s; }
.lb-thumb:hover img    { opacity: .85; }
.lb-thumb.current      { border-color: #5aa0ff; transform: scale(1.08); z-index: 2; }
.lb-thumb.current img  { opacity: 1; }
/* Tiny status dot in thumb corner */
.lb-thumb .lb-thumb-dot {
  position: absolute; top: 2px; right: 2px; width: 7px; height: 7px;
  border-radius: 50%; box-shadow: 0 0 2px #000;
}
.lb-thumb-dot.child { background: #ff5c5c; }
.lb-thumb-dot.teen  { background: #ffd040; }
.lb-thumb-dot.adult { background: #6fda72; }
.lb-thumb-dot.del   { background: #ff6060; box-shadow: 0 0 3px #ff0000; }
@media (max-width: 920px) {
  #lb-thumbs { width: 92vw; flex: 0 0 56px; }
  .lb-thumb { flex: 0 0 44px; height: 44px; }
}
#lb-hints {
  position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,.78); padding: 3px 12px; border-radius: 3px;
  font-family: monospace; font-size: 10px; color: #888;
  white-space: nowrap; pointer-events: none; letter-spacing: 0.5px;
}
.lb-flash      { animation: lbFlash 0.28s ease-out; }
.lb-flash-del  { animation: lbFlashDel 0.28s ease-out; }
@keyframes lbFlash    { 0% { background: rgba(80,200,120,.5); } 100% { background: rgba(0,0,0,.78); } }
@keyframes lbFlashDel { 0% { background: rgba(220,80,80,.55); } 100% { background: rgba(0,0,0,.78); } }


/* pagination */
#pagination { display:flex; align-items:center; justify-content:center; gap:5px; padding:12px 16px; flex-wrap:wrap; }
#pagination button { background:#1e1e1e; border:1px solid #444; color:#aaa; padding:3px 9px; border-radius:3px; cursor:pointer; font-size:11px; font-family:monospace; }
#pagination button:hover { background:#2a2a2a; color:#fff; }
#pagination button.active { background:#2a3a5a; border-color:#4a6aaa; color:#88aaff; font-weight:bold; }
#pagination button:disabled { opacity:.3; cursor:default; }
#pagination .pg-info { color:#555; font-size:10px; padding:0 5px; }


/* ── eval overlay ────────────────────────────────────────────────────────── */
.eval-row {
  display: none; padding: 4px 7px 5px; border-top: 1px solid #1a1a1a;
  flex-direction: column; gap: 3px;
}
.eval-row.visible { display: flex; }

.eval-ver-btns { display: flex; gap: 3px; }
.eval-ver-btns button {
  flex:1; padding: 2px 2px; font-size: 9px; font-family: monospace;
  border-radius: 3px; cursor: pointer; border: 1px solid #2a2a2a;
  background: #181818; color: #444; transition: background .1s;
}
.eval-ver-btns button.active { background: #1a2a3a; border-color: #3a6a9a; color: #7ab0dd; font-weight: bold; }
.eval-ver-btns button:disabled { opacity: .3; cursor: default; }
.eval-ver-btns button.has-data { color: #777; }

/* compact comparison table: V5 | V6 side-by-side */
.eval-cmp {
  display: grid; grid-template-columns: 24px 1fr 1fr; gap: 0 4px;
  font-size: 9px; line-height: 1.55;
}
.eval-cmp.three-col {
  grid-template-columns: 24px 1fr 1fr 1fr;
}
.eval-cmp.four-col {
  grid-template-columns: 24px 1fr 1fr 1fr 1fr;
}
.eval-cmp .cmp-hdr.k30tom { color: #ff9050; }
.eval-cmp .cmp-hdr     { color: #c792ea; font-weight: bold; text-align: center; font-size: 9px; }  /* V8 — purple */
.eval-cmp .cmp-hdr.v11 { color: #ffd060; }   /* V11 — gold */
.eval-cmp .cmp-hdr.v6  { color: #7ec8e3; }   /* V6 — cyan */
.eval-cmp .cmp-hdr.v6, .eval-cmp .cmp-hdr.v11 { /* keep both selectors active */ }
.eval-cmp .cmp-lbl { color: #444; text-transform: uppercase; font-size: 8px; align-self: center; }
.eval-cmp .cmp-val { text-align: center; font-weight: bold; font-size: 11px; }
.cmp-val.hi-lgbm  { color: #ff8060; }            /* score >= thr → orange (blocked) */
.cmp-val.ok-lgbm  { color: #50c860; }            /* score < thr, data exists → green (passes) */
.cmp-val.hi-minor { color: #ffa040; }
.cmp-val.lo       { color: #555; }               /* no data — grey */
.eval-cmp .cmp-outcome { text-align: center; }
.eval-cmp .cmp-sep { grid-column: 1/-1; border-top: 1px solid #1e1e1e; margin: 1px 0; }

.eval-scores {
  display: flex; gap: 5px; align-items: center; font-size: 10px;
}
.eval-score-item { display: flex; flex-direction: column; align-items: center; gap: 1px; }
.eval-score-item .label { font-size: 8px; color: #444; text-transform: uppercase; }
.eval-score-item .value { font-size: 11px; font-weight: bold; }
.eval-score-item .value.hi-lgbm { color: #ff8060; }
.eval-score-item .value.hi-minor { color: #ffa040; }
.eval-score-item .value.lo { color: #555; }

.eval-outcome {
  display: flex; align-items: center; justify-content: space-between; gap: 4px;
}
.eval-outcome .outcome-badge {
  font-size: 10px; font-weight: bold; padding: 1px 6px; border-radius: 3px;
}
.outcome-TP { background: rgba(180,40,40,.25); border: 1px solid #883333; color: #ff8080; }
.outcome-TN { background: rgba(20,80,20,.25); border: 1px solid #337744; color: #70da80; }
.outcome-FP { background: rgba(180,40,180,.25); border: 1px solid #884488; color: #dd80dd; }
.outcome-FN { background: rgba(40,40,180,.25); border: 1px solid #334488; color: #8090ff; }
.outcome-NA { background: rgba(60,60,60,.2); border: 1px solid #333; color: #555; }
.eval-outcome .which-badge {
  font-size: 9px; color: #888; background: rgba(255,255,255,.04);
  padding: 1px 4px; border-radius: 2px; border: 1px solid #2a2a2a;
}
/* ── LGBM threshold bar — collapsible ────────────────────────────────────── */
#lgbm-toggle-btn {
  background: #0e0e2a; border: 1px solid #3a3a8a; color: #8080cc;
  padding: 3px 10px; border-radius: 4px; cursor: pointer;
  font-size: 11px; font-family: monospace; margin-left: 8px;
}
#lgbm-toggle-btn:hover { background: #1a1a3a; border-color: #5a5aaa; color: #b0b0ff; }
#lgbm-toggle-btn.active { background: #1f1f4a; border-color: #6060c0; color: #c0c0ff; }
#lgbm-bar {
  display: none;  /* collapsed by default; toggled via #lgbm-toggle-btn */
  align-items: stretch; gap: 18px; flex-wrap: wrap;
  padding: 8px 14px; background: #0e0e1a; border-bottom: 1px solid #222244;
  font-size: 11px;
}
#lgbm-bar.open { display: flex; }
#lgbm-bar .lgbm-thr-block {
  display: flex;
  align-items: center;
  gap: 14px;
}
#lgbm-bar .lgbm-thr-sliders {
  display: grid;
  grid-template-rows: 18px repeat(3, 22px);
  grid-template-columns: max-content;
  row-gap: 2px;
  align-content: start;
}
.lgbm-thr-title { font-size: 10px; color: #5a5aaa; font-weight: bold;
                  text-transform: uppercase; letter-spacing: .5px;
                  display: flex; align-items: center; }
#lgbm-bar .lgbm-thr-row {
  display: grid;
  grid-template-columns: 38px 130px 44px auto;
  align-items: center;
  gap: 8px;
  height: 22px;
}
.lgbm-thr-buttons {
  display: flex; flex-direction: column; gap: 4px; align-self: center;
}
#lgbm-save-thr-btn, #lgbm-reset-thr-btn {
  background: #0e0e2a; border: 1px solid #3a3a8a; color: #8080cc;
  padding: 4px 12px; border-radius: 4px; cursor: pointer;
  font-size: 11px; font-family: monospace; white-space: nowrap;
}
#lgbm-reset-thr-btn { border-color: #6a4a3a; color: #cc9080; }
#lgbm-reset-thr-btn:hover { background: #2a1a0e; border-color: #aa6a5a; color: #ffb090; }
.lgbm-holdout-toggle {
  display: flex; align-items: center; gap: 4px;
  font-size: 10px; color: #a0c0ee; cursor: pointer;
  padding: 4px 6px; border: 1px solid #2a3a5a; border-radius: 4px;
  background: #0e1424;
}
.lgbm-holdout-toggle input { margin: 0; cursor: pointer; }
.lgbm-holdout-toggle.active { background: #1a2a4a; border-color: #4a6aaa; color: #b0d0ff; }
#lgbm-save-thr-btn:hover { background: #1a1a3a; border-color: #5a5aaa; color: #b0b0ff; }
#lgbm-save-thr-btn.dirty { border-color: #aa8030; color: #ffc060; }
.lgbm-thr-auc {
  font-family: monospace; font-size: 10px; color: #8090b0;
  white-space: nowrap;
}
.lgbm-thr-auc .auc-cv {
  display: inline-block; margin-left: 2px; padding: 0 4px;
  background: #1e2a4a; border-radius: 3px; color: #a0c0ee; font-size: 9px;
}
.lgbm-thr-auc .auc-delta-pos { color: #80e080; }
.lgbm-thr-auc .auc-delta-neg { color: #ff8080; }
.lgbm-bar-title {
  font-size: 10px; color: #5a5aaa; font-weight: bold; text-transform: uppercase;
  letter-spacing: .5px; white-space: nowrap;
}
.lgbm-thr-group { display: flex; align-items: center; gap: 5px; }
.lgbm-thr-label { font-size: 10px; color: #5a5a8a; white-space: nowrap; min-width: 32px; }
.lgbm-thr-input {
  width: 52px; background: #16162a; border: 1px solid #333366; color: #9090dd;
  padding: 2px 5px; border-radius: 3px; font-size: 11px; font-family: monospace; text-align:center;
}
.lgbm-thr-slider {
  -webkit-appearance: none; appearance: none;
  width: 130px; height: 4px;
  background: linear-gradient(to right, #3a3a8a 0%, #6060c0 100%);
  border-radius: 2px; outline: none; cursor: pointer;
}
.lgbm-thr-slider::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 14px; height: 14px; border-radius: 50%;
  background: #c0c0ff; border: 1px solid #6060c0; cursor: pointer;
  box-shadow: 0 0 4px rgba(96,96,192,.6);
}
.lgbm-thr-slider::-moz-range-thumb {
  width: 14px; height: 14px; border-radius: 50%;
  background: #c0c0ff; border: 1px solid #6060c0; cursor: pointer;
}
.lgbm-thr-val {
  display: inline-block; min-width: 34px; text-align: center;
  font-family: monospace; font-size: 11px; color: #b0b0ff;
  background: #16162a; border: 1px solid #333366;
  padding: 1px 4px; border-radius: 3px;
}
#lgbm-apply-btn {
  background: #0e0e2a; border: 1px solid #3a3a8a; color: #8080cc;
  padding: 3px 12px; border-radius: 4px; cursor: pointer; font-size: 11px; font-family: monospace;
}
#lgbm-apply-btn:hover { background: #1a1a3a; border-color: #5a5aaa; color: #b0b0ff; }
#lgbm-stats {
  display: grid; vertical-align: middle;
  grid-template-columns: auto repeat(3, minmax(96px, auto));
  grid-template-rows: 18px repeat(3, 22px);
  align-items: center;
  column-gap: 8px; row-gap: 2px;
  margin-right: 8px;
  font-size: 11px; font-family: monospace;
}
#lgbm-stats .lst-hdr      { font-weight: bold; text-align: center; padding: 1px 4px; border-bottom: 1px solid #2a2a2a; }
#lgbm-stats .lst-rowlbl   { color: #888; text-transform: uppercase; font-size: 9px; padding-right: 4px; align-self: center; }
#lgbm-stats .lst-cell {
  white-space: nowrap;
  padding: 0 4px;
  text-align: left;
}
#lgbm-stats .lst-cell > * { display: inline-block; vertical-align: middle; }
#lgbm-stats .lst-cell .lst-best,
#lgbm-stats .lst-cell .lst-star-spacer {
  width: 14px; text-align: center; color: #80ff80; font-size: 11px; line-height: 1;
}
#lgbm-stats .lst-cell .lst-pct {
  width: 48px; text-align: right; font-weight: bold;
}
#lgbm-stats .lst-cell .lst-dot {
  width: 8px; height: 8px; border-radius: 50%; margin: 0 6px;
}
#lgbm-stats .lst-cell .lst-abs {
  color: #666; font-size: 9px; min-width: 56px; text-align: left;
}
#lgbm-stats .lst-cell .lst-fb { color: #ffa040; font-weight: normal; font-size: 9px; vertical-align: super; cursor: help; }
#lgbm-stats .lst-good     { background: #50c860; }   /* green */
#lgbm-stats .lst-mid      { background: #ffc060; }   /* amber */
#lgbm-stats .lst-bad      { background: #ff6060; }   /* red */
#lgbm-stats .lst-best     { color: #80ff80; }
#lgbm-stats .lst-worst    { color: #ff8080; }
.lgbm-badge {
  position: absolute; bottom: 24px; left: 4px; font-size: 9px; padding: 1px 4px;
  border-radius: 2px; pointer-events: none; z-index: 3;
  background: rgba(10,10,30,.82); border: 1px solid #222244;
  display: flex; gap: 4px; align-items: center;
}
/* Per-model color scheme (sync with cmp-hdr): V8=purple, V11=gold, V6=cyan */
.lgbm-v8-blocked  { color: #c792ea; font-weight: bold; }
.lgbm-v8-ok       { color: #a070c0; font-weight: bold; }
.lgbm-v11-blocked { color: #ffd060; font-weight: bold; }
.lgbm-v11-ok      { color: #c0a050; font-weight: bold; }
.lgbm-v6-blocked  { color: #7ec8e3; font-weight: bold; }
.lgbm-tom-blocked { color: #ff9050; font-weight: bold; }
.lgbm-v6-ok       { color: #5a98b3; font-weight: bold; }
</style>
</head>
<body>

<header>
  <h1>Gallery <span id="subtitle" style="color:#555"></span></h1>

  <label style="display:flex;align-items:center;gap:4px">Источник
    <select id="f-source">
      <option value="all">Все</option>
      <option value="labelstudio">Label Studio</option>
      <option value="grafana">Grafana (все)</option>
      <option value="k30">K30 (Tom)</option>
      <option value="marked">⭐ Отмеченные</option>
    </select>
  </label>

  <label id="session-wrap" style="display:flex;align-items:center;gap:4px">Сессия
    <select id="f-session" style="max-width:170px"><option value="all">все сессии</option></select>
  </label>

  <label style="display:flex;align-items:center;gap:4px">Разметка
    <select id="f-label">
      <option value="all">Все</option>
      <option value="unlabeled">Без разметки</option>
      <option value="unconfirmed">⚡ Не подтверждено</option>
      <option value="child">child</option>
      <option value="teen">teen</option>
      <option value="adult">adult</option>
    </select>
  </label>


  <label style="display:flex;align-items:center;gap:4px">Возраст q3
    <select id="f-age-q3">
      <option value="all">все</option>
      <option value="0-14">≤14</option>
      <option value="15-17">15–17</option>
      <option value="0-17">≤17</option>
      <option value="18+">≥18</option>
      <option value="none">нет данных</option>
    </select>
  </label>

  <label style="display:flex;align-items:center;gap:4px">Возраст fd
    <select id="f-age-fd">
      <option value="all">все</option>
      <option value="0-9">≤9</option>
      <option value="10-19">10–19</option>
      <option value="0-19">≤19</option>
      <option value="20+">≥20</option>
      <option value="none">нет данных</option>
    </select>
  </label>

  <label style="display:flex;align-items:center;gap:4px">На стр.
    <select id="f-pgsize">
      <option value="50">50</option>
      <option value="100" selected>100</option>
      <option value="200">200</option>
      <option value="500">500</option>
    </select>
  </label>

  <label id="eval-wrap" style="display:flex;align-items:center;gap:4px">
    <input type="checkbox" id="eval-toggle" onchange="toggleEval(this.checked)"> LGBM
    <select id="f-eval-ver" style="display:none">
      <option value="v8">V8pas80</option>
      <option value="v11">V11s80</option>
      <option value="v6">V6 (legacy)</option>
    </select>
    <select id="f-eval-outcome" style="display:none">
      <option value="all">Все</option>
      <option value="TP">TP (верно заблок.)</option>
      <option value="FN">FN (пропущено ⚠)</option>
      <option value="TN">TN (верно пропущ.)</option>
      <option value="FP">FP (ложная тревога)</option>
      <option value="NA">Нет данных</option>
    </select>
  </label>


  <span id="status"></span>
  <button id="confirm-btn" onclick="confirmPage()" title="Подтвердить все AI-метки на текущей странице без изменений">✓ Подтвердить страницу</button>
  <button id="save-btn" onclick="saveChanges()">💾 Сохранить</button>
  <button id="export-btn" onclick="exportMarked()" disabled title="Экспортировать отмеченные изображения в JSON">📥 Экспорт ⭐ (<span id="export-cnt">0</span>)</button>
  <button id="lgbm-toggle-btn" onclick="toggleLgbmBar()" title="LGBM пороги и статистика">⚙ LGBM</button>
</header>

<div id="lgbm-bar">
  <div id="lgbm-stats" title="V11 на items с native 317-tag taxonomy (fallback-скоры исключены, их видно на карточках со звёздочкой)"></div>
  <div class="lgbm-thr-block">
    <div class="lgbm-thr-sliders">
      <div class="lgbm-bar-title lgbm-thr-title">Пороги</div>
      <div class="lgbm-thr-row" data-ver="v6">
        <span class="lgbm-thr-label">V6 ≥</span>
        <input type="range" id="thr-v6" class="lgbm-thr-slider" value="0.30" min="0.01" max="1" step="0.01">
        <span class="lgbm-thr-val" id="thr-v6-val">0.30</span>
        <span class="lgbm-thr-auc" id="auc-v6" title="ROC AUC (static) | Op = balanced acc at THR | Δ vs saved baseline">AUC: — | Op: —</span>
      </div>
      <div class="lgbm-thr-row" data-ver="v8">
        <span class="lgbm-thr-label">V8 ≥</span>
        <input type="range" id="thr-v8" class="lgbm-thr-slider" value="0.30" min="0.01" max="1" step="0.01">
        <span class="lgbm-thr-val" id="thr-v8-val">0.30</span>
        <span class="lgbm-thr-auc" id="auc-v8" title="ROC AUC (static) | Op = balanced acc at THR | Δ vs saved baseline">AUC: — | Op: —</span>
      </div>
      <div class="lgbm-thr-row" data-ver="v11">
        <span class="lgbm-thr-label">V11 ≥</span>
        <input type="range" id="thr-v11" class="lgbm-thr-slider" value="0.30" min="0.01" max="1" step="0.01">
        <span class="lgbm-thr-val" id="thr-v11-val">0.30</span>
        <span class="lgbm-thr-auc" id="auc-v11" title="ROC AUC (static) | Op = balanced acc at THR | Δ vs saved baseline">AUC: — | Op: —</span>
      </div>
      <div class="lgbm-thr-row" data-ver="k30tom">
        <span class="lgbm-thr-label" style="color:#ff9050">Tom ≥</span>
        <input type="range" id="thr-k30tom" class="lgbm-thr-slider" value="0.10" min="0.01" max="1" step="0.01">
        <span class="lgbm-thr-val" id="thr-k30tom-val">0.10</span>
        <span class="lgbm-thr-auc" id="auc-k30tom" title="ROC AUC of Tom\'s K=30 model | Op = balanced acc at THR | Δ vs saved baseline">AUC: — | Op: —</span>
      </div>
    </div>
    <div class="lgbm-thr-buttons">
      <button id="lgbm-save-thr-btn" onclick="saveLgbmThresholds()" title="Сохранить текущие пороги как baseline">💾 Сохранить</button>
      <button id="lgbm-reset-thr-btn" onclick="resetLgbmThresholds()" title="Сбросить ползунки к сохранённым значениям">↺ Сбросить</button>
      <label class="lgbm-holdout-toggle" title="Показывать статистику только на 20% holdout-сете (V11 не видел эти items при обучении)">
        <input type="checkbox" id="lgbm-holdout-chk" onchange="onHoldoutToggle()">
        📊 Holdout
      </label>
      <label class="lgbm-holdout-toggle" id="lgbm-disagree-lbl" title="Сортировать карточки по убыванию расхождения V6/V8/V11. Топовые items — самые информативные для разметки (active learning)">
        <input type="checkbox" id="lgbm-disagree-chk" onchange="onDisagreeToggle()">
        🔥 Disagreement
      </label>
    </div>
  </div>
</div>

<div class="stats-bar" id="stats-bar">—</div>
<div class="gallery" id="gallery"></div>
<div id="pagination"></div>

<div id="lb" onclick="closeLb()">
  <div id="lb-main" onclick="event.stopPropagation()">
    <div id="lb-img-col">
      <div id="lb-wrap">
        <img id="lb-img" src="">
        <div id="lb-counter"></div>
        <div id="lb-hints">1 child &nbsp; 2 teen &nbsp; 3 adult &nbsp; 4 del &nbsp; 5 star &nbsp; Enter next &nbsp; Esc close</div>
      </div>
      <div id="lb-thumbs"><div id="lb-thumbs-track"></div></div>
    </div>
    <div id="lb-info"></div>
  </div>
  <div id="lb-prompt"></div>
</div>
<div id="pm" onclick="closePm()">
  <div id="pm-box" onclick="event.stopPropagation()">
    <button id="pm-close" onclick="closePm()">✕</button>
    <h3>Промпт</h3>
    <div id="pm-text"></div>
  </div>
</div>

<script>
let allData = [], sessions = [], changes = {}, hidden = new Set(), toConfirm = new Set(), viewedIds = new Set(), filteredData = [], currentPage = 1;

// Marked-for-followup items — persisted across reloads via localStorage.
let marked = (() => {
  try { return new Set(JSON.parse(localStorage.getItem('marked') || '[]')); }
  catch { return new Set(); }
})();
function updateExportBtn() {
  const btn = document.getElementById('export-btn');
  const cnt = document.getElementById('export-cnt');
  if (!btn || !cnt) return;
  cnt.textContent = marked.size;
  btn.disabled = marked.size === 0;
}

function _exportPayloadFor(r) {
  const e = (typeof evalData === 'object') ? (evalData[r.id] || {}) : {};
  const m = {};
  for (const v of ['v6','v8','v11','k30tom']) {
    const x = e[v];
    if (x && x.lgbm != null) {
      m[v] = {
        lgbm:       +x.lgbm.toFixed(4),
        threshold:  +lgbmThrFor(v).toFixed(2),
        blocked:    x.lgbm >= lgbmThrFor(v),
        fallback:   !!x.fallback_taxonomy,
      };
    }
  }
  let q3 = null;
  try {
    const t = _lbQwen3Age(r);
    if (t) q3 = t;
  } catch (err) {}
  let fd = null;
  try {
    const t = faceAge(r);
    if (t) fd = t;
  } catch (err) {}
  return {
    id:              r.id,
    source:          r.source,
    thumb_url:       r.thumb_url || r._serve_url || null,
    label:           effectiveLabel(r) || null,
    label_confirmed: !!r.label_confirmed,
    deletion_pending: hidden.has(r.id),
    age_qwen3:       q3,
    age_face_detect: fd,
    age_label_studio: (r.source === 'labelstudio' && r.ageFrom != null)
                      ? `${r.ageFrom}-${r.ageTo != null ? r.ageTo : r.ageFrom}` : null,
    prompt:          r.prompt || null,
    models:          m,
  };
}

function exportMarked() {
  if (!marked.size) return;
  const items = [];
  for (const id of marked) {
    const r = allData.find(x => x.id === id);
    if (!r) continue;
    items.push(_exportPayloadFor(r));
  }
  const payload = {
    exported_at: new Date().toISOString(),
    count: items.length,
    items: items,
  };
  const json = JSON.stringify(payload, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
  a.href = url; a.download = `marked_${stamp}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function persistMarked() {
  try { localStorage.setItem('marked', JSON.stringify([...marked])); } catch {}
}
function toggleMark(id) {
  if (!id) return;
  if (marked.has(id)) marked.delete(id); else marked.add(id);
  persistMarked();
  // sync UI: card class + button look
  const cardEl = document.getElementById('card-' + id);
  if (cardEl) {
    cardEl.classList.toggle('marked', marked.has(id));
    const btn = cardEl.querySelector('.mark-btn');
    if (btn) btn.textContent = marked.has(id) ? '★' : '☆';
  }
  // sync thumb in lightbox if open
  const track = document.getElementById('lb-thumbs-track');
  if (track) {
    const idx = lbPageItems.findIndex(x => x.id === id);
    if (idx >= 0 && track.children[idx]) {
      track.children[idx].classList.toggle('marked', marked.has(id));
    }
  }
  updateStats();
  updateExportBtn();
}
let evalData = {}, evalActive = false, evalVersion = 'v8';

// ── live sim ──────────────────────────────────────────────────────────────────
// ── LGBM thresholds & badge logic ────────────────────────────────────────────
let lgbmV6Thr    = 0.30;
let lgbmV8Thr    = 0.30;
let lgbmV11Thr   = 0.30;
let lgbmK30tomThr = 0.10;   // Tom's K=30 default threshold

function lgbmThrFor(version) {
  if (version === 'v8')     return lgbmV8Thr;
  if (version === 'v11')    return lgbmV11Thr;
  if (version === 'k30tom') return lgbmK30tomThr;
  return lgbmV6Thr;
}

function lgbmIsBlocked(entry, version) {
  if (!entry) return null;
  return entry.lgbm >= lgbmThrFor(version);
}

// ── AUC + operating-point score, threshold save/load ─────────────────────────
// Baseline THR: defaults until loaded from server (/api/get_thr) or localStorage.
let lgbmBaseThr = { v6: 0.30, v8: 0.30, v11: 0.30, k30tom: 0.10 };

// Static AUC per version, computed once after evalData is ready.
let modelAuc = { v6: null, v8: null, v11: null, k30tom: null };
let modelMeta = { v6: null, v8: null, v11: null };
let holdoutMode = false;
let holdoutIds  = new Set();   // ids of items in V11 test split (filled at init)
let holdoutMeta = null;
function isInHoldout(r) {
  // Test items use the same id format that was saved: ls_XXX or grafana UUIDs.
  return holdoutIds.has(r.id);
}

function _itemCat(r) {
  // grafana and k30 carry a label string directly (qwen3 draft for k30, human/AI for grafana)
  if (r.source === 'grafana' || r.source === 'k30') return (r.label || '').toLowerCase();
  // LS items expose ageFrom directly as a flat field (load_ls in gallery_server.py)
  const af = (r.ageFrom != null) ? r.ageFrom : (r.age || {}).ageFrom;
  if (af == null) return '';
  if (af <= 14) return 'child';
  if (af <= 17) return 'teen';
  return 'adult';
}
function _sourceMatch(r) {
  if (holdoutMode && !isInHoldout(r)) return false;
  const sel = (document.getElementById('f-source') || {}).value || 'all';
  if (sel === 'labelstudio') return r.source === 'labelstudio';
  if (sel === 'grafana')     return r.source === 'grafana';
  if (sel === 'k30')         return r.source === 'k30';
  return true;
}

// Mann-Whitney-U based ROC-AUC (positive = child+teen, negative = adult).
function computeAucForVersion(ver) {
  const pos = [], neg = [];
  allData.forEach(r => {
    if (!_sourceMatch(r)) return;
    const e = (evalData[r.id] || {})[ver];
    if (!e || e.lgbm == null) return;
    // Include fallback-taxonomy items too — without them V11 on K30 has zero data.
    // Fallback quality issues are signalled by the * marker in stats cells.
    const cat = _itemCat(r);
    if (cat === 'child' || cat === 'teen') pos.push(e.lgbm);
    else if (cat === 'adult')              neg.push(e.lgbm);
  });
  if (!pos.length || !neg.length) return null;
  // Pair sweep: sort+rank for efficiency.
  const all = pos.map(s => [s,1]).concat(neg.map(s => [s,0]));
  all.sort((a,b) => a[0] - b[0]);
  let rank = 0, tieSum = 0, rPosSum = 0;
  // Average ranks for ties
  for (let i = 0; i < all.length;) {
    let j = i;
    while (j+1 < all.length && all[j+1][0] === all[i][0]) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) if (all[k][1] === 1) rPosSum += avgRank;
    i = j + 1;
  }
  const U = rPosSum - pos.length * (pos.length + 1) / 2;
  return U / (pos.length * neg.length);
}

// Balanced accuracy at THR = (TPR + (1-FPR)) / 2 — proxy for "AUC at one point".
function opScoreAt(ver, thr) {
  let p_blk=0, p_tot=0, n_blk=0, n_tot=0;
  allData.forEach(r => {
    if (!_sourceMatch(r)) return;
    const e = (evalData[r.id] || {})[ver];
    if (!e || e.lgbm == null) return;
    // Include fallback too — same reasoning as in computeAucForVersion above
    const cat = _itemCat(r);
    if (cat === 'child' || cat === 'teen') {
      p_tot++; if (e.lgbm >= thr) p_blk++;
    } else if (cat === 'adult') {
      n_tot++; if (e.lgbm >= thr) n_blk++;
    }
  });
  if (!p_tot || !n_tot) return null;
  const tpr = p_blk / p_tot, fpr = n_blk / n_tot;
  return (tpr + (1 - fpr)) / 2;
}

function refreshAucDisplays() {
  for (const ver of ['v6','v8','v11','k30tom']) {
    const el = document.getElementById('auc-' + ver);
    if (!el) continue;
    const auc = modelAuc[ver];
    const thr = lgbmThrFor(ver);
    const opCur  = opScoreAt(ver, thr);
    const opBase = opScoreAt(ver, lgbmBaseThr[ver]);
    if (auc == null || opCur == null) { el.textContent = 'AUC: — | Op: —'; continue; }
    let html = `AUC: ${auc.toFixed(3)}`;
    // Honest out-of-sample AUC — prefer holdout (multi-seed 80/20) over CV
    const meta = modelMeta[ver];
    if (meta && meta.holdout_auc != null) {
      const stdPart = meta.holdout_auc_std != null ? `±${meta.holdout_auc_std.toFixed(3)}` : '';
      let tip = `Holdout test AUC — 20% items the model never saw at training\n` +
                `n_test=${meta.holdout_n_test || '?'}\n` +
                `child recall=${meta.holdout_child?.toFixed(3) || '—'}\n` +
                `teen  recall=${meta.holdout_teen?.toFixed(3) || '—'}\n` +
                `adult FPR  =${meta.holdout_fpr?.toFixed(3) || '—'}`;
      const bySrc = meta.holdout_by_source;
      if (bySrc) {
        tip += '\n\nPer-source:';
        for (const [src, m] of Object.entries(bySrc)) {
          tip += `\n  ${src} (n=${m.n}): AUC=${m.auc?.toFixed(3) || '—'} child=${m.child_recall?.toFixed(3) || '—'} teen=${m.teen_recall?.toFixed(3) || '—'} fpr=${m.adult_fpr?.toFixed(3) || '—'}`;
        }
      }
      html += ` <span class="auc-cv" title="${tip.replace(/"/g,'&quot;')}">Test: ${meta.holdout_auc.toFixed(3)}${stdPart}</span>`;
    } else if (meta && meta.cv_auc != null) {
      const stdPart = meta.cv_std != null ? `±${meta.cv_std.toFixed(3)}` : '';
      html += ` <span class="auc-cv" title="5-fold cross-validation AUC (legacy CV, no holdout meta yet)">CV: ${meta.cv_auc.toFixed(3)}${stdPart}</span>`;
    }
    html += ` | Op: ${opCur.toFixed(3)}`;
    if (opBase != null) {
      const d = opCur - opBase;
      const sign = d >= 0 ? '+' : '−';
      const cls = d >= 0 ? 'auc-delta-pos' : 'auc-delta-neg';
      html += ` <span class="${cls}">(${sign}${Math.abs(d).toFixed(3)})</span>`;
    }
    el.innerHTML = html;
  }
}

async function loadSavedThresholds() {
  try {
    const r = await fetch('/api/get_thr');
    if (!r.ok) throw new Error('http ' + r.status);
    const t = await r.json();
    if (t && typeof t === 'object') {
      for (const k of ['v6','v8','v11']) {
        if (typeof t[k] === 'number') lgbmBaseThr[k] = t[k];
      }
    }
  } catch (e) {
    // Fall back to localStorage if server unreachable
    try {
      const s = JSON.parse(localStorage.getItem('lgbmThr') || '{}');
      for (const k of ['v6','v8','v11']) {
        if (typeof s[k] === 'number') lgbmBaseThr[k] = s[k];
      }
    } catch {}
  }
  // Apply baseline to sliders + globals
  lgbmV6Thr      = lgbmBaseThr.v6;
  lgbmV8Thr      = lgbmBaseThr.v8;
  lgbmV11Thr     = lgbmBaseThr.v11;
  lgbmK30tomThr  = lgbmBaseThr.k30tom != null ? lgbmBaseThr.k30tom : 0.10;
  for (const ver of ['v6','v8','v11','k30tom']) {
    const inp = document.getElementById('thr-' + ver);
    const lbl = document.getElementById('thr-' + ver + '-val');
    const val = lgbmBaseThr[ver] != null ? lgbmBaseThr[ver] : (ver === 'k30tom' ? 0.10 : 0.30);
    if (inp) inp.value = val;
    if (lbl) lbl.textContent = val.toFixed(2);
  }
}

async function saveLgbmThresholds() {
  const payload = {
    v6:     parseFloat(document.getElementById('thr-v6').value),
    v8:     parseFloat(document.getElementById('thr-v8').value),
    v11:    parseFloat(document.getElementById('thr-v11').value),
    k30tom: parseFloat(document.getElementById('thr-k30tom').value),
  };
  try {
    const r = await fetch('/api/save_thr', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    if (j && j.ok) {
      lgbmBaseThr = { ...payload };
      try { localStorage.setItem('lgbmThr', JSON.stringify(payload)); } catch {}
      refreshAucDisplays();
      const btn = document.getElementById('lgbm-save-thr-btn');
      if (btn) {
        btn.classList.remove('dirty');
        const orig = btn.textContent;
        btn.textContent = '✓ Сохранено';
        setTimeout(() => { btn.textContent = orig; }, 1200);
      }
    } else {
      alert('Save failed: ' + (j.error || 'unknown'));
    }
  } catch (e) {
    alert('Save failed: ' + e.message);
  }
}

let disagreeMode = false;
function disagreementScore(r) {
  const e = evalData[r.id] || {};
  // Prefer "our selected model vs Tom's k30 model" when Tom data is present —
  // works for any source (K30 has Tom built-in; LS/Grafana via tom_scores.json).
  const tom  = e['k30tom'];
  const ours = e[evalVersion];
  if (tom && tom.lgbm != null && ours && ours.lgbm != null) {
    return Math.abs(ours.lgbm - tom.lgbm);
  }
  // Fallback: max-min of V6/V8/V11
  const vals = [];
  for (const v of ['v6','v8','v11']) {
    const x = e[v];
    if (x && x.lgbm != null) vals.push(x.lgbm);
  }
  if (vals.length < 2) return 0;
  return Math.max(...vals) - Math.min(...vals);
}
function onDisagreeToggle() {
  const chk = document.getElementById('lgbm-disagree-chk');
  disagreeMode = chk.checked;
  const lbl = document.getElementById('lgbm-disagree-lbl');
  if (lbl) lbl.classList.toggle('active', disagreeMode);
  applyFilter();
}

function onHoldoutToggle() {
  const chk = document.getElementById('lgbm-holdout-chk');
  holdoutMode = chk.checked;
  document.querySelector('.lgbm-holdout-toggle').classList.toggle('active', holdoutMode);
  // Recompute everything that respects source filter
  for (const ver of ['v6','v8','v11']) { modelAuc[ver] = computeAucForVersion(ver); }
  updateLgbmStats();
  refreshAucDisplays();
  applyFilter();   // also constrain visible cards
}

function resetLgbmThresholds() {
  for (const ver of ['v6','v8','v11','k30tom']) {
    const inp = document.getElementById('thr-' + ver);
    const lbl = document.getElementById('thr-' + ver + '-val');
    const val = lgbmBaseThr[ver] != null ? lgbmBaseThr[ver] : (ver === 'k30tom' ? 0.10 : 0.30);
    if (inp) inp.value = val;
    if (lbl) lbl.textContent = val.toFixed(2);
  }
  applyLgbmThresholds();
}

function markThrDirty() {
  const cur = {
    v6:     parseFloat(document.getElementById('thr-v6').value),
    v8:     parseFloat(document.getElementById('thr-v8').value),
    v11:    parseFloat(document.getElementById('thr-v11').value),
    k30tom: parseFloat(document.getElementById('thr-k30tom').value),
  };
  const dirty = ['v6','v8','v11','k30tom'].some(k => Math.abs(cur[k] - (lgbmBaseThr[k]||0)) > 1e-6);
  const btn = document.getElementById('lgbm-save-thr-btn');
  if (btn) btn.classList.toggle('dirty', dirty);
}

function buildLgbmBadge(wrap, r) {
  const edv  = evalData[r.id] || {};
  const ev8  = edv['v8']  || null;
  const ev11 = edv['v11'] || null;
  const ev6  = edv['v6']  || null;
  if (!ev8 && !ev11 && !ev6) return;
  const badge = document.createElement('div');
  badge.className = 'lgbm-badge';
  const parts = [];
  if (ev6) {
    const bl = ev6.lgbm >= lgbmV6Thr;
    parts.push(`<span class="${bl ? 'lgbm-v6-blocked' : 'lgbm-v6-ok'}">${bl ? '⛔' : '✓'} V6:${ev6.lgbm.toFixed(2)}</span>`);
  }
  if (ev8) {
    const bl = ev8.lgbm >= lgbmV8Thr;
    parts.push(`<span class="${bl ? 'lgbm-v8-blocked' : 'lgbm-v8-ok'}">${bl ? '⛔' : '✓'} V8:${ev8.lgbm.toFixed(2)}</span>`);
  }
  if (ev11) {
    const bl = ev11.lgbm >= lgbmV11Thr;
    parts.push(`<span class="${bl ? 'lgbm-v11-blocked' : 'lgbm-v11-ok'}">${bl ? '⛔' : '✓'} V11:${ev11.lgbm.toFixed(2)}</span>`);
  }
  badge.innerHTML = parts.join(' ');
  wrap.appendChild(badge);
}

// ── Pipe-badge — dynamic ok/underage label on image ─────────────────────────
// Rules:
//   - LGBM filter ACTIVE  → show badge for ALL items (including confirmed) based on
//     selected version (V6/V8/V11) and current UI threshold.
//   - LGBM filter INACTIVE → fall back to legacy `pipe_status` snapshot, BUT only
//     for unconfirmed items (AI-labeled). Confirmed items get nothing.
function buildPipeBadge(wrap, r) {
  // Remove any existing badge first (idempotent rebuild)
  wrap.querySelectorAll('.pipe-badge').forEach(el => el.remove());

  if (evalActive) {
    const entry = (evalData[r.id] || {})[evalVersion];
    if (!entry) {
      const pb = document.createElement('div');
      pb.className = 'pipe-badge other';
      pb.textContent = '— нет скоринга';
      wrap.appendChild(pb);
      return;
    }
    const thr = lgbmThrFor(evalVersion);
    const blocked = entry.lgbm >= thr;
    const verLabel = evalVersion.toUpperCase();
    const pb = document.createElement('div');
    pb.className = 'pipe-badge ' + (blocked ? 'underage' : 'passed');
    pb.textContent = blocked
      ? `⛔ ${verLabel} ⩾ ${thr.toFixed(2)}`
      : `✓ ${verLabel} < ${thr.toFixed(2)}`;
    pb.title = `${verLabel} score = ${entry.lgbm.toFixed(3)}`;
    wrap.appendChild(pb);
    return;
  }

  // K30: derive badge from qwen3 age estimate (independent of human label_confirmed)
  // so user can compare q3 draft to manual labels. 18+ → ok, ≤17 → underage.
  if (r.source === 'k30') {
    const q    = r.qwen3_result || {};
    const minA = q.min_age;
    const maxA = q.max_age;
    const pb = document.createElement('div');
    if (minA == null) {
      pb.className = 'pipe-badge other';
      pb.textContent = '⚠ no age';
    } else if (minA >= 18) {
      pb.className = 'pipe-badge passed';
      pb.textContent = '✓ ok';
    } else {
      pb.className = 'pipe-badge underage';
      pb.textContent = '⛔ underage';
    }
    pb.title = `qwen3: min_age=${minA ?? '?'} max_age=${maxA ?? '?'}`;
    wrap.appendChild(pb);
    return;
  }

  // Filter off — legacy snapshot for AI items only
  if (r.label_confirmed) return;
  const ps = pipeStatus(r);
  if (!ps || ps === 'unprocessed') return;
  const pb = document.createElement('div');
  pb.className = 'pipe-badge ' + ps;
  pb.textContent = ps === 'passed' ? '✓ ok' : ps === 'underage' ? '⛔ underage' : '⚠ other';
  wrap.appendChild(pb);
}

function rebuildAllPipeBadges() {
  document.querySelectorAll('.card').forEach(cardEl => {
    const id = cardEl.id.replace('card-', '');
    const item = allData.find(x => x.id === id);
    if (!item) return;
    const wrap = cardEl.querySelector('.img-wrap');
    if (!wrap) return;
    buildPipeBadge(wrap, item);
  });
}

function toggleLgbmBar() {
  const bar = document.getElementById('lgbm-bar');
  const btn = document.getElementById('lgbm-toggle-btn');
  const open = !bar.classList.contains('open');
  bar.classList.toggle('open', open);
  btn.classList.toggle('active', open);
  // Refresh stats when opening (in case data changed)
  if (open) updateLgbmStats();
}

function applyLgbmThresholds() {
  lgbmV8Thr      = parseFloat(document.getElementById('thr-v8').value)      || 0.30;
  lgbmV11Thr     = parseFloat(document.getElementById('thr-v11').value)     || 0.30;
  lgbmV6Thr      = parseFloat(document.getElementById('thr-v6').value)      || 0.30;
  lgbmK30tomThr  = parseFloat(document.getElementById('thr-k30tom').value)  || 0.10;
  rebuildAllPipeBadges();
  refreshAucDisplays();
  markThrDirty();
  document.querySelectorAll('.eval-row').forEach(el => {
    const cid = el.id.replace('eval-', '');
    const item = allData.find(x => x.id === cid);
    if (item) renderEvalRowContent(el, item);
  });
  updateLgbmStats();
}

// Slider drag flow:
//   IMMEDIATE  → update label, push to globals, rebuild pipe-badges, mark dirty.
//   DEBOUNCED  → heavy re-renders (eval-row cells, stats grid, AUC panel).
// Pipe-badge stays in sync with the slider in real time; expensive work is throttled.
let _lgbmDebounce = null;
function onThrSlide(verKey) {
  const inp = document.getElementById('thr-' + verKey);
  const lbl = document.getElementById('thr-' + verKey + '-val');
  const v   = parseFloat(inp.value);
  if (lbl) lbl.textContent = v.toFixed(2);
  // Push to globals NOW so anything reading thr sees the new value
  if (verKey === 'v6')     lgbmV6Thr     = v;
  if (verKey === 'v8')     lgbmV8Thr     = v;
  if (verKey === 'v11')    lgbmV11Thr    = v;
  if (verKey === 'k30tom') lgbmK30tomThr = v;
  rebuildAllPipeBadges();
  markThrDirty();
  clearTimeout(_lgbmDebounce);
  _lgbmDebounce = setTimeout(() => {
    document.querySelectorAll('.eval-row').forEach(el => {
      const cid = el.id.replace('eval-', '');
      const item = allData.find(x => x.id === cid);
      if (item) renderEvalRowContent(el, item);
    });
    updateLgbmStats();
    refreshAucDisplays();
  }, 100);
}

function updateLgbmStats() {
  // Compact 4x4 grid: [metric] | V8 | V11 | V6 — one row per category.
  // Cell shows percentage + status dot + raw counts (small).
  // Coloring rules (absolute, not relative):
  //   recall  (child/teen blocked): ≥95% good, 80-95% mid, <80% bad
  //   FPR     (adult blocked):      ≤5% good,  5-20% mid, >20% bad
  // Best score in each row gets a small ★.
  const cats = ['child', 'teen', 'adult'];
  const totals = {child: 0, teen: 0, adult: 0};
  // Per-version per-category {blk, tot}: tot counts only items WITH a score for that version.
  // Sweep tables use the same denominator, so this keeps gallery numbers comparable.
  const stats = {
    v8:     {child:{blk:0,tot:0,fb:0}, teen:{blk:0,tot:0,fb:0}, adult:{blk:0,tot:0,fb:0}, has: 0},
    v11:    {child:{blk:0,tot:0,fb:0}, teen:{blk:0,tot:0,fb:0}, adult:{blk:0,tot:0,fb:0}, has: 0},
    v6:     {child:{blk:0,tot:0,fb:0}, teen:{blk:0,tot:0,fb:0}, adult:{blk:0,tot:0,fb:0}, has: 0},
    k30tom: {child:{blk:0,tot:0,fb:0}, teen:{blk:0,tot:0,fb:0}, adult:{blk:0,tot:0,fb:0}, has: 0},
  };
  // Respect source filter (f-source): "all" / "labelstudio" / "grafana" / "k30"
  const srcSel = (document.getElementById('f-source') || {}).value || 'all';
  const sourceMatch = (r) => {
    if (srcSel === 'labelstudio') return r.source === 'labelstudio';
    if (srcSel === 'grafana')     return r.source === 'grafana';
    if (srcSel === 'k30')         return r.source === 'k30';
    return true;
  };
  allData.filter(r => {
    if (!sourceMatch(r)) return false;
    if (holdoutMode && !isInHoldout(r)) return false;
    return true;
  }).forEach(r => {
    const cat = (r.label || '').toLowerCase();
    if (!cats.includes(cat)) return;
    totals[cat]++;
    const ed = evalData[r.id] || {};
    for (const ver of ['v8','v11','v6','k30tom']) {
      const e = ed[ver];
      if (!e || e.lgbm == null) continue;
      stats[ver].has++;
      stats[ver][cat].tot++;
      if (e.fallback_taxonomy) {
        stats[ver][cat].fb = (stats[ver][cat].fb || 0) + 1;
      }
      // All four models (incl. Tom) use UI sliders via lgbmThrFor
      if (e.lgbm >= lgbmThrFor(ver)) stats[ver][cat].blk++;
    }
  });

  const dotClass = (cat, pct) => {
    if (cat === 'adult') {  // FPR — lower is better
      if (pct <= 5)   return 'lst-good';
      if (pct <= 20)  return 'lst-mid';
      return 'lst-bad';
    } else {                // recall — higher is better
      if (pct >= 95)  return 'lst-good';
      if (pct >= 80)  return 'lst-mid';
      return 'lst-bad';
    }
  };

  // Transposed: rows = versions (V6, V8, V11), columns = categories (child, teen, adult)
  const versions = ['v6','v8','v11','k30tom'];
  const verNames = {v8: 'V8', v11: 'V11', v6: 'V6', k30tom: 'Tom_K30'};
  const verCls   = {v8: 'lgbm-v8-blocked', v11: 'lgbm-v11-blocked', v6: 'lgbm-v6-blocked', k30tom: 'lgbm-tom-blocked'};
  const catIcons = {child: '👶', teen: '🧒', adult: '🧑'};
  const catLabels = {child: 'child', teen: 'teen', adult: 'adult ✗FP'};
  const html = [];

  // Header row: empty | child | teen | adult
  html.push(`<div class="lst-rowlbl"></div>`);
  for (const cat of cats) {
    html.push(`<div class="lst-hdr">${catIcons[cat]} ${catLabels[cat]}</div>`);
  }

  // Pre-compute best version per category (for ★) — denominator: items with V11/V8/V6 score
  const bestVerPerCat = {};
  for (const cat of cats) {
    let bestVer = null, bestVal = null;
    for (const ver of versions) {
      const s = stats[ver][cat];
      if (!s.tot) continue;
      const sc = s.blk / s.tot * 100;
      if (cat === 'adult') {              // lower is better (FPR)
        if (bestVal == null || sc < bestVal) { bestVal = sc; bestVer = ver; }
      } else {                            // higher is better (recall)
        if (bestVal == null || sc > bestVal) { bestVal = sc; bestVer = ver; }
      }
    }
    bestVerPerCat[cat] = bestVer;
  }

  // Rows for each version
  for (const ver of versions) {
    html.push(`<div class="lst-rowlbl ${verCls[ver]}">${verNames[ver]}</div>`);
    for (const cat of cats) {
      const s = stats[ver][cat];
      if (!s.tot) {
        html.push(`<div class="lst-cell"><span class="lst-star-spacer"></span><span class="lst-pct" style="color:#444">—</span><span></span><span></span></div>`);
        continue;
      }
      const n = s.blk, m = s.tot;
      const pct = n / m * 100;
      const dot = dotClass(cat, pct);
      const isBest = bestVerPerCat[cat] === ver;
      const starSlot = isBest
        ? '<span class="lst-best" title="best in this column">★</span>'
        : '<span class="lst-star-spacer"></span>';
      const colorCls = isBest ? 'lst-best' : '';
      const fb = stats[ver][cat].fb || 0;
      const fbMark = (fb > 0)
        ? `<span class="lst-fb" title="${fb}/${m} scored on off-taxonomy input (fallback) — number is approximate">*</span>`
        : '';
      html.push(
        `<div class="lst-cell" title="${verNames[ver]} on ${cat}: ${n}/${m}${fb>0?` (${fb} fallback)`:''}">`
        + `${starSlot}`
        + `<span class="lst-pct ${colorCls}">${pct.toFixed(1)}%${fbMark}</span>`
        + `<span class="lst-dot ${dot}"></span>`
        + `<span class="lst-abs">${n}/${m}</span>`
        + `</div>`
      );
    }
  }

  const el = document.getElementById('lgbm-stats');
  if (el) el.innerHTML = html.join('');
}

// ── data loading ──────────────────────────────────────────────────────────────
async function loadData() {
  const [dr, sr, er] = await Promise.all([
    fetch('/api/data'), fetch('/api/sessions'), fetch('/api/eval')
  ]);
  allData  = await dr.json();
  sessions = await sr.json();
  evalData = await er.json();
  buildSessionFilter();
  updateStats();
  applyFilter();
  updateLgbmStats();
  // Compute static AUC per version (one-shot after data is ready)
  for (const ver of ['v6','v8','v11','k30tom']) {
    modelAuc[ver] = computeAucForVersion(ver);
  }
  await loadSavedThresholds();
  // Fetch honest CV AUC per model (one-shot)
  try {
    const r = await fetch('/api/model_meta');
    if (r.ok) modelMeta = await r.json();
  } catch (e) { /* not critical */ }
  // Tom k30tom: static reference from his external K=30 dataset evaluation.
  // No holdout/CV from us — we use Tom's reported AUC=0.787 as the "Test" chip target.
  if (!modelMeta.k30tom) {
    modelMeta.k30tom = {
      holdout_auc: 0.787,
      holdout_auc_std: null,
      holdout_n_test: 'Tom external',
      holdout_child: null, holdout_teen: null, holdout_fpr: null,
      n_features: 37,
    };
  }
  // Fetch V11 test holdout split (618 items the model never saw at training)
  try {
    const r = await fetch('/api/get_test_split');
    if (r.ok) {
      const t = await r.json();
      if (t.test_ids) holdoutIds = new Set(t.test_ids);
      holdoutMeta = t;
    }
  } catch (e) { /* optional */ }
  applyLgbmThresholds();
  refreshAucDisplays();
}

function buildSessionFilter() {
  const sel = document.getElementById('f-session');
  while (sel.options.length > 1) sel.remove(1);
  // sessions are already sorted newest-first from server
  sessions.forEach((s, i) => {
    // Count images in this session
    const cnt = allData.filter(r => r.source === 'grafana' && r.session === s).length;
    const label = `${s}  (${cnt})`;
    const opt = new Option(label, s);
    sel.appendChild(opt);
  });
}

// ── filters ───────────────────────────────────────────────────────────────────
function effectiveLabel(r) {
  return (changes[r.id] ? changes[r.id].label : r.label) || null;
}

function pipeStatus(r) {
  const res = r.piper_result;
  if (!res || res.error) return 'unprocessed';
  const lbl = res.siglip2_labels || [];
  if (lbl.includes('underage')) return 'underage';
  if (!res.siglip2_passed) return 'other';
  return 'passed';
}

// ── age helpers ───────────────────────────────────────────────────────────────
function getQ3AgeFrom(r) {
  if (r.source === 'labelstudio') return r.ageFrom ?? null;
  const q = r.qwen3_result;
  if (!q || !q.faces || !q.faces.length) return null;
  let min = Infinity;
  for (const f of q.faces) { const a = f.ageFrom ?? f.age_from; if (a != null && a < min) min = a; }
  return min === Infinity ? null : min;
}
function getFdAgeFrom(r) {
  // Returns ageFrom number | null (not scanned or no face)
  // "none" filter catches both — users want "no FD data" regardless of reason
  const fd = (r.piper_result && r.piper_result.face_detect_result) ?? r.face_detect_result ?? null;
  if (fd === null || fd === undefined) return null;
  return fd.ageFrom ?? fd.age_from ?? null;
}
function matchAge(ageFrom, f) {
  if (f === 'all')   return true;
  if (f === 'none')  return ageFrom == null;
  if (ageFrom == null) return false;
  // q3 ranges (lower bound)
  if (f === '0-14')  return ageFrom <= 14;
  if (f === '15-17') return ageFrom >= 15 && ageFrom <= 17;
  if (f === '0-17')  return ageFrom <= 17;
  if (f === '18+')   return ageFrom >= 18;
  // fd ranges
  if (f === '0-9')   return ageFrom <= 9;
  if (f === '10-19') return ageFrom >= 10 && ageFrom <= 19;
  if (f === '0-19')  return ageFrom <= 19;
  if (f === '20+')   return ageFrom >= 20;
  return true;
}

function applyFilter() {
  const sf  = document.getElementById('f-source').value;
  const ssf = document.getElementById('f-session').value;
  const lf  = document.getElementById('f-label').value;
  const aq3 = document.getElementById('f-age-q3').value;
  const afd = document.getElementById('f-age-fd').value;
  const ef  = evalActive ? document.getElementById('f-eval-outcome').value : 'all';

  // Update active eval version
  if (evalActive) evalVersion = document.getElementById('f-eval-ver').value;

  // Show/hide session & pipeline dropdowns
  const showSession = sf === 'grafana' || sf === 'all';
  document.getElementById('session-wrap').style.display = showSession ? '' : 'none';

  let d = allData;
  if (holdoutMode) d = d.filter(isInHoldout);

  if (sf === 'marked')      d = d.filter(r => marked.has(r.id));
  else if (sf === 'labelstudio') d = d.filter(r => r.source === 'labelstudio');
  else if (sf === 'grafana') d = d.filter(r => r.source === 'grafana');
  else if (sf === 'k30')     d = d.filter(r => r.source === 'k30');

  if (sf !== 'labelstudio' && ssf !== 'all')
    d = d.filter(r => r.session === ssf);

  if (lf === 'unlabeled')   d = d.filter(r => !effectiveLabel(r));
  else if (lf === 'unconfirmed') d = d.filter(r => r.source === 'grafana' && !r.label_confirmed && effectiveLabel(r));
  else if (lf !== 'all')   d = d.filter(r => effectiveLabel(r) === lf);


  // Eval outcome filter
  if (evalActive && ef !== 'all') {
    d = d.filter(r => {
      const entry = getEvalEntry(r.id);
      const outcome = evalOutcome(entry, r.variant);
      if (ef === 'NA') return outcome === 'NA';
      return outcome === ef;
    });
  }

  // Age filters
  if (aq3 !== 'all') d = d.filter(r => matchAge(getQ3AgeFrom(r), aq3));
  if (afd !== 'all') d = d.filter(r => matchAge(getFdAgeFrom(r), afd));

  // Active-learning: sort by descending model disagreement (max-min of V6/V8/V11)
  if (disagreeMode) {
    d = d.slice().sort((a, b) => disagreementScore(b) - disagreementScore(a));
  }

  filteredData = d;
  currentPage  = 1;
  updateStats();
  renderPage();
}

function updateStats() {
  const totals = { ls:0, gr:0, k30:0, child:0, teen:0, adult:0, unlabeled:0, confirmed:0, unconfirmed:0 };
  allData.forEach(r => {
    if (r.source === 'labelstudio') totals.ls++;
    else if (r.source === 'k30')        totals.k30++;
    else                                 totals.gr++;
    const lbl = effectiveLabel(r) || 'unlabeled';
    if (totals[lbl] !== undefined) totals[lbl]++;
    if (r.source === 'grafana') {
      // confirmed = human-labeled OR just saved in this session
      const isConfirmed = changes[r.id] ? true : r.label_confirmed;
      if (effectiveLabel(r) && !isConfirmed) totals.unconfirmed++;
      if (isConfirmed) totals.confirmed++;
    }
  });
  const filtered = filteredData.length;
  const total    = allData.length;
  let html =
    `<span style="color:#ccc"><b>Всего:</b> ${total}</span>` +
    `<span class="s-ls"><b>Label Studio:</b> ${totals.ls}</span>` +
    `<span class="s-gr"><b>Grafana:</b> ${totals.gr}</span>` +
    `<span class="s-k30"><b>K30 (Tom):</b> ${totals.k30}</span>` +
    `<span class="s-ch"><b>child:</b> ${totals.child}</span>` +
    `<span class="s-tn"><b>teen:</b> ${totals.teen}</span>` +
    `<span class="s-ad"><b>adult:</b> ${totals.adult}</span>` +
    `<span class="s-un"><b>без разм.:</b> ${totals.unlabeled}</span>` +
    (totals.unconfirmed ? `<span style="color:#cc8030"><b>⚡ AI (не подтв.):</b> ${totals.unconfirmed}</span>` : '') +
    (totals.confirmed   ? `<span style="color:#50a050"><b>✓ подтверждено:</b> ${totals.confirmed}</span>` : '') +
    (marked.size        ? `<span class="s-mk"><b>⭐ отмечено:</b> ${marked.size}</span>` : '') +
    (filtered < total   ? `<span style="color:#f0c060;margin-left:10px"><b>↳ фильтр:</b> ${filtered}</span>` : '');

  // Eval stats on filtered data when eval is active
  if (evalActive && Object.keys(evalData).length) {
    const ec = { TP:0, TN:0, FP:0, FN:0, NA:0 };
    filteredData.forEach(r => {
      const entry = getEvalEntry(r.id);
      ec[evalOutcome(entry, r.variant)]++;
    });
    const ver = evalVersion.toUpperCase();
    html += `<span style="color:#aaa;margin-left:10px"><b>${ver} (${filteredData.length}):</b></span>` +
      `<span style="color:#ff8080"><b>TP:</b> ${ec.TP}</span>` +
      `<span style="color:#80da90"><b>TN:</b> ${ec.TN}</span>` +
      `<span style="color:#dd80dd"><b>FP:</b> ${ec.FP}</span>` +
      `<span style="color:#8090ff"><b>FN:</b> ${ec.FN}</span>` +
      (ec.NA ? `<span style="color:#444"><b>NA:</b> ${ec.NA}</span>` : '');
  }
  document.getElementById('stats-bar').innerHTML = html;
}

// ── card rendering ────────────────────────────────────────────────────────────
function renderCard(r) {
  const curLabel = effectiveLabel(r);
  const isLS = r.source === 'labelstudio';

  const ps = pipeStatus(r);

  const card = document.createElement('div');
  let cls = 'card src-' + (isLS ? 'ls' : 'grafana');
  if (changes[r.id]) cls += ' modified';
  if (curLabel) cls += ' lbl-' + curLabel;
  card.className = cls;
  card.id = 'card-' + r.id;

  // Image
  const wrap = document.createElement('div'); wrap.className = 'img-wrap';
  const img  = document.createElement('img');
  img.loading = 'lazy';
  img.src = r._serve_url || r.thumb_url || '';
  img.onerror = () => {
    // Fallback to S3 thumb_url if local /img/ fails
    if (r.thumb_url && img.src !== r.thumb_url) img.src = r.thumb_url;
    else wrap.innerHTML = '<span style="color:#333;font-size:10px">no image</span>';
  };
  img.onclick = () => openLb(r);
  wrap.appendChild(img);

  // Hide button
  const hideBtn = document.createElement('div');
  hideBtn.className = 'hide-btn';
  hideBtn.title = 'Удалить из базы при сохранении';
  hideBtn.textContent = '×';
  hideBtn.onclick = e => { e.stopPropagation(); toggleHide(r.id, r.source); };
  wrap.appendChild(hideBtn);

  // Mark-as-favourite (star) — adds to "marked" set for later export
  const markBtn = document.createElement('div');
  markBtn.className = 'mark-btn';
  markBtn.textContent = marked.has(r.id) ? '★' : '☆';
  markBtn.title = 'Отметить (m) — добавить в выборку для экспорта';
  markBtn.onclick = e => { e.stopPropagation(); toggleMark(r.id); };
  wrap.appendChild(markBtn);
  if (marked.has(r.id)) card.classList.add('marked');

  // Source badge
  const sbadge = document.createElement('div');
  const isK30 = r.source === 'k30';
  sbadge.className = 'src-badge ' + (isLS ? 'ls' : isK30 ? 'k30' : 'gr');
  sbadge.textContent = isLS ? 'LS' : isK30 ? 'K30' : 'GR';
  wrap.appendChild(sbadge);

  // Viewed-in-lightbox badge (👁) — shown for items the user has opened in the lb during this session.
  if (viewedIds.has(r.id)) {
    const vb = document.createElement('div');
    vb.className = 'viewed-badge'; vb.textContent = '👁';
    vb.title = 'Просмотрено в лайтбоксе';
    wrap.appendChild(vb);
  }

  // Active-learning hot marker — model disagreement >= 0.30
  const disag = disagreementScore(r);
  if (disag >= 0.30) {
    const hot = document.createElement('div');
    hot.className = 'disagree-badge';
    hot.textContent = '🔥' + disag.toFixed(2);
    const e = evalData[r.id] || {};
    const tomV  = e['k30tom']?.lgbm;
    const oursV = e[evalVersion]?.lgbm;
    if (tomV != null && oursV != null) {
      hot.title = `|${evalVersion.toUpperCase()}=${oursV.toFixed(3)} − Tom_k30=${tomV.toFixed(3)}| = ${disag.toFixed(3)} — manual review`;
    } else {
      hot.title = `Disagreement (max−min of V6/V8/V11) = ${disag.toFixed(3)} — manual review`;
    }
    wrap.appendChild(hot);
  }

  // Pipeline/eval badge — see buildPipeBadge() for the rules.
  buildPipeBadge(wrap, r);

  // LGBM eval badge on image is REDUNDANT — details row below shows same V6/V8/V11.
  // Keeping function for back-compat but not calling it here.
  // buildLgbmBadge(wrap, r);

  // Confirmation badge bottom-right (grafana only)
  if (!isLS && curLabel) {
    const isConfirmed = changes[r.id] ? true : r.label_confirmed;
    const cb = document.createElement('div');
    if (isConfirmed) {
      cb.className = 'confirm-badge human';
      cb.textContent = '✓';
      cb.title = 'Подтверждено';
    } else {
      cb.className = 'confirm-badge auto';
      cb.textContent = '⚡ AI';
      cb.title = 'Авто-разметка (Qwen3) — не подтверждено';
    }
    wrap.appendChild(cb);
  }

  // Info row
  const info = document.createElement('div'); info.className = 'info';
  const iid  = document.createElement('div'); iid.className = 'iid';
  if (isLS && r._ls_id) {
    iid.innerHTML = `<a href="https://ls.artworks.ai/projects/3/data?tab=65&task=${r._ls_id}" target="_blank" rel="noopener">#${r._ls_id}</a>`;
  } else {
    iid.title = r.id;
    iid.textContent = r.id.substring(0, 8) + '…';
  }
  const imeta = document.createElement('div'); imeta.className = 'imeta';
  if (isLS) {
    const curAF = r.ageFrom;
    const curAT = r.ageTo;
    imeta.id = 'age-' + r.id;
    const fd = faceAge(r);
    const fdHtml = fd === false
      ? ' <span title="Face Detect: лицо не обнаружено" style="color:#7a9eff;font-size:9px">fd:👤✕</span>'
      : fd
        ? ` <span title="face detect" style="color:#7a9eff;font-size:9px">fd:${fd}</span>`
        : ' <span title="Face Detect: не сканировалось" style="color:#cc3333;font-size:9px;font-weight:bold">fd:—</span>';
    imeta.innerHTML = `<span title="qwen3">${curAF}–${curAT}</span>` + fdHtml;
  } else {
    const sess = r.session || '';
    const fd  = faceAge(r);
    const q3  = qwen3Age(r);
    const fdHtml = fd === false
      ? ' <span title="Face Detect: лицо не обнаружено" style="color:#7a9eff;font-size:9px">fd:👤✕</span>'
      : fd
        ? ` <span title="face detect" style="color:#7a9eff;font-size:9px">fd:${fd}</span>`
        : ' <span title="Face Detect: не сканировалось" style="color:#cc3333;font-size:9px;font-weight:bold">fd:—</span>';
    imeta.innerHTML =
        (q3 ? `<span title="qwen3" style="color:#c0e080;font-size:9px">q3:${q3}</span>` : '<span style="color:#333;font-size:9px">—</span>')
      + fdHtml;
    imeta.title = sess;
  }
  info.append(iid, imeta);

  // Prompt row (grafana) or empty (LS)
  const promptRow = document.createElement('div');
  const hasPrompt = !isLS && r.prompt;
  promptRow.className = 'prompt-row' + (hasPrompt ? ' has-prompt' : '');
  promptRow.title = hasPrompt ? 'Нажмите чтобы открыть полный промпт' : '';
  promptRow.textContent = hasPrompt ? r.prompt.substring(0, 70) + (r.prompt.length > 70 ? '…' : '') : (isLS ? '(Label Studio)' : '—');
  if (hasPrompt) promptRow.addEventListener('click', () => openPm(r.prompt));

  // Label radios
  const radios = document.createElement('div'); radios.className = 'radios';
  [['child','Дети'],['teen','Подр.'],['adult','Взрослые']].forEach(([lbl, name]) => {
    const label = document.createElement('label'); label.className = 'cat-' + lbl;
    const inp = document.createElement('input');
    inp.type = 'radio'; inp.name = 'lbl-' + r.id; inp.value = lbl; inp.checked = (curLabel === lbl);
    inp.addEventListener('change', () => onLabelChange(r, lbl));
    label.append(inp, Object.assign(document.createElement('span'), {textContent: name}));
    radios.appendChild(label);
  });

  card.append(wrap, info, promptRow, radios);

  // Eval overlay row
  card.appendChild(buildEvalRow(r));

  return card;
}

function faceAge(r) {
  // Extract face detect age from piper_result.face_detect_result
  // Returns: age string | false (scanned, no face) | null (not scanned)
  const fd = (r.piper_result && r.piper_result.face_detect_result)
           ?? r.face_detect_result
           ?? null;
  if (fd === null || fd === undefined) return null;          // not scanned
  const af = fd.ageFrom ?? fd.age_from;
  const at = fd.ageTo   ?? fd.age_to;
  if (af == null) return false;                              // scanned, no face found
  return `${af}–${at != null ? at : af}`;
}

function qwen3Age(r) {
  // Extract age from qwen3_result.faces[0] (grafana cards)
  const q = r.qwen3_result;
  if (!q) return null;
  const faces = q.faces;
  if (!faces || !faces.length) return null;
  // Use the youngest face (min ageFrom across all faces)
  let minAF = Infinity, minAT = Infinity;
  for (const f of faces) {
    const af = f.ageFrom ?? f.age_from;
    const at = f.ageTo   ?? f.age_to;
    if (af != null && af < minAF) { minAF = af; minAT = at ?? af; }
  }
  if (minAF === Infinity) return null;
  return `${minAF}–${minAT}`;
}


function onLabelChange(r, newLabel) {
  changes[r.id] = { label: newLabel, source: r.source };
  markModified(r.id, newLabel);

  // Age data is not updated on label change — it remains as-is (informational only)
  updateDirty();
  updateStats();
}

function markModified(id, label) {
  const el = document.getElementById('card-' + id);
  if (!el) return;
  el.className = el.className.replace(/\b(lbl-\w+|modified)\b/g, '').trim();
  el.classList.add('modified');
  if (label) el.classList.add('lbl-' + label);
}

function toggleHide(id, source) {
  const cardEl = document.getElementById('card-' + id);
  if (hidden.has(id)) {
    hidden.delete(id);
    if (cardEl) cardEl.classList.remove('hidden-pending');
  } else {
    hidden.add(id);
    if (cardEl) cardEl.classList.add('hidden-pending');
  }
  updateDirty();
}

function updateDirty() {
  const n = Object.keys(changes).length + hidden.size + toConfirm.size;
  const btn = document.getElementById('save-btn');
  const cbtn = document.getElementById('confirm-btn');
  if (n > 0) {
    btn.className = 'dirty';
    const parts = [];
    const nc = Object.keys(changes).length;
    if (nc)            parts.push('изм.: ' + nc);
    if (hidden.size)   parts.push('удал.: ' + hidden.size);
    if (toConfirm.size) parts.push('подтв.: ' + toConfirm.size);
    btn.textContent = '💾 Сохранить (' + parts.join('  ') + ')';
    document.getElementById('status').textContent = parts.join('  ');
  } else {
    btn.className = ''; btn.textContent = '💾 Сохранить';
    document.getElementById('status').textContent = '';
  }
  // Highlight confirm button if there are unconfirmed items on this page
  const pageUnconfirmed = filteredData
    .slice((currentPage-1)*parseInt(document.getElementById('f-pgsize').value),
            currentPage*parseInt(document.getElementById('f-pgsize').value))
    .filter(r => (r.source === 'grafana' || r.source === 'k30') && !r.label_confirmed && !toConfirm.has(r.id) && effectiveLabel(r)).length;
  cbtn.className = pageUnconfirmed > 0 ? 'has-pending' : '';
  cbtn.textContent = pageUnconfirmed > 0
    ? `✓ Подтвердить страницу (${pageUnconfirmed})`
    : '✓ Подтвердить страницу';
}

async function confirmPage() {
  const pgSize = parseInt(document.getElementById('f-pgsize').value);
  const start  = (currentPage-1)*pgSize;
  const page   = filteredData.slice(start, start+pgSize);
  let count = 0;
  page.forEach(r => {
    // Accept both grafana and k30 — both have AI drafts needing confirmation
    if ((r.source === 'grafana' || r.source === 'k30') && !r.label_confirmed && !toConfirm.has(r.id) && effectiveLabel(r)) {
      toConfirm.add(r.id);
      const cardEl = document.getElementById('card-' + r.id);
      if (cardEl) {
        const cb = cardEl.querySelector('.confirm-badge');
        if (cb) { cb.className = 'confirm-badge human'; cb.textContent = '✓'; cb.title = 'Подтверждено'; }
      }
      count++;
    }
  });
  if (count > 0) {
    updateDirty();
    // One-click flow: confirmation IS save. Eliminates the "press Сохранить again"
    // step the user found confusing.
    await saveChanges();
  }
}

async function saveChanges() {
  // Auto-confirm pass: items the user has touched or viewed in this session, which
  // have an effective label and are grafana/k30 (LS does not need confirmation),
  // get implicitly added to toConfirm. This eliminates the double-save flow where
  // the user had to click "Подтвердить страницу" after "Сохранить".
  const _autoConfirm = (id) => {
    if (!id || toConfirm.has(id)) return;
    if (hidden.has(id)) return;                       // marked for deletion — don't confirm
    const r = allData.find(x => x.id === id);
    if (!r) return;
    if (r.source !== 'grafana' && r.source !== 'k30') return;
    if (!effectiveLabel(r)) return;
    if (r.label_confirmed) return;                    // already confirmed on server
    toConfirm.add(id);
  };
  for (const id of Object.keys(changes)) _autoConfirm(id);
  for (const id of viewedIds)            _autoConfirm(id);

  if (!Object.keys(changes).length && !hidden.size && !toConfirm.size) return;
  document.getElementById('save-btn').textContent = '⏳ Сохранение…';
  try {
    const r = await fetch('/api/save', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({changes, delete: [...hidden], confirm: [...toConfirm]}),
    });
    const res = await r.json();
    // Reflect server-side label_confirmed=1 locally so the "Подтвердить страницу"
    // counter and confirm-badges flip immediately without a reload.
    for (const cid of toConfirm) {
      const r2 = allData.find(x => x.id === cid);
      if (r2) r2.label_confirmed = 1;
      const cardEl = document.getElementById('card-' + cid);
      if (cardEl) {
        const cb = cardEl.querySelector('.confirm-badge');
        if (cb) { cb.className = 'confirm-badge human'; cb.textContent = '✓'; cb.title = 'Подтверждено'; }
      }
    }
    changes = {}; hidden.clear(); toConfirm.clear();
    // Keep viewedIds across saves — глазик остаётся как индикатор сеансовой истории.
    document.getElementById('save-btn').className = '';
    document.getElementById('save-btn').textContent =
      `✓ Сохранено (${res.saved} изм. / ${res.deleted} удал.)`;
    document.getElementById('status').textContent = '';
    // Preserve current page through the reload — loadData() → applyFilter() resets
    // currentPage to 1, which is jarring if the user was deep in pagination.
    const _savedPage = currentPage;
    await loadData();
    // applyFilter recalculated total pages — clamp restored page if needed
    const _pgSize = parseInt(document.getElementById('f-pgsize').value) || 100;
    const _maxPage = Math.max(1, Math.ceil(filteredData.length / _pgSize));
    currentPage = Math.min(_savedPage, _maxPage);
    renderPage();
    updateDirty();
  } catch(e) {
    document.getElementById('save-btn').textContent = '❌ Ошибка';
    console.error(e);
  }
}

// ── lightbox ──────────────────────────────────────────────────────────────────
function openPm(text) {
  document.getElementById('pm-text').textContent = text || '';
  document.getElementById('pm').classList.add('open');
}
function closePm() { document.getElementById('pm').classList.remove('open'); }

// Lightbox state for moderation mode
let lbCurrent = null;
let lbPageItems = [];
let lbCurrentIdx = -1;

function openLb(r) {
  const pgSize = parseInt(document.getElementById('f-pgsize').value);
  const start  = (currentPage-1)*pgSize;
  lbPageItems  = filteredData.slice(start, start+pgSize);
  lbCurrentIdx = lbPageItems.findIndex(x => x.id === r.id);
  lbCurrent    = r;
  _markViewed(r);
  renderLbThumbs();   // build the strip once for this page
  renderLbView();
  document.getElementById('lb').classList.add('open');
  // Position track without animation on first paint
  requestAnimationFrame(() => _lbScrollToCurrent(false));
}

// Mark a record as viewed in this session — stamp it in viewedIds and add a badge
// on the corresponding card so the user can see at a glance what they have already
// looked at. Idempotent.
function _markViewed(r) {
  if (!r || viewedIds.has(r.id)) return;
  viewedIds.add(r.id);
  const cardEl = document.getElementById('card-' + r.id);
  if (!cardEl) return;
  const wrap = cardEl.querySelector('.img-wrap');
  if (!wrap || wrap.querySelector('.viewed-badge')) return;
  const vb = document.createElement('div');
  vb.className = 'viewed-badge'; vb.textContent = '👁';
  vb.title = 'Просмотрено в лайтбоксе';
  wrap.appendChild(vb);
}

const LB_THUMB_W = 56;  // keep in sync with .lb-thumb width
const LB_THUMB_GAP = 4;

function renderLbThumbs() {
  const track = document.getElementById('lb-thumbs-track');
  if (!track) return;
  track.innerHTML = '';
  // Disable transition while bulk-rebuilding
  track.style.transition = 'none';
  lbPageItems.forEach((r, idx) => {
    const t = document.createElement('div');
    let tCls = 'lb-thumb';
    if (idx === lbCurrentIdx) tCls += ' current';
    if (hidden.has(r.id)) tCls += ' hidden-pending';
    if (marked.has(r.id)) tCls += ' marked';
    t.className = tCls;
    t.dataset.idx = String(idx);
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = r.thumb_url || r._serve_url || '';
    t.appendChild(img);
    // Status dot: deletion mark wins, then effective label colour
    const lbl = effectiveLabel(r);
    if (hidden.has(r.id)) {
      const d = document.createElement('div'); d.className = 'lb-thumb-dot del';
      t.appendChild(d);
    } else if (lbl === 'child' || lbl === 'teen' || lbl === 'adult') {
      const d = document.createElement('div'); d.className = 'lb-thumb-dot ' + lbl;
      t.appendChild(d);
    }
    t.onclick = () => {
      lbCurrentIdx = idx;
      lbCurrent = lbPageItems[idx];
      _markViewed(lbCurrent);
      renderLbView();
    };
    track.appendChild(t);
  });
  // Re-enable transition on next frame
  requestAnimationFrame(() => { track.style.transition = ''; });
}

function highlightCurrentThumb() {
  const track = document.getElementById('lb-thumbs-track');
  if (!track) return;
  const kids = track.children;
  for (let i = 0; i < kids.length; i++) {
    kids[i].classList.toggle('current', i === lbCurrentIdx);
    const r = lbPageItems[i];
    if (!r) continue;
    kids[i].classList.toggle('hidden-pending', hidden.has(r.id));
    kids[i].classList.toggle('marked', marked.has(r.id));
    const existing = kids[i].querySelector('.lb-thumb-dot');
    if (existing) existing.remove();
    const lbl = effectiveLabel(r);
    if (hidden.has(r.id)) {
      const d = document.createElement('div'); d.className = 'lb-thumb-dot del';
      kids[i].appendChild(d);
    } else if (lbl === 'child' || lbl === 'teen' || lbl === 'adult') {
      const d = document.createElement('div'); d.className = 'lb-thumb-dot ' + lbl;
      kids[i].appendChild(d);
    }
  }
  _lbScrollToCurrent(true);
}

function _lbScrollToCurrent(animated) {
  const track = document.getElementById('lb-thumbs-track');
  if (!track) return;
  // After scale 1.08 the current thumb is slightly wider; we center the un-scaled slot.
  const slotW = LB_THUMB_W + LB_THUMB_GAP;
  // Track origin is at left:50%, items laid out from there. Offset so that
  // current-thumb-center sits at 0 (i.e. at left:50% of the container).
  // First item left edge is at 0 within track; center of item i is at
  //   padding(4) + i*(slotW) + LB_THUMB_W/2.
  // Track is translated by -that.
  const offset = -(4 + lbCurrentIdx * slotW + LB_THUMB_W / 2);
  track.style.transition = animated ? '' : 'none';
  track.style.transform = `translateX(${offset}px)`;
  if (!animated) {
    // Force reflow then restore transition
    void track.offsetWidth;
    track.style.transition = '';
  }
}

function _lbQwen3Age(r) {
  // K30 stores qwen3 age as r.qwen3_result.{min_age,max_age}; grafana uses faces[].
  const q = r.qwen3_result;
  if (!q) return null;
  if (q.min_age != null) {
    const lo = q.min_age;
    const hi = (q.max_age != null) ? q.max_age : lo;
    return `${lo}–${hi}`;
  }
  try { return qwen3Age(r); } catch (e) { return null; }
}

function _lbLsAge(r) {
  const af = r.ageFrom ?? (r.age && r.age.ageFrom);
  const at = r.ageTo   ?? (r.age && r.age.ageTo);
  if (af != null) return `${af}–${at != null ? at : af}`;
  return null;
}

function _lbDisagreementRow(r) {
  let d = 0;
  try { d = disagreementScore(r); } catch (e) { return ''; }
  if (d == null) return '';
  const cls = d >= 0.30 ? 'disagree-hi' : d >= 0.15 ? 'disagree-mid' : 'disagree-lo';
  const hot = d >= 0.30 ? ' 🔥' : '';
  return `<div class="lb-row lb-section"><span class="lb-k">🔥 disagree</span>` +
         `<span class="lb-v ${cls}">${d.toFixed(3)}${hot}</span></div>`;
}

function _lbVerdict(r) {
  // Effective human label wins; otherwise majority model vote at current threshold.
  const cur = effectiveLabel(r);
  if (cur === 'child' || cur === 'teen') return { cls: 'underage', text: '⛔ UNDERAGE' };
  if (cur === 'adult')                   return { cls: 'ok',       text: '✓ OK (18+)' };
  const e = evalData[r.id] || {};
  let votes = 0, total = 0;
  for (const v of ['v6','v8','v11','k30tom']) {
    const entry = e[v];
    if (!entry || entry.lgbm == null) continue;
    total++;
    if (entry.lgbm >= lgbmThrFor(v)) votes++;
  }
  if (total === 0)                  return { cls: 'unknown',  text: '? UNKNOWN' };
  if (votes >= Math.ceil(total/2))  return { cls: 'underage', text: `⛔ UNDERAGE (${votes}/${total})` };
  return { cls: 'ok', text: `✓ OK (${votes}/${total})` };
}

function renderLbView() {
  const r = lbCurrent;
  if (!r) return;
  document.getElementById('lb-img').src = r._serve_url || r.thumb_url || '';
  document.getElementById('lb-prompt').textContent = r.prompt || '';
  document.getElementById('lb-counter').textContent =
    `${lbCurrentIdx+1} / ${lbPageItems.length}`;

  const e   = evalData[r.id] || {};
  const cur = effectiveLabel(r) || '—';
  const lblCls = (cur === 'child' || cur === 'teen' || cur === 'adult') ? `lb-lbl-${cur}` : '';

  // Per-model formatting with color grading by its own threshold
  const fmtScore = (entry, ver) => {
    if (!entry || entry.lgbm == null) return '<span class="lb-v lo">—</span>';
    const star = entry.fallback_taxonomy ? '*' : '';
    const thr  = lgbmThrFor(ver);
    const cls  = entry.lgbm >= thr ? 'score-blocked' : 'score-passed';
    return `<span class="lb-v ${cls}" title="thr=${thr.toFixed(2)}">${entry.lgbm.toFixed(3)}${star}</span>`;
  };

  const badges = [];
  if (changes[r.id] || r.label_confirmed) badges.push('<span class="lb-badge-conf">✓</span>');
  if (hidden.has(r.id)) badges.push('<span class="lb-badge-del">🗑</span>');
  const badgeStr = badges.length ? ' ' + badges.join(' ') : '';

  const q3   = _lbQwen3Age(r);
  const fd   = (typeof faceAge === 'function') ? faceAge(r) : null;
  const fdText = (fd === false) ? '<span class="lb-v lo">no face</span>'
              : (fd == null)    ? '<span class="lb-v lo">—</span>'
              :                   `<span class="lb-v">${fd}</span>`;
  const q3Text = (q3 == null) ? '<span class="lb-v lo">—</span>'
                              : `<span class="lb-v">${q3}</span>`;
  const lsAge  = _lbLsAge(r);
  const lsAgeRow = (lsAge && r.source === 'labelstudio')
    ? `<div class="lb-row"><span class="lb-k">age (LS)</span><span class="lb-v">${lsAge}</span></div>`
    : '';

  const v = _lbVerdict(r);

  document.getElementById('lb-info').innerHTML =
    `<div class="lb-verdict ${v.cls}">${v.text}</div>` +
    `<div class="lb-row"><span class="lb-k">label</span><span class="lb-v ${lblCls}">${cur}${badgeStr}</span></div>` +
    `<div class="lb-row"><span class="lb-k">source</span><span class="lb-v">${r.source || '—'}</span></div>` +
    lsAgeRow +
    `<div class="lb-row lb-section"><span class="lb-k">age q3</span>${q3Text}</div>` +
    `<div class="lb-row"><span class="lb-k">age fd</span>${fdText}</div>` +
    `<div class="lb-row lb-section"><span class="lb-k">V6</span>${fmtScore(e.v6, 'v6')}</div>` +
    `<div class="lb-row"><span class="lb-k">V8</span>${fmtScore(e.v8, 'v8')}</div>` +
    `<div class="lb-row"><span class="lb-k">V11</span>${fmtScore(e.v11, 'v11')}</div>` +
    `<div class="lb-row"><span class="lb-k">Tom K30</span>${fmtScore(e.k30tom, 'k30tom')}</div>` +
    _lbDisagreementRow(r) +
    `<div class="lb-row lb-section"><span class="lb-k">⭐ marked</span>` +
      `<span class="lb-v" style="cursor:pointer;color:${marked.has(r.id) ? '#ffd040' : '#666'}"` +
        ` onclick="toggleMark('${r.id}')">${marked.has(r.id) ? '★ ДА (5/m)' : '☆ нет (5/m)'}</span></div>`;

  // Keep thumb strip in sync with current item / labels / deletion mark
  highlightCurrentThumb();
}

function closeLb() {
  document.getElementById('lb').classList.remove('open');
  lbCurrent = null;
  lbCurrentIdx = -1;
}

function nextInLb() {
  if (!lbCurrent) return;
  if (lbCurrentIdx + 1 >= lbPageItems.length) { _lbFlash('lb-flash-del'); return; }
  lbCurrentIdx++;
  lbCurrent = lbPageItems[lbCurrentIdx];
  _markViewed(lbCurrent);
  renderLbView();
}

function prevInLb() {
  if (!lbCurrent) return;
  if (lbCurrentIdx - 1 < 0) { _lbFlash('lb-flash-del'); return; }
  lbCurrentIdx--;
  lbCurrent = lbPageItems[lbCurrentIdx];
  _markViewed(lbCurrent);
  renderLbView();
}

// Mouse wheel paging in lightbox: scroll down → next, up → prev (throttled)
let _lbWheelLockUntil = 0;
function _lbOnWheel(e) {
  if (!document.getElementById('lb').classList.contains('open')) return;
  e.preventDefault();
  const now = Date.now();
  if (now < _lbWheelLockUntil) return;
  _lbWheelLockUntil = now + 180;  // ~180ms throttle
  if (e.deltaY > 0) nextInLb();
  else if (e.deltaY < 0) prevInLb();
}
document.addEventListener('wheel', _lbOnWheel, { passive: false });

function lbSetLabel(cat) {
  if (!lbCurrent) return;
  const r = lbCurrent;
  const inp = document.querySelector(`input[name="lbl-${r.id}"][value="${cat}"]`);
  if (inp) { inp.checked = true; inp.dispatchEvent(new Event('change')); }
  else { onLabelChange(r, cat); }
  renderLbView();
  _lbFlash('lb-flash');
  // Auto-advance to next item after labelling (moderation flow)
  setTimeout(nextInLb, 120);
}

function lbToggleDelete() {
  if (!lbCurrent) return;
  toggleHide(lbCurrent.id, lbCurrent.source);
  renderLbView();
  _lbFlash('lb-flash-del');
  // Auto-advance after marking for deletion
  setTimeout(nextInLb, 120);
}

function _lbFlash(cls) {
  const el = document.getElementById('lb-info');
  if (!el) return;
  el.classList.remove('lb-flash', 'lb-flash-del');
  void el.offsetWidth;
  el.classList.add(cls);
}

document.addEventListener('keydown', e => {
  const lbOpen = document.getElementById('lb').classList.contains('open');
  if (lbOpen) {
    if (e.key === 'Escape') { closeLb(); e.preventDefault(); return; }
    if (e.key === 'Enter')  { nextInLb(); e.preventDefault(); return; }
    if (e.key === '1')      { lbSetLabel('child'); e.preventDefault(); return; }
    if (e.key === '2')      { lbSetLabel('teen');  e.preventDefault(); return; }
    if (e.key === '3')      { lbSetLabel('adult'); e.preventDefault(); return; }
    if (e.key === '4')      { lbToggleDelete();    e.preventDefault(); return; }
    if (e.key === 'm' || e.key === 'M' || e.key === '5') {
      if (lbCurrent) { toggleMark(lbCurrent.id); renderLbView(); }
      e.preventDefault(); return;
    }
  } else {
    if (e.key === 'Escape') { closePm(); }
  }
});

// Hotkeys: hover → 1=child, 2=teen, 3=adult
let hoveredId = null;
document.addEventListener('mouseover', e => {
  const c = e.target.closest('.card'); hoveredId = c ? c.id.replace('card-','') : null;
});
document.addEventListener('keydown', e => {
  if (!hoveredId || document.getElementById('lb').classList.contains('open')) return;
  const map = {'1':'child','2':'teen','3':'adult'};
  if (map[e.key]) {
    const inp = document.querySelector(`input[name="lbl-${hoveredId}"][value="${map[e.key]}"]`);
    if (inp) { inp.checked = true; inp.dispatchEvent(new Event('change')); e.preventDefault(); }
  } else if (e.key === '5' || e.key === 'm' || e.key === 'M') {
    toggleMark(hoveredId);
    e.preventDefault();
  }
});

// ── pagination ────────────────────────────────────────────────────────────────
function renderPage() {
  const pgSize = parseInt(document.getElementById('f-pgsize').value);
  const total  = filteredData.length;
  const pages  = Math.max(1, Math.ceil(total / pgSize));
  if (currentPage > pages) currentPage = pages;
  const start  = (currentPage-1)*pgSize;

  const g = document.getElementById('gallery');
  g.innerHTML = '';
  filteredData.slice(start, start+pgSize).forEach(r => g.appendChild(renderCard(r)));
  window.scrollTo(0,0);
  document.getElementById('subtitle').textContent = `${filteredData.length} / ${allData.length}`;
  renderPagination(pages, pgSize, total);
}

function renderPagination(pages, pgSize, total) {
  const pg = document.getElementById('pagination');
  pg.innerHTML = '';
  if (pages <= 1) return;
  const start = (currentPage-1)*pgSize+1, end = Math.min(currentPage*pgSize, total);
  const btn = (lbl, page, dis, act) => {
    const b = document.createElement('button');
    b.textContent = lbl; if (act) b.className = 'active'; if (dis) b.disabled = true;
    else b.onclick = () => { currentPage = page; renderPage(); };
    return b;
  };
  pg.appendChild(btn('«',1,currentPage===1)); pg.appendChild(btn('‹',currentPage-1,currentPage===1));
  let lo=Math.max(1,currentPage-3), hi=Math.min(pages,lo+6); lo=Math.max(1,hi-6);
  if (lo>1){pg.appendChild(btn('1',1)); if(lo>2)pg.appendChild(Object.assign(document.createElement('span'),{textContent:'…',className:'pg-info'}));}
  for(let p=lo;p<=hi;p++) pg.appendChild(btn(p,p,false,p===currentPage));
  if(hi<pages){if(hi<pages-1)pg.appendChild(Object.assign(document.createElement('span'),{textContent:'…',className:'pg-info'}));pg.appendChild(btn(pages,pages));}
  pg.appendChild(btn('›',currentPage+1,currentPage===pages)); pg.appendChild(btn('»',pages,currentPage===pages));
  const info=document.createElement('span'); info.className='pg-info'; info.textContent=`${start}–${end} из ${total}`; pg.appendChild(info);
}

// ── eval overlay ──────────────────────────────────────────────────────────────

async function toggleEval(on) {
  evalActive = on;
  document.getElementById('f-eval-ver').style.display     = on ? '' : 'none';
  document.getElementById('f-eval-outcome').style.display = on ? '' : 'none';
  // evalData pre-loaded at startup; refresh if somehow empty
  if (on && Object.keys(evalData).length === 0) {
    document.getElementById('status').textContent = 'Загрузка eval…';
    evalData = await fetch('/api/eval').then(r => r.json());
    document.getElementById('status').textContent = '';
  }
  evalVersion = document.getElementById('f-eval-ver').value;
  refreshEvalRows();
  rebuildAllPipeBadges();
  applyFilter();
}

function refreshEvalRows() {
  document.querySelectorAll('.eval-row').forEach(el => {
    el.classList.toggle('visible', evalActive);
  });
}

function getEvalEntry(galleryId) {
  if (!evalData[galleryId]) return null;
  return evalData[galleryId][evalVersion] || null;
}

/**
 * Determine outcome: TP/TN/FP/FN based on variant + blocked
 * variant=positive (child/teen) → should be blocked
 *   blocked=true  → TP
 *   blocked=false → FN
 * variant=negative (adult) → should pass
 *   blocked=false → TN
 *   blocked=true  → FP
 */
function evalOutcome(entry, itemVariant) {
  if (!entry) return 'NA';
  const variant = entry.variant || itemVariant;
  if (!variant) return 'NA';
  const shouldBlock = variant === 'positive';
  if (shouldBlock && entry.blocked)  return 'TP';
  if (shouldBlock && !entry.blocked) return 'FN';
  if (!shouldBlock && !entry.blocked) return 'TN';
  if (!shouldBlock && entry.blocked)  return 'FP';
  return 'NA';
}

function buildEvalRow(r) {
  const row = document.createElement('div');
  row.className = 'eval-row' + (evalActive ? ' visible' : '');
  row.id = 'eval-' + r.id;
  renderEvalRowContent(row, r);
  return row;
}

function renderEvalRowContent(row, r) {
  row.innerHTML = '';
  const allVersionData = evalData[r.id] || {};
  const ev6   = allVersionData['v6']    || null;
  const ev8   = allVersionData['v8']    || null;
  const ev11  = allVersionData['v11']   || null;
  const eTom  = allVersionData['k30tom'] || null;   // Tom's K=30 model (a4aa9dbd9c)

  // ── Comparison columns: V6 | V8 | V11 | Tom_K30 ──────────────────────────
  const cmpLeft      = ev6 || ev8 || null;
  const cmpMid       = ev6 ? (ev8 || ev11) : (ev11 || null);
  const cmpRight     = (ev6 && ev8) ? ev11 : null;
  const cmpLeftName  = ev6 ? 'V6' : (ev8 ? 'V8pas80' : 'V11s80');
  const cmpMidName   = ev6 ? (ev8 ? 'V8pas80' : 'V11s80') : 'V11s80';
  const cmpRightName = 'V11s80';
  const cmpLeftVer   = ev6 ? 'v6' : (ev8 ? 'v8' : 'v11');
  const cmpMidVer    = ev6 ? (ev8 ? 'v8' : 'v11') : 'v11';
  const cmpRightVer  = 'v11';
  const showThree    = !!(ev6 && ev8 && ev11);
  const showFour     = showThree && !!eTom;

  if (cmpLeft && cmpMid) {
    const cmp = document.createElement('div');
    cmp.className = 'eval-cmp' + (showFour ? ' four-col' : (showThree ? ' three-col' : ''));

    // For LGBM: hi=true → orange (blocked), hi=false but data exists → green (passes)
    // For minor (hi-minor): keep original — only hi or grey.
    const mkVal = (val, hi, hiClass) => {
      const el = document.createElement('div');
      let cls;
      if (hi) {
        cls = hiClass;
      } else if (hiClass === 'hi-lgbm') {
        cls = 'ok-lgbm';     // model has data but score below threshold → green
      } else {
        cls = 'lo';          // generic "no signal"
      }
      el.className = 'cmp-val ' + cls;
      el.textContent = val.toFixed(3);
      return el;
    };

    const mkOut = (entry, version) => {
      if (!entry) {
        const el2 = document.createElement('div'); el2.className = 'cmp-outcome';
        const b2  = document.createElement('span'); b2.className  = 'outcome-badge outcome-NA'; b2.textContent = '—';
        el2.appendChild(b2); return el2;
      }
      const bl = lgbmIsBlocked(entry, version || 'v6');
      const variant = entry.variant || r.variant;
      const shouldBlock = variant === 'positive';
      let out;
      if (bl === null) out = 'NA';
      else if (shouldBlock && bl)   out = 'TP';
      else if (shouldBlock && !bl)  out = 'FN';
      else if (!shouldBlock && !bl) out = 'TN';
      else                          out = 'FP';
      const el2 = document.createElement('div'); el2.className = 'cmp-outcome';
      const badge = document.createElement('span');
      badge.className = 'outcome-badge outcome-' + out;
      badge.textContent = out;
      el2.appendChild(badge);
      return el2;
    };

    // Header row — class = cmp-hdr + version-class for color
    const hblank = document.createElement('div'); hblank.className = 'cmp-lbl';
    const hL = document.createElement('div'); hL.className = 'cmp-hdr ' + cmpLeftVer;  hL.textContent = cmpLeftName;
    const hM = document.createElement('div'); hM.className = 'cmp-hdr ' + cmpMidVer;   hM.textContent = cmpMidName;
    cmp.append(hblank, hL, hM);
    if (showThree) {
      const hR = document.createElement('div'); hR.className = 'cmp-hdr ' + cmpRightVer; hR.textContent = cmpRightName;
      cmp.append(hR);
    }
    if (showFour) {
      const hT = document.createElement('div'); hT.className = 'cmp-hdr k30tom'; hT.textContent = 'Tom_K30';
      cmp.append(hT);
    }

    // LGBM row — color depends ONLY on each model's own threshold (no diff coloring)
    const lgbmLbl = document.createElement('div'); lgbmLbl.className = 'cmp-lbl'; lgbmLbl.textContent = 'LGBM';
    const mkValOrDash = (entry, thr) => {
      if (!entry) return Object.assign(document.createElement('div'), {className: 'cmp-val lo', textContent: '—'});
      const el = mkVal(entry.lgbm, entry.lgbm >= thr, 'hi-lgbm');  // no compareTo → pure threshold color
      if (entry.fallback_taxonomy) {
        el.textContent = entry.lgbm.toFixed(3) + '*';
        el.style.opacity = '0.6';
        el.title = 'fallback: 180-tag taxonomy (V11 trained on 317-tag) — score approximate';
      }
      return el;
    };
    cmp.append(lgbmLbl,
      mkValOrDash(cmpLeft, lgbmThrFor(cmpLeftVer)),
      mkValOrDash(cmpMid,  lgbmThrFor(cmpMidVer)));
    if (showThree) cmp.append(mkValOrDash(cmpRight, lgbmThrFor(cmpRightVer)));
    if (showFour) {
      // Tom's K=30 score — threshold 0.1 (per Tom's pipeline default)
      cmp.append(mkValOrDash(eTom, 0.10));
    }

    // separator
    const sep = document.createElement('div'); sep.className = 'cmp-sep'; cmp.appendChild(sep);

    // outcome row
    const outLbl = document.createElement('div'); outLbl.className = 'cmp-lbl'; outLbl.textContent = '';
    cmp.append(outLbl, mkOut(cmpLeft, cmpLeftVer), mkOut(cmpMid, cmpMidVer));
    if (showThree) cmp.append(mkOut(cmpRight, cmpRightVer));
    if (showFour) {
      // Tom outcome uses fixed threshold 0.10
      const mkOutTom = () => {
        const el2 = document.createElement('div'); el2.className = 'cmp-outcome';
        if (!eTom) {
          const b2 = document.createElement('span'); b2.className = 'outcome-badge outcome-NA'; b2.textContent = '—';
          el2.appendChild(b2); return el2;
        }
        const bl = eTom.lgbm >= 0.10;
        const variant = eTom.variant || r.variant;
        const shouldBlock = variant === 'positive';
        let out;
        if (shouldBlock && bl)        out = 'TP';
        else if (shouldBlock && !bl)  out = 'FN';
        else if (!shouldBlock && !bl) out = 'TN';
        else                          out = 'FP';
        const badge = document.createElement('span');
        badge.className = 'outcome-badge outcome-' + out;
        badge.textContent = out;
        el2.appendChild(badge); return el2;
      };
      cmp.append(mkOutTom());
    }

    row.appendChild(cmp);
    return;  // (human_label footer removed — duplicate of label toggle above)
  }

  // ── Single-version fallback ───────────────────────────────────────────────
  const versions = ['v8','v11','v6'];
  const btnRow = document.createElement('div'); btnRow.className = 'eval-ver-btns';
  versions.forEach(v => {
    const btn = document.createElement('button');
    btn.textContent = v.toUpperCase();
    const hasData = !!allVersionData[v];
    if (!hasData) { btn.disabled = true; btn.title = 'Нет данных'; }
    else btn.className = 'has-data';
    if (v === evalVersion && hasData) btn.classList.add('active');
    btn.onclick = () => {
      evalVersion = v;
      document.getElementById('f-eval-ver').value = v;
      document.querySelectorAll('.eval-row').forEach(el => {
        const cid = el.id.replace('eval-', '');
        const item = allData.find(x => x.id === cid);
        if (item) renderEvalRowContent(el, item);
      });
      rebuildAllPipeBadges();
      applyFilter();
    };
    btnRow.appendChild(btn);
  });
  row.appendChild(btnRow);

  const entry = allVersionData[evalVersion] || null;
  if (!entry) {
    const na = document.createElement('div');
    na.style.cssText = 'font-size:10px;color:#333;padding:2px 0';
    na.textContent = 'нет данных для ' + evalVersion.toUpperCase();
    row.appendChild(na);
    return;
  }

  const scores = document.createElement('div'); scores.className = 'eval-scores';
  const mkScore = (label, val, hi) => {
    const si = document.createElement('div'); si.className = 'eval-score-item';
    const lbl = document.createElement('div'); lbl.className = 'label'; lbl.textContent = label;
    const v = document.createElement('div');
    v.className = 'value ' + (hi ? (label === 'LGBM' ? 'hi-lgbm' : 'hi-minor') : 'lo');
    v.textContent = val.toFixed(3);
    si.append(lbl, v); return si;
  };
  const _curThr = lgbmThrFor(evalVersion);
  scores.appendChild(mkScore('LGBM',  entry.lgbm,  entry.lgbm  >= _curThr));
  scores.appendChild(mkScore('minor', entry.minor, entry.minor >= 0.72));
  const outcome = evalOutcome(entry, r.variant);
  const outBadge = document.createElement('span');
  outBadge.className = 'outcome-badge outcome-' + outcome;
  outBadge.textContent = outcome;
  scores.appendChild(outBadge);
  row.appendChild(scores);

  const bottom = document.createElement('div'); bottom.className = 'eval-outcome';
  const whichEl = document.createElement('span'); whichEl.className = 'which-badge';
  if (entry.which) {
    const _tl = lgbmThrFor(evalVersion).toFixed(2);
    const labels = { lgbm: `lgbm≥${_tl}`, pipeline: 'pipeline' };
    whichEl.textContent = '⛔ ' + (labels[entry.which] || entry.which);
    whichEl.style.color = '#ff9060';
  } else {
    whichEl.textContent = '✓ прошёл';
    whichEl.style.color = '#50a060';
  }
  const hlbl = document.createElement('span');
  hlbl.style.cssText = 'font-size:9px;color:#444';
  hlbl.textContent = entry.human_label || '?';
  bottom.append(whichEl, hlbl);
  row.appendChild(bottom);
}

function getItemEvalOutcome(r) {
  if (!evalActive) return null;
  const e = getEvalEntry(r.id);
  return evalOutcome(e, r.variant);
}

// ── init ──────────────────────────────────────────────────────────────────────
document.getElementById('f-source').addEventListener('change', () => {
  applyFilter();
  for (const ver of ['v6','v8','v11']) { modelAuc[ver] = computeAucForVersion(ver); }
  updateLgbmStats();
  refreshAucDisplays();
});
document.getElementById('f-session').addEventListener('change', applyFilter);
document.getElementById('f-label').addEventListener('change', applyFilter);
document.getElementById('f-eval-ver').addEventListener('change', () => {
  evalVersion = document.getElementById('f-eval-ver').value;
  // Re-render all eval rows with new version
  document.querySelectorAll('.eval-row').forEach(el => {
    const cid = el.id.replace('eval-', '');
    const item = allData.find(x => x.id === cid);
    if (item) renderEvalRowContent(el, item);
  });
  rebuildAllPipeBadges();
  applyFilter();
});
document.getElementById('f-eval-outcome').addEventListener('change', applyFilter);
document.getElementById('f-age-q3').addEventListener('change', applyFilter);
document.getElementById('f-age-fd').addEventListener('change', applyFilter);
document.getElementById('f-pgsize').addEventListener('change', () => {
  currentPage = 1; renderPage();
});
document.getElementById('thr-v8').addEventListener('input',  () => onThrSlide('v8'));
document.getElementById('thr-v11').addEventListener('input', () => onThrSlide('v11'));
document.getElementById('thr-v6').addEventListener('input',  () => onThrSlide('v6'));
document.getElementById('thr-k30tom').addEventListener('input', () => onThrSlide('k30tom'));

// initial load
loadData();
</script>
</body></html>
"""


# ─── HTTP handler ────────────────────────────────────────────────────────────
META_PATHS = {
    "v6":  BASE_DIR / "data" / "lgbm_v6_meta.json",
    "v8":  BASE_DIR / "data" / "lgbm_v8pas80_v2_meta.json",
    "v11": BASE_DIR / "data" / "lgbm_v11s80_meta.json",
}


HOLDOUT_META_PATHS = {
    "v6":  BASE_DIR / "data" / "lgbm_v6b_meta.json",
    "v8":  BASE_DIR / "data" / "lgbm_v8bs80_meta.json",
    "v11": BASE_DIR / "data" / "lgbm_v11bs80_meta.json",
}


def load_model_meta() -> dict:
    """Return per-version honest evaluation metrics.

    Prefers HOLDOUT-based test_auc (from train_*b_holdout.py, multi-seed) over
    legacy CV AUC. Both shown when available. Includes per-source breakdown.
    """
    out = {}
    for ver, p in META_PATHS.items():
        d = json.loads(p.read_text()) if p.exists() else {}
        cv_auc = d.get("cv_auc") or d.get("cv_auc_slim") or d.get("cv_auc_full")
        cv_std = d.get("cv_std") or d.get("cv_std_slim") or d.get("cv_std_full")
        entry = {
            "cv_auc":     cv_auc,
            "cv_std":     cv_std,
            "n_samples":  d.get("n_samples"),
            "n_features": d.get("n_features"),
        }
        # Layer in honest holdout metrics (from train_*b_holdout meta) if present
        hp = HOLDOUT_META_PATHS.get(ver)
        if hp and hp.exists():
            try:
                hd = json.loads(hp.read_text())
                ms = hd.get("multi_seed_slim") or hd.get("multi_seed")
                if ms:
                    entry["holdout_auc"]     = ms["auc"]["mean"]
                    entry["holdout_auc_std"] = ms["auc"]["std"]
                    entry["holdout_child"]   = ms["child_recall"]["mean"]
                    entry["holdout_teen"]    = ms["teen_recall"]["mean"]
                    entry["holdout_fpr"]     = ms["adult_fpr"]["mean"]
                entry["holdout_n_test"]      = hd.get("n_test")
                entry["holdout_by_source"]   = hd.get("by_source_slim") or hd.get("by_source")
            except Exception:
                pass
        out[ver] = entry if (p.exists() or hp and hp.exists()) else None
    return out


THR_PATH = BASE_DIR / "data" / "thresholds.json"
THR_DEFAULTS = {"v6": 0.30, "v8": 0.30, "v11": 0.30, "k30tom": 0.10}

def load_thresholds() -> dict:
    """Read saved thresholds from disk; fallback to defaults."""
    if THR_PATH.exists():
        try:
            d = json.loads(THR_PATH.read_text())
            out = dict(THR_DEFAULTS)
            for k, v in d.items():
                if k in out and isinstance(v, (int, float)) and 0 < v <= 1:
                    out[k] = float(v)
            return out
        except Exception:
            pass
    return dict(THR_DEFAULTS)


def save_thresholds(d: dict) -> dict:
    """Persist thresholds; returns final stored dict."""
    cur = load_thresholds()
    for k, v in d.items():
        if k in cur and isinstance(v, (int, float)) and 0 < v <= 1:
            cur[k] = float(v)
    THR_PATH.parent.mkdir(parents=True, exist_ok=True)
    THR_PATH.write_text(json.dumps(cur, indent=2))
    return cur


class GalleryHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Less verbose default logging
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, ctype='text/html; charset=utf-8', code=200):
        body = text.encode('utf-8') if isinstance(text, str) else text
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self):
        self.send_response(404); self.end_headers(); self.wfile.write(b'not found')

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self._send_text(GALLERY_HTML)
        if path == '/api/data':
            return self._send_json(combined_data())
        if path == '/api/sessions':
            return self._send_json(grafana_sessions())
        if path == '/api/eval':
            return self._send_json(load_eval_data())
        if path == '/api/get_thr':
            return self._send_json(load_thresholds())
        if path == '/api/model_meta':
            return self._send_json(load_model_meta())
        if path == '/api/get_test_split':
            p = BASE_DIR / 'data' / 'v11_test_split.json'
            if not p.exists():
                return self._send_json({'test_ids': [], 'note': 'no test split file'})
            try:
                return self._send_json(json.loads(p.read_text()))
            except Exception as e:
                return self._send_json({'error': str(e)}, 500)
        if path.startswith('/img/'):
            name = path[len('/img/'):]
            if '..' in name or name.startswith('/'):
                return self._send_404()
            fpath = IMAGES_DIR / name
            if not fpath.exists() or not fpath.is_file():
                return self._send_404()
            ext = fpath.suffix.lower()
            ctype = {'.webp': 'image/webp', '.png': 'image/png', '.jpg': 'image/jpeg',
                     '.jpeg': 'image/jpeg', '.gif': 'image/gif'}.get(ext, 'application/octet-stream')
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'max-age=86400')
            self.end_headers()
            self.wfile.write(data)
            return
        self._send_404()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/save_thr':
            length = int(self.headers.get('Content-Length') or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode('utf-8') or '{}')
            except Exception as e:
                return self._send_json({'error': f'bad json: {e}'}, 400)
            try:
                stored = save_thresholds(payload)
                return self._send_json({'ok': True, 'thresholds': stored})
            except Exception as e:
                return self._send_json({'error': str(e)}, 500)
        if path == '/api/save':
            length = int(self.headers.get('Content-Length') or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode('utf-8') or '{}')
            except Exception as e:
                return self._send_json({'error': f'bad json: {e}'}, 400)
            changes = payload.get('changes') or {}
            to_del  = payload.get('delete') or []
            to_conf = payload.get('confirm') or []
            def _is_ls(k):  return isinstance(k, str) and k.startswith('ls_')
            def _is_k30(k): return isinstance(k, str) and k.startswith('g/')
            ls_changes  = {k: v for k, v in changes.items() if _is_ls(k)}
            k30_changes = {k: v for k, v in changes.items() if _is_k30(k)}
            gp_changes  = {k: v for k, v in changes.items() if not _is_ls(k) and not _is_k30(k)}
            ls_del  = [x for x in to_del if _is_ls(x)]
            k30_del = [x for x in to_del if _is_k30(x)]
            gp_del  = [x for x in to_del if not _is_ls(x) and not _is_k30(x)]
            k30_conf = [x for x in to_conf if _is_k30(x)]
            gp_conf  = [x for x in to_conf if not _is_k30(x)]
            try:
                n_ls_saved,  n_ls_del  = save_ls(ls_changes, ls_del)
                n_gp_saved,  n_gp_del  = save_grafana(gp_changes, gp_del, gp_conf)
                n_k30_saved, n_k30_del = save_k30(k30_changes, k30_del, k30_conf)
                _db_flush()
                return self._send_json({
                    'saved':   int(n_ls_saved) + int(n_gp_saved) + int(n_k30_saved),
                    'deleted': int(n_ls_del)   + int(n_gp_del)   + int(n_k30_del),
                })
            except Exception as e:
                return self._send_json({'error': str(e)}, 500)
        self._send_404()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=PORT)
    args = ap.parse_args()
    try:
        migrate_variants()
    except Exception as e:
        print(f'  migrate_variants() error: {e}')
    srv = HTTPServer(('0.0.0.0', args.port), GalleryHandler)
    print(f'Gallery server starting on http://localhost:{args.port} ...')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nShutdown.')


if __name__ == '__main__':
    main()
