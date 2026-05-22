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

LGBM_THRESHOLD   = 0.80   # V6 legacy threshold
V6_LGBM_THRESHOLD  = 0.80
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


def _v11_lgbm_score(underage_labels: dict, adult_labels: dict) -> float | None:
    """V11s80 — slim 80 features, BCI-aware (no_underage_ optional)."""
    booster, feats = _load_lgbm_py('V11s80',
                                    'lgbm_underage_v11s80.txt',
                                    'lgbm_v11s80_features.json')
    if booster is None:
        return None
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

    # Load from dedicated eval result files
    for version, files in EVAL_FILES.items():
        for fpath in files:
            if not fpath.exists():
                continue
            items = json.loads(fpath.read_text(encoding="utf-8"))
            for item in items:
                gid = _eval_id_to_gallery_id(item["id"])
                entry = _make_eval_entry(
                    lgbm        = item.get("lgbm",  0.0),
                    minor       = item.get("minor", 0.0),
                    human_label = item.get("human_label"),
                    variant     = item.get("variant"),
                )
                if gid not in result:
                    result[gid] = {}
                if version not in result[gid]:
                    result[gid][version] = entry

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

    # ── V8pas80-v2 + V11s80 inline scoring (replaces former V10 inline) ──────
    # Both models are slim 80-feature LGBM with BCI aggregates. We compute
    # scores in a single pass over the same source data (grafana_pool + LS).
    if _LGB_AVAILABLE:
        try:
            conn_inline = _db_connect()
            for row in conn_inline.execute("SELECT id, label, variant, piper_result FROM grafana_pool"):
                pid = row["id"]
                # Source: either v317 rescore file (full underage+adult), or piper_result siglip2_details
                rec_under, rec_adult, und_minor = None, None, 0.0
                if pid in v317_map:
                    rec = v317_map[pid]
                    rec_under = rec.get('underage_labels', {})
                    rec_adult = rec.get('adult_labels', {})
                    und_minor = rec.get('minor', 0.0)
                else:
                    try:
                        pr  = json.loads(row["piper_result"]) if row["piper_result"] else {}
                        det = (pr.get('siglip2_details') or {}).get('underage', {})
                        lbl_data = det.get('labels', {})
                        rec_under = lbl_data.get('underage', {})
                        rec_adult = lbl_data.get('adult', {})
                        und_minor = det.get('minor', 0.0)
                    except Exception:
                        continue
                if rec_under is None and rec_adult is None:
                    continue
                # V8
                if "v8" not in result.get(pid, {}):
                    score_v8 = _v8_lgbm_score(rec_under or {}, rec_adult or {})
                    if score_v8 is not None:
                        result.setdefault(pid, {})["v8"] = _make_eval_entry(
                            lgbm=score_v8, minor=und_minor,
                            human_label=row["label"], variant=row["variant"], version='v8')
                # V11
                if "v11" not in result.get(pid, {}):
                    score_v11 = _v11_lgbm_score(rec_under or {}, rec_adult or {})
                    if score_v11 is not None:
                        result.setdefault(pid, {})["v11"] = _make_eval_entry(
                            lgbm=score_v11, minor=und_minor,
                            human_label=row["label"], variant=row["variant"], version='v11')

            # ls_rescored_map — items with full underage+adult+no_underage
            for gid, rec in ls_rescored_map.items():
                lbl = rec.get('label')
                rec_under = rec.get('underage_labels', {})
                rec_adult = rec.get('adult_labels', {})
                minor = rec.get('minor', 0.0)
                if "v8" not in result.get(gid, {}):
                    s8 = _v8_lgbm_score(rec_under, rec_adult)
                    if s8 is not None:
                        result.setdefault(gid, {})["v8"] = _make_eval_entry(
                            lgbm=s8, minor=minor, human_label=lbl,
                            variant=default_variant(lbl) if lbl else None, version='v8')
                if "v11" not in result.get(gid, {}):
                    s11 = _v11_lgbm_score(rec_under, rec_adult)
                    if s11 is not None:
                        result.setdefault(gid, {})["v11"] = _make_eval_entry(
                            lgbm=s11, minor=minor, human_label=lbl,
                            variant=default_variant(lbl) if lbl else None, version='v11')

            # qwen3 LS items
            ls_json_inline = BASE_DIR / "qwen3_age_results.json"
            if ls_json_inline.exists():
                try:
                    import ast as _ast
                    raw_inline = ls_json_inline.read_bytes().rstrip(b"\x00").decode("utf-8")
                    ls_data_inline = json.loads(raw_inline)
                    for v in ls_data_inline.values():
                        gid = f"ls_{v.get('task_id', '')}"
                        if "v8" in result.get(gid, {}) and "v11" in result.get(gid, {}):
                            continue
                        siglip_raw = v.get("siglip2_details")
                        if not siglip_raw:
                            continue
                        det = ((_ast.literal_eval(siglip_raw) if isinstance(siglip_raw, str) else siglip_raw) or {}).get("underage", {})
                        lbl_data = det.get("labels", {})
                        u = lbl_data.get("underage", {}); a = lbl_data.get("adult", {})
                        af = (v.get("age") or {}).get("ageFrom")
                        cat = ls_cat(af) if af is not None else None
                        if "v8" not in result.get(gid, {}):
                            s8 = _v8_lgbm_score(u, a)
                            if s8 is not None:
                                result.setdefault(gid, {})["v8"] = _make_eval_entry(
                                    lgbm=s8, minor=det.get("minor", 0.0), human_label=cat, version='v8')
                        if "v11" not in result.get(gid, {}):
                            s11 = _v11_lgbm_score(u, a)
                            if s11 is not None:
                                result.setdefault(gid, {})["v11"] = _make_eval_entry(
                                    lgbm=s11, minor=det.get("minor", 0.0), human_label=cat, version='v11')
                except Exception:
                    pass
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

def combined_data():
    ls      = load_ls()
    grafana = load_grafana()
    # Newest grafana first, then LS
    grafana.sort(key=lambda r: r.get("export_batch") or "", reverse=True)
    return grafana + ls

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
.card.hidden-pending .hide-btn { opacity: 1; background: rgba(180,40,40,.7); color: #fff; }

/* source badge top-left */
.src-badge {
  position: absolute; top: 4px; left: 4px; font-size: 9px; padding: 1px 4px;
  border-radius: 2px; pointer-events: none; font-weight: bold; letter-spacing: .3px;
}
.src-badge.ls    { background: rgba(30,70,120,.85); color: #7ab0dd; }
.src-badge.gr    { background: rgba(60,20,100,.85); color: #c792ea; }
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

/* pagination */
#pagination { display:flex; align-items:center; justify-content:center; gap:5px; padding:12px 16px; flex-wrap:wrap; }
#pagination button { background:#1e1e1e; border:1px solid #444; color:#aaa; padding:3px 9px; border-radius:3px; cursor:pointer; font-size:11px; font-family:monospace; }
#pagination button:hover { background:#2a2a2a; color:#fff; }
#pagination button.active { background:#2a3a5a; border-color:#4a6aaa; color:#88aaff; font-weight:bold; }
#pagination button:disabled { opacity:.3; cursor:default; }
#pagination .pg-info { color:#555; font-size:10px; padding:0 5px; }

/* ── live sim panel ──────────────────────────────────────────────────────── */
#sim-panel {
  display: none; background: #0e1a12; border-bottom: 1px solid #1a3a22;
  padding: 7px 14px; gap: 18px; align-items: center; flex-wrap: wrap;
}
#sim-panel.open { display: flex; }
#sim-panel .sim-title {
  font-size: 10px; color: #4a9a5a; font-weight: bold; text-transform: uppercase;
  letter-spacing: .5px; white-space: nowrap;
}
#sim-panel .sim-slider-group {
  display: flex; align-items: center; gap: 6px;
}
#sim-panel .sim-label { font-size: 10px; color: #5a8a6a; white-space: nowrap; }
#sim-panel input[type=range] {
  width: 110px; height: 3px; cursor: pointer; accent-color: #3aaa5a;
}
#sim-panel .sim-val {
  font-size: 11px; color: #80da90; font-family: monospace;
  min-width: 36px; text-align: right;
}
#sim-panel .sim-sep { color: #1e4a2a; font-size: 14px; }
.sim-btn {
  font-size: 11px; padding: 3px 10px; border-radius: 4px; cursor: pointer;
  background: #0e2010; border: 1px solid #2a6a3a; color: #4aaa5a;
}
.sim-btn.active { background: #1a3a22; border-color: #3aaa5a; color: #80da90; font-weight: bold; }
.sim-badge {
  position: absolute; bottom: 4px; left: 4px; font-size: 9px; padding: 1px 5px;
  border-radius: 2px; pointer-events: none; z-index: 3;
}
.sim-badge.blocked { background: rgba(180,30,30,.88); color: #ffa0a0; border: 1px solid #aa3030; }
.sim-badge.passed  { background: rgba(15,80,25,.88);  color: #80ee90; border: 1px solid #2a8a3a; }

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
  align-items: flex-start; gap: 18px; flex-wrap: wrap;
  padding: 8px 14px; background: #0e0e1a; border-bottom: 1px solid #222244;
  font-size: 11px;
}
#lgbm-bar.open { display: flex; }
#lgbm-bar .lgbm-thr-block { display: flex; flex-direction: column; gap: 6px; }
#lgbm-bar .lgbm-thr-row   { display: flex; align-items: center; gap: 6px; }
.lgbm-bar-title {
  font-size: 10px; color: #5a5aaa; font-weight: bold; text-transform: uppercase;
  letter-spacing: .5px; white-space: nowrap;
}
.lgbm-thr-group { display: flex; align-items: center; gap: 5px; }
.lgbm-thr-label { font-size: 10px; color: #5a5a8a; white-space: nowrap; }
.lgbm-thr-input {
  width: 52px; background: #16162a; border: 1px solid #333366; color: #9090dd;
  padding: 2px 5px; border-radius: 3px; font-size: 11px; font-family: monospace; text-align:center;
}
#lgbm-apply-btn {
  background: #0e0e2a; border: 1px solid #3a3a8a; color: #8080cc;
  padding: 3px 12px; border-radius: 4px; cursor: pointer; font-size: 11px; font-family: monospace;
}
#lgbm-apply-btn:hover { background: #1a1a3a; border-color: #5a5aaa; color: #b0b0ff; }
#lgbm-stats {
  display: inline-grid; vertical-align: middle;
  grid-template-columns: auto repeat(3, minmax(96px, auto));
  gap: 1px 8px;
  margin-left: 8px;
  font-size: 11px; font-family: monospace;
}
#lgbm-stats .lst-hdr      { font-weight: bold; text-align: center; padding: 1px 4px; border-bottom: 1px solid #2a2a2a; }
#lgbm-stats .lst-rowlbl   { color: #888; text-transform: uppercase; font-size: 9px; padding-right: 4px; align-self: center; }
#lgbm-stats .lst-cell     { text-align: right; padding: 0 4px; }
#lgbm-stats .lst-pct      { font-weight: bold; }
#lgbm-stats .lst-abs      { color: #666; font-size: 9px; margin-left: 2px; }
#lgbm-stats .lst-dot      { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-left: 4px; vertical-align: -1px; }
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

  <label id="pipe-wrap" style="display:flex;align-items:center;gap:4px">SigLip2
    <select id="f-pipe">
      <option value="all">Все</option>
      <option value="underage">blocked</option>
      <option value="passed">passed</option>
      <option value="unprocessed">не прогнан</option>
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

  <label id="sim-filter-wrap" style="display:none;align-items:center;gap:4px">Live Sim
    <select id="f-sim">
      <option value="all">все</option>
      <option value="blocked">blocked</option>
      <option value="passed">passed</option>
      <option value="nodata">нет данных</option>
    </select>
  </label>

  <span id="status"></span>
  <button class="sim-btn" id="sim-toggle-btn" onclick="toggleSim()">⚡ Live Sim</button>
  <button id="confirm-btn" onclick="confirmPage()" title="Подтвердить все AI-метки на текущей странице без изменений">✓ Подтвердить страницу</button>
  <button id="save-btn" onclick="saveChanges()">💾 Сохранить</button>
  <button id="lgbm-toggle-btn" onclick="toggleLgbmBar()" title="LGBM пороги и статистика">⚙ LGBM</button>
</header>

<div id="lgbm-bar">
  <div class="lgbm-thr-block">
    <span class="lgbm-bar-title">Пороги</span>
    <div class="lgbm-thr-row"><span class="lgbm-thr-label">V8 ≥</span>
      <input type="number" id="thr-v8" class="lgbm-thr-input" value="0.30" min="0.01" max="1" step="0.01"></div>
    <div class="lgbm-thr-row"><span class="lgbm-thr-label">V11 ≥</span>
      <input type="number" id="thr-v11" class="lgbm-thr-input" value="0.30" min="0.01" max="1" step="0.01"></div>
    <div class="lgbm-thr-row"><span class="lgbm-thr-label">V6 ≥</span>
      <input type="number" id="thr-v6" class="lgbm-thr-input" value="0.80" min="0.01" max="1" step="0.01"></div>
    <button id="lgbm-apply-btn" onclick="applyLgbmThresholds()">Применить</button>
  </div>
  <div id="lgbm-stats"></div>
</div>
<div id="sim-panel">
  <span class="sim-title">⚡ Live Sim</span>
  <span class="sim-sep">|</span>

  <div class="sim-slider-group">
    <span class="sim-label">LGBM thr</span>
    <input type="range" id="sl-lgbm" min="0.5" max="1" step="0.01" value="0.90"
      oninput="simCfg.lgbm_threshold=+this.value; document.getElementById('sv-lgbm').textContent=this.value; _simSliderChange()">
    <span class="sim-val" id="sv-lgbm">0.90</span>
  </div>

  <div class="sim-slider-group">
    <span class="sim-label">LGBM</span>
    <input type="range" id="sl-lgbm-en" min="0" max="1" step="1" value="1"
      oninput="simCfg.lgbm_enabled=+this.value; document.getElementById('sv-lgbm-en').textContent=(+this.value?'ON':'OFF'); _simSliderChange()">
    <span class="sim-val" id="sv-lgbm-en">ON</span>
  </div>

  <span class="sim-sep">|</span>

  <div class="sim-slider-group">
    <span class="sim-label">Auto trig</span>
    <input type="range" id="sl-atrig" min="0" max="1" step="0.01" value="0.85"
      oninput="simCfg.auto_trigger_minor_risk=+this.value; document.getElementById('sv-atrig').textContent=this.value; _simSliderChange()">
    <span class="sim-val" id="sv-atrig">0.85</span>
  </div>

  <div class="sim-slider-group">
    <span class="sim-label">Confidence</span>
    <input type="range" id="sl-conf" min="0" max="1" step="0.01" value="0.60"
      oninput="simCfg.underage_confidence=+this.value; document.getElementById('sv-conf').textContent=this.value; _simSliderChange()">
    <span class="sim-val" id="sv-conf">0.60</span>
  </div>

  <div class="sim-slider-group">
    <span class="sim-label">Min score</span>
    <input type="range" id="sl-minscore" min="0" max="0.05" step="0.001" value="0.004"
      oninput="simCfg.underage_min_score=+this.value; document.getElementById('sv-minscore').textContent=this.value; _simSliderChange()">
    <span class="sim-val" id="sv-minscore">0.004</span>
  </div>

  <span class="sim-sep">|</span>

  <div class="sim-slider-group">
    <span class="sim-label">×20 mult</span>
    <input type="range" id="sl-x20" min="1" max="60" step="0.5" value="20"
      oninput="simCfg.tag_double_bang=+this.value; document.getElementById('sv-x20').textContent=this.value; _simSliderChange()">
    <span class="sim-val" id="sv-x20">20</span>
  </div>

  <div class="sim-slider-group">
    <span class="sim-label">×5 mult</span>
    <input type="range" id="sl-x5" min="1" max="30" step="0.5" value="5"
      oninput="simCfg.tag_single_bang=+this.value; document.getElementById('sv-x5').textContent=this.value; _simSliderChange()">
    <span class="sim-val" id="sv-x5">5</span>
  </div>

  <span class="sim-sep">|</span>
  <button class="sim-btn" onclick="resetSim()" style="font-size:10px;padding:2px 8px">↺ Reset</button>
  <button class="sim-btn" onclick="applyFilter()" style="font-size:10px;padding:2px 8px;margin-left:2px;border-color:#5a9a3a;color:#90da60">🔄 Перестроить</button>
  <span id="sim-stats" style="font-size:10px;color:#3a6a4a;margin-left:4px"></span>
</div>

<div class="stats-bar" id="stats-bar">—</div>
<div class="gallery" id="gallery"></div>
<div id="pagination"></div>

<div id="lb" onclick="closeLb()">
  <img id="lb-img" src="" onclick="event.stopPropagation()">
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
let allData = [], sessions = [], changes = {}, hidden = new Set(), toConfirm = new Set(), filteredData = [], currentPage = 1;
let evalData = {}, evalActive = false, evalVersion = 'v8';

// ── live sim ──────────────────────────────────────────────────────────────────
// ── LGBM thresholds & badge logic ────────────────────────────────────────────
let lgbmV6Thr  = 0.80;
let lgbmV8Thr  = 0.30;
let lgbmV11Thr = 0.30;

function lgbmThrFor(version) {
  if (version === 'v8')  return lgbmV8Thr;
  if (version === 'v11') return lgbmV11Thr;
  return lgbmV6Thr;
}

function lgbmIsBlocked(entry, version) {
  if (!entry) return null;
  return entry.lgbm >= lgbmThrFor(version);
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
  if (ev8) {
    const bl = ev8.lgbm >= lgbmV8Thr;
    parts.push(`<span class="${bl ? 'lgbm-v8-blocked' : 'lgbm-v8-ok'}">${bl ? '⛔' : '✓'} V8:${ev8.lgbm.toFixed(2)}</span>`);
  }
  if (ev11) {
    const bl = ev11.lgbm >= lgbmV11Thr;
    parts.push(`<span class="${bl ? 'lgbm-v11-blocked' : 'lgbm-v11-ok'}">${bl ? '⛔' : '✓'} V11:${ev11.lgbm.toFixed(2)}</span>`);
  }
  if (ev6) {
    const bl = ev6.lgbm >= lgbmV6Thr;
    parts.push(`<span class="${bl ? 'lgbm-v6-blocked' : 'lgbm-v6-ok'}">${bl ? '⛔' : '✓'} V6:${ev6.lgbm.toFixed(2)}</span>`);
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
    if (!wrap || simActive) return;
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
  lgbmV8Thr  = parseFloat(document.getElementById('thr-v8').value)  || 0.30;
  lgbmV11Thr = parseFloat(document.getElementById('thr-v11').value) || 0.30;
  lgbmV6Thr  = parseFloat(document.getElementById('thr-v6').value)  || 0.80;
  rebuildAllPipeBadges();
  document.querySelectorAll('.eval-row').forEach(el => {
    const cid = el.id.replace('eval-', '');
    const item = allData.find(x => x.id === cid);
    if (item) renderEvalRowContent(el, item);
  });
  updateLgbmStats();
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
  const stats = {
    v8:  {child: 0, teen: 0, adult: 0, has: 0},
    v11: {child: 0, teen: 0, adult: 0, has: 0},
    v6:  {child: 0, teen: 0, adult: 0, has: 0},
  };
  allData.forEach(r => {
    const cat = (r.label || '').toLowerCase();
    if (!cats.includes(cat)) return;
    totals[cat]++;
    const ed = evalData[r.id] || {};
    for (const ver of ['v8','v11','v6']) {
      if (!ed[ver]) continue;
      stats[ver].has++;
      if (ed[ver].lgbm >= lgbmThrFor(ver)) stats[ver][cat]++;
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

  // Build rows. Each row: rowlabel | v8 cell | v11 cell | v6 cell
  const versions = ['v8','v11','v6'];
  const verNames = {v8: 'V8', v11: 'V11', v6: 'V6'};
  const verCls   = {v8: 'lgbm-v8-blocked', v11: 'lgbm-v11-blocked', v6: 'lgbm-v6-blocked'};
  const html = [];

  // Header row
  html.push(`<div class="lst-rowlbl"></div>`);
  for (const ver of versions) {
    html.push(`<div class="lst-hdr ${verCls[ver]}">${verNames[ver]}</div>`);
  }

  // Rows for each category
  const catIcons = {child: '👶', teen: '🧒', adult: '🧑'};
  const catLabels = {child: 'child', teen: 'teen', adult: 'adult ✗FP'};
  for (const cat of cats) {
    html.push(`<div class="lst-rowlbl">${catIcons[cat]} ${catLabels[cat]}</div>`);
    // Find best score in this row for ★
    const scores = versions.map(v => {
      if (!stats[v].has || !totals[cat]) return null;
      return stats[v][cat] / totals[cat] * 100;
    });
    let bestIdx = -1, bestVal = null;
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] == null) continue;
      if (cat === 'adult') {  // lower is better
        if (bestVal == null || scores[i] < bestVal) { bestVal = scores[i]; bestIdx = i; }
      } else {                 // higher is better
        if (bestVal == null || scores[i] > bestVal) { bestVal = scores[i]; bestIdx = i; }
      }
    }
    for (let i = 0; i < versions.length; i++) {
      const ver = versions[i];
      if (!stats[ver].has || !totals[cat]) {
        html.push(`<div class="lst-cell"><span class="lst-pct" style="color:#444">—</span></div>`);
        continue;
      }
      const n = stats[ver][cat], m = totals[cat];
      const pct = scores[i];
      const dot = dotClass(cat, pct);
      const star = (i === bestIdx) ? '<span class="lst-best" title="best in this row">★</span> ' : '';
      const colorCls = (i === bestIdx) ? 'lst-best' : '';
      html.push(
        `<div class="lst-cell" title="${verNames[ver]} on ${cat}: ${n}/${m}">`
        + `${star}<span class="lst-pct ${colorCls}">${pct.toFixed(1)}%</span>`
        + `<span class="lst-dot ${dot}"></span>`
        + `<span class="lst-abs">${n}/${m}</span>`
        + `</div>`
      );
    }
  }

  const el = document.getElementById('lgbm-stats');
  if (el) el.innerHTML = html.join('');
}

let simActive = false;
let _simDebounce = null;
function _simSliderChange() {
  // Update stats immediately (cheap); debounce the full re-render
  if (simActive) {
    let blocked = 0, passed = 0, nodata = 0;
    allData.forEach(r => {
      const res = evalSim(r);
      if (!res) nodata++;
      else if (res.blocked) blocked++;
      else passed++;
    });
    document.getElementById('sim-stats').textContent =
      `⛔ ${blocked}  ✓ ${passed}  — ${nodata}`;
  }
  clearTimeout(_simDebounce);
  _simDebounce = setTimeout(() => applyFilter(), 250);
}
let simCfg = {
  lgbm_threshold:           0.90,
  lgbm_enabled:             1,
  auto_trigger_minor_risk:  0.85,
  underage_confidence:      0.60,
  underage_min_score:       0.004,
  tag_double_bang:          20,
  tag_single_bang:          5,
};
const SIM_DEFAULTS = { ...simCfg };

function _combineScores(obj) {
  const vals = Object.values(obj);
  if (!vals.length) return 0;
  return 1 - vals.reduce((p, v) => p * (1 - Math.max(0, Math.min(0.9999, +v || 0))), 1);
}

function _getSimDetails(r) {
  // siglip2_details is always inside piper_result for both sources
  if (r.piper_result && r.piper_result.siglip2_details) return r.piper_result.siglip2_details;
  // fallback: old LS serialisation where it was top-level
  return r.siglip2_details || null;
}

// Returns { blocked: bool, minor: number, lgbm: number|null } or null if no data
function evalSim(r) {
  const det = _getSimDetails(r);
  if (!det || !det.underage) return null;
  const u = det.underage;
  const { lgbm_threshold, lgbm_enabled, auto_trigger_minor_risk,
          underage_confidence, underage_min_score, tag_double_bang, tag_single_bang } = simCfg;

  let minor, adult;
  const labelData = u.labels;
  if (labelData) {
    const uAdj = {};
    for (const [k, v] of Object.entries(labelData.underage || {})) {
      const m = k.match(/:x(\\d+(?:\\.\\d+)?)$/);
      const origMult = m ? parseFloat(m[1]) : 1.0;
      const newMult  = origMult === 20 ? tag_double_bang : origMult === 5 ? tag_single_bang : origMult;
      uAdj[k] = Math.min((origMult !== 1.0 ? +v / origMult : +v) * newMult, 0.999);
    }
    const aAdj = {};
    for (const [k, v] of Object.entries(labelData.adult || {})) aAdj[k] = +v || 0;
    minor = _combineScores(uAdj);
    adult = _combineScores(aAdj);
  } else {
    minor = +u.minor || 0;
    adult = +u.adult || 0;
  }

  const total      = minor + adult;
  const confidence = total > 0 ? minor / total : 0;
  const siglipBlocked = minor >= auto_trigger_minor_risk ||
                        (confidence >= underage_confidence && minor >= underage_min_score);

  const lgbmScore   = (u.lgbm && u.lgbm.score != null) ? +u.lgbm.score : null;
  const lgbmBlocked = lgbm_enabled >= 0.5 && lgbmScore !== null && lgbmScore >= lgbm_threshold;

  return { blocked: siglipBlocked || lgbmBlocked, minor, lgbm: lgbmScore };
}

function toggleSim() {
  simActive = !simActive;
  document.getElementById('sim-panel').classList.toggle('open', simActive);
  document.getElementById('sim-toggle-btn').classList.toggle('active', simActive);
  document.getElementById('sim-filter-wrap').style.display = simActive ? 'flex' : 'none';
  if (!simActive) document.getElementById('f-sim').value = 'all';
  applyFilter();
}

function resetSim() {
  Object.assign(simCfg, SIM_DEFAULTS);
  document.getElementById('sl-lgbm').value      = simCfg.lgbm_threshold;
  document.getElementById('sv-lgbm').textContent = simCfg.lgbm_threshold;
  document.getElementById('sl-lgbm-en').value    = simCfg.lgbm_enabled;
  document.getElementById('sv-lgbm-en').textContent = simCfg.lgbm_enabled ? 'ON' : 'OFF';
  document.getElementById('sl-atrig').value       = simCfg.auto_trigger_minor_risk;
  document.getElementById('sv-atrig').textContent = simCfg.auto_trigger_minor_risk;
  document.getElementById('sl-conf').value        = simCfg.underage_confidence;
  document.getElementById('sv-conf').textContent  = simCfg.underage_confidence;
  document.getElementById('sl-minscore').value    = simCfg.underage_min_score;
  document.getElementById('sv-minscore').textContent = simCfg.underage_min_score;
  document.getElementById('sl-x20').value         = simCfg.tag_double_bang;
  document.getElementById('sv-x20').textContent   = simCfg.tag_double_bang;
  document.getElementById('sl-x5').value          = simCfg.tag_single_bang;
  document.getElementById('sv-x5').textContent    = simCfg.tag_single_bang;
  onSimChange();
}

function onSimChange() {
  applyFilter();
  if (!simActive) return;
  // Update sim stats in panel (counts over ALL data, independent of filter)
  let blocked = 0, passed = 0, nodata = 0;
  allData.forEach(r => {
    const res = evalSim(r);
    if (!res) nodata++;
    else if (res.blocked) blocked++;
    else passed++;
  });
  document.getElementById('sim-stats').textContent =
    `⛔ ${blocked}  ✓ ${passed}  — ${nodata}`;
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
  const pf  = document.getElementById('f-pipe').value;
  const aq3 = document.getElementById('f-age-q3').value;
  const afd = document.getElementById('f-age-fd').value;
  const ef  = evalActive ? document.getElementById('f-eval-outcome').value : 'all';

  // Update active eval version
  if (evalActive) evalVersion = document.getElementById('f-eval-ver').value;

  // Show/hide session & pipeline dropdowns
  const showSession = sf === 'grafana' || sf === 'all';
  document.getElementById('session-wrap').style.display = showSession ? '' : 'none';
  const showPipe = sf !== 'labelstudio';
  document.getElementById('pipe-wrap').style.display = showPipe ? '' : 'none';

  let d = allData;

  if (sf === 'labelstudio') d = d.filter(r => r.source === 'labelstudio');
  else if (sf === 'grafana') d = d.filter(r => r.source === 'grafana');

  if (sf !== 'labelstudio' && ssf !== 'all')
    d = d.filter(r => r.session === ssf);

  if (lf === 'unlabeled')   d = d.filter(r => !effectiveLabel(r));
  else if (lf === 'unconfirmed') d = d.filter(r => r.source === 'grafana' && !r.label_confirmed && effectiveLabel(r));
  else if (lf !== 'all')   d = d.filter(r => effectiveLabel(r) === lf);

  if (pf !== 'all' && showPipe)
    d = d.filter(r => r.source === 'grafana' && pipeStatus(r) === pf);

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

  // Live sim filter
  if (simActive) {
    const sf2 = document.getElementById('f-sim').value;
    if (sf2 === 'blocked') d = d.filter(r => { const s = evalSim(r); return s && s.blocked; });
    else if (sf2 === 'passed')  d = d.filter(r => { const s = evalSim(r); return s && !s.blocked; });
    else if (sf2 === 'nodata')  d = d.filter(r => !evalSim(r));
  }

  filteredData = d;
  currentPage  = 1;
  updateStats();
  renderPage();
}

function updateStats() {
  const totals = { ls:0, gr:0, child:0, teen:0, adult:0, unlabeled:0, confirmed:0, unconfirmed:0 };
  allData.forEach(r => {
    if (r.source === 'labelstudio') totals.ls++; else totals.gr++;
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
    `<span class="s-ch"><b>child:</b> ${totals.child}</span>` +
    `<span class="s-tn"><b>teen:</b> ${totals.teen}</span>` +
    `<span class="s-ad"><b>adult:</b> ${totals.adult}</span>` +
    `<span class="s-un"><b>без разм.:</b> ${totals.unlabeled}</span>` +
    (totals.unconfirmed ? `<span style="color:#cc8030"><b>⚡ AI (не подтв.):</b> ${totals.unconfirmed}</span>` : '') +
    (totals.confirmed   ? `<span style="color:#50a050"><b>✓ подтверждено:</b> ${totals.confirmed}</span>` : '') +
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

  // Source badge
  const sbadge = document.createElement('div');
  sbadge.className = 'src-badge ' + (isLS ? 'ls' : 'gr');
  sbadge.textContent = isLS ? 'LS' : 'GR';
  wrap.appendChild(sbadge);

  // Pipeline/eval badge — see buildPipeBadge() for the rules.
  if (!simActive) buildPipeBadge(wrap, r);

  // Live sim badge (replaces pipe-badge when sim is active)
  if (simActive) {
    const sr = evalSim(r);
    if (sr !== null) {
      const sb = document.createElement('div');
      sb.className = 'sim-badge ' + (sr.blocked ? 'blocked' : 'passed');
      const minorStr = sr.minor != null ? sr.minor.toFixed(3) : '—';
      const lgbmStr  = sr.lgbm  != null ? sr.lgbm.toFixed(2)  : '—';
      sb.textContent = sr.blocked ? `⛔ m:${minorStr}` : `✓ m:${minorStr}`;
      sb.title = `minor=${minorStr}  lgbm=${lgbmStr}`;
      wrap.appendChild(sb);
    }
  }

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
    .filter(r => r.source === 'grafana' && !r.label_confirmed && !toConfirm.has(r.id) && effectiveLabel(r)).length;
  cbtn.className = pageUnconfirmed > 0 ? 'has-pending' : '';
  cbtn.textContent = pageUnconfirmed > 0
    ? `✓ Подтвердить страницу (${pageUnconfirmed})`
    : '✓ Подтвердить страницу';
}

function confirmPage() {
  const pgSize = parseInt(document.getElementById('f-pgsize').value);
  const start  = (currentPage-1)*pgSize;
  const page   = filteredData.slice(start, start+pgSize);
  let count = 0;
  page.forEach(r => {
    if (r.source === 'grafana' && !r.label_confirmed && !toConfirm.has(r.id) && effectiveLabel(r)) {
      toConfirm.add(r.id);
      // Visually update card badge to confirmed
      const cardEl = document.getElementById('card-' + r.id);
      if (cardEl) {
        const cb = cardEl.querySelector('.confirm-badge');
        if (cb) { cb.className = 'confirm-badge human'; cb.textContent = '✓'; cb.title = 'Подтверждено'; }
      }
      count++;
    }
  });
  if (count > 0) updateDirty();
}

async function saveChanges() {
  if (!Object.keys(changes).length && !hidden.size && !toConfirm.size) return;
  document.getElementById('save-btn').textContent = '⏳ Сохранение…';
  try {
    const r = await fetch('/api/save', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({changes, delete: [...hidden], confirm: [...toConfirm]}),
    });
    const res = await r.json();
    changes = {}; hidden.clear(); toConfirm.clear();
    document.getElementById('save-btn').className = '';
    document.getElementById('save-btn').textContent =
      `✓ Сохранено (${res.saved} изм. / ${res.deleted} удал.)`;
    document.getElementById('status').textContent = '';
    await loadData();
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

function openLb(r) {
  document.getElementById('lb-img').src = r._serve_url || r.thumb_url || '';
  document.getElementById('lb-prompt').textContent = r.prompt || '';
  document.getElementById('lb').classList.add('open');
}
function closeLb() { document.getElementById('lb').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeLb(); closePm(); } });

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
  const ev6  = allVersionData['v6']  || null;
  const ev8  = allVersionData['v8']  || null;
  const ev11 = allVersionData['v11'] || null;

  // ── Comparison: V8 | V11 | V6 (production | candidate | legacy) ──────────
  // Three-column when V6 data is available, two-column otherwise.
  const cmpLeft      = ev8 || ev6 || null;
  const cmpMid       = ev11 || null;
  const cmpRight     = ev6 || null;
  const cmpLeftName  = ev8 ? 'V8pas80' : 'V6';
  const cmpMidName   = 'V11s80';
  const cmpRightName = 'V6';
  const cmpLeftVer   = ev8 ? 'v8' : 'v6';
  const cmpMidVer    = 'v11';
  const cmpRightVer  = 'v6';
  const showThree    = !!(cmpLeft && cmpMid && cmpRight && cmpRight !== cmpLeft);

  if (cmpLeft && cmpMid) {
    const cmp = document.createElement('div');
    cmp.className = 'eval-cmp' + (showThree ? ' three-col' : '');

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

    // LGBM row — color depends ONLY on each model's own threshold (no diff coloring)
    const lgbmLbl = document.createElement('div'); lgbmLbl.className = 'cmp-lbl'; lgbmLbl.textContent = 'LGBM';
    const mkValOrDash = (entry, thr) => {
      if (!entry) return Object.assign(document.createElement('div'), {className: 'cmp-val lo', textContent: '—'});
      return mkVal(entry.lgbm, entry.lgbm >= thr, 'hi-lgbm');  // no compareTo → pure threshold color
    };
    cmp.append(lgbmLbl,
      mkValOrDash(cmpLeft, lgbmThrFor(cmpLeftVer)),
      mkValOrDash(cmpMid,  lgbmThrFor(cmpMidVer)));
    if (showThree) cmp.append(mkValOrDash(cmpRight, lgbmThrFor(cmpRightVer)));

    // separator
    const sep = document.createElement('div'); sep.className = 'cmp-sep'; cmp.appendChild(sep);

    // outcome row
    const outLbl = document.createElement('div'); outLbl.className = 'cmp-lbl'; outLbl.textContent = '';
    cmp.append(outLbl, mkOut(cmpLeft, cmpLeftVer), mkOut(cmpMid, cmpMidVer));
    if (showThree) cmp.append(mkOut(cmpRight, cmpRightVer));

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
document.getElementById('f-source').addEventListener('change', applyFilter);
document.getElementById('f-session').addEventListener('change', applyFilter);
document.getElementById('f-label').addEventListener('change', applyFilter);
document.getElementById('f-pipe').addEventListener('change', applyFilter);
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
document.getElementById('f-sim').addEventListener('change', applyFilter);
document.getElementById('thr-v8').addEventListener('change', applyLgbmThresholds);
document.getElementById('thr-v11').addEventListener('change', applyLgbmThresholds);
document.getElementById('thr-v6').addEventListener('change', applyLgbmThresholds);

// initial load
loadData();
</script>
</body></html>
"""


# ─── HTTP handler ────────────────────────────────────────────────────────────
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
        if path == '/api/save':
            length = int(self.headers.get('Content-Length') or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode('utf-8') or '{}')
            except Exception as e:
                return self._send_json({'error': f'bad json: {e}'}, 400)
            changes = payload.get('changes') or {}
            to_del  = payload.get('delete') or []
            to_conf = payload.get('confirm') or []
            ls_changes = {k: v for k, v in changes.items() if isinstance(k, str) and k.startswith('ls_')}
            gp_changes = {k: v for k, v in changes.items() if not (isinstance(k, str) and k.startswith('ls_'))}
            ls_del = [x for x in to_del if isinstance(x, str) and x.startswith('ls_')]
            gp_del = [x for x in to_del if not (isinstance(x, str) and x.startswith('ls_'))]
            try:
                n_ls_saved, n_ls_del = save_ls(ls_changes, ls_del)
                n_gp_saved, n_gp_del = save_grafana(gp_changes, gp_del, to_conf)
                _db_flush()
                return self._send_json({
                    'saved': int(n_ls_saved) + int(n_gp_saved),
                    'deleted': int(n_ls_del) + int(n_gp_del),
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
    print(f'Gallery server starting on http://localhost:{args.port} ...')
    srv = HTTPServer(('0.0.0.0', args.port), GalleryHandler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nShutdown.')


if __name__ == '__main__':
    main()
