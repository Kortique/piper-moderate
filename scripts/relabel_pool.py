#!/usr/bin/env python3
"""
relabel_pool.py
---------------
One-shot fixup: re-derive `label` for items already in the pool using the AI
verdict rule (V8 PASS → adult; V8 BLOCK → child if min_age ≤ 14 else teen),
**without** calling Piper again. Reads piper_result + qwen3_result that are
already there.

Never touches items with label_confirmed=True (human-set labels).
Dry-run by default — pass --execute to actually write.

Usage:
    python scripts/relabel_pool.py                     # dry-run, whole pool
    python scripts/relabel_pool.py --execute           # apply
    python scripts/relabel_pool.py --batch "2026-05-26 20:49 UTC" --execute
    python scripts/relabel_pool.py --source-only qwen3 # only items where label_source==qwen3
"""
import argparse, json, os, shutil, signal, sys, time
from pathlib import Path

try:
    import fcntl as _fcntl_mod
except ImportError:
    _fcntl_mod = None

BASE_DIR = Path(__file__).resolve().parent.parent
POOL_FILE = BASE_DIR / "data" / "disagree_pool.json"
THR_FILE  = BASE_DIR / "data" / "thresholds.json"
LOCK_FILE = str(POOL_FILE) + ".lock"
BACKUP_DIR = BASE_DIR / "backups" / "pool_resets"
GALLERY_DB = BASE_DIR / "gallery.db"


def load_v8_thr() -> float:
    try:
        return float(json.loads(THR_FILE.read_text()).get("v8", 0.51))
    except Exception:
        return 0.51


def try_lock_pool():
    if _fcntl_mod is None:
        return None
    fh = open(LOCK_FILE, "w")
    try:
        _fcntl_mod.flock(fh.fileno(), _fcntl_mod.LOCK_EX | _fcntl_mod.LOCK_NB)
        fh.write(str(os.getpid())); fh.flush()
        return fh
    except (OSError, BlockingIOError):
        fh.close()
        return False


def save_pool(pool):
    try: old_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception: old_term = None
    try: old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception: old_int = None
    try:
        tmp = str(POOL_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, ensure_ascii=False)
            f.flush()
            try: os.fsync(f.fileno())
            except (OSError, AttributeError): pass
        os.replace(tmp, POOL_FILE)
    finally:
        if old_term is not None:
            try: signal.signal(signal.SIGTERM, old_term)
            except Exception: pass
        if old_int is not None:
            try: signal.signal(signal.SIGINT, old_int)
            except Exception: pass


def derive_min_age(v: dict):
    """Min ageFrom across qwen3.faces, fallback to face_detect_result.ageFrom."""
    qr = v.get("qwen3_result") or {}
    faces = qr.get("faces") or []
    ages = []
    for f in faces:
        if isinstance(f, dict):
            af = f.get("ageFrom")
            if isinstance(af, (int, float)): ages.append(int(af))
    if ages:
        return min(ages)
    pr = v.get("piper_result") or {}
    fd = pr.get("face_detect_result") or {}
    af = fd.get("ageFrom")
    if isinstance(af, (int, float)):
        return int(af)
    return None


def derive_label(v: dict, thr: float):
    """Apply the AI verdict rule. Returns (new_label, reason) or (None, why_skip)."""
    pr = v.get("piper_result") or {}
    if not pr or pr.get("error"):
        return None, "no_piper_result"
    sd = (pr.get("siglip2_details") or {}).get("underage") or {}
    v8_score = (sd.get("lgbm") or {}).get("score")
    if v8_score is None:
        # No V8 score at all → treat as adult (no underage signal)
        return "adult", "no_v8_score→adult"
    score = float(v8_score)
    if score < thr:
        return "adult", f"v8_pass ({score:.3f}<{thr})"
    # Blocked
    min_age = derive_min_age(v)
    if min_age is not None and min_age <= 14:
        return "child", f"v8_block ({score:.3f}≥{thr}) + age≤14"
    return "teen", f"v8_block ({score:.3f}≥{thr}) + age>14_or_unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None, help="Only items with exact export_batch")
    ap.add_argument("--batch-prefix", default=None,
                    help="Only items whose export_batch STARTS WITH this string "
                         "(e.g. '2026-05-26' catches '2026-05-26 20:49 UTC' and '2026-05-26 21:01 UTC')")
    ap.add_argument("--source-only", default=None,
                    help="Only items with this label_source (e.g. qwen3)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write the pool (default dry-run)")
    args = ap.parse_args()

    thr = load_v8_thr()
    print(f"V8 threshold       = {thr}")
    print(f"Pool file          = {POOL_FILE}")
    print(f"Filter batch       = {args.batch!r}")
    print(f"Filter batch-prefix= {args.batch_prefix!r}")
    print(f"Filter source      = {args.source_only!r}")
    print(f"Dry-run            = {not args.execute}")
    print()

    if args.execute:
        lh = try_lock_pool()
        if lh is False:
            print("ERROR: pool lock held by another process — refusing to write.")
            sys.exit(2)

    pool = json.loads(POOL_FILE.read_bytes().rstrip(b"\x00"))
    print(f"Pool size: {len(pool)}")

    changes = []
    skipped = {"human_confirmed": 0, "no_piper": 0, "no_change": 0,
               "wrong_batch": 0, "wrong_source": 0}

    for k, v in pool.items():
        if v.get("label_confirmed"):
            skipped["human_confirmed"] += 1; continue
        if args.batch and v.get("export_batch") != args.batch:
            skipped["wrong_batch"] += 1; continue
        if args.batch_prefix:
            eb = v.get("export_batch") or ""
            if not eb.startswith(args.batch_prefix):
                skipped["wrong_batch"] += 1; continue
        if args.source_only and v.get("label_source") != args.source_only:
            skipped["wrong_source"] += 1; continue
        new_lbl, reason = derive_label(v, thr)
        if new_lbl is None:
            skipped["no_piper"] += 1; continue
        old_lbl = v.get("label")
        if new_lbl == old_lbl:
            skipped["no_change"] += 1; continue
        changes.append((k, old_lbl, new_lbl, reason))

    print(f"\nWill change: {len(changes)}")
    print(f"Skipped: {dict(skipped)}")

    # Summary by transition
    from collections import Counter
    trans = Counter((c[1] or "—", c[2]) for c in changes)
    if trans:
        print("\nTransitions (old → new : count):")
        for (o, n), c in sorted(trans.items(), key=lambda x: -x[1]):
            print(f"  {o:>6} → {n:<6}: {c}")

    if changes[:5]:
        print("\nSample (first 5):")
        for k, o, n, r in changes[:5]:
            print(f"  {k[:12]}  {o or '—':>6} → {n:<6}  ({r})")

    if not args.execute:
        print(f"\n[DRY-RUN] {len(changes)} would be relabelled. Re-run with --execute to apply.")
        return

    if not changes:
        print("\nNothing to write.")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bp = BACKUP_DIR / f"disagree_pool_pre_relabel_{ts}.json"
    print(f"\nBacking up → {bp}")
    shutil.copy2(POOL_FILE, bp)

    for k, _o, new, _r in changes:
        v = pool[k]
        v["label"] = new
        v["label_source"] = "ai_verdict"
        v["label_confirmed"] = False

    save_pool(pool)
    print(f"  ✓ relabelled {len(changes)} items, pool saved.")

    # ── ALSO sync gallery.db so the gallery sees the new labels without a
    # separate init_db.py invocation. We touch only the three label columns;
    # piper_result/qwen3_result/etc are unchanged.
    if GALLERY_DB.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(GALLERY_DB))
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            n_synced = 0
            for k, _o, new, _r in changes:
                cur.execute(
                    "UPDATE grafana_pool SET label=?, label_source=?, label_confirmed=0 WHERE id=?",
                    (new, "ai_verdict", k),
                )
                n_synced += cur.rowcount
            conn.commit()
            conn.close()
            print(f"  ✓ gallery.db synced ({n_synced}/{len(changes)} rows updated)")
        except Exception as e:
            print(f"  ⚠ gallery.db sync failed: {e}")
            print(f"    Run `python scripts/init_db.py` to sync manually.")
    else:
        print(f"  (gallery.db not found — skipping DB sync)")


if __name__ == "__main__":
    main()
