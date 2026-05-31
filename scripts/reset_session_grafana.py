#!/usr/bin/env python3
"""
reset_session_grafana.py
------------------------
Selective reset of qwen3_result / piper_result for items in a specific
Grafana export batch — used when items were partially processed without
face_detect/qwen3 because those nodes were not active on Piper.

Default behaviour:
  • Targets export_batch == "2026-05-26 20:49 UTC"
  • Resets ONLY items where piper_result lacks face_detect_result
    (i.e. processed during the broken window)
  • Never touches items with label_confirmed=True (human-labeled)
  • Dry-run by default. Pass --execute to actually write the pool.

Usage:
    python scripts/reset_session_grafana.py                 # dry-run, default batch
    python scripts/reset_session_grafana.py --execute       # do the write
    python scripts/reset_session_grafana.py --batch "2026-05-26 20:49 UTC"
    python scripts/reset_session_grafana.py --all-incomplete --execute
        # reset every item in the batch whose qwen3_result OR piper_result is missing
"""

import argparse
import json
import os
import sys
import shutil
import time
from pathlib import Path

# fcntl + signal masking — mirror moderate_disagree.save_pool atomicity
try:
    import fcntl as _fcntl_mod
except ImportError:
    _fcntl_mod = None
import signal as _signal_mod

BASE_DIR = Path(__file__).resolve().parent.parent
POOL_FILE = BASE_DIR / "data" / "disagree_pool.json"
LOCK_FILE = str(POOL_FILE) + ".lock"
BACKUP_DIR = BASE_DIR / "backups" / "pool_resets"

DEFAULT_BATCH = "2026-05-26 20:49 UTC"


# ─────────────────────────────────────────────────────────────────────────────
# Pool IO (signal-safe atomic write, same recipe as moderate_disagree.save_pool)
# ─────────────────────────────────────────────────────────────────────────────

def try_lock_pool():
    if _fcntl_mod is None:
        return None  # Windows — no advisory lock available
    fh = open(LOCK_FILE, "w")
    try:
        _fcntl_mod.flock(fh.fileno(), _fcntl_mod.LOCK_EX | _fcntl_mod.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except (OSError, BlockingIOError):
        fh.close()
        return False


def load_pool():
    raw = POOL_FILE.read_bytes().rstrip(b'\x00')
    return json.loads(raw)


def save_pool(pool):
    """Atomic, signal-safe save — defers SIGTERM/SIGINT during rename."""
    try:
        old_term = _signal_mod.signal(_signal_mod.SIGTERM, _signal_mod.SIG_IGN)
    except Exception:
        old_term = None
    try:
        old_int = _signal_mod.signal(_signal_mod.SIGINT, _signal_mod.SIG_IGN)
    except Exception:
        old_int = None
    try:
        tmp = str(POOL_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp, POOL_FILE)
    finally:
        if old_term is not None:
            try: _signal_mod.signal(_signal_mod.SIGTERM, old_term)
            except Exception: pass
        if old_int is not None:
            try: _signal_mod.signal(_signal_mod.SIGINT, old_int)
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Item classification
# ─────────────────────────────────────────────────────────────────────────────

def item_in_batch(v, batch):
    if v.get("export_batch") == batch:
        return True
    # Fallback: session string starts with the date part of the batch
    sess = v.get("session", "") or ""
    return sess.startswith(batch.split()[0]) if sess else False


def needs_reset(v, mode):
    """mode = 'incomplete' (default) or 'all'"""
    # Human-confirmed labels never touched
    if v.get("label_confirmed"):
        return False, "human-confirmed"

    qr = v.get("qwen3_result")
    pr = v.get("piper_result")

    if mode == "all":
        return True, "all-mode"

    # Default: only items missing face_detect data
    # (these were processed during the broken window)
    if not pr:
        # Never processed — leave alone (moderate_disagree picks them up)
        return False, "untouched"
    if pr.get("error"):
        return True, "piper_error"
    if not pr.get("face_detect_result"):
        return True, "no_face_detect"
    # Has face_detect_result → processed correctly after fix
    return False, "ok_with_face_detect"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=DEFAULT_BATCH,
                    help=f"export_batch to target (default: {DEFAULT_BATCH!r})")
    ap.add_argument("--all-incomplete", action="store_true",
                    help="Reset every item in the batch (not just those missing face_detect)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write the pool. Without this, dry-run only.")
    args = ap.parse_args()

    mode = "all" if args.all_incomplete else "incomplete"

    print(f"=== reset_session_grafana ===")
    print(f"  batch    : {args.batch!r}")
    print(f"  mode     : {mode}  ({'reset every non-confirmed item in batch' if mode=='all' else 'reset only items missing face_detect_result'})")
    print(f"  dry-run  : {not args.execute}")
    print(f"  pool     : {POOL_FILE}")
    print()

    # Single-instance guard
    if args.execute:
        lh = try_lock_pool()
        if lh is False:
            print("ERROR: pool lock held by another process — refusing to write.")
            sys.exit(2)

    pool = load_pool()
    print(f"  pool size: {len(pool)} items")

    # Classify
    in_batch = [(k, v) for k, v in pool.items() if item_in_batch(v, args.batch)]
    print(f"  in batch : {len(in_batch)} items")

    to_reset = []
    skipped = {"human-confirmed": 0, "untouched": 0, "ok_with_face_detect": 0}
    for k, v in in_batch:
        flag, reason = needs_reset(v, mode)
        if flag:
            to_reset.append((k, v, reason))
        else:
            skipped[reason] = skipped.get(reason, 0) + 1

    print()
    print(f"  to reset : {len(to_reset)}")
    print(f"  skipped  : {dict(skipped)}")

    # Sample first 5 to-reset items
    if to_reset:
        print()
        print("  sample (first 5 candidates):")
        for k, v, reason in to_reset[:5]:
            qr_short = "ok" if v.get("qwen3_result") and not v["qwen3_result"].get("error") else "missing/err"
            pr_short = "ok" if v.get("piper_result") and not v["piper_result"].get("error") else "missing/err"
            has_fd = bool((v.get("piper_result") or {}).get("face_detect_result"))
            print(f"    {k[:12]}  reason={reason}  qwen3={qr_short}  piper={pr_short}  face_detect={has_fd}")

    if not args.execute:
        print()
        print(f"  [DRY-RUN] {len(to_reset)} items would be reset. Re-run with --execute to apply.")
        return

    if not to_reset:
        print()
        print("  Nothing to reset.")
        return

    # Backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"disagree_pool_pre_reset_{ts}.json"
    print()
    print(f"  backing up pool → {backup_path}")
    shutil.copy2(POOL_FILE, backup_path)

    # Apply resets
    for k, v, _ in to_reset:
        v["qwen3_result"] = None
        v["piper_result"] = None
        # If label was set by qwen3 (auto), clear it so moderate picks fresh data
        if v.get("label_source") == "qwen3":
            v["label"] = None
            v["label_confirmed"] = False

    save_pool(pool)
    print(f"  ✓ reset {len(to_reset)} items, pool saved.")
    print()
    print(f"  Next: python -u scripts/moderate_disagree.py --workers 6")


if __name__ == "__main__":
    main()
