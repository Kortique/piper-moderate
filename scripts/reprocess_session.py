#!/usr/bin/env python3
"""
reprocess_session.py
--------------------
Force-redo qwen3_result + piper_result for a specific session (by date prefix or
exact export_batch) in data/disagree_pool.json. Used after fixing the providers
list in moderate_disagree.py so that qwen3 + face_detect actually run.

Usage:
    python scripts/reprocess_session.py --date 2026-05-26
    python scripts/reprocess_session.py --batch "2026-05-26 21:01 UTC"
    python scripts/reprocess_session.py --date 2026-05-26 --workers 6
"""
import argparse, json, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
POOL = BASE / 'data' / 'disagree_pool.json'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date',  default=None, help='YYYY-MM-DD — wipe qwen3+piper for items with that export_batch date')
    ap.add_argument('--batch', default=None, help='exact export_batch string')
    args = ap.parse_args()

    if not args.date and not args.batch:
        ap.error('--date or --batch required')

    pool = json.loads(POOL.read_text(encoding='utf-8'))
    matched = []
    for k, v in pool.items():
        eb = (v.get('export_batch') or '')
        if args.batch and eb == args.batch:
            matched.append(k)
        elif args.date and eb.startswith(args.date):
            matched.append(k)

    print(f'matched items: {len(matched)}')
    if not matched:
        return

    # Reset qwen3_result and piper_result so moderate_disagree picks them up again
    for k in matched:
        pool[k]['qwen3_result'] = None
        pool[k]['piper_result'] = None
        # don't touch label_confirmed=True items (manual labels stay)
    # Atomic save (mirrors moderate_disagree save_pool)
    tmp = str(POOL) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    import os
    os.replace(tmp, POOL)
    print(f'reset qwen3+piper for {len(matched)} items.')
    print('Now run:  python scripts/moderate_disagree.py --workers 12')


if __name__ == '__main__':
    main()
