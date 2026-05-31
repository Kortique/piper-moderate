#!/usr/bin/env python3
"""
recover_json_caches.py — repair JSON cache files damaged by a partial write.

Symptom: `'utf-8' codec can't decode byte 0xNN in position M: invalid
continuation byte` somewhere in the middle of a multi-megabyte JSON cache
(v11_native_scores.json, tom_scores.json, etc).

Cause: typically a process was killed mid-write before the atomic .tmp →
rename completed, leaving a few stray bytes inside a string field. The JSON
structure is otherwise intact (closing `]` present), so we can recover by
dropping the bad bytes (errors='ignore') and re-serialising.

What this script does for each target:
  1. Read raw bytes.
  2. If strict UTF-8 parses → file is fine, skip.
  3. Otherwise decode with errors='ignore', then json.loads.
     If that still fails → bail out, leave file untouched, print the error.
  4. Move the original to <name>.broken-YYYYMMDD-HHMMSS as a backup.
  5. Re-serialise as pretty UTF-8 JSON via atomic .tmp + os.replace.

Usage:
    python scripts/recover_json_caches.py                    # both default files
    python scripts/recover_json_caches.py data/foo.json ...  # custom paths
    python scripts/recover_json_caches.py --dry-run          # report only
"""
import argparse, json, os, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = [
    BASE / 'data' / 'v11_native_scores.json',
    BASE / 'data' / 'tom_scores.json',
]


def repair_one(path: Path, dry_run: bool = False) -> str:
    if not path.exists():
        return f'SKIP (missing): {path}'
    raw = path.read_bytes()
    size_mb = len(raw) / (1024 * 1024)
    # Strict pass
    try:
        data = json.loads(raw.decode('utf-8'))
        return f'OK strict UTF-8 ({size_mb:.1f} MB, {len(data)} records): {path.name}'
    except UnicodeDecodeError as e:
        first_err = str(e)
    except json.JSONDecodeError as e:
        return f'FAIL  strict JSON decode  ({size_mb:.1f} MB): {path.name}\n        {e}'
    # Tolerant pass
    try:
        text = raw.decode('utf-8', errors='ignore')
        data = json.loads(text)
    except Exception as e:
        return (f'FAIL  even with errors=ignore ({size_mb:.1f} MB): {path.name}\n'
                f'        strict: {first_err}\n        ignore: {type(e).__name__}: {e}')

    n = len(data) if isinstance(data, list) else None
    if not isinstance(data, list):
        return f'FAIL  parsed but not a list (root={type(data).__name__}): {path.name}'
    bad_id_rows = sum(1 for r in data if isinstance(r, dict) and not r.get('id'))
    msg = (f'RECOVERABLE ({size_mb:.1f} MB, {n} records, '
           f'{bad_id_rows} rows with empty id): {path.name}')
    if dry_run:
        return msg + '   [dry-run, no write]'

    # Backup
    ts = time.strftime('%Y%m%d-%H%M%S')
    backup = path.with_name(f'{path.name}.broken-{ts}')
    os.replace(path, backup)
    # Atomic write of cleaned JSON
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    new_size_mb = path.stat().st_size / (1024 * 1024)
    return (f'REPAIRED {path.name}: {size_mb:.1f} MB → {new_size_mb:.1f} MB, '
            f'{n} records.  Backup: {backup.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*', help='JSON files to repair (default: v11/tom caches)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report status without writing anything')
    args = ap.parse_args()

    targets = [Path(p) for p in args.paths] if args.paths else DEFAULT_TARGETS
    print(f'\n=== JSON cache repair{" (dry-run)" if args.dry_run else ""} ===\n')
    for p in targets:
        print('  ' + repair_one(p, dry_run=args.dry_run))
    print()


if __name__ == '__main__':
    main()
