#!/usr/bin/env python3
"""
diag_labels.py — Audit current label state and compare with all available
backups. Read-only. No writes. Run from project root.
"""
import sqlite3, sys, shutil, os, tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CUR_DB = BASE / 'gallery.db'
BAK_DIR = BASE / 'backups'


def safe_count(db_path):
    """Open via copy to avoid WAL locking issues with running gallery."""
    if not db_path.exists():
        return None
    tmp = Path(tempfile.gettempdir()) / f'_diag_{os.getpid()}.db'
    try:
        shutil.copy2(db_path, tmp)
    except Exception as e:
        return {'err': f'copy: {e}'}
    try:
        conn = sqlite3.connect(str(tmp))
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return {'err': f'connect: {e}'}
    out = {'size': db_path.stat().st_size}
    for tbl in ('grafana_pool', 'k30_pool', 'ls_images'):
        try:
            row = conn.execute(
                f"SELECT "
                f"  COUNT(*) AS n, "
                f"  SUM(CASE WHEN label_confirmed=1 THEN 1 ELSE 0 END) AS conf, "
                f"  SUM(CASE WHEN label_source='human' THEN 1 ELSE 0 END) AS human, "
                f"  SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) AS labeled, "
                f"  SUM(CASE WHEN deleted=1 THEN 1 ELSE 0 END) AS deleted "
                f"FROM {tbl}"
            ).fetchone()
            out[tbl] = {
                'total':     row['n'],
                'confirmed': row['conf'] or 0,
                'human':     row['human'] or 0,
                'labeled':   row['labeled'] or 0,
                'deleted':   row['deleted'] or 0,
            }
        except Exception as e:
            out[tbl] = {'err': str(e)}
    conn.close()
    try: tmp.unlink()
    except: pass
    return out


def fmt(snapshot):
    if not snapshot:
        return '  (no data)'
    if 'err' in snapshot:
        return f'  ERR: {snapshot["err"]}'
    lines = [f'  size={snapshot["size"]:,} bytes']
    for tbl in ('grafana_pool', 'k30_pool', 'ls_images'):
        d = snapshot.get(tbl)
        if not d:
            continue
        if 'err' in d:
            lines.append(f'  {tbl:14}  ERR: {d["err"]}')
        else:
            lines.append(
                f'  {tbl:14}  total={d["total"]:>5}  '
                f'labeled={d["labeled"]:>5}  '
                f'confirmed={d["confirmed"]:>5}  '
                f'human={d["human"]:>5}  '
                f'deleted={d["deleted"]:>4}'
            )
    return '\n'.join(lines)


def main():
    print('=' * 80)
    print('CURRENT  gallery.db')
    print('=' * 80)
    print(fmt(safe_count(CUR_DB)))
    print()

    backups = sorted(BAK_DIR.glob('gallery_*.db'), key=lambda p: p.stat().st_mtime, reverse=True)
    print('=' * 80)
    print(f'BACKUPS ({len(backups)} found) — newest first')
    print('=' * 80)
    for bp in backups:
        from datetime import datetime
        mt = datetime.fromtimestamp(bp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f'\n{bp.name}  ({mt})')
        print(fmt(safe_count(bp)))

    print()
    print('=' * 80)
    print('SUMMARY — look at "human" / "confirmed" columns:')
    print('=' * 80)
    print('  • If CURRENT shows lower numbers than any BACKUP — labels were wiped')
    print('    between those snapshots. The backup with the highest count is the')
    print('    candidate for restoration.')
    print('  • If all backups show the same counts as CURRENT — the labels never')
    print('    persisted in this DB; check disagree_pool.json or other source.')


if __name__ == '__main__':
    main()
