#!/usr/bin/env python3
"""
check_shadow.py — probe a DB file (anywhere on disk) for label-state counts.
Read-only. Reads via /tmp copy so it can probe even a locked DB.

Usage:
    python scripts/check_shadow.py <path-to-gallery.db>

Example:
    python scripts/check_shadow.py "C:\\Users\\Kortique\\Desktop\\shadow_recover\\gallery.db"
"""
import sqlite3, shutil, sys, tempfile, os
from pathlib import Path
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python scripts/check_shadow.py <path-to-gallery.db>")
    sys.exit(1)

src = Path(sys.argv[1])
if not src.exists():
    print(f"ERR: file not found: {src}")
    sys.exit(1)

mt = datetime.fromtimestamp(src.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
size_mb = src.stat().st_size / 1024 / 1024
print(f"File: {src}")
print(f"  size: {size_mb:.1f} MB    mtime: {mt}")
print()

tmp = Path(tempfile.gettempdir()) / f"_shadow_check_{os.getpid()}.db"
try:
    shutil.copy2(src, tmp)
    c = sqlite3.connect(str(tmp))
    c.row_factory = sqlite3.Row
    for tbl in ("grafana_pool", "k30_pool", "ls_images"):
        try:
            r = c.execute(
                f"SELECT COUNT(*) n, "
                f"  SUM(CASE WHEN label_confirmed=1 THEN 1 ELSE 0 END) conf, "
                f"  SUM(CASE WHEN label_source='human' THEN 1 ELSE 0 END) human, "
                f"  SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) labeled, "
                f"  SUM(CASE WHEN deleted=1 THEN 1 ELSE 0 END) deleted "
                f"FROM {tbl}"
            ).fetchone()
            print(f"  {tbl:14}  total={r['n']:>5}  "
                  f"labeled={r['labeled'] or 0:>5}  "
                  f"confirmed={r['conf'] or 0:>5}  "
                  f"human={r['human'] or 0:>5}  "
                  f"deleted={r['deleted'] or 0:>4}")
        except sqlite3.OperationalError as e:
            try:
                n = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl:14}  total={n:>5}  (no label_confirmed column)")
            except Exception:
                print(f"  {tbl:14}  ERR: {e}")
    c.close()
finally:
    try: tmp.unlink()
    except: pass
