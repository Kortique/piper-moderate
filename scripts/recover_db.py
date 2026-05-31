#!/usr/bin/env python3
"""
recover_db.py — try to recover data from a corrupted SQLite file.

Strategy ladder (each step independent — first one that works wins):
  1. Inspect header — confirm it's actually SQLite, check the version
  2. Try opening with immutable=1 + integrity_check
  3. Try reading raw pages with pragma writable_schema
  4. As last resort: dump readable tables to a fresh DB via INSERT...SELECT
"""
import argparse, os, sqlite3, sys, shutil, tempfile
from pathlib import Path


SQLITE_HEADER = b"SQLite format 3\x00"


def inspect_header(path: Path):
    print(f"=== Header inspection: {path} ===")
    with open(path, "rb") as f:
        header = f.read(100)
    if not header.startswith(SQLITE_HEADER):
        print(f"  ❌ NOT a SQLite file. First 16 bytes: {header[:16]!r}")
        return False
    print(f"  ✓ SQLite magic present")
    page_size = int.from_bytes(header[16:18], "big")
    if page_size == 1:
        page_size = 65536
    print(f"  page_size: {page_size}")
    print(f"  file_change_counter: {int.from_bytes(header[24:28], 'big')}")
    print(f"  in-header page count: {int.from_bytes(header[28:32], 'big')}")
    print(f"  schema cookie: {int.from_bytes(header[40:44], 'big')}")
    print(f"  user_version: {int.from_bytes(header[60:64], 'big')}")
    expected_pages = path.stat().st_size // page_size
    print(f"  expected pages on disk: {expected_pages}")
    return True


def try_open(path: Path, mode_suffix: str):
    uri = f"file:{path.as_posix()}?{mode_suffix}"
    print(f"\n=== Trying open: {uri} ===")
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            print(f"  ✓ opened. Tables: {tables}")
            return conn
        except Exception as e:
            print(f"  ✗ sqlite_master read failed: {e}")
            conn.close()
            return None
    except Exception as e:
        print(f"  ✗ connect failed: {e}")
        return None


def try_dump(conn: sqlite3.Connection, out_path: Path):
    """Dump readable rows into a fresh DB."""
    print(f"\n=== Dumping recoverable rows → {out_path} ===")
    if out_path.exists():
        out_path.unlink()
    dst = sqlite3.connect(str(out_path))
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for tbl in tables:
        try:
            ddl = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (tbl,)).fetchone()[0]
            dst.execute(ddl)
            rows = cur.execute(f"SELECT * FROM {tbl}").fetchall()
            ncols = len(rows[0]) if rows else 0
            placeholders = ",".join("?" * ncols)
            if rows and ncols:
                dst.executemany(f"INSERT INTO {tbl} VALUES ({placeholders})", rows)
            print(f"  ✓ {tbl}: {len(rows)} rows recovered")
        except Exception as e:
            print(f"  ✗ {tbl}: {e}")
    dst.commit()
    dst.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to corrupted gallery.db")
    ap.add_argument("--out", help="Write recovered data here", default=None)
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        print(f"ERR: {src} not found"); sys.exit(1)

    # Step 1: header check
    if not inspect_header(src):
        print("\nFile is not a SQLite database at all — nothing to recover here.")
        sys.exit(1)

    # Make a working copy so we never touch the source
    work = Path(tempfile.gettempdir()) / f"_recover_{os.getpid()}.db"
    shutil.copy2(src, work)

    # Step 2: try plain open
    conn = try_open(work, "mode=ro")
    if conn is None:
        # Step 2b: try immutable
        conn = try_open(work, "mode=ro&immutable=1")
    if conn is None:
        # Step 2c: try nolock + immutable
        conn = try_open(work, "mode=ro&immutable=1&nolock=1")

    if conn is None:
        print("\n❌ All open attempts failed.")
        print("This file is too corrupted for Python's bundled sqlite3 to read.")
        print("You can try a newer sqlite3 CLI (3.40+) which has the .recover command:")
        print(f"  sqlite3 {src} \".recover\" > recovered.sql")
        print(f"  sqlite3 recovered.db < recovered.sql")
        sys.exit(2)

    # Step 3: integrity_check
    try:
        ic = conn.execute("PRAGMA integrity_check").fetchall()
        print(f"\nintegrity_check: {ic[:3]}{'...' if len(ic)>3 else ''}  ({len(ic)} rows)")
    except Exception as e:
        print(f"\nintegrity_check error: {e}")

    # Step 4: per-table counts (read-only)
    print("\n=== Counts in recoverable tables ===")
    for tbl in ("grafana_pool", "k30_pool", "ls_images"):
        try:
            r = conn.execute(
                f"SELECT COUNT(*) n, "
                f"SUM(CASE WHEN label_confirmed=1 THEN 1 ELSE 0 END) conf, "
                f"SUM(CASE WHEN label_source='human' THEN 1 ELSE 0 END) human, "
                f"SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) labeled "
                f"FROM {tbl}"
            ).fetchone()
            print(f"  {tbl:14}  total={r[0]:>5}  labeled={r[3] or 0:>5}  "
                  f"confirmed={r[1] or 0:>5}  human={r[2] or 0:>5}")
        except Exception as e:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl:14}  total={n:>5}  (no label_confirmed column)")
            except Exception as e2:
                print(f"  {tbl:14}  ERR: {e2}")

    # Step 5: optionally dump to fresh DB
    if args.out:
        try_dump(conn, Path(args.out))

    conn.close()

if __name__ == "__main__":
    main()
