#!/usr/bin/env python3
"""
init_db.py — Initialize SQLite database from existing JSON files.

Creates gallery.db with two tables:
  - ls_images    (from qwen3_age_results.json)
  - grafana_pool (from data/disagree_pool.json)

Safe to re-run: uses INSERT OR REPLACE, no data loss.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --dry-run
"""
import json, sqlite3, argparse
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = BASE_DIR / "gallery.db"
LS_FILE    = BASE_DIR / "qwen3_age_results.json"
POOL_FILE  = BASE_DIR / "data" / "disagree_pool.json"


SCHEMA = """
CREATE TABLE IF NOT EXISTS ls_images (
    task_id         INTEGER PRIMARY KEY,
    media           TEXT,
    variant         TEXT,
    category        TEXT,
    age_from        INTEGER,
    age_to          INTEGER,
    launch_id       TEXT,
    siglip2_labels  TEXT,     -- JSON array
    siglip2_passed  INTEGER,  -- 0/1
    siglip2_details TEXT,     -- JSON object
    face_detect     TEXT,     -- JSON object
    error           TEXT,
    processed_at    TEXT,
    extra           TEXT      -- JSON for any other fields
);

CREATE TABLE IF NOT EXISTS grafana_pool (
    id              TEXT PRIMARY KEY,
    thumb_url       TEXT,
    local_path      TEXT,
    prompt          TEXT,
    label           TEXT,
    label_source    TEXT,
    label_confirmed INTEGER,  -- 0/1
    labeled_at      TEXT,
    variant         TEXT,
    export_batch    TEXT,
    exported_at     TEXT,
    piper_result    TEXT,     -- JSON object
    qwen3_result    TEXT,     -- JSON object
    extra           TEXT,     -- JSON for any other fields
    deleted         INTEGER DEFAULT 0,
    deleted_at      TEXT
);
"""

KNOWN_LS_FIELDS = {
    'task_id', 'media', 'variant', 'category', 'age',
    'launch_id', 'siglip2_labels', 'siglip2_passed', 'siglip2_details',
    'face_detect_result', 'error', 'piper_processed_at',
}

KNOWN_POOL_FIELDS = {
    'id', 'thumb_url', 'local_path', 'prompt', 'label', 'label_source',
    'label_confirmed', 'labeled_at', 'variant', 'export_batch', 'exported_at',
    'piper_result', 'qwen3_result', 'deleted', 'deleted_at',
}


def j(v):
    """Serialize to JSON string or None."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


def import_ls(conn, data: dict, dry_run=False):
    cur = conn.cursor()
    count = 0
    for raw in data.values():
        tid = raw.get('task_id')
        if not tid:
            continue
        age = raw.get('age') or {}
        extra = {k: v for k, v in raw.items() if k not in KNOWN_LS_FIELDS}
        if not dry_run:
            cur.execute("""
                INSERT OR REPLACE INTO ls_images
                (task_id, media, variant, category, age_from, age_to,
                 launch_id, siglip2_labels, siglip2_passed, siglip2_details,
                 face_detect, error, processed_at, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                tid,
                raw.get('media'),
                raw.get('variant'),
                raw.get('category'),
                age.get('ageFrom'),
                age.get('ageTo'),
                raw.get('launch_id'),
                j(raw.get('siglip2_labels')),
                1 if raw.get('siglip2_passed') else 0,
                j(raw.get('siglip2_details')),
                j(raw.get('face_detect_result')),
                raw.get('error'),
                raw.get('piper_processed_at'),
                j(extra) if extra else None,
            ))
        count += 1
    conn.commit()
    return count


def import_pool(conn, data: dict, dry_run=False):
    cur = conn.cursor()
    count = 0
    for pid, raw in data.items():
        extra = {k: v for k, v in raw.items() if k not in KNOWN_POOL_FIELDS}
        if not dry_run:
            cur.execute("""
                INSERT OR REPLACE INTO grafana_pool
                (id, thumb_url, local_path, prompt, label, label_source,
                 label_confirmed, labeled_at, variant, export_batch, exported_at,
                 piper_result, qwen3_result, extra, deleted, deleted_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                pid,
                raw.get('thumb_url'),
                raw.get('local_path'),
                raw.get('prompt'),
                raw.get('label'),
                raw.get('label_source'),
                1 if raw.get('label_confirmed') else 0,
                raw.get('labeled_at'),
                raw.get('variant'),
                raw.get('export_batch'),
                raw.get('exported_at'),
                j(raw.get('piper_result')),
                j(raw.get('qwen3_result')),
                j(extra) if extra else None,
                1 if raw.get('deleted') else 0,
                raw.get('deleted_at'),
            ))
        count += 1
    conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent writes
    conn.executescript(SCHEMA)

    ls_count = pool_count = 0

    if LS_FILE.exists():
        ls_data = json.loads(LS_FILE.read_bytes().rstrip(b'\x00'))
        ls_count = import_ls(conn, ls_data, args.dry_run)
        print(f"ls_images:    {ls_count} rows {'(dry run)' if args.dry_run else 'imported'}")

    if POOL_FILE.exists():
        pool_data = json.loads(POOL_FILE.read_bytes().rstrip(b'\x00'))
        pool_count = import_pool(conn, pool_data, args.dry_run)
        print(f"grafana_pool: {pool_count} rows {'(dry run)' if args.dry_run else 'imported'}")

    conn.close()
    if not args.dry_run:
        print(f"\nDB: {DB_PATH}  ({DB_PATH.stat().st_size // 1024} KB)")


if __name__ == '__main__':
    main()
