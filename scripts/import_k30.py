#!/usr/bin/env python3
"""
import_k30.py
-------------
Import Tom's K=30 dataset (data/k30_ls_export_full.json) into gallery.db
as a third source ('k30') alongside LS and Grafana. Images are NOT downloaded
locally — they're served directly from S3 URLs (low-disk-footprint).

Each K30 item gets:
  • id              = external_id (g-XXX UUID)
  • thumb_url       = data.media (full image URL)
  • label           = derived from data.qwen3.min_age (child / teen / adult / NULL)
  • label_source    = 'qwen3'
  • label_confirmed = 0  (manual confirmation pending in gallery)
  • piper_result    = JSON blob of full data object (qwen3+siglip+models+...)

Usage:
    python scripts/import_k30.py             # incremental upsert (skip existing)
    python scripts/import_k30.py --force     # rewrite all rows
"""
import argparse, json, sqlite3, struct, sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / 'gallery.db'
K30_PATH = BASE / 'data' / 'k30_ls_export_full.json'

SCHEMA = """
CREATE TABLE IF NOT EXISTS k30_pool (
    id              TEXT PRIMARY KEY,
    inner_id        INTEGER,
    thumb_url       TEXT,
    local_path      TEXT,
    prompt          TEXT,
    label           TEXT,
    label_source    TEXT,
    label_confirmed INTEGER DEFAULT 0,
    labeled_at      TEXT,
    variant         TEXT,
    piper_result    TEXT,
    qwen3_result    TEXT,
    siglip_result   TEXT,
    extra           TEXT,
    deleted         INTEGER DEFAULT 0,
    deleted_at      TEXT,
    imported_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_k30_label ON k30_pool(label);
CREATE INDEX IF NOT EXISTS idx_k30_confirmed ON k30_pool(label_confirmed);
"""


def open_db_rw():
    """Open gallery.db read/write directly (no /tmp copy — we want writes to persist)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def cat_from_qwen3(qwen3):
    """child <=14, teen 15-17, adult 18+"""
    min_a = (qwen3 or {}).get('min_age')
    if min_a is None: return None
    if min_a <= 14: return 'child'
    if min_a <= 17: return 'teen'
    return 'adult'


def variant_for(label):
    if label in ('child','teen'): return 'positive'
    if label == 'adult': return 'negative'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='Overwrite existing rows')
    ap.add_argument('--limit', type=int, default=None, help='Import only first N items (testing)')
    args = ap.parse_args()

    if not K30_PATH.exists():
        print(f'ERR: {K30_PATH} not found', file=sys.stderr)
        sys.exit(1)

    print(f'Loading K30 export from {K30_PATH}...', flush=True)
    items = json.loads(K30_PATH.read_text())
    if args.limit:
        items = items[:args.limit]
    print(f'  {len(items)} items', flush=True)

    conn = open_db_rw()
    conn.executescript(SCHEMA)

    # Existing IDs
    existing = {r['id'] for r in conn.execute("SELECT id FROM k30_pool")}
    print(f'  existing in DB: {len(existing)}', flush=True)

    now = datetime.utcnow().isoformat()
    n_insert = n_skip = n_update = 0
    for it in items:
        d = it.get('data') or {}
        ext_id = d.get('external_id') or f"k30_{it.get('id')}"  # fall back to LS id
        if not args.force and ext_id in existing:
            n_skip += 1
            continue

        qwen3   = d.get('qwen3') or {}
        siglip  = d.get('siglip') or {}
        models  = d.get('models') or {}
        media   = d.get('media')

        label  = cat_from_qwen3(qwen3)
        prompt_text = qwen3.get('desc') or d.get('prompt') or ''
        # Build a `piper_result`-style structure for compatibility with the gallery's
        # existing rendering logic. The gallery expects siglip2_details with
        # underage.labels.{underage,adult} — we adapt K30's flat top_X arrays.
        sig_compat = {
            'siglip2_details': {
                'underage': {
                    'labels': {
                        'underage': {x['label']: x['score'] for x in siglip.get('top_underage', [])},
                        'adult':    {x['label']: x['score'] for x in siglip.get('top_adult', [])},
                    },
                    'minor': siglip.get('underage_score', 0.0),
                    'adult': siglip.get('adult_score', 0.0),
                }
            },
            'k30_models': models,   # k25_score, k30_score, prod_score, etc.
        }

        row = (
            ext_id,
            it.get('inner_id'),
            media,
            None,                                          # local_path
            prompt_text[:500],
            label,
            'qwen3',
            0,                                             # not confirmed
            None,                                          # labeled_at
            variant_for(label),
            json.dumps(sig_compat),
            json.dumps(qwen3),
            json.dumps(siglip),
            json.dumps({
                'k30_id':         it.get('id'),
                'ls_inner_id':    it.get('inner_id'),
                'source':         d.get('source'),
                'type':           d.get('type'),
                'k30_status':     qwen3.get('status'),
                'k30_block_rsn':  qwen3.get('block_reasons'),
                'k30_models':     models,
            }),
            0, None, now,
        )
        if ext_id in existing:
            conn.execute("""UPDATE k30_pool SET inner_id=?, thumb_url=?, local_path=?, prompt=?,
                            label=?, label_source=?, label_confirmed=?, labeled_at=?, variant=?,
                            piper_result=?, qwen3_result=?, siglip_result=?, extra=?, deleted=?,
                            deleted_at=?, imported_at=?  WHERE id=?""",
                         row[1:] + (ext_id,))
            n_update += 1
        else:
            conn.execute("""INSERT INTO k30_pool (id, inner_id, thumb_url, local_path, prompt,
                            label, label_source, label_confirmed, labeled_at, variant,
                            piper_result, qwen3_result, siglip_result, extra, deleted, deleted_at, imported_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
            n_insert += 1

    conn.commit()

    # Stats
    n_total = conn.execute("SELECT COUNT(*) FROM k30_pool").fetchone()[0]
    by_label = {l: conn.execute("SELECT COUNT(*) FROM k30_pool WHERE label=?", (l,)).fetchone()[0]
                for l in ('child','teen','adult')}
    n_null  = conn.execute("SELECT COUNT(*) FROM k30_pool WHERE label IS NULL").fetchone()[0]
    conn.close()

    print('\n=== Done ===', flush=True)
    print(f'  inserted: {n_insert}', flush=True)
    print(f'  updated:  {n_update}', flush=True)
    print(f'  skipped:  {n_skip}', flush=True)
    print(f'  total in k30_pool: {n_total}', flush=True)
    print(f'  by qwen3 label: {by_label}  (null_age={n_null})', flush=True)
    print(f'\n  → restart gallery_server.py and choose "K30 (Tom)" in source filter', flush=True)


if __name__ == '__main__':
    main()
