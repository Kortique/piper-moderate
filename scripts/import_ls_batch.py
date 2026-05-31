#!/usr/bin/env python3
"""
import_ls_batch.py — Import new Label Studio export into gallery.db.ls_images.

Workflow:
  1. You export a tab/view from Label Studio UI as JSON (full format)
     Project → Export → JSON → saves to data/ls_export_<date>.json
  2. Run this script with --input <that file> --session <label>
  3. Only items with task_id not yet in ls_images get inserted
  4. The siglip2_details / face_detect / qwen3 fields are LEFT EMPTY —
     they're filled later by scripts/moderate_ls_batch.py through Piper
     pipeline d2911d10bb.

Auto-migrates ls_images schema:
  - adds `session` column (legacy rows get session='legacy_initial')
  - adds `imported_at` column

LS JSON shape (Label Studio "JSON Full" format), one record per task:
{
  "id": 12345,
  "data": {
    "image": "https://...",   # primary media URL (one of these keys)
    "media": "https://...",
    "url":   "https://..."
  },
  "annotations": [
    {
      "result": [
        { "from_name": "tag", "value": {"choices": ["Underage"]} },
        { "from_name": "age", "value": {"number": 12} },
        ...
      ]
    }
  ]
}

Script extracts:
  - task_id from .id
  - media URL from .data.image / .data.media / .data.url (first non-empty)
  - age_from / age_to from annotations if present (else NULL)

Usage:
    python scripts/import_ls_batch.py --input data/ls_export_2026-05-28.json \\
                                       --session "2026-05-28_underage"
    python scripts/import_ls_batch.py --input ... --dry-run   # preview only
"""
import argparse, json, sqlite3, sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / 'gallery.db'

MEDIA_KEYS = ('image', 'media', 'url', 'imageUrl', 'image_url')


def ensure_schema(conn: sqlite3.Connection):
    """Lazy schema migration — adds session/imported_at columns if missing."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ls_images)").fetchall()}
    if 'session' not in cols:
        conn.execute("ALTER TABLE ls_images ADD COLUMN session TEXT")
        # Mark all existing rows as 'legacy_initial' so future filters work
        conn.execute("UPDATE ls_images SET session='legacy_initial' WHERE session IS NULL")
        print('  + ALTER ls_images ADD COLUMN session  (legacy rows marked legacy_initial)')
    if 'imported_at' not in cols:
        conn.execute("ALTER TABLE ls_images ADD COLUMN imported_at TEXT")
        print('  + ALTER ls_images ADD COLUMN imported_at')
    conn.commit()


def extract_media(data: dict) -> str | None:
    if not isinstance(data, dict):
        return None
    for k in MEDIA_KEYS:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def extract_age_range(annotations: list) -> tuple:
    """Return (age_from, age_to) if annotations contain age info, else (None, None)."""
    if not annotations:
        return None, None
    for ann in annotations:
        for res in (ann.get('result') or []):
            val = res.get('value') or {}
            # Common shapes: number, ratings, choices with age buckets
            if 'number' in val:
                try:
                    n = int(val['number'])
                    return n, n
                except (TypeError, ValueError):
                    pass
            if 'rating' in val:
                try:
                    n = int(val['rating'])
                    return n, n
                except (TypeError, ValueError):
                    pass
            choices = val.get('choices') or []
            # If choices include age buckets like "10-12" or "10..15"
            for c in choices:
                if not isinstance(c, str): continue
                for sep in ('-', '..', '–'):
                    if sep in c:
                        parts = c.split(sep)
                        try:
                            a = int(parts[0].strip()); b = int(parts[1].strip())
                            return a, b
                        except (TypeError, ValueError, IndexError):
                            pass
    return None, None


def parse_input(path: Path) -> list:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        # Some LS exports wrap in {"tasks": [...]} or similar
        for k in ('tasks', 'items', 'data'):
            if k in raw and isinstance(raw[k], list):
                raw = raw[k]; break
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"unexpected JSON top-level type: {type(raw).__name__}")
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',   required=True, help='LS export JSON file')
    ap.add_argument('--session', required=True, help='Session label (e.g. "2026-05-28_underage")')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force',   action='store_true',
                    help='UPSERT into existing task_ids (default: skip them)')
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERR: {inp} not found", file=sys.stderr); sys.exit(1)
    if not DB_PATH.exists():
        print(f"ERR: {DB_PATH} not found", file=sys.stderr); sys.exit(1)

    print(f"=== Import LS batch ===")
    print(f"  source : {inp}")
    print(f"  session: {args.session!r}")
    print(f"  dry-run: {args.dry_run}\n")

    raw = parse_input(inp)
    print(f"  loaded {len(raw)} tasks from JSON\n")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    existing = {r[0] for r in conn.execute("SELECT task_id FROM ls_images").fetchall()}
    print(f"  ls_images currently: {len(existing)} rows")

    new_rows = []; updated = 0; skipped_no_id = 0; skipped_no_media = 0; skipped_existing = 0
    for it in raw:
        tid = it.get('id')
        if not isinstance(tid, int):
            skipped_no_id += 1; continue
        data = it.get('data') or {}
        media = extract_media(data)
        if not media:
            skipped_no_media += 1; continue
        if tid in existing and not args.force:
            skipped_existing += 1; continue
        age_from, age_to = extract_age_range(it.get('annotations') or [])
        # Variant: 'positive' if any annotation chose Underage/Child/Teen, else None
        variant = None
        for ann in (it.get('annotations') or []):
            for res in (ann.get('result') or []):
                cs = (res.get('value') or {}).get('choices') or []
                for c in cs:
                    cl = (c or '').lower() if isinstance(c, str) else ''
                    if 'underage' in cl or 'child' in cl or 'teen' in cl or 'minor' in cl:
                        variant = 'positive'
        new_rows.append({
            'task_id':     tid,
            'media':       media,
            'age_from':    age_from,
            'age_to':      age_to,
            'variant':     variant,
            'session':     args.session,
        })

    print(f"  new tasks       : {len(new_rows)}")
    print(f"  skipped (existing, --force to upsert): {skipped_existing}")
    print(f"  skipped (no id) : {skipped_no_id}")
    print(f"  skipped (no media URL): {skipped_no_media}")
    if new_rows:
        print(f"  sample (first 3):")
        for r in new_rows[:3]:
            print(f"    id={r['task_id']:>8}  age={r['age_from']!s:>5}-{r['age_to']!s:<5}  variant={r['variant']!s:<8}  media={r['media'][:60]}...")

    if args.dry_run:
        print(f"\n[DRY-RUN] No writes. Re-run without --dry-run to apply.")
        return

    if not new_rows:
        print("\nNothing to insert.")
        return

    now = datetime.utcnow().isoformat()
    cur = conn.cursor()
    n_ins = 0
    for r in new_rows:
        # UPSERT pattern — keeps existing siglip2/etc if a row already exists with --force
        cur.execute("""
            INSERT INTO ls_images (task_id, media, variant, age_from, age_to, session, imported_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                media       = COALESCE(excluded.media, media),
                variant     = COALESCE(excluded.variant, variant),
                age_from    = COALESCE(excluded.age_from, age_from),
                age_to      = COALESCE(excluded.age_to,   age_to),
                session     = COALESCE(excluded.session,  session),
                imported_at = COALESCE(excluded.imported_at, imported_at)
        """, (r['task_id'], r['media'], r['variant'],
              r['age_from'], r['age_to'], r['session'], now))
        n_ins += 1
    conn.commit()
    conn.close()
    print(f"\n  ✓ inserted/upserted {n_ins} rows into ls_images")
    print(f"\nNext steps:")
    print(f"  python scripts/moderate_ls_batch.py --session {args.session!r} --workers 6")
    print(f"  python scripts/rescore_via_v11.py --source ls --workers 8")
    print(f"  python scripts/rescore_via_tom.py --source ls --workers 8")


if __name__ == '__main__':
    main()
