#!/usr/bin/env python3
r"""
import_borderlands.py — register local image files into a new `borderlands_pool`
table so the gallery + moderation pipeline can score them.

Source: a Windows folder of mixed media. Images are copied into
        <project>/data/borderlands/ (so the gallery's /img/ route serves them),
        videos are skipped. Each registered row gets a deterministic id
        bl_<sha8>_<safefilename> so re-runs are idempotent.

Workflow:
    1. python scripts/import_borderlands.py --src "B:\Pop\Pop5\...\..." --dry-run
    2. python scripts/import_borderlands.py --src "B:\Pop\Pop5\...\..."
    3. python scripts/moderate_borderlands.py --workers 6
    4. python scripts/rescore_via_v11.py  --source borderlands --workers 8
    5. python scripts/rescore_via_tom.py  --source borderlands --workers 8

The script auto-creates the borderlands_pool table on first run.
"""
import argparse, hashlib, shutil, sqlite3, sys
from datetime import datetime
from pathlib import Path

# Pillow is required for the resize-on-import step. Fall back to plain copy
# only if --no-resize was passed.
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tif', '.tiff'}
VID_EXTS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v', '.wmv', '.flv'}

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / 'gallery.db'
DEST_DIR_REL = Path('data') / 'borderlands'
MAX_DIM = 1024              # longest side after resize
JPEG_QUALITY = 88           # JPEG/WEBP quality on save
ANIMATED_FRAME_LIMIT = 1    # GIFs: only keep the first frame to keep it simple


def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS borderlands_pool (
            id              TEXT PRIMARY KEY,
            local_path      TEXT NOT NULL,
            original_path   TEXT,
            filename        TEXT,
            label           TEXT,
            label_source    TEXT,
            label_confirmed INTEGER DEFAULT 0,
            labeled_at      TEXT,
            variant         TEXT,
            piper_result    TEXT,
            qwen3_result    TEXT,
            processed_at    TEXT,
            deleted         INTEGER DEFAULT 0,
            deleted_at      TEXT,
            session         TEXT DEFAULT 'borderlands',
            imported_at     TEXT
        )
    """)
    conn.commit()


def make_id(src_path: Path) -> str:
    """Deterministic id from absolute source path. Re-runs on same file -> same id."""
    abs_str = str(src_path.resolve()).lower()
    h = hashlib.sha1(abs_str.encode('utf-8')).hexdigest()[:8]
    # Sanitised filename suffix for human readability
    stem = ''.join(c if c.isalnum() else '_' for c in src_path.stem)[:30]
    return f'bl_{h}_{stem}'


def safe_dest_name(src_path: Path, item_id: str) -> str:
    """Return a unique-yet-readable filename for the copy in data/borderlands/."""
    # id already encodes uniqueness; keep the original extension
    return item_id + src_path.suffix.lower()


def copy_and_resize(src: Path, dest: Path, max_dim: int = MAX_DIM) -> tuple[bool, str]:
    """Open `src`, resize so the longest side <= max_dim (only DOWN-scale, never
    up), save to `dest`. Preserves format (jpg→jpg, png→png, webp→webp, etc.).
    Returns (ok, info) where info is a short status string for logging.

    On any PIL error falls back to shutil.copy2 — better to have an unresized
    image in the dataset than to skip it.
    """
    if not PIL_AVAILABLE:
        shutil.copy2(src, dest)
        return True, 'copied (no PIL)'

    try:
        with Image.open(src) as im:
            # Honour EXIF orientation so portraits don't get rotated mid-pipeline
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass

            w, h = im.size
            longest = max(w, h)
            if longest <= max_dim:
                # Already small enough — keep as-is (still saves through PIL to
                # normalize EXIF & strip metadata, which is fine for game shots).
                resized = im.copy()
                info = f'kept {w}x{h}'
            else:
                ratio = max_dim / longest
                nw, nh = int(w * ratio), int(h * ratio)
                resized = im.resize((nw, nh), Image.LANCZOS)
                info = f'{w}x{h} → {nw}x{nh}'

            ext = dest.suffix.lower()
            fmt_map = {
                '.jpg': 'JPEG', '.jpeg': 'JPEG',
                '.png': 'PNG', '.webp': 'WEBP',
                '.gif': 'GIF', '.bmp': 'BMP',
                '.tif': 'TIFF', '.tiff': 'TIFF',
            }
            fmt = fmt_map.get(ext, 'JPEG')

            save_kwargs = {}
            if fmt in ('JPEG', 'WEBP'):
                if resized.mode in ('RGBA', 'LA', 'P'):
                    # Drop alpha for JPEG; WEBP can take RGBA but quality flag still applies
                    if fmt == 'JPEG':
                        bg = Image.new('RGB', resized.size, (0, 0, 0))
                        if resized.mode == 'P':
                            resized = resized.convert('RGBA')
                        bg.paste(resized, mask=resized.split()[-1]
                                 if resized.mode in ('RGBA', 'LA') else None)
                        resized = bg
                save_kwargs['quality'] = JPEG_QUALITY
                save_kwargs['optimize'] = True
            elif fmt == 'PNG':
                save_kwargs['optimize'] = True
            elif fmt == 'GIF':
                # Animated GIFs: just keep the first frame to keep file size sane
                save_kwargs['save_all'] = False

            resized.save(dest, fmt, **save_kwargs)
            return True, info
    except Exception as e:
        # Fall back to a plain byte copy — at least the file lands in dest
        try:
            shutil.copy2(src, dest)
            return True, f'PIL error ({type(e).__name__}: {e}); raw copy'
        except Exception as e2:
            return False, f'copy failed: {type(e2).__name__}: {e2}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Source folder (recursive scan)')
    ap.add_argument('--session', default='borderlands',
                    help='Session label (default: borderlands)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Scan and report, do not write to DB or copy files')
    ap.add_argument('--no-recurse', action='store_true',
                    help='Only scan the top-level folder, not subfolders')
    ap.add_argument('--no-resize', action='store_true',
                    help='Do not resize images (raw byte copy via shutil.copy2). '
                         'Default: resize so longest side <= --max-dim.')
    ap.add_argument('--max-dim', type=int, default=MAX_DIM,
                    help=f'Maximum longest-side in pixels (default {MAX_DIM}). '
                         f'Smaller images are kept at original size.')
    args = ap.parse_args()
    if not args.no_resize and not PIL_AVAILABLE:
        print('WARN: Pillow not installed — install with `pip install Pillow` '
              'or pass --no-resize to skip resizing.', file=sys.stderr)
        sys.exit(1)

    src_root = Path(args.src)
    if not src_root.exists():
        print(f'ERR: source folder not found: {src_root}', file=sys.stderr); sys.exit(1)
    if not src_root.is_dir():
        print(f'ERR: not a folder: {src_root}', file=sys.stderr); sys.exit(1)

    dest_root = BASE / DEST_DIR_REL
    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        print(f'ERR: {DB_PATH} not found', file=sys.stderr); sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    existing_ids = {r[0] for r in conn.execute('SELECT id FROM borderlands_pool').fetchall()}

    print(f'=== Import Borderlands ===')
    print(f'  source : {src_root}')
    print(f'  dest   : {dest_root}')
    print(f'  session: {args.session!r}')
    print(f'  recurse: {not args.no_recurse}')
    print(f'  resize : {"OFF (raw copy)" if args.no_resize else f"max-dim {args.max_dim}px, JPEG/WEBP q={JPEG_QUALITY}"}')
    print(f'  dry-run: {args.dry_run}')
    print(f'  existing in DB: {len(existing_ids)}\n')

    # Walk
    iterator = src_root.rglob('*') if not args.no_recurse else src_root.glob('*')
    n_img = n_vid = n_other = n_new = n_dup = n_copy_skipped = 0
    rows_to_insert = []
    now = datetime.utcnow().isoformat()

    for p in iterator:
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VID_EXTS:
            n_vid += 1
            continue
        if ext not in IMG_EXTS:
            n_other += 1
            continue
        n_img += 1
        iid = make_id(p)
        if iid in existing_ids:
            n_dup += 1
            continue
        dest_name = safe_dest_name(p, iid)
        rel_dest = str((DEST_DIR_REL / dest_name).as_posix())
        rows_to_insert.append({
            'id':            iid,
            'src':           p,
            'dest_rel':      rel_dest,
            'dest_abs':      dest_root / dest_name,
            'filename':      p.name,
            'original_path': str(p),
        })
        n_new += 1

    print(f'  scanned images:  {n_img}')
    print(f'  scanned videos:  {n_vid}  (SKIPPED)')
    print(f'  scanned other:   {n_other}  (SKIPPED)')
    print(f'  duplicates in DB:{n_dup}')
    print(f'  new to import:   {n_new}')

    if rows_to_insert[:3]:
        print(f'\n  sample (first 3):')
        for r in rows_to_insert[:3]:
            print(f"    {r['id']}  <- {r['src']}")
            print(f"        copy to: {r['dest_rel']}")

    if args.dry_run:
        print('\n[DRY-RUN] No DB writes, no file copies. Re-run without --dry-run to apply.')
        return

    if not rows_to_insert:
        print('\nNothing new to import.')
        return

    cur = conn.cursor()
    print('\n  Copying / resizing files & inserting rows...')
    n_inserted = 0
    bytes_in = bytes_out = 0
    for i, r in enumerate(rows_to_insert, 1):
        try:
            if r['dest_abs'].exists():
                n_copy_skipped += 1
            elif args.no_resize:
                shutil.copy2(r['src'], r['dest_abs'])
            else:
                ok, info = copy_and_resize(r['src'], r['dest_abs'], args.max_dim)
                if not ok:
                    print(f"    ✗ {r['src'].name}: {info}", file=sys.stderr)
                    continue
                if i <= 3 or i % 100 == 0:
                    src_sz = r['src'].stat().st_size
                    dst_sz = r['dest_abs'].stat().st_size
                    print(f"    [{i:>4}] {r['src'].name[:40]:<40} {info:<24} "
                          f"{src_sz//1024:>5} → {dst_sz//1024:<5} KB", flush=True)
                bytes_in  += r['src'].stat().st_size
                bytes_out += r['dest_abs'].stat().st_size
        except Exception as e:
            print(f"    ✗ copy failed {r['src']}: {e}", file=sys.stderr)
            continue
        cur.execute("""
            INSERT INTO borderlands_pool
                (id, local_path, original_path, filename, session, imported_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                local_path    = excluded.local_path,
                original_path = excluded.original_path,
                filename      = excluded.filename,
                session       = excluded.session,
                imported_at   = excluded.imported_at
        """, (r['id'], r['dest_rel'], r['original_path'], r['filename'],
              args.session, now))
        n_inserted += 1
        if i % 100 == 0:
            print(f'    [{i}/{len(rows_to_insert)}] inserted')
            conn.commit()
    conn.commit()
    conn.close()

    print(f'\n  ✓ inserted {n_inserted} rows into borderlands_pool')
    if n_copy_skipped:
        print(f'  ({n_copy_skipped} files already existed in {DEST_DIR_REL}, skipped copy)')
    if bytes_in and not args.no_resize:
        ratio = bytes_out / bytes_in if bytes_in else 0
        print(f'  size: {bytes_in//1024//1024} MB → {bytes_out//1024//1024} MB '
              f'({ratio:.0%} of original)')
    print(f'\nNext:')
    print(f'  python scripts/moderate_borderlands.py --workers 6')


if __name__ == '__main__':
    main()
