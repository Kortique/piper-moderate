#!/usr/bin/env python3
"""
rescore_via_tom.py
------------------
Run our LS + Grafana + K30 images through Tom's pipeline (project a4aa9dbd9c)
to obtain:
  1) Tom's K=30 model score (siglip2_details.underage.lgbm.score) → used as
     `k30tom` model in the gallery for disagreement comparison.
  2) 317-tag taxonomy with :x20 suffixes (siglip2_details.underage.labels.*)
     — this is V11's native input format, so we can score V11 properly on
     items previously stuck in fallback mode.

Output: data/tom_scores.json — list of records:
  {id, source, label, k30_score, k30_blocked, k30_threshold,
   minor, adult, underage_labels, adult_labels, no_underage_labels, done}

Usage:
    python scripts/rescore_via_tom.py                     # all sources, all items
    python scripts/rescore_via_tom.py --source ls         # only LS
    python scripts/rescore_via_tom.py --source grafana --workers 12
    python scripts/rescore_via_tom.py --limit 50          # smoke test
"""
import os, sys, json, time, sqlite3, argparse, shutil, ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()
BASE_DIR   = Path(__file__).resolve().parent.parent
TOKEN      = os.getenv('PIPER_TOKEN', '')
PIPER_BASE = 'https://piper-next.artworks.ai/api'
TOM_PROJECT = 'a4aa9dbd9c'
WORKERS     = 12
OUT_PATH    = BASE_DIR / 'data' / 'tom_scores.json'


def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}


def _open_db():
    src = BASE_DIR / 'gallery.db'
    tmp = Path('/tmp/_tom_rescore.db')
    shutil.copy2(src, tmp)
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def run_one(item_id, src, url, label, max_polls=50, poll_delay=2.5):
    """Submit one image to Tom's pipeline and wait for result."""
    try:
        r = httpx.post(f'{PIPER_BASE}/projects/{TOM_PROJECT}/launch', headers=hdr(),
                       json={'inputs': {'image': url, 'providers': ['siglip2']}}, timeout=30)
        r.raise_for_status()
        run_id = r.json()['_id']
        for _ in range(max_polls):
            time.sleep(poll_delay)
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state', headers=hdr(), timeout=15)
            if rs.status_code != 200: continue
            state = rs.json()
            if state.get('errors'):
                return {'id': item_id, 'source': src, 'label': label, 'error': str(state['errors'][0])[:120]}
            outputs = state.get('outputs') or {}
            if 'siglip2_details' in outputs:
                under = (outputs.get('siglip2_details') or {}).get('underage', {})
                labels = under.get('labels', {})
                lgbm = under.get('lgbm') or {}
                return {
                    'id':                item_id,
                    'source':            src,
                    'label':             label,
                    'k30_score':         lgbm.get('score'),
                    'k30_blocked':       lgbm.get('blocked'),
                    'k30_threshold':     lgbm.get('threshold'),
                    'k30_n_features':    lgbm.get('n_features'),
                    'minor':             under.get('minor'),
                    'adult':             under.get('adult'),
                    'confidence':        under.get('confidence'),
                    'underage_labels':   labels.get('underage', {}),
                    'adult_labels':      labels.get('adult', {}),
                    'no_underage_labels':labels.get('no_underage', {}),
                    'done': True,
                }
        return {'id': item_id, 'source': src, 'label': label, 'error': 'timeout'}
    except Exception as e:
        return {'id': item_id, 'source': src, 'label': label, 'error': str(e)[:120]}


def load_items(source: str):
    """Return list of (id, source, label, url) tuples to score."""
    conn = _open_db()
    items = []

    if source in ('all', 'grafana'):
        for r in conn.execute("""SELECT id, label, thumb_url FROM grafana_pool
                                  WHERE thumb_url IS NOT NULL
                                    AND (deleted IS NULL OR deleted=0)"""):
            items.append((r['id'], 'grafana', r['label'], r['thumb_url']))

    if source in ('all', 'k30'):
        for r in conn.execute("""SELECT id, label, thumb_url FROM k30_pool
                                  WHERE thumb_url IS NOT NULL
                                    AND (deleted IS NULL OR deleted=0)"""):
            items.append((r['id'], 'k30', r['label'], r['thumb_url']))

    if source in ('all', 'ls'):
        # LS items: prefer ls_images table; fall back to qwen3_age_results.json
        ls_rows = []
        try:
            for r in conn.execute("""SELECT task_id, media, age_from
                                      FROM ls_images WHERE media IS NOT NULL"""):
                if r['media']:
                    af = r['age_from']
                    lbl = ('child' if af is not None and af <= 14
                           else 'teen' if af is not None and af <= 17
                           else 'adult' if af is not None else None)
                    ls_rows.append((f"ls_{r['task_id']}", 'labelstudio', lbl, r['media']))
        except Exception:
            pass
        if not ls_rows:
            try:
                qd = json.loads((BASE_DIR / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8'))
                for v in qd.values():
                    tid = v.get('task_id')
                    if tid is None: continue
                    media = v.get('media')
                    if not media: continue
                    af = (v.get('age') or {}).get('ageFrom')
                    lbl = ('child' if af is not None and af <= 14
                           else 'teen' if af is not None and af <= 17
                           else 'adult' if af is not None else None)
                    ls_rows.append((f"ls_{tid}", 'labelstudio', lbl, media))
            except Exception as e:
                print(f'WARN: LS load fallback failed: {e}', file=sys.stderr)
        items.extend(ls_rows)

    conn.close()
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='all', choices=['all','ls','grafana','k30'])
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)

    items = load_items(args.source)
    if args.limit:
        items = items[:args.limit]
    print(f'Total items for source={args.source}: {len(items)}', flush=True)

    # Resume from existing file
    existing = {}
    if OUT_PATH.exists():
        try:
            for r in json.loads(OUT_PATH.read_text()):
                if r.get('done'):
                    existing[r['id']] = r
        except Exception:
            pass

    todo = [it for it in items if it[0] not in existing]
    print(f'  already done: {len(existing)}', flush=True)
    print(f'  todo:         {len(todo)}', flush=True)
    print(f'  workers:      {args.workers}', flush=True)
    print(f'  project:      {TOM_PROJECT}\n', flush=True)

    results = list(existing.values())

    if not todo:
        tmp_p = OUT_PATH.with_suffix(OUT_PATH.suffix + '.tmp'); tmp_p.write_text(json.dumps(results, indent=2, ensure_ascii=False)); os.replace(tmp_p, OUT_PATH)
        print('Nothing to do, file already complete.')
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, iid, src, url, lbl): (iid, src) for (iid, src, lbl, url) in todo}
        n_total = len(existing); n_done = 0; n_err = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n_total += 1
            if res.get('done'):
                n_done += 1; status = '✓'
            else:
                n_err += 1; status = f'✗({res.get("error","?")[:30]})'
            print(f'[{n_total:5d}/{len(items)}] {status} {res["id"][:24]:<26} ({res.get("source")[:6]:<7} {res.get("label")})', flush=True)
            if n_total % 25 == 0:
                tmp_p = OUT_PATH.with_suffix(OUT_PATH.suffix + '.tmp'); tmp_p.write_text(json.dumps(results, indent=2, ensure_ascii=False)); os.replace(tmp_p, OUT_PATH)

    tmp_p = OUT_PATH.with_suffix(OUT_PATH.suffix + '.tmp'); tmp_p.write_text(json.dumps(results, indent=2, ensure_ascii=False)); os.replace(tmp_p, OUT_PATH)
    done = sum(1 for r in results if r.get('done'))
    err  = sum(1 for r in results if not r.get('done'))
    print(f'\n=== Done ===', flush=True)
    print(f'  total: {len(results)}  done: {done}  errors: {err}', flush=True)
    print(f'  saved to: {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
