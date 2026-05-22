#!/usr/bin/env python3
"""
rescore_ls_holdout.py
---------------------
Прогоняет все 2216 LS holdout items через обновлённый Piper pipeline
(ce79f7e299 с 867 тегами включая no_underage) и сохраняет новые
siglip2_details в data/ls_holdout_rescored.json.

Резюме: пропускает уже обработанные записи (по task_id).

Usage:
    python scripts/rescore_ls_holdout.py [--project ID] [--workers N]
"""
import os, sys, json, time, argparse, ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
TOKEN = os.getenv('PIPER_TOKEN', '')
PIPER_BASE = 'https://piper-next.artworks.ai/api'
DEFAULT_PROJECT = 'ce79f7e299'
WORKERS = 4
OUT_PATH = BASE_DIR / 'data' / 'ls_holdout_rescored.json'


def hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}


def run_one(task_id, url, label, project_id):
    try:
        r = httpx.post(f'{PIPER_BASE}/projects/{project_id}/launch', headers=hdr(),
                       json={'inputs': {'image': url, 'providers': ['siglip2']}}, timeout=30)
        r.raise_for_status()
        run_id = r.json()['_id']

        for _ in range(60):
            time.sleep(3)
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state', headers=hdr(), timeout=15)
            if rs.status_code != 200:
                continue
            state = rs.json()
            errors = state.get('errors') or []
            outputs = state.get('outputs') or {}

            if errors:
                return {'task_id': task_id, 'label': label, 'error': str(errors[0])[:120]}

            if 'siglip2_details' in outputs:
                det = (outputs.get('siglip2_details') or {}).get('underage', {})
                labels = det.get('labels', {})
                return {
                    'task_id': task_id,
                    'label': label,
                    'minor': det.get('minor', 0),
                    'adult': det.get('adult', 0),
                    'lgbm_score': (det.get('lgbm') or {}).get('score', 0),
                    'underage_labels':    labels.get('underage', {}),
                    'adult_labels':       labels.get('adult', {}),
                    'no_underage_labels': labels.get('no_underage', {}),
                    'done': True
                }

        return {'task_id': task_id, 'label': label, 'error': 'timeout'}
    except Exception as e:
        return {'task_id': task_id, 'label': label, 'error': str(e)[:120]}


def load_source():
    """Load LS holdout from qwen3_age_results.json."""
    raw = (BASE_DIR / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    data = json.loads(raw)
    items = []
    for v in data.values():
        lbl = v.get('category')
        if lbl not in ('child', 'teen', 'adult'):
            continue
        media = v.get('media')
        if not media:
            continue
        items.append({
            'task_id': v['task_id'],
            'label': lbl,
            'url': media,
        })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=DEFAULT_PROJECT, help='Piper project ID')
    parser.add_argument('--workers', type=int, default=WORKERS)
    args = parser.parse_args()

    if not TOKEN:
        print('ERROR: PIPER_TOKEN not set', file=sys.stderr)
        sys.exit(1)

    items = load_source()
    print(f'LS holdout items: {len(items)}')
    label_counts = {}
    for it in items:
        label_counts[it['label']] = label_counts.get(it['label'], 0) + 1
    print(f'  child={label_counts.get("child",0)}, teen={label_counts.get("teen",0)}, adult={label_counts.get("adult",0)}')

    # Resume
    existing = {}
    if OUT_PATH.exists():
        for r in json.loads(OUT_PATH.read_text()):
            if r.get('done'):
                existing[r['task_id']] = r
    print(f'Resume: done={len(existing)}, todo={len(items) - len(existing)}')

    todo = [it for it in items if it['task_id'] not in existing]
    results = list(existing.values())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, it['task_id'], it['url'], it['label'], args.project): it
                   for it in todo}
        n = len(existing)
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            n += 1
            if res.get('done'):
                nu = len(res.get('no_underage_labels', {}))
                status = f'✓ no_underage={nu}'
            else:
                status = f'✗ {res.get("error","?")[:30]}'
            print(f'[{n:4d}/{len(items)}] {status}  tid={res["task_id"]} ({res["label"]})')

            if n % 20 == 0:
                OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    done = sum(1 for r in results if r.get('done'))
    errors = sum(1 for r in results if not r.get('done'))
    nu_counts = [len(r.get('no_underage_labels', {})) for r in results if r.get('done')]
    print(f'\n{"="*50}')
    print(f'Completed: {done}/{len(items)}  errors: {errors}')
    if nu_counts:
        import statistics
        with_nu = sum(1 for c in nu_counts if c > 0)
        print(f'no_underage coverage: {with_nu}/{done} ({100*with_nu/done:.1f}%)')
        print(f'avg no_underage tags: {statistics.mean(nu_counts):.1f}')
    print(f'Saved → {OUT_PATH}')


if __name__ == '__main__':
    main()
