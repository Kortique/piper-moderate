#!/usr/bin/env python3
"""
rescore_eval616.py
------------------
Rescore 616 items (25 LS + 591 Grafana 17-May sessions) through ce79f7e299
to produce V10-compatible scoring data for the eval benchmark.

- Reads queue from data/eval_set_meta.json (ls_to_rescore + graf_eval).
- Submits launches in parallel (5 workers), polls for results.
- Resumable: skips items already marked done in data/eval616_rescored.json.
- Output schema matches ls_holdout_rescored.json:
    {id, task_id?, label, media, minor, adult, underage_labels, adult_labels, no_underage_labels, done}
"""
import json, os, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')

TOKEN = os.getenv('PIPER_TOKEN')
PROJECT = 'ce79f7e299'
API = 'https://piper-next.artworks.ai/api'
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}

OUT = BASE / 'data' / 'eval616_rescored.json'
META = BASE / 'data' / 'eval_set_meta.json'

WORKERS = 10
POLL_INTERVAL = 2.0
POLL_TIMEOUT  = 90    # sec per item


def load_queue():
    meta = json.loads(META.read_text())
    items = []
    for x in meta['ls_to_rescore']:
        items.append({
            'id': f'ls_{x["task_id"]}',
            'task_id': x['task_id'],
            'label': x['category'],
            'media': x['media'],
            'kind': 'ls',
        })
    for x in meta['graf_eval']:
        items.append({
            'id': x['id'],
            'label': x['label'],
            'media': x['thumb_url'],
            'kind': 'graf',
            'export_batch': x['export_batch'],
        })
    return items


def load_existing():
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text())
            done_ids = {r['id'] for r in data if r.get('done')}
            return data, done_ids
        except Exception:
            pass
    return [], set()


def launch(client, image_url):
    payload = {
        'inputs': {
            'image': image_url,
            'providers_e0': ['siglip2'],   # only siglip2 — фиксирует фокус на тегах
            'providers': ['siglip2'],
        }
    }
    r = client.post(f'{API}/projects/{PROJECT}/launch',
                    headers=HDR, content=json.dumps(payload).encode(), timeout=20)
    r.raise_for_status()
    return r.json()['_id']


def poll(client, run_id):
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT:
        r = client.get(f'{API}/launches/{run_id}/state', headers=HDR, timeout=10)
        if r.status_code == 200:
            st = r.json()
            outputs = st.get('outputs') or {}
            details = outputs.get('siglip2_details')
            if details:
                return details
            status = st.get('status') or st.get('state')
            if status in ('failed', 'error', 'cancelled'):
                return None
        time.sleep(POLL_INTERVAL)
    return None


def extract_record(details, item):
    under = (details.get('underage') or {})
    labels = under.get('labels') or {}
    return {
        'id': item['id'],
        'task_id': item.get('task_id'),
        'label': item['label'],
        'media': item['media'],
        'kind': item['kind'],
        'export_batch': item.get('export_batch'),
        'minor': under.get('minor', 0),
        'adult': under.get('adult', 0),
        'underage_labels':    labels.get('underage') or {},
        'adult_labels':       labels.get('adult') or {},
        'no_underage_labels': labels.get('no_underage') or {},
        'done': True,
    }


def process_one(item):
    with httpx.Client() as client:
        try:
            run_id = launch(client, item['media'])
        except Exception as e:
            return {'id': item['id'], 'task_id': item.get('task_id'),
                    'label': item['label'], 'media': item['media'], 'kind': item['kind'],
                    'export_batch': item.get('export_batch'),
                    'done': False, 'error': f'launch: {e}'}
        details = poll(client, run_id)
        if details is None:
            return {'id': item['id'], 'task_id': item.get('task_id'),
                    'label': item['label'], 'media': item['media'], 'kind': item['kind'],
                    'export_batch': item.get('export_batch'),
                    'done': False, 'error': f'timeout/failed run {run_id}'}
        return extract_record(details, item)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seconds', type=int, default=0, help='Stop after this many seconds (0 = unlimited)')
    parser.add_argument('--max-items',   type=int, default=0, help='Stop after this many items (0 = unlimited)')
    args = parser.parse_args()

    queue = load_queue()
    data, done_ids = load_existing()
    remaining = [x for x in queue if x['id'] not in done_ids]
    print(f'Queue: {len(queue)} | done: {len(done_ids)} | remaining: {len(remaining)}')

    if not remaining:
        print('Nothing to do.')
        return

    # apply limits
    budget = remaining
    if args.max_items > 0:
        budget = budget[:args.max_items]

    out = {r['id']: r for r in data}
    save_each = 5

    t0 = time.time()
    completed = 0
    failed = 0
    stopped_early = False
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_one, it): it for it in budget}
        for fut in as_completed(futs):
            if args.max_seconds and (time.time() - t0) > args.max_seconds:
                # cancel pending and break
                for f2 in futs:
                    if not f2.done():
                        f2.cancel()
                stopped_early = True
                break
            rec = fut.result()
            out[rec['id']] = rec
            completed += 1
            if not rec.get('done'):
                failed += 1
                err = rec.get('error', '?')
                rid = rec['id'][:24]
                print('  X ' + rid + '.. ' + str(err))
            if completed % save_each == 0 or completed == len(budget):
                OUT.write_text(json.dumps(list(out.values()), ensure_ascii=False, indent=2))
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1)
                ok = sum(1 for r in out.values() if r.get('done'))
                print(f'  {completed}/{len(budget)} done '
                      f'(rate={rate:.2f}/sec, ok_total={ok}, failed={failed}, elapsed={elapsed:.0f}s)',
                      flush=True)
    # final save
    OUT.write_text(json.dumps(list(out.values()), ensure_ascii=False, indent=2))
    print(f'\n{"Stopped early." if stopped_early else "Final:"} {completed} processed, {failed} failed in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
