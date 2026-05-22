#!/usr/bin/env python3
"""Simple resumable rescorer for ce79f7e299. Saves after every item."""
import json, os, sys, time, argparse
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


def load_queue():
    m = json.loads(META.read_text())
    out = []
    for x in m['ls_to_rescore']:
        out.append({'id': 'ls_' + str(x['task_id']), 'task_id': x['task_id'],
                    'label': x['category'], 'media': x['media'], 'kind': 'ls'})
    for x in m['graf_eval']:
        out.append({'id': x['id'], 'label': x['label'], 'media': x['thumb_url'],
                    'kind': 'graf', 'export_batch': x['export_batch']})
    return out


def load_existing():
    if OUT.exists():
        try:
            d = json.loads(OUT.read_text())
            return d, {r['id'] for r in d if r.get('done')}
        except Exception:
            return [], set()
    return [], set()


def process_one(item):
    client = httpx.Client()
    try:
        payload = {'inputs': {'image': item['media'],
                              'providers_e0': ['siglip2'],
                              'providers': ['siglip2']}}
        r = client.post(API + '/projects/' + PROJECT + '/launch',
                        headers=HDR, content=json.dumps(payload).encode(), timeout=15)
        r.raise_for_status()
        run_id = r.json()['_id']
    except Exception as e:
        client.close()
        return dict(item, done=False, error='launch:' + str(e))

    t0 = time.time()
    result = None
    while time.time() - t0 < 60:
        try:
            s = client.get(API + '/launches/' + run_id + '/state', headers=HDR, timeout=8).json()
            outs = s.get('outputs') or {}
            det = outs.get('siglip2_details')
            if det:
                under = det.get('underage') or {}
                labels = under.get('labels') or {}
                result = {'id': item['id'], 'task_id': item.get('task_id'),
                          'label': item['label'], 'media': item['media'], 'kind': item['kind'],
                          'export_batch': item.get('export_batch'),
                          'minor': under.get('minor', 0), 'adult': under.get('adult', 0),
                          'underage_labels': labels.get('underage') or {},
                          'adult_labels': labels.get('adult') or {},
                          'no_underage_labels': labels.get('no_underage') or {},
                          'done': True}
                break
            if s.get('status') in ('failed', 'error', 'cancelled'):
                result = dict(item, done=False, error='failed')
                break
        except Exception:
            pass
        time.sleep(1.5)
    client.close()
    if result is None:
        result = dict(item, done=False, error='timeout')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=12)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    queue = load_queue()
    data, done_ids = load_existing()
    remaining = [x for x in queue if x['id'] not in done_ids]
    if not remaining:
        print('Nothing to do. Total done: ' + str(len(done_ids)))
        return
    chunk = remaining[:args.chunk]
    print('queue=' + str(len(queue)) + ' done=' + str(len(done_ids)) +
          ' remaining=' + str(len(remaining)) + ' chunk=' + str(len(chunk)) +
          ' workers=' + str(args.workers), flush=True)
    out = {r['id']: r for r in data}
    t0 = time.time()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, it): it for it in chunk}
        for fut in as_completed(futs):
            rec = fut.result()
            out[rec['id']] = rec
            completed += 1
            if not rec.get('done'):
                failed += 1
            tag = 'OK' if rec.get('done') else ('ERR ' + str(rec.get('error', '')))
            line = '  ' + str(completed) + '/' + str(len(chunk)) + ' t=' + ('%.1f' % (time.time()-t0)) + 's id=' + rec['id'][:18] + ' ' + tag
            print(line, flush=True)
            OUT.write_text(json.dumps(list(out.values()), ensure_ascii=False, indent=2))
    msg = 'Chunk done: ' + str(completed-failed) + '/' + str(completed) + ' in ' + ('%.1f' % (time.time()-t0)) + 's failed=' + str(failed)
    print(msg, flush=True)


if __name__ == '__main__':
    main()
