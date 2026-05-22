#!/usr/bin/env python3
"""
Run Tom's K=30 LS dataset through d2911d10bb (V8pas80-v2).
JSONL format — append-only, robust to timeout/sparse-fs.

Resumable: state in data/k30_ours.jsonl, one record per line.
"""
import os, json, time, sys, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}
API = 'https://piper-next.artworks.ai/api'
PROJECT = 'd2911d10bb'

EXPORT = BASE / 'data' / 'k30_ls_export_full.json'
OUT = BASE / 'data' / 'k30_ours.jsonl'


def launch(image_url):
    try:
        with httpx.Client(timeout=120, follow_redirects=False) as c:
            r = c.post(f'{API}/projects/{PROJECT}/launch', headers=HDR,
                       json={'inputs': {'image': image_url, 'providers': ['siglip2']}})
            if r.status_code != 200:
                return {'error': f'launch HTTP {r.status_code}: {r.text[:120]}'}
            run_id = r.json()['_id']
            for _ in range(40):
                time.sleep(2)
                rs = c.get(f'{API}/launches/{run_id}/state', headers=HDR)
                if rs.status_code != 200: continue
                state = rs.json()
                outputs = state.get('outputs') or {}
                errors = state.get('errors') or []
                if errors:
                    return {'error': f'launch_err: {str(errors)[:200]}'}
                if 'siglip2_details' in outputs:
                    det = outputs.get('siglip2_details') or {}
                    und = det.get('underage') or {}
                    lgbm = und.get('lgbm') or {}
                    return {
                        'lgbm_score': lgbm.get('score'),
                        'lgbm_blocked': lgbm.get('blocked'),
                        'minor': und.get('minor'),
                        'adult': und.get('adult'),
                        'confidence': und.get('confidence'),
                    }
            return {'error': 'timeout'}
    except Exception as e:
        return {'error': str(e)[:120]}


def process_one(item):
    out = launch(item['media'])
    return {**item, 'ours': out}


def load_existing():
    """Read JSONL, skip broken lines, dedupe by gid (keep last)."""
    seen = {}
    if not OUT.exists(): return seen
    with open(OUT, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                if r.get('ours') and 'error' not in r['ours']:
                    seen[r['gid']] = r
            except Exception:
                continue
    return seen


def append_record(rec):
    """Append single record with fsync — atomic on POSIX up to write_size."""
    line = json.dumps(rec, ensure_ascii=False) + '\n'
    with open(OUT, 'a') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=80)
    ap.add_argument('--workers', type=int, default=15)
    args = ap.parse_args()

    full = json.loads(EXPORT.read_text())
    cands = []
    for t in full:
        d = t.get('data') or {}
        if not d.get('media'): continue
        ext_id = d.get('external_id', '')
        gid = ext_id.removeprefix('g/') if isinstance(ext_id, str) else ''
        cands.append({
            'gid': gid,
            'task_id': t.get('id'),
            'media': d['media'],
            'qwen_min_age': (d.get('qwen3') or {}).get('min_age'),
            'qwen_max_age': (d.get('qwen3') or {}).get('max_age'),
            'qwen_underage': (d.get('qwen3') or {}).get('underage'),
            'k30_score': (d.get('models') or {}).get('k30_score'),
            'k30_blocked': (d.get('models') or {}).get('k30_blocked'),
            'k25_score': (d.get('models') or {}).get('k25_score'),
            'prod_score': (d.get('models') or {}).get('prod_score'),
            'prod_blocked': (d.get('models') or {}).get('prod_blocked'),
        })

    existing = load_existing()
    remaining = [c for c in cands if c['gid'] not in existing]
    if not remaining:
        print(f'Nothing to do. Done: {len(existing)}/{len(cands)}', flush=True)
        return
    chunk = remaining[:args.chunk]
    print(f'cands={len(cands)} done={len(existing)} remaining={len(remaining)} chunk={len(chunk)} workers={args.workers}', flush=True)

    t0 = time.time()
    completed = 0
    n_ok = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, c): c for c in chunk}
        for fut in as_completed(futs):
            rec = fut.result()
            completed += 1
            o = rec.get('ours', {})
            if 'error' in o:
                n_err += 1
                tag = 'ERR ' + str(o['error'])[:50]
            else:
                n_ok += 1
                tag = 'sc={:.3f} b={}'.format(o.get('lgbm_score', 0), int(bool(o.get('lgbm_blocked'))))
            if completed % 5 == 0 or completed == len(chunk):
                print('  {}/{} t={:.0f}s {} -> {}'.format(completed, len(chunk), time.time()-t0, str(rec['gid'])[:18], tag), flush=True)
            # Append-only: durable per-record
            append_record(rec)

    print('Session: ok={} err={}  total={}/{}'.format(n_ok, n_err, len(existing)+n_ok, len(cands)), flush=True)


if __name__ == '__main__':
    main()
