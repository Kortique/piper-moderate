#!/usr/bin/env python3
"""Benchmark V6 / V7 / V7nx / V10 / V10nx on 2803 eval set."""
import json, sqlite3, struct, os, tempfile, ast
from pathlib import Path
import numpy as np
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'


def open_db():
    raw = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', raw, 28, len(raw) // 4096)
    fd, tmp = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd)
    Path(tmp).write_bytes(bytes(raw))
    conn = sqlite3.connect(tmp); conn.row_factory = sqlite3.Row
    return conn, tmp


def sanitize(n):
    return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')


def build_X(items, feats):
    idx = {f:i for i,f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    for i, it in enumerate(items):
        for k,v in (it.get('underage_labels') or {}).items():
            fk = sanitize(k)
            if fk in idx: X[i, idx[fk]] = float(v)
        for k,v in (it.get('adult_labels') or {}).items():
            fk = 'adult__' + sanitize(k)
            if fk in idx: X[i, idx[fk]] = float(v)
        for k,v in (it.get('no_underage_labels') or {}).items():
            fk = 'no_underage__' + sanitize(k)
            if fk in idx: X[i, idx[fk]] = float(v)
    return X


def load_old_items():
    items = []
    raw = (BASE / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    qd = json.loads(raw)
    for v in qd.values():
        cat = v.get('category')
        if cat not in ('child','teen','adult'): continue
        sd = v.get('siglip2_details')
        if isinstance(sd, str):
            try: sd = ast.literal_eval(sd)
            except: sd = None
        det = (sd or {}).get('underage') or {}
        if not det: continue
        labels = det.get('labels') or {}
        items.append({'id': 'ls_'+str(v['task_id']), 'label': cat, 'source': 'ls',
                      'underage_labels': labels.get('underage') or {},
                      'adult_labels': labels.get('adult') or {}})
    conn, tmp = open_db()
    rows = conn.execute("""SELECT id,label,piper_result FROM grafana_pool
                           WHERE (deleted IS NULL OR deleted=0) AND label IN ('child','teen','adult')
                           AND piper_result IS NOT NULL AND export_batch != '2026-05-20 UTC'""").fetchall()
    conn.close(); os.unlink(tmp)
    for r in rows:
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage') or {}
            if not det: continue
            labels = det.get('labels') or {}
            items.append({'id': r['id'], 'label': r['label'], 'source': 'graf17',
                          'underage_labels': labels.get('underage') or {},
                          'adult_labels': labels.get('adult') or {}})
        except: pass
    return items


def load_new_items():
    items, seen = [], set()
    def add(rec, src):
        if rec['id'] in seen: return
        seen.add(rec['id'])
        items.append({'id': rec['id'], 'label': rec['label'], 'source': src,
                      'underage_labels': rec.get('underage_labels') or {},
                      'adult_labels': rec.get('adult_labels') or {},
                      'no_underage_labels': rec.get('no_underage_labels') or {}})
    for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        add({**r, 'id': 'ls_'+str(r['task_id'])}, 'ls')
    for r in json.loads((DATA / 'eval616_rescored.json').read_text()):
        if not r.get('done') or r.get('label') not in ('child','teen','adult'): continue
        if r.get('export_batch') == '2026-05-20 UTC': continue
        add(r, 'ls' if r.get('kind') == 'ls' else 'graf17')
    return items


def metrics(items, preds, thr):
    by = {'child': [], 'teen': [], 'adult': []}
    for it, p in zip(items, preds):
        if it['label'] in by:
            by[it['label']].append(p >= thr)
    nc, nt, na = len(by['child']), len(by['teen']), len(by['adult'])
    cr = 100 * sum(by['child']) / nc if nc else 0
    tr = 100 * sum(by['teen']) / nt if nt else 0
    fpr = 100 * sum(by['adult']) / na if na else 0
    return cr, tr, fpr, nc, nt, na


def find_best(items, preds, thrs):
    best = None
    for thr in thrs:
        cr, tr, fpr, _, _, _ = metrics(items, preds, thr)
        meets = cr >= 95 and tr >= 80 and fpr <= 20
        violation = max(0, 95-cr) + max(0, 80-tr) + max(0, fpr-20)
        cand = (meets, -violation, cr, tr, -fpr, thr)
        if best is None or cand > best: best = cand
    return {'thr': best[-1], 'cr': best[2], 'tr': best[3], 'fpr': -best[4], 'met': best[0]}


def to_py(o):
    if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_py(x) for x in o]
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o


def main():
    print('Loading...', flush=True)
    old = load_old_items()
    new = load_new_items()
    print(f'  old (V6/V7/V7nx): {len(old)}', flush=True)
    print(f'  new (V10/V10nx): {len(new)}', flush=True)

    models = [
        ('V6',    'lgbm_underage_v6.txt',    'lgbm_v6_features.json',    old),
        ('V7',    'lgbm_underage_v7.txt',    'lgbm_v7_features.json',    old),
        ('V7nx',  'lgbm_underage_v7nx.txt',  'lgbm_v7nx_features.json',  old),
        ('V10',   'lgbm_underage_v10.txt',   'lgbm_v10_features.json',   new),
        ('V10nx', 'lgbm_underage_v10nx.txt', 'lgbm_v10nx_features.json', new),
    ]

    thrs = [round(0.10 + 0.05 * i, 2) for i in range(13)]   # 0.10..0.70

    summary = {'thrs': thrs, 'sweep': {}, 'best': {}}
    for name, mp, fp, src in models:
        print(f'Scoring {name}...', flush=True)
        feats = json.load(open(DATA / fp))
        booster = lgb.Booster(model_file=str(DATA / mp))
        preds = booster.predict(build_X(src, feats))
        rows = []
        for thr in thrs:
            rec = {'thr': thr}
            rec['overall'] = dict(zip(['child','teen','fpr','n_c','n_t','n_a'], metrics(src, preds, thr)))
            for source in ('ls', 'graf17'):
                sub_i = [x for x in src if x['source'] == source]
                sub_p = [p for x, p in zip(src, preds) if x['source'] == source]
                if sub_i:
                    rec[source] = dict(zip(['child','teen','fpr','n_c','n_t','n_a'], metrics(sub_i, sub_p, thr)))
            rows.append(rec)
        summary['sweep'][name] = rows
        summary['best'][name] = {
            'overall': find_best(src, preds, thrs),
            'ls':     find_best([x for x in src if x['source']=='ls'],
                                [p for x,p in zip(src,preds) if x['source']=='ls'], thrs),
        }

    (DATA / 'benchmark_all_summary.json').write_text(json.dumps(to_py(summary), indent=2))
    print(f'Saved: data/benchmark_all_summary.json', flush=True)

    # Print compact summaries
    print('\n=== BEST THR per model — LS only (n=2212) ===')
    print(f'{"Model":>6} {"THR":>5} {"Child%":>8} {"Teen%":>7} {"FPR%":>6} {"Met?":>5}')
    for name, _, _, _ in models:
        b = summary['best'][name]['ls']
        print(f'  {name:>6} {b["thr"]:>5.2f} {b["cr"]:>7.1f} {b["tr"]:>6.1f} {b["fpr"]:>5.1f} {"YES" if b["met"] else "no":>5}')

    print('\n=== BEST THR per model — overall (n=2803) ===')
    for name, _, _, _ in models:
        b = summary['best'][name]['overall']
        print(f'  {name:>6} {b["thr"]:>5.2f} {b["cr"]:>7.1f} {b["tr"]:>6.1f} {b["fpr"]:>5.1f} {"YES" if b["met"] else "no":>5}')

    print('\n=== LS-only sweep — Child / Teen / FPR for each model at every THR ===')
    h = f'{"THR":>5}'
    for name, _, _, _ in models:
        h += f' | {name:>6}'
    h += '   (child / teen / fpr per model)'
    print(h)
    for i, thr in enumerate(thrs):
        row = f'{thr:>5.2f}'
        for name, _, _, _ in models:
            ls = summary['sweep'][name][i].get('ls', {})
            row += f' | {ls.get("child",0):4.1f}/{ls.get("teen",0):4.1f}/{ls.get("fpr",0):4.1f}'
        print(row)


if __name__ == '__main__':
    main()
