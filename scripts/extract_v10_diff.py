#!/usr/bin/env python3
"""Extract V10-introduced FPs (71) and V10-missed teen FNs (22) with metadata."""
import json, sys, ast, struct, os, tempfile, sqlite3
from pathlib import Path
import numpy as np
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

def sanitize(n): return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')

def build_X(items, feats):
    idx = {f:i for i,f in enumerate(feats)}
    X = np.zeros((len(items), len(feats)), dtype=np.float32)
    for i, it in enumerate(items):
        for k,v in (it.get('underage_labels') or {}).items():
            fk=sanitize(k)
            if fk in idx: X[i,idx[fk]]=float(v)
        for k,v in (it.get('adult_labels') or {}).items():
            fk='adult__'+sanitize(k)
            if fk in idx: X[i,idx[fk]]=float(v)
        for k,v in (it.get('no_underage_labels') or {}).items():
            fk='no_underage__'+sanitize(k)
            if fk in idx: X[i,idx[fk]]=float(v)
    return X

# Load V7 items (old siglip scores) — qwen3_age_results.json
v7_items = {}
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
    iid = 'ls_' + str(v['task_id'])
    v7_items[iid] = {
        'id': iid, 'label': cat,
        'media': v.get('media'),
        'underage_labels': labels.get('underage') or {},
        'adult_labels':    labels.get('adult') or {},
    }

# Load V10 items (rescored)
v10_items = {}
for r in json.loads((DATA / 'ls_holdout_rescored.json').read_text()):
    if not r.get('done'): continue
    if r.get('label') not in ('child','teen','adult'): continue
    iid = 'ls_' + str(r['task_id'])
    v10_items[iid] = {
        'id': iid, 'label': r['label'],
        'media': r.get('media'),
        'underage_labels':    r.get('underage_labels') or {},
        'adult_labels':       r.get('adult_labels') or {},
        'no_underage_labels': r.get('no_underage_labels') or {},
    }

common = set(v7_items) & set(v10_items)
print(f'common LS items: {len(common)}', flush=True)

paired_ids = sorted(common)
v7_in = [v7_items[i] for i in paired_ids]
v10_in = [v10_items[i] for i in paired_ids]

v7_feats = json.load(open(DATA / 'lgbm_v7_features.json'))
v10_feats = json.load(open(DATA / 'lgbm_v10_features.json'))
m7 = lgb.Booster(model_file=str(DATA / 'lgbm_underage_v7.txt'))
m10 = lgb.Booster(model_file=str(DATA / 'lgbm_underage_v10.txt'))

p7 = m7.predict(build_X(v7_in, v7_feats))
p10 = m10.predict(build_X(v10_in, v10_feats))

T7, T10 = 0.35, 0.25

def top_tags(labels_dict, n=8):
    return sorted([(k, float(v)) for k, v in (labels_dict or {}).items()], key=lambda x: -x[1])[:n]

fp_v10_only = []   # adult, V10 blocks, V7 passes — new FPs by V10
fn_v10_only = []   # teen, V10 passes, V7 blocks — V10 missed teens

for iid, item_v7, item_v10, s7, s10 in zip(paired_ids, v7_in, v10_in, p7, p10):
    b7 = s7 >= T7
    b10 = s10 >= T10
    lbl = item_v7['label']
    if lbl == 'adult' and b10 and not b7:
        fp_v10_only.append({
            'id': iid, 'label': lbl, 'media': item_v7['media'],
            'v7_score': float(s7), 'v10_score': float(s10),
            'v7_blocks': bool(b7), 'v10_blocks': bool(b10),
            'top_underage_old':  top_tags(item_v7['underage_labels']),
            'top_underage_new':  top_tags(item_v10['underage_labels']),
            'top_adult_new':     top_tags(item_v10['adult_labels']),
            'top_no_underage':   top_tags(item_v10['no_underage_labels']),
        })
    if lbl == 'teen' and b7 and not b10:
        fn_v10_only.append({
            'id': iid, 'label': lbl, 'media': item_v7['media'],
            'v7_score': float(s7), 'v10_score': float(s10),
            'top_underage_old':  top_tags(item_v7['underage_labels']),
            'top_underage_new':  top_tags(item_v10['underage_labels']),
            'top_adult_new':     top_tags(item_v10['adult_labels']),
            'top_no_underage':   top_tags(item_v10['no_underage_labels']),
        })

print(f'V10-introduced adult FPs: {len(fp_v10_only)}', flush=True)
print(f'V10-missed teen FNs:      {len(fn_v10_only)}', flush=True)

result = {'v10_fps_adults': fp_v10_only, 'v10_fns_teens': fn_v10_only}
(DATA / 'v10_diff_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f'Saved: data/v10_diff_analysis.json', flush=True)

# Print aggregate of no_underage tags that fire on FPs (which "fail" to suppress correctly)
print('\n=== AGGREGATE: top no_underage_* tags fired on the 71 NEW FP adults ===')
nu_score_sum = {}
nu_count = {}
for it in fp_v10_only:
    for k, v in it['top_no_underage']:
        nu_score_sum[k] = nu_score_sum.get(k, 0) + v
        nu_count[k] = nu_count.get(k, 0) + 1
top_nu = sorted(nu_count.items(), key=lambda x: -x[1])[:15]
print(f'{"tag":<45} {"count":>6} {"sum_score":>10}')
for k, c in top_nu:
    print(f'  {k[:43]:<45} {c:>6} {nu_score_sum[k]:>10.2f}')

print('\n=== AGGREGATE: top underage_* tags fired on the 22 V10-missed teen FNs ===')
u_score_sum = {}
u_count = {}
for it in fn_v10_only:
    for k, v in it['top_underage_new']:
        u_score_sum[k] = u_score_sum.get(k, 0) + v
        u_count[k] = u_count.get(k, 0) + 1
top_u = sorted(u_count.items(), key=lambda x: -x[1])[:15]
print(f'{"tag":<45} {"count":>6} {"sum_score":>10}')
for k, c in top_u:
    print(f'  {k[:43]:<45} {c:>6} {u_score_sum[k]:>10.2f}')

# Print top no_underage_* tags fired on V10-missed teens (these should NOT fire on teens but do)
print('\n=== AGGREGATE: no_underage_* tags fired on the 22 V10-missed teen FNs (false suppression!) ===')
nu2_score_sum = {}
nu2_count = {}
for it in fn_v10_only:
    for k, v in it['top_no_underage']:
        if v < 0.1: continue
        nu2_score_sum[k] = nu2_score_sum.get(k, 0) + v
        nu2_count[k] = nu2_count.get(k, 0) + 1
top_nu2 = sorted(nu2_count.items(), key=lambda x: -x[1])[:15]
if top_nu2:
    print(f'{"tag":<45} {"count":>6} {"sum_score":>10}')
    for k, c in top_nu2:
        print(f'  {k[:43]:<45} {c:>6} {nu2_score_sum[k]:>10.2f}')
else:
    print('  (none — no_underage_ tags didn\'t fire strongly on missed teens)')
