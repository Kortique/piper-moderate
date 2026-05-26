#!/usr/bin/env python3
"""
extract_v8cs80_ls_fps.py
------------------------
Score every confirmed-adult LS item with V8cs80 and dump the top-N items where
the model blocks (lgbm >= 0.30). These are the next round of hard-neg candidates
for Grok validation → train_v8d (next sprint).

Output: data/v8cs80_ls_fps_2026-05.json
  { thr, n_adult_total, n_fp, items: [{ id, lgbm, underage_labels, adult_labels, prompt }] }
"""
import json, ast, re, sys
from pathlib import Path
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')
BCI = ['_child_body','_child_context','_child_interaction','_body_vs_context']
THR = 0.30
TOP_N = 100


def unmult(k, v):
    m = X20_RE.search(k)
    if not m: return float(v)
    mu = float(m.group(1))
    return 0.999/mu if v >= 0.999 else float(v)/mu
def strip_mult(k): return X20_RE.sub('', k)
def sanitize(n): return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')
def noisy_or(d, st):
    p = 1.0
    for k, v in d.items():
        if k in st: p *= 1.0 - float(v)
    return 1.0 - p


def score(model, feats, u_raw, a_raw):
    idx = {f: i for i, f in enumerate(feats)}
    vec = [0.0]*len(feats)
    u = {}
    for k, v in (u_raw or {}).items():
        raw = unmult(k, v); k2 = strip_mult(k)
        if k2 not in u or u[k2] < raw: u[k2] = raw
    for k, v in u.items():
        fk = sanitize(k)
        if fk in idx: vec[idx[fk]] = v
    for k, v in (a_raw or {}).items():
        fk = 'adult__' + sanitize(k)
        if fk in idx: vec[idx[fk]] = float(v)
    body = noisy_or(u, BODY_LABELS); ctx = noisy_or(u, CONTEXT_LABELS); inter = noisy_or(u, INTERACTION_LABELS)
    bc = body + ctx
    if '_child_body' in idx: vec[idx['_child_body']] = body
    if '_child_context' in idx: vec[idx['_child_context']] = ctx
    if '_child_interaction' in idx: vec[idx['_child_interaction']] = inter
    if '_body_vs_context' in idx: vec[idx['_body_vs_context']] = (body/bc) if bc>0 else 0
    return float(model.predict([vec])[0])


def main():
    booster = lgb.Booster(model_file=str(DATA / 'lgbm_underage_v8cs80.txt'))
    feats = json.loads((DATA / 'lgbm_v8cs80_features.json').read_text())
    print(f'V8cs80 loaded: features={len(feats)} trees={booster.num_trees()}')

    raw = (BASE / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    qd = json.loads(raw)
    fps = []
    n_adult = 0
    for v in qd.values():
        age = (v.get('age') or {}).get('ageFrom')
        if age is None: continue
        if age <= 17: continue   # only adults
        n_adult += 1
        sd = v.get('siglip2_details')
        if isinstance(sd, str):
            try: sd = ast.literal_eval(sd)
            except: continue
        det = (sd or {}).get('underage') or {}
        labels = det.get('labels') or {}
        u = labels.get('underage') or {}; a = labels.get('adult') or {}
        if not u and not a: continue
        s = score(booster, feats, u, a)
        if s < THR: continue
        fps.append({
            'id': f"ls_{v['task_id']}", 'lgbm': round(s, 4),
            'media': v.get('media'),
            'prompt': (v.get('prompt') or '')[:300],
            'age_from': age, 'age_to': (v.get('age') or {}).get('ageTo'),
            'underage_labels': u, 'adult_labels': a,
        })
    fps.sort(key=lambda x: -x['lgbm'])
    top = fps[:TOP_N]

    out = {
        'model': 'v8cs80', 'threshold': THR,
        'n_adult_total': n_adult, 'n_fp_total': len(fps),
        'fpr': round(len(fps)/n_adult, 4) if n_adult else None,
        'top_n_saved': len(top),
        'items': top,
    }
    out_path = DATA / 'v8cs80_ls_fps_2026-05.json'
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f'\nLS adults: {n_adult}')
    print(f'Blocked (lgbm>={THR}): {len(fps)}  FPR={out["fpr"]*100:.1f}%')
    print(f'Saved top-{TOP_N} → {out_path}')
    if top:
        print(f'\nTop 5 LS FPs (highest V8cs80 score on adult):')
        for it in top[:5]:
            print(f'  {it["id"]}  lgbm={it["lgbm"]:.3f}  age={it["age_from"]}-{it["age_to"]}  prompt={it["prompt"][:80]!r}')


if __name__ == '__main__':
    main()
