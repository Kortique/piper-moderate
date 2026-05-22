#!/usr/bin/env python3
"""
Test V8pas80 vs V8pas80-v2 on Screenshot_141.png (maid woman case).

Flow:
  1. Encode PNG → base64 data URI
  2. Launch Piper d2911d10bb to get siglip2_details
  3. Run both V8pas80 and V8pas80-v2 locally on those details
  4. Print scores + verdict
"""
import json, base64, time, sys, os, re
from pathlib import Path
import httpx
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
load_dotenv(BASE / '.env')
TOKEN = os.getenv('PIPER_TOKEN')
PROJECT = 'ce79f7e299'  # full no_underage labels — needed for V8pas80 features
API = 'https://piper-next.artworks.ai/api'
HDR = {'User-Token': TOKEN, 'Content-Type': 'application/json'}

from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')
def unmult(k, v):
    m = X20_RE.search(k)
    if not m: return float(v)
    mult = float(m.group(1))
    return 0.999/mult if v >= 0.999 else float(v)/mult
def strip_mult(k): return X20_RE.sub('', k)
def sanitize(n): return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')
def noisy_or(d, f):
    p = 1.0
    for k, v in d.items():
        if f(k): p *= 1.0 - float(v)
    return 1.0 - p


def run_piper(image_data_uri):
    """Launch + poll Piper ce79f7e299. Returns siglip2_details dict or {error: ...}."""
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        r = client.post(f'{API}/projects/{PROJECT}/launch',
                        headers=HDR,
                        json={'inputs': {'image': image_data_uri, 'providers': ['siglip2']}})
        if r.status_code != 200:
            return {'error': f'HTTP {r.status_code}: {r.text[:300]}'}
        run_id = r.json()['_id']
        print(f'  launched run: {run_id}', flush=True)
        for i in range(40):
            time.sleep(3)
            rs = client.get(f'{API}/launches/{run_id}/state', headers=HDR)
            if rs.status_code != 200:
                continue
            state = rs.json()
            outputs = state.get('outputs') or {}
            errors = state.get('errors') or []
            if errors:
                return {'error': str(errors)}
            if 'siglip2_details' in outputs:
                return outputs
            print(f'    poll {i}: state.statusType={state.get("statusType")}', flush=True)
        return {'error': 'timeout'}


def build_X_for_one(siglip_details, feats):
    """Build single-row X from siglip2_details.underage.labels structure."""
    und = (siglip_details.get('underage') or {}).get('labels') or {}
    underage_labels = und.get('underage') or {}
    adult_labels = und.get('adult') or {}

    u = {}
    for k, v in underage_labels.items():
        raw = unmult(k, v); k2 = strip_mult(k)
        u[k2] = max(u.get(k2, 0.0), raw)
    a = dict((k, float(v)) for k, v in adult_labels.items())

    idx = {f: i for i, f in enumerate(feats)}
    X = np.zeros((1, len(feats)), dtype=np.float32)
    for k, v in u.items():
        f = sanitize(k)
        if f in idx: X[0, idx[f]] = v
    for k, v in a.items():
        f = 'adult__' + sanitize(k)
        if f in idx: X[0, idx[f]] = float(v)
    body = noisy_or(u, lambda k: k in BODY_LABELS)
    ctx  = noisy_or(u, lambda k: k in CONTEXT_LABELS)
    inter = noisy_or(u, lambda k: k in INTERACTION_LABELS)
    bc = body + ctx
    if '_child_body' in idx: X[0, idx['_child_body']] = body
    if '_child_context' in idx: X[0, idx['_child_context']] = ctx
    if '_child_interaction' in idx: X[0, idx['_child_interaction']] = inter
    if '_body_vs_context' in idx: X[0, idx['_body_vs_context']] = (body/bc) if bc>0 else 0.0
    return X, u, a


def main():
    img_path = DATA / 'disagree_images' / 'Screenshot_141.png'
    if not img_path.exists():
        print(f'NOT FOUND: {img_path}'); sys.exit(1)

    img_bytes = img_path.read_bytes()
    print(f'Image: {img_path.name}  size={len(img_bytes)/1024:.1f}KB', flush=True)
    data_uri = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode()

    print('\nLaunching Piper ce79f7e299...', flush=True)
    out = run_piper(data_uri)
    if 'error' in out:
        print(f'  ERROR: {out["error"]}', flush=True); sys.exit(1)

    siglip_details = out.get('siglip2_details') or {}
    if not siglip_details:
        print(f'  No siglip2_details. Outputs keys: {list(out.keys())}', flush=True); sys.exit(1)

    und_section = siglip_details.get('underage') or {}
    print(f'\nSigLIP underage section:')
    print(f'  minor={und_section.get("minor"):.4f}  adult={und_section.get("adult"):.4f}', flush=True)
    print(f'  pipeline-lgbm-score (V8pas80, what production sees): {(und_section.get("lgbm") or {}).get("score")}')
    print(f'  blocked-by-pipeline: {(und_section.get("lgbm") or {}).get("blocked")}')

    # Now run V8pas80 and V8pas80-v2 locally
    for name, mfile, ffile in [
        ('V8pas80',     'lgbm_underage_v8pas80.txt',    'lgbm_v8pas80_features.json'),
        ('V8pas80-v2',  'lgbm_underage_v8pas80_v2.txt', 'lgbm_v8pas80_v2_features.json'),
    ]:
        booster = lgb.Booster(model_file=str(DATA / mfile))
        feats = json.loads((DATA / ffile).read_text())
        X, u, a = build_X_for_one(siglip_details, feats)
        score = float(booster.predict(X)[0])
        # Top contributors via SHAP
        shap = booster.predict(X, pred_contrib=True)[0]
        # shap[:-1] feature contribs + shap[-1] bias
        contribs = sorted(zip(feats, shap[:-1]), key=lambda x: -abs(x[1]))[:8]
        print(f'\n[{name}]  score={score:.4f}  features={len(feats)}', flush=True)
        print(f'  Top SHAP contributors:')
        for f, c in contribs:
            mark = 'BCI' if f.startswith('_') else 'adult' if f.startswith('adult__') else 'underage'
            val = X[0, feats.index(f)]
            sign = '+' if c > 0 else '-'
            print(f'    {sign}  {f:<45} contrib={c:+.4f}  val={val:.4f}  [{mark}]', flush=True)

    # Save raw output
    (DATA / 'screenshot141_siglip.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print('\n  raw siglip output → data/screenshot141_siglip.json', flush=True)


if __name__ == '__main__':
    main()
