#!/usr/bin/env python3
"""Convert LightGBM .txt model + feature list into a compact JS evaluator
for Piper lgbm_evaluate node.

Usage:
    python scripts/export_lgbm_js.py <name>
    # reads data/lgbm_underage_<name>.txt and data/lgbm_<name>_features.json
    # writes data/lgbm_evaluate_<name>.js
"""
import json, sys, re, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'


def parse_trees(model_str):
    trees = []
    blocks = re.split(r'Tree=\d+', model_str)
    for b in blocks[1:]:
        lv = re.search(r'leaf_value=([^\n]+)', b)
        nf = re.search(r'split_feature=([^\n]+)', b)
        nt = re.search(r'threshold=([^\n]+)', b)
        lc = re.search(r'left_child=([^\n]+)', b)
        rc = re.search(r'right_child=([^\n]+)', b)
        if not (lv and nf): continue
        leaves = [float(x) for x in lv.group(1).split()]
        sf = [int(x) for x in nf.group(1).split()]
        thr = [float(x) for x in nt.group(1).split()]
        l = [int(x) for x in lc.group(1).split()]
        r2 = [int(x) for x in rc.group(1).split()]
        n_int = len(sf)
        all_ch = set(l + r2)
        root = next(ni for ni in range(n_int) if ni not in all_ch)
        splits = [[sf[i], thr[i]] for i in range(n_int)]
        children = [[l[i], r2[i]] for i in range(n_int)]
        trees.append({'r': root, 's': splits, 'c': children, 'l': leaves})
    return trees


def build_js(name, meta, feats, trees):
    today = datetime.date.today().isoformat()
    feats_js = json.dumps(feats)
    trees_js = json.dumps(trees, separators=(',', ':'))
    cv_auc = meta.get('cv_auc', 0)
    note = meta.get('note', '')
    n_features = len(feats)
    n_trees = len(trees)
    # Build feature reader: features can come from underage/adult/no_underage groups
    # adult__X → labelsObj.adult.X
    # no_underage__X → labelsObj.no_underage.X
    # else → labelsObj.underage.X
    js = f"""// lgbm_evaluate_{name} — LGBM Underage Scorer ({name.upper()})
// Generated: {today}
// Model: {n_trees} trees, CV AUC={cv_auc:.4f}, features={n_features}
// Notes: {note}
// v_{name}_{today.replace('-','')}

const LGBM_FEATURES = {feats_js};

const LGBM_TREES = {trees_js};

function lgbm_predict_{name}(vec) {{
  let score = 0;
  for (const t of LGBM_TREES) {{
    let node = t.r;
    while (node >= 0) {{
      const [fi, thr] = t.s[node];
      const [l, r] = t.c[node];
      node = vec[fi] <= thr ? l : r;
    }}
    score += t.l[-(node + 1)];
  }}
  return 1 / (1 + Math.exp(-score));
}}

function lgbm_evaluate_{name}(labelsObj) {{
  const vec = LGBM_FEATURES.map(f => {{
    if (f.startsWith('adult__')) {{
      const key = f.slice(7);
      return (labelsObj.adult && labelsObj.adult[key]) || 0;
    }}
    if (f.startsWith('no_underage__')) {{
      const key = f.slice(13);
      return (labelsObj.no_underage && labelsObj.no_underage[key]) || 0;
    }}
    return (labelsObj.underage && labelsObj.underage[f]) || 0;
  }});
  return lgbm_predict_{name}(vec);
}}
"""
    return js


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'v7r'
    model_path = DATA / f'lgbm_underage_{name}.txt'
    feat_path = DATA / f'lgbm_{name}_features.json'
    meta_path = DATA / f'lgbm_{name}_meta.json'
    out_path = DATA / f'lgbm_evaluate_{name}.js'

    model_str = model_path.read_text()
    feats = json.loads(feat_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    trees = parse_trees(model_str)
    print(f'{name}: features={len(feats)}, trees={len(trees)}', flush=True)

    js = build_js(name, meta, feats, trees)
    out_path.write_text(js)
    print(f'  Saved: {out_path} ({len(js)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
