#!/usr/bin/env python3
"""
Export a complete Piper-compatible lgbm_evaluate JS by:
 - taking the production V6 script (from a backup) as scaffold
 - replacing LGBM_FEATURES + LGBM_TREES with new model
 - removing :x20/:x5 multiplier from getAdjustedScores (Path A)
 - making buildVec strip :x20/:x5 from key when looking up featIdx

Usage:
    python scripts/export_full_lgbm_js.py <model_name> <backup_path>
    e.g.   python scripts/export_full_lgbm_js.py v7pa backups/piper_pre_rename/d2911d10bb_20260521_140848.json
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


def build_script(name, feats, trees, meta):
    """Build full JS using new helper functions matching production structure."""
    today = datetime.date.today().isoformat()
    cv_auc = meta.get('cv_auc', 0)
    note = meta.get('note', '')
    feats_js = json.dumps(feats)
    trees_js = json.dumps(trees, separators=(',', ':'))
    # NOTE: getAdjustedScores no longer applies the :x20 multiplier (Path A).
    # buildVec strips :x20/:x5 suffix from key when looking up featIdx
    # so model features can be stored without the suffix.
    return f"""// lgbm_evaluate — Piper node ({name.upper()})
// Path A: :x20 multiplier removed from getAdjustedScores.
// LGBM features have :x20/:x5 suffix STRIPPED from names.
// Model: {len(trees)} trees, {len(feats)} features, CV AUC={cv_auc:.4f}, trained {today}
// {note}

const LGBM_FEATURES = {feats_js};

const LGBM_TREES = {trees_js};

function lgbmPredict(vec) {{
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

function combineScores(labels, group) {{
  return 1 - group.reduce((prod, label) => prod * (1 - (labels[label] || 0)), 1);
}}

// Path A: no multiplier. Returns rawLabels as-is (defensive copy).
function getAdjustedScores(rawLabels) {{
  return Object.assign({{}}, rawLabels);
}}

function stripMultSuffix(name) {{
  return name.replace(/:x\\d+(?:\\.\\d+)?$/, '');
}}

function buildVec(labelsFlat) {{
  const featIdx = Object.create(null);
  for (let i = 0; i < LGBM_FEATURES.length; i++) featIdx[LGBM_FEATURES[i]] = i;
  const vec = new Float32Array(LGBM_FEATURES.length);
  for (const [key, val] of Object.entries(labelsFlat)) {{
    if (typeof val !== 'number') continue;
    if (key.startsWith('adult__')) {{
      if (key in featIdx) vec[featIdx[key]] = val;
    }} else if (key.startsWith('adult_')) {{
      const fname = 'adult__' + key.slice(6);
      if (fname in featIdx) vec[featIdx[fname]] = val;
    }} else if (key.startsWith('underage_')) {{
      const fname = stripMultSuffix(key.slice(9));
      if (fname in featIdx) vec[featIdx[fname]] = val;
    }} else if (key.startsWith('no_underage_')) {{
      // Path A + merge: no_underage_X scores are routed into adult__X feature
      const fname = 'adult__' + key.slice(12);
      if (fname in featIdx) vec[featIdx[fname]] = val;
    }}
  }}
  return vec;
}}

export async function run({{ inputs }}) {{
  const {{ NextNode }} = DEFINITIONS;

  const LGBM_THRESHOLD       = (typeof inputs.lgbm_threshold === 'number') ? inputs.lgbm_threshold : 0.35;
  const UNDERAGE_CONFIDENCE  = 0.72;
  const UNDERAGE_MIN_SCORE   = 0.035;
  const AUTO_TRIGGER         = 0.70;

  const rawLabels = Object.fromEntries(
    Object.entries(inputs.labels || {{}}).filter(([, v]) => v > 0)
  );
  const affected = getAdjustedScores(rawLabels);

  const MINOR_LABELS = Object.keys(affected).filter(k => k.startsWith('underage'));
  const ADULT_LABELS = Object.keys(affected).filter(k => k.startsWith('adult'));
  const minorRisk    = combineScores(affected, MINOR_LABELS);
  const adultConf    = combineScores(affected, ADULT_LABELS);
  const total        = minorRisk + adultConf;
  const underageConf = total > 0 ? minorRisk / total : 0;
  const ratioBlocked =
    (underageConf >= UNDERAGE_CONFIDENCE && minorRisk >= UNDERAGE_MIN_SCORE)
    || minorRisk >= AUTO_TRIGGER;

  const vec       = buildVec(affected);
  const lgbmScore = lgbmPredict(vec);
  const lgbmBlocked = lgbmScore >= LGBM_THRESHOLD;

  const topFeats = LGBM_FEATURES
    .map((name, i) => ({{ name, val: vec[i] }}))
    .filter(f => f.val > 0.001)
    .sort((a, b) => b.val - a.val)
    .slice(0, 5)
    .map(f => `${{f.name}}=${{f.val.toFixed(3)}}`)
    .join(', ');

  return NextNode.from({{
    outputs: {{
      labels:  lgbmBlocked ? ['underage'] : [],
      details: {{
        score:        Math.round(lgbmScore * 10000) / 10000,
        blocked:      lgbmBlocked,
        disagree:     ratioBlocked !== lgbmBlocked,
        top_features: topFeats,
        version:      '{name}',
        minor:        Math.round(minorRisk * 100000) / 100000,
      }},
    }},
  }});
}}
// {name}
"""


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'v7pa'
    feats = json.loads((DATA / f'lgbm_{name}_features.json').read_text())
    model_str = (DATA / f'lgbm_underage_{name}.txt').read_text()
    meta = json.loads((DATA / f'lgbm_{name}_meta.json').read_text())
    trees = parse_trees(model_str)
    js = build_script(name, feats, trees, meta)
    out = DATA / f'lgbm_evaluate_{name}.js'
    out.write_text(js)
    print(f'{name}: {len(feats)} features, {len(trees)} trees → {out} ({len(js)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
