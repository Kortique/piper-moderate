#!/usr/bin/env python3
"""Export V8pas80 as Piper lgbm_evaluate JS (BCI + path A, no no_underage merge)."""
import json, re, datetime, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS


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


def build_script(feats, trees, meta):
    today = datetime.date.today().isoformat()
    cv_auc = meta.get('cv_auc', 0)
    note = meta.get('note', '')
    feats_js = json.dumps(feats)
    trees_js = json.dumps(trees, separators=(',', ':'))
    body_js = json.dumps(sorted(BODY_LABELS))
    ctx_js  = json.dumps(sorted(CONTEXT_LABELS))
    int_js  = json.dumps(sorted(INTERACTION_LABELS))
    # NOTE: V8pas80 is for d2911d10bb (no no_underage). buildVec routes underage_/adult_ only.
    return f"""// lgbm_evaluate — Piper node (V8pas80)
// Path A + BCI feature split + hard-neg mining x20 (V7pa-FPs)
// Model: {len(trees)} trees, {len(feats)} features, CV AUC={cv_auc:.4f}, trained {today}
// {note}

const LGBM_FEATURES = {feats_js};

const LGBM_TREES = {trees_js};

const BCI_BODY = new Set({body_js});
const BCI_CONTEXT = new Set({ctx_js});
const BCI_INTERACTION = new Set({int_js});

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

function getAdjustedScores(rawLabels) {{
  return Object.assign({{}}, rawLabels);
}}

function stripMultSuffix(name) {{
  return name.replace(/:x\\d+(?:\\.\\d+)?$/, '');
}}

function noisyOrSet(scoresDict, keySet) {{
  let p = 1.0;
  for (const k in scoresDict) {{
    if (keySet.has(k)) p *= 1.0 - scoresDict[k];
  }}
  return 1.0 - p;
}}

function buildVec(labelsFlat) {{
  const featIdx = Object.create(null);
  for (let i = 0; i < LGBM_FEATURES.length; i++) featIdx[LGBM_FEATURES[i]] = i;
  const vec = new Float32Array(LGBM_FEATURES.length);

  const underageRaw = Object.create(null);
  for (const key in labelsFlat) {{
    const val = labelsFlat[key];
    if (typeof val !== 'number') continue;
    if (key.startsWith('adult__')) {{
      if (key in featIdx) vec[featIdx[key]] = val;
    }} else if (key.startsWith('adult_')) {{
      const fname = 'adult__' + key.slice(6);
      if (fname in featIdx) vec[featIdx[fname]] = val;
    }} else if (key.startsWith('underage_')) {{
      const fname = stripMultSuffix(key.slice(9));
      if (fname in featIdx) vec[featIdx[fname]] = val;
      underageRaw[fname] = Math.max(underageRaw[fname] || 0, val);
    }}
  }}

  const body  = noisyOrSet(underageRaw, BCI_BODY);
  const ctx   = noisyOrSet(underageRaw, BCI_CONTEXT);
  const inter = noisyOrSet(underageRaw, BCI_INTERACTION);
  const bcTot = body + ctx;
  const bodyVsCtx = bcTot > 0 ? body / bcTot : 0;
  if ('_child_body' in featIdx)        vec[featIdx['_child_body']]        = body;
  if ('_child_context' in featIdx)     vec[featIdx['_child_context']]     = ctx;
  if ('_child_interaction' in featIdx) vec[featIdx['_child_interaction']] = inter;
  if ('_body_vs_context' in featIdx)   vec[featIdx['_body_vs_context']]   = bodyVsCtx;
  return vec;
}}

export async function run({{ inputs }}) {{
  const {{ NextNode }} = DEFINITIONS;

  const LGBM_THRESHOLD       = (typeof inputs.lgbm_threshold === 'number') ? inputs.lgbm_threshold : 0.30;
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
        version:      'v8pas80',
        minor:        Math.round(minorRisk * 100000) / 100000,
      }},
    }},
  }});
}}
// v8pas80
"""


def main():
    feats = json.loads((DATA / 'lgbm_v8pas80_features.json').read_text())
    model_str = (DATA / 'lgbm_underage_v8pas80.txt').read_text()
    meta = json.loads((DATA / 'lgbm_v8pas80_meta.json').read_text())
    trees = parse_trees(model_str)
    js = build_script(feats, trees, meta)
    out = DATA / 'lgbm_evaluate_v8pas80.js'
    out.write_text(js)
    print(f'V8pas80: features={len(feats)}, trees={len(trees)} → {out} ({len(js)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
