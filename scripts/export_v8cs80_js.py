#!/usr/bin/env python3
"""Export V8cs80 as Piper lgbm_evaluate JS."""
import json, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from export_v8pas80_js import build_script, parse_trees


def main():
    feats = json.loads((DATA / 'lgbm_v8cs80_features.json').read_text())
    model_str = (DATA / 'lgbm_underage_v8cs80.txt').read_text()
    meta = json.loads((DATA / 'lgbm_v8cs80_meta.json').read_text())
    # The v8pas80 template expects a 'cv_auc' key — use the slim multi-seed mean
    ss = (meta.get('multi_seed_slim') or {}).get('auc') or {}
    meta['cv_auc'] = float(ss.get('mean', 0))
    trees = parse_trees(model_str)
    js = build_script(feats, trees, meta)
    js = js.replace("version:      'v8pas80'", "version:      'v8cs80'")
    js = js.replace('// v8pas80', '// v8cs80')
    out = DATA / 'lgbm_evaluate_v8cs80.js'
    out.write_text(js)
    print(f'V8cs80: features={len(feats)}, trees={len(trees)} -> {out} ({len(js)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
