#!/usr/bin/env python3
"""Export V11cs80 as Piper lgbm_evaluate JS (uses the V11 template)."""
import json, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from export_v11_js import build_script, parse_trees


def main():
    feats = json.loads((DATA / 'lgbm_v11cs80_features.json').read_text())
    model_str = (DATA / 'lgbm_underage_v11cs80.txt').read_text()
    meta = json.loads((DATA / 'lgbm_v11cs80_meta.json').read_text())
    ss = (meta.get('multi_seed_slim') or {}).get('auc') or {}
    meta['cv_auc'] = float(ss.get('mean', 0))
    trees = parse_trees(model_str)
    js = build_script(feats, trees, meta)
    js = js.replace("version:      'v11'", "version:      'v11cs80'")
    js = js.replace("version:      'v11s80'", "version:      'v11cs80'")
    js = js.replace('// v11', '// v11cs80')
    out = DATA / 'lgbm_evaluate_v11cs80.js'
    out.write_text(js)
    print(f'V11cs80: features={len(feats)}, trees={len(trees)} -> {out} ({len(js)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
