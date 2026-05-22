#!/usr/bin/env python3
"""Export V8pas80-v2 as Piper lgbm_evaluate JS (delegates to export_v8pas80_js.build_script)."""
import json, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from export_v8pas80_js import build_script, parse_trees


def main():
    feats = json.loads((DATA / 'lgbm_v8pas80_v2_features.json').read_text())
    model_str = (DATA / 'lgbm_underage_v8pas80_v2.txt').read_text()
    meta_path = DATA / 'lgbm_v8pas80_v2_meta.json'
    meta = json.loads(meta_path.read_text())
    # Reuse v8pas80 JS template — version label patched below
    meta['cv_auc'] = meta.get('cv_auc_slim', meta.get('cv_auc', 0))
    trees = parse_trees(model_str)
    js = build_script(feats, trees, meta)
    # Patch version label to v2 (find/replace single 'v8pas80' → 'v8pas80_v2')
    js = js.replace("version:      'v8pas80'", "version:      'v8pas80_v2'")
    js = js.replace('// v8pas80', '// v8pas80_v2')
    out = DATA / 'lgbm_evaluate_v8pas80_v2.js'
    out.write_text(js)
    print(f'V8pas80-v2: features={len(feats)}, trees={len(trees)} → {out} ({len(js)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
