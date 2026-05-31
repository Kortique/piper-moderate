#!/usr/bin/env python3
"""
retrain_v6_v8_v11.py — safe retraining wrapper for V6c / V8cs80 / V11cs80.

Steps:
  1. Pre-flight: verify anchor snapshot exists; refuse to run without one
  2. Archive current model files to backups/models_pre_retrain_<ts>/
  3. Run train_v6c_holdout, train_v8c_holdout, train_v11c_holdout sequentially
  4. Print before/after multi-seed AUC + child/teen recall + adult FPR per model
  5. Suggest deploy commands

The script does NOT auto-deploy to Piper — that's a separate manual step after
you eyeball the metrics and confirm the new model is at least as good as old.

If you don't like the result, restore archived models from
backups/models_pre_retrain_<ts>/ and skip the deploy.

Usage:
    python scripts/retrain_v6_v8_v11.py             # all three
    python scripts/retrain_v6_v8_v11.py --only v8   # one model
    python scripts/retrain_v6_v8_v11.py --skip-train  # only print before/after if you already trained
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
ARCHIVE_ROOT = BASE / 'backups' / 'models_pre_retrain'

MODELS = {
    'v6':  {
        'train_script': 'scripts/train_v6c_holdout.py',
        'files': ['lgbm_underage_v6c.txt', 'lgbm_v6c_features.json', 'lgbm_v6c_meta.json'],
        'meta':  'lgbm_v6c_meta.json',
        'multi_seed_key': 'multi_seed',
    },
    'v8':  {
        'train_script': 'scripts/train_v8c_holdout.py',
        'files': ['lgbm_underage_v8cs80.txt', 'lgbm_underage_v8c.txt',
                  'lgbm_v8cs80_features.json', 'lgbm_v8c_features.json',
                  'lgbm_v8cs80_meta.json'],
        'meta':  'lgbm_v8cs80_meta.json',
        'multi_seed_key': 'multi_seed_slim',
    },
    'v11': {
        'train_script': 'scripts/train_v11c_holdout.py',
        'files': ['lgbm_underage_v11cs80.txt', 'lgbm_underage_v11c.txt',
                  'lgbm_v11cs80_features.json', 'lgbm_v11c_features.json',
                  'lgbm_v11cs80_meta.json'],
        'meta':  'lgbm_v11cs80_meta.json',
        'multi_seed_key': 'multi_seed_slim',
    },
}


def latest_anchor():
    """Return path to newest anchor in backups/anchors/."""
    anchors_dir = BASE / 'backups' / 'anchors'
    if not anchors_dir.exists():
        return None
    files = sorted(anchors_dir.glob('anchor_*.db'), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def archive_models(archive_dir: Path, only: list = None):
    archive_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for key, info in MODELS.items():
        if only and key not in only:
            continue
        for fn in info['files']:
            src = DATA / fn
            if src.exists():
                shutil.copy2(src, archive_dir / fn)
                n += 1
    return n


def load_meta(key: str) -> dict:
    p = DATA / MODELS[key]['meta']
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt_metrics(meta: dict, key: str) -> dict:
    if not meta:
        return {}
    ms = meta.get(MODELS[key]['multi_seed_key']) or meta.get('multi_seed_slim') or meta.get('multi_seed') or {}
    auc = (ms.get('auc') or {})
    cr  = (ms.get('child_recall') or {})
    tr  = (ms.get('teen_recall') or {})
    fpr = (ms.get('adult_fpr') or {})
    return {
        'auc':     auc.get('mean'),
        'auc_std': auc.get('std'),
        'child':   cr.get('mean'),
        'teen':    tr.get('mean'),
        'adult_fpr': fpr.get('mean'),
        'n_train': meta.get('n_train'),
        'n_test':  meta.get('n_test'),
    }


def print_metrics(label: str, key: str, m: dict):
    if not m:
        print(f'  {label:<8} {key:<4}  (no meta)')
        return
    def f(v, p=3): return f'{v:.{p}f}' if isinstance(v,(int,float)) else '—'
    print(f'  {label:<8} {key:<4}  '
          f'AUC={f(m.get("auc"))}±{f(m.get("auc_std"),4)}  '
          f'child={f(m.get("child"))}  teen={f(m.get("teen"))}  '
          f'adult_FPR={f(m.get("adult_fpr"))}  '
          f'n_train={m.get("n_train")}  n_test={m.get("n_test")}')


def run_train(key: str):
    script = MODELS[key]['train_script']
    print(f'\n=== Training {key.upper()} via {script} ===', flush=True)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, '-u', script], cwd=str(BASE), check=True)
    except subprocess.CalledProcessError as e:
        print(f'  ✗ training failed: exit code {e.returncode}', flush=True)
        return False
    print(f'  ✓ {key.upper()} done in {int(time.time()-t0)}s', flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['v6','v8','v11'], action='append',
                    help='Train only this model (can repeat)')
    ap.add_argument('--skip-train', action='store_true',
                    help='Skip training, just print before/after from existing meta')
    ap.add_argument('--no-anchor-check', action='store_true',
                    help='Skip the "anchor must exist" pre-flight check')
    args = ap.parse_args()

    only = args.only or ['v6','v8','v11']

    print('='*70)
    print('SAFE RETRAIN — V6c / V8cs80 / V11cs80')
    print('='*70)
    print(f'  models to train: {", ".join(only)}')

    # Pre-flight: anchor must exist
    if not args.no_anchor_check:
        anchor = latest_anchor()
        if not anchor:
            print('\n❌ No anchor snapshot found in backups/anchors/.')
            print('   Run first:  python scripts/snapshot.py --label pre_retrain')
            print('   This protects your label data in case retraining is interrupted.')
            sys.exit(1)
        age_min = int((time.time() - anchor.stat().st_mtime) / 60)
        print(f'  latest anchor: {anchor.name}  ({age_min} min old)')
        if age_min > 60:
            print(f'  ⚠ Anchor is more than 1 hour old. Consider fresh snapshot:')
            print(f'    python scripts/snapshot.py --label pre_retrain')

    # Step 1: read BEFORE metrics
    before = {}
    for k in only:
        before[k] = fmt_metrics(load_meta(k), k)
    print('\nBEFORE retrain (current models):')
    for k in only:
        print_metrics('before', k, before[k])

    if args.skip_train:
        print('\n--skip-train passed; not retraining. Done.')
        return

    # Step 2: archive
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = ARCHIVE_ROOT / f'archive_{ts}'
    n_archived = archive_models(archive_dir, only)
    print(f'\nArchived {n_archived} model files → {archive_dir}')

    # Step 3: train
    failures = []
    for k in only:
        if not run_train(k):
            failures.append(k)

    if failures:
        print(f'\n❌ Failures: {", ".join(failures)}')
        print(f'   Restore archived models from {archive_dir} if you want to roll back.')
        sys.exit(2)

    # Step 4: read AFTER metrics + diff
    after = {}
    for k in only:
        after[k] = fmt_metrics(load_meta(k), k)
    print('\n' + '='*70)
    print('AFTER retrain:')
    for k in only:
        print_metrics('after', k, after[k])

    print('\nDELTA (after - before):')
    for k in only:
        b, a = before.get(k) or {}, after.get(k) or {}
        if not b or not a:
            continue
        def d(key, sign='+'):
            bv, av = b.get(key), a.get(key)
            if not isinstance(bv,(int,float)) or not isinstance(av,(int,float)):
                return '—'
            delta = av - bv
            s = ('+' if delta >= 0 else '')
            return f'{s}{delta:+.4f}'.replace('++','+')
        print(f'  {k.upper():<4}  AUC: {d("auc")}   child_recall: {d("child")}   '
              f'teen_recall: {d("teen")}   adult_FPR: {d("adult_fpr")}')

    print(f'\n📦 Archived previous models in: {archive_dir}')
    print(f'   To roll back if needed: copy *.txt and *.json from there back into data/')
    print('\n📤 To deploy new models to Piper:')
    if 'v8' in only:
        print('   python scripts/export_v8cs80_js.py')
        print('   python scripts/deploy_piper_lgbm.py d2911d10bb data/lgbm_evaluate_v8cs80.js --threshold 0.51')
    if 'v11' in only:
        print('   python scripts/export_v11cs80_js.py')
        print('   python scripts/deploy_piper_lgbm.py ce79f7e299 data/lgbm_evaluate_v11cs80.js')
    if 'v6' in only:
        print('   (V6 is gallery-only, not in Piper d2911d10bb — no deploy needed)')
    print('\nVerify in gallery first by hitting Ctrl+Shift+R after deploy.')


if __name__ == '__main__':
    main()
