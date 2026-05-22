#!/usr/bin/env python3
"""
Compare our V8pas80-v2 vs Tom's K=30 on his 6675-image dataset.

Ground truth: qwen3.min_age <= 14 (same convention as Tom).
Tom's automated baseline: k30_blocked (threshold 0.10 internal).
Our automated baseline: ours.lgbm_blocked (V8pas80-v2 @ thr 0.30).

Outputs:
  - Confusion matrix per model
  - Per-class recall/FPR/AUC
  - Disagreement list (cases where Tom and we disagree)
  - data/k30_vs_v8pas80_v2_report.json
  - data/k30_vs_v8pas80_v2_disagreements.json (sample of ~200 disagreements)
"""
import json, os
from pathlib import Path
from collections import Counter
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

# Load our results from JSONL (dedupe by gid, keep last)
records = {}
with open(DATA / 'k30_ours.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
            if r.get('ours') and 'error' not in r['ours']:
                records[r['gid']] = r
        except Exception:
            pass

print(f'Loaded {len(records)} records from data/k30_ours.jsonl\n')

# Build evaluable items — require qwen3.min_age and at least one model verdict
items = []
for r in records.values():
    age = r.get('qwen_min_age')
    if age is None: continue
    our = r.get('ours', {})
    if 'error' in our: continue
    items.append({
        'gid': r['gid'],
        'task_id': r['task_id'],
        'media': r['media'],
        'gt_underage': age <= 14,        # Tom's ground-truth convention
        'qwen_min_age': age,
        'qwen_max_age': r.get('qwen_max_age'),
        'tom_k30_score':    r.get('k30_score'),
        'tom_k30_blocked':  r.get('k30_blocked'),
        'tom_k25_score':    r.get('k25_score'),
        'tom_prod_score':   r.get('prod_score'),
        'tom_prod_blocked': r.get('prod_blocked'),
        'our_score':        our.get('lgbm_score'),
        'our_blocked':      our.get('lgbm_blocked'),
    })

n = len(items)
n_pos = sum(1 for it in items if it['gt_underage'])
n_neg = n - n_pos
print(f'Evaluable items: {n}  (underage ≤14: {n_pos}  adult: {n_neg})\n')


def confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    recall = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    return tp, fp, fn, tn, recall, fpr, precision


def evaluate(name, scores, blocked):
    """Return metrics dict."""
    y_true = [it['gt_underage'] for it in items if it[scores] is not None and it[blocked] is not None]
    y_pred = [it[blocked] for it in items if it[scores] is not None and it[blocked] is not None]
    y_score = [it[scores] for it in items if it[scores] is not None and it[blocked] is not None]
    n_use = len(y_true)
    tp, fp, fn, tn, recall, fpr, prec = confusion(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if roc_auc_score and len(set(y_true)) > 1 else None
    return {
        'name': name, 'n': n_use, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'recall': recall, 'fpr': fpr, 'precision': prec, 'auc': auc,
    }


print('=== HEAD-TO-HEAD ===\n')
print(f'{"Model":<25} {"n":>6} {"TP":>5} {"FP":>5} {"FN":>5} {"TN":>5} {"Recall":>8} {"FPR":>7} {"AUC":>7}')
print('-' * 88)
models = [
    ('V8pas80-v2 @ thr=0.30',    'our_score',      'our_blocked'),
    ('Tom K=30 @ thr=0.10',      'tom_k30_score',  'tom_k30_blocked'),
    ('Tom prod (detect.ts)',     'tom_prod_score', 'tom_prod_blocked'),
]
results = {}
for name, sk, bk in models:
    m = evaluate(name, sk, bk)
    results[name] = m
    auc = f'{m["auc"]:.4f}' if m['auc'] else '   —  '
    print(f'{name:<25} {m["n"]:>6} {m["TP"]:>5} {m["FP"]:>5} {m["FN"]:>5} {m["TN"]:>5} {m["recall"]*100:>7.2f}% {m["fpr"]*100:>6.2f}% {auc:>7}')

print()
print('Tom\'s K=30 paper headline (from README): Recall=88.0%, FPR=5.55%, AUC=0.97')
print('Ground truth: qwen3.min_age ≤ 14 (Qwen3-VL face age, not human-annotated)\n')

# Threshold sweep for our V8pas80-v2
print('=== V8pas80-v2 — threshold sweep ===\n')
print(f'{"THR":>5} {"TP":>5} {"FP":>5} {"FN":>5} {"TN":>5} {"Recall":>8} {"FPR":>7}')
print('-' * 50)
sweep = []
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    y_true = [it['gt_underage'] for it in items if it['our_score'] is not None]
    y_pred = [it['our_score'] >= thr for it in items if it['our_score'] is not None]
    tp, fp, fn, tn, rec, fpr, _ = confusion(y_true, y_pred)
    print(f'{thr:>5.2f} {tp:>5} {fp:>5} {fn:>5} {tn:>5} {rec*100:>7.2f}% {fpr*100:>6.2f}%')
    sweep.append({'thr': thr, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'recall': rec, 'fpr': fpr})

# Disagreement list — where V8pas80-v2 blocks but Tom K30 doesn't (or vice versa)
print('\n=== DISAGREEMENTS (V8pas80-v2 @0.30 vs K30 @0.10) ===\n')
disagree = []
for it in items:
    if it['our_blocked'] is None or it['tom_k30_blocked'] is None: continue
    if it['our_blocked'] != it['tom_k30_blocked']:
        disagree.append(it)

# Subcategories
ours_blocks_tom_doesnt = [d for d in disagree if d['our_blocked'] and not d['tom_k30_blocked']]
tom_blocks_we_dont      = [d for d in disagree if not d['our_blocked'] and d['tom_k30_blocked']]
print(f'Total disagreements: {len(disagree)}')
print(f'  V8pas80-v2 BLOCKS, K30 PASSES: {len(ours_blocks_tom_doesnt)}')
print(f'  V8pas80-v2 PASSES, K30 BLOCKS: {len(tom_blocks_we_dont)}\n')

# Breakdown by ground truth
def breakdown(group, label):
    minor = sum(1 for d in group if d['gt_underage'])
    adult = sum(1 for d in group if not d['gt_underage'])
    print(f'  {label}: {len(group)}  (underage={minor}, adult={adult})')

print('By ground truth:')
breakdown(ours_blocks_tom_doesnt, 'V8 blocks, K30 passes')
breakdown(tom_blocks_we_dont,      'V8 passes, K30 blocks')

# Save samples for human review
samples = {
    'v8_blocks_k30_passes_underage_gt': [d for d in ours_blocks_tom_doesnt if d['gt_underage']][:50],
    'v8_blocks_k30_passes_adult_gt':    [d for d in ours_blocks_tom_doesnt if not d['gt_underage']][:50],
    'v8_passes_k30_blocks_underage_gt': [d for d in tom_blocks_we_dont if d['gt_underage']][:50],
    'v8_passes_k30_blocks_adult_gt':    [d for d in tom_blocks_we_dont if not d['gt_underage']][:50],
}
out_dis = DATA / 'k30_vs_v8pas80_v2_disagreements.json'
out_dis.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
print(f'\nSaved 4 × up-to-50 disagreement samples → {out_dis}')

# Full report JSON
report = {
    'dataset': "Tom's K=30 LS eval set (project 5, 6675 images)",
    'ground_truth': 'qwen3.min_age <= 14',
    'evaluable_items': n,
    'pos_count': n_pos,
    'neg_count': n_neg,
    'models': results,
    'v8pas80_v2_thr_sweep': sweep,
    'disagreement_counts': {
        'total': len(disagree),
        'v8_blocks_k30_passes': len(ours_blocks_tom_doesnt),
        'v8_passes_k30_blocks': len(tom_blocks_we_dont),
        'v8_blocks_k30_passes_underage_gt': sum(1 for d in ours_blocks_tom_doesnt if d['gt_underage']),
        'v8_blocks_k30_passes_adult_gt':    sum(1 for d in ours_blocks_tom_doesnt if not d['gt_underage']),
        'v8_passes_k30_blocks_underage_gt': sum(1 for d in tom_blocks_we_dont if d['gt_underage']),
        'v8_passes_k30_blocks_adult_gt':    sum(1 for d in tom_blocks_we_dont if not d['gt_underage']),
    },
}
out_rep = DATA / 'k30_vs_v8pas80_v2_report.json'
out_rep.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f'Saved full report → {out_rep}')
