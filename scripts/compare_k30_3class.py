#!/usr/bin/env python3
"""
3-class comparison V8pas80-v2 vs Tom K=30 on Tom's 6675-image dataset.

Categories from qwen3:
  child  = qwen3.min_age ≤ 14
  teen   = 15 ≤ qwen3.min_age ≤ 17
  adult  = qwen3.min_age ≥ 18

Each model is binary (blocked / not blocked). Per-class metrics:
  - recall(child) = blocked / total_child   (want HIGH)
  - recall(teen)  = blocked / total_teen    (we want high, Tom doesn't aim for this)
  - FPR(adult)    = blocked / total_adult   (want LOW)

Also AUC for:
  - child vs (teen+adult)     — Tom's training task
  - (child+teen) vs adult     — our training task (≤17 vs ≥18)
"""
import json
from pathlib import Path
from collections import Counter
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'

# Load JSONL
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

# Categorize
def categorize(min_age):
    if min_age is None: return None
    if min_age <= 14: return 'child'
    if min_age <= 17: return 'teen'
    return 'adult'


items = []
for r in records.values():
    age = r.get('qwen_min_age')
    cat = categorize(age)
    if cat is None: continue
    our = r.get('ours', {})
    if 'error' in our: continue
    if our.get('lgbm_score') is None or r.get('k30_score') is None: continue
    items.append({
        'gid': r['gid'],
        'media': r['media'],
        'category': cat,
        'qwen_min_age': age,
        'qwen_max_age': r.get('qwen_max_age'),
        'our_score':   our.get('lgbm_score'),
        'our_blocked': our.get('lgbm_blocked'),
        'k30_score':   r.get('k30_score'),
        'k30_blocked': r.get('k30_blocked'),
        'k25_score':   r.get('k25_score'),
        'prod_blocked': r.get('prod_blocked'),
    })

n = len(items)
c = Counter(it['category'] for it in items)
print(f'Evaluable items: {n}')
print(f'  child (≤14): {c["child"]}')
print(f'  teen  (15-17): {c["teen"]}')
print(f'  adult (≥18): {c["adult"]}\n')


def per_class_block_rates(blocked_key):
    """blocked rate per category for given model."""
    out = {}
    for cat in ['child', 'teen', 'adult']:
        sub = [it for it in items if it['category'] == cat]
        if not sub:
            out[cat] = (0, 0, 0.0); continue
        n_blocked = sum(1 for it in sub if it[blocked_key])
        out[cat] = (n_blocked, len(sub), n_blocked / len(sub))
    return out


def auc(score_key, positive_cats):
    """Binary AUC where positive = item['category'] in positive_cats."""
    if not roc_auc_score: return None
    y = [1 if it['category'] in positive_cats else 0 for it in items]
    s = [it[score_key] for it in items]
    if len(set(y)) < 2: return None
    return roc_auc_score(y, s)


print('=== PER-CLASS BLOCK RATES ===\n')
print(f'{"Model":<28} {"child blocked":>16} {"teen blocked":>16} {"adult blocked":>16}')
print('-' * 80)
models = [
    ('V8pas80-v2 @ thr=0.30',  'our_blocked',  'our_score'),
    ('Tom K=30 @ thr=0.10',    'k30_blocked',  'k30_score'),
    ('Tom prod (detect.ts)',   'prod_blocked', 'prod_score'),
]
metrics = {}
for name, bk, sk in models:
    rates = per_class_block_rates(bk)
    auc_child = auc(sk, {'child'}) if sk != 'prod_score' else None  # prod has no score field
    auc_under = auc(sk, {'child', 'teen'}) if sk != 'prod_score' else None
    fmt_cell = lambda rt: f'{rt[0]:>4}/{rt[1]:<4} {rt[2]*100:>5.1f}%'
    print(f'{name:<28} {fmt_cell(rates["child"]):>16} {fmt_cell(rates["teen"]):>16} {fmt_cell(rates["adult"]):>16}')
    metrics[name] = {'rates': rates, 'auc_child_vs_rest': auc_child, 'auc_under_vs_adult': auc_under}

print()
print('=== AUC ===\n')
print(f'{"Model":<28} {"AUC child-vs-rest":>20} {"AUC ≤17-vs-≥18":>20}')
print('-' * 72)
for name in ['V8pas80-v2 @ thr=0.30', 'Tom K=30 @ thr=0.10']:
    a1 = metrics[name].get('auc_child_vs_rest')
    a2 = metrics[name].get('auc_under_vs_adult')
    a1s = f'{a1:.4f}' if a1 else '  —   '
    a2s = f'{a2:.4f}' if a2 else '  —   '
    print(f'{name:<28} {a1s:>20} {a2s:>20}')

print()
print('How to read:')
print('  - V8pas80-v2 was trained on ≤17 → it correctly aims for both child AND teen.')
print('  - K=30 was trained on ≤14 only → it doesn\'t aim at teen.')
print('  - "child blocked"  — both models WANT high (catching real underage).')
print('  - "teen blocked"   — V8 WANTS high (15-17 still underage in our def).')
print('                       K30 has no opinion: blocking teens is incidental.')
print('  - "adult blocked"  — both WANT low (false positive rate).')

# Threshold sweep — V8 only, showing both child + teen recall and adult FPR
print('\n=== V8pas80-v2 — per-class threshold sweep ===\n')
print(f'{"THR":>5} {"child blk":>12} {"teen blk":>12} {"adult blk":>12}')
print('-' * 50)
sweep = []
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
    rates = {}
    for cat in ['child', 'teen', 'adult']:
        sub = [it for it in items if it['category'] == cat]
        n_blk = sum(1 for it in sub if it['our_score'] >= thr)
        rates[cat] = (n_blk, len(sub), n_blk / len(sub) if sub else 0)
    print(f'{thr:>5.2f}  {rates["child"][2]*100:>10.1f}% {rates["teen"][2]*100:>10.1f}% {rates["adult"][2]*100:>10.1f}%')
    sweep.append({'thr': thr, **{f'{k}_blocked': v for k, v in rates.items()}})

# K30 threshold sweep
print('\n=== Tom K=30 — per-class threshold sweep ===\n')
print(f'{"THR":>5} {"child blk":>12} {"teen blk":>12} {"adult blk":>12}')
print('-' * 50)
for thr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
    rates = {}
    for cat in ['child', 'teen', 'adult']:
        sub = [it for it in items if it['category'] == cat]
        n_blk = sum(1 for it in sub if it['k30_score'] >= thr)
        rates[cat] = (n_blk, len(sub), n_blk / len(sub) if sub else 0)
    print(f'{thr:>5.2f}  {rates["child"][2]*100:>10.1f}% {rates["teen"][2]*100:>10.1f}% {rates["adult"][2]*100:>10.1f}%')

# Save
report = {
    'dataset': "Tom's K=30 LS eval set (project 5, 6675 images)",
    'ground_truth_categories': {
        'child': 'qwen3.min_age <= 14',
        'teen':  '15 <= qwen3.min_age <= 17',
        'adult': 'qwen3.min_age >= 18',
    },
    'counts': dict(c),
    'evaluable_items': n,
    'metrics_at_default_thr': {k: {'block_rates': v['rates'],
                                    'auc_child_vs_rest': v['auc_child_vs_rest'],
                                    'auc_under_vs_adult': v['auc_under_vs_adult']}
                                for k, v in metrics.items()},
    'v8pas80_v2_thr_sweep_3class': sweep,
}
(DATA / 'k30_3class_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f'\nSaved → data/k30_3class_report.json')
