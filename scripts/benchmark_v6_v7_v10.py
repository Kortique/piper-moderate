#!/usr/bin/env python3
"""
benchmark_v6_v7_v10.py
----------------------
Single-shot eval of V6 / V7 / V10 over the 2803 eval set (2212 LS + 591 Grafana 17-May).

Inputs:
  - V6 (312 features) and V7 (314 features): scored on OLD SigLIP labels stored in
    qwen3_age_results.json (LS) and gallery.db.grafana_pool (Grafana, batches 2026-05-17 only).
  - V10 (510 features): scored on NEW (rescored) SigLIP labels in
    ls_holdout_rescored.json + eval616_rescored.json.

Blocking rule:  lgbm >= THR  (LGBM-only, minor ignored — per user request).
Targets:        child recall >= 95%, teen recall >= 80%, adult FPR <= 20%.
Threshold sweep:  0.30 to 0.70 step 0.05.

Output:
  data/benchmark_v6v7v10_results.json   — raw scores per item
  data/benchmark_v6v7v10_summary.json   — sweep + best THR
  prints a human-readable table to stdout
"""
import json, sqlite3, struct, os, tempfile, ast
from pathlib import Path
import numpy as np
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'


def open_db():
    raw = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', raw, 28, len(raw) // 4096)
    fd, tmp = tempfile.mkstemp(suffix='.db', dir='/tmp')
    os.close(fd)
    Path(tmp).write_bytes(bytes(raw))
    conn = sqlite3.connect(tmp)
    conn.row_factory = sqlite3.Row
    return conn, tmp


def sanitize_feat(name):
    return name.replace(':', '_x').replace('"', '').replace("'", '').replace('{', '').replace('}', '')


def build_X(items, feat_names):
    idx = {f: i for i, f in enumerate(feat_names)}
    X = np.zeros((len(items), len(feat_names)), dtype=np.float32)
    for i, it in enumerate(items):
        for k, v in (it.get('underage_labels') or {}).items():
            fk = sanitize_feat(k)
            if fk in idx: X[i, idx[fk]] = float(v)
        for k, v in (it.get('adult_labels') or {}).items():
            fk = 'adult__' + sanitize_feat(k)
            if fk in idx: X[i, idx[fk]] = float(v)
        for k, v in (it.get('no_underage_labels') or {}).items():
            fk = 'no_underage__' + sanitize_feat(k)
            if fk in idx: X[i, idx[fk]] = float(v)
    return X


def load_v6v7_inputs():
    """Old-tag scores: qwen3_age_results.json (LS) + gallery.db (Grafana 17-May).
    Returns list of items: {id, label, source, underage_labels, adult_labels}.
    """
    items = []

    # LS
    raw = (BASE / 'qwen3_age_results.json').read_bytes().rstrip(b'\x00').decode('utf-8')
    qd = json.loads(raw)
    for v in qd.values():
        cat = v.get('category')
        if cat not in ('child', 'teen', 'adult'): continue
        sd = v.get('siglip2_details')
        if isinstance(sd, str):
            try: sd = ast.literal_eval(sd)
            except: sd = None
        det = (sd or {}).get('underage') or {}
        if not det: continue
        labels = det.get('labels') or {}
        items.append({
            'id': 'ls_' + str(v['task_id']),
            'label': cat,
            'source': 'ls',
            'minor': det.get('minor', 0),
            'adult': det.get('adult', 0),
            'underage_labels': labels.get('underage') or {},
            'adult_labels':    labels.get('adult') or {},
            'no_underage_labels': labels.get('no_underage') or {},
        })

    # Grafana 17-May (исключаем 317-session)
    conn, tmp = open_db()
    rows = conn.execute("""
        SELECT id, label, piper_result FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
        AND export_batch != '2026-05-20 UTC'
    """).fetchall()
    conn.close()
    os.unlink(tmp)
    for r in rows:
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage') or {}
            if not det: continue
            labels = det.get('labels') or {}
            items.append({
                'id': r['id'], 'label': r['label'], 'source': 'graf17',
                'minor': det.get('minor', 0), 'adult': det.get('adult', 0),
                'underage_labels': labels.get('underage') or {},
                'adult_labels':    labels.get('adult') or {},
                'no_underage_labels': labels.get('no_underage') or {},
            })
        except: pass
    return items


def load_v10_inputs():
    """New-tag (rescored) scores from ls_holdout_rescored.json + eval616_rescored.json."""
    items = []
    seen = set()

    def add(rec, source):
        if rec.get('id') in seen: return
        seen.add(rec['id'])
        items.append({
            'id': rec['id'], 'label': rec['label'], 'source': source,
            'minor': rec.get('minor', 0), 'adult': rec.get('adult', 0),
            'underage_labels': rec.get('underage_labels') or {},
            'adult_labels':    rec.get('adult_labels') or {},
            'no_underage_labels': rec.get('no_underage_labels') or {},
        })

    # ls_holdout_rescored.json
    lsr = json.loads((DATA / 'ls_holdout_rescored.json').read_text())
    for r in lsr:
        if not r.get('done'): continue
        if r.get('label') not in ('child','teen','adult'): continue
        add({**r, 'id': 'ls_' + str(r['task_id'])}, 'ls')

    # eval616_rescored.json
    e616 = json.loads((DATA / 'eval616_rescored.json').read_text())
    # exclude 317-session
    for r in e616:
        if not r.get('done'): continue
        if r.get('label') not in ('child','teen','adult'): continue
        if r.get('export_batch') == '2026-05-20 UTC': continue
        source = 'ls' if r.get('kind') == 'ls' else 'graf17'
        add(r, source)

    return items


def score(items, model_path, feat_path):
    feat_names = json.loads(Path(feat_path).read_text())
    booster = lgb.Booster(model_file=str(model_path))
    X = build_X(items, feat_names)
    return booster.predict(X)


def metrics(items, preds, thr):
    by_label = {'child': [], 'teen': [], 'adult': []}
    for it, p in zip(items, preds):
        if it['label'] in by_label:
            by_label[it['label']].append(p >= thr)
    n_c = len(by_label['child']); n_t = len(by_label['teen']); n_a = len(by_label['adult'])
    cr = 100 * sum(by_label['child']) / n_c if n_c else 0
    tr = 100 * sum(by_label['teen'])  / n_t if n_t else 0
    fpr= 100 * sum(by_label['adult']) / n_a if n_a else 0
    return cr, tr, fpr, n_c, n_t, n_a


def find_best_thr(items, preds, thrs):
    """Best THR: prefer all three targets met; if none, minimise constraint violation."""
    targets = {'child': 95.0, 'teen': 80.0, 'fpr': 20.0}
    best = None
    for thr in thrs:
        cr, tr, fpr, _, _, _ = metrics(items, preds, thr)
        meets = cr >= targets['child'] and tr >= targets['teen'] and fpr <= targets['fpr']
        # score for ranking: penalty for each violation
        violation = max(0, targets['child']-cr) + max(0, targets['teen']-tr) + max(0, fpr-targets['fpr'])
        cand = (meets, -violation, cr, tr, -fpr, thr)
        if best is None or cand > best:
            best = cand
    return best[-1], best[2], best[3], -best[4], best[0]  # thr, cr, tr, fpr, meets


def main():
    print('Loading old-tag inputs (V6/V7)...', flush=True)
    old_items = load_v6v7_inputs()
    print(f'  V6/V7 input items: {len(old_items)}', flush=True)
    print(f'    by label: child={sum(1 for x in old_items if x["label"]=="child")}, '
          f'teen={sum(1 for x in old_items if x["label"]=="teen")}, '
          f'adult={sum(1 for x in old_items if x["label"]=="adult")}', flush=True)
    print(f'    by source: ls={sum(1 for x in old_items if x["source"]=="ls")}, '
          f'graf17={sum(1 for x in old_items if x["source"]=="graf17")}', flush=True)

    print('Loading new-tag inputs (V10)...', flush=True)
    new_items = load_v10_inputs()
    print(f'  V10 input items: {len(new_items)}', flush=True)
    print(f'    by label: child={sum(1 for x in new_items if x["label"]=="child")}, '
          f'teen={sum(1 for x in new_items if x["label"]=="teen")}, '
          f'adult={sum(1 for x in new_items if x["label"]=="adult")}', flush=True)

    # Scoring
    print('Scoring V6...', flush=True)
    pred_v6 = score(old_items, DATA / 'lgbm_underage_v6.txt', DATA / 'lgbm_v6_features.json')
    print('Scoring V7...', flush=True)
    pred_v7 = score(old_items, DATA / 'lgbm_underage_v7.txt', DATA / 'lgbm_v7_features.json')
    print('Scoring V10...', flush=True)
    pred_v10 = score(new_items, DATA / 'lgbm_underage_v10.txt', DATA / 'lgbm_v10_features.json')

    thrs = [round(0.10 + 0.05 * i, 2) for i in range(13)]  # 0.10..0.70 step 0.05

    # Best THR
    def best(name, items, preds):
        thr, cr, tr, fpr, meets = find_best_thr(items, preds, thrs)
        return {'model': name, 'best_thr': thr, 'child_recall': cr, 'teen_recall': tr,
                'adult_fpr': fpr, 'targets_met': meets, 'n': len(items)}

    summary = {
        'best': [best('V6', old_items, pred_v6),
                 best('V7', old_items, pred_v7),
                 best('V10', new_items, pred_v10)],
        'sweep': {'thrs': thrs, 'models': {}},
    }
    for name, items, preds in [('V6', old_items, pred_v6),
                               ('V7', old_items, pred_v7),
                               ('V10', new_items, pred_v10)]:
        rows = []
        # overall
        for thr in thrs:
            cr, tr, fpr, n_c, n_t, n_a = metrics(items, preds, thr)
            rows.append({'thr': thr, 'overall': {'child': cr, 'teen': tr, 'fpr': fpr,
                                                 'n_c': n_c, 'n_t': n_t, 'n_a': n_a}})
            # per-source
            for source in ('ls', 'graf17'):
                sub_items = [x for x in items if x['source'] == source]
                sub_preds = [p for x, p in zip(items, preds) if x['source'] == source]
                if not sub_items: continue
                cr_s, tr_s, fpr_s, nc, nt, na = metrics(sub_items, sub_preds, thr)
                rows[-1][source] = {'child': cr_s, 'teen': tr_s, 'fpr': fpr_s,
                                    'n_c': nc, 'n_t': nt, 'n_a': na}
        summary['sweep']['models'][name] = rows

    def to_py(o):
        import numpy as _np
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_py(x) for x in o]
        if isinstance(o, (_np.bool_, bool)): return bool(o)
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        return o
    (DATA / 'benchmark_v6v7v10_summary.json').write_text(json.dumps(to_py(summary), indent=2))
    print(f'\nSaved: data/benchmark_v6v7v10_summary.json', flush=True)

    # Print compact table
    print('\n' + '='*72)
    print('BEST THR PER MODEL — targets: child>=95% teen>=80% fpr<=20% (LGBM-only)')
    print('='*72)
    print(f'{"Model":>6} {"Best THR":>9} {"Child%":>8} {"Teen%":>8} {"FPR%":>8} {"Met?":>6} {"N":>6}')
    for b in summary['best']:
        print(f'{b["model"]:>6} {b["best_thr"]:>9.2f} {b["child_recall"]:>7.1f}% '
              f'{b["teen_recall"]:>7.1f}% {b["adult_fpr"]:>7.1f}% '
              f'{"YES" if b["targets_met"] else "no":>6} {b["n"]:>6}')

    print('\n' + '='*72)
    print('DETAILED SWEEP — overall (LS + Grafana-17-May combined)')
    print('='*72)
    header = f'{"THR":>5}'
    for name in ('V6', 'V7', 'V10'):
        header += f'   {name+" child%":>10} {"teen%":>7} {"fpr%":>6}'
    print(header)
    print('-' * len(header))
    for i, thr in enumerate(thrs):
        row = f'{thr:>5.2f}'
        for name in ('V6', 'V7', 'V10'):
            o = summary['sweep']['models'][name][i]['overall']
            ok_c = 'v' if o['child'] >= 95 else ' '
            ok_t = 'v' if o['teen']  >= 80 else ' '
            ok_f = 'v' if o['fpr']  <= 20 else ' '
            row += f'   {o["child"]:>8.1f}{ok_c} {o["teen"]:>5.1f}{ok_t} {o["fpr"]:>4.1f}{ok_f}'
        print(row)


if __name__ == '__main__':
    main()
