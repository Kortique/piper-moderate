#!/usr/bin/env python3
"""
report_new_grafana.py
---------------------
After fetching new Grafana items and scoring them through all pipelines,
this script aggregates how V6c / V8cs80 / V11cs80 / Tom K30 scored them.

It identifies "new" Grafana items as those imported after the cutoff timestamp
(or all items added in the latest export_batch).

Usage:
    python scripts/report_new_grafana.py --since 2026-05-26
    python scripts/report_new_grafana.py --batch latest
    python scripts/report_new_grafana.py --top-fps 20
"""
import argparse, json, sqlite3, struct, sys
from pathlib import Path
import lightgbm as lgb
import re

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
sys.path.insert(0, str(BASE / 'scripts'))
from bci_taxonomy import BODY_LABELS, CONTEXT_LABELS, INTERACTION_LABELS

X20_RE = re.compile(r':x(\d+(?:\.\d+)?)$')
BCI = ['_child_body','_child_context','_child_interaction','_body_vs_context']

def unmult(k, v):
    m = X20_RE.search(k)
    if not m: return float(v)
    mu = float(m.group(1))
    return 0.999/mu if v >= 0.999 else float(v)/mu
def strip_mult(k): return X20_RE.sub('', k)
def sanitize(n): return n.replace(':','_x').replace('"','').replace("'",'').replace('{','').replace('}','')
def noisy_or(d, st):
    p = 1.0
    for k, v in d.items():
        if k in st: p *= 1.0 - float(v)
    return 1.0 - p

def score_v6(model, feats, u, a):
    idx = {f: i for i, f in enumerate(feats)}
    vec = [0.0]*len(feats)
    for k, v in (u or {}).items():
        if k in idx: vec[idx[k]] = float(v)
    for k, v in (a or {}).items():
        fk = 'adult__' + k
        if fk in idx: vec[idx[fk]] = float(v)
    return float(model.predict([vec])[0])

def score_bci(model, feats, u_raw, a_raw, nu_raw=None):
    idx = {f: i for i, f in enumerate(feats)}
    vec = [0.0]*len(feats)
    u = {}
    for k, v in (u_raw or {}).items():
        raw = unmult(k, v); k2 = strip_mult(k)
        if k2 not in u or u[k2] < raw: u[k2] = raw
    for k, v in u.items():
        fk = sanitize(k)
        if fk in idx: vec[idx[fk]] = v
    a = dict(a_raw or {})
    if nu_raw:
        for k, v in nu_raw.items():
            a[k] = max(a.get(k, 0.0), float(v))
    for k, v in a.items():
        fk = 'adult__' + sanitize(k)
        if fk in idx: vec[idx[fk]] = float(v)
    body = noisy_or(u, BODY_LABELS); ctx = noisy_or(u, CONTEXT_LABELS); inter = noisy_or(u, INTERACTION_LABELS)
    bc = body + ctx
    if '_child_body' in idx: vec[idx['_child_body']] = body
    if '_child_context' in idx: vec[idx['_child_context']] = ctx
    if '_child_interaction' in idx: vec[idx['_child_interaction']] = inter
    if '_body_vs_context' in idx: vec[idx['_body_vs_context']] = (body/bc) if bc>0 else 0
    return float(model.predict([vec])[0])

def open_db():
    db = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', db, 28, len(db)//4096)
    import tempfile
    tmp = Path(tempfile.gettempdir()) / '_rpt.db'
    tmp.write_bytes(bytes(db))
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default=None, help='ISO timestamp (e.g. 2026-05-26) to include items exported after')
    ap.add_argument('--batch', default=None, help='Specific export_batch value, or "latest"')
    ap.add_argument('--top-fps', type=int, default=10, help='How many top FPs to show per model')
    ap.add_argument('--thresholds', default='data/thresholds.json')
    args = ap.parse_args()

    thrs = json.loads(Path(args.thresholds).read_text())
    print(f'Thresholds: {thrs}\n')

    conn = open_db()
    # Determine the cohort
    if args.batch == 'latest':
        r = conn.execute("SELECT MAX(export_batch) FROM grafana_pool").fetchone()
        args.batch = r[0]
        print(f'Latest export_batch: {args.batch!r}')
    if args.batch:
        rows = conn.execute("""SELECT id, label, label_confirmed, piper_result, prompt
                                FROM grafana_pool
                                WHERE export_batch=? AND (deleted IS NULL OR deleted=0)""",
                            (args.batch,)).fetchall()
    elif args.since:
        rows = conn.execute("""SELECT id, label, label_confirmed, piper_result, prompt
                                FROM grafana_pool
                                WHERE exported_at >= ? AND (deleted IS NULL OR deleted=0)""",
                            (args.since,)).fetchall()
    else:
        print('Either --since or --batch is required', file=sys.stderr); sys.exit(1)

    print(f'New Grafana cohort: {len(rows)} items\n')
    if not rows:
        print('Nothing to score.'); return

    # Load models
    b6 = lgb.Booster(model_file=str(DATA / 'lgbm_underage_v6c.txt'))
    f6 = json.loads((DATA / 'lgbm_v6c_features.json').read_text())
    b8 = lgb.Booster(model_file=str(DATA / 'lgbm_underage_v8cs80.txt'))
    f8 = json.loads((DATA / 'lgbm_v8cs80_features.json').read_text())
    b11 = lgb.Booster(model_file=str(DATA / 'lgbm_underage_v11cs80.txt'))
    f11 = json.loads((DATA / 'lgbm_v11cs80_features.json').read_text())

    # Pull rescore dumps for V11 and Tom (these were filled by remote scoring scripts)
    v11_map = {r['id']: r for r in json.loads((DATA / 'v11_native_scores.json').read_text()) if r.get('done')}
    tom_map = {r['id']: r for r in json.loads((DATA / 'tom_scores.json').read_text()) if r.get('done')}

    # Score each item with all 4 models
    scores = {'v6': {}, 'v8': {}, 'v11': {}, 'tom': {}}
    labels = {}
    item_min_age = {}   # rid -> minimum age from qwen3 or face_detect
    missing_piper = 0
    missing_v11 = 0
    missing_tom = 0
    no_face_v11 = 0
    no_face_tom = 0
    for row in rows:
        rid = row['id']
        labels[rid] = row['label']
        # V6 / V8 from local piper_result siglip2_details
        try:
            pr = json.loads(row['piper_result']) if row['piper_result'] else {}
        except Exception:
            pr = {}
        det = (pr.get('siglip2_details') or {}).get('underage', {})
        siglip_labels = det.get('labels', {})
        u = siglip_labels.get('underage') or {}
        a = siglip_labels.get('adult') or {}
        if u or a:
            scores['v6'][rid]  = score_v6(b6, f6, u, a)
            scores['v8'][rid]  = score_bci(b8, f8, u, a)
        else:
            missing_piper += 1
        # Min age: qwen3.faces[].ageFrom (min), fallback to face_detect.ageFrom
        ages = []
        # qwen3_details may be under piper_result.qwen3_details or top-level qwen3_result via a separate column
        q3 = pr.get('qwen3_details') or {}
        for f in (q3.get('faces') or []):
            af = f.get('ageFrom') if isinstance(f, dict) else None
            if isinstance(af, (int, float)): ages.append(af)
        if not ages:
            fd = pr.get('face_detect_result') or {}
            af = fd.get('ageFrom')
            if isinstance(af, (int, float)): ages.append(af)
        if ages:
            item_min_age[rid] = min(ages)
        # V11 from native scores
        v11_rec = v11_map.get(rid)
        if v11_rec:
            if v11_rec.get('no_face'):
                no_face_v11 += 1
            else:
                u11 = v11_rec.get('underage_labels') or {}
                a11 = v11_rec.get('adult_labels') or {}
                nu = v11_rec.get('no_underage_labels') or {}
                if u11 or a11:
                    scores['v11'][rid] = score_bci(b11, f11, u11, a11, nu)
        else:
            missing_v11 += 1
        # Tom from rescore dump
        tom_rec = tom_map.get(rid)
        if tom_rec:
            k30 = tom_rec.get('k30_score')
            if k30 is not None:
                scores['tom'][rid] = float(k30)
        else:
            missing_tom += 1

    print(f'Coverage:')
    print(f'  V6/V8  (piper_result):    {len(scores["v6"])}/{len(rows)}  (missing piper_result: {missing_piper})')
    print(f'  V11cs80 (native):         {len(scores["v11"])}/{len(rows)}  (missing rescore: {missing_v11}, no_face: {no_face_v11})')
    print(f'  Tom K30 (rescore):        {len(scores["tom"])}/{len(rows)}  (missing rescore: {missing_tom}, no_face: {no_face_tom})')

    # === Model verdict → status ===
    # Rule (per Ivan's spec):
    #   score < threshold              → adult  (model passed; age IGNORED)
    #   score ≥ threshold  +  min_age ≤ 14   → child
    #   score ≥ threshold  +  min_age > 14   → teen
    #   score ≥ threshold  +  no age info   → teen
    # min_age = min qwen3.faces[].ageFrom; fallback face_detect.ageFrom
    verdict = {m: {'child': [], 'teen': [], 'adult': []} for m in scores}
    print(f'\n=== Model verdict → status ({len(rows)} items in cohort) ===')
    print(f'{"model":<8} {"thr":>5}  {"n":>6}  {"child":>14}  {"teen":>14}  {"adult":>14}')
    print('-'*75)
    for model_key, thr_key in [('v6','v6'), ('v8','v8'), ('v11','v11'), ('tom','k30tom')]:
        thr = thrs.get(thr_key, 0.30)
        sc = scores[model_key]
        n = len(sc)
        if n == 0:
            print(f'  {model_key:<6} {thr:>5}  {n:>6}  {"—":>14}  {"—":>14}  {"—":>14}')
            continue
        for rid, s in sc.items():
            if s < thr:
                verdict[model_key]['adult'].append(rid)
            else:
                age = item_min_age.get(rid)
                if age is not None and age <= 14:
                    verdict[model_key]['child'].append(rid)
                else:
                    verdict[model_key]['teen'].append(rid)
        def fmt(k):
            v = len(verdict[model_key][k])
            return f'{v:>4} ({v/n*100:>5.1f}%)' if n else '—'
        print(f'  {model_key:<6} {thr:>5}  {n:>6}  {fmt("child"):>14}  {fmt("teen"):>14}  {fmt("adult"):>14}')

    # Save full breakdown for downstream consumption
    out = {
        'cohort_size': len(rows),
        'thresholds': thrs,
        'scores': {m: {i: round(s, 4) for i, s in d.items()} for m, d in scores.items()},
        'verdict': {m: {k: vs for k, vs in vv.items()} for m, vv in verdict.items()},
    }
    out_path = DATA / f'grafana_cohort_scores_{args.batch or args.since}.json'.replace(':', '-').replace(' ', '_')
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f'\nSaved cohort scores → {out_path.name}')


if __name__ == '__main__':
    main()
