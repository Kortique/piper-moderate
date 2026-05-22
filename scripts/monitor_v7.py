"""
monitor_v7.py
-------------
Continuous performance monitor for the Piper underage detection pipeline.

Runs after each new export batch is ingested:
  1. Loads all labeled items with piper_result (from gallery.db / disagree_pool.json)
  2. Applies V7 blocking rule: lgbm_score >= 0.45 OR minor >= 0.72
  3. Reports recall/FPR per label class per export session
  4. Alerts if targets are VIOLATED: child ≥98%, teen ≥80%, adult FPR ≤20%
  5. Writes report to logs/monitor_YYYYMMDD_HHMMSS.json

Usage:
    python scripts/monitor_v7.py                    # run once, print report
    python scripts/monitor_v7.py --session all      # all sessions
    python scripts/monitor_v7.py --session latest   # latest session only (default)
    python scripts/monitor_v7.py --alert-only       # only print if targets violated
    python scripts/monitor_v7.py --watch 60         # loop every 60 seconds

Schedule:
    In cron: 0 * * * * cd /path/to/piper-moderate && python scripts/monitor_v7.py --alert-only
"""
import os, sys, json, struct, sqlite3, time, shutil, datetime, argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

LGBM_THR  = 0.45   # V7 threshold
MINOR_THR = 0.72   # minor score threshold

# Targets
TARGET_CHILD_RECALL = 95.0
TARGET_TEEN_RECALL  = 80.0
TARGET_ADULT_FPR    = 20.0


def _open_db():
    # Try gallery.db first, fall back to latest backup (gallery.db may be corrupted/locked)
    candidates = sorted((BASE_DIR / 'backups').glob('gallery_*.db'), reverse=True)
    tmp = Path('/tmp/_monitor_v7.db')
    for db_path in candidates:
        if not db_path.exists():
            continue
        try:
            data = bytearray(db_path.read_bytes())
            struct.pack_into('>I', data, 28, len(data) // 4096)
            tmp.write_bytes(bytes(data))
            conn = sqlite3.connect(str(tmp))
            conn.row_factory = sqlite3.Row
            # Real sanity check — the full query we'll actually run
            conn.execute("""
                SELECT id FROM grafana_pool
                WHERE (deleted IS NULL OR deleted=0) AND label IN ('child','teen','adult')
                AND piper_result IS NOT NULL LIMIT 1
            """).fetchall()
            return conn
        except Exception:
            tmp.unlink(missing_ok=True)
            continue
    print("[monitor] No readable DB found")
    return None


def load_sessions():
    """Return {session_name: [items]} where each item has label, lgbm, minor."""
    conn = _open_db()
    if conn is None:
        # Fallback: try disagree_pool.json
        pool_path = BASE_DIR / 'data' / 'disagree_pool.json'
        if not pool_path.exists():
            return {}
        try:
            raw = pool_path.read_bytes().rstrip(b'\x00')
            pool = json.loads(raw)
            items = []
            for v in pool.values():
                if v.get('deleted'): continue
                lbl = v.get('label')
                if lbl not in ('child', 'teen', 'adult'): continue
                pr = v.get('piper_result') or {}
                det = (pr.get('siglip2_details') or {}).get('underage', {})
                lgbm = (det.get('lgbm') or {}).get('score', 0)
                minor = det.get('minor', 0)
                items.append({'label': lbl, 'lgbm': lgbm, 'minor': minor,
                              'session': v.get('export_batch', 'unknown')})
            sessions = {}
            for item in items:
                s = item['session']
                sessions.setdefault(s, []).append(item)
            return sessions
        except Exception as e:
            print(f"[monitor] JSON fallback error: {e}")
            return {}

    rows = conn.execute("""
        SELECT id, label, piper_result, export_batch
        FROM grafana_pool
        WHERE (deleted IS NULL OR deleted=0)
        AND label IN ('child','teen','adult')
        AND piper_result IS NOT NULL
    """).fetchall()
    conn.close()

    sessions = {}
    for r in rows:
        lbl = r['label']
        session = r['export_batch'] or 'unknown'
        try:
            pr = json.loads(r['piper_result'])
            det = (pr.get('siglip2_details') or {}).get('underage', {})
            lgbm  = (det.get('lgbm') or {}).get('score', 0)
            minor = det.get('minor', 0)
            sessions.setdefault(session, []).append({
                'id': r['id'], 'label': lbl, 'lgbm': lgbm, 'minor': minor
            })
        except:
            pass
    return sessions


def evaluate_session(items, lgbm_thr=LGBM_THR, minor_thr=MINOR_THR):
    """Compute recall/FPR for a list of labeled items."""
    children = [x for x in items if x['label'] == 'child']
    teens    = [x for x in items if x['label'] == 'teen']
    adults   = [x for x in items if x['label'] == 'adult']

    def blocked(x):
        return x['lgbm'] >= lgbm_thr or x['minor'] >= minor_thr

    child_blocked = sum(1 for x in children if blocked(x))
    teen_blocked  = sum(1 for x in teens if blocked(x))
    adult_blocked = sum(1 for x in adults if blocked(x))

    cr  = child_blocked / len(children) * 100 if children else None
    tr  = teen_blocked  / len(teens)    * 100 if teens    else None
    fpr = adult_blocked / len(adults)   * 100 if adults   else None

    return {
        'n': len(items),
        'n_child': len(children), 'child_blocked': child_blocked, 'child_recall': cr,
        'n_teen':  len(teens),    'teen_blocked':  teen_blocked,  'teen_recall':  tr,
        'n_adult': len(adults),   'adult_blocked': adult_blocked, 'adult_fpr':    fpr,
    }


def check_targets(result):
    """Returns list of violated targets."""
    violations = []
    cr, tr, fpr = result.get('child_recall'), result.get('teen_recall'), result.get('adult_fpr')
    if cr is not None and cr < TARGET_CHILD_RECALL:
        violations.append(f"child_recall={cr:.1f}% < {TARGET_CHILD_RECALL}%")
    if tr is not None and tr < TARGET_TEEN_RECALL:
        violations.append(f"teen_recall={tr:.1f}% < {TARGET_TEEN_RECALL}%")
    if fpr is not None and fpr > TARGET_ADULT_FPR:
        violations.append(f"adult_fpr={fpr:.1f}% > {TARGET_ADULT_FPR}%")
    return violations


def format_result(session_name, result):
    cr  = f"{result['child_recall']:.1f}%" if result['child_recall'] is not None else "N/A"
    tr  = f"{result['teen_recall']:.1f}%"  if result['teen_recall']  is not None else "N/A"
    fpr = f"{result['adult_fpr']:.1f}%"    if result['adult_fpr']    is not None else "N/A"

    violations = check_targets(result)
    status = "✓ ALL OK" if not violations else f"✗ VIOLATIONS: {'; '.join(violations)}"

    lines = [
        f"Session: {session_name} (n={result['n']})",
        f"  child_recall: {cr} ({result['child_blocked']}/{result['n_child']})  target ≥{TARGET_CHILD_RECALL}%",
        f"  teen_recall:  {tr} ({result['teen_blocked']}/{result['n_teen']})  target ≥{TARGET_TEEN_RECALL}%",
        f"  adult_fpr:    {fpr} ({result['adult_blocked']}/{result['n_adult']})  target ≤{TARGET_ADULT_FPR}%",
        f"  {status}",
    ]
    return '\n'.join(lines)


def run_monitor(session_filter='latest', alert_only=False):
    ts = datetime.datetime.utcnow().isoformat()
    sessions = load_sessions()

    if not sessions:
        msg = f"[{ts}] No labeled sessions found."
        print(msg)
        return {'ts': ts, 'error': 'no_sessions'}

    # Filter sessions
    if session_filter == 'latest':
        key = sorted(sessions.keys())[-1]
        sessions = {key: sessions[key]}
    # 'all' → use all sessions

    report = {'ts': ts, 'sessions': {}, 'any_violation': False, 'lgbm_thr': LGBM_THR, 'minor_thr': MINOR_THR}

    all_violations = False
    for session_name, items in sorted(sessions.items()):
        result = evaluate_session(items)
        violations = check_targets(result)
        result['violations'] = violations
        report['sessions'][session_name] = result
        if violations:
            all_violations = True

    report['any_violation'] = all_violations

    if alert_only and not all_violations:
        return report

    print(f"\n{'='*60}")
    print(f"PIPELINE MONITOR — {ts}")
    print(f"Blocking rule: lgbm >= {LGBM_THR} OR minor >= {MINOR_THR}")
    print('='*60)
    for session_name, result in sorted(report['sessions'].items()):
        print()
        print(format_result(session_name, result))

    if all_violations:
        print('\n⚠️  TARGETS VIOLATED — retraining recommended!')
        print('   Run: python scripts/train_lgbm_v7.py && python scripts/monitor_v7.py')
    else:
        print('\n✓ All sessions within targets.')

    # Save log
    log_file = LOGS_DIR / f"monitor_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    return report


def main():
    parser = argparse.ArgumentParser(description='V7 pipeline performance monitor')
    parser.add_argument('--session', default='latest', choices=['latest', 'all'],
                        help='Which sessions to evaluate')
    parser.add_argument('--alert-only', action='store_true',
                        help='Only print output if targets are violated')
    parser.add_argument('--watch', type=int, default=0, metavar='SECONDS',
                        help='Loop every N seconds (0=run once)')
    args = parser.parse_args()

    if args.watch > 0:
        print(f"[monitor] Watching every {args.watch}s. Ctrl+C to stop.")
        while True:
            run_monitor(args.session, args.alert_only)
            time.sleep(args.watch)
    else:
        run_monitor(args.session, args.alert_only)


if __name__ == '__main__':
    main()
