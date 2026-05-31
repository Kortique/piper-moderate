#!/usr/bin/env python3
"""
check_borderlands_scores.py — diagnostic: how many Borderlands cards are missing
which model's score in the gallery.

Counts per-model gaps:
  • V6 / V8 — fail when piper_result.siglip2_details.underage.labels is missing
              or empty (both underage and adult dicts). Filled by:
                python scripts/moderate_borderlands.py --missing-scores --workers 6
  • V11    — primary source is data/v11_native_scores.json. Items not in this
              file get the gallery's fallback (siglip2-based, marked with *).
              Filled by:
                python scripts/rescore_via_v11.py --source borderlands --workers 8
  • Tom K30 — sourced from data/tom_scores.json. Filled by:
                python scripts/rescore_via_tom.py --source borderlands --workers 8

Prints a per-model "missing N / total M" line. Total time estimate at default
worker counts is included so you can plan the run.
"""
import json, sqlite3, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH         = BASE / 'gallery.db'
V11_NATIVE_PATH = BASE / 'data' / 'v11_native_scores.json'
TOM_PATH        = BASE / 'data' / 'tom_scores.json'


def _safe_load_json_list(p: Path) -> set:
    """Return set of ids in a list-of-dicts JSON (resume cache file).

    If strict UTF-8 parsing fails (common symptom: 'invalid continuation byte'
    in the middle of the file after a killed write), fall back to
    errors='ignore' so we still report a meaningful coverage number, and
    nudge the user to run scripts/recover_json_caches.py to clean it up.
    """
    if not p.exists():
        return set()
    raw = p.read_bytes()
    try:
        data = json.loads(raw.decode('utf-8'))
    except UnicodeDecodeError as e:
        print(f'WARN: {p.name} has corrupt bytes ({e}); '
              f'recovering via errors=ignore. '
              f'Run `python scripts/recover_json_caches.py` to clean.',
              file=sys.stderr)
        try:
            data = json.loads(raw.decode('utf-8', errors='ignore'))
        except Exception as e2:
            print(f'ERR: still unparseable: {e2}', file=sys.stderr)
            return set()
    except Exception as e:
        print(f'WARN: could not parse {p.name}: {e}', file=sys.stderr)
        return set()
    return {r.get('id') for r in data if isinstance(r, dict) and r.get('id')}


def main():
    if not DB_PATH.exists():
        print(f'ERR: {DB_PATH} not found', file=sys.stderr); sys.exit(1)

    # Read-only mode — no lock contention with gallery_server.py
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True, timeout=10)
    conn.row_factory = sqlite3.Row

    # All borderlands items (not deleted)
    all_ids = [r['id'] for r in conn.execute(
        "SELECT id FROM borderlands_pool WHERE deleted IS NULL OR deleted = 0"
    ).fetchall()]
    total = len(all_ids)
    print(f'\n=== Borderlands scoring coverage ({total} items) ===\n')

    # ── V6 / V8: need siglip2_details to exist ─
    # Items where siglip2_details exists but labels are empty are NOT missing —
    # that's a valid result for clean images, and gallery LGBM scores them on a
    # 0-vector. Same logic as moderate_borderlands.py --missing-scores.
    missing_v6v8 = conn.execute("""
        SELECT COUNT(*) FROM borderlands_pool
        WHERE (deleted IS NULL OR deleted = 0)
          AND (
              piper_result IS NULL
              OR json_extract(piper_result, '$.siglip2_details') IS NULL
          )
    """).fetchone()[0]
    have_v6v8 = total - missing_v6v8
    print(f'V6 / V8:  have {have_v6v8:>5} / missing {missing_v6v8:>5}  '
          f'({100*have_v6v8/total:.1f}% covered)')

    # ── V11 native: id present in v11_native_scores.json (regardless of value) ─
    v11_ids = _safe_load_json_list(V11_NATIVE_PATH)
    have_v11_native = sum(1 for iid in all_ids if iid in v11_ids)
    missing_v11_native = total - have_v11_native
    # Note: items NOT in v11_native still get a gallery fallback score from
    # siglip2_details (with `*` marker), so V11 column won't be empty for them
    # — provided V6/V8 condition is satisfied (i.e. siglip2_details exists).
    print(f'V11 native: have {have_v11_native:>5} / missing {missing_v11_native:>5}  '
          f'({100*have_v11_native/total:.1f}% covered)')
    print(f'           (missing → gallery falls back to siglip2-based V11 with * marker)')

    # ── Tom K30: id present in tom_scores.json ─────────────────────────────
    tom_ids = _safe_load_json_list(TOM_PATH)
    have_tom = sum(1 for iid in all_ids if iid in tom_ids)
    missing_tom = total - have_tom
    print(f'Tom K30:  have {have_tom:>5} / missing {missing_tom:>5}  '
          f'({100*have_tom/total:.1f}% covered)')

    conn.close()

    # ── Recommendation ──────────────────────────────────────────────────────
    print(f'\n--- TO FILL THE GAPS ---')
    if missing_v6v8:
        # ~10-15s per item with siglip2-only (qwen3 + face_detect skipped)
        est_min = missing_v6v8 * 12 / 6 / 60
        print(f'V6/V8  ({missing_v6v8}):  python -u scripts/moderate_borderlands.py '
              f'--missing-scores --workers 6')
        print(f'        estimated ~{est_min:.0f} min @ 6 workers')
    if missing_v11_native:
        est_min = missing_v11_native * 8 / 8 / 60
        print(f'V11   ({missing_v11_native}):  python -u scripts/rescore_via_v11.py '
              f'--source borderlands --workers 8')
        print(f'        estimated ~{est_min:.0f} min @ 8 workers')
    if missing_tom:
        est_min = missing_tom * 6 / 8 / 60
        print(f'Tom   ({missing_tom}):  python -u scripts/rescore_via_tom.py '
              f'--source borderlands --workers 8')
        print(f'        estimated ~{est_min:.0f} min @ 8 workers')
    if not (missing_v6v8 or missing_v11_native or missing_tom):
        print('All scores complete — no rescore needed.')
    print()


if __name__ == '__main__':
    main()
