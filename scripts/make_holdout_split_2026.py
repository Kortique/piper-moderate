#!/usr/bin/env python3
"""
make_holdout_split_2026.py — stratified 80/20 split across ALL labelled items
(LS + Grafana + K30), preserving label AND source proportion in test set.

Output: data/v11_test_split_2026.json   { seed, test_ids, by_label, by_source, created_at }
"""
import json, sqlite3, datetime, struct
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
SEED = 1337
TEST_FRAC = 0.20


def open_db():
    db = bytearray((BASE / 'gallery.db').read_bytes())
    struct.pack_into('>I', db, 28, len(db)//4096)
    tmp = Path('/tmp/_split_db.db'); tmp.write_bytes(bytes(db))
    return sqlite3.connect(str(tmp))


def main():
    conn = open_db()
    rows = []
    # LS
    for r in conn.execute("SELECT task_id, age_from FROM ls_images WHERE age_from IS NOT NULL"):
        af = r[1]
        lbl = 'child' if af <= 14 else ('teen' if af <= 17 else 'adult')
        rows.append((f'ls_{r[0]}', lbl, 'ls'))
    # Grafana
    for r in conn.execute("""SELECT id, label FROM grafana_pool
                              WHERE label IS NOT NULL AND label_confirmed=1
                                AND (deleted IS NULL OR deleted=0)
                                AND label IN ('child','teen','adult')"""):
        rows.append((r[0], r[1], 'grafana'))
    # K30
    for r in conn.execute("""SELECT id, label FROM k30_pool
                              WHERE label IS NOT NULL AND label_confirmed=1
                                AND (deleted IS NULL OR deleted=0)
                                AND label IN ('child','teen','adult')"""):
        rows.append((r[0], r[1], 'k30'))
    conn.close()

    print(f'Total labelled items: {len(rows)}')
    from collections import Counter
    print(f'  by label:  {dict(Counter(r[1] for r in rows))}')
    print(f'  by source: {dict(Counter(r[2] for r in rows))}')

    # Stratify by combined (label, source) to preserve both axes in test
    strat = [f'{lbl}|{src}' for (_id, lbl, src) in rows]
    train, test = train_test_split(rows, test_size=TEST_FRAC, stratify=strat, random_state=SEED)

    test_ids = [t[0] for t in test]
    by_lab = Counter(t[1] for t in test)
    by_src = Counter(t[2] for t in test)
    print(f'\nTest set: {len(test)} items ({TEST_FRAC*100:.0f}%)')
    print(f'  by label:  {dict(by_lab)}')
    print(f'  by source: {dict(by_src)}')

    out = {
        'seed': SEED,
        'test_frac': TEST_FRAC,
        'n_total': len(rows),
        'n_test':  len(test),
        'n_train': len(train),
        'test_ids': test_ids,
        'by_label':  dict(by_lab),
        'by_source': dict(by_src),
        'created_at': datetime.datetime.utcnow().isoformat(),
    }
    (DATA / 'v11_test_split_2026.json').write_text(json.dumps(out, indent=2))
    print(f'\nSaved → data/v11_test_split_2026.json')


if __name__ == '__main__':
    main()
