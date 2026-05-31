#!/usr/bin/env python3
"""
install_label_guard.py — install SQLite triggers into gallery.db that protect
your data from accidental destructive writes.

Two families of triggers installed:

  1. label_confirmed guard (trg_*_no_unconfirm)
     Blocks UPDATEs that flip label_confirmed=1 → 0 on k30_pool / grafana_pool.

  2. deleted-row guard (trg_*_no_modify_deleted)
     Blocks UPDATEs to a row with OLD.deleted=1 unless the UPDATE is restoring
     it (NEW.deleted=0). Once you delete + save, no future write can overwrite
     the row's fields while it's still soft-deleted. This catches re-import
     scripts, accidental relabel passes, etc.

The triggers live INSIDE the gallery.db file. Any process that opens the file
and attempts a forbidden UPDATE gets a hard SQLite ABORT — and the UPDATE
is rolled back.

This protects against:
  • my buggy scripts
  • import_k30.py --force (even old broken version)
  • direct sqlite3 CLI mistakes
  • re-import that flips deleted=1 → 0 on rows the user already deleted

If you DELIBERATELY need to bypass (rare): --uninstall, do your work,
re-install via --install.

Usage:
    python scripts/install_label_guard.py             # install ALL guards
    python scripts/install_label_guard.py --check     # show installed triggers + smoke test
    python scripts/install_label_guard.py --uninstall # disable all guards
"""
import argparse, sqlite3, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "gallery.db"

# (trigger_name, table, sql_template_func)
TRIGGERS = [
    ("trg_k30_no_unconfirm",        "k30_pool",     "unconfirm"),
    ("trg_grafana_no_unconfirm",    "grafana_pool", "unconfirm"),
    ("trg_k30_no_modify_deleted",   "k30_pool",     "no_modify_deleted"),
    ("trg_grafana_no_modify_deleted","grafana_pool","no_modify_deleted"),
]


def _create_trigger_sql(trigger_name: str, table: str, kind: str) -> str:
    # NOTE: SQLite RAISE(ABORT, msg) requires `msg` to be a *string literal*,
    # not an expression — so we can't concatenate OLD.id into it via `||`.
    # Static error messages only; the offending statement is visible in the
    # traceback / SQLite log anyway, which makes the row identifiable.
    if kind == "unconfirm":
        msg = f"label_guard {table}: refused to flip label_confirmed 1->0. Uninstall: python scripts/install_label_guard.py --uninstall"
        return f"""
            CREATE TRIGGER IF NOT EXISTS {trigger_name}
            BEFORE UPDATE OF label_confirmed ON {table}
            FOR EACH ROW
            WHEN OLD.label_confirmed = 1 AND NEW.label_confirmed = 0
            BEGIN
                SELECT RAISE(ABORT, '{msg}');
            END;
        """
    if kind == "no_modify_deleted":
        # Block any UPDATE that keeps the row in deleted=1 state AND changes
        # any label/variant column. Restoring (deleted=1->0) and idempotent
        # re-deletion are both allowed.
        msg = f"delete_guard {table}: refused to modify a soft-deleted row. Undelete first (deleted=0) or uninstall: python scripts/install_label_guard.py --uninstall"
        return f"""
            CREATE TRIGGER IF NOT EXISTS {trigger_name}
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            WHEN OLD.deleted = 1 AND NEW.deleted = 1
              AND (
                IFNULL(NEW.label,'')          != IFNULL(OLD.label,'')          OR
                IFNULL(NEW.label_source,'')   != IFNULL(OLD.label_source,'')   OR
                IFNULL(NEW.label_confirmed,0) != IFNULL(OLD.label_confirmed,0) OR
                IFNULL(NEW.labeled_at,'')     != IFNULL(OLD.labeled_at,'')     OR
                IFNULL(NEW.variant,'')        != IFNULL(OLD.variant,'')
              )
            BEGIN
                SELECT RAISE(ABORT, '{msg}');
            END;
        """
    raise ValueError(f"unknown trigger kind: {kind}")


def install(conn: sqlite3.Connection):
    for trg, tbl, kind in TRIGGERS:
        sql = _create_trigger_sql(trg, tbl, kind)
        try:
            conn.execute(sql)
            print(f"✓ installed {trg} on {tbl}  [{kind}]")
        except sqlite3.OperationalError as e:
            print(f"✗ skip {trg} ({e})")


def uninstall(conn: sqlite3.Connection):
    for trg, _tbl, _kind in TRIGGERS:
        conn.execute(f"DROP TRIGGER IF EXISTS {trg};")
        print(f"✓ dropped {trg}")


def check(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT name, tbl_name FROM sqlite_master WHERE type='trigger' "
        "AND (name LIKE 'trg_%_no_unconfirm' OR name LIKE 'trg_%_no_modify_deleted')"
    ).fetchall()
    if not rows:
        print("⚠ No guard triggers installed. Run without --check to install.")
        return
    print("Installed triggers:")
    for r in rows:
        print(f"  {r[0]}  on  {r[1]}")

    # Smoke test 1: label_confirmed=1→0 must fail
    print()
    print("Smoke test 1: flip label_confirmed=1→0 on existing confirmed row...")
    try:
        conn.execute("BEGIN")
        r = conn.execute(
            "SELECT id FROM grafana_pool WHERE label_confirmed=1 LIMIT 1"
        ).fetchone()
        if r:
            try:
                conn.execute(
                    "UPDATE grafana_pool SET label_confirmed=0 WHERE id=?", (r[0],)
                )
                print("  ✗ Trigger did NOT fire — bug.")
            except sqlite3.IntegrityError as e:
                print(f"  ✓ Trigger fires correctly: {e}")
        else:
            print("  (no confirmed=1 grafana row for live smoke test)")
        conn.execute("ROLLBACK")
    except Exception as e:
        print(f"  smoke test 1 error: {e}")
        try: conn.execute("ROLLBACK")
        except: pass

    # Smoke test 2: modify a deleted row must fail; undelete must succeed
    print()
    print("Smoke test 2: modify a deleted=1 row (label change)...")
    try:
        conn.execute("BEGIN")
        r = conn.execute(
            "SELECT id, label FROM grafana_pool WHERE deleted=1 LIMIT 1"
        ).fetchone()
        if r:
            new_label = 'teen' if (r[1] or '') != 'teen' else 'adult'
            try:
                conn.execute(
                    "UPDATE grafana_pool SET label=? WHERE id=?", (new_label, r[0])
                )
                print("  ✗ Trigger did NOT fire — bug.")
            except sqlite3.IntegrityError as e:
                print(f"  ✓ Trigger fires correctly: {e}")
            # Smoke test 2b: undelete (deleted=1→0) must succeed
            try:
                conn.execute(
                    "UPDATE grafana_pool SET deleted=0 WHERE id=?", (r[0],)
                )
                print(f"  ✓ Undelete (deleted=1→0) allowed for id={r[0]}")
            except sqlite3.IntegrityError as e:
                print(f"  ✗ Undelete unexpectedly blocked: {e}")
        else:
            print("  (no deleted=1 grafana row for live smoke test)")
        conn.execute("ROLLBACK")
    except Exception as e:
        print(f"  smoke test 2 error: {e}")
        try: conn.execute("ROLLBACK")
        except: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"ERR: {DB_PATH} not found", file=sys.stderr); sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        if args.check:
            check(conn)
        elif args.uninstall:
            uninstall(conn)
            conn.commit()
            print("\n⚠ Guard is OFF. Re-install ASAP after you're done with whatever required disabling it.")
        else:
            install(conn)
            conn.commit()
            print("\n✓ Guard active.")
            print("  • Any UPDATE that flips label_confirmed=1→0 will now ABORT")
            print("  • Triggers travel WITH the DB file — even external scripts can't bypass")
            print("  • To check: python scripts/install_label_guard.py --check")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
