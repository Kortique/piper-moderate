#!/usr/bin/env python3
"""
rollback.py — list snapshots and safely restore gallery.db from one of them.

Safe restore flow:
  1. Probe the chosen snapshot (counts per table)
  2. Auto-create a fresh anchor of CURRENT gallery.db (so the rollback itself
     is reversible — you can always go back to where you were)
  3. Copy snapshot → gallery.db
  4. Print before/after diff

You MUST stop gallery_server.py before running --restore (it locks the file
and any concurrent write will produce torn pages).

Usage:
    python scripts/rollback.py --list                                  # all snapshots
    python scripts/rollback.py --restore anchor_20260527_113000.db     # restore named
    python scripts/rollback.py --restore anchor_20260527_113000.db --dry-run
"""
import argparse, json, sqlite3, shutil, sys, tempfile, os
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "gallery.db"
ANCHORS_DIR = BASE / "backups" / "anchors"
ROTATING_DIR = BASE / "backups"   # rotating gallery_*.db backups


def db_counts(p: Path) -> dict:
    if not p.exists():
        return {"err": "missing"}
    tmp = Path(tempfile.gettempdir()) / f"_rb_probe_{os.getpid()}.db"
    try:
        shutil.copy2(p, tmp)
        conn = sqlite3.connect(str(tmp))
        conn.row_factory = sqlite3.Row
        out = {"size": p.stat().st_size}
        for tbl in ("grafana_pool", "k30_pool", "ls_images"):
            try:
                r = conn.execute(
                    f"SELECT COUNT(*) AS n, "
                    f"SUM(CASE WHEN label_confirmed=1 THEN 1 ELSE 0 END) AS conf, "
                    f"SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) AS labeled, "
                    f"SUM(CASE WHEN deleted=1 THEN 1 ELSE 0 END) AS deleted "
                    f"FROM {tbl}"
                ).fetchone()
                out[tbl] = {
                    "total": r["n"], "labeled": r["labeled"] or 0,
                    "confirmed": r["conf"] or 0, "deleted": r["deleted"] or 0,
                }
            except sqlite3.OperationalError:
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    out[tbl] = {"total": n}
                except Exception:
                    pass
        conn.close()
        return out
    except Exception as e:
        return {"err": str(e)}
    finally:
        try: tmp.unlink()
        except Exception: pass


def collect_snapshots():
    """Yield (path, kind) for every restorable file we know about."""
    out = []
    if ANCHORS_DIR.exists():
        for p in ANCHORS_DIR.glob("anchor_*.db"):
            out.append((p, "anchor"))
    if ROTATING_DIR.exists():
        for p in ROTATING_DIR.glob("gallery_*.db"):
            out.append((p, "rotating"))
    out.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    return out


def fmt_counts(c: dict) -> str:
    if not c or "err" in c:
        return f"  ERR: {c.get('err','?') if c else 'no data'}"
    lines = []
    for tbl in ("grafana_pool", "k30_pool", "ls_images"):
        d = c.get(tbl)
        if not d: continue
        if "confirmed" in d:
            lines.append(
                f"  {tbl:14}  total={d['total']:>5}  "
                f"labeled={d.get('labeled',0):>5}  "
                f"confirmed={d['confirmed']:>5}  "
                f"deleted={d.get('deleted',0):>4}"
            )
        else:
            lines.append(f"  {tbl:14}  total={d['total']:>5}")
    return "\n".join(lines)


def cmd_list():
    snaps = collect_snapshots()
    cur = db_counts(DB_PATH) if DB_PATH.exists() else None

    if cur:
        print("=" * 86)
        print(f"CURRENT  gallery.db")
        print("=" * 86)
        print(fmt_counts(cur))
        print()

    if not snaps:
        print("No snapshots found. Run scripts/snapshot.py to create your first anchor.")
        return

    print("=" * 86)
    print(f"Available snapshots ({len(snaps)}) — newest first")
    print("=" * 86)
    for p, kind in snaps:
        when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = p.stat().st_size / 1024 / 1024
        kind_tag = "🔒 ANCHOR" if kind == "anchor" else "🔄 ROTATING"
        print(f"\n{kind_tag}  {p.name}")
        print(f"  ts: {when}    size: {size_mb:.1f} MB")
        meta_path = p.with_suffix(".json") if kind == "anchor" else None
        if meta_path and meta_path.exists():
            try:
                info = json.loads(meta_path.read_text())
                if info.get("reason"):
                    print(f"  reason: {info['reason']}")
            except Exception:
                pass
        print(fmt_counts(db_counts(p)))


def cmd_restore(name: str, dry_run: bool):
    # Find the snapshot
    candidates = []
    for d in (ANCHORS_DIR, ROTATING_DIR):
        p = d / name
        if p.exists() and p.is_file():
            candidates.append(p)
    if not candidates:
        # Fuzzy: any file named like `name`
        for p, _kind in collect_snapshots():
            if p.name == name:
                candidates.append(p)
    if not candidates:
        print(f"ERR: snapshot '{name}' not found in {ANCHORS_DIR} or {ROTATING_DIR}")
        print("Run --list to see what's available.")
        sys.exit(1)
    src = candidates[0]

    print("=" * 86)
    print(f"RESTORE PLAN")
    print("=" * 86)
    print(f"  from: {src}")
    print(f"  to:   {DB_PATH}")
    print()
    print("BEFORE (current gallery.db):")
    cur = db_counts(DB_PATH) if DB_PATH.exists() else {}
    print(fmt_counts(cur) if cur else "  (no current db)")
    print()
    print("AFTER (the snapshot will replace current):")
    new = db_counts(src)
    print(fmt_counts(new))
    print()

    # Diff per table
    def _delta(tbl, key):
        a = (cur.get(tbl) or {}).get(key, 0) or 0
        b = (new.get(tbl) or {}).get(key, 0) or 0
        d = b - a
        sign = "+" if d > 0 else ("-" if d < 0 else " ")
        return f"{sign}{abs(d)}" if d != 0 else "—"
    print("DELTA  (snapshot − current):")
    for tbl in ("grafana_pool", "k30_pool"):
        if tbl in cur or tbl in new:
            print(f"  {tbl:14}  total: {_delta(tbl,'total'):>6}   "
                  f"labeled: {_delta(tbl,'labeled'):>6}   "
                  f"confirmed: {_delta(tbl,'confirmed'):>6}   "
                  f"deleted: {_delta(tbl,'deleted'):>6}")

    if dry_run:
        print()
        print("[DRY-RUN] No changes made. Re-run without --dry-run to apply.")
        return

    print()
    confirm = input("Are you sure? Stop gallery_server.py first. Type 'restore' to proceed: ")
    if confirm.strip().lower() != "restore":
        print("Aborted.")
        return

    # Step 1: anchor current state as a safety net BEFORE restore
    if DB_PATH.exists():
        ANCHORS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pre = ANCHORS_DIR / f"anchor_{ts}_pre_restore.db"
        shutil.copy2(DB_PATH, pre)
        meta_pre = {
            "created_at": datetime.now().isoformat(),
            "reason": f"automatic pre-restore safety snapshot before rolling back to {src.name}",
            "counts": cur,
        }
        pre.with_suffix(".json").write_text(json.dumps(meta_pre, indent=2, ensure_ascii=False))
        print(f"✓ pre-restore anchor saved: {pre.name}")

    # Step 2: replace gallery.db
    tmp = DB_PATH.with_suffix(".db.restoring")
    shutil.copy2(src, tmp)
    os.replace(tmp, DB_PATH)
    print(f"✓ gallery.db restored from {src.name}")
    print()
    print("Verify counts:")
    print(fmt_counts(db_counts(DB_PATH)))
    print()
    print("Now restart gallery_server.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list",    action="store_true", help="Show all available snapshots")
    ap.add_argument("--restore", metavar="NAME",      help="Restore from given snapshot filename")
    ap.add_argument("--dry-run", action="store_true", help="With --restore, show plan but don't apply")
    args = ap.parse_args()

    if args.list:
        cmd_list()
    elif args.restore:
        cmd_restore(args.restore, args.dry_run)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
