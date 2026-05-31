#!/usr/bin/env python3
"""
snapshot.py — create an immutable anchor snapshot of gallery.db.

Anchors live in backups/anchors/ and are NEVER touched by any rotation logic
in the rest of the codebase. They're your manual checkpoints — pin one before
any potentially-destructive operation (script, big edit, deploy) so you can
always roll back to a known-good state via scripts/rollback.py.

Read-only on gallery.db. Writes only to backups/anchors/.

Usage:
    python scripts/snapshot.py                    # auto-name by timestamp
    python scripts/snapshot.py --label safe_state # adds suffix for findability
    python scripts/snapshot.py --reason "pre import_k30"  # stores note alongside
    python scripts/snapshot.py --list             # show existing anchors
"""
import argparse, json, sqlite3, shutil, sys, tempfile, os
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "gallery.db"
ANCHORS_DIR = BASE / "backups" / "anchors"


def db_counts(p: Path) -> dict:
    """Read-only count probe via /tmp copy (avoids WAL conflicts with running gallery)."""
    if not p.exists():
        return {"err": "missing"}
    tmp = Path(tempfile.gettempdir()) / f"_snapshot_probe_{os.getpid()}.db"
    try:
        shutil.copy2(p, tmp)
        conn = sqlite3.connect(str(tmp))
        conn.row_factory = sqlite3.Row
        out = {"size": p.stat().st_size}
        for tbl in ("grafana_pool", "k30_pool", "ls_images", "borderlands_pool"):
            try:
                r = conn.execute(
                    f"SELECT COUNT(*) AS n, "
                    f"SUM(CASE WHEN label_confirmed=1 THEN 1 ELSE 0 END) AS conf, "
                    f"SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) AS labeled, "
                    f"SUM(CASE WHEN deleted=1 THEN 1 ELSE 0 END) AS deleted "
                    f"FROM {tbl}"
                ).fetchone()
                out[tbl] = {
                    "total": r["n"],
                    "labeled": r["labeled"] or 0,
                    "confirmed": r["conf"] or 0,
                    "deleted": r["deleted"] or 0,
                }
            except sqlite3.OperationalError:
                # Table missing or column missing (e.g. ls_images has no label_confirmed)
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


def cmd_list():
    if not ANCHORS_DIR.exists():
        print(f"No anchors yet. {ANCHORS_DIR} doesn't exist.")
        return
    anchors = sorted(ANCHORS_DIR.glob("anchor_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not anchors:
        print(f"{ANCHORS_DIR} is empty.")
        return
    print(f"{'='*86}")
    print(f"Anchors in {ANCHORS_DIR}  ({len(anchors)} total) — newest first")
    print(f"{'='*86}")
    for a in anchors:
        meta = a.with_suffix(".json")
        info = {}
        if meta.exists():
            try: info = json.loads(meta.read_text())
            except: pass
        when = datetime.fromtimestamp(a.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = a.stat().st_size / 1024 / 1024
        c = info.get("counts", {})
        gp = c.get("grafana_pool", {})
        k30 = c.get("k30_pool", {})
        print(f"\n{a.name}")
        print(f"  ts:       {when}    size: {size_mb:.1f} MB")
        if info.get("reason"):
            print(f"  reason:   {info['reason']}")
        if gp:
            print(f"  grafana:  total={gp.get('total',0):>5}  "
                  f"labeled={gp.get('labeled',0):>5}  "
                  f"confirmed={gp.get('confirmed',0):>5}  "
                  f"deleted={gp.get('deleted',0):>4}")
        if k30:
            print(f"  k30:      total={k30.get('total',0):>5}  "
                  f"labeled={k30.get('labeled',0):>5}  "
                  f"confirmed={k30.get('confirmed',0):>5}  "
                  f"deleted={k30.get('deleted',0):>4}")


def cmd_create(label: str = None, reason: str = None):
    if not DB_PATH.exists():
        print(f"ERR: {DB_PATH} not found", file=sys.stderr)
        sys.exit(1)
    ANCHORS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    target = ANCHORS_DIR / f"anchor_{ts}{suffix}.db"

    # Probe counts BEFORE we copy (so the JSON metadata reflects the actual snapshot)
    counts = db_counts(DB_PATH)

    print(f"Source : {DB_PATH}")
    print(f"Target : {target}")
    if reason:
        print(f"Reason : {reason}")
    print()

    shutil.copy2(DB_PATH, target)

    # Sidecar JSON with metadata + counts
    meta = {
        "created_at": datetime.now().isoformat(),
        "source": str(DB_PATH),
        "size":   target.stat().st_size,
        "label":  label,
        "reason": reason,
        "counts": counts,
    }
    meta_path = target.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("✓ Anchor created.")
    print(f"  file: {target.name}")
    print(f"  meta: {meta_path.name}")
    gp  = (counts.get("grafana_pool")     or {})
    k30 = (counts.get("k30_pool")         or {})
    ls  = (counts.get("ls_images")        or {})
    bl  = (counts.get("borderlands_pool") or {})
    print(f"  grafana_pool:     total={gp.get('total',0):>5}  confirmed={gp.get('confirmed',0):>5}  deleted={gp.get('deleted',0):>4}")
    print(f"  k30_pool:         total={k30.get('total',0):>5}  confirmed={k30.get('confirmed',0):>5}  deleted={k30.get('deleted',0):>4}")
    if ls.get('total'):
        print(f"  ls_images:        total={ls.get('total',0):>5}  (no confirmed/deleted columns)")
    if bl.get('total'):
        print(f"  borderlands_pool: total={bl.get('total',0):>5}  confirmed={bl.get('confirmed',0):>5}  deleted={bl.get('deleted',0):>4}")
    print()
    print("To roll back to this state later:")
    print(f"  python scripts/rollback.py --restore {target.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",  help="Friendly label appended to filename (e.g. safe_state)")
    ap.add_argument("--reason", help="Free-text note stored in sidecar JSON")
    ap.add_argument("--list", action="store_true", help="Show existing anchors")
    args = ap.parse_args()

    if args.list:
        cmd_list()
    else:
        cmd_create(label=args.label, reason=args.reason)


if __name__ == "__main__":
    main()
