#!/usr/bin/env python3
"""
scan_ls_images.py — run images through Piper V4.

Writes results to gallery.db (SQLite) — atomic per-row writes, no truncation risk.
Also keeps JSON files in sync for backward compatibility.

Safe to restart — skips items that already have the requested data in DB.

Usage:
    # siglip2 scan (default, LS only)
    python scripts/scan_ls_images.py

    # face_detect scan for ALL images (LS + Grafana)
    python scripts/scan_ls_images.py --providers face_detect --source all

    # both providers, grafana only
    python scripts/scan_ls_images.py --providers siglip2,face_detect --source grafana

    # more workers
    python scripts/scan_ls_images.py --workers 8
"""
import os, sys, json, time, sqlite3, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = BASE_DIR / "gallery.db"
LS_FILE    = BASE_DIR / "qwen3_age_results.json"
POOL_FILE  = BASE_DIR / "data" / "disagree_pool.json"
PIPER_BASE = "https://piper-next.artworks.ai/api"
PROJECT    = "d2911d10bb"
TOKEN      = os.getenv("PIPER_TOKEN", "")
WORKERS    = 6
JSON_SYNC_EVERY = 50


def hdr():
    return {"User-Token": TOKEN, "Content-Type": "application/json", "Accept": "application/json"}


def run_one(item_id, url, providers):
    """Run one image through Piper. Returns (item_id, result_dict)."""
    try:
        r = httpx.post(
            f"{PIPER_BASE}/projects/{PROJECT}/launch",
            headers=hdr(),
            json={"inputs": {"image": url, "providers": providers}},
            timeout=30,
        )
        r.raise_for_status()
        run_id = r.json()["_id"]

        want_siglip2 = "siglip2" in providers
        want_face    = "face_detect" in providers

        for attempt in range(40):
            time.sleep(3)
            rs = httpx.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=hdr(), timeout=15)
            if rs.status_code != 200:
                continue
            state   = rs.json()
            outputs = state.get("outputs") or {}
            errors  = state.get("errors") or []

            if errors:
                return item_id, {"error": str(errors[0])[:120]}

            # --- Extract face detect result ---
            # Piper returns face data under key "features" (not "face_detect")
            face_result    = None
            face_key_found = False
            for key, val in outputs.items():
                if isinstance(val, dict) and ("ageFrom" in val or "age_from" in val):
                    face_result    = val
                    face_key_found = True
                    break
                if "face" in key.lower() or "detect" in key.lower() or key == "features":
                    face_key_found = True
                    if isinstance(val, dict):
                        face_result = val

            # --- Completion checks ---
            siglip2_ready = (not want_siglip2) or ("siglip2_labels" in outputs)

            # face_detect: done when key appears, OR pipeline status is terminal,
            # OR siglip2 is already done and we've been polling ≥20 times
            pipeline_done = state.get("status") in ("done", "completed", "finished", "success")
            face_ready = (
                (not want_face)
                or face_key_found
                or pipeline_done
                or (want_siglip2 and siglip2_ready and attempt >= 20)
            )

            if siglip2_ready and face_ready:
                return item_id, {
                    "launch_id":          run_id,
                    "siglip2_labels":     outputs.get("siglip2_labels"),
                    "siglip2_passed":     outputs.get("siglip2_passed", True),
                    "siglip2_details":    outputs.get("siglip2_details"),
                    "face_detect_result": face_result,
                    "error":              None,
                }

        return item_id, {"error": "timeout"}

    except Exception as e:
        return item_id, {"error": str(e)[:120]}


def j(v):
    if v is None: return None
    if isinstance(v, (dict, list)): return json.dumps(v, ensure_ascii=False)
    return v


# ── DB save helpers ───────────────────────────────────────────────────────────

def _face_val(result):
    """face_detect_result to store: real data if found, {} sentinel if no face/error, never null."""
    v = result.get("face_detect_result")
    if v is not None:
        return v
    # Mark as scanned even if no face detected (prevents infinite retry)
    return {"error": result["error"]} if result.get("error") else {}


def save_ls(conn, task_id, result, providers):
    """Atomic single-row update for ls_images."""
    face_only = "face_detect" in providers and "siglip2" not in providers

    if face_only:
        conn.execute("""
            UPDATE ls_images SET
                face_detect  = ?,
                processed_at = datetime('now')
            WHERE task_id = ?
        """, (j(_face_val(result)), int(task_id)))
    else:
        conn.execute("""
            UPDATE ls_images SET
                launch_id       = ?,
                siglip2_labels  = ?,
                siglip2_passed  = ?,
                siglip2_details = ?,
                face_detect     = ?,
                error           = ?,
                processed_at    = datetime('now')
            WHERE task_id = ?
        """, (
            result.get("launch_id"),
            j(result.get("siglip2_labels")),
            1 if result.get("siglip2_passed") else 0,
            j(result.get("siglip2_details")),
            j(_face_val(result)),
            result.get("error"),
            int(task_id),
        ))
    conn.commit()


def save_grafana(conn, gen_id, result):
    """Update face_detect_result inside piper_result JSON for grafana_pool."""
    row = conn.execute("SELECT piper_result FROM grafana_pool WHERE id=?", (gen_id,)).fetchone()
    pr = json.loads(row[0]) if row and row[0] else {}
    pr["face_detect_result"] = _face_val(result)  # {} sentinel if no face, never null
    conn.execute("UPDATE grafana_pool SET piper_result=? WHERE id=?",
                 (json.dumps(pr, ensure_ascii=False), gen_id))
    conn.commit()


# ── JSON sync helpers ─────────────────────────────────────────────────────────

def sync_ls_json(conn):
    """Rebuild qwen3_age_results.json from DB."""
    rows = conn.execute("""
        SELECT task_id, media, variant, category, age_from, age_to,
               launch_id, siglip2_labels, siglip2_passed, siglip2_details,
               face_detect, error, processed_at, extra
        FROM ls_images
    """).fetchall()
    data = {}
    for row in rows:
        (tid, media, variant, category, af, at,
         launch_id, labels, passed, details, face, error, proc_at, extra) = row
        item = {
            "task_id":            tid,
            "media":              media,
            "variant":            variant,
            "category":           category,
            "age":                {"ageFrom": af, "ageTo": at} if af is not None else None,
            "launch_id":          launch_id,
            "siglip2_labels":     json.loads(labels) if labels else None,
            "siglip2_passed":     bool(passed) if passed is not None else None,
            "siglip2_details":    json.loads(details) if details else None,
            "face_detect_result": json.loads(face) if face else None,
            "error":              error,
            "piper_processed_at": proc_at,
        }
        if extra:
            item.update(json.loads(extra))
        data[str(tid)] = item
    tmp = str(LS_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, LS_FILE)


def sync_pool_json(conn):
    """Rebuild disagree_pool.json from DB."""
    rows = conn.execute("""
        SELECT id, thumb_url, local_path, prompt, label, label_source,
               label_confirmed, labeled_at, variant, export_batch, exported_at,
               piper_result, qwen3_result, extra
        FROM grafana_pool
    """).fetchall()
    data = {}
    for row in rows:
        (pid, thumb_url, local_path, prompt, label, label_source,
         label_confirmed, labeled_at, variant, export_batch, exported_at,
         piper_result, qwen3_result, extra) = row
        item = {
            "id":             pid,
            "thumb_url":      thumb_url,
            "local_path":     local_path,
            "prompt":         prompt,
            "label":          label,
            "label_source":   label_source,
            "label_confirmed": bool(label_confirmed),
            "labeled_at":     labeled_at,
            "variant":        variant,
            "export_batch":   export_batch,
            "exported_at":    exported_at,
            "piper_result":   json.loads(piper_result) if piper_result else None,
            "qwen3_result":   json.loads(qwen3_result) if qwen3_result else None,
        }
        if extra:
            item.update(json.loads(extra))
        data[pid] = item
    tmp = str(POOL_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, POOL_FILE)


# ── Todo selection ────────────────────────────────────────────────────────────

def get_todo(conn, providers, source):
    """Return list of (source, item_id, url) to scan."""
    face_only = "face_detect" in providers and "siglip2" not in providers
    todo = []

    if source in ("ls", "all"):
        if face_only:
            rows = conn.execute("""
                SELECT task_id, media FROM ls_images
                WHERE face_detect IS NULL AND media IS NOT NULL
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT task_id, media FROM ls_images
                WHERE siglip2_labels IS NULL AND media IS NOT NULL
            """).fetchall()
        todo += [("ls", str(r[0]), r[1]) for r in rows]

    if source in ("grafana", "all"):
        if face_only:
            rows = conn.execute("""
                SELECT id, thumb_url FROM grafana_pool
                WHERE (
                    piper_result IS NULL
                    OR json_extract(piper_result, '$.face_detect_result') IS NULL
                )
                AND thumb_url IS NOT NULL
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, thumb_url FROM grafana_pool
                WHERE piper_result IS NULL AND thumb_url IS NOT NULL
            """).fetchall()
        todo += [("grafana", r[0], r[1]) for r in rows]

    return todo


def count_done(conn, providers, source):
    face_only = "face_detect" in providers and "siglip2" not in providers
    done = 0
    if source in ("ls", "all"):
        if face_only:
            done += conn.execute("SELECT COUNT(*) FROM ls_images WHERE face_detect IS NOT NULL").fetchone()[0]
        else:
            done += conn.execute("SELECT COUNT(*) FROM ls_images WHERE siglip2_labels IS NOT NULL").fetchone()[0]
    if source in ("grafana", "all"):
        if face_only:
            done += conn.execute("""
                SELECT COUNT(*) FROM grafana_pool
                WHERE json_extract(piper_result, '$.face_detect_result') IS NOT NULL
            """).fetchone()[0]
        else:
            done += conn.execute("SELECT COUNT(*) FROM grafana_pool WHERE piper_result IS NOT NULL").fetchone()[0]
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",   type=int,    default=WORKERS)
    parser.add_argument("--providers", type=str,    default="siglip2",
                        help="Comma-separated: siglip2,face_detect,qwen3,hive,hal9")
    parser.add_argument("--source",    type=str,    default="ls",
                        choices=["ls", "grafana", "all"],
                        help="Which table to scan: ls, grafana, or all")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",")]

    if not TOKEN:
        print("ERROR: PIPER_TOKEN not set in .env"); sys.exit(1)
    if not DB_PATH.exists():
        print("ERROR: gallery.db not found. Run: python scripts/init_db.py"); sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    todo        = get_todo(conn, providers, args.source)
    already     = count_done(conn, providers, args.source)
    total_in_db = (
        conn.execute("SELECT COUNT(*) FROM ls_images").fetchone()[0]  if args.source in ("ls", "all")    else 0
    ) + (
        conn.execute("SELECT COUNT(*) FROM grafana_pool").fetchone()[0] if args.source in ("grafana", "all") else 0
    )

    print(f"Providers       : {', '.join(providers)}")
    print(f"Source          : {args.source}")
    print(f"Total in DB     : {total_in_db}")
    print(f"Already done    : {already}")
    print(f"To scan         : {len(todo)}")
    print(f"Workers         : {args.workers}")
    print()

    if not todo:
        print("Nothing to do.")
        conn.close()
        return

    face_only   = "face_detect" in providers and "siglip2" not in providers
    n           = already
    since_sync  = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, item_id, url, providers): (src, item_id)
            for src, item_id, url in todo
        }
        for fut in as_completed(futures):
            src, item_id = futures[fut]
            item_result  = fut.result()[1]
            n           += 1
            since_sync  += 1

            # Atomic DB write
            if src == "ls":
                save_ls(conn, item_id, item_result, providers)
            else:
                save_grafana(conn, item_id, item_result)

            # Progress line
            err  = item_result.get("error")
            face = item_result.get("face_detect_result")
            if err:
                status = f"✗ {err[:35]}"
            elif face_only:
                if face:
                    status = f"✓ face {face.get('ageFrom','?')}-{face.get('ageTo','?')}"
                else:
                    status = "· no face"
            else:
                labels = item_result.get("siglip2_labels")
                if labels is None:
                    status = "✗ no labels"
                elif not labels:
                    status = "✓ passed"
                else:
                    status = "⛔ " + ",".join(labels)
                if face:
                    status += f"  face={face.get('ageFrom','?')}-{face.get('ageTo','?')}"

            det   = (item_result.get("siglip2_details") or {}).get("underage", {})
            minor = det.get("minor", 0)
            tag   = f"[{src[:2]}]" if args.source == "all" else ""
            print(f"  [{n:4d}/{total_in_db}]{tag} {status:40s}  minor={minor:.3f}  id={item_id}")

            if since_sync >= JSON_SYNC_EVERY:
                if args.source in ("ls", "all"):
                    sync_ls_json(conn)
                if args.source in ("grafana", "all"):
                    sync_pool_json(conn)
                since_sync = 0

    # Final JSON sync
    if args.source in ("ls", "all"):
        sync_ls_json(conn)
    if args.source in ("grafana", "all"):
        sync_pool_json(conn)
    conn.close()

    print(f"\nFinished. DB is source of truth: {DB_PATH}")


if __name__ == "__main__":
    main()
