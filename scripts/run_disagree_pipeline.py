"""
run_disagree_pipeline.py
------------------------
Run labeled disagree images through the Piper pipeline and store results
back into gallery.db (grafana_pool table).

Only processes entries where:
  - label is set (child | teen | adult)
  - piper_result is null (not yet processed)

After processing, prints accuracy stats comparing manual labels vs pipeline.

Usage:
    python scripts/run_disagree_pipeline.py            # process all unprocessed labeled
    python scripts/run_disagree_pipeline.py --limit 50 # process up to 50
    python scripts/run_disagree_pipeline.py --all      # re-process already-processed too
    python scripts/run_disagree_pipeline.py --stats    # just print stats, no processing

Piper project: PIPER_PROJECT from .env (default b2fb1af977)
"""

import os
import sys
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv()
console = Console()

BASE_DIR      = Path(__file__).resolve().parent.parent
DB_PATH       = BASE_DIR / "gallery.db"
POOL_FILE     = BASE_DIR / "data" / "disagree_pool.json"

PIPER_BASE    = "https://piper-next.artworks.ai/api"
PIPER_PROJECT = os.getenv("PIPER_PROJECT", "b2fb1af977")
PIPER_TOKEN   = os.getenv("PIPER_TOKEN")


def _db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def get_headers():
    if not PIPER_TOKEN:
        console.print("[red]PIPER_TOKEN not set in .env[/red]")
        raise SystemExit(1)
    return {
        "User-Token": PIPER_TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def run_piper(image_url: str, providers: list[str] = None) -> dict:
    """Launch pipeline and poll until complete. Returns outputs dict."""
    if providers is None:
        providers = ["siglip2"]
    headers = get_headers()

    with httpx.Client(timeout=60, follow_redirects=False) as client:
        r = client.post(
            f"{PIPER_BASE}/projects/{PIPER_PROJECT}/launch",
            headers=headers,
            json={"inputs": {"image": image_url, "providers": providers}},
        )
        r.raise_for_status()
        run_id = r.json()["_id"]

        for _ in range(40):
            time.sleep(3)
            rs = client.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=headers)
            if rs.status_code != 200:
                continue
            state   = rs.json()
            outputs = state.get("outputs") or {}
            errors  = state.get("errors") or []

            if errors:
                return {"error": str(errors)}

            if any(k.startswith("siglip2_") for k in outputs):
                return outputs

    return {"error": "timeout"}


def load_pool() -> dict:
    """Load grafana_pool from SQLite, fall back to JSON if DB missing."""
    if DB_PATH.exists():
        conn = _db_connect()
        rows = conn.execute("""
            SELECT id, thumb_url, local_path, prompt, label, label_source,
                   label_confirmed, labeled_at, variant, export_batch, exported_at,
                   piper_result, qwen3_result, extra
            FROM grafana_pool
        """).fetchall()
        conn.close()
        pool = {}
        for row in rows:
            item = {
                "id":             row["id"],
                "thumb_url":      row["thumb_url"],
                "local_path":     row["local_path"],
                "prompt":         row["prompt"],
                "label":          row["label"],
                "label_source":   row["label_source"],
                "label_confirmed": bool(row["label_confirmed"]),
                "labeled_at":     row["labeled_at"],
                "variant":        row["variant"],
                "export_batch":   row["export_batch"],
                "exported_at":    row["exported_at"],
                "piper_result":   json.loads(row["piper_result"]) if row["piper_result"] else None,
                "qwen3_result":   json.loads(row["qwen3_result"]) if row["qwen3_result"] else None,
            }
            if row["extra"]:
                item.update(json.loads(row["extra"]))
            pool[row["id"]] = item
        return pool
    # Fallback to JSON
    if POOL_FILE.exists():
        with open(POOL_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_piper_result(gen_id: str, piper_result: dict):
    """Atomic single-row update in SQLite + sync JSON."""
    pr_json = json.dumps(piper_result, ensure_ascii=False)
    if DB_PATH.exists():
        conn = _db_connect()
        conn.execute(
            "UPDATE grafana_pool SET piper_result=? WHERE id=?",
            (pr_json, gen_id)
        )
        conn.commit()
        conn.close()
    # Also keep JSON in sync for backward compat
    _sync_pool_json_from_db()


def _sync_pool_json_from_db():
    """Regenerate disagree_pool.json from DB."""
    if not DB_PATH.exists(): return
    conn = _db_connect()
    rows = conn.execute("""
        SELECT id, thumb_url, local_path, prompt, label, label_source,
               label_confirmed, labeled_at, variant, export_batch, exported_at,
               piper_result, qwen3_result, extra
        FROM grafana_pool
    """).fetchall()
    conn.close()
    data = {}
    for row in rows:
        item = {
            "id":             row["id"],
            "thumb_url":      row["thumb_url"],
            "local_path":     row["local_path"],
            "prompt":         row["prompt"],
            "label":          row["label"],
            "label_source":   row["label_source"],
            "label_confirmed": bool(row["label_confirmed"]),
            "labeled_at":     row["labeled_at"],
            "variant":        row["variant"],
            "export_batch":   row["export_batch"],
            "exported_at":    row["exported_at"],
            "piper_result":   json.loads(row["piper_result"]) if row["piper_result"] else None,
            "qwen3_result":   json.loads(row["qwen3_result"]) if row["qwen3_result"] else None,
        }
        if row["extra"]:
            item.update(json.loads(row["extra"]))
        data[row["id"]] = item
    tmp = str(POOL_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, POOL_FILE)


def save_pool(pool: dict):
    """Legacy: update all changed piper_results in DB then sync JSON."""
    if DB_PATH.exists():
        conn = _db_connect()
        for gen_id, entry in pool.items():
            pr = entry.get("piper_result")
            conn.execute(
                "UPDATE grafana_pool SET piper_result=? WHERE id=?",
                (json.dumps(pr, ensure_ascii=False) if pr else None, gen_id)
            )
        conn.commit()
        conn.close()
    _sync_pool_json_from_db()


def print_stats(pool: dict):
    """Print confusion matrix: manual label vs pipeline underage detection."""
    processed = [v for v in pool.values() if v.get("piper_result") and not v["piper_result"].get("error")]

    if not processed:
        console.print("[yellow]No processed records yet.[/yellow]")
        return

    # Confusion matrix: label (rows) vs pipeline underage (cols)
    matrix = {
        "child": {"blocked_underage": 0, "blocked_other": 0, "passed": 0},
        "teen":  {"blocked_underage": 0, "blocked_other": 0, "passed": 0},
        "adult": {"blocked_underage": 0, "blocked_other": 0, "passed": 0},
    }

    for v in processed:
        label  = v.get("label")
        result = v.get("piper_result", {})
        labels = result.get("siglip2_labels") or []
        passed = result.get("siglip2_passed")

        if label not in matrix:
            continue

        has_underage = "underage" in labels
        if has_underage:
            matrix[label]["blocked_underage"] += 1
        elif not passed:
            matrix[label]["blocked_other"] += 1
        else:
            matrix[label]["passed"] += 1

    table = Table(title=f"Pipeline vs Manual Label  (n={len(processed)})", show_header=True)
    table.add_column("Manual label")
    table.add_column("Blocked: underage", justify="right")
    table.add_column("Blocked: other",    justify="right")
    table.add_column("Passed",            justify="right")
    table.add_column("Total",             justify="right")

    for lbl, row in matrix.items():
        total = sum(row.values())
        if total == 0:
            continue
        table.add_row(
            lbl,
            str(row["blocked_underage"]),
            str(row["blocked_other"]),
            str(row["passed"]),
            str(total),
        )

    console.print(table)

    # False positive rate on adult images
    adult = matrix.get("adult", {})
    adult_total = sum(adult.values())
    if adult_total:
        fp = adult.get("blocked_underage", 0)
        console.print(
            f"\n[bold]Adult FP rate:[/bold] {fp}/{adult_total} = [red]{fp/adult_total*100:.1f}%[/red] "
            f"(adults incorrectly blocked as underage)"
        )

    # Recall on underage (child + teen)
    u_total = sum(matrix["child"].values()) + sum(matrix["teen"].values())
    u_detected = matrix["child"]["blocked_underage"] + matrix["teen"]["blocked_underage"]
    if u_total:
        console.print(
            f"[bold]Underage recall:[/bold] {u_detected}/{u_total} = [green]{u_detected/u_total*100:.1f}%[/green] "
            f"(child+teen correctly blocked)"
        )


@click.command()
@click.option("--limit",  default=0,     help="Max images to process (0 = all)")
@click.option("--all",    "reprocess",   is_flag=True, help="Re-process already-processed entries too")
@click.option("--stats",  is_flag=True,  help="Print stats only, no processing")
@click.option("--providers", default="siglip2", show_default=True)
def main(limit, reprocess, stats, providers):
    """Run labeled disagree images through Piper pipeline."""

    pool = load_pool()

    if stats:
        print_stats(pool)
        return

    provider_list = [p.strip() for p in providers.split(",")]

    # Select candidates
    candidates = []
    for v in pool.values():
        if not v.get("label"):
            continue
        if not reprocess and v.get("piper_result"):
            continue
        candidates.append(v)

    if limit > 0:
        candidates = candidates[:limit]

    console.print(
        f"[bold]{len(candidates)}[/bold] images to process "
        f"(providers: {', '.join(provider_list)})"
    )

    if not candidates:
        console.print("[dim]Nothing to do. Label some images in the gallery first.[/dim]")
        print_stats(pool)
        return

    saved_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=len(candidates))

        for i, entry in enumerate(candidates):
            gen_id    = entry["id"]
            thumb_url = entry["thumb_url"]

            progress.update(task, description=f"[dim]{gen_id[:8]}...[/dim]")

            outputs = run_piper(thumb_url, provider_list)

            face_result = None
            for key, val in outputs.items():
                if ('face' in key.lower() or 'detect' in key.lower()) and isinstance(val, dict):
                    if 'ageFrom' in val or 'age_from' in val:
                        face_result = val
                        break

            piper_result = {
                "siglip2_labels":     outputs.get("siglip2_labels"),
                "siglip2_passed":     outputs.get("siglip2_passed"),
                "siglip2_details":    outputs.get("siglip2_details"),
                "face_detect_result": face_result,
                "error":              outputs.get("error"),
                "processed_at":       datetime.now(timezone.utc).isoformat(),
            }
            pool[gen_id]["piper_result"] = piper_result

            # Atomic per-row write to DB (safe against process kill)
            save_piper_result(gen_id, piper_result)

            saved_count += 1
            progress.advance(task)

    # Final full sync (no-op if DB already up to date)
    save_pool(pool)
    console.print(f"[green]Done. Processed {saved_count} images.[/green]")
    console.print()
    print_stats(pool)


if __name__ == "__main__":
    main()
