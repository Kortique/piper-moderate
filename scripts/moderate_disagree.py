"""
moderate_disagree.py
--------------------
Auto-moderate new entries in data/disagree_pool.json via the Piper pipeline
that includes both Siglip2 and Qwen3-VL.

One Piper call per image returns:
  • qwen3_details  — faces [{ageFrom, ageTo}], description, underage bool, status
  • qwen3_labels   — e.g. ["asian"]
  • siglip2_labels — e.g. ["underage"]  (empty = passed)
  • siglip2_details — per-label risk scores

From these we derive:
  • age label (child / teen / adult)  — from youngest face in qwen3_details.faces
  • piper_result  — siglip2 blocked/passed
  • qwen3_result  — qwen3 age faces + description

Only processes entries where qwen3_result is None (new images).
Never overwrites a human-confirmed label.

Usage:
    python scripts/moderate_disagree.py              # all unmoderated
    python scripts/moderate_disagree.py --limit 50   # up to 50
    python scripts/moderate_disagree.py --workers 3  # parallel Piper launches
    python scripts/moderate_disagree.py --stats      # stats only
    python scripts/moderate_disagree.py --reprocess  # re-run already processed
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv()
console = Console()

BASE_DIR         = Path(__file__).resolve().parent.parent
POOL_FILE        = BASE_DIR / "data" / "disagree_pool.json"

PIPER_BASE       = "https://piper-next.artworks.ai/api"
MODERATE_PROJECT = "d2911d10bb"   # Siglip2 + Qwen3-VL combined pipeline
PIPER_TOKEN      = os.getenv("PIPER_TOKEN", "")


# ─────────────────────────────────────────────────────────────────────────────
# Pool helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pool() -> dict:
    if POOL_FILE.exists():
        raw = POOL_FILE.read_bytes().rstrip(b'\x00')
        return json.loads(raw)
    return {}


def save_pool(pool: dict):
    tmp = str(POOL_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    os.replace(tmp, POOL_FILE)


def migrate_legacy_labels(pool: dict) -> int:
    """Mark pre-existing manually-set labels (label set, label_source=None) as
    human-confirmed so they are never overwritten by auto-moderation."""
    migrated = 0
    for v in pool.values():
        if v.get("label") and v.get("label_source") is None:
            v["label_source"]    = "human"
            v["label_confirmed"] = True
            migrated += 1
    return migrated


# ─────────────────────────────────────────────────────────────────────────────
# Age derivation from Qwen3-VL faces
# ─────────────────────────────────────────────────────────────────────────────

def age_label_from_faces(faces: list) -> str:
    """Derive child/teen/adult from youngest face in qwen3_details.faces.
    Uses ageFrom (minimum bound) of the youngest person:
    child  : ageFrom 1–14
    teen   : ageFrom 15–17
    adult  : ageFrom 18+  (or no faces)
    """
    if not faces:
        return "adult"
    youngest = min(f.get("ageFrom", 99) for f in faces)
    if youngest <= 14:
        return "child"
    if youngest <= 17:
        return "teen"
    return "adult"


def age_label_from_detect_face(features: dict) -> str:
    """Derive child/teen/adult from detect_face node output.
    features = {'ageFrom': N, 'ageTo': N, 'gender': ..., 'race': ..., 'emotion': ...}
    Uses ageFrom (minimum bound):
    child  : ageFrom 1–14
    teen   : ageFrom 15–17
    adult  : ageFrom 18+  (or missing)
    """
    if not features:
        return "adult"
    age_from = features.get("ageFrom")
    if age_from is None:
        return "adult"
    if age_from <= 14:
        return "child"
    if age_from <= 17:
        return "teen"
    return "adult"


# ─────────────────────────────────────────────────────────────────────────────
# Piper
# ─────────────────────────────────────────────────────────────────────────────

def _piper_headers() -> dict:
    if not PIPER_TOKEN:
        raise RuntimeError("PIPER_TOKEN not set in .env")
    return {
        "User-Token": PIPER_TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def run_pipeline(image_url: str) -> dict:
    """Launch Piper project d2911d10bb and poll until both qwen3 and siglip2
    outputs are present. Returns raw outputs dict."""
    headers = _piper_headers()

    with httpx.Client(timeout=60, follow_redirects=False) as client:
        r = client.post(
            f"{PIPER_BASE}/projects/{MODERATE_PROJECT}/launch",
            headers=headers,
            json={"inputs": {"image": image_url}},
        )
        r.raise_for_status()
        run_id = r.json()["_id"]

        for _ in range(60):   # up to 3 min
            time.sleep(3)
            rs = client.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=headers)
            if rs.status_code != 200:
                continue
            state   = rs.json()
            outputs = state.get("outputs") or {}
            errors  = state.get("errors") or []

            if errors:
                # "no faces detected" is a soft error — siglip2 may still have results
                if "siglip2_labels" in outputs:
                    outputs["_face_error"] = str(errors)
                    return outputs
                return {"error": str(errors)}

            # Wait for siglip2 output (qwen3 no longer in live pipeline)
            if "siglip2_labels" in outputs:
                return outputs

    return {"error": "timeout"}


# ─────────────────────────────────────────────────────────────────────────────
# Process one entry
# ─────────────────────────────────────────────────────────────────────────────

def process_entry(entry: dict) -> tuple[str, dict, dict, str]:
    """Run pipeline on one entry. Returns (gen_id, qwen3_result, piper_result, age_label)."""
    gen_id    = entry["id"]
    thumb_url = entry["thumb_url"]

    outputs = run_pipeline(thumb_url)

    now = datetime.now(timezone.utc).isoformat()

    if outputs.get("error"):
        err_str = str(outputs["error"])
        # "no faces detected" = detect_face ran but found no face — treat as adult/passed
        if "no faces detected" in err_str:
            qwen3_result = {
                "label":       "adult",
                "faces":       [],
                "description": "",
                "underage":    False,
                "status":      "no_face",
                "processed_at": now,
            }
            piper_result = {
                "siglip2_labels":  [],
                "siglip2_passed":  True,
                "siglip2_details": None,
                "processed_at":    now,
            }
            return gen_id, qwen3_result, piper_result, "adult"
        return gen_id, {"error": outputs["error"], "processed_at": now}, {"error": outputs["error"], "processed_at": now}, None

    # ── age from detect_face features (replaces qwen3_details in live pipeline) ──
    feat = outputs.get("features") or {}
    age_lbl = age_label_from_detect_face(feat)

    qwen3_result = {
        "label":       age_lbl,
        "faces":       [feat] if feat else [],   # store detect_face output in faces slot for compat
        "description": "",
        "underage":    age_lbl in ("child", "teen"),
        "status":      "detect_face",
        "processed_at": now,
    }

    # ── siglip2 ──
    sl = outputs.get("siglip2_labels") or []
    piper_result = {
        "siglip2_labels":  sl,
        "siglip2_passed":  "underage" not in sl,
        "siglip2_details": outputs.get("siglip2_details"),
        "processed_at":    now,
    }

    return gen_id, qwen3_result, piper_result, age_lbl


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(pool: dict):
    total     = len(pool)
    q_done    = sum(1 for v in pool.values() if v.get("qwen3_result") and not v["qwen3_result"].get("error"))
    p_done    = sum(1 for v in pool.values() if v.get("piper_result") and not v["piper_result"].get("error"))
    confirmed = sum(1 for v in pool.values() if v.get("label_confirmed"))

    labels = {"child": 0, "teen": 0, "adult": 0}
    for v in pool.values():
        lbl = v.get("label")
        if lbl in labels:
            labels[lbl] += 1

    console.print(
        f"\n[bold]Pool stats:[/bold]  total={total}  "
        f"qwen3={q_done}  piper={p_done}  confirmed={confirmed}"
    )
    console.print(
        f"[dim]Labels:[/dim]  "
        f"child=[red]{labels['child']}[/red]  "
        f"teen=[yellow]{labels['teen']}[/yellow]  "
        f"adult=[green]{labels['adult']}[/green]  "
        f"unconfirmed=[dim]{total - confirmed}[/dim]"
    )

    # Confusion: qwen3 age label vs siglip2 underage detection
    processed = [v for v in pool.values()
                 if v.get("qwen3_result") and not v["qwen3_result"].get("error")
                 and v.get("piper_result") and not v["piper_result"].get("error")]

    if not processed:
        return

    matrix = {
        "child": {"underage": 0, "other": 0, "passed": 0},
        "teen":  {"underage": 0, "other": 0, "passed": 0},
        "adult": {"underage": 0, "other": 0, "passed": 0},
    }
    for v in processed:
        lbl = v.get("label")
        if lbl not in matrix:
            continue
        pr  = v.get("piper_result", {})
        sl  = pr.get("siglip2_labels") or []
        if "underage" in sl:
            matrix[lbl]["underage"] += 1
        elif not pr.get("siglip2_passed"):
            matrix[lbl]["other"] += 1
        else:
            matrix[lbl]["passed"] += 1

    console.print(f"\n[bold]Siglip2 × Qwen3 age cross-tab  (n={len(processed)}):[/bold]")
    for lbl, row in matrix.items():
        t = sum(row.values())
        if t:
            console.print(
                f"  {lbl:6s}:  underage=[red]{row['underage']}[/red]  "
                f"other=[yellow]{row['other']}[/yellow]  "
                f"passed=[green]{row['passed']}[/green]  total={t}"
            )

    # Recall / FP
    u_total    = sum(matrix["child"].values()) + sum(matrix["teen"].values())
    u_detected = matrix["child"]["underage"]   + matrix["teen"]["underage"]
    adult_t    = sum(matrix["adult"].values())
    adult_fp   = matrix["adult"]["underage"]
    if u_total:
        console.print(
            f"\n  [bold]Underage recall:[/bold] {u_detected}/{u_total} = "
            f"[green]{u_detected/u_total*100:.1f}%[/green]"
        )
    if adult_t:
        console.print(
            f"  [bold]Adult FP rate:[/bold]  {adult_fp}/{adult_t} = "
            f"[red]{adult_fp/adult_t*100:.1f}%[/red]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--limit",     default=0,  help="Max images to process (0 = all new)")
@click.option("--workers",   default=2,  show_default=True, help="Parallel Piper launches")
@click.option("--reprocess", is_flag=True, help="Re-run even if already processed")
@click.option("--stats",     is_flag=True, help="Print stats only")
def main(limit, workers, reprocess, stats):
    """Auto-moderate disagree images via Piper (Siglip2 + Qwen3-VL)."""

    pool = load_pool()

    # Protect pre-existing manual labels
    migrated = migrate_legacy_labels(pool)
    if migrated:
        save_pool(pool)
        console.print(f"[dim]Migrated {migrated} legacy manual labels → human/confirmed[/dim]")

    if stats:
        print_stats(pool)
        return

    def needs_processing(v):
        if reprocess:
            return True
        qr = v.get("qwen3_result") or {}
        pr = v.get("piper_result") or {}
        # Process if either result is missing or errored
        return not v.get("qwen3_result") or qr.get("error") or not v.get("piper_result") or pr.get("error")

    candidates = [v for v in pool.values() if needs_processing(v)]

    if limit > 0:
        candidates = candidates[:limit]

    console.print(
        f"[bold]Project:[/bold] {MODERATE_PROJECT}  "
        f"[bold]To process:[/bold] {len(candidates)}  "
        f"[bold]Workers:[/bold] {workers}"
    )

    if not candidates:
        console.print("[dim]Nothing to do.[/dim]")
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
        task = progress.add_task("Running pipeline…", total=len(candidates))

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(process_entry, entry): entry for entry in candidates}

            for future in as_completed(futures):
                gen_id, qwen3_result, piper_result, age_lbl = future.result()

                pool[gen_id]["qwen3_result"] = qwen3_result
                pool[gen_id]["piper_result"] = piper_result

                # Set age label only if not human-confirmed
                existing_source = pool[gen_id].get("label_source")
                can_overwrite   = (not pool[gen_id].get("label")) or existing_source == "qwen3"
                if can_overwrite and age_lbl:
                    pool[gen_id]["label"]           = age_lbl
                    pool[gen_id]["label_source"]    = "qwen3"
                    pool[gen_id]["label_confirmed"] = False

                status = f"[dim]{gen_id[:8]}…[/dim] → age=[bold]{age_lbl or '?'}[/bold]"
                if piper_result.get("siglip2_labels"):
                    status += f"  siglip2=[red]{piper_result['siglip2_labels']}[/red]"
                progress.update(task, description=status, advance=1)

                saved_count += 1
                if saved_count % 5 == 0:
                    save_pool(pool)

    save_pool(pool)
    console.print(f"\n[green]Done. Processed {saved_count} images.[/green]")
    print_stats(pool)


if __name__ == "__main__":
    main()
