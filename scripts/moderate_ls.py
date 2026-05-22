"""
moderate_ls.py
--------------
Run Label Studio images through Piper pipeline d2911d10bb (Siglip2 + Qwen3-VL)
and store siglip2 results back into qwen3_age_results.json.

Adds to each entry:
    siglip2_passed  : bool
    siglip2_labels  : list
    siglip2_details : dict
    piper_processed_at : ISO timestamp

Only processes entries where siglip2_passed is missing or None.

Usage:
    python scripts/moderate_ls.py               # all unprocessed
    python scripts/moderate_ls.py --limit 100
    python scripts/moderate_ls.py --workers 3
    python scripts/moderate_ls.py --stats
    python scripts/moderate_ls.py --reprocess
"""

import os, sys, json, time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx, click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv()
console = Console()

BASE_DIR         = Path(__file__).resolve().parent.parent
LS_FILE          = BASE_DIR / "qwen3_age_results.json"

PIPER_BASE       = "https://piper-next.artworks.ai/api"
MODERATE_PROJECT = "d2911d10bb"
PIPER_TOKEN      = os.getenv("PIPER_TOKEN", "")


def load_ls():
    return json.loads(LS_FILE.read_text(encoding="utf-8"))

def save_ls(data: dict):
    tmp = str(LS_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(LS_FILE))

def _headers():
    return {"User-Token": PIPER_TOKEN, "Content-Type": "application/json", "Accept": "application/json"}

def run_pipeline(image_url: str) -> dict:
    headers = _headers()
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        r = client.post(
            f"{PIPER_BASE}/projects/{MODERATE_PROJECT}/launch",
            headers=headers,
            json={"inputs": {"image": image_url}},
        )
        r.raise_for_status()
        run_id = r.json()["_id"]
        for _ in range(60):
            time.sleep(3)
            rs = client.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=headers)
            if rs.status_code != 200: continue
            state   = rs.json()
            outputs = state.get("outputs") or {}
            errors  = state.get("errors") or []
            if errors: return {"error": str(errors)}
            if "siglip2_labels" in outputs and "qwen3_details" in outputs:
                return outputs
    return {"error": "timeout"}

def process_entry(key: str, entry: dict) -> tuple[str, dict]:
    outputs = run_pipeline(entry["media"])
    now = datetime.now(timezone.utc).isoformat()
    if outputs.get("error"):
        return key, {"siglip2_passed": None, "siglip2_labels": None,
                     "siglip2_details": None, "piper_error": outputs["error"],
                     "piper_processed_at": now}
    sl = outputs.get("siglip2_labels") or []
    return key, {
        "siglip2_passed":  "underage" not in sl,
        "siglip2_labels":  sl,
        "siglip2_details": outputs.get("siglip2_details"),
        "piper_processed_at": now,
    }

def print_stats(data: dict):
    total     = len(data)
    processed = sum(1 for v in data.values() if v.get("siglip2_passed") is not None)
    errors    = sum(1 for v in data.values() if v.get("piper_error"))

    def lbl(v):
        af = (v.get("age") or {}).get("ageFrom")
        if af is None: return None
        if af <= 14: return "child"
        if af <= 17: return "teen"
        return "adult"

    matrix = {"child":{"blocked":0,"passed":0},"teen":{"blocked":0,"passed":0},"adult":{"blocked":0,"passed":0}}
    for v in data.values():
        if v.get("siglip2_passed") is None: continue
        l = lbl(v)
        if l not in matrix: continue
        if not v["siglip2_passed"]: matrix[l]["blocked"] += 1
        else:                       matrix[l]["passed"]  += 1

    console.print(f"\n[bold]LS stats:[/bold]  total={total}  processed={processed}  errors={errors}")
    for l, row in matrix.items():
        t = sum(row.values())
        if t:
            pct = row['blocked']/t*100
            console.print(f"  {l:6s}: blocked={row['blocked']}  passed={row['passed']}  ({pct:.1f}% blocked)")

    pos = [v for v in data.values() if v.get("variant")=="positive" and v.get("siglip2_passed") is not None]
    neg = [v for v in data.values() if v.get("variant")=="negative" and v.get("siglip2_passed") is not None]
    if pos:
        blocked = sum(1 for v in pos if not v["siglip2_passed"])
        console.print(f"\n  Recall  (positive blocked): {blocked}/{len(pos)} = {blocked/len(pos)*100:.1f}%")
    if neg:
        fp = sum(1 for v in neg if not v["siglip2_passed"])
        console.print(f"  FP rate (negative blocked): {fp}/{len(neg)} = {fp/len(neg)*100:.1f}%")

@click.command()
@click.option("--limit",     default=0, help="Max entries to process (0=all)")
@click.option("--workers",   default=2, show_default=True)
@click.option("--reprocess", is_flag=True)
@click.option("--stats",     is_flag=True)
def main(limit, workers, reprocess, stats):
    """Run LS images through Piper d2911d10bb, store siglip2 results."""
    data = load_ls()

    if stats:
        print_stats(data)
        return

    candidates = [
        (k, v) for k, v in data.items()
        if reprocess or v.get("siglip2_passed") is None
    ]
    if limit > 0:
        candidates = candidates[:limit]

    console.print(f"[bold]Project:[/bold] {MODERATE_PROJECT}  [bold]To process:[/bold] {len(candidates)}  [bold]Workers:[/bold] {workers}")
    if not candidates:
        console.print("[dim]Nothing to do.[/dim]")
        print_stats(data)
        return

    saved_count = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), console=console) as progress:
        task = progress.add_task("Running…", total=len(candidates))
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(process_entry, k, v): k for k, v in candidates}
            for future in as_completed(futures):
                key, result = future.result()
                data[key].update(result)
                saved_count += 1
                status = f"[dim]{key[:8]}…[/dim]"
                if result.get("siglip2_passed") is not None:
                    status += " → " + ("[green]pass[/green]" if result["siglip2_passed"] else "[red]block[/red]")
                progress.update(task, description=status, advance=1)
                if saved_count % 10 == 0:
                    save_ls(data)

    save_ls(data)
    console.print(f"[green]Done. Processed {saved_count} entries.[/green]")
    print_stats(data)

if __name__ == "__main__":
    main()
