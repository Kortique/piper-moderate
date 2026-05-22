"""
patch_and_simulate.py
---------------------
1. Runs specified test IDs through Piper pipeline (fresh siglip2 scores).
2. Patches those items in the snapshot JSON.
3. Saves updated snapshot.
4. Runs accuracy simulation (mirrors simulator/index.html logic) and prints stats.

Usage:
    python scripts/patch_and_simulate.py \
        --snapshot /path/to/tests.json \
        --category weapons_and_military \
        --ids 685,689,696,697

    # Or patch ALL failed items in the category:
    python scripts/patch_and_simulate.py \
        --snapshot /path/to/tests.json \
        --category weapons_and_military \
        --patch-failed

    # Simulate only (no Piper calls), uses snapshot as-is:
    python scripts/patch_and_simulate.py \
        --snapshot /path/to/tests.json \
        --simulate-only
"""

import sys
import os
import json
import time
import datetime
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console()

PIPER_BASE    = "https://piper-next.artworks.ai/api"
PIPER_PROJECT = os.getenv("PIPER_PROJECT", "b2fb1af977")
PIPER_TOKEN   = os.getenv("PIPER_TOKEN")

# Simulator default thresholds (match simulator/index.html DEFAULT_CONFIG)
# but use 0.01 for weapons since that's what the actual pipeline uses
SIM_THRESHOLDS = {
    "weapons":    0.01,   # actual pipeline threshold
    "drugs":      0.01,
    "death":      0.01,
    "rape":       0.03,
    "humanWaste": 0.05,
    "bestiality": 0.05,
    "blood":      0.01,
}

CATEGORIES = [
    "underage", "bestiality", "human_waste", "death_and_murder",
    "weapons_and_military", "blood", "drugs", "rape",
]

DETAIL_KEY_MAP = {
    "weapons_and_military": "weapons",
    "death_and_murder":     "death",
    "human_waste":          "humanWaste",
}


# ─── Piper helpers ────────────────────────────────────────────────────────────

def get_headers():
    if not PIPER_TOKEN:
        console.print("[red]PIPER_TOKEN not set in .env[/red]")
        raise SystemExit(1)
    return {
        "User-Token": PIPER_TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def run_piper(client: httpx.Client, image_url: str) -> dict:
    """Run image through Piper and return raw outputs dict."""
    headers = get_headers()

    r = client.post(
        f"{PIPER_BASE}/projects/{PIPER_PROJECT}/launch",
        headers=headers,
        json={"inputs": {"image": image_url, "providers": ["siglip2"]}},
        timeout=30,
    )
    r.raise_for_status()
    run_id = r.json()["_id"]
    console.print(f"  [dim]launched [cyan]{run_id}[/cyan][/dim]")

    for _ in range(50):
        time.sleep(3)
        rs = client.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=headers, timeout=15)
        if rs.status_code != 200:
            continue
        state = rs.json()
        errors = state.get("errors") or []
        if errors:
            console.print(f"  [red]Pipeline error:[/red] {errors}")
            return {}
        outputs = state.get("outputs") or {}
        if any(k.startswith("siglip2_") for k in outputs):
            return outputs

    console.print("  [yellow]Timeout waiting for Piper results[/yellow]")
    return {}


def patch_siglip2(item: dict, outputs: dict) -> dict:
    """
    Replace item['models']['siglip2'] with fresh Piper outputs.
    Maps siglip2_details / siglip2_labels / siglip2_passed → snapshot structure.
    """
    item = deepcopy(item)
    details  = outputs.get("siglip2_details") or {}
    labels   = outputs.get("siglip2_labels")  or []
    passed   = outputs.get("siglip2_passed")

    if passed is None:
        passed = len(labels) == 0

    # Keep existing details structure, overwrite each category that Piper returned
    existing = item.setdefault("models", {}).setdefault("siglip2", {})
    existing["passed"] = passed

    if "details" not in existing:
        existing["details"] = {}

    for cat_key, det in details.items():
        existing["details"][cat_key] = det

    return item


# ─── Simulator (mirrors simulator/index.html evaluateSiglip2) ─────────────────

def evaluate_siglip2(item: dict) -> bool | None:
    """Returns True=blocked, False=passed, None=no data."""
    d   = (item.get("models") or {}).get("siglip2", {}).get("details") or {}
    cat = item.get("category")

    if cat == "weapons_and_military":
        risk = float(d.get("weapons", {}).get("risk") or 0)
        # Use threshold from data if available, else fallback
        thr  = float(d.get("weapons", {}).get("threshold") or SIM_THRESHOLDS["weapons"])
        return risk >= thr

    if cat == "drugs":
        risk = float(d.get("drugs", {}).get("risk") or 0)
        thr  = float(d.get("drugs", {}).get("threshold") or SIM_THRESHOLDS["drugs"])
        return risk >= thr

    if cat == "death_and_murder":
        risk = float(d.get("death", {}).get("risk") or 0)
        thr  = float(d.get("death", {}).get("threshold") or SIM_THRESHOLDS["death"])
        return risk >= thr

    if cat == "human_waste":
        risk = float(d.get("humanWaste", {}).get("risk") or 0)
        thr  = float(d.get("humanWaste", {}).get("threshold") or SIM_THRESHOLDS["humanWaste"])
        return risk >= thr

    if cat == "rape":
        risk = float(d.get("rape", {}).get("risk") or 0)
        thr  = float(d.get("rape", {}).get("threshold") or SIM_THRESHOLDS["rape"])
        return risk >= thr

    if cat == "blood":
        b  = float(d.get("blood", {}).get("blood")   or 0)
        nb = float(d.get("blood", {}).get("noBlood") or 0)
        thr = float(d.get("blood", {}).get("threshold") or SIM_THRESHOLDS["blood"])
        return b >= thr and b > nb

    if cat == "bestiality":
        risk  = float(d.get("bestiality", {}).get("risk")  or 0)
        furry = float(d.get("bestiality", {}).get("furry") or 0)
        thr   = float(d.get("bestiality", {}).get("threshold") or SIM_THRESHOLDS["bestiality"])
        return risk >= thr and risk > furry

    # underage: complex logic, skip for now
    return None


def compute_stats(data: list) -> dict:
    """Compute accuracy / error per category, same formula as simulator."""
    stats = {}

    for cat in CATEGORIES:
        items = [x for x in data if x.get("category") == cat]
        if not items:
            continue

        pos, neg, unsure = 0, 0, 0
        fp, fn = 0, 0
        items_detail = []

        for item in items:
            blocked = evaluate_siglip2(item)
            if blocked is None:
                continue

            variant = item.get("variant")
            is_uncertain = item.get("uncertain", False)

            if variant == "positive":
                pos += 1
            elif variant == "negative":
                neg += 1

            if is_uncertain:
                unsure += 1
                items_detail.append((item.get("id"), variant, blocked, "unsure"))
                continue

            if variant == "positive" and not blocked:
                fn += 1
                items_detail.append((item.get("id"), variant, blocked, "FAILED+"))
            elif variant == "negative" and blocked:
                fp += 1
                items_detail.append((item.get("id"), variant, blocked, "FAILED-"))
            else:
                items_detail.append((item.get("id"), variant, blocked, "ok"))

        accuracy = (pos - fn) / pos * 100 if pos else 0
        error    = fp / (neg - unsure)  * 100 if (neg - unsure) > 0 else 0

        stats[cat] = {
            "pos": pos, "neg": neg, "unsure": unsure,
            "fn": fn, "fp": fp,
            "accuracy": accuracy, "error": error,
            "items": items_detail,
        }

    return stats


def print_stats(stats: dict, title: str = "Accuracy stats"):
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Category")
    table.add_column("Pos", justify="right")
    table.add_column("Neg", justify="right")
    table.add_column("Failed+", justify="right")
    table.add_column("Failed-", justify="right")
    table.add_column("Accuracy %", justify="right")
    table.add_column("Error %", justify="right")

    for cat, s in stats.items():
        acc_style  = "green" if s["accuracy"] == 100 else ("yellow" if s["accuracy"] >= 95 else "red")
        err_style  = "green" if s["error"] == 0 else ("yellow" if s["error"] <= 5 else "red")
        table.add_row(
            cat,
            str(s["pos"]),
            str(s["neg"]),
            f"[red]{s['fn']}[/red]" if s["fn"] else "0",
            f"[red]{s['fp']}[/red]" if s["fp"] else "0",
            f"[{acc_style}]{s['accuracy']:.0f}%[/{acc_style}]",
            f"[{err_style}]{s['error']:.0f}%[/{err_style}]",
        )
    console.print(table)


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--snapshot",       "-s", required=True, help="Path to snapshot JSON")
@click.option("--category",       "-c", default=None,  help="Filter to specific category for detail view")
@click.option("--ids",            "-i", default=None,  help="Comma-separated test IDs to run through Piper")
@click.option("--patch-failed",   is_flag=True,        help="Auto-detect failed items and patch them")
@click.option("--simulate-only",  is_flag=True,        help="Skip Piper, just compute stats on snapshot")
@click.option("--output",         "-o", default=None,  help="Save patched snapshot to this path")
def main(snapshot, category, ids, patch_failed, simulate_only, output):
    """Run Piper on failed items, patch snapshot, show accuracy stats."""

    # ── Load snapshot ────────────────────────────────────────────────────────
    console.print(f"[cyan]Loading snapshot:[/cyan] {snapshot}")
    with open(snapshot, encoding="utf-8") as f:
        raw = json.load(f)
    data: list = raw if isinstance(raw, list) else (raw.get("items") or list(raw.values())[0])
    console.print(f"  [dim]{len(data)} items total[/dim]")

    # ── Baseline stats ───────────────────────────────────────────────────────
    baseline = compute_stats(data)
    print_stats(baseline, title="📊 Baseline (snapshot as-is)")

    if simulate_only:
        if category:
            s = baseline.get(category)
            if s:
                console.print(f"\n[bold]Detail — {category}:[/bold]")
                for (iid, var, blocked, status) in s["items"]:
                    color = "red" if "FAILED" in status else ("dim" if status == "ok" else "yellow")
                    console.print(f"  [{color}]id={iid}  variant={var}  blocked={blocked}  {status}[/{color}]")
        return

    # ── Determine which IDs to patch ─────────────────────────────────────────
    patch_ids: set[str] = set()

    if ids:
        patch_ids = {x.strip() for x in ids.split(",")}
    elif patch_failed and category:
        # Auto-detect failed items for the given category
        for item in data:
            if item.get("category") != category:
                continue
            blocked = evaluate_siglip2(item)
            if blocked is None:
                continue
            variant = item.get("variant")
            if (variant == "positive" and not blocked) or (variant == "negative" and blocked):
                patch_ids.add(str(item.get("id")))
        console.print(f"[cyan]Auto-detected {len(patch_ids)} failed items:[/cyan] {sorted(patch_ids)}")
    else:
        console.print("[red]Provide --ids or --patch-failed --category[/red]")
        raise SystemExit(1)

    # ── Build id → item map ──────────────────────────────────────────────────
    id_map: dict[str, int] = {}
    for idx, item in enumerate(data):
        id_map[str(item.get("id"))] = idx

    # ── Run Piper for each ID ────────────────────────────────────────────────
    console.print(f"\n[bold]Running {len(patch_ids)} images through Piper...[/bold]")
    patched_data = list(data)  # shallow copy, items will be deepcopied when patched

    with httpx.Client(follow_redirects=False) as client:
        for item_id in sorted(patch_ids):
            idx = id_map.get(item_id)
            if idx is None:
                console.print(f"  [yellow]id={item_id} not found in snapshot[/yellow]")
                continue

            item = patched_data[idx]
            image_url = item.get("media") or item.get("mediaUrl")
            if not image_url:
                console.print(f"  [red]id={item_id} — no media URL[/red]")
                continue

            console.print(f"\n  [bold]id={item_id}[/bold]  ({item.get('variant')}/{item.get('category')})")
            console.print(f"  [dim]{image_url[:80]}...[/dim]")

            try:
                outputs = run_piper(client, image_url)
                if outputs:
                    patched_data[idx] = patch_siglip2(item, outputs)
                    # Print fresh risk score
                    detail_key = DETAIL_KEY_MAP.get(item.get("category"), item.get("category", ""))
                    new_risk = float(
                        patched_data[idx].get("models",{}).get("siglip2",{})
                        .get("details",{}).get(detail_key,{}).get("risk",0) or 0
                    )
                    old_risk = float(
                        item.get("models",{}).get("siglip2",{})
                        .get("details",{}).get(detail_key,{}).get("risk",0) or 0
                    )
                    change = f"{old_risk:.5f} → [bold]{new_risk:.5f}[/bold]"
                    color  = "green" if new_risk > old_risk else ("red" if new_risk < old_risk else "dim")
                    console.print(f"  [cyan]risk ({detail_key}):[/cyan] [{color}]{change}[/{color}]")
                    passed_new = patched_data[idx]["models"]["siglip2"]["passed"]
                    console.print(f"  [cyan]passed:[/cyan] {'[green]yes[/green]' if passed_new else '[red]no[/red]'}")
                else:
                    console.print(f"  [red]No outputs returned from Piper[/red]")
            except Exception as e:
                console.print(f"  [red]Error:[/red] {e}")

    # ── Post-patch stats ─────────────────────────────────────────────────────
    patched_stats = compute_stats(patched_data)
    console.print()
    print_stats(patched_stats, title="🚀 After patching (new Piper scores)")

    # ── Delta summary ────────────────────────────────────────────────────────
    if category and category in baseline and category in patched_stats:
        b = baseline[category]
        p = patched_stats[category]
        delta_acc = p["accuracy"] - b["accuracy"]
        delta_err = p["error"]    - b["error"]
        acc_icon  = "✅" if delta_acc >= 0 else "❌"
        err_icon  = "✅" if delta_err <= 0 else "❌"
        console.print(Panel(
            f"[bold]{category}[/bold]\n"
            f"  Accuracy: {b['accuracy']:.0f}% → [bold]{p['accuracy']:.0f}%[/bold]  "
            f"{acc_icon}  (Δ{delta_acc:+.0f}%)\n"
            f"  Error:    {b['error']:.0f}% → [bold]{p['error']:.0f}%[/bold]  "
            f"{err_icon}  (Δ{delta_err:+.0f}%)\n"
            f"  Failed+:  {b['fn']} → {p['fn']}\n"
            f"  Failed-:  {b['fp']} → {p['fp']}",
            title="Delta",
            border_style="cyan",
        ))

        # Show per-item status
        console.print(f"\n[bold]Per-item status after patch:[/bold]")
        for (iid, var, blocked, status) in p["items"]:
            color = "red" if "FAILED" in status else ("dim" if status == "ok" else "yellow")
            console.print(f"  [{color}]id={iid}  {var}  blocked={blocked}  → {status}[/{color}]")

    # ── Save patched snapshot ─────────────────────────────────────────────────
    out_path = output or snapshot.replace(".json", f"_patched_{datetime.date.today()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(patched_data, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]✅ Patched snapshot saved:[/green] {out_path}")


if __name__ == "__main__":
    main()
