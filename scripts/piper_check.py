"""
piper_check.py
--------------
Run a single image through the Piper SigLIP-2 pipeline and show results.

Usage:
    python scripts/piper_check.py --url https://fsn1.your-objectstorage.com/artworks-assets/test-1768.jpg
    python scripts/piper_check.py --url <image_url> --providers siglip2,hive
    python scripts/piper_check.py --id 1058   # fetch media URL from mod.artworks.ai by test ID

API:
    POST https://piper-next.artworks.ai/api/projects/{PROJECT}/launch
         body: {"inputs": {"image": <url>, "providers": [...]}}
    GET  https://piper-next.artworks.ai/api/launches/{run_id}/state
         → poll until outputs appear
"""

import sys
import os
import json
import time
import asyncio
from pathlib import Path

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

MOD_API_BASE  = "https://mod.artworks.ai/api"
MOD_PIN       = os.getenv("MOD_PIN", "332211")


def get_headers():
    token = PIPER_TOKEN
    if not token:
        console.print("[red]PIPER_TOKEN not set in .env[/red]")
        raise SystemExit(1)
    return {
        "User-Token": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def fetch_media_url_by_id(test_id: str) -> str:
    """Fetch the direct media URL for a test item from mod.artworks.ai."""
    with httpx.Client(timeout=15) as client:
        r = client.get(
            f"{MOD_API_BASE}/tests/export",
            params={"date": "2026-04-14"},
            headers={"x-pin": MOD_PIN},
        )
        items = r.json()
        for item in items:
            if str(item.get("id")) == str(test_id):
                url = item.get("media")
                if url:
                    return url
                raise SystemExit(f"Item {test_id} has no 'media' field")
        raise SystemExit(f"Test item {test_id} not found in today's export")


def run_piper(image_url: str, providers: list[str]) -> dict:
    """Launch pipeline and poll until complete. Returns outputs dict."""
    headers = get_headers()

    with httpx.Client(timeout=60, follow_redirects=False) as client:
        # 1. Launch
        r = client.post(
            f"{PIPER_BASE}/projects/{PIPER_PROJECT}/launch",
            headers=headers,
            json={"inputs": {"image": image_url, "providers": providers}},
        )
        r.raise_for_status()
        run_id = r.json()["_id"]
        console.print(f"[dim]  Launched run [cyan]{run_id}[/cyan][/dim]")

        # 2. Poll state
        for attempt in range(40):
            time.sleep(3)
            rs = client.get(f"{PIPER_BASE}/launches/{run_id}/state", headers=headers)
            if rs.status_code != 200:
                continue
            state = rs.json()
            outputs = state.get("outputs") or {}
            errors  = state.get("errors") or []

            if errors:
                console.print(f"[red]Pipeline errors:[/red] {errors}")
                break

            # Done when siglip2 outputs appear (or all requested providers)
            if any(k.startswith("siglip2_") for k in outputs):
                return outputs

        console.print("[yellow]Timed out waiting for results[/yellow]")
        return {}


def print_results(outputs: dict, image_url: str):
    """Display siglip2 results in a readable table."""
    details = outputs.get("siglip2_details") or {}
    labels  = outputs.get("siglip2_labels") or []
    passed  = outputs.get("siglip2_passed")

    console.print(Panel(
        f"[bold]Image:[/bold] {image_url}\n"
        f"[bold]Triggered categories:[/bold] {', '.join(labels) if labels else '—'}\n"
        f"[bold]Passed:[/bold] {'[green]Yes[/green]' if passed else '[red]No — blocked[/red]'}",
        title="SigLIP-2 Result",
        border_style="cyan",
    ))

    if not details:
        console.print("[yellow]No siglip2_details in output[/yellow]")
        return

    # Categories with risk/score above 0
    table = Table(show_header=True, title="Category scores")
    table.add_column("Category")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status")
    table.add_column("Top tags")

    for cat, det in sorted(details.items()):
        if not isinstance(det, dict):
            continue
        risk      = float(det.get("risk") or det.get("score") or 0)
        threshold = float(det.get("threshold") or 0)
        if risk == 0 and not labels:
            continue

        blocked = risk >= threshold
        status  = "[red]BLOCKED[/red]" if blocked else "[green]ok[/green]"

        # Gather top 5 tags
        top_tags = []
        raw_labels = det.get("labels") or {}
        flat = {}
        if isinstance(raw_labels, dict):
            for v in raw_labels.values():
                if isinstance(v, dict):
                    flat.update(v)
            if not flat:
                flat = {k: v for k, v in raw_labels.items() if isinstance(v, (int, float))}
        top_tags = sorted(flat.items(), key=lambda x: float(x[1] or 0), reverse=True)[:5]
        tags_str = "  ".join(f"{k}={v:.4f}" for k, v in top_tags) if top_tags else "—"

        table.add_row(
            cat,
            f"{risk:.5f}",
            f"{threshold}",
            status,
            tags_str,
        )

    console.print(table)


@click.command()
@click.option("--url", "-u", default=None, help="Direct image URL to analyze")
@click.option("--id",  "-i", "test_id", default=None, help="Test item ID from mod.artworks.ai")
@click.option("--providers", default="siglip2", show_default=True,
              help="Comma-separated providers: siglip2,hive,qwen3")
@click.option("--json-output", "json_out", is_flag=True, help="Print raw JSON output")
def main(url, test_id, providers, json_out):
    """Run an image through the Piper pipeline and show SigLIP-2 results."""

    if not url and not test_id:
        console.print("[red]Provide --url or --id[/red]")
        raise SystemExit(1)

    image_url = url
    if test_id:
        console.print(f"[dim]Fetching media URL for test id={test_id}...[/dim]")
        image_url = fetch_media_url_by_id(test_id)
        console.print(f"[dim]  → {image_url}[/dim]")

    provider_list = [p.strip() for p in providers.split(",")]

    console.print(f"\n[bold]Running pipeline...[/bold]  image={image_url[:80]}...")
    outputs = run_piper(image_url, provider_list)

    if json_out:
        console.print_json(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        print_results(outputs, image_url)


if __name__ == "__main__":
    main()
