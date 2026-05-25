"""
run_category.py
---------------
Triggers the n8n agent to reset and re-run moderation for a given category.

Usage:
    python scripts/run_category.py --category underage
    python scripts/run_category.py --category rape
    python scripts/run_category.py --list

After running, wait for the agent to respond "✅ Reset complete: N <category> tests were reset",
then go to https://mod.artworks.ai/, click Refresh, then Export to download results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--category", "-c", help="Category to reset (e.g. underage, rape, blood)")
@click.option("--list", "list_categories", is_flag=True, help="List available categories")
@click.option("--message", "-m", default=None, help="Override: send a raw message to the agent")
def main(category, list_categories, message):
    """Trigger n8n agent to re-run moderation for a category."""

    cfg = load_config()
    categories = cfg.get("categories", {})

    if list_categories:
        console.print("\n[bold cyan]Available categories:[/bold cyan]")
        for name, info in categories.items():
            console.print(f"  [green]{name}[/green] → command: [yellow]{info['reset_command']}[/yellow]")
        return

    if not category and not message:
        console.print("[red]Error:[/red] provide --category or --message")
        raise SystemExit(1)

    # Resolve message to send
    if message:
        msg_to_send = message
    else:
        cat_cfg = categories.get(category)
        if not cat_cfg:
            console.print(f"[red]Unknown category:[/red] {category}")
            console.print(f"Available: {', '.join(categories.keys())}")
            raise SystemExit(1)
        msg_to_send = cat_cfg["reset_command"]

    # Load credentials
    webhook_url = os.getenv("N8N_WEBHOOK_URL")
    n8n_user    = os.getenv("N8N_USER")
    n8n_pass    = os.getenv("N8N_PASSWORD")

    if not webhook_url:
        console.print("[red]N8N_WEBHOOK_URL not set in .env[/red]")
        raise SystemExit(1)

    console.print(Panel(
        f"Sending command: [bold yellow]{msg_to_send}[/bold yellow]\n"
        f"Endpoint: {webhook_url}",
        title="[bold]Run Category[/bold]",
        border_style="cyan"
    ))

    auth = (n8n_user, n8n_pass) if n8n_user and n8n_pass else None

    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(
                webhook_url,
                json={"message": msg_to_send, "chatInput": msg_to_send},
                auth=auth,
            )
            resp.raise_for_status()

        console.print(f"\n[green]✅ Agent responded (HTTP {resp.status_code}):[/green]")
        try:
            data = resp.json()
            # n8n chat webhooks typically return {"output": "..."}
            output = data.get("output") or data.get("text") or str(data)
            console.print(f"[bold white]{output}[/bold white]")
        except Exception:
            console.print(resp.text[:500])

        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Wait until you see '✅ Reset complete: N tests were reset'")
        console.print("  2. Open [link=https://mod.artworks.ai/]https://mod.artworks.ai/[/link]")
        console.print("  3. Click [bold]Refresh[/bold] → [bold]Export[/bold]")
        console.print("  4. Run: [bold]python scripts/analyze_failed.py --category " + (category or "?") + "[/bold]")

    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP error:[/red] {e.response.status_code} — {e.response.text[:300]}")
        raise SystemExit(1)
    except httpx.RequestError as e:
        console.print(f"[red]Connection error:[/red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
