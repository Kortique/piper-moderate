"""
update_tags.py
--------------
Reviews Grok suggestions and applies approved changes to data/tags.json.

Usage:
    python scripts/update_tags.py --suggestions suggestions/underage_2026-04-13.json
    python scripts/update_tags.py --suggestions suggestions/underage_2026-04-13.json --auto   # no confirmation

Outputs an updated data/tags.json (old version backed up as data/tags.<date>.bak.json).
"""

import sys
import os
import json
import re
import datetime
import shutil
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tags(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tag_prefix(key: str) -> str:
    """Return category prefix for grouping: 'no_drugs' or 'drugs', etc."""
    m = re.match(r'^(no_[a-z]+)_', key)
    if m:
        return m.group(1)
    m = re.match(r'^([a-z]+)_', key)
    if m:
        return m.group(1)
    return key


def save_tags(tags_dict: dict, keys_ordered: list, path: str):
    """Write tags.json with blank lines between category groups."""
    lines = ['{']
    prev_prefix = None
    total = len(keys_ordered)
    for idx, key in enumerate(keys_ordered):
        prefix = get_tag_prefix(key)
        if prev_prefix is not None and prefix != prev_prefix:
            lines.append('')  # blank line between groups
        value = tags_dict[key]
        comma = ',' if idx < total - 1 else ''
        lines.append(f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}{comma}')
        prev_prefix = prefix
    lines.append('}')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def insert_tag(keys: list, new_key: str) -> list:
    """
    Insert new_key into keys at the correct position within its category block.

    Rules:
    - For 'drugs_xyz'    → insert after the last 'drugs_*' key (not 'no_drugs_*')
    - For 'no_drugs_xyz' → insert after the last 'no_drugs_*' key
    - If no existing keys with that prefix, insert after the last key of the
      parent category (e.g. for 'no_drugs_xyz' fall back to last 'drugs_*')
    - Last resort: append at end
    """
    prefix = get_tag_prefix(new_key)

    # Find last position of same prefix
    insert_after = None
    for i, k in enumerate(keys):
        if get_tag_prefix(k) == prefix:
            insert_after = i

    # Fallback: for no_X prefix, try to find last X_ key
    if insert_after is None and prefix.startswith('no_'):
        parent = prefix[3:]  # strip 'no_'
        for i, k in enumerate(keys):
            if get_tag_prefix(k) == parent:
                insert_after = i

    if insert_after is not None:
        keys.insert(insert_after + 1, new_key)
    else:
        keys.append(new_key)

    return keys


def aggregate_suggestions(results: list) -> dict:
    """
    Aggregate all Grok suggestions across items.
    Returns:
      {
        "improve_existing": {key: [new_descriptions...]},
        "add_new_tags":     {key: [descriptions...]},
        "remove_tags":      {key: [reasons...]},
      }
    """
    agg = {
        "improve_existing": defaultdict(list),
        "add_new_tags":     defaultdict(list),
        "remove_tags":      defaultdict(list),
    }

    for r in results:
        s = r.get("suggestion")
        if not s:
            continue

        for item in s.get("improve_existing") or []:
            key = item.get("key")
            new_desc = item.get("new_description")
            if key and new_desc:
                agg["improve_existing"][key].append(new_desc)

        for item in s.get("add_new_tags") or []:
            key = item.get("key")
            desc = item.get("description")
            if key and desc:
                agg["add_new_tags"][key].append(desc)

        for item in s.get("remove_tags") or []:
            key = item.get("key")
            reason = item.get("reason", "")
            if key:
                agg["remove_tags"][key].append(reason)

    return agg


def pick_best_description(descriptions: list) -> str:
    """When multiple VLM runs suggest descriptions for the same key, pick the most common or longest."""
    if len(descriptions) == 1:
        return descriptions[0]
    # Pick the one that appears most often; tie-break by length
    from collections import Counter
    c = Counter(descriptions)
    most_common = c.most_common()
    top_count = most_common[0][1]
    candidates = [d for d, cnt in most_common if cnt == top_count]
    return max(candidates, key=len)


@click.command()
@click.option("--suggestions", "-s", required=True, help="Path to suggestions JSON from analyze_failed.py")
@click.option("--auto", is_flag=True, help="Apply all suggestions without confirmation prompts")
@click.option("--dry-run", is_flag=True, help="Show changes but don't write anything")
def main(suggestions, auto, dry_run):
    """Review and apply tag suggestions to data/tags.json."""

    cfg = load_config()
    tags_file = cfg["tags_file"]

    # Load suggestions
    with open(suggestions, "r", encoding="utf-8") as f:
        sug_data = json.load(f)

    category = sug_data.get("category", "?")
    results  = sug_data.get("results", [])
    success  = [r for r in results if "suggestion" in r]

    console.print(Panel(
        f"Category: [bold cyan]{category}[/bold cyan]\n"
        f"Date: {sug_data.get('date')}\n"
        f"Model: {sug_data.get('model')}\n"
        f"Successful analyses: {len(success)} / {len(results)}",
        title="Suggestions file",
        border_style="cyan"
    ))

    if not success:
        console.print("[yellow]No successful suggestions to apply.[/yellow]")
        return

    agg = aggregate_suggestions(success)

    tags = load_tags(tags_file)
    keys = list(tags.keys())   # maintain explicit order for grouped writing
    changes_applied = []

    # ── Improve existing tags ─────────────────────────────────────────────────
    if agg["improve_existing"]:
        console.print(f"\n[bold yellow]→ Improve existing tags ({len(agg['improve_existing'])})[/bold yellow]")
        table = Table(show_header=True)
        table.add_column("Key")
        table.add_column("Current description")
        table.add_column("Suggested description")
        table.add_column("Votes", justify="right")

        for key, descs in sorted(agg["improve_existing"].items(), key=lambda x: -len(x[1])):
            best = pick_best_description(descs)
            current = tags.get(key, "[NOT IN TAGS]")
            table.add_row(key, current[:60] + ("…" if len(current) > 60 else ""),
                          best[:60] + ("…" if len(best) > 60 else ""), str(len(descs)))
        console.print(table)

        for key, descs in sorted(agg["improve_existing"].items(), key=lambda x: -len(x[1])):
            best = pick_best_description(descs)
            if key not in tags:
                console.print(f"[dim]  Skipping {key} — not in current tags.json[/dim]")
                continue
            do_apply = auto or dry_run or Confirm.ask(f"  Apply improvement to [cyan]{key}[/cyan]?")
            if do_apply and not dry_run:
                old = tags[key]
                tags[key] = best
                changes_applied.append({"action": "improve", "key": key, "old": old, "new": best})

    # ── Add new tags ──────────────────────────────────────────────────────────
    if agg["add_new_tags"]:
        console.print(f"\n[bold green]→ Add new tags ({len(agg['add_new_tags'])})[/bold green]")
        table = Table(show_header=True)
        table.add_column("Key")
        table.add_column("Suggested description")
        table.add_column("Votes", justify="right")
        table.add_column("Status")

        for key, descs in sorted(agg["add_new_tags"].items(), key=lambda x: -len(x[1])):
            best = pick_best_description(descs)
            status = "[dim]already exists[/dim]" if key in tags else "[green]new[/green]"
            table.add_row(key, best[:70] + ("…" if len(best) > 70 else ""), str(len(descs)), status)
        console.print(table)

        # Balancing pairs: for each positive category, a sibling no_* category
        # that prevents SigLIP-2 from over-triggering on visually similar safe content.
        BALANCE_PAIRS = {
            "drugs":     "no_drugs",
            "underage":  "adult",
            "rape":      "no_rape",
            "weapons":   "no_weapon",
            "blood":     "no_blood",
            "death":     "no_death",
        }

        missing_balance: list[tuple[str, str]] = []  # (new_key, expected_counter_prefix)

        for key, descs in sorted(agg["add_new_tags"].items(), key=lambda x: -len(x[1])):
            best = pick_best_description(descs)
            if key in tags:
                console.print(f"[dim]  {key} already exists — skipping[/dim]")
                continue
            do_apply = auto or dry_run or Confirm.ask(f"  Add new tag [green]{key}[/green]?")
            if do_apply and not dry_run:
                tags[key] = best
                insert_tag(keys, key)  # insert at correct position within category block
                changes_applied.append({"action": "add", "key": key, "description": best})

                # Check balance: positive tag needs a corresponding no_* counter-tag
                prefix = get_tag_prefix(key)
                counter_prefix = BALANCE_PAIRS.get(prefix)
                if counter_prefix:
                    # Check whether any no_* tag already covers this concept
                    suffix = key[len(prefix) + 1:]  # e.g. 'cocaine_pile' from 'drugs_cocaine_pile'
                    expected = f"{counter_prefix}_{suffix}"
                    if expected not in tags:
                        missing_balance.append((key, counter_prefix))

        if missing_balance:
            console.print(
                f"\n[bold yellow]⚠  Balance check: {len(missing_balance)} new positive tag(s) "
                f"have no counter-tag in the sibling no_* category.[/bold yellow]"
            )
            console.print("[yellow]SigLIP-2 needs both sides to avoid over-triggering on safe images.[/yellow]")
            for pos_key, ctr_prefix in missing_balance:
                console.print(f"  [cyan]{pos_key}[/cyan] → add a matching [yellow]{ctr_prefix}_…[/yellow] tag")

    # ── Remove tags ───────────────────────────────────────────────────────────
    if agg["remove_tags"]:
        console.print(f"\n[bold red]→ Remove tags ({len(agg['remove_tags'])})[/bold red]")
        table = Table(show_header=True)
        table.add_column("Key")
        table.add_column("Reason (most common)")
        table.add_column("Votes", justify="right")

        for key, reasons in sorted(agg["remove_tags"].items(), key=lambda x: -len(x[1])):
            top_reason = pick_best_description(reasons)
            table.add_row(key, top_reason[:80], str(len(reasons)))
        console.print(table)

        for key, reasons in sorted(agg["remove_tags"].items(), key=lambda x: -len(x[1])):
            if key not in tags:
                continue
            top_reason = pick_best_description(reasons)
            do_apply = auto or dry_run or Confirm.ask(
                f"  [red]Remove[/red] tag [red]{key}[/red]?\n  Reason: {top_reason[:100]}"
            )
            if do_apply and not dry_run:
                old_desc = tags.pop(key, None)
                if key in keys:
                    keys.remove(key)
                changes_applied.append({"action": "remove", "key": key, "old": old_desc})

    # ── Save ──────────────────────────────────────────────────────────────────
    if dry_run:
        console.print("\n[yellow]Dry run — no files written.[/yellow]")
        return

    if not changes_applied:
        console.print("\n[yellow]No changes selected.[/yellow]")
        return

    # Backup old tags
    today = datetime.date.today().isoformat()
    bak = Path(tags_file).with_suffix(f".{today}.bak.json")
    shutil.copy(tags_file, bak)

    save_tags(tags, keys, tags_file)

    # Save changelog
    changelog_path = Path("suggestions") / f"{category}_{today}_applied.json"
    with open(changelog_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": today,
            "category": category,
            "source": suggestions,
            "changes": changes_applied,
        }, f, indent=2, ensure_ascii=False)

    console.print(Panel(
        f"[green]✅ Done![/green]\n"
        f"  Applied {len(changes_applied)} changes to [bold]{tags_file}[/bold]\n"
        f"  Backup: [dim]{bak}[/dim]\n"
        f"  Changelog: [bold]{changelog_path}[/bold]\n\n"
        f"Next step: re-run moderation to verify improvements:\n"
        f"  [bold]python scripts/run_category.py --category {category}[/bold]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
