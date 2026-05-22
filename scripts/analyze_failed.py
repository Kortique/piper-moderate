"""
analyze_failed.py
-----------------
1. Fetches test results from mod.artworks.ai API (or reads a local JSON file).
2. Finds failed items: Failed+ (missed positives) and Failed- (false alarms).
3. Sends each failed image URL to Grok via OpenRouter for tag analysis.
4. Saves tag suggestions to suggestions/<category>_<date>.json

Usage:
    # Auto-fetch today's results from mod.artworks.ai API:
    python scripts/analyze_failed.py --category rape

    # Fetch a specific date:
    python scripts/analyze_failed.py --category rape --date 2026-04-13

    # Use a local JSON file instead:
    python scripts/analyze_failed.py --category rape --results results/export.json

    # Analyze only first 5 failed items (quick test):
    python scripts/analyze_failed.py --category rape --limit 5
"""

import sys
import os
import json
import asyncio
import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console()

OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
MOD_API_BASE    = "https://mod.artworks.ai/api"
MOD_PIN         = os.getenv("MOD_PIN", "332211")

# ─── Prompt templates ────────────────────────────────────────────────────────

PROMPT_FAILED_POSITIVE = """You are a computer vision tag expert for a content moderation system using SigLIP-2 image classification.

TASK: This image is a REAL POSITIVE — it belongs to the "{category}" category and should have been BLOCKED, but the model MISSED it (False Negative / Failed+).

CURRENT TOP TAGS that fired (with scores):
{top_tags}

ALL CURRENT TAGS for this category (key → description):
{category_tags}

Analyze the image and answer:
1. What visual elements in the image are characteristic of "{category}"?
2. Which existing tags are too vague and failed to catch this image?
3. What NEW tags would help detect this type of image?

Return ONLY a valid JSON object (no markdown, no explanation outside the JSON):
{{
  "analysis": "brief explanation of why the model missed this image",
  "improve_existing": [
    {{"key": "existing_tag_key", "old_description": "current text", "new_description": "improved text"}}
  ],
  "add_new_tags": [
    {{"key": "{category}_descriptive_name", "description": "precise SigLIP-2 visual prompt"}}
  ],
  "remove_tags": []
}}"""

PROMPT_FAILED_NEGATIVE = """You are a computer vision tag expert for a content moderation system using SigLIP-2 image classification.

TASK: This image is a REAL NEGATIVE — it does NOT belong to the "{category}" category, but the model INCORRECTLY blocked it (False Positive / Failed-).

CURRENT TOP TAGS that fired (with scores):
{top_tags}

ALL CURRENT TAGS for this category (key → description):
{category_tags}

Analyze the image and answer:
1. What visual elements triggered a false detection?
2. Which tags are too broad and fire on safe images?
3. What counter-tags (no_* prefix) would suppress this false positive?

Return ONLY a valid JSON object (no markdown, no explanation outside the JSON):
{{
  "analysis": "brief explanation of why this safe image was incorrectly blocked",
  "improve_existing": [
    {{"key": "existing_tag_key", "old_description": "current text", "new_description": "more precise text"}}
  ],
  "add_new_tags": [
    {{"key": "no_{category}_descriptive_name", "description": "visual prompt for counter-tag"}}
  ],
  "remove_tags": [
    {{"key": "tag_key", "reason": "why this tag causes false positives"}}
  ]
}}"""


# ─── mod.artworks.ai API ──────────────────────────────────────────────────────

def fetch_export(date: str) -> list:
    """Download test export from mod.artworks.ai API."""
    url = f"{MOD_API_BASE}/tests/export"
    console.print(f"[cyan]Fetching from mod.artworks.ai API (date={date})...[/cyan]")
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url, params={"date": date}, headers={"x-pin": MOD_PIN})
        resp.raise_for_status()
    data = resp.json()
    # API may return a list directly or wrapped in a key
    if isinstance(data, list):
        return data
    for key in ("items", "data", "tests", "results"):
        if key in data:
            return data[key]
    # Try first value
    if isinstance(data, dict):
        first = next(iter(data.values()), None)
        if isinstance(first, list):
            return first
    return []


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tags(tags_file: str) -> dict:
    with open(tags_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_category_tags(all_tags: dict, category: str) -> dict:
    prefixes = [f"{category}_", f"no_{category}_"]
    if category == "weapons_and_military":
        prefixes = ["weapons_", "military_", "no_weapon_"]
    elif category == "death_and_murder":
        prefixes = ["death_", "no_death_"]
    elif category == "human_waste":
        prefixes = ["human_waste_", "no_waste_"]
    return {k: v for k, v in all_tags.items() if any(k.startswith(p) for p in prefixes)}


def find_latest_local(results_dir: str, watch_dir: str) -> Path | None:
    candidates = []
    for d in [results_dir, watch_dir]:
        p = Path(d)
        if p.exists():
            candidates.extend(p.glob("*.json"))
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


def get_detail_key(category: str) -> str:
    """
    Return the siglip2 details key for a given category name.
    Category names in the snapshot may differ from the detail keys:
      weapons_and_military → weapons
      death_and_murder     → death
      human_waste          → humanWaste
    Reads from config.yaml if available, falls back to hardcoded map.
    """
    try:
        cfg = load_config()
        cat_cfg = cfg.get("categories", {}).get(category, {})
        if "detail_key" in cat_cfg:
            return cat_cfg["detail_key"]
    except Exception:
        pass
    _FALLBACK = {
        "weapons_and_military": "weapons",
        "death_and_murder":     "death",
        "human_waste":          "humanWaste",
    }
    return _FALLBACK.get(category, category)


def extract_failed_items(data: list, category: str) -> tuple[list, list]:
    """
    Returns (failed_positive, failed_negative).

    Uses per-category risk vs threshold from details.<detail_key>.
    detail_key may differ from category (e.g. weapons_and_military → weapons).
    """
    detail_key = get_detail_key(category)
    failed_pos, failed_neg = [], []

    for item in data:
        if item.get("category") != category:
            continue
        if item.get("uncertain"):
            continue

        variant = item.get("variant")
        siglip2 = item.get("models", {}).get("siglip2") or {}
        details = siglip2.get("details") or {}
        cat_det = details.get(detail_key) or {}

        # Category-specific blocked determination
        if cat_det and "risk" in cat_det and "threshold" in cat_det:
            blocked = float(cat_det["risk"]) >= float(cat_det["threshold"])
        elif cat_det and "score" in cat_det and "threshold" in cat_det:
            blocked = float(cat_det["score"]) >= float(cat_det["threshold"])
        elif "passed" in siglip2:
            # Fallback to global flag only when no per-category data available
            blocked = not siglip2["passed"]
        else:
            s_labels = siglip2.get("labels") or []
            if isinstance(s_labels, str):
                s_labels = [s_labels]
            blocked = detail_key in s_labels

        if variant == "positive" and not blocked:
            failed_pos.append(item)
        elif variant == "negative" and blocked:
            failed_neg.append(item)

    return failed_pos, failed_neg


def get_top_tags(item: dict, category: str, n: int = 15) -> str:
    details  = item.get("models", {}).get("siglip2", {}).get("details") or {}
    # Use the correct detail_key (may differ from category name)
    detail_key = get_detail_key(category)
    cat_det  = details.get(detail_key) or details.get(category) or {}
    labels_d = {}

    if isinstance(cat_det, dict):
        raw_labels = cat_det.get("labels") or {}
        if isinstance(raw_labels, dict):
            for sub in raw_labels.values():
                if isinstance(sub, dict):
                    labels_d.update(sub)
            if not labels_d:
                labels_d = raw_labels

    if not labels_d:
        return "(no per-tag scores in test data)"

    sorted_tags = sorted(labels_d.items(), key=lambda x: float(x[1] or 0), reverse=True)[:n]
    return "\n".join(f"  {k}: {v}" for k, v in sorted_tags)


def get_image_url(item: dict) -> str | None:
    """Extract direct image URL from test item.

    mod.artworks.ai API returns:
      - `media`  — direct object-storage URL (e.g. fsn1.your-objectstorage.com/...)
      - `url`    — Label Studio proxy URL (requires auth, not usable by Grok)
    Always prefer `media` over `url`.
    """
    # Primary: direct CDN/object-storage URL
    for field in ["media", "mediaUrl", "image_url", "imageUrl", "image", "src"]:
        val = item.get(field)
        if val and isinstance(val, str) and val.startswith("http"):
            return val
    # Avoid `url` field — it's the Label Studio proxy that requires auth
    # Only use it as absolute last resort if nothing else found
    meta = item.get("meta") or item.get("metadata") or {}
    for field in ["media", "mediaUrl", "imageUrl", "image_url", "url"]:
        val = meta.get(field)
        if val and isinstance(val, str) and val.startswith("http"):
            return val
    return None


# ─── Grok API ────────────────────────────────────────────────────────────────

async def call_grok(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    image_url: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Kortique/piper-moderate",
        "X-Title": "piper-moderate",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
    }

    async with semaphore:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90.0)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())


async def analyze_batch(
    items: list,
    item_type: str,
    category: str,
    category_tags: dict,
    api_key: str,
    model: str,
    max_tokens: int,
    concurrency: int,
) -> list:
    semaphore = asyncio.Semaphore(concurrency)
    template  = PROMPT_FAILED_POSITIVE if item_type == "positive" else PROMPT_FAILED_NEGATIVE
    cat_tags_str = json.dumps(category_tags, indent=2, ensure_ascii=False)
    results = []

    async with httpx.AsyncClient() as client:

        async def process_one(item):
            item_id   = str(item.get("id") or item.get("_id") or "?")
            image_url = get_image_url(item)
            if not image_url:
                return {"item_id": item_id, "error": "no_image_url", "type": item_type}

            top_tags = get_top_tags(item, category)
            prompt   = template.format(
                category=category,
                top_tags=top_tags,
                category_tags=cat_tags_str,
            )
            try:
                suggestion = await call_grok(client, api_key, model, image_url, prompt, max_tokens, semaphore)
                return {"item_id": item_id, "image_url": image_url, "type": item_type, "suggestion": suggestion}
            except json.JSONDecodeError as e:
                return {"item_id": item_id, "image_url": image_url, "type": item_type, "error": f"json_parse: {e}"}
            except httpx.HTTPStatusError as e:
                return {"item_id": item_id, "image_url": image_url, "type": item_type, "error": f"http_{e.response.status_code}: {e.response.text[:200]}"}
            except Exception as e:
                return {"item_id": item_id, "image_url": image_url, "type": item_type, "error": str(e)}

        aws = [process_one(item) for item in items]

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            tid = progress.add_task(f"[yellow]Analyzing {item_type} failures...", total=len(aws))
            for coro in asyncio.as_completed(aws):
                r = await coro
                results.append(r)
                icon = "✓" if "suggestion" in r else "✗"
                console.log(f"  {icon} id={r['item_id']} {'(no url)' if r.get('error') == 'no_image_url' else ''}")
                progress.advance(tid)

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--category", "-c", required=True)
@click.option("--date", "-d", default=None, help="Date to fetch (YYYY-MM-DD). Defaults to today.")
@click.option("--results", "-r", default=None, help="Local JSON file. Skips API fetch if set.")
@click.option("--limit",   "-l", default=0,  type=int, help="Analyze only first N failed items per type (0=all)")
@click.option("--type", "fail_type", default="both",
              type=click.Choice(["positive", "negative", "both"]))
def main(category, date, results, limit, fail_type):
    """Analyze failed test items with Grok and generate tag suggestions."""

    cfg        = load_config()
    api_key    = os.getenv("OPENROUTER_API_KEY")
    model      = cfg["openrouter"]["model"]
    max_tokens = cfg["openrouter"]["max_tokens"]
    concurrency= cfg["openrouter"]["concurrency"]
    watch_dir  = os.getenv("EXPORT_WATCH_DIR", "results")

    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set in .env[/red]")
        raise SystemExit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    if results:
        console.print(f"[cyan]Loading local file:[/cyan] {results}")
        with open(results, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = raw if isinstance(raw, list) else (raw.get("items") or raw.get("data") or list(raw.values())[0])
    else:
        target_date = date or datetime.date.today().isoformat()
        try:
            data = fetch_export(target_date)
            console.print(f"[green]✓ Fetched {len(data)} items from API (date={target_date})[/green]")
        except httpx.HTTPStatusError as e:
            console.print(f"[red]API error {e.response.status_code}[/red] — falling back to local files")
            local = find_latest_local(cfg.get("default_results_dir","results"), watch_dir)
            if not local:
                console.print("[red]No local export found either. Export manually from mod.artworks.ai[/red]")
                raise SystemExit(1)
            console.print(f"[cyan]Using:[/cyan] {local}")
            with open(local, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = data.get("items") or list(data.values())[0]

    # ── Filter ─────────────────────────────────────────────────────────────
    all_tags      = load_tags(cfg["tags_file"])
    category_tags = get_category_tags(all_tags, category)
    failed_pos, failed_neg = extract_failed_items(data, category)

    pos_n = min(limit, len(failed_pos)) if limit else len(failed_pos)
    neg_n = min(limit, len(failed_neg)) if limit else len(failed_neg)

    table = Table(title=f"Failed items — {category}", show_header=True, header_style="bold")
    table.add_column("Type")
    table.add_column("Total", justify="right")
    table.add_column("Will analyze", justify="right")
    table.add_row("Failed+ (missed threats)",  str(len(failed_pos)), str(pos_n) if fail_type in ("positive","both") else "0")
    table.add_row("Failed- (false alarms)",    str(len(failed_neg)), str(neg_n) if fail_type in ("negative","both") else "0")
    console.print(table)

    if pos_n == 0 and neg_n == 0:
        console.print("[green]No failed items — category looks clean![/green]")
        return

    # ── Analyze ────────────────────────────────────────────────────────────
    all_results = []

    if fail_type in ("positive", "both") and failed_pos:
        batch = failed_pos[:limit] if limit else failed_pos
        r = asyncio.run(analyze_batch(batch, "positive", category, category_tags, api_key, model, max_tokens, concurrency))
        all_results.extend(r)

    if fail_type in ("negative", "both") and failed_neg:
        batch = failed_neg[:limit] if limit else failed_neg
        r = asyncio.run(analyze_batch(batch, "negative", category, category_tags, api_key, model, max_tokens, concurrency))
        all_results.extend(r)

    # ── Save ───────────────────────────────────────────────────────────────
    today    = datetime.date.today().isoformat()
    out_path = Path("suggestions") / f"{category}_{today}.json"
    out_path.parent.mkdir(exist_ok=True)

    output = {
        "category": category,
        "date": today,
        "model": model,
        "stats": {
            "failed_positive_total": len(failed_pos),
            "failed_negative_total": len(failed_neg),
            "analyzed": len(all_results),
            "success": sum(1 for r in all_results if "suggestion" in r),
            "errors":  sum(1 for r in all_results if "error" in r),
        },
        "results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    console.print(Panel(
        f"[green]✅ Done[/green]  success={output['stats']['success']}  errors={output['stats']['errors']}\n"
        f"Saved: [bold]{out_path}[/bold]\n\n"
        f"Next: [bold]python scripts/update_tags.py --suggestions {out_path}[/bold]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
