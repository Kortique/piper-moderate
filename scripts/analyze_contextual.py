"""
analyze_contextual.py
---------------------
Enhanced Grok analysis with full tag history, before/after scores,
threshold analysis, and balancing constraints.

Usage:
    python scripts/analyze_contextual.py \
        --category weapons_and_military \
        --snapshot results/tests-2026-04-14-patched.json \
        --ids 696,697
"""

import sys, os, json, asyncio, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx, click, yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MOD_API_BASE   = "https://mod.artworks.ai/api"
MOD_PIN        = os.getenv("MOD_PIN", "332211")


# ─── Context builders ────────────────────────────────────────────────────────

def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_tags(tags_file: str) -> dict:
    with open(tags_file, encoding="utf-8") as f:
        return json.load(f)

def get_detail_key(category: str) -> str:
    try:
        cfg = load_config()
        cat_cfg = cfg.get("categories", {}).get(category, {})
        if "detail_key" in cat_cfg:
            return cat_cfg["detail_key"]
    except Exception:
        pass
    return {"weapons_and_military": "weapons", "death_and_murder": "death",
            "human_waste": "humanWaste"}.get(category, category)

def get_category_tags(all_tags: dict, category: str) -> dict:
    prefixes = [f"{category}_", f"no_{category}_"]
    if category == "weapons_and_military":
        prefixes = ["weapons_", "military_", "no_weapon_"]
    elif category == "death_and_murder":
        prefixes = ["death_", "no_death_"]
    elif category == "human_waste":
        prefixes = ["human_waste_", "no_waste_"]
    return {k: v for k, v in all_tags.items() if any(k.startswith(p) for p in prefixes)}

def get_image_url(item: dict) -> str | None:
    for field in ["media", "mediaUrl", "image_url", "imageUrl", "image", "src"]:
        val = item.get(field)
        if val and isinstance(val, str) and val.startswith("http"):
            return val
    return None

def load_changelog(category: str) -> list[dict]:
    """Load the most recent applied changelog for this category."""
    base = Path("suggestions")
    pattern = f"{category}_*_applied.json"
    files = sorted(base.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return []
    with open(files[0], encoding="utf-8") as f:
        d = json.load(f)
    return d.get("changes", [])

def compute_all_scores(data: list, category: str, detail_key: str) -> list[dict]:
    """Return all items for category with their siglip2 risk scores."""
    items_out = []
    for item in data:
        if item.get("category") != category:
            continue
        det = item.get("models",{}).get("siglip2",{}).get("details",{}).get(detail_key,{})
        risk = float(det.get("risk",0) or 0)
        thr  = float(det.get("threshold", 0.01))
        blocked = risk >= thr
        variant = item.get("variant")
        is_ok = (variant == "positive" and blocked) or (variant == "negative" and not blocked)
        top_labels = {}
        raw_labels = det.get("labels") or {}
        if isinstance(raw_labels, dict):
            top_labels = {k: float(v or 0) for k, v in raw_labels.items() if float(v or 0) > 0}
        items_out.append({
            "id": item["id"],
            "variant": variant,
            "risk": risk,
            "threshold": thr,
            "blocked": blocked,
            "is_ok": is_ok,
            "top_labels": dict(sorted(top_labels.items(), key=lambda x: x[1], reverse=True)[:5]),
        })
    return sorted(items_out, key=lambda x: x["id"])

def threshold_analysis(all_scores: list) -> dict:
    """Find the threshold range and report on the gap between pos/neg scores."""
    pos = sorted([x["risk"] for x in all_scores if x["variant"] == "positive"])
    neg = sorted([x["risk"] for x in all_scores if x["variant"] == "negative"])

    min_pos = min(pos) if pos else 0
    max_neg_fail = max([x["risk"] for x in all_scores
                        if x["variant"] == "negative" and x["blocked"]], default=0)
    # Optimal threshold: block all positives, unblock all negatives
    # If min_pos > max_neg_fail: a perfect threshold exists between max_neg_fail and min_pos
    gap = min_pos - max_neg_fail
    return {
        "min_positive": min_pos,
        "max_failing_negative": max_neg_fail,
        "gap": gap,
        "perfect_threshold_possible": gap > 0,
        "suggested_threshold": (min_pos + max_neg_fail) / 2 if gap > 0 else None,
        "all_pos_sorted": pos,
        "all_neg_sorted": neg,
    }


# ─── Prompt builder ──────────────────────────────────────────────────────────

CONTEXTUAL_PROMPT = """You are a computer vision tag expert for a content moderation system using SigLIP-2 image classification.

## CONTEXT: What SigLIP-2 does
SigLIP-2 compares an image against a vocabulary of text prompts ("tags"). Each tag gets a score 0–1.
The final **risk** for a category = combined probability from ALL positive tags in that category.
Counter-tags (no_* prefix) are SEPARATE category scores that are NOT subtracted from risk directly —
they work by giving the model vocabulary to recognize safe images. Improving counter-tags means
making positive tags more specific, so they fire less on safe content.

## TASK TYPE
This image is a REAL NEGATIVE — it does NOT contain "{category}" but the model INCORRECTLY blocked it (False Positive / Failed-).

## ITEM INFO
- ID: {item_id}
- Category: {category}
- Variant: negative (safe image — SHOULD PASS, should NOT be blocked)
- Current risk: {current_risk} (threshold: {threshold}) → INCORRECTLY BLOCKED

## TAGS CURRENTLY FIRING ON THIS IMAGE (top scores)
{top_tags_firing}

## ALL CURRENT TAGS FOR THIS CATEGORY
{all_category_tags}

## TAG CHANGE HISTORY (last round of improvements)
{changelog}

## FULL SCORE DISTRIBUTION — ALL ITEMS IN CATEGORY (post last-round changes)
This shows you what each item scores now so you don't break what's working.
{all_scores_table}

## THRESHOLD ANALYSIS
Current threshold: {threshold}
Minimum positive risk: {min_positive:.5f} (id={min_positive_id}) — this is the LOWEST positive score we must keep
Maximum failing negative risk: {max_neg_fail:.5f} (id={max_neg_fail_id})
GAP (min_pos - max_neg_fail): {gap:.5f}

{threshold_advice}

## YOUR TASK
1. **Analyze** the image carefully. What does it actually show?
2. **Explain** why the tags listed under "TAGS CURRENTLY FIRING" are triggering on this safe image — what visual similarity causes them to score high?
3. **Review the changelog** — did any of the last round's improvements make things worse for this image?
4. **Propose improvements** with these constraints:
   - Every new no_* counter-tag you add MUST have a matching positive tag in the pair (weapons_xyz ↔ no_weapon_xyz)
   - You may narrow existing positive tags to be MORE SPECIFIC so they stop firing on toys/replicas
   - Do NOT remove critical positive tags that are currently catching real threats (check the scores table)
   - New descriptions must be visually precise SigLIP-2 prompts (describe exact visual features)
   - Target: reduce risk for this image below {target_risk:.5f}

Return ONLY valid JSON (no markdown, no comments outside JSON):
{{
  "analysis": "what the image shows and why it triggers false positive",
  "changelog_review": "assessment of how last round changes affected this image",
  "threshold_recommendation": {{
    "suggested_threshold": <number or null>,
    "reasoning": "why this threshold (or why not to change it)"
  }},
  "improve_existing": [
    {{"key": "tag_key", "old_description": "current", "new_description": "improved — more specific to avoid firing on toys"}}
  ],
  "add_new_tags": [
    {{"key": "no_weapon_descriptive_name", "description": "precise visual prompt for safe content"}}
  ],
  "remove_tags": []
}}"""


def build_prompt(item: dict, category: str, detail_key: str,
                 all_category_tags: dict, changelog: list,
                 all_scores: list, thr_analysis: dict) -> str:

    item_id = item["id"]
    top_tags_firing = "\n".join(
        f"  {k}: {v:.4f}" for k, v in item["top_labels"].items()
    ) or "  (none)"

    all_tags_str = "\n".join(
        f"  {k}: {v}" for k, v in all_category_tags.items()
    )

    # Changelog formatted
    if changelog:
        cl_str = "\n".join(
            f"  [{c['action']}] {c['key']}: "
            + (f"\"{c.get('old','')}\" → \"{c.get('new','')}\"" if c['action'] == 'improve'
               else f"\"{c.get('description', c.get('reason',''))}\"")
            for c in changelog
        )
    else:
        cl_str = "  (no previous changes recorded)"

    # Scores table
    scores_lines = []
    for s in all_scores:
        ok_str = "✓ OK" if s["is_ok"] else "✗ FAIL"
        tags_str = " ".join(f"{k}={v:.3f}" for k, v in s["top_labels"].items())
        scores_lines.append(
            f"  id={s['id']} {s['variant']:8} risk={s['risk']:.5f} blocked={s['blocked']} {ok_str}  [{tags_str}]"
        )
    scores_str = "\n".join(scores_lines)

    # Find min positive and max failing negative
    min_pos_items = sorted([s for s in all_scores if s["variant"] == "positive"],
                            key=lambda x: x["risk"])
    max_neg_items = sorted([s for s in all_scores if s["variant"] == "negative" and s["blocked"]],
                            key=lambda x: x["risk"], reverse=True)
    min_pos_id = min_pos_items[0]["id"] if min_pos_items else "?"
    max_neg_id = max_neg_items[0]["id"] if max_neg_items else "?"

    # Threshold advice
    if thr_analysis["perfect_threshold_possible"]:
        thr_advice = (
            f"✅ A PERFECT THRESHOLD EXISTS: if risk for this image drops below {thr_analysis['min_positive']:.5f}, "
            f"threshold can be set to {thr_analysis['suggested_threshold']:.5f} to achieve 100% accuracy + 0% error."
        )
    else:
        thr_advice = (
            f"❌ NO PERFECT THRESHOLD EXISTS: minimum positive ({thr_analysis['min_positive']:.5f}) "
            f"< max failing negative ({thr_analysis['max_failing_negative']:.5f}). "
            f"Counter-tags must push this image's risk below {thr_analysis['min_positive']:.5f}."
        )

    target_risk = thr_analysis["min_positive"] * 0.9  # 10% below min positive

    return CONTEXTUAL_PROMPT.format(
        category=category,
        item_id=item_id,
        current_risk=item["risk"],
        threshold=item["threshold"],
        top_tags_firing=top_tags_firing,
        all_category_tags=all_tags_str,
        changelog=cl_str,
        all_scores_table=scores_str,
        min_positive=thr_analysis["min_positive"],
        min_positive_id=min_pos_id,
        max_neg_fail=thr_analysis["max_failing_negative"],
        max_neg_fail_id=max_neg_id,
        gap=thr_analysis["gap"],
        threshold_advice=thr_advice,
        target_risk=target_risk,
    )


# ─── Grok API ────────────────────────────────────────────────────────────────

async def call_grok(client: httpx.AsyncClient, api_key: str, model: str,
                    image_url: str, prompt: str, max_tokens: int) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Kortique/piper-moderate",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
    }
    resp = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120.0)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--category",  "-c", required=True)
@click.option("--snapshot",  "-s", required=True, help="Patched snapshot JSON (with fresh Piper scores)")
@click.option("--ids",       "-i", required=True, help="Comma-separated item IDs to analyze")
@click.option("--output",    "-o", default=None)
def main(category, snapshot, ids, output):
    """Rich contextual Grok analysis for failed items with full tag history."""

    cfg        = load_config()
    api_key    = os.getenv("OPENROUTER_API_KEY")
    model      = cfg["openrouter"]["model"]
    max_tokens = cfg["openrouter"].get("max_tokens", 2000)
    tags_file  = cfg["tags_file"]
    detail_key = get_detail_key(category)

    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set[/red]")
        raise SystemExit(1)

    target_ids = {x.strip() for x in ids.split(",")}

    # Load data
    with open(snapshot, encoding="utf-8") as f:
        raw = json.load(f)
    data = raw if isinstance(raw, list) else list(raw.values())[0]

    all_tags      = load_tags(tags_file)
    category_tags = get_category_tags(all_tags, category)
    changelog     = load_changelog(category)
    all_scores    = compute_all_scores(data, category, detail_key)
    thr_analysis  = threshold_analysis(all_scores)

    # Print overview
    console.print(Panel(
        f"Category: [bold]{category}[/bold]  detail_key={detail_key}\n"
        f"Tags in category: {len(category_tags)}\n"
        f"Changelog entries: {len(changelog)}\n"
        f"Items with scores: {len(all_scores)}\n"
        f"Min positive risk: {thr_analysis['min_positive']:.5f}\n"
        f"Max failing negative: {thr_analysis['max_failing_negative']:.5f}\n"
        f"Gap: {thr_analysis['gap']:.5f}  "
        f"{'✅ perfect threshold possible' if thr_analysis['perfect_threshold_possible'] else '❌ tag fix needed'}",
        title="Analysis Context", border_style="cyan"
    ))

    # Get items to analyze
    items_map = {str(item["id"]): item for item in all_scores if str(item["id"]) in target_ids}
    if not items_map:
        console.print(f"[red]No items found for IDs: {target_ids}[/red]")
        raise SystemExit(1)

    # Also get full item data (for image URLs)
    full_items = {str(item.get("id")): item for item in data
                  if str(item.get("id")) in target_ids}

    results = []

    async def run_all():
        async with httpx.AsyncClient() as client:
            for item_id, score_item in items_map.items():
                full_item = full_items.get(item_id)
                if not full_item:
                    console.print(f"[red]No full item data for id={item_id}[/red]")
                    continue

                image_url = get_image_url(full_item)
                if not image_url:
                    console.print(f"[red]No image URL for id={item_id}[/red]")
                    continue

                console.print(f"\n[bold]Analyzing id={item_id}[/bold]  "
                              f"variant={score_item['variant']}  "
                              f"risk={score_item['risk']:.5f}  "
                              f"url={image_url[:70]}...")

                prompt = build_prompt(
                    item=score_item,
                    category=category,
                    detail_key=detail_key,
                    all_category_tags=category_tags,
                    changelog=changelog,
                    all_scores=all_scores,
                    thr_analysis=thr_analysis,
                )

                try:
                    suggestion = await call_grok(client, api_key, model, image_url, prompt, max_tokens)
                    console.print(f"  [green]✓ Grok responded[/green]")
                    console.print(f"  Analysis: {suggestion.get('analysis','')[:120]}")
                    console.print(f"  Changelog review: {suggestion.get('changelog_review','')[:120]}")
                    thr_rec = suggestion.get("threshold_recommendation", {})
                    console.print(f"  Threshold rec: {thr_rec.get('suggested_threshold')} — {thr_rec.get('reasoning','')[:100]}")
                    results.append({
                        "item_id": item_id,
                        "image_url": image_url,
                        "type": "negative",
                        "current_risk": score_item["risk"],
                        "suggestion": suggestion
                    })
                except Exception as e:
                    console.print(f"  [red]Error: {e}[/red]")
                    results.append({"item_id": item_id, "error": str(e), "type": "negative"})

    asyncio.run(run_all())

    # Save
    today    = datetime.date.today().isoformat()
    out_path = output or f"suggestions/{category}_{today}_contextual.json"
    Path(out_path).parent.mkdir(exist_ok=True)

    output_data = {
        "category": category,
        "date": today,
        "model": model,
        "context": {
            "threshold_analysis": thr_analysis,
            "changelog_entries": len(changelog),
            "total_category_items": len(all_scores),
        },
        "stats": {
            "analyzed": len(results),
            "success": sum(1 for r in results if "suggestion" in r),
            "errors": sum(1 for r in results if "error" in r),
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Convert to format compatible with update_tags.py
    # update_tags.py expects {"category": ..., "results": [{"type": ..., "suggestion": {...}}]}
    compat_path = out_path.replace("_contextual.json", "_contextual_compat.json")
    compat = {
        "category": category,
        "date": today,
        "model": model,
        "stats": output_data["stats"],
        "results": results,
    }
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump(compat, f, indent=2, ensure_ascii=False)

    console.print(Panel(
        f"[green]✅ Done[/green]  success={output_data['stats']['success']}  errors={output_data['stats']['errors']}\n"
        f"Saved: [bold]{out_path}[/bold]\n"
        f"Compatible: [bold]{compat_path}[/bold]\n\n"
        f"Next: [bold]python scripts/update_tags.py --suggestions {compat_path}[/bold]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
