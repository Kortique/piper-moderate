"""
analyze_fp_69.py
----------------
Grok analysis of 69 adult FP images falsely blocked by Piper underage model.

For each image:
  - Loads prompt, siglip2 top tags, fd/q3 age data from gallery.db
  - Sends image URL + context to Grok (grok-2-vision)
  - Asks for tag fixes: improve existing, add no_* counter-tags, remove over-broad tags

Output: suggestions/underage_fp69_<date>.json  (compatible with update_tags.py)

Usage:
    python scripts/analyze_fp_69.py
    python scripts/analyze_fp_69.py --limit 10   # test run
    python scripts/analyze_fp_69.py --resume      # skip already analyzed
"""

import sys, os, json, asyncio, datetime, sqlite3
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx, click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv()
console = Console()

BASE_DIR       = Path(__file__).resolve().parent.parent
DB_PATH        = BASE_DIR / "gallery.db"
FP_IDS_FILE    = BASE_DIR / "fp_69_ids.txt"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL          = "x-ai/grok-4.3"   # vision-capable Grok via OpenRouter

EVAL_FILES = [
    BASE_DIR / "data" / f"eval_g{i}_results.json" for i in range(1, 7)
]
LGBM_THRESHOLD  = 0.80
MINOR_THRESHOLD = 0.72


PROMPT_FP = """You are an expert in computer vision tag calibration for a content moderation system (SigLIP-2 visual classifier).

CONTEXT:
This image was INCORRECTLY blocked as underage content. It is a confirmed adult image.
- Face Detect age: {fd_age}
- Qwen3 faces: {q3_faces}
- Image prompt: {prompt}
- LGBM underage score: {lgbm:.4f}  (threshold=0.80 — this image is above threshold)
- SigLIP minor score: {minor:.4f} / adult score: {adult:.4f}

TAGS THAT FIRED MOST (key: score):
{top_tags}

ALL CURRENT UNDERAGE TAGS (key → description):
{all_underage_tags}

TASK:
1. Look at the image. Identify what adult features/anatomy are visible.
2. Identify which specific existing tags are too vague and incorrectly fire on this adult content.
3. Suggest improvements to existing tag descriptions to be MORE restrictive (require literal child proportions, round baby head, etc.).
4. Suggest new "no_underage_*" counter-tags that explicitly describe adult visual features that should NOT trigger the underage model.

Return ONLY valid JSON (no markdown):
{{
  "visual_triggers": "1-2 sentence description of why this adult image falsely triggered",
  "improve_existing": [
    {{"key": "underage_tag_key", "new_description": "more restrictive description requiring literal child anatomy"}}
  ],
  "add_new_tags": [
    {{"key": "no_underage_descriptive_name", "description": "visual prompt for adult counter-tag"}}
  ],
  "remove_tags": []
}}"""


def load_fp_data():
    """Load 69 FP images from DB with all relevant context."""
    fp_ids = [l.strip() for l in FP_IDS_FILE.read_text().splitlines() if l.strip()]

    # Load eval JSON for accurate LGBM scores
    eval_data = {}
    for f in EVAL_FILES:
        if not f.exists(): continue
        for item in json.loads(f.read_text()):
            eid = item["id"]
            gid = eid[3:] if eid.startswith("dg_") else ("ls_" + eid[5:] if eid.startswith("qwen_") else eid)
            lgbm = item.get("lgbm", 0.0)
            minor = item.get("minor", 0.0)
            blocked = (lgbm >= LGBM_THRESHOLD) or (minor >= MINOR_THRESHOLD)
            eval_data[gid] = {"lgbm": lgbm, "minor": minor, "blocked": blocked}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Load all underage tags
    tags_all = json.load(open(BASE_DIR / "data" / "tags.json"))
    underage_tags = {k: v for k, v in tags_all.items()
                     if k.startswith("underage_") or k.startswith("no_underage_")}

    items = []
    for gid in fp_ids:
        row = conn.execute("""
            SELECT id, thumb_url, prompt,
                   json_extract(piper_result, '$.siglip2_details.underage.minor') as minor,
                   json_extract(piper_result, '$.siglip2_details.underage.adult') as adult,
                   json_extract(piper_result, '$.siglip2_details.underage.labels') as labels_json,
                   json_extract(piper_result, '$.face_detect_result.ageFrom') as fd_from,
                   json_extract(piper_result, '$.face_detect_result.ageTo') as fd_to,
                   qwen3_result
            FROM grafana_pool WHERE id=?
        """, (gid,)).fetchone()
        if not row:
            continue

        # Get accurate LGBM score from eval JSON if available
        ev = eval_data.get(gid, {})
        lgbm = ev.get("lgbm") or 0.0

        labels = json.loads(row["labels_json"]) if row["labels_json"] else {}
        underage_labels = labels.get("underage", {})
        top_tags = sorted(underage_labels.items(), key=lambda x: -float(x[1]))[:12]

        q3 = json.loads(row["qwen3_result"]) if row["qwen3_result"] else {}
        q3_faces = q3.get("faces") or []

        items.append({
            "id": gid,
            "image_url": row["thumb_url"],
            "prompt": (row["prompt"] or "")[:300],
            "lgbm": lgbm,
            "minor": float(row["minor"] or 0),
            "adult": float(row["adult"] or 0),
            "fd_age": f"{row['fd_from']}-{row['fd_to']}" if row["fd_from"] else "n/a",
            "q3_faces": q3_faces,
            "top_tags": top_tags,
            "underage_tags": underage_tags,
        })

    conn.close()
    return items


async def call_grok(client, api_key, item, semaphore):
    top_tags_str = "\n".join(f"  {k}: {v:.4f}" for k, v in item["top_tags"])
    all_tags_str = json.dumps(item["underage_tags"], indent=2, ensure_ascii=False)

    prompt = PROMPT_FP.format(
        fd_age=item["fd_age"],
        q3_faces=item["q3_faces"],
        prompt=item["prompt"],
        lgbm=item["lgbm"],
        minor=item["minor"],
        adult=item["adult"],
        top_tags=top_tags_str,
        all_underage_tags=all_tags_str,
    )

    payload = {
        "model": MODEL,
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": item["image_url"]}},
            ],
        }],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Kortique/piper-moderate",
        "X-Title": "piper-moderate",
    }

    async with semaphore:
        try:
            resp = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120.0)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
            return {
                "item_id": "dg_" + item["id"],
                "image_url": item["image_url"],
                "type": "negative",
                "suggestion": json.loads(content.strip()),
            }
        except json.JSONDecodeError as e:
            return {"item_id": "dg_" + item["id"], "image_url": item["image_url"],
                    "type": "negative", "error": f"json_parse: {e}"}
        except httpx.HTTPStatusError as e:
            return {"item_id": "dg_" + item["id"], "image_url": item["image_url"],
                    "type": "negative", "error": f"http_{e.response.status_code}: {e.response.text[:200]}"}
        except Exception as e:
            return {"item_id": "dg_" + item["id"], "image_url": item["image_url"],
                    "type": "negative", "error": str(e)}


async def run_all(items, api_key, concurrency, resume_ids):
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient() as client:
        todo = [it for it in items if it["id"] not in resume_ids]
        console.print(f"[cyan]Processing {len(todo)} images (skipping {len(resume_ids)} already done)[/cyan]")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), console=console,
        ) as prog:
            tid = prog.add_task("Analyzing with Grok...", total=len(todo))
            coros = [call_grok(client, api_key, item, semaphore) for item in todo]
            for coro in asyncio.as_completed(coros):
                r = await coro
                results.append(r)
                icon = "✓" if "suggestion" in r else f"✗({r.get('error','?')[:30]})"
                console.log(f"  {icon}  {r['item_id']}")
                prog.advance(tid)

    return results


@click.command()
@click.option("--limit", "-l", default=0, type=int, help="Analyze only first N images (0=all)")
@click.option("--resume", is_flag=True, help="Skip images already in output file")
@click.option("--concurrency", default=3, type=int, help="Parallel Grok requests")
@click.option("--output", "-o", default=None, help="Output file path (default: suggestions/underage_fp69_<date>.json)")
def main(limit, resume, concurrency, output):
    """Analyze 69 adult FP images with Grok to generate tag improvement suggestions."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set[/red]")
        raise SystemExit(1)

    items = load_fp_data()
    console.print(f"[green]Loaded {len(items)} FP images from DB[/green]")

    if limit:
        items = items[:limit]

    today = datetime.date.today().isoformat()
    out_path = Path(output) if output else Path("suggestions") / f"underage_fp69_{today}.json"
    out_path.parent.mkdir(exist_ok=True)

    # Resume: load existing results
    resume_ids = set()
    existing_results = []
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        existing_results = existing.get("results", [])
        resume_ids = {r["item_id"].replace("dg_", "") for r in existing_results if "suggestion" in r}
        console.print(f"[cyan]Resuming: {len(resume_ids)} already done[/cyan]")

    new_results = asyncio.run(run_all(items, api_key, concurrency, resume_ids))
    all_results = existing_results + new_results

    success = sum(1 for r in all_results if "suggestion" in r)
    errors  = sum(1 for r in all_results if "error" in r)

    output_data = {
        "category": "underage",
        "date": today,
        "model": MODEL,
        "description": "FP analysis: 69 adult images falsely blocked by Piper V4 underage model",
        "stats": {
            "total": len(all_results),
            "success": success,
            "errors": errors,
        },
        "results": all_results,
    }

    out_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    console.print(f"\n[bold green]✅ Done! success={success} errors={errors}[/bold green]")
    console.print(f"Saved: [bold]{out_path}[/bold]")
    console.print(f"\nNext: [bold]python scripts/update_tags.py --suggestions {out_path} --auto[/bold]")


if __name__ == "__main__":
    main()
