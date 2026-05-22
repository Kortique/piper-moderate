"""
analyze_v6_targets.py
---------------------
Grok-4.3 analysis of:
  A) Child FN (false negatives in V5): children NOT blocked — need stronger tags
  B) Adult FP (false positives in V5): adults incorrectly blocked — need counter-tags

Output: suggestions/v6_targets_<date>.json

Usage:
    python scripts/analyze_v6_targets.py           # both A+B
    python scripts/analyze_v6_targets.py --type fn  # only child FN
    python scripts/analyze_v6_targets.py --type fp  # only adult FP
    python scripts/analyze_v6_targets.py --resume
    python scripts/analyze_v6_targets.py --limit 10
"""

import sys, os, json, asyncio, datetime, sqlite3, struct, base64, io
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx, click
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv()
console = Console()

BASE_DIR       = Path(__file__).resolve().parent.parent
DB_PATH        = BASE_DIR / "gallery.db"
TAGS_FILE      = BASE_DIR / "data" / "tags.json"
OUT_DIR        = BASE_DIR / "suggestions"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL          = "x-ai/grok-4.3"

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_CHILD_FN = """You are a computer vision engineer calibrating a SigLIP-2 age-detection classifier for a content moderation pipeline.

CONTEXT:
This image was reviewed by human moderators and flagged as depicting a subject with apparent age significantly below adult (estimated age category: young / pre-teen). The automated classifier scored it below its detection threshold — a false negative.
- Classifier underage score: {lgbm:.4f}  (threshold=0.80 — MISSED)
- SigLIP minor signal: {minor:.4f}  (threshold=0.72)

TOP UNDERAGE TAGS THAT FIRED (but too weakly to trigger block):
{top_tags}

YOUR TASK — visual feature analysis for classifier calibration:
1. What age-indicating visual features are visible? (facial proportions, body size, developmental stage, style, scenario context)
2. Which existing underage tags partially match but need stronger/more precise descriptions?
3. What new tags would better capture the distinctive visual patterns present?

Return ONLY valid JSON (no markdown, no explanation):
{{
  "what_i_see": "1-2 sentence description of age-indicating visual features",
  "strengthen_existing": [
    {{"key": "existing_tag_key", "new_description": "improved description that better captures this pattern"}}
  ],
  "add_new_tags": [
    {{"key": "underage_descriptive_name", "description": "precise visual description of the age-indicating pattern"}}
  ]
}}"""

PROMPT_ADULT_FP = """You are an expert in computer vision tag calibration for a content moderation system (SigLIP-2 visual classifier).

CONTEXT:
This image was INCORRECTLY blocked as underage content. It is a confirmed adult image.
- LGBM underage score: {lgbm:.4f}  (threshold=0.80)
- SigLIP minor score: {minor:.4f}  (threshold=0.72)
- Blocking reason: {block_reason}

TAGS THAT FIRED MOST:
{top_tags}

TASK:
Look at the image carefully.
1. Confirm: is this genuinely an adult-only image, or does it contain minors?
2. If ADULT: identify which tags incorrectly fired and what adult visual features are present.
3. Suggest improvements to reduce false triggering:
   - Make existing tags more restrictive (require literal child anatomy)
   - Add new "adult__descriptive_name" counter-tags for adult visual features visible here

Return ONLY valid JSON (no markdown):
{{
  "contains_minor": false,
  "visual_triggers": "why this adult image falsely triggered",
  "improve_existing": [
    {{"key": "underage_tag_key", "new_description": "more restrictive description"}}
  ],
  "add_counter_tags": [
    {{"key": "adult__descriptive_name", "description": "visual prompt for adult counter-tag"}}
  ]
}}"""

# ── DB helpers ────────────────────────────────────────────────────────────────

def open_db():
    data = bytearray(DB_PATH.read_bytes())
    actual = len(data) // 4096
    struct.pack_into('>I', data, 28, actual)
    tmp = Path('/tmp/_gal_v6.db')
    tmp.write_bytes(data)
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    return conn


def get_siglip(conn, gid: str):
    if gid.startswith('ls_'):
        tid = gid[3:]
        try:
            rows = conn.execute("SELECT siglip2_details FROM ls_images WHERE task_id=?", (tid,)).fetchall()
            if rows and rows[0][0]:
                return json.loads(rows[0][0])
        except: pass
    else:
        for offset in [0, 100, 200, 300, 500, 600]:
            try:
                rows = conn.execute(
                    "SELECT piper_result FROM grafana_pool LIMIT 100 OFFSET ?", (offset,)
                ).fetchall()
                for row in rows:
                    if row[0]:
                        pr = json.loads(row[0])
                        det = pr.get('siglip2_details') or {}
                        if det:
                            return det
            except: pass
    return None


# ── Build item list ───────────────────────────────────────────────────────────

def build_items(mode: str, limit: int):
    tags = json.loads(TAGS_FILE.read_text())
    underage_tags = {k[9:]: v for k, v in tags.items() if k.startswith('underage_')}

    # Load V5 eval data
    data_dir = BASE_DIR / "data"
    v5_map = {}
    for fname in ['eval_v5_db_all_std.json', 'eval_v5_g1g2_results.json', 'eval_v5_g3456_results.json']:
        p = data_dir / fname
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            eid = r['id']
            if eid.startswith('qwen_'): gid = 'ls_' + eid[5:]
            elif eid.startswith('dg_'): gid = eid[3:]
            else: gid = eid
            v5_map[gid] = r

    # URL map from DB
    conn = open_db()
    url_map = {}
    for offset in range(0, 2500, 200):
        try:
            rows = conn.execute(f"SELECT task_id, media FROM ls_images LIMIT 200 OFFSET {offset}").fetchall()
            if not rows: break
            for r in rows: url_map[f"ls_{r['task_id']}"] = r['media']
        except: pass
    for offset in [0, 100, 200, 300, 500, 600]:
        try:
            rows = conn.execute(f"SELECT id, thumb_url FROM grafana_pool LIMIT 100 OFFSET {offset}").fetchall()
            for r in rows: url_map[str(r['id'])] = r['thumb_url']
        except: pass
    conn.close()

    items = []
    conn2 = open_db()

    for gid, entry in v5_map.items():
        lbl = entry.get('human_label')
        lgbm  = entry.get('lgbm', 0)
        minor = entry.get('minor', 0)
        blocked = (lgbm >= 0.80) or (minor >= 0.72)
        url = url_map.get(gid)
        if not url: continue

        if mode in ('fn', 'both') and lbl == 'child' and not blocked:
            det = get_siglip(conn2, gid)
            und = (det or {}).get('underage', {}) if det else {}
            labels = und.get('labels', {})
            top_tags = sorted(labels.get('underage', {}).items(), key=lambda x: -x[1])[:8]
            top_tags_str = '\n'.join(f"  {k}: {v:.4f}" for k, v in top_tags) or '  (none)'
            items.append({
                'id': gid, 'url': url, 'type': 'fn',
                'lgbm': lgbm, 'minor': minor,
                'top_tags_str': top_tags_str,
            })

        elif mode in ('fp', 'both') and lbl == 'adult' and blocked:
            block_reason = []
            if lgbm >= 0.80: block_reason.append(f'LGBM={lgbm:.3f}')
            if minor >= 0.72: block_reason.append(f'minor={minor:.3f}')
            det = get_siglip(conn2, gid)
            und = (det or {}).get('underage', {}) if det else {}
            labels = und.get('labels', {})
            top_und = sorted(labels.get('underage', {}).items(), key=lambda x: -x[1])[:6]
            top_adt = sorted(labels.get('adult',    {}).items(), key=lambda x: -x[1])[:4]
            all_top = [(k,v) for k,v in top_und] + [(f'(adult){k}',v) for k,v in top_adt]
            top_tags_str = '\n'.join(f"  {k}: {v:.4f}" for k, v in all_top) or '  (none)'
            items.append({
                'id': gid, 'url': url, 'type': 'fp',
                'lgbm': lgbm, 'minor': minor,
                'block_reason': ' + '.join(block_reason),
                'top_tags_str': top_tags_str,
            })

    conn2.close()
    if limit: items = items[:limit]
    return items


# ── Image helpers ─────────────────────────────────────────────────────────────

async def to_data_url(client: httpx.AsyncClient, url: str) -> str | None:
    """Download an image and return a JPEG base64 data URL. Returns None on failure."""
    if not HAS_PIL:
        return None
    try:
        resp = await client.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        img = PILImage.open(io.BytesIO(resp.content))
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


# ── Grok call ─────────────────────────────────────────────────────────────────

async def _grok_request(client, api_key, prompt, image_ref, semaphore):
    """Send one request to Grok. image_ref is a URL string or data URL."""
    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": image_ref}},
        ]}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Kortique/piper-moderate",
        "X-Title": "piper-moderate",
    }
    async with semaphore:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()


def _parse_grok_response(item, raw_json):
    """Parse a Grok API response dict into a result dict."""
    raw_msg = raw_json["choices"][0]["message"]
    content = raw_msg.get("content")
    if content is None:
        refusal = raw_msg.get("refusal") or "null content"
        return {"id": item['id'], "url": item['url'], "type": item['type'],
                "error": f"null_content: {refusal}"}
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1]
        if content.startswith("json"): content = content[4:]
    try:
        return {"id": item['id'], "url": item['url'], "type": item['type'],
                "suggestion": json.loads(content.strip())}
    except json.JSONDecodeError as e:
        return {"id": item['id'], "url": item['url'], "type": item['type'],
                "error": f"json_parse: {e}", "raw": content}


async def call_grok(client, api_key, item, semaphore):
    if item['type'] == 'fn':
        prompt = PROMPT_CHILD_FN.format(
            lgbm=item['lgbm'], minor=item['minor'],
            top_tags=item['top_tags_str'],
        )
    else:
        prompt = PROMPT_ADULT_FP.format(
            lgbm=item['lgbm'], minor=item['minor'],
            block_reason=item.get('block_reason','?'),
            top_tags=item['top_tags_str'],
        )

    url = item['url']

    try:
        raw_json = await _grok_request(client, api_key, prompt, url, semaphore)
        result = _parse_grok_response(item, raw_json)

        # If Grok refused (json_parse with refusal text) — don't retry, just return
        if 'error' in result and result['error'].startswith('json_parse'):
            return result

        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code != 412:
            return {"id": item['id'], "url": url, "type": item['type'],
                    "error": f"http_{e.response.status_code}: {e.response.text[:200]}"}
        # 412 = unsupported content type (e.g. WebP) — download + convert and retry
        data_url = await to_data_url(client, url)
        if data_url is None:
            return {"id": item['id'], "url": url, "type": item['type'],
                    "error": "http_412_and_convert_failed"}
        try:
            raw_json = await _grok_request(client, api_key, prompt, data_url, semaphore)
            return _parse_grok_response(item, raw_json)
        except Exception as e2:
            return {"id": item['id'], "url": url, "type": item['type'], "error": f"retry: {e2}"}

    except Exception as e:
        return {"id": item['id'], "url": url, "type": item['type'], "error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option('--type', 'mode', default='both', type=click.Choice(['fn','fp','both']),
              help='fn=child FN only, fp=adult FP only, both=all')
@click.option('--limit', default=0, type=int, help='Max items to process (0=all)')
@click.option('--concurrency', default=4, type=int)
@click.option('--resume', is_flag=True)
def main(mode, limit, concurrency, resume):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set[/red]")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    today = datetime.date.today().isoformat()
    out_path = OUT_DIR / f"v6_targets_{today}.json"

    resume_ids = set()
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        resume_ids = {r['id'] for r in existing if 'suggestion' in r}
        console.print(f"[yellow]Resuming: {len(resume_ids)} already done[/yellow]")

    items = build_items(mode, limit)
    console.print(f"[cyan]Items to analyze: {len(items)} (fn={sum(1 for x in items if x['type']=='fn')}, fp={sum(1 for x in items if x['type']=='fp')})[/cyan]")

    todo = [it for it in items if it['id'] not in resume_ids]
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def run():
        async with httpx.AsyncClient() as client:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), MofNCompleteColumn(), console=console) as prog:
                tid = prog.add_task("Analyzing...", total=len(todo))
                coros = [call_grok(client, api_key, item, semaphore) for item in todo]
                for coro in asyncio.as_completed(coros):
                    r = await coro
                    results.append(r)
                    icon = "✓" if "suggestion" in r else f"✗({r.get('error','?')[:40]})"
                    console.log(f"  [{r['type']}] {icon}  {r['id']}")
                    prog.advance(tid)

    asyncio.run(run())

    # Merge with existing if resuming
    # New results take priority over existing (allows retrying failures)
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        new_by_id = {r['id']: r for r in results}
        merged = []
        for r in existing:
            # Replace old error with new result (even if new is also error — we re-ran it)
            # But keep old success if new run didn't touch this ID
            if r['id'] in new_by_id:
                merged.append(new_by_id.pop(r['id']))
            else:
                merged.append(r)
        # Append any brand-new IDs that weren't in existing
        merged.extend(new_by_id.values())
        results = merged

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    ok = sum(1 for r in results if 'suggestion' in r)
    console.print(f"\n[green]Done: {ok}/{len(results)} successful → {out_path}[/green]")

if __name__ == '__main__':
    main()
