#!/usr/bin/env python3
"""Validate V8pas80 FP candidates through Grok-4-fast.

For each candidate, send image + ask: "is this clearly an adult (>=18) or any chance of being a minor?"
Keep only those Grok confirms is_minor='no' AND age_min >= 18.

Resumable. Saves results to data/v8pas80_fps_grok.json after each item.
"""
import json, os, sys, time, base64, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')
KEY = os.getenv('OPENROUTER_API_KEY')
URL = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'x-ai/grok-4.3'

CANDS = BASE / 'data' / 'v8pas80_top100_fps.json'
OUT = BASE / 'data' / 'v8pas80_fps_grok.json'

PROMPT = """You are a content moderation classifier. Analyze this image and assess the age of the subject(s).

Return STRICT JSON only:
{
  "age_min": <int — youngest age you see in image>,
  "age_max": <int — oldest age you see in image>,
  "is_clearly_adult": "yes | maybe | no",
  "reasoning": "1 sentence — what visual cues support this assessment",
  "scene_type": "1-3 words — cosplay, lingerie, nude, etc."
}"""


def fetch_image(url):
    """Fetch image and return data URI. Convert non-JPEG (e.g. webp) to JPEG
       because xAI provider rejects some content-types."""
    try:
        with httpx.Client(timeout=15) as c:
            r = c.get(url)
            r.raise_for_status()
            content = r.content
        # Always normalize to JPEG (Grok content-type strict)
        try:
            from PIL import Image as PILImage
            from io import BytesIO
            img = PILImage.open(BytesIO(content)).convert('RGB')
            out = BytesIO()
            img.save(out, 'JPEG', quality=85)
            jpeg_bytes = out.getvalue()
            return 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes).decode()
        except Exception:
            # fallback: pass through
            ct = r.headers.get('content-type', 'image/jpeg')
            return f'data:{ct};base64,' + base64.b64encode(content).decode()
    except Exception:
        return None


def grok_call(img_data, retries=2):
    last_err = None
    for _ in range(retries):
        try:
            with httpx.Client(timeout=90) as c:
                payload = {
                    'model': MODEL,
                    'messages': [{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': img_data}},
                        {'type': 'text', 'text': PROMPT},
                    ]}],
                    'max_tokens': 300,
                }
                r = c.post(URL, headers={'Authorization': f'Bearer {KEY}'},
                           content=json.dumps(payload).encode())
                if r.status_code != 200:
                    last_err = f'HTTP {r.status_code}: {r.text[:150]}'
                    time.sleep(1)
                    continue
                content = r.json()['choices'][0]['message']['content']
                # Try to extract JSON from possibly markdown-wrapped text
                import re as _re
                m = _re.search(r'\{[^{}]*"is_clearly_adult"[^{}]*\}', content, _re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except: pass
                try: return json.loads(content)
                except:
                    return {'raw': content[:300], 'parse_err': True}
        except Exception as e:
            last_err = str(e)[:120]
            time.sleep(1)
    return {'error': last_err or 'all retries failed'}


def process_one(item):
    img = fetch_image(item['media'])
    if not img:
        return {**item, 'grok': {'error': 'image_fetch_failed'}}
    ans = grok_call(img)
    return {**item, 'grok': ans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=20)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()

    cands = json.loads(CANDS.read_text())
    existing = {}
    if OUT.exists():
        try:
            for r in json.loads(OUT.read_text()):
                if r.get('grok') and 'error' not in r['grok']:
                    existing[r['id']] = r
        except Exception: pass
    remaining = [c for c in cands if c["id"] not in existing]
    if not remaining:
        print(f"Nothing to do. Done: {len(existing)}")
        return
    chunk = remaining[:args.chunk]
    print(f"cands={len(cands)} done={len(existing)} remaining={len(remaining)} chunk={len(chunk)}", flush=True)

    out = dict(existing)
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, c): c for c in chunk}
        for fut in as_completed(futs):
            rec = fut.result()
            out[rec["id"]] = rec
            completed += 1
            g = rec.get('grok', {})
            if 'error' in g:
                err = str(g.get('error', ''))[:60]
                tag = 'ERR ' + err
            else:
                tag = g.get('is_clearly_adult', '?')
            rid = rec.get('id', '?')
            print('  ' + str(completed) + '/' + str(len(chunk)) + ' t=' + ('%.0f' % (time.time()-t0)) + 's ' + str(rid) + ' -> ' + str(tag), flush=True)
            OUT.write_text(json.dumps(list(out.values()), ensure_ascii=False, indent=2))

    confirmed = sum(1 for r in out.values()
                    if r.get('grok', {}).get('is_clearly_adult') == 'yes')
    print('\nTotal confirmed-adult so far: ' + str(confirmed) + '/' + str(len(out)), flush=True)


if __name__ == "__main__":
    main()
