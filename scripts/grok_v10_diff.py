#!/usr/bin/env python3
"""Grok analysis of V10 diff items — pattern discovery."""
import json, os, sys, base64, time
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')

KEY = os.getenv('OPENROUTER_API_KEY')
URL = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'x-ai/grok-4-fast'  # fast tier — cheaper

OUT = BASE / 'data' / 'v10_diff_grok.json'

PROMPT_FP = """You are evaluating a content moderation classifier. This image was BLOCKED by V10 model as suspected underage content, but human reviewers confirmed it is ADULT. We want to understand what visual features caused the false alarm.

Top underage_ tags that fired on this image (with scores):
{top_und}

Top no_underage_ counter-tags that fired (which SHOULD have suppressed but didn't):
{top_nu}

Top adult__ tags:
{top_ad}

Return strict JSON only:
{{
  "subject_age_estimate_years": [min, max],
  "is_actually_minor": "no | maybe | yes",
  "scene_type": "one phrase: e.g. classroom_roleplay, college_uniform, anime_adult, petite_adult_nude, ...",
  "why_underage_tags_misfired": "1 sentence",
  "which_no_underage_tag_should_be_strengthened": "tag_name_or_none",
  "which_underage_tag_should_be_narrowed": "tag_name_or_none"
}}"""

PROMPT_FN = """You are evaluating a content moderation classifier. This image is labeled by humans as TEEN (15-17), but V10 model failed to block it (passed it as adult). V7 correctly blocked it. We want to understand what V10 missed.

Top underage_ tags that fired (weakly):
{top_und}

Top no_underage_ counter-tags:
{top_nu}

Return strict JSON only:
{{
  "subject_age_estimate_years": [min, max],
  "is_actually_minor": "no | maybe | yes",
  "visual_evidence_of_minor": "1 sentence — what makes this look like a teen?",
  "which_underage_tag_should_be_strengthened": "tag_name_or_none",
  "is_no_underage_tag_falsely_suppressing": "tag_name_or_none"
}}"""


def encode_image(url, timeout=15):
    """Download image and return data URI."""
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(url)
            r.raise_for_status()
            ct = r.headers.get('content-type', 'image/jpeg')
            b64 = base64.b64encode(r.content).decode()
            return f'data:{ct};base64,{b64}'
    except Exception as e:
        return None


def call_grok(prompt, img_data, retries=2):
    for _ in range(retries):
        try:
            with httpx.Client(timeout=60) as c:
                payload = {
                    'model': MODEL,
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'image_url', 'image_url': {'url': img_data}},
                            {'type': 'text', 'text': prompt},
                        ]
                    }],
                    'max_tokens': 400,
                    'response_format': {'type': 'json_object'},
                }
                r = c.post(URL, headers={'Authorization': f'Bearer {KEY}'},
                           content=json.dumps(payload).encode())
                if r.status_code != 200:
                    time.sleep(2)
                    continue
                msg = r.json()['choices'][0]['message']['content']
                return json.loads(msg)
        except Exception as e:
            time.sleep(2)
    return {'error': 'all retries failed'}


def fmt_tags(tags_list, top=5):
    return '\n'.join(f'  - {k}: {v:.3f}' for k, v in tags_list[:top])


def analyze(items, kind, n_max=10):
    """Process top-N items by V10 score (descending for FPs, ascending for FNs)."""
    if kind == 'fp':
        items = sorted(items, key=lambda x: -x['v10_score'])[:n_max]
    else:
        items = sorted(items, key=lambda x: -x['v7_score'])[:n_max]
    results = []
    for i, it in enumerate(items, 1):
        print(f'  [{i}/{len(items)}] {kind} {it["id"]} v7={it["v7_score"]:.2f} v10={it["v10_score"]:.2f}', flush=True)
        img = encode_image(it['media'])
        if not img:
            results.append({'id': it['id'], 'error': 'image_fetch_failed'})
            continue
        prompt = PROMPT_FP if kind == 'fp' else PROMPT_FN
        prompt = prompt.format(
            top_und=fmt_tags(it['top_underage_new']),
            top_nu=fmt_tags(it['top_no_underage']),
            top_ad=fmt_tags(it.get('top_adult_new', [])) if 'top_adult_new' in it else '',
        )
        ans = call_grok(prompt, img)
        results.append({
            'id': it['id'], 'media': it['media'], 'label': it['label'],
            'v7_score': it['v7_score'], 'v10_score': it['v10_score'],
            'grok': ans,
        })
    return results


def main():
    diff = json.loads((BASE / 'data' / 'v10_diff_analysis.json').read_text())
    print(f'Loaded: {len(diff["v10_fps_adults"])} FPs, {len(diff["v10_fns_teens"])} FNs', flush=True)

    n_fp = int(os.environ.get('N_FP', 10))
    n_fn = int(os.environ.get('N_FN', 10))

    # Resume support
    cache = {'fps': {}, 'fns': {}}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text())
            for r in old.get('fps', []):
                if 'grok' in r and 'error' not in r.get('grok', {}):
                    cache['fps'][r['id']] = r
            for r in old.get('fns', []):
                if 'grok' in r and 'error' not in r.get('grok', {}):
                    cache['fns'][r['id']] = r
            print(f'  Resumed: {len(cache["fps"])} fps + {len(cache["fns"])} fns from prior run', flush=True)
        except Exception:
            pass

    def analyze_resumable(items, kind, n_max):
        if kind == 'fp':
            items = sorted(items, key=lambda x: -x['v10_score'])[:n_max]
        else:
            items = sorted(items, key=lambda x: -x['v7_score'])[:n_max]
        results = []
        for i, it in enumerate(items, 1):
            if it['id'] in cache[kind+'s']:
                print(f'  [{i}/{len(items)}] {kind} {it["id"]} cached', flush=True)
                results.append(cache[kind+'s'][it['id']])
                continue
            print(f'  [{i}/{len(items)}] {kind} {it["id"]} v7={it["v7_score"]:.2f} v10={it["v10_score"]:.2f}', flush=True)
            img = encode_image(it['media'])
            if not img:
                results.append({'id': it['id'], 'error': 'image_fetch_failed'})
                continue
            prompt_t = PROMPT_FP if kind == 'fp' else PROMPT_FN
            prompt = prompt_t.format(
                top_und=fmt_tags(it['top_underage_new']),
                top_nu=fmt_tags(it['top_no_underage']),
                top_ad=fmt_tags(it.get('top_adult_new', [])),
            )
            ans = call_grok(prompt, img)
            rec = {'id': it['id'], 'media': it['media'], 'label': it['label'],
                   'v7_score': it['v7_score'], 'v10_score': it['v10_score'],
                   'grok': ans}
            results.append(rec)
            cache[kind+'s'][it['id']] = rec
            # save after every item
            OUT.write_text(json.dumps(
                {'fps': list(cache['fps'].values()),
                 'fns': list(cache['fns'].values())},
                ensure_ascii=False, indent=2))
        return results

    print(f'\n=== Analyzing {n_fp} V10 FPs ===', flush=True)
    fp_results = analyze_resumable(diff['v10_fps_adults'], 'fp', n_fp)
    print(f'\n=== Analyzing {n_fn} V10 missed teens ===', flush=True)
    fn_results = analyze_resumable(diff['v10_fns_teens'], 'fn', n_fn)

    OUT.write_text(json.dumps({'fps': fp_results, 'fns': fn_results}, ensure_ascii=False, indent=2))
    print(f'\nSaved: {OUT}', flush=True)


if __name__ == '__main__':
    main()
