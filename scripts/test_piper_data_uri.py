#!/usr/bin/env python3
"""
test_piper_data_uri.py — does Piper d2911d10bb accept a base64 data URI in the
`image` input field instead of an http(s) URL?

If yes — we can score local files (B:\\Pop\\Pop5\\...) without uploading anywhere.
If no — we need an upload path (LS API or S3).

Approach: take ONE known-good image URL, download bytes, b64-encode, send as
`data:image/...;base64,...` and check Piper's launch response. If launch succeeds
AND polling returns siglip2_details, the path works.
"""
import base64, json, os, sys, time
from pathlib import Path
import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')

TOKEN = os.getenv('PIPER_TOKEN', '')
PIPER = 'https://piper-next.artworks.ai/api'
PROJECT = 'd2911d10bb'

# A known LS image (small enough to fit comfortably in a single request)
SAMPLE_URL = 'https://fsn1.your-objectstorage.com/artworks-assets/review-019e2efe-5bc1-7bed-b4c4-f9cb83ab0ee5.webp'

if not TOKEN:
    print('ERR: PIPER_TOKEN not set'); sys.exit(1)

hdr = {'User-Token': TOKEN, 'Content-Type': 'application/json',
       'Accept': 'application/json'}

print(f'=== Test 1: download {SAMPLE_URL} ===')
r = httpx.get(SAMPLE_URL, timeout=30)
r.raise_for_status()
img_bytes = r.content
ctype = r.headers.get('content-type', 'image/webp').split(';')[0].strip()
print(f'  bytes: {len(img_bytes)}  content-type: {ctype}')

data_uri = f'data:{ctype};base64,' + base64.b64encode(img_bytes).decode('ascii')
print(f'  data URI length: {len(data_uri)} chars  (first 80: {data_uri[:80]}...)')

print(f'\n=== Test 2: launch d2911d10bb with data URI (full providers) ===')
payload = {'inputs': {'image': data_uri,
                      'providers_e0': ['siglip2', 'qwen3', 'face_detect']}}
try:
    r = httpx.post(f'{PIPER}/projects/{PROJECT}/launch', headers=hdr,
                   json=payload, timeout=60)
except Exception as e:
    print(f'  POST error: {type(e).__name__}: {e}'); sys.exit(2)

print(f'  HTTP {r.status_code}')
if r.status_code != 200:
    print(f'  body: {r.text[:500]}')
    print(f'\n  -> data URI rejected. Need to fall back to upload path.')
    sys.exit(2)

run_id = r.json().get('_id')
print(f'  run_id: {run_id}')

print(f'\n=== Test 3: poll until siglip2_details (up to ~3 min) ===')
for i in range(60):
    time.sleep(3)
    rs = httpx.get(f'{PIPER}/launches/{run_id}/state', headers=hdr, timeout=20)
    if rs.status_code != 200:
        print(f'  poll {i}: HTTP {rs.status_code}'); continue
    st = rs.json()
    outs = st.get('outputs') or {}
    errs = st.get('errors') or []
    if errs:
        print(f'  ERRORS: {errs}'); sys.exit(2)
    if 'siglip2_details' in outs:
        print(f'  ✓ siglip2_details returned after ~{(i+1)*3}s')
        u = ((outs.get('siglip2_details') or {}).get('underage') or {})
        labels = (u.get('labels') or {})
        print(f'    siglip2 underage tags: {sorted(labels.get("underage", {}).items(), key=lambda x: -x[1])[:5]}')
        # Check qwen3 / face_detect outputs
        print(f'    qwen3 output present:  {bool(outs.get("qwen3_age"))}')
        print(f'    face_detect present:   {bool(outs.get("face_detect_result"))}')
        print(f'\n=== VERDICT: data URI WORKS for Piper d2911d10bb ===')
        sys.exit(0)
else:
    print(f'  TIMEOUT after 3 min — no siglip2_details'); sys.exit(2)
2)
