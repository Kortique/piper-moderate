#!/usr/bin/env python3
"""
moderate_borderlands.py — run Piper d2911d10bb on local Borderlands files using
base64 data URIs (no upload needed; Piper accepts data: URLs).

Pipeline output stored per-row in borderlands_pool:
  - piper_result   : JSON {siglip2_labels, siglip2_passed, siglip2_details,
                            face_detect_result={ageFrom,ageTo}}
  - qwen3_result   : JSON {faces:[{ageFrom,ageTo},...]}
  - label          : derived from qwen3 min ageFrom (child ≤14, teen 15-17, adult 18+)
  - label_source   : 'ai'
  - variant        : default_variant(label) — 'positive' for child/teen, 'negative' for adult
  - processed_at   : ISO timestamp

CRITICAL: pipeline d2911d10bb expects the `providers` input key (verified in
moderate_disagree.py — the working reference). Without it the pipeline falls
back to its built-in defaults (siglip2 + hive), which leaves qwen3 and
face_detect inactive. Pass all three explicitly.

Output keys from Piper (cf. moderate_disagree.py):
  - siglip2_labels, siglip2_details   ← siglip2 provider
  - features                          ← face_detect provider (single Index=0 face)
  - qwen3_faces, qwen3_description,
    qwen3_underage, qwen3_details     ← qwen3 provider
We persist ONLY the age fields the gallery actually consumes.

Usage:
    python scripts/moderate_borderlands.py --workers 6
    python scripts/moderate_borderlands.py --workers 6 --limit 5   # smoke test
    python scripts/moderate_borderlands.py --workers 6 --redo      # rescore everything
"""
import argparse, base64, json, mimetypes, os, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / '.env')

DB_PATH        = BASE / 'gallery.db'
PIPER_BASE     = 'https://piper-next.artworks.ai/api'
PROJECT        = 'd2911d10bb'
TOKEN          = os.getenv('PIPER_TOKEN', '')
PROVIDERS      = ['siglip2', 'qwen3', 'face_detect']


def _hdr():
    return {'User-Token': TOKEN, 'Content-Type': 'application/json',
            'Accept': 'application/json'}


def _ls_cat(min_age):
    if min_age is None: return None
    if min_age <= 14:   return 'child'
    if min_age <= 17:   return 'teen'
    return 'adult'


def _variant_for(lbl):
    if lbl in ('child', 'teen'): return 'positive'
    if lbl == 'adult':           return 'negative'
    return None


_MAX_SIDE = 768          # px — second-pass resize before sending to Piper
_MAX_PAYLOAD_BYTES = 3_500_000  # ~3.5 MB raw, ~4.7 MB base64 — well under 413 limit
_JPEG_QUALITY_HI = 88
_JPEG_QUALITY_LO = 72


def _build_data_uri(local_path: Path) -> str:
    """Read file, normalise via PIL to a JPEG data URI.

    Why always JPEG (regardless of source):
      • Animated GIFs crash siglip2 with 500. We take only the first frame and
        re-encode → guaranteed static image.
      • PNG with alpha or 16-bit channels can hit obscure Piper edges. JPEG is
        the lowest common denominator.
      • Lets us re-resize to MAX_SIDE and back off quality if the resulting
        base64 payload would exceed Piper's request size.

    On any PIL failure falls back to the raw bytes (matches old behaviour) so
    a corrupted file still surfaces a clean Piper error rather than crashing
    locally.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        # No Pillow → fall back to raw bytes path
        raw = local_path.read_bytes()
        mime, _ = mimetypes.guess_type(local_path.name)
        if not mime or not mime.startswith('image/'):
            ext = local_path.suffix.lower().lstrip('.')
            mime = f'image/{"jpeg" if ext == "jpg" else ext}'
        return f'data:{mime};base64,' + base64.b64encode(raw).decode('ascii')

    import io
    try:
        with Image.open(local_path) as im:
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            # Force RGB (drops alpha, drops palette, gives JPEG-ready data)
            if im.mode in ('RGBA', 'LA', 'P'):
                bg = Image.new('RGB', im.size, (0, 0, 0))
                src = im.convert('RGBA') if im.mode == 'P' else im
                bg.paste(src, mask=src.split()[-1] if src.mode in ('RGBA', 'LA') else None)
                im = bg
            elif im.mode != 'RGB':
                im = im.convert('RGB')

            w, h = im.size
            longest = max(w, h)
            if longest > _MAX_SIDE:
                ratio = _MAX_SIDE / longest
                im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=_JPEG_QUALITY_HI, optimize=True)
            if buf.tell() > _MAX_PAYLOAD_BYTES:
                # Re-encode at lower quality to fit Piper's request limit
                buf = io.BytesIO()
                im.save(buf, 'JPEG', quality=_JPEG_QUALITY_LO, optimize=True)
        raw = buf.getvalue()
    except Exception:
        # PIL choked — let Piper see the original bytes; if it 500s, the
        # permanent-error path in run_pipeline catches it.
        raw = local_path.read_bytes()

    return 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')


def run_pipeline(local_path: Path, providers=None) -> dict:
    """Returns full outputs dict on success, or {'error': ...}.

    Polling exits as soon as siglip2 + (face_detect OR qwen3) reported.
    Mirrors moderate_disagree.py — face_detect 'no faces detected' is a soft
    error: siglip2 may have run anyway, so we still return.
    """
    try:
        data_uri = _build_data_uri(local_path)
    except Exception as e:
        return {'error': f'read: {type(e).__name__}: {str(e)[:120]}'}

    prov = providers if providers is not None else PROVIDERS
    # NB: top-level pipeline input is named `providers_e0` (with _e0 suffix),
    # which Piper internally flows to prepare_params.inputs.providers. Sending
    # plain `providers` is ignored → pipeline uses whatever default the UI
    # persisted (often qwen3 only from manual tests). See task #115 for the
    # original fix in moderate_disagree.py — this is the same regression.
    #
    # Pipeline requires `image` + `question` (no defaults), plus `model` and
    # `labels` (have defaults). We always send `question` and `model` explicitly
    # because (a) `question` has no default → launch stays "Inputs not filled
    # yet", (b) model default is `deepseek-r1:8b` which is text-only — qwen3
    # node needs the vision variant `qwen3-vl:8b-instruct` to actually analyse
    # the image. `labels` uses pipeline default (big shared NSFW taxonomy).
    payload = {'inputs': {
        'image':         data_uri,
        'providers_e0':  prov,
        'model':         'qwen3-vl:8b-instruct',
        'question':      'Detect violations',
        # categories_e0 tells Evaluate SigLIP which LGBM heads to run.
        # Without it Underage/Bestiality LGBM don't evaluate → no V8 score.
        'categories_e0': ['underage_slim', 'bestiality'],
        # LGBM Underage node only emits scores when this flag is true.
        # Default in pipeline is false (would skip LGBM and leave V8=null).
        'lgbm_enabled':  True,
    }}
    try:
        r = httpx.post(f'{PIPER_BASE}/projects/{PROJECT}/launch',
                       headers=_hdr(), json=payload, timeout=60)
        # 4xx at launch = client-side issue with this specific file.
        #   413 → payload still too big after our JPEG resize.
        #   400 → Piper rejected the input shape/content (often malformed image).
        # Both are permanent — mark unprocessable so resume skips this item.
        if r.status_code in (400, 413):
            return {'unprocessable': True,
                    '_face_error': f'HTTP {r.status_code}: {r.text[:160]}'}
        # 5xx at launch (server hiccup) → retryable, surface as error.
        r.raise_for_status()
        run_id = r.json()['_id']
    except httpx.HTTPStatusError as e:
        return {'error': f'launch: HTTP {e.response.status_code}: {str(e)[:120]}'}
    except Exception as e:
        return {'error': f'launch: {type(e).__name__}: {str(e)[:120]}'}

    # Wait for siglip2 + qwen3 specifically — qwen3 is the slowest (~20-30s) AND
    # the source of our age label, so we MUST not exit before it lands.
    # face_detect is independent; we grab whatever it produced.
    # Tracks soft errors so we can fall back to siglip2-only-with-face-errored
    # results if qwen3 also fails to return.
    face_errored = False
    face_err_str = None
    for _ in range(60):  # ~3 min total
        time.sleep(3)
        try:
            rs = httpx.get(f'{PIPER_BASE}/launches/{run_id}/state',
                           headers=_hdr(), timeout=20)
            if rs.status_code != 200: continue
            st = rs.json()
        except Exception:
            continue
        outs = st.get('outputs') or {}
        errs = st.get('errors') or []

        if errs:
            err_str = str(errs)
            err_low = err_str.lower()
            # Soft "no faces detected" from face_detect — don't bail out, qwen3
            # might still find faces itself (qwen3-vl is a VLM, not a face
            # detector). Mark it and keep polling.
            if 'no faces detected' in err_low:
                face_errored = True
                face_err_str = err_str
            else:
                # Hard permanent errors — bail immediately so we don't retry.
                if any(s in err_low for s in (
                        'premature end', 'vipsjpeg', 'unable to decode',
                        'invalid image', 'corrupt jpeg', 'bad image data')):
                    return {'unprocessable': True, '_face_error': err_str}
                # Any other error string but with siglip2 result already in:
                # surface what we have so the row is processed.
                if 'siglip2_labels' in outs:
                    outs['_face_error'] = err_str
                    return outs
                return {'error': err_str[:200]}

        # Pipeline d2911d10bb emits siglip2 output under either key depending on
        # provider mix: `siglip2_labels` for plain siglip2-only legacy mode, or
        # `siglip2_details` (aggregated by Evaluate SigLIP node downstream of
        # LGBM) which is what we actually get back in production. The working
        # rescore_via_v11.py picks up `siglip2_details` — mirror that here.
        has_siglip = ('siglip2_labels' in outs) or ('siglip2_details' in outs)
        has_qwen3  = any(k.startswith('qwen3_') for k in outs)
        # qwen3-only run (recovery mode): exit as soon as qwen3 lands.
        if 'siglip2' not in prov:
            if has_qwen3:
                return outs
            continue
        # siglip2-only run (--missing-scores recovery): qwen3 wasn't requested,
        # so don't wait for it — exit as soon as siglip2 lands. Otherwise we sit
        # in this loop for the full 180s timeout and return {'error':'timeout'}
        # for every item, which is exactly what happened on the first attempt.
        if 'qwen3' not in prov:
            if has_siglip:
                return outs
            continue
        # Primary done condition: siglip2 AND qwen3 both reported.
        if has_siglip and has_qwen3:
            if face_errored:
                outs['_face_error'] = face_err_str
            return outs

    # Timeout. If siglip2 landed, salvage that (qwen3 was just slow / hung).
    # If face_detect errored with no-face, this is effectively a "no_face" row.
    if face_errored:
        return {'no_face': True, '_face_error': face_err_str}
    return {'error': 'timeout'}


def process_one(item_id: str, local_path_rel: str,
                qwen3_only: bool = False, siglip2_only: bool = False):
    """Run pipeline and return DB-update payload for the row.

    qwen3_only=True   — send only ['qwen3'] (recovery for missing qwen3 faces).
                        UPDATE touches qwen3_result + label only.
    siglip2_only=True — send only ['siglip2'] (recovery for missing V6/V8/V11
                        which need siglip2_details). UPDATE touches piper_result
                        only; qwen3_result/label preserved.
    Default           — send all three providers, full UPDATE.
    """
    abs_path = BASE / local_path_rel
    if not abs_path.exists():
        return item_id, {'error': f'file missing: {local_path_rel}'}
    if siglip2_only:
        providers = ['siglip2']
    elif qwen3_only:
        providers = ['qwen3']
    else:
        providers = PROVIDERS
    outs = run_pipeline(abs_path, providers=providers)
    now = datetime.utcnow().isoformat()
    if 'error' in outs:
        return item_id, {'error': outs['error'], 'processed_at': now}

    # ── "no faces detected" path ─────────────────────────────────────────────
    # face_detect failed permanently because the image has no people. Persist a
    # sentinel piper_result/qwen3_result so the resume filter (piper_result IS
    # NULL) won't retry this image. label stays None — the user can manually
    # mark it as adult/none-of-the-above in the gallery if needed.
    if outs.get('no_face'):
        piper_result = {
            'siglip2_labels':     [],
            'siglip2_passed':     True,
            'siglip2_details':    None,
            'face_detect_result': None,
            '_face_error':        outs.get('_face_error'),
            'no_face':            True,
        }
        return item_id, {
            'piper_result':  json.dumps(piper_result, ensure_ascii=False),
            'qwen3_result':  json.dumps({'faces': []}, ensure_ascii=False),
            'label':         None,
            'label_source':  None,
            'variant':       None,
            'processed_at':  now,
        }

    # ── Permanent client-side failure (413 Payload Too Large, malformed file,
    #     unsupported format that Piper 500s on). Marked unprocessable=True so
    #     the resume filter skips them. User can review or delete in gallery.
    if outs.get('unprocessable'):
        piper_result = {
            'siglip2_labels':     [],
            'siglip2_passed':     True,
            'siglip2_details':    None,
            'face_detect_result': None,
            '_face_error':        outs.get('_face_error'),
            'unprocessable':      True,
        }
        return item_id, {
            'piper_result':  json.dumps(piper_result, ensure_ascii=False),
            'qwen3_result':  json.dumps({'faces': []}, ensure_ascii=False),
            'label':         None,
            'label_source':  None,
            'variant':       None,
            'processed_at':  now,
        }

    # ── siglip2 ─────────────────────────────────────────────────────────────
    siglip_labels  = outs.get('siglip2_labels') or []
    siglip_passed  = 1 if 'underage' not in siglip_labels else 0
    siglip_details = outs.get('siglip2_details')

    # ── face_detect (Index=0 face). Keep only ageFrom/ageTo. ───────────────
    feat = outs.get('features') or outs.get('face_detect_result') or {}
    fd_from = feat.get('ageFrom') if feat.get('ageFrom') is not None else feat.get('age_from')
    fd_to   = feat.get('ageTo')   if feat.get('ageTo')   is not None else feat.get('age_to')
    face_detect = ({'ageFrom': fd_from, 'ageTo': fd_to}
                   if (fd_from is not None or fd_to is not None) else None)

    piper_result = {
        'siglip2_labels':     siglip_labels,
        'siglip2_passed':     bool(siglip_passed),
        'siglip2_details':    siglip_details,
        'face_detect_result': face_detect,
    }

    # ── qwen3 (faces + description). Description is the VLM's free-form
    #     summary of the image; useful for manual review of edge cases.
    qwen3_faces_raw = (outs.get('qwen3_faces')
                       or (outs.get('qwen3_details') or {}).get('faces')
                       or [])
    qwen3_faces = []
    for f in qwen3_faces_raw:
        a = f.get('ageFrom') if f.get('ageFrom') is not None else f.get('age_from')
        b = f.get('ageTo')   if f.get('ageTo')   is not None else f.get('age_to')
        if a is None and b is None:
            continue
        qwen3_faces.append({'ageFrom': a, 'ageTo': b})

    qwen3_desc = (outs.get('qwen3_description')
                  or (outs.get('qwen3_details') or {}).get('description')
                  or '')

    qwen3_result = {'faces': qwen3_faces}
    if qwen3_desc:
        qwen3_result['description'] = qwen3_desc

    # min_age is used locally to derive the AI label; not persisted.
    min_age = None
    for f in qwen3_faces:
        a = f.get('ageFrom')
        if a is not None and (min_age is None or a < min_age):
            min_age = a

    label   = _ls_cat(min_age)
    variant = _variant_for(label)

    if siglip2_only:
        # Only refresh piper_result (with new siglip2_details for V6/V8/V11).
        # Don't overwrite qwen3_result / label — those may already be set by
        # earlier runs or manual labelling.
        return item_id, {
            'piper_result':  json.dumps(piper_result, ensure_ascii=False),
            'processed_at':  now,
            'siglip2_only':  True,
        }

    out = {
        'qwen3_result':  json.dumps(qwen3_result, ensure_ascii=False),
        'label':         label,
        'label_source':  'ai' if label else None,
        'variant':       variant,
        'processed_at':  now,
    }
    if qwen3_only:
        # Recovery mode: don't touch piper_result (siglip2/face_detect kept as-is).
        out['qwen3_only'] = True
    else:
        out['piper_result'] = json.dumps(piper_result, ensure_ascii=False)
    return item_id, out


def _check_pipeline_providers(needed):
    """Fetch the Piper project config and verify the prepare_params node's
    `inputs.providers.enum` includes every provider we plan to request.

    The Prepare params script's JS does `if (providers.includes('X'))` checks
    per provider — if X isn't wired into the pipeline, the corresponding
    downstream node simply doesn't receive an image and we silently lose that
    provider's data. Catch this BEFORE burning hours on a half-broken pipeline.

    Returns True if OK to proceed, False if abort.
    """
    try:
        r = httpx.get(f'{PIPER_BASE}/projects/{PROJECT}',
                      headers=_hdr(), timeout=20)
        if r.status_code != 200:
            print(f'WARN: pipeline config fetch failed (HTTP {r.status_code}); '
                  'skipping providers validation.', file=sys.stderr)
            return True
        data = r.json()
        pipe = data.get('pipeline')
        if isinstance(pipe, str):
            pipe = json.loads(pipe)
        # The top-level pipeline input is `providers_e0` (suffix avoids clash
        # with node-level `providers`). Its enum lists supported values.
        # Fall back to the node-level enum if the top-level one is missing
        # (older pipeline shape).
        top_inputs = (pipe or {}).get('inputs') or {}
        nodes = (pipe or {}).get('nodes') or {}
        prep = nodes.get('prepare_params') or {}
        enum = (((top_inputs.get('providers_e0') or {}).get('enum'))
                or ((prep.get('inputs') or {}).get('providers') or {}).get('enum')
                or [])
        if not enum:
            print('WARN: pipeline.inputs.providers_e0.enum missing in config; '
                  'skipping providers validation.', file=sys.stderr)
            return True
        missing = [p for p in needed if p not in enum]
        if missing:
            print('\n!!! Pipeline {} does NOT support requested providers: {}'
                  .format(PROJECT, missing), file=sys.stderr)
            print('!!! Allowed by prepare_params.enum: {}'.format(enum), file=sys.stderr)
            print('!!! Add the missing providers to https://piper-next.artworks.ai/'
                  'en/projects/{} → Prepare params → Providers enum, save the '
                  'pipeline, then re-run.'.format(PROJECT), file=sys.stderr)
            return False
        print(f'  pipeline providers OK — prepare_params enum: {enum}', flush=True)
        return True
    except Exception as e:
        print(f'WARN: pipeline validation failed ({type(e).__name__}: {e}); '
              'proceeding without check.', file=sys.stderr)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit',   type=int, default=0,
                    help='Process at most N items (0 = all unprocessed)')
    ap.add_argument('--redo',    action='store_true',
                    help='Re-score items that already have piper_result')
    ap.add_argument('--missing-qwen3', action='store_true',
                    help='Re-score items where face_detect found a face but '
                         'qwen3_result.faces is empty (recovers items that were '
                         'returned early before qwen3 finished — see polling bug fix).')
    ap.add_argument('--redo-no-face', action='store_true',
                    help='Re-score items previously marked no_face=True. With the '
                         'new polling that waits for qwen3+siglip2 together, '
                         'these items now get full siglip2_details (so V6/V8/V11 '
                         'can score them) — the old early-exit-on-face_detect '
                         'bug stored no usable data.')
    ap.add_argument('--missing-scores', action='store_true',
                    help='Re-score items where V6/V8/V11 would show "—" in the '
                         'gallery: siglip2_details is None (no_face/unprocessable '
                         'sentinels), OR underage labels are empty. Full pipeline '
                         'run, populates everything from scratch.')
    args = ap.parse_args()

    if not TOKEN:
        print('ERR: PIPER_TOKEN not set in .env', file=sys.stderr); sys.exit(1)
    if not DB_PATH.exists():
        print(f'ERR: {DB_PATH} not found', file=sys.stderr); sys.exit(1)

    # Pipeline providers sanity check — fail-fast if pipeline can't deliver what
    # we plan to ask. --missing-qwen3 only needs qwen3; everything else needs all 3.
    if args.missing_qwen3:
        needed_providers = ['qwen3']
    elif args.missing_scores:
        needed_providers = ['siglip2']
    else:
        needed_providers = PROVIDERS
    if not _check_pipeline_providers(needed_providers):
        sys.exit(2)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    if args.missing_scores:
        # Cards showing "—" for V6/V8/V11 in the gallery. Two cases:
        #   (1) siglip2_details missing entirely (sentinels, old broken runs)
        #   (2) siglip2_details exists but labels.underage/.adult empty → V8 fails
        # Either way the gallery can't compute V6/V8/V11. Re-run full pipeline.
        # ONLY consider items where siglip2 truly didn't run — i.e. no
        # siglip2_details at all. Items where siglip2_details exists but
        # labels.underage and labels.adult are both empty are NOT missing:
        # that's a valid result for clean images, and gallery LGBM can score
        # them on a 0-vector. Earlier SQL marked them as "missing" and forced
        # 163 pointless re-runs that returned identical empty labels.
        rows = conn.execute("""
            SELECT id, local_path FROM borderlands_pool
            WHERE local_path IS NOT NULL
              AND (deleted IS NULL OR deleted=0)
              AND (
                  piper_result IS NULL
                  OR json_extract(piper_result, '$.siglip2_details') IS NULL
              )
        """).fetchall()
    elif args.redo_no_face:
        # Select old no_face=True sentinels — re-run full pipeline so siglip2
        # data lands (then V6/V8/V11 in gallery can score them).
        rows = conn.execute("""
            SELECT id, local_path FROM borderlands_pool
            WHERE local_path IS NOT NULL
              AND piper_result IS NOT NULL
              AND (deleted IS NULL OR deleted=0)
              AND COALESCE(json_extract(piper_result, '$.no_face'), 0) = 1
        """).fetchall()
    elif args.missing_qwen3:
        # Recovery mode: items with face_detect age but no qwen3 faces.
        # Skip no_face/unprocessable sentinels — those are confirmed bad.
        rows = conn.execute("""
            SELECT id, local_path FROM borderlands_pool
            WHERE local_path IS NOT NULL
              AND piper_result IS NOT NULL
              AND (deleted IS NULL OR deleted=0)
              AND COALESCE(json_extract(piper_result, '$.no_face'),       0) = 0
              AND COALESCE(json_extract(piper_result, '$.unprocessable'), 0) = 0
              AND json_extract(piper_result, '$.face_detect_result.ageFrom') IS NOT NULL
              AND (qwen3_result IS NULL
                   OR COALESCE(json_array_length(json_extract(qwen3_result, '$.faces')), 0) = 0)
        """).fetchall()
    elif args.redo:
        rows = conn.execute(
            "SELECT id, local_path FROM borderlands_pool "
            "WHERE local_path IS NOT NULL AND (deleted IS NULL OR deleted=0)"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, local_path FROM borderlands_pool "
            "WHERE local_path IS NOT NULL AND piper_result IS NULL "
            "  AND (deleted IS NULL OR deleted=0)"
        ).fetchall()
    if args.limit:
        rows = rows[:args.limit]

    print(f'=== Moderate Borderlands ===')
    print(f'  to process:  {len(rows)}')
    print(f'  workers:     {args.workers}')
    # Print the ACTUAL providers we'll send, not the global PROVIDERS constant.
    # --missing-scores → ['siglip2'], --missing-qwen3 → ['qwen3'], default → all 3.
    print(f'  providers:   {needed_providers}')
    print(f'  project:     {PROJECT}\n')

    if not rows:
        print('Nothing to do.'); return

    saved = errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, r['id'], r['local_path'],
                                args.missing_qwen3, args.missing_scores): r['id']
                   for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            item_id, res = fut.result()
            if 'error' in res:
                errors += 1
                conn.execute("UPDATE borderlands_pool SET processed_at=? WHERE id=?",
                             (res['processed_at'], item_id))
                print(f'  [{i:>4}/{len(rows)}] x {item_id[:30]:<30}  {res["error"][:80]}',
                      flush=True)
                continue
            if res.get('siglip2_only'):
                # Refresh ONLY piper_result (new siglip2_details for V6/V8/V11).
                # qwen3_result / label / variant preserved — may already be set.
                conn.execute("""
                    UPDATE borderlands_pool SET
                        piper_result=?, processed_at=?
                    WHERE id=?
                """, (res['piper_result'], res['processed_at'], item_id))
            elif res.get('qwen3_only'):
                # Recovery mode: preserve existing piper_result; only refresh qwen3 + label.
                conn.execute("""
                    UPDATE borderlands_pool SET
                        qwen3_result=?, label=?, label_source=?, variant=?,
                        processed_at=?
                    WHERE id=?
                """, (res['qwen3_result'], res['label'], res['label_source'],
                      res['variant'], res['processed_at'], item_id))
            else:
                conn.execute("""
                    UPDATE borderlands_pool SET
                        piper_result=?, qwen3_result=?,
                        label=?, label_source=?, variant=?,
                        processed_at=?
                    WHERE id=?
                """, (res['piper_result'], res['qwen3_result'],
                      res['label'], res['label_source'], res['variant'],
                      res['processed_at'], item_id))
            saved += 1
            if saved % 25 == 0:
                conn.commit()
            if res.get('qwen3_only'):
                q3_faces = json.loads(res['qwen3_result'] or '{}').get('faces') or []
                tag = f'q3 {len(q3_faces)} face(s)'
                age = '-'
            elif res.get('siglip2_only'):
                tag = ', '.join(json.loads(res['piper_result'] or '{}').get('siglip2_labels', [])[:3]) or 'passed'
                age = '(siglip2 only)'
            else:
                tag = ', '.join(json.loads(res['piper_result'] or '{}').get('siglip2_labels', [])[:3]) or 'passed'
                age = res['label'] or '-'
            print(f'  [{i:>4}/{len(rows)}] OK {item_id[:30]:<30}  age={age:<18}  {tag[:40]}',
                  flush=True)

    conn.commit()
    conn.close()
    dt = time.time() - t0
    print(f'\n=== Done in {dt:.0f}s ===')
    print(f'  saved:  {saved}')
    print(f'  errors: {errors}')


if __name__ == '__main__':
    main()
