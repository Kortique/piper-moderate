# commit_push.ps1
# ---------------
# Logical commits + push for piper-moderate repo.
# Run from project root: powershell -ExecutionPolicy Bypass .\commit_push.ps1

# Continue (not Stop) -- git emits warnings on stderr (e.g. LF->CRLF) which PS
# treats as terminating errors under Stop, breaking the script unfairly.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

# Suppress the LF->CRLF informational warning for this session
$env:GIT_CONFIG_PARAMETERS = "'core.safecrlf=false'"

# 0. Safety: refuse if .env is tracked
$tracked_env = (git ls-files .env)
if ($tracked_env) {
    Write-Host "WARN: .env is tracked! Run: git rm --cached .env" -ForegroundColor Red
    Write-Host "Aborting."
    exit 1
}

# Offer to untrack big files that are now in .gitignore
$big_tracked = git ls-files gallery.db backups/ 2>$null
if ($big_tracked) {
    Write-Host "INFO: these are currently tracked but now in .gitignore:" -ForegroundColor Yellow
    Write-Host $big_tracked
    $ans = Read-Host "Untrack them (git rm --cached)? They stay on disk. [y/N]"
    if ($ans -eq 'y' -or $ans -eq 'Y') {
        git rm -r --cached gallery.db backups/ 2>$null | Out-Null
        Write-Host "Untracked." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== Current state ===" -ForegroundColor Cyan
git status --short
Write-Host ""
$ans = Read-Host "Proceed with commits? [y/N]"
if ($ans -ne 'y' -and $ans -ne 'Y') { Write-Host "Aborted."; exit 0 }

function Has-Staged {
    $diff = (git diff --cached --name-only 2>$null) -join ""
    return [bool]$diff
}

# --- Commit 1: .gitignore ---
git add .gitignore 2>&1 | Out-Null
if (Has-Staged) {
    git commit -m "chore: whitelist data/ - keep lgbm models, k30 subdir, shared splits and JS evals; ignore stale dataset dumps"
    Write-Host "Committed: .gitignore" -ForegroundColor Green
} else {
    Write-Host "Skipped: .gitignore (no diff)" -ForegroundColor DarkGray
}

# --- Commit 2: project meta ---
git add requirements.txt 2>&1 | Out-Null
git add .env.example    2>&1 | Out-Null
git add commit_push.ps1 2>&1 | Out-Null
if (Has-Staged) {
    git commit -m "chore: pin python deps, add .env template and commit/push helper"
    Write-Host "Committed: project meta" -ForegroundColor Green
} else {
    Write-Host "Skipped: project meta (no diff)" -ForegroundColor DarkGray
}

# --- Commit 3: scripts/ ---
git add scripts/ 2>&1 | Out-Null
if (Has-Staged) {
    $smsg = @'
feat(scripts): borderlands batch + rescore hardening + diagnostics

Borderlands ingestion
- scripts/import_borderlands.py   - import ~9.8k local images into gallery.db
- scripts/moderate_borderlands.py - run d2911d10bb pipeline on local data URIs;
  PIL JPEG normalisation (768px, <3.5MB), permanent-error sentinels for HTTP
  400/413 and libvips decode failures, no_face as a soft success; modes:
  --missing-qwen3 (qwen3-only recovery), --redo-no-face (re-run old sentinels),
  --missing-scores (siglip2-only fill of V6/V8/V11 gaps with providers_e0);
  polling now waits for the providers that were actually requested
  (siglip2_details OR siglip2_labels as the siglip2-done signal)

Rescore hardening (V11 native + Tom K30)
- rescore_via_v11.py / rescore_via_tom.py: _atomic_write with explicit
  encoding=utf-8 + fsync; Windows cp1251 default was silently mangling
  cyrillic IDs (e.g. bl_..._эротические_) and corrupting whole JSON caches
- resume now reads raw bytes + errors=ignore fallback so a few corrupt bytes
  don't force a full 25k-record re-score
- Tom resume only treats done=True records as already-scored (was skipping
  prior-error items by mistake)
- providers_e0 enum validation up-front; counter shows 1..len(todo)

Diagnostics + recovery utilities
- scripts/recover_json_caches.py    - atomic UTF-8 rewrite of v11/tom caches
  with timestamped .broken backups, dry-run mode
- scripts/check_piper_pipeline.py   - dump pipeline.inputs/prepare_params,
  recommend providers key per project (providers vs providers_e0)
- scripts/check_borderlands_scores.py - per-model gap coverage report
- scripts/check_shadow.py, diag_labels.py, install_label_guard.py,
  snapshot.py, rollback.py - label-safety + audit trail utilities

LS batch ingestion + V6/V8/V11 holdout retrain
- scripts/fetch_ls_batch.py, import_ls_batch.py, moderate_ls_batch.py
  - end-to-end LS view 65 pull → DB → siglip2
- scripts/train_v6b_holdout.py / train_v8b_holdout.py / train_v11b_holdout.py
  - shared 80/20 holdout (v11_test_split.json), multi-seed, per-source AUC
- scripts/retrain_v6_v8_v11.py, reprocess_session.py, reset_session_grafana.py,
  report_new_grafana.py, deploy_piper_lgbm.py, relabel_pool.py - operations
'@
    git commit -m $smsg
    Write-Host "Committed: scripts/" -ForegroundColor Green
} else {
    Write-Host "Skipped: scripts/ (no diff)" -ForegroundColor DarkGray
}

# --- Commit 4: data/ whitelist ---
git add data/ 2>&1 | Out-Null
if (Has-Staged) {
    $dmsg = @'
data: holdout retrain artifacts + thresholds + V11 native test split

- data/lgbm_*v6b*, *v8b*, *v8bs80*, *v11b*, *v11bs80*  - new holdout-trained models
- data/lgbm_v6_meta.json                                - V6 CV AUC computed retroactively
- data/v11_test_split.json                              - frozen 618-item shared holdout
- data/thresholds.json                                  - persisted UI thresholds (v6/v8/v11/k30tom)
- data/k30/feature_keepers.json, piper_lgbm_model.json,
        side_by_side_results.json, three_way_results.json - K30 study artifacts
'@
    git commit -m $dmsg
    Write-Host "Committed: data/ whitelist" -ForegroundColor Green
} else {
    Write-Host "Skipped: data/ (no diff)" -ForegroundColor DarkGray
}

# --- Commit 5: suggestions/ ---
git add suggestions/ 2>&1 | Out-Null
if (Has-Staged) {
    git commit -m "data(suggestions): add AI-generated label suggestions for review history"
    Write-Host "Committed: suggestions/" -ForegroundColor Green
} else {
    Write-Host "Skipped: suggestions/ (no diff)" -ForegroundColor DarkGray
}

# --- Commit 6: gallery_server.py ---
git add gallery_server.py 2>&1 | Out-Null
if (Has-Staged) {
    $gmsg = @'
feat(gallery): borderlands source + colored borders + lightbox moderation

Borderlands as a first-class source
- borderlands_pool wired into load_eval_data: V6/V8/V11 inline scoring on
  piper_result.siglip2_details (same shape as grafana); V11 native pulled
  from data/v11_native_scores.json with siglip2-based fallback (* marker)
- load_borderlands returns qwen3_description as prompt slot
- borderlands in source dropdown, exclude-sessions panel groups,
  _exclKeyFor → borderlands:all
- Broken-image handler replaces only <img>, keeps action buttons clickable
  (x, star, src-badge)

Drop empty-labels guards on V8/V11 scoring (10 sites)
- Was: "if not u and not a: continue" skipped LGBM whenever siglip2 found
  zero underage AND zero adult tags, leaving ~163 clean-image cards without
  V6/V8/V11 scores even though piper_result.siglip2_details was populated
- Now: LGBM runs on a 0-vector and returns the baseline (~0.05) — correct
  prediction for completely clean adult content, matches Piper-side LGBM
  Underage that already produced lgbm.score in the same shape

Colored frame by label (card + lightbox)
- child=red, teen=orange, adult=green; lightbox uses ::after layer with
  z-index 4 + pointer-events:none so the image cannot occlude the border
- !important on card .lbl-X / lightbox #lb-wrap::after

Exclude-sessions reach AUC + stats
- _sourceMatch() honours excludedSessions when source='all'
- Exclude checkbox handler triggers computeAucForVersion, updateLgbmStats,
  refreshAucDisplays so toolbar numbers stay in sync with visible cards

Lightbox moderation
- Sidebar layout, verdict badge (UNDERAGE / OK / UNKNOWN), per-model
  color-graded scores, age rows, disagreement row, marked toggle,
  thumbnail strip with auto-centering, position counter and hint bar
- Hotkeys: 1/2/3 = child/teen/adult, 4 = delete, 5/m = mark, Enter = next;
  auto-advance after labelling (120ms), mouse-wheel paging (180ms throttle)

Marked items workflow
- localStorage-persisted Set, star button top-center, golden outline
  (outline, not clipped inset shadow), source dropdown "marked" option,
  JSON export marked_YYYYMMDDHHMMSS.json

Card + workflow fixes
- Badge in LGBM mode uses local gallery score, not Piper
- TP/FP/TN/FN computed from live slider threshold
- Viewed-in-lightbox eye badge bottom-center
- "Confirm page" auto-saves in one click; save preserves current page
- Page-size auto-refresh; V11 native primary, no_face skipped safely
- K30 (Tom) count in header stats; /api/save_thr handler restored
- LS session dropdown (formatted like Grafana, working filter)
- V8 scoring for new LS batch reads siglip2_details from DB
- V11 LS fallback uses DB siglip2_details for no_face/missing-rescore items
'@
    git commit -m $gmsg
    Write-Host "Committed: gallery_server.py" -ForegroundColor Green
} else {
    Write-Host "Skipped: gallery_server.py (no diff)" -ForegroundColor DarkGray
}

# --- Push ---
Write-Host ""
$br = (git branch --show-current).Trim()
Write-Host ("Branch: " + $br) -ForegroundColor Cyan
$ans = Read-Host ("Push to origin/" + $br + "? [y/N]")
if ($ans -eq 'y' -or $ans -eq 'Y') {
    git push origin $br
    Write-Host "Pushed." -ForegroundColor Green
} else {
    Write-Host "Push skipped. Commits are local." -ForegroundColor Yellow
}

git log --oneline -12
