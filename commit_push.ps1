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
feat(scripts): rescore + holdout retrain + K30 import utilities

- scripts/rescore_via_v11.py  - score LS+Grafana via V11 pipeline ce79f7e299
  with retry/backoff on 5xx and timeouts, no_face handling
- scripts/rescore_k30_v6v8.py - fill 180-tag K30 input via d2911d10bb
- scripts/rescore_via_tom.py  - score via Tom pipeline a4aa9dbd9c
- scripts/rescore_k30_v9.py   - earlier K30 rescore variant
- scripts/import_k30.py       - import Tom K=30 dataset into gallery.db
- scripts/run_category.py     - category-scoped runner
- scripts/train_v6b_holdout.py / train_v8b_holdout.py / train_v11b_holdout.py
  - retrain V6/V8/V11 with shared 80/20 holdout (v11_test_split.json),
    multi-seed lgb, per-source AUC breakdown
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
feat(gallery): lightbox moderation mode + marked items workflow

Lightbox
- Sidebar layout: image left, info column right (overlay fallback on narrow viewports)
- Verdict badge: UNDERAGE / OK / UNKNOWN (effective label first, then majority model vote)
- Per-model LGBM scores with color grading by their own thresholds
- Age rows: qwen3, face_detect, LS source range
- Disagreement row with low/mid/hi color grading
- Marked toggle row in sidebar
- Thumbnails strip with auto-centering on current item (cubic-bezier transition)
- Position counter "N / TOTAL" + hint bar

Moderation hotkeys
- 1/2/3 = child/teen/adult, 4 = delete, 5/m = mark, Enter = next
- Auto-advance after labelling and deletion (120ms flash delay)
- Mouse wheel paging (180ms throttle)

Marked items workflow
- Global Set persisted to localStorage across sessions
- Star button top-center of card; click or hotkey toggles
- Full-perimeter golden outline on marked cards (outline, not clipped inset shadow)
- New "marked" option in source filter
- JSON export: marked_YYYYMMDDHHMMSS.json with id, source, label, ages, model scores

Card UX
- Viewed-in-lightbox eye badge bottom-center of preview
- Deletion mark synced between lightbox and gallery card

Workflow fixes
- "Confirm page" auto-saves in one click (was two clicks)
- Save preserves current page (previously reset to page 1)
- Page-size select auto-refreshes on change
- V11 native scores wired as primary source (data/v11_native_scores.json), no_face skipped without crashing
- K30 (Tom) count in header stats line
- /api/save_thr handler fixed (orphan except after mount-write truncation)
- Auto-confirm pass on save for grafana/k30 items touched or viewed
- Local label_confirmed flip after save (counter drops to 0 without reload)
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
