# commit_sprint_2026-05.ps1 - commit + push the V*c sprint
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

git status --short
Write-Host ""
$ans = Read-Host "Proceed with staging + commit + push? [y/N]"
if ($ans -ne 'y' -and $ans -ne 'Y') { Write-Host "Aborted."; exit 0 }

# Stage relevant files
git add scripts/make_holdout_split_2026.py 2>&1 | Out-Null
git add scripts/train_v6c_holdout.py       2>&1 | Out-Null
git add scripts/train_v8c_holdout.py       2>&1 | Out-Null
git add scripts/train_v11c_holdout.py      2>&1 | Out-Null
git add scripts/export_v8cs80_js.py        2>&1 | Out-Null
git add scripts/export_v11cs80_js.py       2>&1 | Out-Null
git add scripts/extract_v8cs80_ls_fps.py   2>&1 | Out-Null
git add data/lgbm_underage_v6c.txt         2>&1 | Out-Null
git add data/lgbm_v6c_features.json        2>&1 | Out-Null
git add data/lgbm_v6c_meta.json            2>&1 | Out-Null
git add data/lgbm_underage_v8c.txt         2>&1 | Out-Null
git add data/lgbm_underage_v8cs80.txt      2>&1 | Out-Null
git add data/lgbm_v8c_features.json        2>&1 | Out-Null
git add data/lgbm_v8cs80_features.json     2>&1 | Out-Null
git add data/lgbm_v8cs80_meta.json         2>&1 | Out-Null
git add data/lgbm_underage_v11c.txt        2>&1 | Out-Null
git add data/lgbm_underage_v11cs80.txt     2>&1 | Out-Null
git add data/lgbm_v11c_features.json       2>&1 | Out-Null
git add data/lgbm_v11cs80_features.json    2>&1 | Out-Null
git add data/lgbm_v11cs80_meta.json        2>&1 | Out-Null
git add data/lgbm_evaluate_v8cs80.js       2>&1 | Out-Null
git add data/lgbm_evaluate_v11cs80.js      2>&1 | Out-Null
git add data/v11_test_split_2026.json      2>&1 | Out-Null
git add data/v8cs80_ls_fps_2026-05.json    2>&1 | Out-Null
git add data/lgbm_v6b_meta.json            2>&1 | Out-Null
git add data/lgbm_v8bs80_meta.json         2>&1 | Out-Null
git add data/lgbm_v11bs80_meta.json        2>&1 | Out-Null
git add data/lgbm_underage_v6b.txt         2>&1 | Out-Null
git add data/lgbm_underage_v8bs80.txt      2>&1 | Out-Null
git add data/lgbm_underage_v11bs80.txt     2>&1 | Out-Null
git add data/lgbm_v6b_features.json        2>&1 | Out-Null
git add data/lgbm_v8bs80_features.json     2>&1 | Out-Null
git add data/lgbm_v11bs80_features.json    2>&1 | Out-Null
git add gallery_server.py                  2>&1 | Out-Null
git add README.md                          2>&1 | Out-Null
git add sprint_2026-05_commit_msg.txt      2>&1 | Out-Null
git add commit_sprint_2026-05.ps1          2>&1 | Out-Null

$diff = (git diff --cached --name-only 2>$null) -join "|"
if (-not $diff) { Write-Host "Nothing staged."; exit 0 }
Write-Host ("staged: " + $diff)

git commit -F sprint_2026-05_commit_msg.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Commit failed." -ForegroundColor Red; exit 1 }
Write-Host "Committed." -ForegroundColor Green

git push origin main
git log --oneline -5
