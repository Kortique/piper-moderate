#!/usr/bin/env bash
# run_v10_pipeline.sh
# Ждёт завершения rescore_ls_holdout.py, затем обучает V10 и сравнивает.
# Usage: bash scripts/run_v10_pipeline.sh

set -e
BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"

TARGET=2216

echo "=== Ожидаем завершения ресcore LS holdout (target: $TARGET items) ==="
while true; do
    DONE=$(python3 -c "
import json, sys
try:
    d = json.load(open('data/ls_holdout_rescored.json'))
    print(sum(1 for r in d if r.get('done')))
except:
    print(0)
" 2>/dev/null)
    echo "  done=$DONE / $TARGET  ($(date +%H:%M:%S))"
    if [ "$DONE" -ge "$TARGET" ]; then
        echo "  Ресcore завершён!"
        break
    fi
    # Проверяем, жив ли процесс
    if ! pgrep -f rescore_ls_holdout.py > /dev/null 2>&1; then
        echo "  Процесс завершился (done=$DONE)"
        break
    fi
    sleep 30
done

echo ""
echo "=== Обучаем V10 ==="
python3 scripts/train_lgbm_v10.py

echo ""
echo "=== Сравниваем V7/V8/V9/V10 ==="
python3 scripts/compare_v7_v8_v9_v10.py

echo ""
echo "=== Done ==="
