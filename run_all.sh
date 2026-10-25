#!/usr/bin/env bash
# Полный прогон эксперимента с нуля. На ноутбуке с 2 ядрами — около 2,5 часов.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f data/raw/interactions.csv ]; then
    bash scripts/00_download_data.sh
fi

python scripts/01_prepare_data.py
python scripts/02_build_candidates.py
python tests/test_no_leakage.py
python scripts/03_train_baseline.py
python scripts/04_auto_features.py
python scripts/05_family_selection.py logloss
python scripts/05_family_selection.py ndcg
python scripts/06_information_ablation.py
python scripts/07_compare.py
python scripts/08_rolling_evaluation.py
python scripts/09_rolling_summary.py
