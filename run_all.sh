#!/usr/bin/env bash
# рядом должны лежать train/test csv и папка "Архив 3"
set -euo pipefail
cd "$(dirname "$0")"
python extract_content.py   # ~20 мин
python cnn_url.py           # ~15 мин
python pipeline.py          # ~15 мин, итог в submission_v2.csv
