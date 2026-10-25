#!/usr/bin/env bash
# Скачивает датасет KION (МТС) в data/raw.
#
# Важно: файл interactions.csv.zip в корне репозитория KION обрезан (~1.6 млн строк из 5.5 млн),
# поэтому полную таблицу взаимодействий берём из data_en.zip.
# Метаданные пользователей и фильмов берём из data_original (на русском).
set -euo pipefail

RAW_DIR="$(dirname "$0")/../data/raw"
TMP_DIR="$(mktemp -d)"
mkdir -p "$RAW_DIR"

git clone --depth 1 https://github.com/irsafilo/KION_DATASET "$TMP_DIR/KION_DATASET"

unzip -o -q -j "$TMP_DIR/KION_DATASET/data_en.zip" data_en/interactions.csv -d "$RAW_DIR"
cp "$TMP_DIR/KION_DATASET/data_original/users.csv" "$RAW_DIR/"
cp "$TMP_DIR/KION_DATASET/data_original/items.csv" "$RAW_DIR/"

rm -rf "$TMP_DIR"
echo "Готово. Строк во взаимодействиях: $(($(wc -l < "$RAW_DIR/interactions.csv") - 1))"
