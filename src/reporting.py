"""Сохранение результатов экспериментов в reports/results.json."""

import json

from src import config

RESULTS_PATH = config.REPORTS_DIR / "results.json"


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_result(name, result):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    results[name] = result
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def print_result(name, result):
    parts = [f"{key}={value:.4f}" for key, value in result.items() if isinstance(value, float)]
    print(f"{name}: " + ", ".join(parts))
