"""Шаг 6. Честное сравнение по доступной информации.

В ручном наборе есть признаки генератора кандидатов: флаги источников (src_*) и
covisitation-скор — связь item–item через пользователей. Генератор признаков такую
связь выразить не может: она требует self-join таблицы событий, а не GROUP BY.

Чтобы понять, откуда разница в качестве, считаем две модели:
  * ручные признаки БЕЗ признаков кандидатов;
  * отобранные автопризнаки + признаки кандидатов.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.experiment import fit_and_evaluate, free_memory, load_inputs, save_test_predictions, load_week
from src.reporting import print_result, save_result

CANDIDATE_FEATURES = ["src_popular", "src_genre", "src_covisit", "covisit_score"]
KEYS = ["user_id", "item_id", "week"]


def run_experiment(name, data, targets, features):
    model, result = fit_and_evaluate(data, targets, features)
    print_result(name, result)
    save_result(name, result)
    save_test_predictions(name, data["test"], model.predictions_["test"])


def main():
    events, _, _ = load_inputs()
    targets = {part: load_week(events, week)[0].target
               for part, week in [("valid", config.VALID_WEEK), ("test", config.TEST_WEEK)]}

    manual = {part: pd.read_pickle(config.PROCESSED_DIR / f"manual_{part}.pkl")
              for part in ["train", "valid", "test"]}
    manual_features = pd.read_csv(config.REPORTS_DIR / "manual_features.csv", header=None)[0].to_list()
    selected = pd.read_csv(config.REPORTS_DIR / "selection" / "selected_features.csv")["feature"].to_list()

    no_candidate = [f for f in manual_features if f not in CANDIDATE_FEATURES]
    run_experiment("manual_without_candidate_features", manual, targets, no_candidate)

    combined = {}
    for part in ["train", "valid", "test"]:
        auto = pd.read_pickle(config.PROCESSED_DIR / f"auto_{part}.pkl")[KEYS + ["label"] + selected]
        combined[part] = auto.merge(manual[part][KEYS + CANDIDATE_FEATURES], on=KEYS, how="inner")
        del auto
        free_memory()
    run_experiment("auto_selected_plus_candidate_features", combined, targets,
                   selected + CANDIDATE_FEATURES)


if __name__ == "__main__":
    main()
