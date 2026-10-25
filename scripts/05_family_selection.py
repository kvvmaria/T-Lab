"""Шаг 5. Отбор автопризнаков целыми семействами (все окна одного агрегата вместе).

Мотивация — результат шага 4: поштучная permutation importance занижает важность
коррелирующих окон, и отбор выкидывает, например, популярность фильма за разные окна.
Здесь важность считается для семейства целиком.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, features_auto, selection
from src.experiment import fit_and_evaluate, load_inputs, load_week, save_test_predictions
from src.reporting import print_result, save_result

KEYS = ["user_id", "item_id", "week"]


def run_experiment(name, data, targets, features, extra=None):
    model, result = fit_and_evaluate(data, targets, features)
    result.update(extra or {})
    print_result(name, result)
    save_result(name, result)
    save_test_predictions(name, data["test"], model.predictions_["test"])


def main():
    events, _, _ = load_inputs()
    targets = {part: load_week(events, week)[0].target
               for part, week in [("valid", config.VALID_WEEK), ("test", config.TEST_WEEK)]}
    data = {part: pd.read_pickle(config.PROCESSED_DIR / f"auto_{part}.pkl")
            for part in ["train", "valid", "test"]}

    spec_by_name = {s.name: s for s in features_auto.default_specs()}
    features = features_auto.feature_columns(data["test"])
    family_of = {f: spec_by_name[f].family for f in features}

    started = time.time()
    selected, details = selection.run_by_family(data, targets, features, family_of)
    seconds = time.time() - started
    run_experiment("auto_family_selected", data, targets, selected, {"selection_seconds": round(seconds, 1)})

    manual_features = pd.read_csv(config.REPORTS_DIR / "manual_features.csv", header=None)[0].to_list()
    combined = {}
    for part in ["train", "valid", "test"]:
        manual = pd.read_pickle(config.PROCESSED_DIR / f"manual_{part}.pkl")
        combined[part] = manual.merge(data[part][KEYS + selected], on=KEYS, how="inner")
    del data
    run_experiment("manual_plus_auto_family", combined, targets, manual_features + selected)

    out = config.REPORTS_DIR / "selection"
    details["family_importance"].to_csv(out / "family_importance.csv", index=False)
    details["family_curve"].to_csv(out / "family_curve.csv", index=False)
    pd.DataFrame({"feature": selected, "sql": [spec_by_name[f].to_sql() for f in selected]}).to_csv(
        out / "family_selected_features.csv", index=False)


if __name__ == "__main__":
    main()
