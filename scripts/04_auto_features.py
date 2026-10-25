"""Шаг 4. Автоматические признаки: генерация по сетке, абляции и отбор."""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, features_auto, selection
from src.experiment import build_split, fit_and_evaluate, load_inputs, save_test_predictions
from src.reporting import print_result, save_result


def run_experiment(name, data, targets, features, extra=None):
    model, result = fit_and_evaluate(data, targets, features)
    result.update(extra or {})
    print_result(name, result)
    save_result(name, result)
    save_test_predictions(name, data["test"], model.predictions_["test"])
    return result


def main():
    events, users, items = load_inputs()
    specs = features_auto.default_specs()
    spec_by_name = {s.name: s for s in specs}

    def build(week, candidates):
        return features_auto.build(week, candidates, users, items, specs)

    data, targets, build_seconds = build_split(events, build)
    all_features = features_auto.feature_columns(data["test"])
    print(f"Сгенерировано признаков: {len(all_features)}, построение: {build_seconds:.0f} с")

    # 1. Все признаки сетки сразу, без отбора
    run_experiment("auto_all", data, targets, all_features, {"build_seconds": round(build_seconds, 1)})

    # 2. Абляции: какие части сетки важны
    def subset(condition):
        return [f for f in all_features if condition(spec_by_name[f])]

    ablations = {
        "auto_only_user_item": subset(lambda s: s.entity in ("user", "item")),
        "auto_without_segments": subset(lambda s: not s.entity.startswith("item×")),
        "auto_windows_7_30_all": subset(lambda s: s.window in (7, 30, None)),
        "auto_only_all_events": subset(lambda s: s.event == "all"),
    }
    for name, features in ablations.items():
        run_experiment(name, data, targets, features)

    # 3. Отбор признаков (только train + valid)
    started = time.time()
    selected, details = selection.run(data, targets, all_features)
    selection_seconds = time.time() - started
    run_experiment("auto_selected", data, targets, selected,
                   {"selection_seconds": round(selection_seconds, 1)})

    # 4. Ручные + отобранные автоматические
    combined = {}
    for part in ["train", "valid", "test"]:
        manual = pd.read_pickle(config.PROCESSED_DIR / f"manual_{part}.pkl")
        combined[part] = manual.merge(data[part][["user_id", "item_id", "week"] + selected],
                                      on=["user_id", "item_id", "week"], how="inner")
    manual_features = pd.read_csv(config.REPORTS_DIR / "manual_features.csv", header=None)[0].to_list()
    run_experiment("manual_plus_auto_selected", combined, targets, manual_features + selected)

    # 5. Сохраняем артефакты отбора для анализа
    out = config.REPORTS_DIR / "selection"
    out.mkdir(parents=True, exist_ok=True)
    screening = details["screening"]
    screening["entity"] = screening["feature"].map(lambda f: spec_by_name[f].entity)
    screening["family"] = screening["feature"].map(lambda f: spec_by_name[f].family)
    screening["window"] = screening["feature"].map(lambda f: spec_by_name[f].window)
    screening.to_csv(out / "screening.csv", index=False)
    details["importance"].to_csv(out / "permutation_importance.csv", index=False)
    details["curve"].to_csv(out / "top_n_curve.csv", index=False)
    pd.Series(details["dropped_by"], name="kept_twin").to_csv(out / "dropped_as_redundant.csv")
    pd.DataFrame({"feature": selected,
                  "sql": [spec_by_name[f].to_sql() for f in selected]}).to_csv(
        out / "selected_features.csv", index=False)

    for part, df in data.items():
        df.to_pickle(config.PROCESSED_DIR / f"auto_{part}.pkl")


if __name__ == "__main__":
    main()
