"""Шаг 3. Бейзлайны: популярность и бустинг на ручных признаках."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, features_manual
from src.experiment import build_split, fit_and_evaluate, load_inputs, save_test_predictions
from src.metrics import ranking_metrics
from src.reporting import print_result, save_result


def main():
    events, users, items = load_inputs()

    def build(week, candidates):
        return features_manual.build(week, candidates, users, items)

    data, targets, build_seconds = build_split(events, build)
    features = features_manual.feature_columns(data["test"])
    print(f"Признаков: {len(features)}, строк в трейне: {len(data['train']):,}, "
          f"построение признаков: {build_seconds:.0f} с")

    # Бейзлайн 0: сортировка кандидатов по популярности за 7 дней.
    # Это не вероятность, поэтому LogLoss/AUC для него не считаем — только ранжирование.
    popularity = {}
    for part in ["valid", "test"]:
        scored = data[part][["user_id", "item_id"]].assign(score=data[part]["i_cnt_7d"])
        for key, value in ranking_metrics(scored, targets[part], k=config.K).items():
            popularity[f"{part}_{key}"] = value
    print_result("popularity", popularity)
    save_result("popularity_7d", popularity)

    model, result = fit_and_evaluate(data, targets, features)
    result["build_seconds"] = round(build_seconds, 1)
    print_result("manual_gbdt", result)
    save_result("manual_gbdt", result)
    save_test_predictions("manual_gbdt", data["test"], model.predictions_["test"])
    save_test_predictions("popularity_7d", data["test"], data["test"]["i_cnt_7d"].to_numpy(np.float32))

    # сохраняем данные бейзлайна — пригодятся для объединения с автопризнаками
    for part, df in data.items():
        df.to_pickle(config.PROCESSED_DIR / f"manual_{part}.pkl")
    pd.Series(features).to_csv(config.REPORTS_DIR / "manual_features.csv", index=False, header=False)


if __name__ == "__main__":
    main()
