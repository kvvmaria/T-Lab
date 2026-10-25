"""Шаг 8. Rolling evaluation: ключевые модели на трёх фолдах (сдвинутых по неделям).

Зачем: на одной тестовой неделе разница между лучшими моделями меньше шума между
неделями. Здесь тот же эксперимент повторяется на трёх тестовых неделях, и в каждом
фолде отбор признаков делается заново — только на его train и valid.

Запуск:  python scripts/08_rolling_evaluation.py        # все фолды
         python scripts/08_rolling_evaluation.py 1      # один фолд
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, features_auto, features_manual, selection
from src.experiment import build_split, fit_and_evaluate, free_memory, load_inputs
from src.metrics import ranking_metrics
from src.reporting import print_result

ROLLING_DIR = config.REPORTS_DIR / "rolling"
PER_USER_DIR = config.PROCESSED_DIR / "rolling"
KEYS = ["user_id", "item_id", "week"]
CANDIDATE_FEATURES = ["src_popular", "src_genre", "src_covisit", "covisit_score"]


def popularity_result(df, column, targets):
    """Сортировка кандидатов по одному признаку популярности — без обучения."""
    result, per_user = {}, None
    for part in ["valid", "test"]:
        scored = df[part][["user_id", "item_id"]].assign(score=df[part][column].to_numpy())
        for key, value in ranking_metrics(scored, targets[part], k=config.K).items():
            result[f"{part}_{key}"] = value
        if part == "test":
            per_user = ranking_metrics(scored, targets[part], k=config.K, per_user=True)
    return result, per_user[f"ndcg@{config.K}"]


def model_result(data, targets, features):
    model, result = fit_and_evaluate(data, targets, features)
    scored = data["test"][["user_id", "item_id"]].assign(score=model.predictions_["test"])
    per_user = ranking_metrics(scored, targets["test"], k=config.K, per_user=True)
    return result, per_user[f"ndcg@{config.K}"]


def run_fold(fold_id):
    fold = config.FOLDS[fold_id]
    print(f"=== Фолд {fold_id}: train {fold['train']}, valid {fold['valid']}, test {fold['test']}")
    events, users, items = load_inputs()
    specs = features_auto.default_specs()
    spec_by_name = {s.name: s for s in specs}

    results, per_user = {}, {}

    def record(name, outcome):
        results[name], per_user[name] = outcome
        print_result(name, results[name])

    # --- ручные признаки
    manual, targets, _ = build_split(
        events, lambda w, c: features_manual.build(w, c, users, items), fold)
    manual_features = features_manual.feature_columns(manual["test"])
    no_candidate = [f for f in manual_features if f not in CANDIDATE_FEATURES]

    record("popularity_7d", popularity_result(manual, "i_cnt_7d", targets))
    record("manual_gbdt", model_result(manual, targets, manual_features))
    record("manual_without_candidate_features", model_result(manual, targets, no_candidate))

    # ручные таблицы пока выгружаем на диск: иначе вместе с 311 автопризнаками не хватает памяти
    manual_path = PER_USER_DIR / "manual_tmp.pkl"
    PER_USER_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(manual, manual_path)
    del manual
    free_memory()

    # --- автоматические признаки
    auto, _, _ = build_split(
        events, lambda w, c: features_auto.build(w, c, users, items, specs), fold)
    auto_features = features_auto.feature_columns(auto["test"])
    record("popularity_3d", popularity_result(auto, "a__item__count__3d__all", targets))
    record("auto_all", model_result(auto, targets, auto_features))

    v4, v4_details = selection.run_v4(auto, targets, auto_features)
    record("auto_v4", model_result(auto, targets, v4))

    family_of = {f: spec_by_name[f].family for f in auto_features}
    v3, _ = selection.run_by_family(auto, targets, auto_features, family_of, metric="ndcg")
    record("auto_v3_family_ndcg", model_result(auto, targets, v3))

    # --- объединения: та же информация, что у ручных, и ручные + авто
    combined = {}
    manual = pd.read_pickle(manual_path)
    for part in ["train", "valid", "test"]:
        combined[part] = auto[part][KEYS + ["label"] + v4].merge(
            manual[part][KEYS + manual_features], on=KEYS, how="inner")
    del auto, manual
    manual_path.unlink()
    free_memory()
    record("auto_v4_plus_candidate_features", model_result(combined, targets, v4 + CANDIDATE_FEATURES))
    record("manual_plus_auto_v4", model_result(combined, targets, manual_features + v4))

    # --- сохранение
    ROLLING_DIR.mkdir(parents=True, exist_ok=True)
    PER_USER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"fold": fold, "results": results,
               "selected": {"auto_v4": v4, "auto_v3_family_ndcg": v3},
               "v4_constant_features": v4_details["constant"]}
    (ROLLING_DIR / f"fold_{fold_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    v4_details["importance"].to_csv(ROLLING_DIR / f"fold_{fold_id}_v4_importance.csv", index=False)
    v4_details["curve"].to_csv(ROLLING_DIR / f"fold_{fold_id}_v4_curve.csv", index=False)
    pd.DataFrame(per_user).to_pickle(PER_USER_DIR / f"per_user_fold_{fold_id}.pkl")


def main():
    folds = [int(sys.argv[1])] if len(sys.argv) > 1 else range(len(config.FOLDS))
    for fold_id in folds:
        run_fold(fold_id)
        free_memory()


if __name__ == "__main__":
    main()
