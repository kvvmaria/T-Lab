"""Отбор автоматически сгенерированных признаков.

Три ступени, от дешёвой к дорогой:

  1. screening   — одномерная оценка каждого признака на ТРЕЙНЕ: grouped AUC,
                   то есть насколько хорошо признак сам по себе ранжирует кандидатов
                   внутри одного пользователя. Ничего не обучаем, это быстро.
  2. redundancy  — из почти одинаковых признаков (|корреляция Спирмена| > порога)
                   оставляем тот, у которого лучше grouped AUC. Например, count за
                   30 и 60 дней у редко заходящих пользователей почти совпадают.
  3. model-based — обучаем бустинг на оставшихся признаках, считаем permutation
                   importance на ВАЛИДАЦИИ и выбираем, сколько топ-признаков оставить,
                   по NDCG@10 на валидации.

Тестовая неделя в отборе не участвует вообще.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src import config, model as gbdt
from src.experiment import fit_and_evaluate
from src.metrics import ranking_metrics


# ------------------------------------------------------------------ 1. screening

def grouped_auc(df, feature, group_cols=("week", "user_id")):
    """AUC признака внутри каждого пользователя, усреднённый по пользователям.

    Возвращает число в [0, 1]; 0.5 — признак не ранжирует, далеко от 0.5 в любую
    сторону — ранжирует (знак для бустинга неважен).
    """
    values = df[feature].fillna(-1e9)
    ranks = values.groupby([df[c] for c in group_cols]).rank(method="average")
    stats = pd.DataFrame({"rank_pos": ranks * df["label"], "pos": df["label"], "n": 1})
    stats = stats.groupby([df[c] for c in group_cols]).sum()
    stats["neg"] = stats["n"] - stats["pos"]
    stats = stats[(stats["pos"] > 0) & (stats["neg"] > 0)]
    auc = (stats["rank_pos"] - stats["pos"] * (stats["pos"] + 1) / 2) / (stats["pos"] * stats["neg"])
    return float(auc.mean())


def screen(train, features):
    rows = []
    for feature in features:
        auc = grouped_auc(train, feature)
        rows.append({"feature": feature, "gauc": auc, "gauc_strength": abs(auc - 0.5)})
    return pd.DataFrame(rows).sort_values("gauc_strength", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ 2. redundancy

def drop_redundant(train, screening, threshold=0.97, sample_size=60_000, min_strength=0.005):
    """Жадно идём по признакам от сильных к слабым и не берём те, что почти
    повторяют уже взятые. Совсем бесполезные (|gAUC − 0.5| < min_strength) отбрасываем."""
    candidates = screening[screening["gauc_strength"] >= min_strength]["feature"].to_list()
    sample = train.sample(min(sample_size, len(train)), random_state=config.SEED)
    ranks = sample[candidates].rank().fillna(-1).to_numpy(np.float64)
    corr = np.nan_to_num(np.corrcoef(ranks, rowvar=False))

    kept, dropped_by = [], {}
    for i, feature in enumerate(candidates):
        twin = next((candidates[j] for j in kept if abs(corr[i, j]) > threshold), None)
        if twin is None:
            kept.append(i)
        else:
            dropped_by[feature] = twin
    return [candidates[i] for i in kept], dropped_by


# ------------------------------------------------------------------ 3. model-based

def permutation_importance(model, valid, features, n_users=3_000, seed=config.SEED):
    """Насколько растёт LogLoss на валидации, если перемешать признак."""
    rng = np.random.default_rng(seed)
    users = rng.choice(valid["user_id"].unique(), min(n_users, valid["user_id"].nunique()), replace=False)
    part = valid[valid["user_id"].isin(users)]
    X = gbdt.to_matrix(part, features)
    y = part["label"].to_numpy()
    base = log_loss(y, model.predict_proba(X)[:, 1])

    rows = []
    for j, feature in enumerate(features):
        shuffled = X.copy()
        shuffled[:, j] = rng.permutation(shuffled[:, j])
        rows.append({"feature": feature,
                     "importance": log_loss(y, model.predict_proba(shuffled)[:, 1]) - base})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def choose_top_n(data, targets, ranked_features, cutoffs, metric="valid_ndcg@10"):
    """Обучаем модель на топ-N признаках для разных N и выбираем N по валидации."""
    rows = []
    for n in cutoffs:
        features = ranked_features[:n]
        _, result = fit_and_evaluate(data, targets, features, use_test=False)
        rows.append({"n_features": len(features), metric: result[metric],
                     "valid_logloss": result["valid_logloss"], "fit_seconds": result["fit_seconds"]})
        print(f"  top-{len(features):3d}: {metric}={result[metric]:.4f}")
    curve = pd.DataFrame(rows)
    best_n = int(curve.loc[curve[metric].idxmax(), "n_features"])
    return ranked_features[:best_n], curve


# ------------------------------------------------------------------ отбор семействами

def group_permutation_importance(model, valid, features, groups, metric="logloss", target=None,
                                 n_users=4_000, seed=config.SEED):
    """Как permutation importance, но перемешиваем сразу всю группу признаков.

    Зачем: окна 7/14/30 дней одного и того же агрегата сильно коррелируют. Если
    перемешать только одно окно, модель возьмёт информацию из соседнего, и признак
    покажется бесполезным. Перемешивание целой группы этой проблемы не имеет.
    Строки перемешиваются одной перестановкой на группу, чтобы связи внутри группы сохранились.

    metric="logloss" — рост LogLoss; metric="ndcg" — падение NDCG@10 (нужен target валидации).
    """
    rng = np.random.default_rng(seed)
    users = rng.choice(valid["user_id"].unique(), min(n_users, valid["user_id"].nunique()), replace=False)
    part = valid[valid["user_id"].isin(users)]
    X = gbdt.to_matrix(part, features)
    y = part["label"].to_numpy()

    if metric == "ndcg":
        part_target = target[target["user_id"].isin(users)]

        def loss(matrix):
            scored = part[["user_id", "item_id"]].assign(score=model.predict_proba(matrix)[:, 1])
            return -ranking_metrics(scored, part_target, k=config.K)[f"ndcg@{config.K}"]
    else:
        def loss(matrix):
            return log_loss(y, model.predict_proba(matrix)[:, 1])

    base = loss(X)
    position = {f: j for j, f in enumerate(features)}
    rows = []
    for group, members in groups.items():
        cols = [position[f] for f in members]
        shuffled = X.copy()
        shuffled[:, cols] = shuffled[rng.permutation(len(X))][:, cols]
        rows.append({"family": group, "n_features": len(members), "importance": loss(shuffled) - base})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def run_by_family(data, targets, features, family_of, metric="logloss",
                  cutoffs=(3, 5, 8, 12, 16, 20, 30, 40)):
    """Отбор семействами: семейство = сущность + агрегат + фильтр событий, все окна сразу.

    1. Модель на всех признаках -> групповая permutation importance на валидации.
    2. Модели на топ-K семействах для разных K -> выбираем K по NDCG@10 на валидации.
    """
    groups = {}
    for feature in features:
        groups.setdefault(family_of[feature], []).append(feature)

    model = gbdt.train(data["train"], data["valid"], features)
    importance = group_permutation_importance(model, data["valid"], features, groups,
                                              metric=metric, target=targets["valid"])
    ranked = importance[importance["importance"] > 0]["family"].to_list()
    print(f"Семейств: {len(groups)}, с положительной важностью: {len(ranked)}")

    rows = []
    for k in sorted({k for k in cutoffs if k < len(ranked)} | {len(ranked)}):
        chosen = [f for family in ranked[:k] for f in groups[family]]
        _, result = fit_and_evaluate(data, targets, chosen, use_test=False)
        rows.append({"n_families": k, "n_features": len(chosen),
                     "valid_ndcg@10": result["valid_ndcg@10"], "valid_logloss": result["valid_logloss"]})
        print(f"  top-{k:2d} семейств ({len(chosen):3d} признаков): valid_ndcg@10={result['valid_ndcg@10']:.4f}")
    curve = pd.DataFrame(rows)
    best_k = int(curve.loc[curve["valid_ndcg@10"].idxmax(), "n_families"])
    selected = [f for family in ranked[:best_k] for f in groups[family]]
    print(f"Выбрано семейств: {best_k}, признаков: {len(selected)}")
    return selected, {"family_importance": importance, "family_curve": curve}


def run(data, targets, features, cutoffs=(5, 10, 15, 20, 30, 40, 60, 80, 120)):
    """Полный отбор. Возвращает отобранные признаки и все промежуточные таблицы."""
    screening = screen(data["train"], features)
    print(f"Скрининг: {len(features)} признаков, у {int((screening.gauc_strength >= 0.005).sum())} "
          f"|gAUC − 0.5| ≥ 0.005")

    kept, dropped_by = drop_redundant(data["train"], screening)
    print(f"После удаления дублей: {len(kept)} признаков")

    model = gbdt.train(data["train"], data["valid"], kept)
    importance = permutation_importance(model, data["valid"], kept)
    useful = importance[importance["importance"] > 0]["feature"].to_list()
    print(f"Положительная permutation importance у {len(useful)} признаков")

    cutoffs = sorted({n for n in cutoffs if n < len(useful)} | {len(useful)})
    selected, curve = choose_top_n(data, targets, useful, cutoffs)
    print(f"Выбрано признаков: {len(selected)}")
    return selected, {"screening": screening, "dropped_by": dropped_by,
                      "importance": importance, "curve": curve}
