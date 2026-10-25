"""Метрики качества.

Ранжирующие метрики считаются по ВСЕМ тёплым пользователям целевой недели и по ВСЕМ их
реальным просмотрам, а не только по тем, что попали в кандидаты. Поэтому полнота
кандидатов ограничивает метрики сверху, и это честно: модель не может порекомендовать
то, чего нет в кандидатах.

Классификационные метрики (LogLoss, ROC-AUC) считаются по строкам кандидатов.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def top_k(scored, k):
    """scored: user_id, item_id, score -> top-k строк на пользователя с рангом 1..k."""
    ranked = scored.sort_values(["user_id", "score"], ascending=[True, False])
    ranked = ranked.groupby("user_id").head(k).copy()
    ranked["rank"] = ranked.groupby("user_id").cumcount() + 1
    return ranked


def ranking_metrics(scored, target, k=10):
    relevant = target.groupby("user_id")["item_id"].apply(set)
    n_relevant = relevant.map(len)

    recs = top_k(scored, k).merge(target[["user_id", "item_id"]].assign(hit=1),
                                  on=["user_id", "item_id"], how="left")
    recs["hit"] = recs["hit"].fillna(0)
    recs["gain"] = recs["hit"] / np.log2(recs["rank"] + 1)

    # для MAP: precision@i в позициях попаданий
    recs["cum_hits"] = recs.groupby("user_id")["hit"].cumsum()
    recs["prec_at_hit"] = recs["hit"] * recs["cum_hits"] / recs["rank"]

    per_user = pd.DataFrame(index=relevant.index)
    per_user["hits"] = recs.groupby("user_id")["hit"].sum()
    per_user["dcg"] = recs.groupby("user_id")["gain"].sum()
    per_user["ap_sum"] = recs.groupby("user_id")["prec_at_hit"].sum()
    per_user = per_user.fillna(0)

    ideal_hits = np.minimum(n_relevant, k)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    idcg = ideal_hits.map(lambda n: discounts[:n].sum())

    return {
        f"recall@{k}": float((per_user["hits"] / n_relevant).mean()),
        f"ndcg@{k}": float((per_user["dcg"] / idcg).mean()),
        f"map@{k}": float((per_user["ap_sum"] / ideal_hits).mean()),
    }


def classification_metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return {"logloss": float(log_loss(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_pred))}


def candidate_recall(candidates, target):
    """Какая доля реальных просмотров вообще попала в кандидаты (верхняя граница)."""
    return float(candidates["label"].sum() / len(target))
