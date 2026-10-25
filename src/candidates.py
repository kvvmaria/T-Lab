"""Генерация кандидатов для ранжирования.

Ранжировать все ~10 тысяч фильмов для каждого пользователя дорого, поэтому сначала
отбираем около сотни кандидатов тремя простыми источниками:

  1. popular  — самые популярные фильмы за последние 14 дней;
  2. genre    — популярное в двух любимых жанрах пользователя;
  3. covisit  — фильмы, которые часто смотрят вместе с последними просмотрами пользователя.

Все источники считаются только по истории (до начала целевой недели).
Уже просмотренные пользователем фильмы из кандидатов убираем: в KION одна строка
на пару user–item, поэтому повторный просмотр не может стать новым позитивом.

Набор кандидатов один и тот же для всех сравниваемых моделей.
"""

import pandas as pd

from src import config


def popular_items(history, start, days, top_n):
    recent = history[history["date"] >= start - pd.Timedelta(days=days)]
    return recent["item_id"].value_counts().head(top_n).index.to_list()


def _popular_candidates(users, history, start):
    top = popular_items(history, start, days=14, top_n=config.N_POPULAR)
    return pd.MultiIndex.from_product([users, top], names=["user_id", "item_id"]).to_frame(index=False)


def _genre_candidates(history, items, start):
    """Для каждого пользователя — топ фильмов в двух его самых частых жанрах."""
    item_genres = items[["item_id", "genres"]].explode("genres").dropna()

    user_genres = (history[["user_id", "item_id"]]
                   .merge(item_genres, on="item_id")
                   .groupby(["user_id", "genres"]).size()
                   .rename("cnt").reset_index()
                   .sort_values(["user_id", "cnt"], ascending=[True, False])
                   .groupby("user_id").head(2))

    recent = history[history["date"] >= start - pd.Timedelta(days=30)]
    genre_top = (recent[["item_id"]]
                 .merge(item_genres, on="item_id")
                 .groupby(["genres", "item_id"]).size()
                 .rename("cnt").reset_index()
                 .sort_values(["genres", "cnt"], ascending=[True, False])
                 .groupby("genres").head(config.N_GENRE_POPULAR))

    return user_genres[["user_id", "genres"]].merge(genre_top[["genres", "item_id"]], on="genres")[
        ["user_id", "item_id"]]


def covisitation_table(history, last_n=30):
    """Сколько пользователей посмотрели оба фильма пары. Длинные истории обрезаем до last_n."""
    recent = (history.sort_values("date")
              .groupby("user_id").tail(last_n)[["user_id", "item_id"]])
    pairs = recent.merge(recent, on="user_id", suffixes=("", "_other"))
    pairs = pairs[pairs["item_id"] != pairs["item_id_other"]]
    return pairs.groupby(["item_id", "item_id_other"]).size().rename("covisit").reset_index()


def _covisit_candidates(history, covisit):
    """Соседи трёх последних просмотров пользователя, взвешенные по covisit."""
    last_items = (history.sort_values("date")
                  .groupby("user_id").tail(3)[["user_id", "item_id"]])
    neighbours = last_items.merge(covisit, on="item_id")
    scores = (neighbours.groupby(["user_id", "item_id_other"])["covisit"].sum()
              .rename("score").reset_index()
              .sort_values(["user_id", "score"], ascending=[True, False])
              .groupby("user_id").head(config.N_COVISIT))
    return scores.rename(columns={"item_id_other": "item_id", "score": "covisit_score"})


def generate(week, items):
    """Кандидаты и разметка для одной целевой недели."""
    history, start = week.history, week.start
    users = week.target["user_id"].unique()
    user_history = history[history["user_id"].isin(users)]

    covisit = covisitation_table(history)
    sources = {
        "popular": _popular_candidates(users, history, start),
        "genre": _genre_candidates(user_history, items, start),
        "covisit": _covisit_candidates(user_history, covisit),
    }

    frames = []
    for name, frame in sources.items():
        frame = frame.drop_duplicates(["user_id", "item_id"]).copy()
        frame[f"src_{name}"] = 1
        frames.append(frame.set_index(["user_id", "item_id"]))
    # covisit_score остаётся как признак: это "ручная" связь item–item через пользователей
    candidates = pd.concat(frames, axis=1).fillna(0).reset_index()
    for name in sources:
        candidates[f"src_{name}"] = candidates[f"src_{name}"].astype("int8")
    candidates["covisit_score"] = candidates["covisit_score"].astype("float32")

    # убираем уже просмотренное
    seen = user_history[["user_id", "item_id"]].assign(seen=1)
    candidates = candidates.merge(seen, on=["user_id", "item_id"], how="left")
    candidates = candidates[candidates["seen"].isna()].drop(columns="seen")

    # drop_duplicates: в KION пары уникальны, но без этого дубли в таргете размножили бы кандидатов
    positives = week.target[["user_id", "item_id"]].drop_duplicates().assign(label=1)
    candidates = candidates.merge(positives, on=["user_id", "item_id"], how="left")
    candidates["label"] = candidates["label"].fillna(0).astype("int8")
    candidates["week"] = week.name
    return candidates.reset_index(drop=True)
