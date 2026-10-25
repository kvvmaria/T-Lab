"""Загрузка KION, сэмплирование пользователей и подготовка таблиц.

На выходе три таблицы:
  * events — взаимодействия (user_id, item_id, date, total_dur, watched_pct);
  * users  — демография пользователей;
  * items  — метаданные фильмов, многозначные поля (жанры, страны, режиссёры) — списки.
"""

import numpy as np
import pandas as pd

from src import config


def _split_list(value):
    """'драмы, комедии' -> ['драмы', 'комедии']; NaN -> []."""
    if pd.isna(value):
        return []
    return [part.strip().lower() for part in str(value).split(",") if part.strip()]


def load_events(path=None):
    path = path or config.RAW_DIR / "interactions.csv"
    events = pd.read_csv(path, parse_dates=["last_watch_dt"])
    events = events.rename(columns={"last_watch_dt": "date"})
    # в паре строк watched_pct пропущен — считаем, что почти не смотрели
    events["watched_pct"] = events["watched_pct"].fillna(0).astype("float32")
    events["total_dur"] = events["total_dur"].astype("float32")
    return events


def load_users(path=None):
    path = path or config.RAW_DIR / "users.csv"
    users = pd.read_csv(path)
    for col in ["age", "income", "sex"]:
        users[col] = users[col].fillna("unknown")
    users["kids_flg"] = users["kids_flg"].fillna(0).astype("int8")
    return users


def load_items(path=None):
    path = path or config.RAW_DIR / "items.csv"
    items = pd.read_csv(path)
    items = items[["item_id", "content_type", "release_year", "genres", "countries",
                   "for_kids", "age_rating", "studios", "directors"]].copy()

    for col in ["genres", "countries", "studios", "directors"]:
        items[col] = items[col].map(_split_list)

    items["release_year"] = items["release_year"].fillna(items["release_year"].median())
    # десятилетие выпуска — грубая категория, по ней тоже можно строить связи
    items["decade"] = (items["release_year"] // 10 * 10).astype(int).astype(str)
    items["age_rating"] = items["age_rating"].fillna(-1).astype(int).astype(str)
    items["for_kids"] = items["for_kids"].fillna(0).astype("int8")
    return items


def sample_users(events, n_users, seed=config.SEED):
    """Берём n_users случайных пользователей и ВСЮ их историю.

    Сэмплируем именно пользователей, а не строки: иначе мы бы порвали истории
    и признаки вида "что пользователь смотрел раньше" стали бы бессмысленными.
    """
    rng = np.random.default_rng(seed)
    all_users = events["user_id"].unique()
    chosen = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)
    return events[events["user_id"].isin(chosen)].reset_index(drop=True)


def prepare(n_users=config.N_USERS):
    events = sample_users(load_events(), n_users)
    users = load_users()
    users = users[users["user_id"].isin(events["user_id"].unique())].reset_index(drop=True)
    items = load_items()
    return events, users, items
