"""Проверка на утечки: признаки и кандидаты не должны зависеть от будущего.

Идея теста: берём неделю, затем "портим" все события с датой >= начала недели
(меняем фильмы, проценты просмотра, добавляем мусор) и строим всё заново.
Если признаки хоть где-то подглядывают в будущее, они изменятся. Меняться имеет право
только таргет (label).

Запуск: python tests/test_no_leakage.py  (или pytest tests/)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, features_auto, features_manual
from src.candidates import generate
from src.split import make_week

WEEK = config.VALID_WEEK
N_TEST_USERS = 5_000


def _small_sample():
    events = pd.read_pickle(config.PROCESSED_DIR / "events.pkl")
    users = pd.read_pickle(config.PROCESSED_DIR / "users.pkl")
    items = pd.read_pickle(config.PROCESSED_DIR / "items.pkl")
    rng = np.random.default_rng(0)
    chosen = rng.choice(events["user_id"].unique(), N_TEST_USERS, replace=False)
    return events[events["user_id"].isin(chosen)].reset_index(drop=True), users, items


def _corrupt_future(events, start):
    """Портим всё, что происходит начиная с start."""
    rng = np.random.default_rng(1)
    events = events.copy()
    future = events["date"] >= pd.Timestamp(start)
    events.loc[future, "watched_pct"] = rng.uniform(0, 100, future.sum()).astype("float32")
    events.loc[future, "total_dur"] = rng.uniform(0, 1e5, future.sum()).astype("float32")
    events.loc[future, "item_id"] = rng.permutation(events.loc[future, "item_id"].to_numpy())
    # плюс добавляем лишние события в будущем
    extra = events[future].sample(frac=0.5, random_state=2).assign(
        item_id=lambda d: rng.permutation(d["item_id"].to_numpy()))
    return pd.concat([events, extra], ignore_index=True)


def _build_all(events, users, items):
    week = make_week(events, WEEK, name=WEEK)
    candidates = generate(week, items)
    manual = features_manual.build(week, candidates, users, items)
    auto = features_auto.build(week, candidates, users, items, features_auto.default_specs())
    return manual, auto


def _compare(clean, corrupted):
    # Множество пользователей зависит от того, кто активен в целевой неделе (это нормально:
    # мы оцениваем тех, кто пришёл). Поэтому сравниваем признаки на общих парах.
    keys = ["user_id", "item_id"]
    merged = clean.merge(corrupted, on=keys, suffixes=("", "_corrupted"))
    assert len(merged) > 0
    features = [c for c in clean.columns if c not in keys + ["label", "week"]]
    for col in features:
        a, b = merged[col].to_numpy(np.float64), merged[f"{col}_corrupted"].to_numpy(np.float64)
        assert np.allclose(a, b, equal_nan=True), f"признак {col} зависит от будущего!"
    return len(features)


def test_features_do_not_see_future():
    events, users, items = _small_sample()
    manual, auto = _build_all(events, users, items)
    manual_bad, auto_bad = _build_all(_corrupt_future(events, WEEK), users, items)
    n_manual = _compare(manual, manual_bad)
    n_auto = _compare(auto, auto_bad)
    print(f"OK: {n_manual} ручных и {n_auto} автопризнаков не зависят от событий после {WEEK}")


def test_candidates_exclude_seen_items():
    events, _, items = _small_sample()
    week = make_week(events, WEEK, name=WEEK)
    candidates = generate(week, items)
    seen = week.history[["user_id", "item_id"]]
    overlap = candidates.merge(seen, on=["user_id", "item_id"])
    assert overlap.empty, "в кандидатах есть уже просмотренные фильмы"
    print("OK: в кандидатах нет уже просмотренного")


if __name__ == "__main__":
    test_candidates_exclude_seen_items()
    test_features_do_not_see_future()
