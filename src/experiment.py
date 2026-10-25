"""Общий каркас эксперимента: одинаковые недели, кандидаты и оценка для всех подходов."""

import time

import numpy as np
import pandas as pd

from src import config, model as gbdt
from src.metrics import classification_metrics, ranking_metrics
from src.split import make_week


def load_inputs():
    events = pd.read_pickle(config.PROCESSED_DIR / "events.pkl")
    users = pd.read_pickle(config.PROCESSED_DIR / "users.pkl")
    items = pd.read_pickle(config.PROCESSED_DIR / "items.pkl")
    return events, users, items


def load_week(events, week_start):
    week = make_week(events, week_start, name=week_start)
    candidates = pd.read_pickle(config.PROCESSED_DIR / f"candidates_{week_start}.pkl")
    return week, candidates


def subsample_train(df, neg_fraction=config.TRAIN_NEG_FRACTION, seed=config.SEED):
    """Оставляем пользователей, у которых есть хотя бы один позитив среди кандидатов,
    и долю их негативов. Для валидации и теста ничего не выкидываем."""
    has_positive = df.groupby(["week", "user_id"])["label"].transform("max") == 1
    df = df[has_positive]
    rng = np.random.default_rng(seed)
    keep = (df["label"] == 1) | (rng.random(len(df)) < neg_fraction)
    return df[keep].reset_index(drop=True)


def build_split(events, build_features):
    """Строит признаки для всех недель. build_features(week, candidates) -> DataFrame."""
    started = time.time()
    frames, weeks = {}, {}
    for week_start in config.TRAIN_WEEKS + [config.VALID_WEEK, config.TEST_WEEK]:
        week, candidates = load_week(events, week_start)
        weeks[week_start] = week
        frame = build_features(week, candidates)
        # трейн прореживаем сразу, чтобы не держать в памяти полные недели
        frames[week_start] = subsample_train(frame) if week_start in config.TRAIN_WEEKS else frame
    train = pd.concat([frames[w] for w in config.TRAIN_WEEKS], ignore_index=True)
    data = {"train": train, "valid": frames[config.VALID_WEEK], "test": frames[config.TEST_WEEK]}
    targets = {"valid": weeks[config.VALID_WEEK].target, "test": weeks[config.TEST_WEEK].target}
    return data, targets, time.time() - started


def evaluate(df, scores, target, k=config.K):
    scored = df[["user_id", "item_id"]].assign(score=scores)
    result = ranking_metrics(scored, target, k=k)
    result.update(classification_metrics(df["label"].to_numpy(), scores))
    return result


def fit_and_evaluate(data, targets, features, params=None, use_test=True):
    """Обучение + оценка. Во время отбора признаков use_test=False: тест не смотрим вообще."""
    model = gbdt.train(data["train"], data["valid"], features, params)
    result = {"n_features": len(features), "fit_seconds": round(model.fit_seconds_, 1),
              "n_iter": int(model.n_iter_)}
    for part in (["valid", "test"] if use_test else ["valid"]):
        scores = gbdt.predict(model, data[part], features)
        for key, value in evaluate(data[part], scores, targets[part]).items():
            result[f"{part}_{key}"] = value
    return model, result
