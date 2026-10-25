"""Шаг 1. Сэмплируем пользователей, сохраняем таблицы и рисуем пару графиков для EDA."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data import prepare


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    events, users, items = prepare()
    events.to_pickle(config.PROCESSED_DIR / "events.pkl")
    users.to_pickle(config.PROCESSED_DIR / "users.pkl")
    items.to_pickle(config.PROCESSED_DIR / "items.pkl")

    print(f"Пользователей: {events.user_id.nunique():,}")
    print(f"Взаимодействий: {len(events):,}")
    print(f"Фильмов и сериалов: {events.item_id.nunique():,}")
    print(f"Период: {events.date.min().date()} — {events.date.max().date()}")
    print(f"Пользователей без демографии: {1 - len(users) / events.user_id.nunique():.1%}")

    history_len = events.groupby("user_id").size()
    print("Длина истории пользователя:")
    print(history_len.describe(percentiles=[0.5, 0.75, 0.9, 0.99]).round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    weekly = events.set_index("date").resample("W-SUN")["user_id"].agg(["size", "nunique"])
    axes[0].plot(weekly.index, weekly["size"], label="взаимодействия")
    axes[0].plot(weekly.index, weekly["nunique"], label="активные пользователи")
    for week in config.TRAIN_WEEKS + [config.VALID_WEEK, config.TEST_WEEK]:
        axes[0].axvline(pd.Timestamp(week), color="grey", linestyle=":", linewidth=1)
    axes[0].set_title("Активность по неделям (пунктир — начала целевых недель)")
    axes[0].legend()

    axes[1].hist(history_len.clip(upper=50), bins=50)
    axes[1].set_yscale("log")
    axes[1].set_title("Длина истории пользователя (обрезано на 50)")

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda.png", dpi=120)


if __name__ == "__main__":
    main()
