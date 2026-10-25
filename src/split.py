"""Разбиение по времени.

Схема (каждая "неделя" — 7 дней):

    история (всё, что раньше начала недели) | целевая неделя
    ----------------------------------------|---------------
    признаки и кандидаты считаются здесь     | отсюда берём только таргет

Train — несколько недель подряд, затем valid, затем test. Для каждой недели история
своя: для test она включает и валидационную неделю — в момент предсказания
эти данные уже были бы доступны, так что утечки здесь нет.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class WeekSplit:
    name: str
    start: pd.Timestamp
    history: pd.DataFrame   # события строго до start
    target: pd.DataFrame    # события в [start, start + 7 дней)


def make_week(events, week_start, name):
    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=7)

    history = events[events["date"] < start]
    target = events[(events["date"] >= start) & (events["date"] < end)]

    # Оцениваем только "тёплых" пользователей: у кого есть история.
    # Холодный старт — отдельная задача, для него ранжирование по признакам истории бессмысленно.
    warm_users = history["user_id"].unique()
    target = target[target["user_id"].isin(warm_users)]

    return WeekSplit(name=name, start=start,
                     history=history.reset_index(drop=True),
                     target=target.reset_index(drop=True))
