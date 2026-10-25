"""Автоматическая генерация признаков по декларативной сетке.

Почти любой "ручной" признак рекомендательной системы — это SQL-запрос вида

    SELECT <ключ>, <агрегат>(<поле>)
    FROM events [JOIN items / users]
    WHERE date >= <начало недели> - <окно> AND date < <начало недели> AND <фильтр события>
    GROUP BY <ключ>

с последующим присоединением к парам (user, item) из кандидатов. Поэтому признак
описывается пятью параметрами (см. FeatureSpec):

  * entity — по какому ключу группируем:
      user                — пользователь;
      item                — фильм;
      user×<атрибут>      — пользователь и значение атрибута фильма (жанр, страна, режиссёр...),
                            "косвенная связь" из условия задания;
      item×<сегмент>      — фильм и демографический сегмент пользователя (возраст, доход...);
  * window — окно в днях (None — вся история);
  * event — фильтр событий: все просмотры или только досмотренные (watched_pct >= 80);
            это аналог типов событий (показ / клик / покупка);
  * agg — что считаем: count, sum_dur, mean_pct, share, days_since_last, ratio;
  * reduce — как свернуть, если у фильма несколько значений атрибута (sum / max / min).

Генератор перебирает декартово произведение этих параметров, а дальше отбор
(src/selection.py) решает, какие окна и связи действительно полезны.
Человек задаёт только словарь сущностей и операций, но не конкретные признаки.
"""

from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

from src.features_manual import encode_users

WINDOWS = [3, 7, 14, 30, 60, None]
EVENTS = ["all", "done"]          # done = досмотрел хотя бы на 80%
DONE_THRESHOLD = 80

ITEM_ATTRIBUTES = ["genres", "countries", "directors", "studios", "content_type", "decade", "age_rating"]
USER_SEGMENTS = ["age", "income", "sex", "kids_flg"]


@dataclass(frozen=True)
class FeatureSpec:
    entity: str                 # "user", "item", "user×genres", "item×age", ...
    agg: str
    window: Optional[int] = None
    event: str = "all"
    reduce: Optional[str] = None

    @property
    def name(self):
        window = "all" if self.window is None else f"{self.window}d"
        entity = self.entity.replace("×", "_x_")
        parts = [entity, self.agg, window, self.event]
        if self.reduce:
            parts.append(self.reduce)
        return "a__" + "__".join(parts)

    @property
    def family(self):
        """Признак без окна — чтобы сравнивать, какое окно лучше для одного и того же смысла."""
        return f"{self.entity}|{self.agg}|{self.event}|{self.reduce}"

    def to_sql(self):
        """Человекочитаемое SQL-описание признака (для отчёта и самопроверки)."""
        where = ["e.date < :week_start"]
        if self.window:
            where.append(f"e.date >= :week_start - INTERVAL '{self.window} days'")
        if self.event == "done":
            where.append(f"e.watched_pct >= {DONE_THRESHOLD}")
        agg_sql = {
            "count": "COUNT(*)", "sum_dur": "SUM(e.total_dur)", "mean_pct": "AVG(e.watched_pct)",
            "days_since_last": "DATE_DIFF(:week_start, MAX(e.date))",
            "share": "COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY <группа>)",
            "ratio": "COUNT(*) FILTER (WHERE e.watched_pct >= 80) / COUNT(*)",
        }[self.agg]
        if self.entity == "user":
            key, join = "e.user_id", ""
        elif self.entity == "item":
            key, join = "e.item_id", ""
        elif self.entity.startswith("user×"):
            attr = self.entity.split("×")[1]
            key, join = f"e.user_id, i.{attr}", "JOIN item_attrs i USING (item_id)"
        else:
            segment = self.entity.split("×")[1]
            key, join = f"u.{segment}, e.item_id", "JOIN users u USING (user_id)"
        sql = f"SELECT {key}, {agg_sql} FROM events e {join} WHERE {' AND '.join(where)} GROUP BY {key}"
        if self.reduce:
            sql += f"  -- затем {self.reduce.upper()} по значениям атрибута фильма-кандидата"
        return sql


def default_specs(windows=WINDOWS):
    specs = []
    for entity in ["user", "item"]:
        for window, event in product(windows, EVENTS):
            specs += [FeatureSpec(entity, agg, window, event) for agg in ["count", "sum_dur", "mean_pct"]]
            if event == "all":
                specs.append(FeatureSpec(entity, "ratio", window, "all"))
        specs += [FeatureSpec(entity, "days_since_last", None, event) for event in EVENTS]

    for attr in ITEM_ATTRIBUTES:
        entity = f"user×{attr}"
        for window, event in product(windows, EVENTS):
            specs.append(FeatureSpec(entity, "count", window, event, reduce="sum"))
            specs.append(FeatureSpec(entity, "share", window, event, reduce="max"))
        specs.append(FeatureSpec(entity, "days_since_last", None, "all", reduce="min"))

    for segment in USER_SEGMENTS:
        entity = f"item×{segment}"
        for window in windows:
            specs.append(FeatureSpec(entity, "share", window, "all"))
            specs.append(FeatureSpec(entity, "mean_pct", window, "all"))
    return specs


# ---------------------------------------------------------------- вычисление

def _windowed(history, start, window, event):
    h = history
    if window is not None:
        h = h[h["date"] >= start - pd.Timedelta(days=window)]
    if event == "done":
        h = h[h["watched_pct"] >= DONE_THRESHOLD]
    return h


def _key_stats(history, start, key, specs):
    """user / item: одна широкая таблица со всеми окнами и агрегатами для ключа."""
    columns = {}
    for spec in specs:
        h = _windowed(history, start, spec.window, spec.event)
        g = h.groupby(key)
        if spec.agg == "count":
            columns[spec.name] = g.size()
        elif spec.agg == "sum_dur":
            columns[spec.name] = g["total_dur"].sum()
        elif spec.agg == "mean_pct":
            columns[spec.name] = g["watched_pct"].mean()
        elif spec.agg == "days_since_last":
            columns[spec.name] = (start - g["date"].max()).dt.days
        elif spec.agg == "ratio":
            columns[spec.name] = (h["watched_pct"] >= DONE_THRESHOLD).groupby(h[key]).mean()
    table = pd.DataFrame(columns)
    fill_zero = [s.name for s in specs if s.agg in ("count", "sum_dur")]
    return table, fill_zero


def _user_attribute_stats(history, candidates, items, start, attr, specs):
    """user×атрибут: считаем статистики по парам (user, значение), затем сворачиваем
    по значениям атрибута у фильма-кандидата."""
    item_attr = items[["item_id", attr]].explode(attr).dropna()
    h = history.merge(item_attr, on="item_id")
    pairs = candidates[["user_id", "item_id"]].merge(item_attr, on="item_id")

    stats = {}
    for spec in specs:
        hw = _windowed(h, start, spec.window, spec.event)
        g = hw.groupby(["user_id", attr])
        if spec.agg == "count":
            stats[spec.name] = g.size()
        elif spec.agg == "share":
            # знаменатель — сколько всего событий у пользователя в том же окне
            user_total = _windowed(history, start, spec.window, spec.event).groupby("user_id").size()
            cnt = g.size()
            stats[spec.name] = cnt / user_total.reindex(cnt.index.get_level_values(0)).to_numpy()
        elif spec.agg == "days_since_last":
            stats[spec.name] = (start - g["date"].max()).dt.days
    stats = pd.DataFrame(stats).reset_index()

    pairs = pairs.merge(stats, on=["user_id", attr], how="left")
    reduce_map = {s.name: s.reduce for s in specs}
    reduced = pairs.groupby(["user_id", "item_id"]).agg(reduce_map)
    fill_zero = [s.name for s in specs if s.agg in ("count", "share")]
    return reduced, fill_zero


def _item_segment_stats(history, candidates, users_enc, start, segment, specs):
    """item×сегмент: популярность фильма среди похожих по демографии пользователей."""
    h = history.merge(users_enc[["user_id", segment]], on="user_id", how="left").fillna({segment: -1})
    stats = {}
    for spec in specs:
        hw = _windowed(h, start, spec.window, spec.event)
        g = hw.groupby([segment, "item_id"])
        if spec.agg == "share":
            seg_total = hw.groupby(segment).size()
            cnt = g.size()
            stats[spec.name] = cnt / seg_total.reindex(cnt.index.get_level_values(0)).to_numpy()
        elif spec.agg == "mean_pct":
            stats[spec.name] = g["watched_pct"].mean()
    stats = pd.DataFrame(stats).reset_index()

    pairs = (candidates[["user_id", "item_id"]]
             .merge(users_enc[["user_id", segment]], on="user_id", how="left")
             .fillna({segment: -1})
             .merge(stats, on=[segment, "item_id"], how="left"))
    fill_zero = [s.name for s in specs if s.agg == "share"]
    return pairs.drop(columns=segment).set_index(["user_id", "item_id"]), fill_zero


def build(week, candidates, users, items, specs):
    """Считает все признаки из specs для кандидатов одной недели."""
    history, start = week.history, week.start
    users_enc = encode_users(users)
    items = items.assign(content_type=items["content_type"].map(lambda v: [v]),
                         decade=items["decade"].map(lambda v: [v]),
                         age_rating=items["age_rating"].map(lambda v: [v]))

    cand_users = candidates["user_id"].unique()
    user_history = history[history["user_id"].isin(cand_users)]

    by_entity = {}
    for spec in specs:
        by_entity.setdefault(spec.entity, []).append(spec)

    df = candidates[["user_id", "item_id", "label", "week"]].copy()
    fill_zero = []
    for entity, entity_specs in by_entity.items():
        if entity == "user":
            table, zeros = _key_stats(user_history, start, "user_id", entity_specs)
            df = df.merge(table, left_on="user_id", right_index=True, how="left")
        elif entity == "item":
            table, zeros = _key_stats(history, start, "item_id", entity_specs)
            df = df.merge(table, left_on="item_id", right_index=True, how="left")
        elif entity.startswith("user×"):
            attr = entity.split("×")[1]
            table, zeros = _user_attribute_stats(user_history, candidates, items, start, attr, entity_specs)
            df = df.merge(table, left_on=["user_id", "item_id"], right_index=True, how="left")
        else:
            segment = entity.split("×")[1]
            table, zeros = _item_segment_stats(history, candidates, users_enc, start, segment, entity_specs)
            df = df.merge(table, left_on=["user_id", "item_id"], right_index=True, how="left")
        fill_zero += zeros

    df[fill_zero] = df[fill_zero].fillna(0)
    names = [s.name for s in specs]
    df[names] = df[names].astype(np.float32)
    return df


def feature_columns(df):
    return [c for c in df.columns if c.startswith("a__")]
