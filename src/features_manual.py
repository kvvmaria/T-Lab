"""Ручные признаки — сильный бейзлайн.

Это то, что инженер написал бы руками, исходя из здравого смысла:
активность пользователя, популярность и свежесть фильма, совпадение жанров,
стран и режиссёров с историей пользователя, популярность фильма в возрастной группе.

Все признаки считаются по week.history — событиям строго до начала целевой недели.
"""

import numpy as np
import pandas as pd

AGE_ORDER = ["age_18_24", "age_25_34", "age_35_44", "age_45_54", "age_55_64", "age_65_inf"]
INCOME_ORDER = ["income_0_20", "income_20_40", "income_40_60", "income_60_90",
                "income_90_150", "income_150_inf"]


def encode_users(users):
    """Демография в числа: порядковые категории, неизвестное -> -1."""
    out = users[["user_id"]].copy()
    out["age"] = users["age"].map({v: i for i, v in enumerate(AGE_ORDER)}).fillna(-1)
    out["income"] = users["income"].map({v: i for i, v in enumerate(INCOME_ORDER)}).fillna(-1)
    out["sex"] = users["sex"].map({"М": 0, "Ж": 1}).fillna(-1)
    out["kids_flg"] = users["kids_flg"]
    return out


def user_features(history, start, items):
    h = history.merge(items[["item_id", "content_type", "release_year"]], on="item_id", how="left")
    last_7 = h[h["date"] >= start - pd.Timedelta(days=7)]
    last_30 = h[h["date"] >= start - pd.Timedelta(days=30)]

    g = h.groupby("user_id")
    feats = pd.DataFrame({
        "u_cnt_all": g.size(),
        "u_mean_pct": g["watched_pct"].mean(),
        "u_share_series": g["content_type"].apply(lambda s: (s == "series").mean()),
        "u_mean_release_year": g["release_year"].mean(),
        "u_days_since_last": (start - g["date"].max()).dt.days,
        "u_days_since_first": (start - g["date"].min()).dt.days,
    })
    feats["u_cnt_7d"] = last_7.groupby("user_id").size()
    feats["u_cnt_30d"] = last_30.groupby("user_id").size()
    feats["u_dur_30d"] = last_30.groupby("user_id")["total_dur"].sum()
    feats[["u_cnt_7d", "u_cnt_30d", "u_dur_30d"]] = feats[["u_cnt_7d", "u_cnt_30d", "u_dur_30d"]].fillna(0)
    return feats.reset_index()


def item_features(history, start, items):
    last_7 = history[history["date"] >= start - pd.Timedelta(days=7)]
    last_30 = history[history["date"] >= start - pd.Timedelta(days=30)]

    g = history.groupby("item_id")
    feats = pd.DataFrame({
        "i_cnt_all": g.size(),
        "i_days_since_first": (start - g["date"].min()).dt.days,
        "i_days_since_last": (start - g["date"].max()).dt.days,
    })
    feats["i_cnt_7d"] = last_7.groupby("item_id").size()
    feats["i_cnt_30d"] = last_30.groupby("item_id").size()
    feats["i_mean_pct_30d"] = last_30.groupby("item_id")["watched_pct"].mean()
    feats[["i_cnt_7d", "i_cnt_30d"]] = feats[["i_cnt_7d", "i_cnt_30d"]].fillna(0)
    feats["i_trend_7_30"] = feats["i_cnt_7d"] / (feats["i_cnt_30d"] + 1)
    feats = feats.reset_index()

    static = items[["item_id", "release_year", "age_rating", "for_kids"]].copy()
    static["is_series"] = (items["content_type"] == "series").astype("int8")
    static["age_rating"] = static["age_rating"].astype(int)
    static = static.rename(columns={"release_year": "i_release_year", "age_rating": "i_age_rating",
                                    "for_kids": "i_for_kids", "is_series": "i_is_series"})
    return static.merge(feats, on="item_id", how="left")


def user_attribute_share(candidates, history, items, attr, name):
    """Косвенная связь: доля просмотров пользователя, пришедшаяся на значения атрибута фильма.

    Например, для жанров: у фильма жанры {драма, комедия}, пользователь 40% времени
    смотрел драмы и 10% — комедии. Признаки: среднее (25%) и максимум (40%) по жанрам фильма.
    """
    item_attr = items[["item_id", attr]].explode(attr).dropna()
    user_attr = (history[["user_id", "item_id"]].merge(item_attr, on="item_id")
                 .groupby(["user_id", attr]).size().rename("cnt").reset_index())
    user_total = history.groupby("user_id").size().rename("total")
    user_attr = user_attr.merge(user_total, on="user_id")
    user_attr["share"] = user_attr["cnt"] / user_attr["total"]

    pairs = (candidates[["user_id", "item_id"]].merge(item_attr, on="item_id")
             .merge(user_attr[["user_id", attr, "share"]], on=["user_id", attr], how="left")
             .fillna({"share": 0}))
    agg = pairs.groupby(["user_id", "item_id"])["share"].agg(["mean", "max"])
    agg.columns = [f"ui_{name}_share_mean", f"ui_{name}_share_max"]
    return agg.reset_index()


def segment_popularity(candidates, history, users_enc, start, segment):
    """Популярность фильма среди пользователей того же сегмента (например, возраста) за 30 дней."""
    recent = history[history["date"] >= start - pd.Timedelta(days=30)]
    recent = recent.merge(users_enc[["user_id", segment]], on="user_id", how="left")
    recent[segment] = recent[segment].fillna(-1)
    seg_item = recent.groupby([segment, "item_id"]).size().rename("cnt").reset_index()
    seg_total = recent.groupby(segment).size().rename("total")
    seg_item = seg_item.merge(seg_total, on=segment)
    seg_item[f"ui_{segment}_segment_pop"] = seg_item["cnt"] / seg_item["total"]

    pairs = candidates[["user_id", "item_id"]].merge(users_enc[["user_id", segment]], on="user_id", how="left")
    pairs[segment] = pairs[segment].fillna(-1)
    pairs = pairs.merge(seg_item[[segment, "item_id", f"ui_{segment}_segment_pop"]],
                        on=[segment, "item_id"], how="left")
    return pairs[["user_id", "item_id", f"ui_{segment}_segment_pop"]].fillna(0)


def build(week, candidates, users, items):
    history, start = week.history, week.start
    users_enc = encode_users(users)

    df = candidates.merge(users_enc, on="user_id", how="left")
    df[["age", "income", "sex", "kids_flg"]] = df[["age", "income", "sex", "kids_flg"]].fillna(-1)
    df = df.merge(user_features(history, start, items), on="user_id", how="left")
    df = df.merge(item_features(history, start, items), on="item_id", how="left")

    for attr, name in [("genres", "genre"), ("countries", "country"), ("directors", "director")]:
        df = df.merge(user_attribute_share(candidates, history, items, attr, name),
                      on=["user_id", "item_id"], how="left")
    df = df.merge(segment_popularity(candidates, history, users_enc, start, "age"),
                  on=["user_id", "item_id"], how="left")

    # насколько год выпуска фильма отличается от "типичного" года для пользователя
    df["ui_year_diff"] = (df["i_release_year"] - df["u_mean_release_year"]).abs()
    # насколько тип контента (фильм/сериал) совпадает с привычками пользователя
    df["ui_type_match"] = np.where(df["i_is_series"] == 1, df["u_share_series"], 1 - df["u_share_series"])
    return df


def feature_columns(df):
    service = {"user_id", "item_id", "label", "week"}
    return [c for c in df.columns if c not in service]
