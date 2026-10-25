"""Градиентный бустинг для ранжирования кандидатов.

Используем HistGradientBoostingClassifier из scikit-learn — это гистограммный бустинг
того же типа, что LightGBM. Гиперпараметры одинаковые для всех сравниваемых наборов
признаков: сравниваем признаки, а не тюнинг модели.

Ранняя остановка — по валидационной неделе (а не по случайной доле трейна), чтобы
не перемешивать время.
"""

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src import config

PARAMS = dict(
    learning_rate=0.05,
    max_iter=600,
    max_leaf_nodes=63,
    min_samples_leaf=200,
    l2_regularization=1.0,
    early_stopping=True,
    n_iter_no_change=30,
    scoring="loss",
    random_state=config.SEED,
)


def train(train_df, valid_df, features, params=None):
    model = HistGradientBoostingClassifier(**(params or PARAMS))
    started = time.time()
    model.fit(train_df[features].to_numpy(np.float32), train_df["label"].to_numpy(),
              X_val=valid_df[features].to_numpy(np.float32), y_val=valid_df["label"].to_numpy())
    model.fit_seconds_ = time.time() - started
    return model


def predict(model, df, features):
    return model.predict_proba(df[features].to_numpy(np.float32))[:, 1]
