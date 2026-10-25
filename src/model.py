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


def to_matrix(df, features):
    """float32-матрица без промежуточной копии DataFrame (при 300 признаках это экономит гигабайты)."""
    matrix = np.empty((len(df), len(features)), dtype=np.float32)
    for j, feature in enumerate(features):
        matrix[:, j] = df[feature].to_numpy(np.float32)
    return matrix


def train(train_df, valid_df, features, params=None):
    model = HistGradientBoostingClassifier(**(params or PARAMS))
    started = time.time()
    model.fit(to_matrix(train_df, features), train_df["label"].to_numpy(),
              X_val=to_matrix(valid_df, features), y_val=valid_df["label"].to_numpy())
    model.fit_seconds_ = time.time() - started
    return model


def predict(model, df, features, chunk_size=100_000):
    parts = [model.predict_proba(to_matrix(df.iloc[i:i + chunk_size], features))[:, 1]
             for i in range(0, len(df), chunk_size)]
    return np.concatenate(parts)
