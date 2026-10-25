"""Шаг 2. Для каждой целевой недели строим историю, кандидатов и разметку."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.candidates import generate
from src.metrics import candidate_recall
from src.split import make_week


def all_weeks():
    weeks = [(w, "train") for w in config.TRAIN_WEEKS]
    weeks += [(config.VALID_WEEK, "valid"), (config.TEST_WEEK, "test")]
    return weeks


def main():
    events = pd.read_pickle(config.PROCESSED_DIR / "events.pkl")
    items = pd.read_pickle(config.PROCESSED_DIR / "items.pkl")

    for week_start, part in all_weeks():
        week = make_week(events, week_start, name=week_start)
        candidates = generate(week, items)
        candidates.to_pickle(config.PROCESSED_DIR / f"candidates_{week_start}.pkl")
        print(f"{part:5s} {week_start}: пользователей {week.target.user_id.nunique():,}, "
              f"кандидатов {len(candidates):,}, "
              f"полнота кандидатов {candidate_recall(candidates, week.target):.3f}, "
              f"доля позитивов {candidates.label.mean():.4f}")


if __name__ == "__main__":
    main()
