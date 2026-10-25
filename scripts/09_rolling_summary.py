"""Шаг 9. Сводка rolling evaluation: средние по трём фолдам, бутстрап и график."""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

ROLLING_DIR = config.REPORTS_DIR / "rolling"
PER_USER_DIR = config.PROCESSED_DIR / "rolling"
BASE = "manual_gbdt"

MODELS = [  # имя, подпись, цвет
    ("manual_gbdt", "Ручные (37)", "#2a78d6"),
    ("manual_without_candidate_features", "Ручные без признаков кандидатов (33)", "#2a78d6"),
    ("auto_all", "Авто: вся сетка (311)", "#eb6834"),
    ("auto_v4", "Авто: отбор v4", "#eb6834"),
    ("auto_v3_family_ndcg", "Авто: отбор v3 (семействами)", "#eb6834"),
    ("auto_v4_plus_candidate_features", "Авто v4 + признаки кандидатов", "#1baf7a"),
    ("manual_plus_auto_v4", "Ручные + авто v4", "#1baf7a"),
    ("popularity_3d", "Популярность за 3 дня", "#8a8984"),
    ("popularity_7d", "Популярность за 7 дней", "#8a8984"),
]
INK, INK_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
METRIC = f"ndcg@{config.K}"


def load_folds():
    folds = []
    for path in sorted(ROLLING_DIR.glob("fold_*.json")):
        fold_id = int(path.stem.split("_")[1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_user = pd.read_pickle(PER_USER_DIR / f"per_user_fold_{fold_id}.pkl")
        folds.append((fold_id, payload, per_user))
    return folds


def paired_bootstrap(deltas, n_boot=2000, seed=config.SEED):
    """deltas: разницы по парам (пользователь, неделя), собранные со всех фолдов.
    Перевыборка внутри каждого фолда отдельно, чтобы сохранить вес недель."""
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for values in deltas:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        boots += values[idx].mean(axis=1) / len(deltas)
    point = np.mean([v.mean() for v in deltas])
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def summarize(folds):
    rows = []
    for name, label, _ in MODELS:
        row = {"model": name, "label": label}
        valid, test, deltas = [], [], []
        for fold_id, payload, per_user in folds:
            result = payload["results"][name]
            valid.append(result[f"valid_{METRIC}"])
            test.append(result[f"test_{METRIC}"])
            row[f"valid_fold{fold_id}"] = result[f"valid_{METRIC}"]
            row[f"test_fold{fold_id}"] = result[f"test_{METRIC}"]
            if name != BASE:
                deltas.append((per_user[name] - per_user[BASE]).to_numpy())
        row["valid_mean"], row["test_mean"] = np.mean(valid), np.mean(test)
        if name != BASE:
            base = [payload["results"][BASE] for _, payload, _ in folds]
            row["test_wins_vs_manual"] = sum(t > b[f"test_{METRIC}"] for t, b in zip(test, base))
            row["valid_wins_vs_manual"] = sum(v > b[f"valid_{METRIC}"] for v, b in zip(valid, base))
            row["delta_test"], row["ci_low"], row["ci_high"] = paired_bootstrap(deltas)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("valid_mean", ascending=False).reset_index(drop=True)


def plot(summary, folds, path):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })
    colors = {name: color for name, _, color in MODELS}
    df = summary[summary["model"] != BASE].iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axvline(0, color=colors[BASE], linewidth=1.2, linestyle=":")
    for y, row in df.iterrows():
        color = colors[row["model"]]
        # маленькие точки — разница на отдельных тестовых неделях
        for fold_id, payload, _ in folds:
            delta = payload["results"][row["model"]][f"test_{METRIC}"] - payload["results"][BASE][f"test_{METRIC}"]
            ax.plot(delta, y, "o", color=color, alpha=0.35, markersize=5)
        ax.errorbar(row["delta_test"], y, xerr=[[row["delta_test"] - row["ci_low"]],
                                                 [row["ci_high"] - row["delta_test"]]],
                    fmt="o", color=color, markersize=8, elinewidth=2, capsize=0)
        ax.text(row["ci_high"] + 0.0006, y, f"{row['delta_test']:+.4f}", va="center", fontsize=9)
    ax.set_yticks(range(len(df)), df["label"])
    ax.set_xlabel("Разница NDCG@10 с ручным бустингом на тесте\n"
                  "крупная точка — среднее по 3 неделям с 95% бутстрап-интервалом, мелкие — отдельные недели")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def to_markdown(summary, folds):
    fold_ids = [f for f, _, _ in folds]
    header = ("| Модель | NDCG@10 valid, среднее ↓ | NDCG@10 test, среднее | "
              + " | ".join(f"test, фолд {f}" for f in fold_ids)
              + " | Δ test к ручным [95% ДИ] | побед над ручными (valid / test) |")
    lines = [header, "|" + "---|" * (5 + len(fold_ids))]
    for _, row in summary.iterrows():
        per_fold = " | ".join(f"{row[f'test_fold{f}']:.4f}" for f in fold_ids)
        if row["model"] == BASE:
            delta, wins = "—", "—"
        else:
            delta = f"{row['delta_test']:+.4f} [{row['ci_low']:+.4f}; {row['ci_high']:+.4f}]"
            wins = f"{int(row['valid_wins_vs_manual'])} / {int(row['test_wins_vs_manual'])} из {len(fold_ids)}"
        lines.append(f"| {row['label']} | {row['valid_mean']:.4f} | {row['test_mean']:.4f} | "
                     f"{per_fold} | {delta} | {wins} |")
    return "\n".join(lines)


def main():
    folds = load_folds()
    summary = summarize(folds)
    summary.to_csv(ROLLING_DIR / "summary.csv", index=False)
    markdown = to_markdown(summary, folds)
    (ROLLING_DIR / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)

    best = summary.iloc[0]
    print(f"\nЛучшая модель по среднему valid: {best['label']} "
          f"(valid {best['valid_mean']:.4f}, test {best['test_mean']:.4f})")
    for fold_id, payload, _ in folds:
        print(f"Фолд {fold_id}: v4 выбрал {len(payload['selected']['auto_v4'])} признаков, "
              f"v3 — {len(payload['selected']['auto_v3_family_ndcg'])}")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot(summary, folds, config.FIGURES_DIR / "rolling_comparison.png")


if __name__ == "__main__":
    main()
