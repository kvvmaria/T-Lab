"""Шаг 5. Итоговое сравнение: таблица, бутстрап-интервалы и графики для отчёта."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src import config
from src.experiment import load_inputs
from src.metrics import ranking_metrics
from src.reporting import load_results
from src.split import make_week

# Палитра: фиксированный порядок цветов, серый — для "просто популярности"
COLORS = {"manual": "#2a78d6", "auto": "#eb6834", "combined": "#1baf7a", "reference": "#8a8984"}
INK, INK_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", "#9ec5f4", "#3987e5", "#184f95"])

MODELS = [  # имя в results.json, подпись, цвет
    ("popularity_7d", "Популярность за 7 дней", "reference"),
    ("manual_gbdt", "Ручные (37)", "manual"),
    ("manual_without_candidate_features", "Ручные без признаков кандидатов (33)", "manual"),
    ("auto_all", "Авто: вся сетка (311)", "auto"),
    ("auto_selected", "Авто: поштучный отбор (55)", "auto"),
    ("auto_family_ndcg_selected", "Авто: отбор семействами по NDCG (67)", "auto"),
    ("auto_selected_plus_candidate_features", "Авто (55) + признаки кандидатов (4)", "combined"),
    ("manual_plus_auto_selected", "Ручные (37) + авто (55)", "combined"),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def per_user_test_metrics(name, target):
    scored = pd.read_pickle(config.PROCESSED_DIR / f"pred_test_{name}.pkl")
    return ranking_metrics(scored, target, k=config.K, per_user=True)


def paired_bootstrap(a, b, n_boot=2000, seed=config.SEED):
    """Доверительный интервал для разницы средних (b − a) по одним и тем же пользователям."""
    rng = np.random.default_rng(seed)
    diff = (b - a).to_numpy()
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def results_table(results):
    rows = []
    for name, result in results.items():
        row = {"model": name}
        for key in ["n_features", "test_ndcg@10", "test_recall@10", "test_map@10",
                    "test_roc_auc", "test_logloss", "valid_ndcg@10",
                    "build_seconds", "selection_seconds", "fit_seconds"]:
            row[key] = result.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_main_comparison(per_user, path):
    labels, means, lows, highs, colors = [], [], [], [], []
    for name, label, color in MODELS:
        values = per_user[name]["ndcg@10"]
        boots = np.random.default_rng(0).choice(values.to_numpy(), (1000, len(values))).mean(axis=1)
        labels.append(label)
        means.append(values.mean())
        lows.append(values.mean() - np.percentile(boots, 2.5))
        highs.append(np.percentile(boots, 97.5) - values.mean())
        colors.append(COLORS[color])

    # точки с интервалами, а не столбики: разницы маленькие, а у столбика ось обязана начинаться с нуля
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(labels))[::-1]
    for yi, m, lo, hi, c in zip(y, means, lows, highs, colors):
        ax.errorbar(m, yi, xerr=[[lo], [hi]], fmt="o", color=c, markersize=8,
                    ecolor=c, elinewidth=2, capsize=0)
        ax.text(m + hi + 0.0008, yi, f"{m:.4f}", va="center", fontsize=9, color=INK)
    manual_mean = means[[name for name, _, _ in MODELS].index("manual_gbdt")]
    ax.axvline(manual_mean, color=COLORS["manual"], linewidth=1, linestyle=":")
    ax.set_yticks(y, labels)
    ax.set_xlim(min(means) - 0.006, max(means) + 0.009)
    ax.set_xlabel("NDCG@10 на тестовой неделе (линии — 95% бутстрап-интервал по пользователям)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_window_heatmap(screening, path, top_families=18):
    """Для каждой "семьи" признаков (сущность + агрегат + фильтр) — сила по окнам."""
    df = screening[~screening["family"].str.contains("days_since")].copy()
    df["window"] = df["window"].fillna(10_000).astype(int)
    best = df.groupby("family")["gauc_strength"].max().sort_values(ascending=False)
    families = best.head(top_families).index
    pivot = df[df["family"].isin(families)].pivot_table(
        index="family", columns="window", values="gauc_strength").reindex(families)
    windows = sorted(pivot.columns)
    pivot = pivot[windows]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(pivot.to_numpy(), cmap=BLUES, aspect="auto")
    ax.set_xticks(range(len(windows)), [("вся\nистория" if w == 10_000 else f"{w} дн.") for w in windows])
    ax.set_yticks(range(len(families)),
                  [f.replace("|None", "").replace("|", " · ") for f in families], fontsize=8)
    # отмечаем лучшее окно в каждой строке
    for row, values in enumerate(pivot.to_numpy()):
        if np.all(np.isnan(values)):
            continue
        col = int(np.nanargmax(values))
        ax.text(col, row, "★", ha="center", va="center", color=INK, fontsize=9)
    ax.set_title("Какое окно лучше: |gAUC − 0.5| признака на трейне (★ — лучшее окно)", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_top_n_curve(curve, family_curves, manual_valid, auto_all_valid, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(curve["n_features"], curve["valid_ndcg@10"], color=COLORS["auto"], linewidth=2,
            marker="o", markersize=5, label="поштучная важность (LogLoss)")
    styles = {"logloss": ("#e87ba4", "семейства, важность по LogLoss"),
              "ndcg": ("#6250d6", "семейства, важность по NDCG@10")}
    for metric, fam in family_curves.items():
        color, label = styles[metric]
        ax.plot(fam["n_features"], fam["valid_ndcg@10"], color=color, linewidth=2,
                marker="s", markersize=5, label=label)
    ax.axhline(auto_all_valid, color=COLORS["reference"], linewidth=1.5, linestyle=":",
               label="вся сетка (311)")
    ax.axhline(manual_valid, color=COLORS["manual"], linewidth=2, linestyle="--",
               label="ручные признаки (37)")
    ax.set_xlabel("Сколько признаков оставили")
    ax.set_ylabel("NDCG@10, валидация")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(importance, selected, path, top=20):
    df = importance[importance["feature"].isin(selected)].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"].str.replace("a__", ""), df["importance"], color=COLORS["auto"], height=0.6)
    ax.set_xlabel("Рост LogLoss на валидации при перемешивании признака")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_by_history_length(per_user, history, path):
    lengths = history.groupby("user_id").size()
    buckets = pd.cut(lengths, [0, 2, 5, 10, 20, 10_000], labels=["1–2", "3–5", "6–10", "11–20", "21+"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [("manual_gbdt", "ручные", "manual"), ("auto_selected", "отобранные авто", "auto"),
             ("manual_plus_auto_selected", "ручные + авто", "combined")]
    width = 0.26
    x = np.arange(len(buckets.cat.categories))
    for i, (name, label, color) in enumerate(names):
        values = per_user[name]["ndcg@10"].groupby(buckets.reindex(per_user[name].index)).mean()
        ax.bar(x + (i - 1) * width, values.to_numpy(), width=width - 0.02, color=COLORS[color], label=label)
    counts = buckets.reindex(per_user["manual_gbdt"].index).value_counts().reindex(buckets.cat.categories)
    ax.set_xticks(x, [f"{c}\n({n:,} польз.)" for c, n in zip(buckets.cat.categories, counts)])
    ax.set_xlabel("Сколько фильмов пользователь посмотрел до тестовой недели")
    ax.set_ylabel("NDCG@10, тест")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    events, _, _ = load_inputs()
    test_week = make_week(events, config.TEST_WEEK, name=config.TEST_WEEK)

    table = results_table(results)
    table.to_csv(config.REPORTS_DIR / "results_table.csv", index=False)
    print(table.round(4).to_string(index=False))

    names = [name for name in results if (config.PROCESSED_DIR / f"pred_test_{name}.pkl").exists()]
    per_user = {name: per_user_test_metrics(name, test_week.target) for name in names}

    print("\nРазница NDCG@10 с ручным бустингом на тесте (парный бутстрап по пользователям):")
    rows = []
    for name in names:
        if name == "manual_gbdt":
            continue
        mean, low, high = paired_bootstrap(per_user["manual_gbdt"]["ndcg@10"], per_user[name]["ndcg@10"])
        rows.append({"model": name, "delta_ndcg@10": mean, "ci_low": low, "ci_high": high})
        print(f"  {name:28s} {mean:+.4f}  [{low:+.4f}, {high:+.4f}]")
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "bootstrap_vs_manual.csv", index=False)

    selection_dir = config.REPORTS_DIR / "selection"
    screening = pd.read_csv(selection_dir / "screening.csv")
    importance = pd.read_csv(selection_dir / "permutation_importance.csv")
    curve = pd.read_csv(selection_dir / "top_n_curve.csv")
    selected = pd.read_csv(selection_dir / "selected_features.csv")["feature"].to_list()

    plot_main_comparison(per_user, config.FIGURES_DIR / "main_comparison.png")
    plot_window_heatmap(screening, config.FIGURES_DIR / "window_heatmap.png")
    family_curves = {"logloss": pd.read_csv(selection_dir / "family_curve.csv"),
                     "ndcg": pd.read_csv(selection_dir / "family_ndcg_curve.csv")}
    plot_top_n_curve(curve, family_curves, results["manual_gbdt"]["valid_ndcg@10"],
                     results["auto_all"]["valid_ndcg@10"], config.FIGURES_DIR / "top_n_curve.png")
    plot_importance(importance, selected, config.FIGURES_DIR / "importance_selected.png")
    plot_by_history_length(per_user, test_week.history, config.FIGURES_DIR / "ndcg_by_history_length.png")
    print("\nГрафики сохранены в", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
