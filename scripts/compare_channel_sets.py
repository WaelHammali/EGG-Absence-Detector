"""Compare the archived 8-channel results with the current 19-channel results.

Both runs use the same 21 recordings, the same labels and the same folds; only the feature
channels differ. Writes docs/CHANNELS_8_VS_19.md and a figure of importance by channel.
"""

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import CHANNELS, SCORING_CHANNELS


OLD = ROOT / "outputs/archive_8ch"
NEW = ROOT / "outputs"
FIGURE = ROOT / "outputs/figures/official/channels/importance_by_channel.png"
MODELS = ["Decision Tree", "Random Forest", "KNN", "SVM"]
REGIONS = {"Fp": "Frontal", "F": "Frontal", "C": "Central", "T": "Temporal", "P": "Parietal", "O": "Occipital"}
# Same region hues as the app (light theme); parietal and occipital share the posterior hue.
REGION_COLORS = {"Frontal": "#0D9488", "Central": "#2563EB", "Temporal": "#DB2777", "Parietal": "#0284C7", "Occipital": "#0284C7"}
REGION_HATCH = {"Occipital": "//"}
INK = "#52514e"


def region(channel: str) -> str:
    return "All channels (average)" if channel == "avg" else REGIONS[channel.rstrip("0123456789z")]


def importances(path: Path) -> pd.DataFrame:
    bundle = joblib.load(path)
    frame = pd.DataFrame({"feature": bundle["features"], "importance": bundle["model"].feature_importances_})
    frame[["channel", "kind"]] = frame["feature"].str.split("_", n=1, expand=True)
    frame["region"] = frame["channel"].map(region)
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def both(name: str, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(OLD / name, **kwargs), pd.read_csv(NEW / name, **kwargs)


def delta(new: float, old: float) -> str:
    return f"{new - old:+.3f}"


def importance_figure(table: pd.DataFrame) -> None:
    channels = table.loc[table.channel != "avg"].groupby("channel")["importance"].sum().reindex(CHANNELS)
    figure, axis = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    seen = set()
    for index, (channel, value) in enumerate(channels.items()):
        name = region(channel)
        axis.bar(
            index, value, width=0.7, color=REGION_COLORS[name], hatch=REGION_HATCH.get(name), edgecolor="white", linewidth=0.8,
            label=None if name in seen else name,
        )
        seen.add(name)
        axis.text(index, value, f"{value:.1%}", ha="center", va="bottom", fontsize=8, color=INK)
    axis.set_xticks(range(len(channels)), channels.index)
    axis.set_ylabel("Share of Random Forest importance")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axis.set_title("Importance by channel — final Random Forest, 19 channels (channel-average features not shown)")
    axis.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.legend(frameon=False, ncol=5, loc="upper right")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=150)
    plt.close(figure)


def main() -> None:
    old_random, new_random = both("models_random_split.csv")
    old_group, new_group = both("models_groupkfold_summary.csv", index_col=[0, 1])
    old_folds, new_folds = both("models_groupkfold_folds.csv")
    old_event, new_event = (frame.set_index("model") for frame in both("model_comparison.csv"))
    old_importance = importances(OLD / "rf_final_8ch.joblib")
    new_importance = importances(ROOT / "models/rf_final.joblib")
    importance_figure(new_importance)

    lines = [
        "# 8 channels vs 19 channels",
        "",
        f"Same 21 recordings, same annotations and labels, same windows and the same folds. Only the channels used for the features change: "
        f"{len(SCORING_CHANNELS)} channels ({', '.join(SCORING_CHANNELS)}; {len(old_importance)} features) versus the {len(CHANNELS)} EEG channels of the 10-20 montage "
        f"({', '.join(CHANNELS)}; {len(new_importance)} features). ECG, EMG and SLI are excluded in both. "
        "`200625A_F` has only 8 EEG channels, but it is excluded for lack of annotations, so every recording used has all 19.",
        "",
        "The 8-channel files are kept in `outputs/archive_8ch/` and `docs/archive_8ch/`. Annotation decisions (duplicate rows, corrected end time) are still scored on the original 8 channels, so the labels are identical in both runs.",
        "",
        "## Window level — GroupKFold by recording (5 folds), unbalanced training",
        "",
        "| Model | F1 8 ch | F1 19 ch | Δ F1 | Recall 8 ch | Recall 19 ch | Precision 8 ch | Precision 19 ch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        a, b = old_group.loc[("non équilibré", model)], new_group.loc[("non équilibré", model)]
        lines.append(
            f"| {model} | {a.f1_mean:.3f} ± {a.f1_std:.3f} | {b.f1_mean:.3f} ± {b.f1_std:.3f} | {delta(b.f1_mean, a.f1_mean)} | "
            f"{a.recall_mean:.3f} | {b.recall_mean:.3f} | {a.precision_mean:.3f} | {b.precision_mean:.3f} |"
        )
    lines += [
        "",
        "## Window level — GroupKFold, undersampled (balanced) training",
        "",
        "| Model | F1 8 ch | F1 19 ch | Δ F1 | Recall 8 ch | Recall 19 ch | Precision 8 ch | Precision 19 ch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        a, b = old_group.loc[("sous-échantillonné", model)], new_group.loc[("sous-échantillonné", model)]
        lines.append(
            f"| {model} | {a.f1_mean:.3f} ± {a.f1_std:.3f} | {b.f1_mean:.3f} ± {b.f1_std:.3f} | {delta(b.f1_mean, a.f1_mean)} | "
            f"{a.recall_mean:.3f} | {b.recall_mean:.3f} | {a.precision_mean:.3f} | {b.precision_mean:.3f} |"
        )
    lines += [
        "",
        "## Window level — random stratified 80/20 split, unbalanced training",
        "",
        "| Model | Accuracy 8 / 19 | Precision 8 / 19 | Recall 8 / 19 | F1 8 ch | F1 19 ch | Δ F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    pick = lambda frame, model: frame.loc[(frame.training == "non équilibré") & (frame.model == model)].iloc[0]
    for model in MODELS:
        a, b = pick(old_random, model), pick(new_random, model)
        lines.append(
            f"| {model} | {a.accuracy:.3f} / {b.accuracy:.3f} | {a.precision:.3f} / {b.precision:.3f} | {a.recall:.3f} / {b.recall:.3f} | "
            f"{a.f1:.3f} | {b.f1:.3f} | {delta(b.f1, a.f1)} |"
        )
    lines += [
        "",
        "## Event level — whole seizures, out-of-fold predictions, each model at its best threshold",
        "",
        "| Model | Threshold 8 / 19 | Event F1 8 ch | Event F1 19 ch | Δ | Found 8 / 19 (of 93) | False alarms 8 / 19 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        a, b = old_event.loc[model], new_event.loc[model]
        lines.append(
            f"| {model} | {a.threshold:.2f} / {b.threshold:.2f} | {a.event_f1:.3f} | {b.event_f1:.3f} | {delta(b.event_f1, a.event_f1)} | "
            f"{int(a.found)} / {int(b.found)} | {int(a.false_alarms)} / {int(b.false_alarms)} |"
        )
    lines += [
        "",
        "The threshold of each run is chosen on its own out-of-fold predictions, so both event-level columns are slightly optimistic in the same way.",
        "",
        "## Training time",
        "",
        "Seconds to fit and predict, measured inside `train_models.py`. The two runs were timed separately on the same machine, so differences of a few tenths of a second are noise.",
        "",
        "| Model | Random split, unbalanced: 8 ch | 19 ch | Ratio | GroupKFold mean per fold, unbalanced: 8 ch | 19 ch | Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fold_time = lambda frame, model: frame.loc[(frame.training == "non équilibré") & (frame.model == model), "seconds"].mean()
    for model in MODELS:
        a, b = pick(old_random, model).seconds, pick(new_random, model).seconds
        c, d = fold_time(old_folds, model), fold_time(new_folds, model)
        lines.append(f"| {model} | {a:.1f} | {b:.1f} | ×{b / a:.1f} | {c:.1f} | {d:.1f} | ×{d / c:.1f} |")

    by_region = new_importance.groupby("region")["importance"].sum().sort_values(ascending=False)
    counts = pd.Series({name: sum(region(channel) == name for channel in CHANNELS) for name in set(REGIONS.values())})
    by_channel = new_importance.loc[new_importance.channel != "avg"].groupby("channel")["importance"].sum().sort_values(ascending=False)
    by_kind = new_importance.groupby("kind")["importance"].sum().sort_values(ascending=False)
    new_channels = [channel for channel in CHANNELS if channel not in SCORING_CHANNELS]
    added_share = new_importance.loc[new_importance.channel.isin(new_channels), "importance"].sum()
    lines += [
        "",
        "## Feature importance — final Random Forest trained on all 21 recordings",
        "",
        "Impurity-based importances (they sum to 1). When several features carry the same information the importance is split between them, "
        "so read these as indications of where the model looks, not as a ranking of medical relevance.",
        "",
        "### Top 20 features with 19 channels",
        "",
        "| Rank | Feature | Channel | Region | Importance |",
        "|---:|---|---|---|---:|",
    ]
    for rank, row in enumerate(new_importance.head(20).itertuples(index=False), 1):
        channel = "all (average)" if row.channel == "avg" else row.channel
        lines.append(f"| {rank} | `{row.feature}` | {channel} | {row.region} | {row.importance:.4f} |")
    lines += [
        "",
        "For reference, the top 10 with 8 channels: " + ", ".join(f"`{row.feature}` ({row.importance:.3f})" for row in old_importance.head(10).itertuples(index=False)) + ".",
        "",
        "### Which regions matter most",
        "",
        "| Region | Channels | Share of importance | Share per channel |",
        "|---|---:|---:|---:|",
    ]
    for name, value in by_region.items():
        if name == "All channels (average)":
            lines.append(f"| {name} | — | {value:.1%} | — |")
        else:
            lines.append(f"| {name} | {counts[name]} | {value:.1%} | {value / counts[name]:.1%} |")
    per_channel = (by_region.drop("All channels (average)") / counts).sort_values(ascending=False)
    lines += [
        "",
        f"Per channel, the {per_channel.index[0].lower()} region carries the most importance ({per_channel.iloc[0]:.1%} per channel), "
        f"followed by the {per_channel.index[1].lower()} region ({per_channel.iloc[1]:.1%}); the {per_channel.index[-1].lower()} region carries the least ({per_channel.iloc[-1]:.1%}). "
        f"The five most used channels are {', '.join(f'{channel} ({value:.1%})' for channel, value in by_channel.head(5).items())}. "
        f"The 11 channels that were not used before ({', '.join(new_channels)}) receive {added_share:.1%} of the importance. "
        f"Features averaged over all channels receive {by_region.get('All channels (average)', 0):.1%}.",
        "",
        "By type of feature: " + ", ".join(f"{kind} {value:.1%}" for kind, value in by_kind.head(8).items()) + ".",
        "",
        "Figure: [importance by channel](../outputs/figures/official/channels/importance_by_channel.png).",
        "",
        "## Summary",
        "",
    ]
    group_delta = pd.Series({model: new_group.loc[("non équilibré", model), "f1_mean"] - old_group.loc[("non équilibré", model), "f1_mean"] for model in MODELS})
    event_delta = (new_event["event_f1"] - old_event["event_f1"]).loc[MODELS]
    best_old, best_new = old_event["event_f1"].idxmax(), new_event["event_f1"].idxmax()
    lines += [
        f"- Window level (GroupKFold, unbalanced): F1 changes by {', '.join(f'{model} {value:+.3f}' for model, value in group_delta.items())}.",
        f"- Event level: F1 changes by {', '.join(f'{model} {value:+.3f}' for model, value in event_delta.items())}. "
        f"Best model: {best_old} with 8 channels ({old_event.loc[best_old, 'event_f1']:.3f}), {best_new} with 19 channels ({new_event.loc[best_new, 'event_f1']:.3f}).",
        "- These differences come from 5 folds over 21 recordings and 93 seizures. A change of 0.01–0.02 in event F1 is one or two seizures or false alarms and is within fold-to-fold variation; only larger, consistent changes should be read as a real effect.",
        "",
        "Reproduction: `python scripts/compare_channel_sets.py` (needs `outputs/archive_8ch/`).",
        "",
    ]
    (ROOT / "docs/CHANNELS_8_VS_19.md").write_text("\n".join(lines))
    print(f"Report: {ROOT / 'docs/CHANNELS_8_VS_19.md'}")
    print(f"Event F1 change: {event_delta.round(3).to_dict()}")


if __name__ == "__main__":
    main()
