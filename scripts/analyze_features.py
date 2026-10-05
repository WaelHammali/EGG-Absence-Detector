"""Part VII: class comparisons and feature-correlation analysis."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


METADATA = {"recording", "subject", "window", "start_s", "end_s", "class"}
Q10_FEATURES = [
    "avg_energy",
    "avg_delta_power",
    "avg_theta_power",
    "avg_std",
    "avg_rms",
    "avg_amplitude",
    "avg_delta_relative_power",
    "avg_theta_relative_power",
]


def dataset_path(mode: str) -> Path:
    preferred = ROOT / f"data/processed/windows_features_{mode}.parquet"
    if preferred.exists():
        return preferred
    if mode == "provisional":
        legacy = ROOT / "data/processed/windows_features.parquet"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f"Build {mode} dataset first: {preferred}")


def class_statistics(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    target = frame["class"].to_numpy()
    rows = []
    for feature in features:
        values = frame[feature].to_numpy(dtype=np.float64)
        zero = values[target == 0]
        one = values[target == 1]
        pooled = np.sqrt(((len(zero) - 1) * zero.var(ddof=1) + (len(one) - 1) * one.var(ddof=1)) / max(len(zero) + len(one) - 2, 1))
        effect = (one.mean() - zero.mean()) / pooled if pooled > 0 else np.nan
        auc = roc_auc_score(target, values)
        rows.append(
            {
                "feature": feature,
                "class_0_mean": zero.mean(),
                "class_1_mean": one.mean(),
                "class_0_median": np.median(zero),
                "class_1_median": np.median(one),
                "cohens_d": effect,
                "roc_auc": auc,
                "separation_auc": max(auc, 1 - auc),
                "higher_in_class": 1 if one.mean() > zero.mean() else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["separation_auc", "cohens_d"], ascending=[False, False])


def high_correlation_pairs(correlation: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    values = correlation.to_numpy()
    rows = []
    for row in range(len(correlation)):
        for column in range(row + 1, len(correlation)):
            value = float(values[row, column])
            if abs(value) >= threshold:
                rows.append(
                    {
                        "feature_1": correlation.index[row],
                        "feature_2": correlation.columns[column],
                        "correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "correlation", "absolute_correlation"]
    )


def boxplots(frame: pd.DataFrame, output: Path) -> None:
    available = [feature for feature in Q10_FEATURES if feature in frame]
    rows = []
    for feature in available:
        values = frame[feature].to_numpy(dtype=np.float64)
        positive = values[values > 0]
        epsilon = float(positive.min() / 10) if positive.size else 1e-12
        transformed = np.log10(np.maximum(values, 0) + epsilon)
        rows.append(pd.DataFrame({"feature": feature, "log10_value": transformed, "class": frame["class"].astype(str)}))
    long = pd.concat(rows, ignore_index=True)
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for axis, feature in zip(axes.flat, available):
        sns.boxplot(
            data=long.loc[long["feature"] == feature],
            x="class",
            y="log10_value",
            hue="class",
            legend=False,
            showfliers=False,
            ax=axis,
            palette={"0": "#4C78A8", "1": "#E45756"},
        )
        axis.set(
            title=feature,
            xlabel="Classe (0 = hors crise, 1 = crise)",
            ylabel="log10(valeur + ε)",
        )
    figure.suptitle("Q10 — Comparaison des caractéristiques EEG entre les classes", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def heatmaps(correlation: pd.DataFrame, average_correlation: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 11), constrained_layout=True)
    sns.heatmap(average_correlation, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Q11 — Corrélation des caractéristiques moyennes sur les 8 canaux")
    figure.savefig(output_dir / "correlation_average_features.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(24, 21), constrained_layout=True)
    sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, xticklabels=False, yticklabels=False, ax=axis)
    axis.set_title("Q11 — Matrice de corrélation des 144 caractéristiques")
    axis.set_xlabel("Caractéristiques")
    axis.set_ylabel("Caractéristiques")
    figure.savefig(output_dir / "correlation_all_features.png", dpi=160)
    plt.close(figure)


def write_report(
    mode: str,
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    pairs: pd.DataFrame,
    average_correlation: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected = statistics.set_index("feature").loc[[feature for feature in Q10_FEATURES if feature in statistics["feature"].values]].reset_index()
    top = statistics.head(15)
    average_pairs = high_correlation_pairs(average_correlation, 0.90)
    lines = [
        f"# Partie VII — Analyse exploratoire ({mode})",
        "",
        f"Analyse de {len(frame):,} fenêtres : {(frame['class'] == 0).sum():,} de classe 0 et {(frame['class'] == 1).sum():,} de classe 1. Aucun modèle n’est entraîné dans cette partie.",
        "",
        "## Q10 — Comparaison entre les classes",
        "",
        "Les boxplots utilisent une échelle logarithmique car l’énergie et les puissances spectrales sont très asymétriques. Les valeurs aberrantes ne sont pas affichées dans les boîtes, mais elles restent présentes dans les calculs.",
        "",
        "| Caractéristique | Médiane classe 0 | Médiane classe 1 | Cohen d | AUC séparatrice | Plus élevée en classe |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.class_0_median:.6g} | {row.class_1_median:.6g} | {row.cohens_d:.3f} | {row.separation_auc:.3f} | {row.higher_in_class} |"
        )
    lines += [
        "",
        "Le graphique correspondant est [class_comparisons.png](../outputs/figures/" + mode + "/part_vii/class_comparisons.png). Une AUC séparatrice proche de 1 indique une bonne séparation univariée; 0,5 indique une absence de séparation.",
        "",
        "### Caractéristiques les plus discriminantes",
        "",
        "| Rang | Caractéristique | AUC séparatrice | Cohen d | Direction moyenne |",
        "|---:|---|---:|---:|---|",
    ]
    for rank, row in enumerate(top.itertuples(index=False), 1):
        direction = "crise > hors crise" if row.higher_in_class == 1 else "crise < hors crise"
        lines.append(f"| {rank} | {row.feature} | {row.separation_auc:.3f} | {row.cohens_d:.3f} | {direction} |")
    lines += [
        "",
        "Ces résultats mesurent chaque variable séparément. Ils ne remplacent pas une évaluation par enregistrement et ne constituent pas encore un modèle prédictif.",
        "",
        "## Q11 — Corrélations et redondance",
        "",
        f"La matrice complète contient {len(pairs)} paires avec |r| ≥ 0,95. La matrice des 16 moyennes inter-canaux contient {len(average_pairs)} paires avec |r| ≥ 0,90.",
        "",
        "Paires moyennes fortement corrélées :",
        "",
        "| Caractéristique 1 | Caractéristique 2 | Corrélation |",
        "|---|---|---:|",
    ]
    for row in average_pairs.head(20).itertuples(index=False):
        lines.append(f"| {row.feature_1} | {row.feature_2} | {row.correlation:.3f} |")
    lines += [
        "",
        "Les redondances attendues sont `std` avec `variance`, `RMS` avec `energy`, et les versions absolues d’une même bande entre canaux voisins. Les puissances relatives partagent le même dénominateur 0,5–30 Hz et peuvent également être corrélées ou anticorrélées. Une sélection de variables devra conserver une seule représentation parmi les groupes presque équivalents avant les modèles sensibles à la colinéarité.",
        "",
        "Figures : [corrélations moyennes](../outputs/figures/" + mode + "/part_vii/correlation_average_features.png) et [matrice complète](../outputs/figures/" + mode + "/part_vii/correlation_all_features.png). Les tableaux complets sont dans `outputs/feature_class_comparison_" + mode + ".csv` et `outputs/high_correlations_" + mode + ".csv`.",
        "",
    ]
    (ROOT / f"docs/PART_VII_ANALYSIS_{mode.upper()}.md").write_text("\n".join(lines))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--annotations", choices=["official", "provisional"], required=True)
    args = parser.parse_args()
    mode = args.annotations
    path = dataset_path(mode)
    frame = pd.read_parquet(path)
    feature_columns = [column for column in frame.columns if column not in METADATA]
    if frame[feature_columns].isna().any().any():
        raise ValueError("Feature dataset contains missing values")
    output_dir = ROOT / f"outputs/figures/{mode}/part_vii"
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics = class_statistics(frame, feature_columns)
    correlation = frame[feature_columns].corr(method="pearson")
    average_columns = [column for column in feature_columns if column.startswith("avg_")]
    average_correlation = correlation.loc[average_columns, average_columns]
    pairs = high_correlation_pairs(correlation)

    statistics.to_csv(ROOT / f"outputs/feature_class_comparison_{mode}.csv", index=False)
    pairs.to_csv(ROOT / f"outputs/high_correlations_{mode}.csv", index=False)
    correlation.to_csv(ROOT / f"outputs/feature_correlations_{mode}.csv")
    boxplots(frame, output_dir / "class_comparisons.png")
    heatmaps(correlation, average_correlation, output_dir)
    write_report(mode, frame, statistics, pairs, average_correlation, output_dir)
    print(f"Part VII report: {ROOT / f'docs/PART_VII_ANALYSIS_{mode.upper()}.md'}")
    print(f"Top discriminants:\n{statistics.head(10)[['feature', 'separation_auc', 'cohens_d']].to_string(index=False)}")
    print(f"Highly correlated pairs |r|>=0.95: {len(pairs)}")


if __name__ == "__main__":
    main()
