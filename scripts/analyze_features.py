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
        # Pooled AUC can reflect between-recording differences; also score within each recording.
        within = [
            roc_auc_score(group["class"], group[feature])
            for _, group in frame.groupby("recording")
            if group["class"].nunique() == 2
        ]
        within_auc = float(np.median(within))
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
                "within_recording_auc_median": within_auc,
                "within_recording_separation_auc": max(within_auc, 1 - within_auc),
                "higher_in_class": 1 if np.median(one) > np.median(zero) else 0,
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


def redundancy_groups(correlation: pd.DataFrame, threshold: float = 0.95) -> list[list[str]]:
    """Connected components of the |rho| >= threshold graph, largest first."""
    adjacency = (correlation.abs().to_numpy() >= threshold)
    names = correlation.index.tolist()
    unseen = set(range(len(names)))
    groups = []
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = [other for other in np.flatnonzero(adjacency[node]) if other in unseen]
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
        if len(component) > 1:
            groups.append(sorted(names[index] for index in component))
    return sorted(groups, key=len, reverse=True)


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


def heatmaps(correlation: pd.DataFrame, average_columns: list[str], output_dir: Path, method: str) -> None:
    label = {"pearson": "Pearson r", "spearman": "Spearman ρ"}[method]
    suffix = "" if method == "pearson" else f"_{method}"
    figure, axis = plt.subplots(figsize=(13, 11), constrained_layout=True)
    sns.heatmap(
        correlation.loc[average_columns, average_columns],
        cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axis,
        cbar_kws={"label": label},
    )
    axis.set_title(f"Q11 — Corrélation ({label}) des caractéristiques moyennes sur les 8 canaux")
    figure.savefig(output_dir / f"correlation_average_features{suffix}.png", dpi=180)
    plt.close(figure)

    # Order by feature type, then channel, so same-feature blocks sit on the diagonal.
    ordered = sorted(correlation.index, key=lambda name: (name.split("_", 1)[1], name.split("_", 1)[0]))
    matrix = correlation.loc[ordered, ordered]
    figure, axis = plt.subplots(figsize=(24, 21), constrained_layout=True)
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, xticklabels=False, yticklabels=False, ax=axis, cbar_kws={"label": label})
    feature_names = [name.split("_", 1)[1] for name in ordered]
    boundaries = [index for index in range(1, len(feature_names)) if feature_names[index] != feature_names[index - 1]]
    for boundary in boundaries:
        axis.axhline(boundary, color="white", linewidth=1)
        axis.axvline(boundary, color="white", linewidth=1)
    centers = [(start + end) / 2 for start, end in zip([0, *boundaries], [*boundaries, len(ordered)])]
    groups = [feature_names[int(start)] for start in [0, *boundaries]]
    axis.set_xticks(centers, groups, rotation=90)
    axis.set_yticks(centers, groups, rotation=0)
    axis.set_title(f"Q11 — Matrice de corrélation ({label}) des {len(ordered)} caractéristiques (8 canaux + moyenne par bloc)")
    figure.savefig(output_dir / f"correlation_all_features{suffix}.png", dpi=160)
    plt.close(figure)


def write_report(
    mode: str,
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    pairs: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    average_columns: list[str],
) -> None:
    selected = statistics.set_index("feature").loc[[feature for feature in Q10_FEATURES if feature in statistics["feature"].values]].reset_index()
    top = statistics.head(15)
    average_spearman = spearman.loc[average_columns, average_columns]
    average_pearson = pearson.loc[average_columns, average_columns]
    average_pairs = high_correlation_pairs(average_spearman, 0.90)
    groups = redundancy_groups(spearman, 0.95)
    spearman_pairs = high_correlation_pairs(spearman, 0.95)
    within_top = statistics.sort_values("within_recording_separation_auc", ascending=False).head(10)
    dc_features = [f"avg_{name}" for name in ("mean", "min", "max", "rms", "energy")]
    dc = statistics.set_index("feature").loc[dc_features]
    figure_base = "../outputs/figures/" + mode + "/part_vii/"
    lines = [
        f"# Partie VII — Analyse exploratoire ({mode})",
        "",
        f"Analyse de {len(frame):,} fenêtres : {(frame['class'] == 0).sum():,} de classe 0 et {(frame['class'] == 1).sum():,} de classe 1. Aucun modèle n’est entraîné dans cette partie.",
        "",
        "## Q10 — Comparaison entre les classes",
        "",
        "Les boxplots utilisent une échelle logarithmique car l’énergie et les puissances spectrales sont très asymétriques. Les valeurs aberrantes ne sont pas affichées dans les boîtes, mais elles restent présentes dans les calculs.",
        "",
        "| Caractéristique | Médiane classe 0 | Médiane classe 1 | Cohen d | AUC séparatrice (globale) | AUC séparatrice (médiane intra-enregistrement) | Plus élevée en classe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.class_0_median:.6g} | {row.class_1_median:.6g} | {row.cohens_d:.3f} | "
            f"{row.separation_auc:.3f} | {row.within_recording_separation_auc:.3f} | {row.higher_in_class} |"
        )
    lines += [
        "",
        f"Le graphique correspondant est [class_comparisons.png]({figure_base}class_comparisons.png). Une AUC séparatrice proche de 1 indique une bonne séparation univariée; 0,5 indique une absence de séparation. L’AUC globale mélange toutes les fenêtres; l’AUC intra-enregistrement est calculée dans chaque enregistrement puis résumée par la médiane, ce qui neutralise les différences de niveau entre enregistrements. « Plus élevée en classe » compare les médianes.",
        "",
        "Les distributions sont très asymétriques : un Cohen d faible (calculé sur les moyennes et écarts-types) peut coexister avec une AUC élevée (fondée sur les rangs). L’AUC est donc la mesure de référence ici.",
        "",
        "### Caractéristiques les plus discriminantes (AUC globale)",
        "",
        "| Rang | Caractéristique | AUC globale | AUC intra-enregistrement | Cohen d | Direction (médianes) |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(top.itertuples(index=False), 1):
        direction = "crise > hors crise" if row.higher_in_class == 1 else "crise < hors crise"
        lines.append(
            f"| {rank} | {row.feature} | {row.separation_auc:.3f} | {row.within_recording_separation_auc:.3f} | {row.cohens_d:.3f} | {direction} |"
        )
    lines += [
        "",
        "### Caractéristiques les plus discriminantes à l’intérieur des enregistrements",
        "",
        "| Rang | Caractéristique | AUC intra-enregistrement | AUC globale |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(within_top.itertuples(index=False), 1):
        lines.append(f"| {rank} | {row.feature} | {row.within_recording_separation_auc:.3f} | {row.separation_auc:.3f} |")
    lines += [
        "",
        "### Avertissement : composante continue (offset DC)",
        "",
        "Plusieurs enregistrements ont une moyenne de signal très éloignée de zéro (médiane par enregistrement de `avg_mean` entre environ −830 et +1000), ce qui indique un signal non filtré passe-haut. `mean`, `min`, `max`, `rms` et `energy` sont calculés sans retrait de la moyenne : leur niveau dépend donc de cet offset propre à chaque enregistrement. À l’intérieur d’un enregistrement (offset constant), `min` et `max` suivent encore l’amplitude des pointes-ondes, mais entre enregistrements l’offset domine, d’où l’écart entre AUC globale et AUC intra-enregistrement :",
        "",
        "| Caractéristique | AUC globale | AUC intra-enregistrement |",
        "|---|---:|---:|",
    ]
    for feature, row in dc.iterrows():
        lines.append(f"| {feature} | {row.separation_auc:.3f} | {row.within_recording_separation_auc:.3f} |")
    lines += [
        "",
        "Un modèle évalué sur des enregistrements non vus risque d’utiliser ces variables pour reconnaître l’enregistrement plutôt que la crise. `std`, `variance`, `amplitude` (crête à crête) et les puissances spectrales (calculées après retrait de la moyenne) ne sont pas affectées. À décider avant les modèles : retirer ces variables ou retirer la moyenne de chaque fenêtre.",
        "",
        "Ces résultats mesurent chaque variable séparément et ne constituent pas encore un modèle prédictif.",
        "",
        "## Q11 — Corrélations et redondance",
        "",
        "Deux matrices sont calculées. Pearson mesure une relation linéaire et est dominée par les valeurs extrêmes (artefacts) de ces distributions asymétriques. Spearman mesure une relation monotone sur les rangs : c’est la mesure retenue pour juger la redondance, car `variance = std²` ou `energy ∝ rms²` sont des relations parfaitement monotones mais non linéaires.",
        "",
        f"- Paires avec |r de Pearson| ≥ 0,95 : {len(pairs)} sur {len(pearson) * (len(pearson) - 1) // 2}.",
        f"- Paires avec |ρ de Spearman| ≥ 0,95 : {len(spearman_pairs)}.",
        "",
        "Exemples Pearson vs Spearman sur les moyennes inter-canaux :",
        "",
        "| Paire | Pearson r | Spearman ρ |",
        "|---|---:|---:|",
    ]
    for first, second in [("avg_std", "avg_variance"), ("avg_rms", "avg_energy"), ("avg_std", "avg_amplitude"), ("avg_theta_power", "avg_alpha_power")]:
        lines.append(f"| {first} / {second} | {average_pearson.loc[first, second]:.3f} | {average_spearman.loc[first, second]:.3f} |")
    lines += [
        "",
        "### Caractéristiques fortement corrélées (moyennes inter-canaux, |ρ| ≥ 0,90)",
        "",
        "| Caractéristique 1 | Caractéristique 2 | Spearman ρ |",
        "|---|---|---:|",
    ]
    for row in average_pairs.itertuples(index=False):
        lines.append(f"| {row.feature_1} | {row.feature_2} | {row.correlation:.3f} |")
    lines += [
        "",
        "### Groupes redondants (|ρ| ≥ 0,95, toutes caractéristiques)",
        "",
        "Chaque groupe est une composante connexe : ses membres sont reliés par des chaînes de paires avec |ρ| ≥ 0,95. Un seul représentant par groupe suffit avant les modèles sensibles à la colinéarité.",
        "",
    ]
    auc = statistics.set_index("feature")["within_recording_separation_auc"]
    for index, group in enumerate(groups, 1):
        best = max(group, key=lambda name: auc[name])
        lines.append(f"{index}. ({len(group)}) {', '.join(f'`{name}`' for name in group)} — représentant suggéré : `{best}` (meilleure AUC intra-enregistrement {auc[best]:.3f}).")
    lines += [
        "",
        f"Figures : corrélations moyennes [Spearman]({figure_base}correlation_average_features_spearman.png) et [Pearson]({figure_base}correlation_average_features.png); matrice complète [Spearman]({figure_base}correlation_all_features_spearman.png) et [Pearson]({figure_base}correlation_all_features.png). Tableaux : `outputs/feature_class_comparison_{mode}.csv`, `outputs/high_correlations_{mode}.csv` (Pearson), `outputs/high_correlations_spearman_{mode}.csv`, `outputs/feature_correlations_{mode}.csv` et `outputs/feature_correlations_spearman_{mode}.csv`.",
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
    spearman = frame[feature_columns].corr(method="spearman")
    average_columns = [column for column in feature_columns if column.startswith("avg_")]
    pairs = high_correlation_pairs(correlation)

    statistics.to_csv(ROOT / f"outputs/feature_class_comparison_{mode}.csv", index=False)
    pairs.to_csv(ROOT / f"outputs/high_correlations_{mode}.csv", index=False)
    high_correlation_pairs(spearman).to_csv(ROOT / f"outputs/high_correlations_spearman_{mode}.csv", index=False)
    correlation.to_csv(ROOT / f"outputs/feature_correlations_{mode}.csv")
    spearman.to_csv(ROOT / f"outputs/feature_correlations_spearman_{mode}.csv")
    boxplots(frame, output_dir / "class_comparisons.png")
    heatmaps(correlation, average_columns, output_dir, "pearson")
    heatmaps(spearman, average_columns, output_dir, "spearman")
    write_report(mode, frame, statistics, pairs, correlation, spearman, average_columns)
    print(f"Part VII report: {ROOT / f'docs/PART_VII_ANALYSIS_{mode.upper()}.md'}")
    columns = ["feature", "separation_auc", "within_recording_separation_auc", "cohens_d"]
    print(f"Top discriminants:\n{statistics.head(10)[columns].to_string(index=False)}")
    print(f"Highly correlated pairs |r|>=0.95: {len(pairs)}")


if __name__ == "__main__":
    main()
