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


DC_FEATURES = [f"avg_{name}" for name in ("mean", "min", "max", "rms", "energy")]


def dataset_path(mode: str, suffix: str = "") -> Path:
    preferred = ROOT / f"data/processed/windows_features_{mode}{suffix}.parquet"
    if preferred.exists():
        return preferred
    if mode == "provisional" and suffix == "_raw":
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


def pair_types(pairs: pd.DataFrame) -> pd.Series:
    """Count high-correlation pairs by (feature type, feature type, same/different channel)."""
    def kind(row) -> str:
        channel_1, feature_1 = row.feature_1.split("_", 1)
        channel_2, feature_2 = row.feature_2.split("_", 1)
        first, second = sorted([feature_1, feature_2])
        if channel_1 == channel_2:
            where = "même canal"
        elif "avg" in (channel_1, channel_2):
            where = "canal vs moyenne"
        else:
            where = "canaux différents"
        return f"{first} – {second} ({where})"
    if pairs.empty:
        return pd.Series(dtype=int)
    return pairs.apply(kind, axis=1).value_counts()


def preprocessing_comparison(
    mode: str, statistics: pd.DataFrame, spearman: pd.DataFrame, reference: dict
) -> list[str]:
    """Report section comparing the filtered dataset with the unfiltered reference."""
    raw = reference["statistics"].set_index("feature")
    new = statistics.set_index("feature")
    lines = [
        "## Effet du prétraitement (comparaison avec les données non filtrées)",
        "",
        "Le filtre passe-bande 0,5–40 Hz retire l’offset continu propre à chaque enregistrement (< 0,5 Hz) et le bruit secteur à 50 Hz, très présent dans le signal brut (pic à 50 Hz 300 à 35 000 fois au-dessus du niveau 30–45 Hz selon les enregistrements). Illustration : [avant/après filtrage](../outputs/figures/" + mode + "/preprocessing/211104B_D_seizure3_filtering.png). Le rapport non filtré complet reste disponible : [PART_VII_ANALYSIS_" + mode.upper() + "_RAW.md](PART_VII_ANALYSIS_" + mode.upper() + "_RAW.md).",
        "",
        "### Écart AUC globale / intra-enregistrement des variables sensibles à l’offset",
        "",
        "| Caractéristique | Brut : globale | Brut : intra | Brut : écart | Filtré : globale | Filtré : intra | Filtré : écart |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in DC_FEATURES:
        a, b = raw.loc[feature], new.loc[feature]
        lines.append(
            f"| {feature} | {a.separation_auc:.3f} | {a.within_recording_separation_auc:.3f} | "
            f"{a.within_recording_separation_auc - a.separation_auc:+.3f} | {b.separation_auc:.3f} | "
            f"{b.within_recording_separation_auc:.3f} | {b.within_recording_separation_auc - b.separation_auc:+.3f} |"
        )
    gaps_raw = (raw.within_recording_separation_auc - raw.separation_auc).loc[DC_FEATURES]
    gaps_new = (new.within_recording_separation_auc - new.separation_auc).loc[DC_FEATURES]
    lines += [
        "",
        f"Écart moyen sur ces cinq variables : {gaps_raw.mean():+.3f} avant filtrage, {gaps_new.mean():+.3f} après. "
        "Après filtrage, la moyenne de chaque fenêtre est proche de 0 : `mean` n’apporte plus d’information (elle est conservée car demandée par l’énoncé), "
        "`rms` devient presque identique à `std`, et `min`/`max` mesurent l’amplitude des oscillations au lieu de l’offset.",
        "",
        "### Caractéristiques les plus discriminantes avant / après filtrage (AUC globale)",
        "",
        "| Rang | Brut | AUC | Filtré | AUC |",
        "|---:|---|---:|---|---:|",
    ]
    raw_top = reference["statistics"].head(10)
    new_top = statistics.head(10)
    for rank, (a, b) in enumerate(zip(raw_top.itertuples(index=False), new_top.itertuples(index=False)), 1):
        lines.append(f"| {rank} | {a.feature} | {a.separation_auc:.3f} | {b.feature} | {b.separation_auc:.3f} |")
    raw_groups = redundancy_groups(reference["spearman"], 0.95)
    new_groups = redundancy_groups(spearman, 0.95)
    raw_pairs = high_correlation_pairs(reference["spearman"], 0.95)
    new_pairs = high_correlation_pairs(spearman, 0.95)
    raw_keys = set(zip(raw_pairs.feature_1, raw_pairs.feature_2))
    new_keys = set(zip(new_pairs.feature_1, new_pairs.feature_2))
    gained = new_pairs.loc[[pair not in raw_keys for pair in zip(new_pairs.feature_1, new_pairs.feature_2)]]
    lost = raw_pairs.loc[[pair not in new_keys for pair in zip(raw_pairs.feature_1, raw_pairs.feature_2)]]
    lines += [
        "",
        "### Groupes redondants avant / après filtrage (|ρ de Spearman| ≥ 0,95)",
        "",
        f"Brut : {len(raw_pairs)} paires, {len(raw_groups)} groupes (tailles {', '.join(str(len(group)) for group in raw_groups)}). "
        f"Filtré : {len(new_pairs)} paires, {len(new_groups)} groupes (tailles {', '.join(str(len(group)) for group in new_groups)}).",
        "",
        "| Type de paire | Apparues après filtrage | Disparues après filtrage |",
        "|---|---:|---:|",
    ]
    gained_types, lost_types = pair_types(gained), pair_types(lost)
    for kind in sorted(set(gained_types.index) | set(lost_types.index)):
        lines.append(f"| {kind} | {int(gained_types.get(kind, 0))} | {int(lost_types.get(kind, 0))} |")
    lines.append("")
    return lines


def write_report(
    mode: str,
    suffix: str,
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    pairs: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    average_columns: list[str],
    reference: dict | None,
) -> None:
    selected = statistics.set_index("feature").loc[[feature for feature in Q10_FEATURES if feature in statistics["feature"].values]].reset_index()
    top = statistics.head(15)
    average_spearman = spearman.loc[average_columns, average_columns]
    average_pearson = pearson.loc[average_columns, average_columns]
    average_pairs = high_correlation_pairs(average_spearman, 0.90)
    groups = redundancy_groups(spearman, 0.95)
    spearman_pairs = high_correlation_pairs(spearman, 0.95)
    within_top = statistics.sort_values("within_recording_separation_auc", ascending=False).head(10)
    dc = statistics.set_index("feature").loc[DC_FEATURES]
    variant = mode + suffix
    figure_base = "../outputs/figures/" + variant + "/part_vii/"
    preprocessing = "signal non filtré" if suffix else "signal filtré passe-bande 0,5–40 Hz"
    lines = [
        f"# Partie VII — Analyse exploratoire ({mode}, {preprocessing})",
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
    if suffix:
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
            "Un modèle évalué sur des enregistrements non vus risque d’utiliser ces variables pour reconnaître l’enregistrement plutôt que la crise. `std`, `variance`, `amplitude` (crête à crête) et les puissances spectrales (calculées après retrait de la moyenne) ne sont pas affectées. Ce constat motive le filtrage passe-bande du jeu de données principal.",
        ]
    else:
        lines += [
            "",
            "`mean` est quasi nulle après filtrage : son AUC reflète du bruit résiduel, pas une information utile. Elle est conservée car demandée par l’énoncé. L’effet du filtre sur les variables sensibles à l’offset est détaillé dans la dernière section.",
        ]
    lines += [
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
        f"Figures : corrélations moyennes [Spearman]({figure_base}correlation_average_features_spearman.png) et [Pearson]({figure_base}correlation_average_features.png); matrice complète [Spearman]({figure_base}correlation_all_features_spearman.png) et [Pearson]({figure_base}correlation_all_features.png). Tableaux : `outputs/feature_class_comparison_{variant}.csv`, `outputs/high_correlations_{variant}.csv` (Pearson), `outputs/high_correlations_spearman_{variant}.csv`, `outputs/feature_correlations_{variant}.csv` et `outputs/feature_correlations_spearman_{variant}.csv`.",
        "",
    ]
    if reference is not None:
        lines += preprocessing_comparison(mode, statistics, spearman, reference)
    (ROOT / f"docs/PART_VII_ANALYSIS_{variant.upper()}.md").write_text("\n".join(lines))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--annotations", choices=["official", "provisional"], required=True)
    parser.add_argument("--preprocess", choices=["bandpass", "none"], default="bandpass")
    args = parser.parse_args()
    mode = args.annotations
    suffix = "" if args.preprocess == "bandpass" else "_raw"
    variant = mode + suffix
    path = dataset_path(mode, suffix)
    frame = pd.read_parquet(path)
    feature_columns = [column for column in frame.columns if column not in METADATA]
    if frame[feature_columns].isna().any().any():
        raise ValueError("Feature dataset contains missing values")
    output_dir = ROOT / f"outputs/figures/{variant}/part_vii"
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics = class_statistics(frame, feature_columns)
    correlation = frame[feature_columns].corr(method="pearson")
    spearman = frame[feature_columns].corr(method="spearman")
    average_columns = [column for column in feature_columns if column.startswith("avg_")]
    pairs = high_correlation_pairs(correlation)

    statistics.to_csv(ROOT / f"outputs/feature_class_comparison_{variant}.csv", index=False)
    pairs.to_csv(ROOT / f"outputs/high_correlations_{variant}.csv", index=False)
    high_correlation_pairs(spearman).to_csv(ROOT / f"outputs/high_correlations_spearman_{variant}.csv", index=False)
    correlation.to_csv(ROOT / f"outputs/feature_correlations_{variant}.csv")
    spearman.to_csv(ROOT / f"outputs/feature_correlations_spearman_{variant}.csv")
    boxplots(frame, output_dir / "class_comparisons.png")
    heatmaps(correlation, average_columns, output_dir, "pearson")
    heatmaps(spearman, average_columns, output_dir, "spearman")
    reference = None
    raw_statistics = ROOT / f"outputs/feature_class_comparison_{mode}_raw.csv"
    raw_spearman = ROOT / f"outputs/feature_correlations_spearman_{mode}_raw.csv"
    if not suffix and raw_statistics.exists() and raw_spearman.exists():
        reference = {
            "statistics": pd.read_csv(raw_statistics),
            "spearman": pd.read_csv(raw_spearman, index_col=0),
        }
    elif not suffix:
        print("Unfiltered reference not found; run with --preprocess none first for the comparison section.")
    write_report(mode, suffix, frame, statistics, pairs, correlation, spearman, average_columns, reference)
    print(f"Part VII report: {ROOT / f'docs/PART_VII_ANALYSIS_{variant.upper()}.md'}")
    columns = ["feature", "separation_auc", "within_recording_separation_auc", "cohens_d"]
    print(f"Top discriminants:\n{statistics.head(10)[columns].to_string(index=False)}")
    print(f"Highly correlated pairs |r|>=0.95: {len(pairs)}")


if __name__ == "__main__":
    main()
