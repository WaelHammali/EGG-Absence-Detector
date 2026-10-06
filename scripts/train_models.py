"""Parts VIII–IX: train and evaluate classifiers on the filtered official dataset."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import CHANNELS


SEED = 42
METADATA = {"recording", "subject", "window", "start_s", "end_s", "class"}
BASELINE = "Toujours 0"
METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "balanced_accuracy": "Balanced accuracy",
}
# Categorical slots 1–3 of the reference palette, in fixed order.
SETTING_COLORS = {
    "Split aléatoire, train non équilibré": "#2a78d6",
    "Split aléatoire, train sous-échantillonné": "#eb6834",
    "GroupKFold, train sous-échantillonné": "#1baf7a",
    "GroupKFold, train non équilibré": "#eda100",
}


def models() -> dict[str, Pipeline]:
    return {
        "Decision Tree": Pipeline([("model", DecisionTreeClassifier(random_state=SEED))]),
        "Random Forest": Pipeline([("model", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(random_state=SEED))]),
    }


def baseline() -> Pipeline:
    return Pipeline([("model", DummyClassifier(strategy="constant", constant=0))])


def undersample(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Indices keeping every class-1 sample and as many randomly drawn class-0 samples."""
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    chosen = np.random.RandomState(seed).choice(negatives, size=len(positives), replace=False)
    return np.sort(np.concatenate([positives, chosen]))


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_and_score(estimator, X_train, y_train, X_test, y_test) -> dict:
    started = time.perf_counter()
    fitted = clone(estimator).fit(X_train, y_train)
    result = scores(y_test, fitted.predict(X_test))
    result["seconds"] = time.perf_counter() - started
    return result


def random_split(X: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=SEED)
    balanced = train[undersample(y[train])]
    split = {
        "train_windows": len(train),
        "train_class_1": int(y[train].sum()),
        "balanced_train_windows": len(balanced),
        "balanced_train_class_1": int(y[balanced].sum()),
        "test_windows": len(test),
        "test_class_1": int(y[test].sum()),
    }
    rows = []
    for training, train_index in [("sous-échantillonné", balanced), ("non équilibré", train)]:
        for name, estimator in {BASELINE: baseline(), **models()}.items():
            result = fit_and_score(estimator, X[train_index], y[train_index], X[test], y[test])
            rows.append({"training": training, "model": name, **result})
            print(f"random/{training}/{name}: F1={result['f1']:.3f} recall={result['recall']:.3f} ({result['seconds']:.1f}s)", flush=True)
    return pd.DataFrame(rows), split


def group_kfold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[pd.DataFrame, list[dict]]:
    rows, folds = [], []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), 1):
        # Undersampling is drawn from this fold's training recordings only.
        balanced = train[undersample(y[train])]
        folds.append(
            {
                "fold": fold,
                "test_recordings": sorted(set(groups[test])),
                "test_windows": len(test),
                "test_class_1": int(y[test].sum()),
                "balanced_train_windows": len(balanced),
            }
        )
        for training, train_index in [("sous-échantillonné", balanced), ("non équilibré", train)]:
            for name, estimator in {BASELINE: baseline(), **models()}.items():
                result = fit_and_score(estimator, X[train_index], y[train_index], X[test], y[test])
                rows.append({"training": training, "fold": fold, "model": name, **result})
                print(f"group/{training}/fold {fold}/{name}: F1={result['f1']:.3f} recall={result['recall']:.3f}", flush=True)
    return pd.DataFrame(rows), folds


def plot_confusions(panels: list[tuple[str, dict]], path: Path, title: str, columns: int = 5) -> None:
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.3 * rows), constrained_layout=True, squeeze=False)
    for axis, (label, result) in zip(axes.flat, panels):
        matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
        share = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                axis.text(
                    j, i, f"{matrix[i, j]:,}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.6 else "#0b0b0b",
                )
        axis.set_xticks([0, 1], ["Prédit 0", "Prédit 1"])
        axis.set_yticks([0, 1], ["Vrai 0", "Vrai 1"])
        axis.set_title(label, fontsize=10)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(title + "\n(nombre de fenêtres et % de la classe réelle)", fontsize=12)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_settings(random_results: pd.DataFrame, group_summary: pd.DataFrame, path: Path) -> None:
    names = list(models())
    random_by = {training: random_results.loc[random_results.training == training].set_index("model") for training in ["non équilibré", "sous-échantillonné"]}
    group_by = {training: group_summary.xs(training, level="training") for training in ["non équilibré", "sous-échantillonné"]}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True, constrained_layout=True)
    width = 0.2
    x = np.arange(len(names))
    for axis, metric in zip(axes, ["precision", "recall", "f1"]):
        for offset, training in enumerate(["non équilibré", "sous-échantillonné"]):
            label = f"Split aléatoire, train {training}"
            axis.bar(x + (2 * offset - 1.5) * width, random_by[training].loc[names, metric], width - 0.03, color=SETTING_COLORS[label], label=label)
            label = f"GroupKFold, train {training}"
            axis.bar(
                x + (2 * offset - 0.5) * width, group_by[training].loc[names, f"{metric}_mean"], width - 0.03,
                color=SETTING_COLORS[label], label=label, yerr=group_by[training].loc[names, f"{metric}_std"],
                error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 2},
            )
        axis.set_xticks(x, names)
        axis.set_title(METRICS[metric] + " (classe 1)")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("Effet de l’équilibrage et du protocole d’évaluation (barres d’erreur : écart-type sur 5 plis)", fontsize=12)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def table(frame: pd.DataFrame, columns: list[str], formatter) -> list[str]:
    header = "| Modèle | " + " | ".join(METRICS.get(column, column) for column in columns) + " |"
    lines = [header, "|---|" + "---:|" * len(columns)]
    for model, row in frame.iterrows():
        lines.append(f"| {model} | " + " | ".join(formatter(row, column) for column in columns) + " |")
    return lines


def write_report(
    dataset: Path, split: dict, random_results: pd.DataFrame, group_results: pd.DataFrame, group_summary: pd.DataFrame, folds: list[dict]
) -> None:
    names = [BASELINE, *models()]
    learned = list(models())
    feature_count = len([column for column in pq.read_schema(dataset).names if column not in METADATA])
    random_by = {t: random_results.loc[random_results.training == t].set_index("model").loc[names] for t in ["sous-échantillonné", "non équilibré"]}
    group_by = {t: group_summary.xs(t, level="training").loc[names] for t in ["sous-échantillonné", "non équilibré"]}
    balanced, unbalanced = random_by["sous-échantillonné"], random_by["non équilibré"]
    group_balanced, group_unbalanced = group_by["sous-échantillonné"], group_by["non équilibré"]
    plain = lambda row, column: f"{row[column]:.3f}"
    count = lambda row, column: f"{int(row[column]):,}"
    base = balanced.loc[BASELINE]
    test_share = split["test_class_1"] / split["test_windows"]
    gap_balanced = balanced.loc[learned, "f1"] - group_balanced.loc[learned, "f1_mean"]
    gap_unbalanced = unbalanced.loc[learned, "f1"] - group_unbalanced.loc[learned, "f1_mean"]
    delta_recall = balanced.loc[learned, "recall"] - unbalanced.loc[learned, "recall"]
    delta_precision = balanced.loc[learned, "precision"] - unbalanced.loc[learned, "precision"]
    delta_f1 = balanced.loc[learned, "f1"] - unbalanced.loc[learned, "f1"]
    group_delta_recall = group_balanced.loc[learned, "recall_mean"] - group_unbalanced.loc[learned, "recall_mean"]
    group_delta_f1 = group_balanced.loc[learned, "f1_mean"] - group_unbalanced.loc[learned, "f1_mean"]
    others = pd.concat([gap_balanced, gap_unbalanced]).drop(index="Decision Tree")
    best_group = group_balanced.loc[learned, "f1_mean"].idxmax()
    best_group_unbalanced = group_unbalanced.loc[learned, "f1_mean"].idxmax()
    fold_f1 = group_results.loc[(group_results.training == "sous-échantillonné") & group_results.model.isin(learned)].groupby("fold")["f1"].mean()
    worst_fold, best_fold = int(fold_f1.idxmin()), int(fold_f1.idxmax())
    fold_recordings = {fold["fold"]: ", ".join(fold["test_recordings"]) for fold in folds}

    def group_table(summary: pd.DataFrame) -> list[str]:
        lines = ["| Modèle | " + " | ".join(METRICS.values()) + " |", "|---|" + "---:|" * len(METRICS)]
        for model in names:
            row = summary.loc[model]
            lines.append(f"| {model} | " + " | ".join(f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}" for m in METRICS) + " |")
        return lines

    lines = [
        "# Parties VIII–IX — Modèles de classification",
        "",
        f"Jeu de données : `{dataset.relative_to(ROOT)}` (annotations officielles, signal filtré passe-bande 0,5–40 Hz, {feature_count} caractéristiques sur {len(CHANNELS)} canaux EEG). "
        "Hyperparamètres par défaut de scikit-learn, sans réglage (prévu en Partie X). Arbre de décision et Random Forest sans normalisation; "
        "KNN et SVM avec `StandardScaler` dans le `Pipeline`, donc ajusté sur les seules données d’entraînement. `random_state=42` partout.",
        "",
        "## Protocole",
        "",
        f"1. Split stratifié 80/20 : {split['train_windows']:,} fenêtres d’entraînement ({split['train_class_1']:,} crises) et "
        f"{split['test_windows']:,} fenêtres de test ({split['test_class_1']:,} crises, {test_share:.2%}).",
        f"2. Sous-échantillonnage aléatoire du **train uniquement** : les {split['balanced_train_class_1']:,} fenêtres de classe 1 + "
        f"{split['balanced_train_windows'] - split['balanced_train_class_1']:,} fenêtres de classe 0 tirées au hasard ({split['balanced_train_windows']:,} fenêtres). "
        "Le test garde les proportions réelles : un test équilibré surestimerait la precision.",
        "3. Comparaison : mêmes modèles entraînés sur le train complet non équilibré.",
        "4. Évaluation honnête : `GroupKFold` à 5 plis par enregistrement, sous-échantillonnage tiré dans chaque pli d’entraînement seulement. "
        "La même validation avec un train non équilibré est ajoutée comme référence.",
        "",
        "Precision, Recall et F1 concernent la classe 1 (crise). La balanced accuracy (moyenne des rappels des deux classes) est ajoutée pour Q13.",
        "",
        "## Split aléatoire — train sous-échantillonné",
        "",
        *table(balanced, list(METRICS), plain),
        "",
        *table(balanced, ["tn", "fp", "fn", "tp"], count),
        "",
        "## Split aléatoire — train non équilibré (comparaison)",
        "",
        *table(unbalanced, list(METRICS), plain),
        "",
        *table(unbalanced, ["tn", "fp", "fn", "tp"], count),
        "",
        "tn/fp/fn/tp : vrais négatifs, faux positifs (fausses alarmes), faux négatifs (crises manquées), vrais positifs. "
        "Matrices de confusion : [split aléatoire](../outputs/figures/official/models/confusion_matrices_random_split.png).",
        "",
        "## Évaluation par enregistrement — GroupKFold (5 plis)",
        "",
        "### Train sous-échantillonné (protocole demandé)",
        "",
        *group_table(group_balanced),
        "",
        "### Train non équilibré (référence)",
        "",
        *group_table(group_unbalanced),
        "",
        "| Pli | Enregistrements de test | Fenêtres | Crises |",
        "|---:|---|---:|---:|",
    ]
    for fold in folds:
        lines.append(f"| {fold['fold']} | {', '.join(fold['test_recordings'])} | {fold['test_windows']:,} | {fold['test_class_1']:,} |")
    lines += [
        "",
        "F1 par pli (train sous-échantillonné) :",
        "",
        "| Modèle | " + " | ".join(f"Pli {fold['fold']}" for fold in folds) + " |",
        "|---|" + "---:|" * len(folds),
    ]
    for model in learned:
        values = group_results.loc[(group_results.model == model) & (group_results.training == "sous-échantillonné")].sort_values("fold")["f1"]
        lines.append(f"| {model} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += [
        "",
        "Matrices de confusion cumulées sur les 5 plis : [GroupKFold](../outputs/figures/official/models/confusion_matrices_groupkfold.png). "
        "Synthèse des quatre configurations : [metrics_by_setting.png](../outputs/figures/official/models/metrics_by_setting.png).",
        "",
        "### Split aléatoire vs GroupKFold",
        "",
        "| Modèle | F1 aléatoire (équilibré) | F1 GroupKFold (équilibré) | Écart | F1 aléatoire (non équilibré) | F1 GroupKFold (non équilibré) | Écart |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in learned:
        lines.append(
            f"| {model} | {balanced.loc[model, 'f1']:.3f} | {group_balanced.loc[model, 'f1_mean']:.3f} | {gap_balanced[model]:+.3f} | "
            f"{unbalanced.loc[model, 'f1']:.3f} | {group_unbalanced.loc[model, 'f1_mean']:.3f} | {gap_unbalanced[model]:+.3f} |"
        )
    lines += [
        "",
        f"Le split aléatoire est optimiste dans les deux cas : F1 supérieur de {gap_balanced.mean():+.3f} en moyenne avec le train équilibré, "
        f"et de {gap_unbalanced.mean():+.3f} avec le train non équilibré. Deux raisons :",
        "",
        "- **Fuite entre fenêtres voisines.** Avec 50 % de recouvrement, deux fenêtres consécutives partagent une seconde de signal, et les fenêtres d’une même crise se ressemblent. "
        "Un split aléatoire place ces quasi-copies à la fois dans le train et dans le test.",
        "- **Spécificités de l’enregistrement.** Niveau d’amplitude de fond, électrodes, bruit : le modèle apprend aussi ce qui caractérise chaque enregistrement, ce qui ne se généralise pas à un nouveau patient.",
        "",
        f"Hors arbre de décision, l’écart est du même ordre avec ou sans équilibrage "
        f"({others.min():+.3f} à {others.max():+.3f}). Il est le plus fort pour l’arbre de décision non équilibré ({gap_unbalanced['Decision Tree']:+.3f}) : "
        "un arbre unique et profond mémorise les fenêtres voisines, donc profite le plus de la fuite. "
        "GroupKFold teste sur des enregistrements jamais vus, ce qui correspond à l’usage réel. Les identifiants patients n’étant pas confirmés, un même patient peut toutefois apparaître dans deux enregistrements : "
        "même GroupKFold peut rester légèrement optimiste.",
        "",
        f"La variabilité entre plis est l’autre enseignement : le F1 moyen des quatre modèles va de {fold_f1.min():.3f} (pli {worst_fold} : {fold_recordings[worst_fold]}) "
        f"à {fold_f1.max():.3f} (pli {best_fold} : {fold_recordings[best_fold]}). Un seul split aléatoire ne montre pas cette dépendance aux enregistrements testés.",
        "",
        "## Q12 — Pourquoi l’accuracy est trompeuse",
        "",
        f"La classe 1 ne représente que {test_share:.2%} des fenêtres de test. Le modèle « {BASELINE} », qui ne détecte aucune crise, obtient "
        f"une accuracy de **{base['accuracy']:.3f}**, avec un recall, une precision et un F1 de 0. "
        f"Il dépasse même en accuracy l’arbre de décision entraîné sur le train équilibré ({balanced.loc['Decision Tree', 'accuracy']:.3f}), qui détecte pourtant "
        f"{balanced.loc['Decision Tree', 'recall']:.0%} des crises. L’accuracy est dominée par la classe majoritaire et ne dit rien des crises manquées.",
        "",
        "## Q13 — Quelle métrique privilégier",
        "",
        "- **Recall (sensibilité)** : part des fenêtres de crise détectées. C’est la priorité clinique : une crise manquée est l’erreur la plus coûteuse.",
        "- **Precision** : part des alarmes qui sont de vraies crises. Elle mesure le coût des fausses alarmes, nombreuses dès que la classe 0 domine.",
        "- **F1** : moyenne harmonique des deux, nulle si l’un est nul. C’est la métrique principale pour comparer les modèles, toujours accompagnée du recall et de la precision.",
        f"- La **balanced accuracy** corrige aussi le déséquilibre ({base['balanced_accuracy']:.3f} pour « {BASELINE} », soit le hasard), mais elle est peu sensible aux fausses alarmes : "
        "quelques centaines de faux positifs pèsent peu face à plus de 6 000 vraies fenêtres de classe 0.",
        "",
        f"En évaluation par enregistrement, le meilleur F1 est obtenu par **{best_group}** avec le train équilibré "
        f"({group_balanced.loc[best_group, 'f1_mean']:.3f} ± {group_balanced.loc[best_group, 'f1_std']:.3f}) et par **{best_group_unbalanced}** avec le train non équilibré "
        f"({group_unbalanced.loc[best_group_unbalanced, 'f1_mean']:.3f} ± {group_unbalanced.loc[best_group_unbalanced, 'f1_std']:.3f}).",
        "",
        "## Effet de l’équilibrage (compromis recall / precision)",
        "",
        "Split aléatoire :",
        "",
        "| Modèle | Recall non équilibré | Recall équilibré | Δ recall | Precision non équilibrée | Precision équilibrée | Δ precision | Δ F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in learned:
        lines.append(
            f"| {model} | {unbalanced.loc[model, 'recall']:.3f} | {balanced.loc[model, 'recall']:.3f} | {delta_recall[model]:+.3f} | "
            f"{unbalanced.loc[model, 'precision']:.3f} | {balanced.loc[model, 'precision']:.3f} | {delta_precision[model]:+.3f} | {delta_f1[model]:+.3f} |"
        )
    lines += [
        "",
        f"Le sous-échantillonnage fait passer la classe 1 de {split['train_class_1'] / split['train_windows']:.2%} à 50 % du train. "
        f"Le modèle apprend une frontière plus favorable aux crises : le recall augmente en moyenne de {delta_recall.mean():+.3f}, "
        f"mais la precision chute de {delta_precision.mean():+.3f}, et le F1 baisse de {delta_f1.mean():+.3f}. "
        f"La cause : le modèle s’entraîne avec 50 % de crises mais est testé avec {test_share:.0%}. Un faible taux de fausses alarmes sur la classe 0, très nombreuse, "
        + (
            "produit alors davantage de faux positifs que de vrais positifs pour chacun des quatre modèles. "
            if (balanced.loc[learned, "fp"] > balanced.loc[learned, "tp"]).all()
            else "produit alors un nombre de faux positifs comparable à celui des vraies détections. "
        )
        + f"Le sous-échantillonnage jette aussi {(split['train_windows'] - split['balanced_train_windows']) / split['train_windows']:.0%} des fenêtres d’entraînement, "
        "et donc de l’information sur la variabilité de l’activité normale.",
        "",
        f"En GroupKFold, la même tendance se confirme : Δ recall moyen {group_delta_recall.mean():+.3f}, Δ F1 moyen {group_delta_f1.mean():+.3f}.",
        "",
        "Le choix dépend donc du coût des erreurs. Pour un outil d’aide à la lecture d’EEG, manquer une crise est généralement plus grave qu’une fausse alarme qu’un neurologue peut écarter : "
        "c’est l’argument pour l’équilibrage. Si l’on optimise le F1, le train non équilibré est meilleur ici. "
        "Le seuil de décision peut aussi être déplacé après l’entraînement pour choisir un compromis recall / precision; ce réglage relève de la Partie X.",
        "",
        "Résultats bruts : `outputs/models_random_split.csv`, `outputs/models_groupkfold_folds.csv`, `outputs/models_groupkfold_summary.csv` et `outputs/models_summary.json`. "
        "Reproduction : `python scripts/train_models.py`.",
        "",
    ]
    (ROOT / "docs/PARTS_VIII_IX.md").write_text("\n".join(lines))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data/processed/windows_features_official.parquet"))
    args = parser.parse_args()
    dataset = Path(args.dataset)
    frame = pd.read_parquet(dataset)
    features = [column for column in frame.columns if column not in METADATA]
    X = frame[features].to_numpy(dtype=np.float64)
    y = frame["class"].to_numpy(dtype=np.int64)
    groups = frame["recording"].to_numpy()
    print(f"{dataset.name}: {X.shape[0]:,} windows, {X.shape[1]} features, class 1 = {y.mean():.2%}", flush=True)

    random_results, split = random_split(X, y)
    group_results, folds = group_kfold(X, y, groups)
    summary = group_results.groupby(["training", "model"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summed = group_results.groupby(["training", "model"])[["tn", "fp", "fn", "tp"]].sum()

    outputs = ROOT / "outputs"
    figures = outputs / "figures/official/models"
    figures.mkdir(parents=True, exist_ok=True)
    random_results.to_csv(outputs / "models_random_split.csv", index=False)
    group_results.to_csv(outputs / "models_groupkfold_folds.csv", index=False)
    summary.join(summed).to_csv(outputs / "models_groupkfold_summary.csv")
    (outputs / "models_summary.json").write_text(
        json.dumps({"dataset": str(dataset.relative_to(ROOT)), "features": len(features), "seed": SEED, "random_split": split, "group_folds": folds}, indent=2) + "\n"
    )

    names = [BASELINE, *models()]
    panels = []
    for training in ["sous-échantillonné", "non équilibré"]:
        subset = random_results.loc[random_results.training == training].set_index("model")
        panels += [(f"{name}\ntrain {training}", subset.loc[name]) for name in names]
    plot_confusions(panels, figures / "confusion_matrices_random_split.png", "Split aléatoire 80/20 — test non équilibré")
    plot_confusions(
        [(f"{name}\ntrain {training}", summed.loc[(training, name)]) for training in ["sous-échantillonné", "non équilibré"] for name in names],
        figures / "confusion_matrices_groupkfold.png",
        "GroupKFold par enregistrement — matrices cumulées sur 5 plis",
    )
    plot_settings(random_results, summary, figures / "metrics_by_setting.png")
    write_report(dataset, split, random_results, group_results, summary, folds)
    print(f"Report: {ROOT / 'docs/PARTS_VIII_IX.md'}")


if __name__ == "__main__":
    main()
