"""Parts X–XI: threshold selection, detected intervals, per-recording figures and error analysis."""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import MIN_DURATION_S, MIN_GAP_S, detect_all, event_metrics, sweep_thresholds


PREDICTIONS = ROOT / "outputs/predictions_oof.parquet"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
APP_DATA = ROOT / "data/processed/app"
FIGURES = ROOT / "outputs/figures/official/detection"
# Same semantic colors as the app.
REAL = "#34D399"
DETECTED = "#F87171"
THRESHOLD = "#A78BFA"
EVENT = "#FBBF24"
TRACE = "#2a78d6"
INK = "#52514e"
EXAMPLE_CHANNELS = ["Fp1", "C3", "O1"]


def choose_threshold(sweep: pd.DataFrame) -> float:
    """Highest event-level F1; ties go to the lowest threshold (higher recall)."""
    best = sweep["f1"].max()
    return float(sweep.loc[np.isclose(sweep["f1"], best), "threshold"].min())


def interval_table(detected: pd.DataFrame, matches: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for recording in sorted(annotations["recording"].unique()):
        found = matches.loc[(matches.recording == recording) & (matches.status != "FP")].reset_index(drop=True)
        for index, row in enumerate(found.itertuples(index=False), 1):
            rows.append(
                {
                    "recording": recording, "type": "real", "number": index,
                    "start_s": row.real_start_s, "end_s": row.real_end_s, "duration_s": row.real_end_s - row.real_start_s,
                    "status": "found" if row.status == "TP" else "missed",
                    "onset_error_s": row.onset_error_s, "offset_error_s": row.offset_error_s, "max_proba": row.max_proba,
                }
            )
        false_starts = set(matches.loc[(matches.recording == recording) & (matches.status == "FP"), "detected_start_s"])
        for index, row in enumerate(detected.loc[detected.recording == recording].itertuples(index=False), 1):
            rows.append(
                {
                    "recording": recording, "type": "detected", "number": index,
                    "start_s": row.start_s, "end_s": row.end_s, "duration_s": row.duration_s,
                    "status": "false alarm" if row.start_s in false_starts else "true detection",
                    "onset_error_s": np.nan, "offset_error_s": np.nan, "max_proba": row.max_proba,
                }
            )
    return pd.DataFrame(rows)


def listing(intervals: pd.DataFrame) -> list[str]:
    """Q14 text output, one block per recording."""
    lines = []
    for recording, group in intervals.groupby("recording", sort=True):
        lines.append(f"{recording}")
        detected = group.loc[group.type == "detected"]
        if detected.empty:
            lines.append("  Aucune crise détectée")
        for row in detected.itertuples(index=False):
            note = "" if row.status == "true detection" else "  (fausse alarme)"
            lines.append(f"  Crise détectée {row.number} : {row.start_s:.1f} s → {row.end_s:.1f} s{note}")
        for row in group.loc[group.type == "real"].itertuples(index=False):
            note = "" if row.status == "found" else "  (manquée)"
            lines.append(f"  Crise réelle {row.number} : {row.start_s:.1f} s → {row.end_s:.1f} s{note}")
        lines.append("")
    return lines


def shade(axis, intervals, color: str, alpha: float = 0.25) -> None:
    for start, end in intervals:
        axis.axvspan(start, end, color=color, alpha=alpha, linewidth=0)


def strips(axis, real, detected) -> None:
    """Two labeled strips: real seizures above detections, so neither hides the other."""
    for start, end in real:
        axis.broken_barh([(start, end - start)], (1.1, 0.8), color=REAL)
    for start, end in detected:
        axis.broken_barh([(start, end - start)], (0.1, 0.8), color=DETECTED)
    axis.set_ylim(0, 2)
    axis.set_yticks([0.5, 1.5], ["Détecté", "Réel"])
    axis.tick_params(axis="x", labelbottom=False, length=0)
    for spine in ("top", "right", "bottom"):
        axis.spines[spine].set_visible(False)


def clean(axis) -> None:
    axis.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def recording_figure(recording: str, predictions: pd.DataFrame, real, detected, threshold: float, counts: dict) -> None:
    signal = pd.read_parquet(APP_DATA / f"signals/{recording}.parquet", columns=["Time", "C3"])
    figure, axes = plt.subplots(
        3, 1, figsize=(16, 6.5), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [0.5, 3, 2]}
    )
    strips(axes[0], real, detected)
    axes[0].set_title(
        f"{recording} — {counts['tp']} crise(s) trouvée(s), {counts['fn']} manquée(s), {counts['fp']} fausse(s) alarme(s) "
        f"(seuil {threshold:.2f}, prédictions hors-pli)",
        fontsize=12,
    )
    axes[1].plot(signal["Time"], signal["C3"], color=TRACE, linewidth=0.4, rasterized=True)
    limit = float(np.percentile(np.abs(signal["C3"]), 99.9)) * 1.2
    axes[1].set_ylim(-limit, limit)
    axes[1].set_ylabel("C3 filtré\n(unité non renseignée)")
    shade(axes[1], real, REAL)
    clean(axes[1])
    group = predictions.loc[predictions.recording == recording]
    centers = (group["start_s"] + group["end_s"]) / 2
    axes[2].plot(centers, group["proba"], color=INK, linewidth=0.8)
    axes[2].axhline(threshold, color=THRESHOLD, linewidth=1.4, linestyle="--", label=f"Seuil {threshold:.2f}")
    shade(axes[2], real, REAL)
    shade(axes[2], detected, DETECTED, 0.18)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("Probabilité de crise")
    axes[2].set_xlabel("Temps depuis le début de l’enregistrement (s)")
    axes[2].legend(loc="upper right", frameon=False)
    axes[2].set_xlim(0, float(signal["Time"].iloc[-1]))
    clean(axes[2])
    figure.savefig(FIGURES / f"{recording}.png", dpi=130)
    plt.close(figure)


def sweep_figure(sweep: pd.DataFrame, threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for column, label, color in [("recall", "Recall", "#2a78d6"), ("precision", "Precision", "#eb6834"), ("f1", "F1", "#1baf7a")]:
        axis.plot(sweep["threshold"], sweep[column], color=color, linewidth=2, label=label)
        axis.text(sweep["threshold"].iloc[-1] + 0.01, sweep[column].iloc[-1], label, color=INK, va="center", fontsize=9)
    axis.axvline(threshold, color=THRESHOLD, linewidth=1.4, linestyle="--", label=f"Seuil retenu {threshold:.2f}")
    axis.set_xlabel("Seuil sur la probabilité")
    axis.set_ylabel("Score au niveau événement")
    axis.set_ylim(0, 1.02)
    axis.set_xlim(0, 1.06)
    axis.set_title("Choix du seuil sur les prédictions hors-pli (niveau événement)")
    axis.legend(loc="lower center", ncol=4, frameon=False)
    clean(axis)
    figure.savefig(FIGURES / "threshold_sweep.png", dpi=150)
    plt.close(figure)


def pick_examples(matches: pd.DataFrame, predictions: pd.DataFrame, detected: pd.DataFrame, annotations: pd.DataFrame, threshold: float) -> list[dict]:
    examples = []
    found = matches.loc[matches.status == "TP"].assign(error=lambda f: f.onset_error_s.abs() + f.offset_error_s.abs())
    found = found.loc[(found.real_end_s - found.real_start_s) >= 8]
    if len(found):
        row = found.sort_values("error").iloc[0]
        examples.append({"kind": "TP — crise détectée", "recording": row.recording, "start": row.real_start_s, "end": row.real_end_s})
    missed = matches.loc[matches.status == "FN"].assign(duration=lambda f: f.real_end_s - f.real_start_s)
    if len(missed):
        row = missed.sort_values("duration", ascending=False).iloc[0]
        examples.append({"kind": "FN — crise manquée", "recording": row.recording, "start": row.real_start_s, "end": row.real_end_s})
    false = matches.loc[matches.status == "FP"].assign(duration=lambda f: f.detected_end_s - f.detected_start_s)
    if len(false):
        row = false.sort_values(["max_proba", "duration"], ascending=False).iloc[0]
        examples.append({"kind": "FP — fausse alarme", "recording": row.recording, "start": row.detected_start_s, "end": row.detected_end_s})
    # TN: the quietest 10 s stretch far from any real or detected event in the TP example's recording.
    recording = examples[0]["recording"] if examples else predictions["recording"].iloc[0]
    group = predictions.loc[predictions.recording == recording]
    busy = pd.concat(
        [annotations.loc[annotations.recording == recording, ["start_s", "end_s"]], detected.loc[detected.recording == recording, ["start_s", "end_s"]]]
    )
    candidates = group.loc[(group.proba < threshold) & (group.true_class == 0)]
    for start in candidates["start_s"].sort_values().iloc[len(candidates) // 3 :]:
        if not ((busy.start_s < start + 40) & (busy.end_s > start - 30)).any():
            examples.append({"kind": "TN — activité normale", "recording": recording, "start": float(start), "end": float(start) + 10})
            break
    return examples


def examples_figure(examples: list[dict], predictions: pd.DataFrame, detected: pd.DataFrame, annotations: pd.DataFrame, threshold: float) -> None:
    rows = len(EXAMPLE_CHANNELS) + 1
    figure, axes = plt.subplots(rows, len(examples), figsize=(5.2 * len(examples), 8.5), constrained_layout=True, squeeze=False, sharex="col")
    for column, example in enumerate(examples):
        recording = example["recording"]
        low, high = example["start"] - 10, example["end"] + 10
        signal = pd.read_parquet(APP_DATA / f"signals/{recording}.parquet", columns=["Time", *EXAMPLE_CHANNELS])
        signal = signal.loc[(signal.Time >= low) & (signal.Time <= high)]
        real = annotations.loc[(annotations.recording == recording) & (annotations.end_s > low) & (annotations.start_s < high), ["start_s", "end_s"]].to_numpy()
        found = detected.loc[(detected.recording == recording) & (detected.end_s > low) & (detected.start_s < high), ["start_s", "end_s"]].to_numpy()
        for row, channel in enumerate(EXAMPLE_CHANNELS):
            axis = axes[row, column]
            axis.plot(signal["Time"], signal[channel], color=TRACE, linewidth=0.7)
            shade(axis, real, REAL)
            shade(axis, found, DETECTED, 0.18)
            axis.set_ylabel(channel)
            clean(axis)
            if row == 0:
                axis.set_title(f"{example['kind']}\n{recording}, {example['start']:.0f}–{example['end']:.0f} s", fontsize=11)
        axis = axes[-1, column]
        group = predictions.loc[(predictions.recording == recording) & (predictions.end_s > low) & (predictions.start_s < high)]
        axis.step((group.start_s + group.end_s) / 2, group.proba, where="mid", color=INK, linewidth=1.2)
        axis.axhline(threshold, color=THRESHOLD, linewidth=1.4, linestyle="--")
        shade(axis, real, REAL)
        shade(axis, found, DETECTED, 0.18)
        axis.set_ylim(0, 1.02)
        axis.set_xlim(low, high)
        axis.set_ylabel("Probabilité")
        axis.set_xlabel("Temps (s)")
        clean(axis)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REAL, alpha=0.4, label="Crise réelle (annotation)"),
        plt.Rectangle((0, 0), 1, 1, color=DETECTED, alpha=0.3, label="Crise détectée"),
        plt.Line2D([0], [0], color=THRESHOLD, linestyle="--", label=f"Seuil {threshold:.2f}"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    figure.savefig(FIGURES / "error_examples.png", dpi=140)
    plt.close(figure)


def nearby_notes(matches: pd.DataFrame, events: pd.DataFrame, status: str, margin: float = 10.0) -> pd.DataFrame:
    """Technician notes within ``margin`` seconds of each FP detection or FN seizure."""
    rows = []
    prefix = "detected" if status == "FP" else "real"
    for row in matches.loc[matches.status == status].itertuples(index=False):
        start, end = getattr(row, f"{prefix}_start_s"), getattr(row, f"{prefix}_end_s")
        near = events.loc[(events.recording == row.recording) & (events.time_s >= start - margin) & (events.time_s <= end + margin)]
        rows.append(
            {
                "recording": row.recording, "start_s": start, "end_s": end, "max_proba": row.max_proba,
                "categories": sorted(set(near["category"])), "labels": "; ".join(near["label"].str.slice(0, 30)),
            }
        )
    return pd.DataFrame(rows)


def in_hpn(events: pd.DataFrame, recording: str, time_s: float) -> bool:
    """True when ``time_s`` falls between the first and last HPN marker of the recording (+60 s)."""
    hpn = events.loc[(events.recording == recording) & (events.category == "HPN"), "time_s"]
    return bool(len(hpn)) and hpn.min() <= time_s <= hpn.max() + 60


def write_report(
    threshold: float, sweep: pd.DataFrame, summary: dict, intervals: pd.DataFrame, matches: pd.DataFrame,
    per_recording: pd.DataFrame, false_notes: pd.DataFrame, missed_notes: pd.DataFrame, metadata: pd.DataFrame, examples: list[dict],
) -> None:
    event, window, half = summary["event"], summary["window"], summary["event_at_0.50"]
    plateau = sweep.loc[sweep.f1 >= event["f1"] - 0.02, "threshold"]
    ages = metadata.set_index("recording")["age"]
    false = matches.loc[matches.status == "FP"].assign(duration=lambda f: f.detected_end_s - f.detected_start_s)
    short = int((false.duration <= 3).sum())
    noted = false_notes.loc[false_notes.categories.map(lambda value: "Seizure note" in value)]
    artifact = false_notes.loc[false_notes.categories.map(lambda value: bool({"Artifact", "Movement"} & set(value)))]
    top_false = per_recording.sort_values("fp", ascending=False).head(3)
    lines = [
        "# Parties X–XI — Détection temporelle des crises et analyse des erreurs",
        "",
        "Modèle : Random Forest (hyperparamètres par défaut, entraînement non équilibré) sur le jeu officiel filtré. "
        "Toutes les probabilités utilisées ici sont **hors-pli** : `GroupKFold` à 5 plis par enregistrement (mêmes plis qu’en Partie IX), "
        "donc chaque fenêtre est prédite par un modèle qui n’a jamais vu son enregistrement. "
        "Le modèle final, entraîné sur les 21 enregistrements, est sauvegardé dans `models/rf_final.joblib` avec la liste des 144 caractéristiques; "
        "il sert aux futurs enregistrements, pas à l’évaluation.",
        "",
        "## Partie X — Des fenêtres aux intervalles",
        "",
        "Règles (`src/detection.py`) :",
        "",
        "1. une fenêtre est positive si sa probabilité ≥ seuil;",
        "2. les fenêtres positives consécutives forment un intervalle, du début de la première à la fin de la dernière;",
        f"3. deux intervalles séparés de moins de {MIN_GAP_S:g} s sont fusionnés;",
        f"4. les événements de moins de {MIN_DURATION_S:g} s sont supprimés. Une fenêtre isolée dure exactement 2 s : elle est donc conservée, et cette règle n’élimine rien avec des fenêtres de 2 s.",
        "",
        "Comparaison avec les annotations, au niveau événement : une crise réelle est **trouvée (TP)** si au moins une détection la chevauche, sinon **manquée (FN)**; "
        "une détection qui ne chevauche aucune crise réelle est une **fausse alarme (FP)**. "
        "Recall = crises trouvées / crises réelles. Precision = détections chevauchant une crise / détections. "
        "Erreur de début = début détecté − début réel (négatif : détecté en avance); erreur de fin = fin détectée − fin réelle.",
        "",
        "### Choix du seuil",
        "",
        f"Le seuil qui maximise le F1 événement sur les prédictions hors-pli est **{threshold:.2f}** (balayage de {sweep.threshold.min():.2f} à {sweep.threshold.max():.2f} par pas de 0,01) :",
        "",
        "| Seuil | Détections | Crises trouvées | Manquées | Fausses alarmes | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shown = sorted({0.30, 0.40, 0.50, 0.60, round(threshold, 2), 0.70, 0.80, 0.90})
    for value in shown:
        row = sweep.loc[np.isclose(sweep.threshold, value)].iloc[0]
        mark = "**" if np.isclose(value, threshold) else ""
        lines.append(
            f"| {mark}{value:.2f}{mark} | {int(row.detections)} | {int(row.tp)} | {int(row.fn)} | {int(row.fp)} | {row.precision:.3f} | {row.recall:.3f} | {mark}{row.f1:.3f}{mark} |"
        )
    lines += [
        "",
        f"Le maximum est peu marqué : le F1 reste à moins de 0,02 du maximum pour tous les seuils de {plateau.min():.2f} à {plateau.max():.2f} "
        f"(quelques événements d’écart sur {event['tp'] + event['fn']} crises). Dans ce plateau, le seuil règle surtout le compromis : "
        f"à 0,50, {half['tp']} crises trouvées et {half['fp']} fausses alarmes (F1 {half['f1']:.3f}); "
        f"à {threshold:.2f}, {event['tp']} trouvées et {event['fp']} fausses alarmes (F1 {event['f1']:.3f}). "
        "Le seuil est choisi et évalué sur les mêmes prédictions hors-pli : le F1 annoncé est donc légèrement optimiste. "
        "Figure : [threshold_sweep.png](../outputs/figures/official/detection/threshold_sweep.png).",
        "",
        f"### Résultats au seuil {threshold:.2f}",
        "",
        f"- Crises réelles : {event['tp'] + event['fn']}; trouvées : **{event['tp']}**; manquées : **{event['fn']}**.",
        f"- Détections : {summary['detections']}; fausses alarmes : **{event['fp']}** ({summary['false_alarms_per_hour']:.1f} par heure d’enregistrement).",
        f"- Precision {event['precision']:.3f}, recall {event['recall']:.3f}, **F1 {event['f1']:.3f}**.",
        f"- Erreur de début : moyenne {event['onset_error_mean_s']:+.2f} s, médiane absolue {event['onset_error_abs_median_s']:.1f} s. "
        f"Erreur de fin : moyenne {event['offset_error_mean_s']:+.2f} s, médiane absolue {event['offset_error_abs_median_s']:.1f} s.",
        "",
        "La résolution temporelle est limitée par le pas de 1 s des fenêtres. Une fenêtre est étiquetée crise dès qu’elle contient 1 s de crise : "
        "un détecteur parfait commencerait donc jusqu’à 1 s avant le début réel et finirait jusqu’à 1 s après la fin réelle. "
        + (
            "En pratique, les détections commencent plutôt en retard et finissent plutôt en avance : les fenêtres de bord, à moitié normales, "
            "reçoivent une probabilité plus faible et passent sous ce seuil élevé."
            if event["onset_error_mean_s"] > 0 and event["offset_error_mean_s"] < 0 else ""
        ),
        "",
        "| Enregistrement | Crises réelles | Trouvées | Manquées | Détections | Fausses alarmes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_recording.itertuples(index=False):
        lines.append(f"| {row.recording} | {row.tp + row.fn} | {row.tp} | {row.fn} | {row.detections} | {row.fp} |")
    lines += [
        "",
        "### Q14 — Intervalles détectés et intervalles réels",
        "",
        "Liste complète : `outputs/detected_intervals.csv` (une ligne par intervalle réel ou détecté) et `outputs/detected_intervals.txt`. "
        "Une figure par enregistrement (signal C3 filtré, crises réelles, détections, probabilité et seuil) : `outputs/figures/official/detection/<enregistrement>.png`. Extrait :",
        "",
        "```text",
    ]
    text = listing(intervals)
    shown_recordings = [examples[0]["recording"]] if examples else []
    shown_recordings += [recording for recording in ["190304A_E", "210427B_C"] if recording not in shown_recordings]
    for recording in shown_recordings:
        start = text.index(recording)
        lines += text[start : text.index("", start) + 1]
    lines += [
        "```",
        "",
        "CLI : `python scripts/detect.py --recording 190304A_E [--threshold 0.5]`.",
        "",
        "## Partie XI — Analyse des erreurs",
        "",
        f"### Comptages au seuil {threshold:.2f}",
        "",
        "| Niveau | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|",
        f"| Fenêtres (2 s) | {window['tp']:,} | {window['fp']:,} | {window['fn']:,} | {window['tn']:,} |",
        f"| Événements | {event['tp']} | {event['fp']} | {event['fn']} | non défini |",
        "",
        f"Au niveau fenêtre : precision {window['precision']:.3f}, recall {window['recall']:.3f}, F1 {window['f1']:.3f}. "
        "Au niveau événement, les vrais négatifs ne sont pas définis : il n’existe pas de liste d’« événements non-crise » à compter. "
        "On rapporte à la place le nombre de fausses alarmes par heure. "
        "Le recall événement est bien plus élevé que le recall fenêtre : il suffit qu’une partie de la crise soit détectée pour la trouver, "
        "alors que les fenêtres de bord (mi-crise, mi-normal) sont souvent manquées.",
        "",
        "### Exemples",
        "",
        "Figure : [error_examples.png](../outputs/figures/official/detection/error_examples.png) — Fp1, C3, O1 filtrés et probabilité, ±10 s autour de l’événement.",
        "",
    ]
    for example in examples:
        lines.append(f"- **{example['kind']}** : {example['recording']}, {example['start']:.1f} s → {example['end']:.1f} s.")
    lines += [
        "",
        f"### Crises manquées (FN) : {event['fn']}",
        "",
        "| Enregistrement | Âge | Début (s) | Fin (s) | Durée (s) | Probabilité max dans la crise | Notes du technicien à ±10 s |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in missed_notes.itertuples(index=False):
        lines.append(
            f"| {row.recording} | {ages[row.recording]} | {row.start_s:.1f} | {row.end_s:.1f} | {row.end_s - row.start_s:.1f} | {row.peak_proba:.2f} | {row.labels or '—'} |"
        )
    adult = missed_notes.loc[missed_notes.recording.map(ages) >= 17]
    lines += [
        "",
        "Causes probables :",
        "",
        f"- **Patients plus âgés, tracé différent.** {len(adult)} des {len(missed_notes)} crises manquées viennent de `190304A_E` (17 ans) et `191113A_D` (25 ans), "
        "les deux seuls patients de plus de 15 ans; les 19 autres ont entre 5 et 15 ans. Aucune de leurs 3 crises n’est trouvée, même au seuil 0,50. "
        "Le modèle, évalué hors-pli, n’a alors vu que des enfants à l’entraînement. Les notes du technicien y sont en outre hésitantes (« crise?? », « ABSNCE? »).",
        "- **Probabilité juste sous le seuil.** Les autres crises manquées ont une probabilité maximale proche du seuil : elles sont trouvées avec un seuil plus bas, au prix de fausses alarmes supplémentaires.",
        "- **Amplitude plus faible.** Dans l’exemple manqué de `191113A_D`, la décharge est visible sur C3 mais avec une amplitude deux à trois fois plus faible que dans l’exemple de crise détectée, "
        "et elle se distingue à peine du fond sur O1; or les caractéristiques dominantes du modèle sont des mesures d’amplitude.",
        "",
        f"### Fausses alarmes (FP) : {event['fp']}",
        "",
        f"- Durée : médiane {false.duration.median():.0f} s; {short} sur {len(false)} durent 3 s ou moins (une ou deux fenêtres).",
        f"- {len(noted)} sur {len(false)} ont une note « absence / crise / clonies » du technicien à ±10 s.",
        f"- {len(artifact)} sur {len(false)} ont une note d’artefact, de saturation ou de mouvement à ±10 s.",
        f"- {summary['false_in_hpn']} sur {len(false)} tombent pendant l’hyperpnée (HPN) ou dans la minute qui suit.",
        "- Enregistrements les plus touchés : " + ", ".join(f"`{row.recording}` ({row.fp})" for row in top_false.itertuples(index=False)) + ".",
        "",
        "| Enregistrement | Début (s) | Fin (s) | Probabilité max | Notes du technicien à ±10 s |",
        "|---|---:|---:|---:|---|",
    ]
    for row in false_notes.sort_values("max_proba", ascending=False).head(12).itertuples(index=False):
        lines.append(f"| {row.recording} | {row.start_s:.1f} | {row.end_s:.1f} | {row.max_proba:.2f} | {row.labels or '—'} |")
    lines += [
        "",
        "Causes probables :",
        "",
        "- **Crises probablement réelles mais non annotées.** Dans `210427B_C`, le fichier Excel ne contient qu’une crise (1031–1049 s), mais le tracé montre cinq bouffées "
        "de même aspect et de même amplitude (vers 280–293, 327–348, 817–835, 890–901 et 1031–1049 s; voir [210427B_C.png](../outputs/figures/official/detection/210427B_C.png)). "
        "Le technicien a noté « absence » à 290 s, 821 s et 831 s, et le modèle, qui n’a jamais vu cet enregistrement, détecte les cinq. "
        f"Ces {int(per_recording.set_index('recording').loc['210427B_C', 'fp'])} « fausses alarmes » sont vraisemblablement de vraies crises absentes du fichier : la precision réelle est sous-estimée. À confirmer avec le professeur.",
        "- **Décharges pendant les épreuves d’activation.** La fausse alarme la plus probable (`211027B_C`, 424–439 s, probabilité 1,00, pendant la stimulation lumineuse SLI) "
        "montre 15 s de décharge rythmique ample sur Fp1, C3 et O1 (figure des exemples). "
        f"{summary['false_in_hpn']} fausses alarmes sur {len(false)} surviennent pendant l’hyperpnée ou juste après. SLI et hyperpnée sont justement utilisées pour provoquer des décharges : "
        "une partie de ces événements peut être de l’activité pointe-onde que l’annotateur n’a pas retenue comme crise clinique. L’hyperpnée provoque aussi un ralentissement ample du tracé, proche en amplitude d’une crise.",
        f"- **Fragments d’une ou deux fenêtres.** {short} fausses alarmes sur {len(false)} durent 3 s ou moins. Une durée minimale supérieure à 2 s les supprimerait, au risque de manquer les crises très courtes.",
        f"- **Artefacts et mouvements : rôle limité ici.** Seulement {len(artifact)} fausse(s) alarme(s) sur {len(false)} se trouve(nt) près d’une note d’artefact ou de mouvement.",
        "",
        "Ces causes sont des hypothèses tirées des notes du technicien et de l’aspect du tracé; elles ne remplacent pas une relecture par un neurologue.",
        "",
        "Résultats : `outputs/detection_summary.json`, `outputs/detection_threshold_sweep.csv`, `outputs/event_matches.csv`. Reproduction : "
        "`python scripts/predict_oof.py`, `python scripts/prepare_app_data.py`, `python scripts/evaluate_detection.py`.",
        "",
    ]
    (ROOT / "docs/PARTS_X_XI.md").write_text("\n".join(lines))


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(PREDICTIONS)
    annotations = pd.read_csv(ANNOTATIONS)
    events = pd.read_csv(APP_DATA / "technician_events.csv")
    metadata = pd.read_csv(APP_DATA / "recording_metadata.csv")

    sweep = sweep_thresholds(predictions, annotations, np.round(np.arange(0.05, 0.955, 0.01), 2))
    threshold = choose_threshold(sweep)
    detected, matches = detect_all(predictions, annotations, threshold)
    event = event_metrics(matches, len(detected))
    _, matches_half = detect_all(predictions, annotations, 0.5)
    half = sweep.loc[np.isclose(sweep.threshold, 0.5)].iloc[0]

    predicted = (predictions["proba"] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(predictions["true_class"], predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    hours = metadata["duration_s"].sum() / 3600
    false_notes = nearby_notes(matches, events, "FP")
    missed_notes = nearby_notes(matches, events, "FN")
    peaks = []
    for row in missed_notes.itertuples(index=False):
        group = predictions.loc[(predictions.recording == row.recording) & (predictions.end_s > row.start_s) & (predictions.start_s < row.end_s)]
        peaks.append(float(group["proba"].max()))
    missed_notes["peak_proba"] = peaks
    false_in_hpn = sum(in_hpn(events, row.recording, row.start_s) for row in false_notes.itertuples(index=False))

    per_recording = (
        matches.groupby(["recording", "status"]).size().unstack(fill_value=0).reindex(columns=["TP", "FN", "FP"], fill_value=0)
        .rename(columns=str.lower).reset_index()
    )
    per_recording["detections"] = per_recording["recording"].map(detected.groupby("recording").size()).fillna(0).astype(int)

    summary = {
        "model": "RandomForestClassifier (default hyperparameters, unbalanced training)",
        "predictions": "out-of-fold, GroupKFold by recording (5 folds)",
        "threshold": threshold,
        "min_gap_s": MIN_GAP_S,
        "min_duration_s": MIN_DURATION_S,
        "detections": len(detected),
        "recorded_hours": hours,
        "false_alarms_per_hour": event["fp"] / hours,
        "false_in_hpn": int(false_in_hpn),
        "event": event,
        "event_at_0.50": {key: (int(half[key]) if key in ("tp", "fn", "fp") else float(half[key])) for key in ["tp", "fn", "fp", "precision", "recall", "f1"]},
        "window": {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision), "recall": float(recall), "f1": float(2 * precision * recall / (precision + recall)),
        },
    }
    intervals = interval_table(detected, matches, annotations)
    outputs = ROOT / "outputs"
    sweep.to_csv(outputs / "detection_threshold_sweep.csv", index=False)
    intervals.to_csv(outputs / "detected_intervals.csv", index=False)
    (outputs / "detected_intervals.txt").write_text("\n".join(listing(intervals)))
    matches.to_csv(outputs / "event_matches.csv", index=False)
    (outputs / "detection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    sweep_figure(sweep, threshold)
    for row in per_recording.itertuples(index=False):
        real = annotations.loc[annotations.recording == row.recording, ["start_s", "end_s"]].to_numpy()
        found = detected.loc[detected.recording == row.recording, ["start_s", "end_s"]].to_numpy()
        recording_figure(row.recording, predictions, real, found, threshold, {"tp": row.tp, "fn": row.fn, "fp": row.fp})
    examples = pick_examples(matches, predictions, detected, annotations, threshold)
    examples_figure(examples, predictions, detected, annotations, threshold)
    write_report(threshold, sweep, summary, intervals, matches, per_recording, false_notes, missed_notes, metadata, examples)
    print(f"Threshold {threshold:.2f}: event F1 {event['f1']:.3f} (TP {event['tp']}, FN {event['fn']}, FP {event['fp']})")
    print(f"Report: {ROOT / 'docs/PARTS_X_XI.md'}")


if __name__ == "__main__":
    main()
