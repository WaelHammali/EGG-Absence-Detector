"""Write docs/MODEL_COMPARISON.md: the 8 registered models side by side.

Reads models/registry.json, the out-of-fold predictions and the app data (technician notes and
reference channels). Run after scripts/train_registry.py and scripts/prepare_app_data.py.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context import emg_ratio, hpn_periods, overlapping, sli_channel_active, sli_periods
from src.detection import detect_all


REGISTRY = ROOT / "models/registry.json"
APP_DATA = ROOT / "data/processed/app"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
MODE_NAMES = {"unbalanced": "non équilibré", "balanced": "équilibré"}
EMG_BURST_RATIO = 3.0


def context(recordings: list[str]) -> dict:
    """Hyperventilation and photic-stimulation periods, plus the reference channels, per recording."""
    events = pd.read_csv(APP_DATA / "technician_events.csv")
    result = {}
    for recording in recordings:
        notes = events.loc[events["recording"] == recording]
        reference = pd.read_parquet(APP_DATA / f"reference/{recording}.parquet")
        result[recording] = {
            "hpn": hpn_periods(notes),
            "sli": sli_periods(notes),
            "seizure_notes": notes.loc[notes["category"] == "Seizure note", "time_s"].to_numpy(dtype=float),
            "sli_channel": reference["SLI"].to_numpy() if "SLI" in reference else None,
            "emg": [reference[name].to_numpy() for name in ("EMG1", "EMG2") if name in reference],
        }
    return result


def describe_interval(recording_context: dict, start: float, end: float) -> dict:
    steps = overlapping(start, end, recording_context["sli"])
    channel = recording_context["sli_channel"]
    emg = recording_context["emg"]
    ratios = [emg_ratio(signal, start, end) for signal in emg]
    return {
        "hpn": bool(overlapping(start, end, recording_context["hpn"])),
        "sli": bool(steps),
        "sli_2_4_hz": any(2 <= step[2] <= 4 for step in steps),
        "sli_channel": None if channel is None else sli_channel_active(channel, start, end),
        "emg_burst": None if not emg else bool(np.nanmax(ratios) >= EMG_BURST_RATIO),
        "seizure_note": bool(((recording_context["seizure_notes"] >= start - 10) & (recording_context["seizure_notes"] <= end + 10)).any()),
    }


def share(count: int, total: int) -> str:
    return "—" if not total else f"{count}/{total} ({count / total:.0%})"


def spread(values: dict, digits: int = 3) -> str:
    return f"{values['mean']:.{digits}f} ± {values['std']:.{digits}f}"


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    models = registry["models"]
    annotations = pd.read_csv(ANNOTATIONS)
    metadata = pd.read_csv(APP_DATA / "recording_metadata.csv")
    recordings = metadata["recording"].tolist()
    durations = metadata.set_index("recording")["duration_s"]
    information = context(recordings)
    total_seconds = float(durations.sum())
    hpn_seconds = sum(min(end, durations[r]) - start for r in recordings for start, end in information[r]["hpn"])
    sli_seconds = sum(end - start for r in recordings for start, end, _ in information[r]["sli"])
    with_sli_channel = [r for r in recordings if information[r]["sli_channel"] is not None]
    with_emg = [r for r in recordings if information[r]["emg"]]

    def analyse(frame: pd.DataFrame) -> dict:
        rows = [dict(describe_interval(information[row.recording], row.start_s, row.end_s), recording=row.recording) for row in frame.itertuples(index=False)]
        table = pd.DataFrame(rows, columns=["hpn", "sli", "sli_2_4_hz", "sli_channel", "emg_burst", "seizure_note", "recording"])
        in_channel = table.loc[table["recording"].isin(with_sli_channel)]
        in_emg = table.loc[table["recording"].isin(with_emg)]
        return {
            "total": len(table), "hpn": int(table["hpn"].sum()), "sli": int(table["sli"].sum()), "sli_2_4_hz": int(table["sli_2_4_hz"].sum()),
            "note": int(table["seizure_note"].sum()),
            "channel": (int(in_channel["sli_channel"].sum()), len(in_channel)), "emg": (int(in_emg["emg_burst"].sum()), len(in_emg)),
        }

    false_alarms = {}
    for key, entry in models.items():
        table = pd.read_parquet(ROOT / entry["predictions"])
        detected, matches = detect_all(table, annotations, entry["best_threshold"])
        false = matches.loc[matches["status"] == "FP", ["recording", "detected_start_s", "detected_end_s"]]
        false_alarms[key] = analyse(false.rename(columns={"detected_start_s": "start_s", "detected_end_s": "end_s"}))
    seizures = analyse(annotations[["recording", "start_s", "end_s"]])

    default = registry["default_model"]
    lines = [
        "# Comparaison des 8 modèles",
        "",
        f"8 modèles = 4 algorithmes (Decision Tree, Random Forest, KNN, SVM) × 2 modes d’entraînement, sur le jeu officiel filtré à {len(registry['channels'])} canaux EEG "
        f"({len(registry['features'])} caractéristiques). Hyperparamètres par défaut de scikit-learn; `StandardScaler` pour KNN et SVM.",
        "",
        "- **Non équilibré** : chaque pli d’entraînement est utilisé tel quel.",
        "- **Équilibré** : sous-échantillonnage aléatoire du pli d’entraînement seulement : toutes les fenêtres de classe 1 + autant de fenêtres de classe 0 tirées au hasard (50/50). "
        f"Répété avec {len(registry['balanced_seeds'])} graines ({', '.join(map(str, registry['balanced_seeds']))}); la graine {registry['saved_seed']} sert aux prédictions et aux modèles sauvegardés.",
        "- **Test** : toujours les proportions réelles. `GroupKFold` à 5 plis par enregistrement (mêmes plis partout) : chaque fenêtre est prédite par un modèle qui n’a jamais vu son enregistrement.",
        "- **Seuil** : pour chaque modèle, le seuil qui maximise le F1 événement sur ses prédictions hors-pli; les résultats à 0,50 sont donnés aussi. "
        "Ce seuil est choisi et évalué sur les mêmes prédictions, donc les scores au meilleur seuil sont légèrement optimistes.",
        "",
        f"Modèle par défaut (meilleur F1 événement) : **{models[default]['algorithm_name']}, {MODE_NAMES[models[default]['mode']]}**.",
        "",
        "## Entraînement : distribution des classes",
        "",
        "| Mode | Fenêtres d’entraînement par pli (moyenne) | Classe 0 | Classe 1 | Part de classe 1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("unbalanced", "balanced"):
        training = models[f"rf_{mode}"]["training"]
        lines.append(f"| {MODE_NAMES[mode].capitalize()} | {training['windows']:,.0f} | {training['class_0']:,.0f} | {training['class_1']:,.0f} | {training['class_1_share']:.1%} |")

    def table(state: str, title: str) -> None:
        lines.extend(
            [
                "",
                title,
                "",
                "| Algorithme | Mode | Seuil | Fenêtres : precision | recall | F1 | Événements : F1 | recall | precision | Trouvées / manquées | Fausses alarmes | FA / heure | Erreur début (s) | Erreur fin (s) | Temps d’entraînement (s/pli) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        best = max(entry[state]["event"]["f1"] for entry in models.values())
        for entry in models.values():
            result = entry[state]
            window, event = result["window"], result["event"]
            mark = "**" if event["f1"] == best else ""
            onset = "—" if event["tp"] == 0 else f"{event['onset_error_mean_s']:+.2f}"
            offset = "—" if event["tp"] == 0 else f"{event['offset_error_mean_s']:+.2f}"
            lines.append(
                f"| {entry['algorithm_name']} | {MODE_NAMES[entry['mode']]} | {result['threshold']:.2f} | {window['precision']:.3f} | {window['recall']:.3f} | {window['f1']:.3f} | "
                f"{mark}{event['f1']:.3f}{mark} | {event['recall']:.3f} | {event['precision']:.3f} | {event['tp']} / {event['fn']} | {event['fp']} | "
                f"{event['false_alarms_per_hour']:.1f} | {onset} | {offset} | {entry['fit_seconds_per_fold']:.2f} |"
            )

    table("at_best_threshold", "## Les 8 modèles au meilleur seuil de chacun")
    lines += [
        "",
        "Fenêtres : fenêtres de 2 s. Événements : une crise est trouvée si au moins une détection la chevauche; une fausse alarme est une détection hors de toute crise annotée. "
        "Erreur de début / de fin : détecté − réel, moyenne sur les crises trouvées (positif = en retard). "
        "Temps d’entraînement : ajustement du modèle sur un pli; le mode équilibré est plus rapide car il s’entraîne sur environ 12 fois moins de fenêtres. Pour KNN, l’ajustement est presque instantané : son coût est à la prédiction, non mesurée ici.",
    ]
    table("at_0.50", "## Les 8 modèles au seuil 0,50")
    lines += [
        "",
        "L’arbre de décision ne produit que des probabilités 0 ou 1 : son résultat ne dépend pas du seuil.",
        "",
        f"## Mode équilibré : variabilité sur {len(registry['balanced_seeds'])} graines",
        "",
        "Moyenne ± écart-type selon le tirage des fenêtres de classe 0.",
        "",
        "| Algorithme | Meilleur seuil | F1 événement (meilleur seuil) | Recall événement | Precision événement | Fausses alarmes | F1 fenêtre (meilleur seuil) | F1 événement à 0,50 | F1 fenêtre à 0,50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in models.items():
        if entry["mode"] != "balanced":
            continue
        seeds = entry["seeds"]
        lines.append(
            f"| {entry['algorithm_name']} | {spread(seeds['best_threshold'], 2)} | {spread(seeds['event_f1_best'])} | {spread(seeds['event_recall_best'])} | "
            f"{spread(seeds['event_precision_best'])} | {spread(seeds['false_alarms_best'], 1)} | {spread(seeds['window_f1_best'])} | {spread(seeds['event_f1_0.50'])} | {spread(seeds['window_f1_0.50'])} |"
        )
    lines += [
        "",
        "## Fausses alarmes : coïncidence avec SLI, EMG et hyperpnée",
        "",
        "ECG, EMG et SLI ne sont jamais utilisés comme caractéristiques. Ils servent ici, avec les notes du technicien, à interpréter les fausses alarmes de chaque modèle à son meilleur seuil.",
        "",
        f"- **Hyperpnée (HPN)** : périodes reconstruites à partir des notes « HPN » (notes espacées de moins de 60 s, plus 60 s après la dernière). Elles couvrent {hpn_seconds / total_seconds:.1%} du temps enregistré.",
        f"- **SLI (notes)** : chaque note « SLI=x Hz » ouvre un palier jusqu’à la note suivante (20 s au plus). Les paliers couvrent {sli_seconds / total_seconds:.1%} du temps enregistré. « 2–4 Hz » : palier dont la fréquence est entre 2 et 4 Hz.",
        f"- **SLI (canal)** : le canal SLI est un marqueur d’impulsions. Il n’est exploitable que dans {len(with_sli_channel)} enregistrements sur {len(recordings)} (constant dans les autres); la colonne compte les fausses alarmes de ces enregistrements pendant lesquelles il pulse.",
        f"- **Bouffée EMG** : RMS de l’EMG pendant l’événement ≥ {EMG_BURST_RATIO:g} fois sa valeur typique. L’EMG n’existe que dans {len(with_emg)} enregistrements; la colonne ne porte que sur eux.",
        "- **Note de crise** : note « absence », « crise » ou « clonies » du technicien à ±10 s.",
        "",
        "| Modèle | Fausses alarmes | Pendant HPN | Pendant SLI (notes) | dont SLI 2–4 Hz | SLI actif (canal) | Bouffée EMG | Note de crise à ±10 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in models.items():
        result = false_alarms[key]
        lines.append(
            f"| {entry['algorithm_name']}, {MODE_NAMES[entry['mode']]} | {result['total']} | {share(result['hpn'], result['total'])} | {share(result['sli'], result['total'])} | "
            f"{share(result['sli_2_4_hz'], result['total'])} | {share(*result['channel'])} | {share(*result['emg'])} | {share(result['note'], result['total'])} |"
        )
    lines.append(
        f"| *Crises réelles (référence)* | {seizures['total']} | {share(seizures['hpn'], seizures['total'])} | {share(seizures['sli'], seizures['total'])} | "
        f"{share(seizures['sli_2_4_hz'], seizures['total'])} | {share(*seizures['channel'])} | {share(*seizures['emg'])} | {share(seizures['note'], seizures['total'])} |"
    )
    chosen = false_alarms[default]
    lines += [
        "",
        f"Lecture pour le modèle par défaut ({models[default]['algorithm_name']}, {MODE_NAMES[models[default]['mode']]}) : "
        f"{share(chosen['hpn'], chosen['total'])} de ses fausses alarmes tombent pendant l’hyperpnée, qui ne représente que {hpn_seconds / total_seconds:.0%} du temps; "
        f"{share(chosen['sli'], chosen['total'])} pendant la SLI ({sli_seconds / total_seconds:.0%} du temps), dont {chosen['sli_2_4_hz']} pendant un palier à 2–4 Hz; "
        f"bouffée EMG : {share(*chosen['emg'])}. "
        f"Les crises réelles suivent la même tendance : {share(seizures['hpn'], seizures['total'])} surviennent pendant l’hyperpnée, qui sert justement à les provoquer. "
        "Une fausse alarme pendant HPN ou SLI peut donc être une décharge réelle non annotée aussi bien qu’une erreur du modèle; ces chiffres ne permettent pas de trancher.",
        "",
        "## Conclusion",
        "",
    ]

    best = lambda key: models[key]["at_best_threshold"]
    ranking = sorted(models, key=lambda key: best(key)["event"]["f1"], reverse=True)
    top, second = ranking[0], ranking[1]
    lines.append(
        f"- **Meilleur modèle** : {models[top]['algorithm_name']} {MODE_NAMES[models[top]['mode']]} (F1 événement {best(top)['event']['f1']:.3f}, "
        f"{best(top)['event']['tp']} crises trouvées sur {best(top)['event']['tp'] + best(top)['event']['fn']}, {best(top)['event']['fp']} fausses alarmes), "
        f"devant {models[second]['algorithm_name']} {MODE_NAMES[models[second]['mode']]} ({best(second)['event']['f1']:.3f}). "
        "L’écart entre les premiers correspond à quelques événements : il n’est pas assez grand pour les départager avec certitude."
    )
    effects = []
    for algorithm in ("dt", "rf", "knn", "svm"):
        a, b = best(f"{algorithm}_unbalanced"), best(f"{algorithm}_balanced")
        effects.append(
            f"{models[f'{algorithm}_unbalanced']['algorithm_name']} : F1 événement {a['event']['f1']:.3f} → {b['event']['f1']:.3f}, "
            f"fausses alarmes {a['event']['fp']} → {b['event']['fp']}, seuil {a['threshold']:.2f} → {b['threshold']:.2f}"
        )
    lines.append("- **Effet de l’équilibrage au meilleur seuil** (non équilibré → équilibré) : " + "; ".join(effects) + ".")
    half = lambda key: models[key]["at_0.50"]
    balanced_thresholds = [f"{models[key]['algorithm_name']} {best(key)['threshold']:.2f}" for key in ("rf_balanced", "knn_balanced", "svm_balanced")]
    recall_gain = np.mean([half(f"{a}_balanced")["window"]["recall"] - half(f"{a}_unbalanced")["window"]["recall"] for a in ("dt", "rf", "knn", "svm")])
    precision_loss = np.mean([half(f"{a}_balanced")["window"]["precision"] - half(f"{a}_unbalanced")["window"]["precision"] for a in ("dt", "rf", "knn", "svm")])
    alarms_half = sum(half(f"{a}_balanced")["event"]["fp"] for a in ("rf", "knn", "svm")) / max(1, sum(half(f"{a}_unbalanced")["event"]["fp"] for a in ("rf", "knn", "svm")))
    lines += [
        f"- **Au seuil fixe 0,50**, l’équilibrage change surtout le compromis : recall fenêtre {recall_gain:+.3f} en moyenne, precision fenêtre {precision_loss:+.3f}, "
        f"et {alarms_half:.1f} fois plus de fausses alarmes pour Random Forest, KNN et SVM réunis. Un modèle entraîné à 50 % de crises surestime la probabilité de crise quand il est testé sur 4 %.",
        "- **Le seuil ne compense qu’en partie l’équilibrage** : les modèles équilibrés demandent un seuil bien plus élevé "
        f"({', '.join(balanced_thresholds)}) "
        "et restent en dessous de leur version non équilibrée. Pour Random Forest et SVM, le meilleur seuil atteint la limite haute explorée (0,99) : "
        "presque toutes leurs probabilités utiles sont comprimées près de 1, ce qui rend le réglage fragile. Ici, entraîner sur les proportions réelles et choisir le seuil donne de meilleurs résultats que rééquilibrer.",
        f"- **Arbre de décision** : inutilisable pour la détection temporelle dans les deux modes ({best('dt_unbalanced')['event']['fp']} et {best('dt_balanced')['event']['fp']} fausses alarmes), car il ne donne que 0 ou 1.",
        "- **Limites** : 21 enregistrements, 93 crises, hyperparamètres par défaut, identité des patients non confirmée. Les différences de 0,01 à 0,02 en F1 événement sont dans la variabilité entre plis et entre graines.",
        "",
        "Reproduction : `python scripts/train_registry.py`, `python scripts/prepare_app_data.py`, `python scripts/compare_models.py`. Détails par modèle : `models/registry.json`.",
        "",
    ]
    (ROOT / "docs/MODEL_COMPARISON.md").write_text("\n".join(lines))
    print(f"Report: {ROOT / 'docs/MODEL_COMPARISON.md'}")


if __name__ == "__main__":
    main()
