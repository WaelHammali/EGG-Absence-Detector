"""Print detected seizure intervals for one recording and compare them with the annotations.

Uses the out-of-fold probabilities, i.e. predictions from a model that never saw the recording.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import event_metrics, match_events, windows_to_intervals


PREDICTIONS = ROOT / "outputs/predictions_oof.parquet"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
SUMMARY = ROOT / "outputs/detection_summary.json"


def default_threshold() -> float:
    if SUMMARY.exists():
        return float(json.loads(SUMMARY.read_text())["threshold"])
    return 0.5


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="Recording ID, e.g. 190304A_E")
    parser.add_argument("--threshold", type=float, help="Probability threshold (default: the event-F1 optimum from Part X)")
    args = parser.parse_args()
    if not PREDICTIONS.exists():
        sys.exit(f"Missing {PREDICTIONS.relative_to(ROOT)}: run `python scripts/predict_oof.py` first.")
    predictions = pd.read_parquet(PREDICTIONS)
    available = sorted(predictions["recording"].unique())
    if args.recording not in available:
        sys.exit(f"Unknown recording {args.recording!r}. Available: {', '.join(available)}")
    threshold = default_threshold() if args.threshold is None else args.threshold
    if not 0 < threshold <= 1:
        sys.exit("--threshold must be in (0, 1].")

    group = predictions.loc[predictions["recording"] == args.recording]
    annotations = pd.read_csv(ANNOTATIONS)
    real = annotations.loc[annotations["recording"] == args.recording, ["start_s", "end_s"]].sort_values("start_s").reset_index(drop=True)
    detected = windows_to_intervals(group["start_s"], group["end_s"], group["proba"], threshold)
    matches = match_events(detected, real)
    metrics = event_metrics(matches, len(detected))
    false_starts = set(matches.loc[matches["status"] == "FP", "detected_start_s"])

    print(f"Enregistrement {args.recording} — seuil {threshold:.2f} (probabilités hors-pli)")
    print()
    if detected.empty:
        print("Aucune crise détectée")
    for number, row in enumerate(detected.itertuples(index=False), 1):
        note = "  (fausse alarme)" if row.start_s in false_starts else ""
        print(f"Crise détectée {number} : {row.start_s:.1f} s → {row.end_s:.1f} s  (proba max {row.max_proba:.2f}){note}")
    print()
    for number, row in enumerate(matches.loc[matches["status"] != "FP"].itertuples(index=False), 1):
        if row.status == "TP":
            note = (
                f"trouvée : {row.detected_start_s:.1f} s → {row.detected_end_s:.1f} s, "
                f"erreur début {row.onset_error_s:+.1f} s, erreur fin {row.offset_error_s:+.1f} s"
            )
        else:
            note = "manquée"
        print(f"Crise réelle {number} : {row.real_start_s:.1f} s → {row.real_end_s:.1f} s  ({note})")
    print()
    print(
        f"Bilan : {metrics['tp']} trouvée(s), {metrics['fn']} manquée(s), {metrics['fp']} fausse(s) alarme(s) — "
        f"precision {metrics['precision']:.2f}, recall {metrics['recall']:.2f}, F1 {metrics['f1']:.2f}"
    )


if __name__ == "__main__":
    main()
