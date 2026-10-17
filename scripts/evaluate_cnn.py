"""Score the CNN predictions trained on Kaggle (kaggle/train_cnn.py) like the classical models.

Reads the zip downloaded from the Kaggle notebook (see docs/KAGGLE.md), re-scores every CNN's
out-of-fold predictions with this project's own code (src/detection.py: same event matching,
same threshold sweep as the classical models) and with SzCORE (reusing scripts/evaluate_nested.py
for the exact same scoring code), and writes a report comparing them with the SVM baseline.

Two thresholds per model, as for the classical models:
- nested: each outer fold scored with its own threshold, taken from the Kaggle run's
  fold_thresholds (chosen only on 3 held-out training recordings, never the test fold);
- optimistic: one threshold chosen on all out-of-fold predictions (reference only).

Usage, after unzipping the Kaggle output into outputs/kaggle/cnn_outputs/:
    python scripts/evaluate_cnn.py
    python scripts/evaluate_cnn.py --dir path/to/unzipped/cnn_outputs

Outputs: outputs/cnn_evaluation.csv, docs/DEEP_LEARNING_RESULTS.md. Nothing else is changed:
the registered models and the app are not touched.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.detection import detect_all, detect_all_hysteresis, event_metrics, smooth_probabilities
from evaluate_nested import szcore
from train_registry import THRESHOLDS, best_threshold


DEFAULT_DIR = ROOT / "outputs/kaggle/cnn_outputs"
REGISTRY = ROOT / "models/registry.json"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
APP_METADATA = ROOT / "data/processed/app/recording_metadata.csv"
TABLE = ROOT / "outputs/cnn_evaluation.csv"
REPORT = ROOT / "docs/DEEP_LEARNING_RESULTS.md"
ARCH_NAMES = {"shallow": "ShallowConvNet", "eegnet": "EEGNet"}


def label(config: dict) -> str:
    return f"{ARCH_NAMES.get(config['arch'], config['arch'])}, {'augmented' if config['augment'] else 'plain'}"


def find_runs(directory: Path) -> list[tuple[str, Path, dict]]:
    """Every CNN run in the Kaggle output: (name, predictions file, config)."""
    runs = []
    for config_path in sorted(directory.glob("*_config.json")):
        config = json.loads(config_path.read_text())
        predictions_path = directory / "predictions" / f"{config['name']}.parquet"
        if not predictions_path.exists():
            print(f"skipping {config['name']}: {predictions_path.name} not found", flush=True)
            continue
        runs.append((config["name"], predictions_path, config))
    return runs


def nested_score(predictions: pd.DataFrame, annotations: pd.DataFrame, fold_thresholds: dict) -> tuple[dict, pd.DataFrame]:
    """Each outer fold scored with its own threshold (never chosen on that fold's recordings)."""
    table = predictions.copy()
    for fold_key, threshold in fold_thresholds.items():
        rows = table["fold"] == int(fold_key)
        table.loc[rows, "proba"] = np.where(table.loc[rows, "proba"] >= threshold, 1.0, -1.0)
    detected, matches = detect_all(table, annotations, 0.0)
    return event_metrics(matches, len(detected)), detected


def smoothed_score(predictions: pd.DataFrame, annotations: pd.DataFrame, fold_thresholds: dict, window: int, gap: float) -> tuple[dict, pd.DataFrame]:
    """Nested thresholds again, but on a smoothed probability with hysteresis instead of a hard
    cut (research report, part A5): removes isolated one-window spikes without retraining, and
    lets a confirmed event survive a brief dip instead of being cut into several false alarms.
    Each fold keeps its own nested ``high`` threshold; ``low = high - gap``.
    """
    smoothed = smooth_probabilities(predictions, window=window)
    detected_frames, match_frames = [], []
    for fold_key, high in fold_thresholds.items():
        subset = smoothed.loc[smoothed["fold"] == int(fold_key)]
        low = max(THRESHOLDS[0], high - gap)
        detected, matches = detect_all_hysteresis(subset, annotations, high, low)
        detected_frames.append(detected)
        match_frames.append(matches)
    detected = pd.concat(detected_frames, ignore_index=True)
    matches = pd.concat(match_frames, ignore_index=True)
    return event_metrics(matches, len(detected)), detected


def optimistic_score(predictions: pd.DataFrame, annotations: pd.DataFrame) -> tuple[dict, float, pd.DataFrame]:
    """One threshold chosen on all out-of-fold predictions (slightly optimistic, kept for reference)."""
    threshold = best_threshold(predictions, annotations)
    detected, matches = detect_all(predictions, annotations, threshold)
    return event_metrics(matches, len(detected)), threshold, detected


def baseline_row(registry: dict) -> dict:
    default = registry["default_model"]
    entry = registry["models"][default]
    event = entry["at_best_threshold"]["event"]
    return {
        "name": default, "label": entry["label"], "event_f1": event["f1"], "found": event["tp"], "missed": event["fn"],
        "false_alarms": event["fp"], "false_alarms_per_hour": event["false_alarms_per_hour"], "threshold": entry["best_threshold"],
        "kind": "optimistic (registry; nested evaluation of the classical models is separate, see docs/NESTED_EVALUATION.md)",
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="Unzipped Kaggle output folder (contains *_config.json and predictions/)")
    parser.add_argument("--smooth-window", type=int, default=3, help="Moving-median window (odd number of windows) before hysteresis; 1 disables it")
    parser.add_argument("--hysteresis-gap", type=float, default=0.3, help="low threshold = each fold's threshold minus this; an event survives a dip down to low")
    parser.add_argument("--no-smooth", action="store_true", help="Skip the smoothed+hysteresis scoring entirely")
    args = parser.parse_args()
    directory = Path(args.dir)
    if not directory.exists():
        raise SystemExit(
            f"{directory} not found.\nDownload cnn_outputs.zip from the Kaggle notebook's Output tab, then:\n"
            f"  mkdir -p {DEFAULT_DIR.parent}\n  unzip cnn_outputs.zip -d {DEFAULT_DIR}\nSee docs/KAGGLE.md."
        )
    runs = find_runs(directory)
    if not runs:
        raise SystemExit(f"No *_config.json found in {directory}. Did the Kaggle notebook finish at least one full run?")

    annotations = pd.read_csv(ANNOTATIONS)
    durations = pd.read_csv(APP_METADATA).set_index("recording")["duration_s"]
    registry = json.loads(REGISTRY.read_text())
    hours = durations.sum() / 3600

    rows = [baseline_row(registry)]
    for name, predictions_path, config in runs:
        predictions = pd.read_parquet(predictions_path)
        nested, nested_detected = nested_score(predictions, annotations, config["fold_thresholds"])
        optimistic, threshold, _ = optimistic_score(predictions, annotations)
        ap = average_precision_score(predictions["true_class"], predictions["proba"])
        variants = [("nested", nested, nested_detected, {"threshold": None}), ("optimistic", optimistic, None, {"threshold": threshold})]
        if not args.no_smooth:
            smoothed, smoothed_detected = smoothed_score(predictions, annotations, config["fold_thresholds"], args.smooth_window, args.hysteresis_gap)
            variants.append(("nested_smoothed", smoothed, smoothed_detected, {"threshold": None}))
        for kind, event, detected, extra in variants:
            sz = szcore(detected, annotations, durations) if detected is not None else None
            rows.append({
                "name": name, "label": label(config), "event_f1": event["f1"], "found": event["tp"], "missed": event["fn"],
                "false_alarms": event["fp"], "false_alarms_per_hour": event["fp"] / hours, "kind": kind,
                "window_ap": float(ap), "szcore_event_f1": sz["event"]["f1"] if sz else None,
                "szcore_sensitivity": sz["event"]["sensitivity"] if sz else None,
                "szcore_fp_per_day": sz["event"]["fp_per_day"] if sz else None,
                "fold_thresholds": config["fold_thresholds"] if kind.startswith("nested") else None, **extra,
            })
            print(f"{name} ({label(config)}) [{kind}]: event F1 {event['f1']:.3f} (found {event['tp']}/{event['tp'] + event['fn']}, "
                  f"false alarms {event['fp']})" + (f"; SzCORE event F1 {sz['event']['f1']:.3f}" if sz else "") + (f"; window AP {ap:.3f}" if kind == "nested" else ""), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(TABLE, index=False)
    write_report(table)
    print(f"\nTable: {TABLE}\nReport: {REPORT}")


def write_report(table: pd.DataFrame) -> None:
    baseline = table.iloc[0]
    nested = table.loc[table["kind"] == "nested"].set_index("name")
    smoothed = table.loc[table["kind"] == "nested_smoothed"].set_index("name")
    optimistic = table.loc[table["kind"] == "optimistic"].set_index("name")
    has_smoothed = len(smoothed) > 0

    # "Effective" = whichever of plain / smoothed+hysteresis nested scoring is better for that
    # model; this is what the headline table and the verdict are ranked on.
    effective = {}
    for name in nested.index:
        candidates = [("plain", nested.loc[name])] + ([("smoothed", smoothed.loc[name])] if has_smoothed and name in smoothed.index else [])
        effective[name] = max(candidates, key=lambda item: item[1]["event_f1"])
    order = sorted(effective, key=lambda name: effective[name][1]["event_f1"], reverse=True)
    best = effective[order[0]][1] if order else None
    best_kind = effective[order[0]][0] if order else None

    lines = [
        "# Deep learning results (Kaggle)",
        "",
        "Compact CNNs trained directly on the cleaned 19-channel EEG (no hand-made features), on Kaggle's free GPU. "
        "Method, input pipeline and training recipe: [`kaggle/train_cnn.py`](../kaggle/train_cnn.py); how to reproduce: [`docs/KAGGLE.md`](KAGGLE.md).",
        "",
        "- **Nested threshold** (main number): each of the 5 outer folds (same `GroupKFold` as every other model here) is scored with its own threshold, "
        "chosen only on 3 training recordings held out from that fold and never seen by the test fold. This is the fair number to compare with the SVM.",
    ]
    if has_smoothed:
        lines.append(
            "- **Smoothed**: the same nested thresholds, but the probability is smoothed first (moving median) and an event, once started at the "
            "threshold, survives a dip down to threshold − 0.3 instead of being cut into several false alarms (research report, part A5; "
            "`src/detection.py`: `smooth_probabilities`, `windows_to_intervals_hysteresis`). Pure post-processing, no retraining."
        )
    lines += [
        "- **Optimistic threshold**: one threshold chosen on all out-of-fold predictions at once, kept only for reference (this is how the classical "
        "models' registry numbers below are computed too, pending their own nested re-evaluation — see `docs/NESTED_EVALUATION.md`).",
        "- **SzCORE**: event scoring from the [SzCORE framework](https://pypi.org/project/timescoring/) (30 s / 60 s tolerance, events closer than 90 s merged), "
        "given for comparison with the literature; not directly comparable to this project's own event F1 (see `docs/NESTED_EVALUATION.md` for why).",
        "- **Window AP**: average precision of the raw per-window probability (threshold-independent), each window scored by a model that never saw its patient.",
        "",
        "## Results",
        "",
        "Best of plain / smoothed nested scoring per model (bold marks which one won):",
        "",
        f"| Model | Event F1 | Found (of 93) | False alarms | FA / hour | SzCORE event F1 | Window AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **{baseline['label']}** (classical baseline) | **{baseline['event_f1']:.3f}** | {baseline['found']} | {baseline['false_alarms']} | "
        f"{baseline['false_alarms_per_hour']:.1f} | — | — |",
    ]
    for name in order:
        kind, row = effective[name]
        mark = "**" if name == order[0] else ""
        tag = " *(smoothed)*" if kind == "smoothed" else ""
        lines.append(
            f"| {mark}{row['label']}{tag}{mark} | {mark}{row['event_f1']:.3f}{mark} | {row['found']} | {row['false_alarms']} | "
            f"{row['false_alarms_per_hour']:.1f} | {row['szcore_event_f1']:.3f} | {row['window_ap']:.3f} |"
        )
    if has_smoothed:
        lines += [
            "",
            "Plain vs. smoothed, each model (same fold thresholds, only the post-processing differs):",
            "",
            "| Model | Event F1 plain | Event F1 smoothed | Δ |",
            "|---|---:|---:|---:|",
        ]
        for name in nested.index:
            plain_f1 = nested.loc[name, "event_f1"]
            if name in smoothed.index:
                smooth_f1 = smoothed.loc[name, "event_f1"]
                lines.append(f"| {nested.loc[name, 'label']} | {plain_f1:.3f} | {smooth_f1:.3f} | {smooth_f1 - plain_f1:+.3f} |")
    lines += [
        "",
        "Optimistic threshold, for reference (not the number to compare with the SVM):",
        "",
        "| Model | Event F1 (optimistic) | Threshold | Found | False alarms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in optimistic.iterrows():
        lines.append(f"| {row['label']} | {row['event_f1']:.3f} | {row['threshold']:.2f} | {row['found']} | {row['false_alarms']} |")

    lines += ["", "## Reading", ""]
    if best is not None:
        gap = baseline["event_f1"] - best["event_f1"]
        if gap > 0.01:
            verdict = (
                f"The best CNN ({best['label']}, event F1 {best['event_f1']:.3f}) does not beat the SVM baseline "
                f"({baseline['event_f1']:.3f}), a gap of {gap:.3f}. This matches the research report's expectation: "
                "with 93 seizures in 21 patients, raw-signal deep learning is unlikely to beat hand-made amplitude and "
                "band-power features on a pattern as regular as the 3 Hz absence rhythm."
            )
        elif gap > -0.01:
            verdict = (
                f"The best CNN ({best['label']}, event F1 {best['event_f1']:.3f}) ties the SVM baseline "
                f"({baseline['event_f1']:.3f}) within the fold-to-fold noise. A tie without hand-made features is a "
                "genuine result for a student project, but not a reason to replace the SVM, which is simpler and faster."
            )
        else:
            verdict = (
                f"The best CNN ({best['label']}, event F1 {best['event_f1']:.3f}) beats the SVM baseline "
                f"({baseline['event_f1']:.3f}) by {-gap:.3f} under the nested threshold. Check this is not noise "
                "(re-run with different seeds) before reporting it as a clear win."
            )
        if best_kind == "smoothed":
            verdict += " Its winning score is the *smoothed* one — plain thresholding alone did worse (see the plain vs. smoothed table above)."
        lines.append(f"- {verdict}")
        drop = [(row["label"], row["event_f1"] - optimistic.loc[name, "event_f1"]) for name, row in nested.iterrows()]
        lines.append(
            "- **Optimism of the CNNs' own threshold choice** (nested − optimistic event F1): "
            + ", ".join(f"{label} {value:+.3f}" for label, value in drop)
            + ". A negative value is how much the single-threshold number overstates the model; the same gap the classical models show "
            "(see `docs/NESTED_EVALUATION.md`), and the reason the nested number is the one to trust."
        )
    lines += [
        "- **Augmentation** (amplitude scaling, noise, channel dropout, left-right mirror, applied to both classes so the seizure share "
        "stays near 4 %, unlike the classical models' first augmentation attempt) is compared against the plain CNN when both were run.",
        "",
        "Reproduction: train on Kaggle (`kaggle/absence_seizure_cnn.ipynb`, see `docs/KAGGLE.md`), download the output, then "
        "`python scripts/evaluate_cnn.py`. Raw numbers: `outputs/cnn_evaluation.csv`.",
        "",
    ]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
