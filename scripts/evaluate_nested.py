"""Stricter evaluation of the registered models: nested threshold choice and SzCORE scoring.

Until now each model's threshold was the one giving the best event F1 on its own out-of-fold
predictions, then scored on those same predictions (slightly optimistic). Here, for each outer
fold of the same GroupKFold, the threshold is chosen with an inner GroupKFold (4 folds) on the
training recordings only: inner models are trained and the threshold maximizing event F1 on
their inner out-of-fold predictions is applied to the outer test fold. Training modes are
reproduced in the inner folds (undersampling with the saved seed, augmented windows of the inner
training recordings only).

Detections at the nested thresholds are then scored three ways:
- this project's event matching (any overlap, events merged under 2 s, events under 2 s dropped);
- SzCORE event scoring (timescoring library, default parameters: 30 s / 60 s tolerance, events
  closer than 90 s merged, events longer than 5 min split);
- SzCORE sample scoring (10 Hz).

Outputs: outputs/nested/<model>.json (inner thresholds, cached), outputs/nested_evaluation.csv
and docs/NESTED_EVALUATION.md. The saved models and predictions are only read.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from timescoring import scoring
from timescoring.annotations import Annotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.detection import detect_all, event_metrics
from src.paths import AUGMENTED, OFFICIAL, dataset_file
from train_models import METADATA, undersample
from train_registry import SAVED_SEED, best_threshold, pipeline


REGISTRY = ROOT / "models/registry.json"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
CACHE = ROOT / "outputs/nested"
TABLE = ROOT / "outputs/nested_evaluation.csv"
REPORT = ROOT / "docs/NESTED_EVALUATION.md"
SZCORE_FS = 10


def load():
    frame = pd.read_parquet(dataset_file(OFFICIAL))
    features = [column for column in frame.columns if column not in METADATA]
    augmented = pd.read_parquet(dataset_file(AUGMENTED))
    return {
        "X": frame[features].to_numpy(dtype=np.float64),
        "y": frame["class"].to_numpy(dtype=np.int64),
        "groups": frame["recording"].to_numpy(),
        "reference": frame[["recording", "window", "start_s", "end_s"]].assign(true_class=frame["class"].astype(np.uint8)),
        "extra_X": augmented[features].to_numpy(dtype=np.float64),
        "extra_groups": augmented["recording"].to_numpy(),
    }


def inner_threshold(data: dict, annotations: pd.DataFrame, algorithm: str, mode: str, train: np.ndarray) -> float:
    """Threshold chosen with an inner GroupKFold on the training recordings of one outer fold."""
    X, y, groups = data["X"][train], data["y"][train], data["groups"][train]
    proba = np.full(len(y), np.nan)
    for inner_train, inner_test in GroupKFold(n_splits=4).split(X, y, groups):
        if mode == "balanced":
            inner_train = inner_train[undersample(y[inner_train], SAVED_SEED)]
        X_fit, y_fit = X[inner_train], y[inner_train]
        if mode == "augmented":
            added = data["extra_X"][np.isin(data["extra_groups"], groups[inner_train])]
            X_fit, y_fit = np.vstack([X_fit, added]), np.r_[y_fit, np.ones(len(added), dtype=y.dtype)]
        model = pipeline(algorithm)
        if algorithm == "rf":
            model.set_params(model__n_jobs=2)
        proba[inner_test] = model.fit(X_fit, y_fit).predict_proba(X[inner_test])[:, 1]
    table = data["reference"].iloc[train].assign(proba=proba.astype(np.float32))
    return best_threshold(table, annotations)


def nested_thresholds(key: str, entry: dict, data: dict, annotations: pd.DataFrame, jobs: int) -> list[float]:
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["thresholds"]
    folds = list(GroupKFold(n_splits=5).split(data["X"], data["y"], data["groups"]))
    started = time.perf_counter()
    thresholds = Parallel(n_jobs=jobs)(
        delayed(inner_threshold)(data, annotations, entry["algorithm"], entry["mode"], train) for train, _ in folds
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": key, "thresholds": [float(value) for value in thresholds], "seconds": time.perf_counter() - started}, indent=2) + "\n")
    print(f"{key}: inner thresholds {thresholds} ({time.perf_counter() - started:.0f} s)", flush=True)
    return thresholds


def szcore(detected: pd.DataFrame, annotations: pd.DataFrame, durations: pd.Series) -> dict:
    """SzCORE event and sample scores, counts summed over all recordings."""
    totals = {"event": [0, 0, 0], "sample": [0, 0, 0]}  # true detections, false detections, reference events
    for recording, duration in durations.items():
        samples = int(round(duration * SZCORE_FS))
        ref = Annotation([(row.start_s, row.end_s) for row in annotations.loc[annotations["recording"] == recording].itertuples()], SZCORE_FS, samples)
        hyp = Annotation([(row.start_s, row.end_s) for row in detected.loc[detected["recording"] == recording].itertuples()], SZCORE_FS, samples)
        for name, score in (("event", scoring.EventScoring(ref, hyp)), ("sample", scoring.SampleScoring(ref, hyp, SZCORE_FS))):
            totals[name][0] += score.tp
            totals[name][1] += score.fp
            totals[name][2] += score.refTrue
    result = {}
    hours = durations.sum() / 3600
    for name, (tp, fp, ref) in totals.items():
        result[name] = {
            "tp": int(tp), "fp": int(fp), "ref": int(ref),
            "sensitivity": tp / ref if ref else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + (ref - tp)) if ref + fp else float("nan"),
        }
    result["event"]["fp_per_day"] = result["event"]["fp"] / hours * 24
    return result


def evaluate(key: str, entry: dict, data: dict, annotations: pd.DataFrame, durations: pd.Series, thresholds: list[float]) -> dict:
    predictions = pd.read_parquet(ROOT / entry["predictions"])
    folds = list(GroupKFold(n_splits=5).split(data["X"], data["y"], data["groups"]))
    for number, (_, test) in enumerate(folds, 1):
        # The saved out-of-fold predictions must come from the same folds.
        assert set(predictions.loc[predictions["fold"] == number, "recording"]) == set(data["groups"][test]), key
    hours = durations.sum() / 3600

    def score(table: pd.DataFrame) -> dict:
        detected, matches = detect_all(table, annotations, 0.0)  # thresholds already applied through proba
        event = event_metrics(matches, len(detected))
        return {"detected": detected, "event": event, "false_alarms_per_hour": event["fp"] / hours, "szcore": szcore(detected, annotations, durations)}

    # Nested: each test fold binarized with its own inner threshold (proba >= t  <=>  shifted >= 0).
    nested = predictions.copy()
    for number, threshold in enumerate(thresholds, 1):
        rows = nested["fold"] == number
        nested.loc[rows, "proba"] = np.where(nested.loc[rows, "proba"] >= threshold, 1.0, -1.0)
    optimistic = predictions.assign(proba=np.where(predictions["proba"] >= entry["best_threshold"], 1.0, -1.0))
    return {"nested": score(nested), "optimistic": score(optimistic)}


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=5, help="Outer folds computed in parallel")
    parser.add_argument("--models", nargs="*", help="Model keys (default: all registered models)")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    annotations = pd.read_csv(ANNOTATIONS)
    data = load()
    durations = pd.read_csv(ROOT / "data/processed/app/recording_metadata.csv").set_index("recording")["duration_s"]
    keys = args.models or list(registry["models"])
    # Slowest models first so that the long jobs start early.
    keys.sort(key=lambda key: -registry["models"][key]["fit_seconds_per_fold"])
    rows, results = [], {}
    for key in keys:
        entry = registry["models"][key]
        thresholds = nested_thresholds(key, entry, data, annotations, args.jobs)
        result = evaluate(key, entry, data, annotations, durations, thresholds)
        results[key] = (entry, thresholds, result)
        for kind in ("optimistic", "nested"):
            part = result[kind]
            event, sz = part["event"], part["szcore"]
            rows.append({
                "model": key, "algorithm": entry["algorithm_name"], "mode": entry["mode"], "threshold_choice": kind,
                "thresholds": " ".join(f"{t:.2f}" for t in (thresholds if kind == "nested" else [entry["best_threshold"]])),
                "event_f1": event["f1"], "found": event["tp"], "missed": event["fn"], "false_alarms": event["fp"],
                "false_alarms_per_hour": part["false_alarms_per_hour"], "event_precision": event["precision"], "event_recall": event["recall"],
                "szcore_event_f1": sz["event"]["f1"], "szcore_sensitivity": sz["event"]["sensitivity"], "szcore_precision": sz["event"]["precision"],
                "szcore_ref_events": sz["event"]["ref"], "szcore_fp": sz["event"]["fp"], "szcore_fp_per_day": sz["event"]["fp_per_day"],
                "szcore_sample_f1": sz["sample"]["f1"],
            })
        nested = result["nested"]["event"]
        print(f"{key}: optimistic F1 {result['optimistic']['event']['f1']:.3f} -> nested {nested['f1']:.3f} "
              f"(found {nested['tp']}, false alarms {nested['fp']}); SzCORE event F1 {result['nested']['szcore']['event']['f1']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE, index=False)
    write_report(table, registry)
    print(f"Table: {TABLE}\nReport: {REPORT}")


def write_report(table: pd.DataFrame, registry: dict) -> None:
    modes = {"unbalanced": "Unbalanced", "balanced": "Balanced", "augmented": "Augmented"}
    nested = table.loc[table["threshold_choice"] == "nested"].set_index("model")
    optimistic = table.loc[table["threshold_choice"] == "optimistic"].set_index("model")
    order = nested.sort_values("event_f1", ascending=False).index
    best = order[0]
    svm = "svm_unbalanced" if "svm_unbalanced" in nested.index else best
    lines = [
        "# Nested evaluation and SzCORE scoring",
        "",
        "Stricter evaluation of the registered models, on the same out-of-fold predictions and the same 5 folds (GroupKFold by recording).",
        "",
        "- **Optimistic** (previous numbers): one threshold per model, chosen to maximize event F1 on all its out-of-fold predictions, then scored on those same predictions.",
        "- **Nested** (new): for each outer fold, the threshold is chosen by an inner GroupKFold (4 folds) on the training recordings only. "
        "The inner models are retrained with the same training mode (undersampling with seed 0, augmented windows of the inner training recordings only). "
        "The test fold never influences its threshold.",
        "- **This project's event scoring**: any overlap; detections closer than 2 s merged, shorter than 2 s dropped.",
        "- **SzCORE event scoring** ([timescoring](https://pypi.org/project/timescoring/), default parameters of the SzCORE framework): "
        "a seizure is detected if a detection overlaps it extended by 30 s before and 60 s after; reference and detected events closer than 90 s are merged "
        "and events longer than 5 min split. F1 = 2·TP / (2·TP + FP + FN), counts summed over the 21 recordings.",
        "- **SzCORE sample scoring**: the same comparison sample by sample at 10 Hz.",
        "",
        "SzCORE's defaults were designed for long-term EEG with longer seizures. Here, absences often occur less than 90 s apart, so several annotated absences "
        f"become one reference event (93 annotated seizures → {int(nested['szcore_ref_events'].iloc[0])} SzCORE events) and the 30/60 s tolerance is generous. "
        "Its event F1 is therefore not comparable with this project's event F1; it is given for comparison with the literature.",
        "",
        "## Results",
        "",
        "| Model | Event F1 optimistic | Event F1 nested | Nested thresholds (5 folds) | Found (of 93) | False alarms | FA / hour | SzCORE event F1 | SzCORE sensitivity | SzCORE precision | SzCORE FP / day | SzCORE sample F1 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in order:
        row, old = nested.loc[key], optimistic.loc[key]
        mark = "**" if key == best else ""
        lines.append(
            f"| {row['algorithm']} · {modes[row['mode']]} | {old['event_f1']:.3f} | {mark}{row['event_f1']:.3f}{mark} | {row['thresholds']} | "
            f"{row['found']} | {row['false_alarms']} | {row['false_alarms_per_hour']:.1f} | {row['szcore_event_f1']:.3f} | {row['szcore_sensitivity']:.3f} | "
            f"{row['szcore_precision']:.3f} | {row['szcore_fp_per_day']:.1f} | {row['szcore_sample_f1']:.3f} |"
        )
    drop = optimistic["event_f1"] - nested["event_f1"]
    s_new, s_old = nested.loc[svm], optimistic.loc[svm]
    lines += [
        "",
        "## Reading",
        "",
        f"- **Best model with the nested threshold**: {nested.loc[best, 'algorithm']} · {modes[nested.loc[best, 'mode']]} "
        f"(event F1 {nested.loc[best, 'event_f1']:.3f}, {nested.loc[best, 'found']} of 93 found, {nested.loc[best, 'false_alarms']} false alarms).",
        f"- **SVM unbalanced**: {s_old['event_f1']:.3f} → {s_new['event_f1']:.3f} "
        f"({s_old['found']} → {s_new['found']} found, {s_old['false_alarms']} → {s_new['false_alarms']} false alarms). "
        f"This is the reference value for every new model.",
        f"- **Optimism of the previous numbers**: event F1 drops by {drop.median():.3f} in median over the {len(drop)} models "
        f"(from {drop.min():+.3f} to {drop.max():+.3f}; a negative value means the nested threshold did better).",
        "- **Spread of the nested thresholds**: when they vary a lot between folds, the model's probabilities are not stable from one group of patients to another, "
        "and its threshold is fragile.",
        "",
        "Reproduction: `python scripts/evaluate_nested.py` (inner thresholds cached in `outputs/nested/`; delete a file there to recompute that model). "
        "Table: `outputs/nested_evaluation.csv`.",
        "",
    ]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
