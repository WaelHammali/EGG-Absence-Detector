"""Train the selectable models: 4 algorithms x 3 training modes.

For every model: out-of-fold probabilities with GroupKFold by recording (the same 5 folds as
the other parts), the threshold maximizing event-level F1, scores at that threshold and at
0.50, and a final model trained on all recordings. Balanced training keeps every class-1
training window plus as many random class-0 windows. Augmented training adds the extra seizure
windows of scripts/build_augmented.py for the training recordings only. Test data always keeps
real proportions and real windows.

--modes trains only some modes and keeps the other models of the existing registry.

Outputs: outputs/predictions/<algo>_<mode>.parquet, models/<algo>_<mode>.joblib and
models/registry.json.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import detect_all, event_metrics, sweep_thresholds
from src.io import CHANNELS
from src.paths import AUGMENTED as AUGMENTED_NAME, OFFICIAL, dataset_file
from src.preprocessing import describe
from train_models import METADATA, models as pipelines, undersample


DATASET = dataset_file(OFFICIAL)
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
PREDICTIONS = ROOT / "outputs/predictions"
MODELS = ROOT / "models"
ALGORITHMS = {"dt": "Decision Tree", "rf": "Random Forest", "knn": "KNN", "svm": "SVM"}
AUGMENTED = dataset_file(AUGMENTED_NAME)
MODES = ("unbalanced", "balanced", "augmented")
BALANCED_SEEDS = (0, 1, 2, 3, 4)
SAVED_SEED = 0
# Up to 0.99: models trained on balanced data need high thresholds on real proportions.
THRESHOLDS = np.round(np.arange(0.05, 0.995, 0.01), 2)


def pipeline(algorithm: str):
    estimator = pipelines()[ALGORITHMS[algorithm]]
    if algorithm == "svm":
        # Platt scaling gives the SVM a probability output; the decision boundary is unchanged.
        estimator.set_params(model__probability=True)
    return estimator


def out_of_fold(algorithm: str, X, y, groups, seed: int | None, extra=None) -> tuple[np.ndarray, np.ndarray, list[float], dict]:
    """Probabilities for every window from models that never saw its recording.

    ``seed`` selects balanced training (undersampling of each training fold); None keeps
    the training folds as they are. ``extra`` = (features, recordings) of augmented class-1
    windows: those of the training recordings are added to each training fold.
    """
    proba = np.full(len(y), np.nan)
    fold_of = np.zeros(len(y), dtype=np.int8)
    seconds, sizes = [], []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        assert not set(groups[train]) & set(groups[test])
        if seed is not None:
            train = train[undersample(y[train], seed)]
        X_train, y_train = X[train], y[train]
        if extra is not None:
            added = extra[0][np.isin(extra[1], groups[train])]
            assert not set(extra[1][np.isin(extra[1], groups[train])]) & set(groups[test])
            X_train, y_train = np.vstack([X_train, added]), np.r_[y_train, np.ones(len(added), dtype=y.dtype)]
        started = time.perf_counter()
        fitted = pipeline(algorithm).fit(X_train, y_train)
        seconds.append(time.perf_counter() - started)
        proba[test] = fitted.predict_proba(X[test])[:, 1]
        fold_of[test] = fold
        sizes.append((len(y_train), int(y_train.sum())))
    windows = float(np.mean([size for size, _ in sizes]))
    positives = float(np.mean([count for _, count in sizes]))
    training = {"windows": windows, "class_1": positives, "class_0": windows - positives, "class_1_share": positives / windows}
    return proba, fold_of, seconds, training


def window_scores(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    predicted = proba >= threshold
    truth = y == 1
    tp, fp, fn, tn = int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum()), int((~predicted & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / len(y),
    }


def evaluate(table: pd.DataFrame, annotations: pd.DataFrame, threshold: float, hours: float) -> dict:
    detected, matches = detect_all(table, annotations, threshold)
    event = event_metrics(matches, len(detected))
    event.update({"detections": len(detected), "false_alarms_per_hour": event["fp"] / hours})
    return {"threshold": threshold, "window": window_scores(table["true_class"].to_numpy(), table["proba"].to_numpy(), threshold), "event": event}


def best_threshold(table: pd.DataFrame, annotations: pd.DataFrame) -> float:
    """Highest event-level F1; ties go to the lowest threshold (higher recall)."""
    sweep = sweep_thresholds(table, annotations, THRESHOLDS)
    return float(sweep.loc[np.isclose(sweep["f1"], sweep["f1"].max()), "threshold"].min())


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="Training modes to train (default: all)")
    selected = parser.parse_args().modes
    frame = pd.read_parquet(DATASET)
    features = [column for column in frame.columns if column not in METADATA]
    X = frame[features].to_numpy(dtype=np.float64)
    y = frame["class"].to_numpy(dtype=np.int64)
    groups = frame["recording"].to_numpy()
    annotations = pd.read_csv(ANNOTATIONS)
    hours = float(frame.groupby("recording")["end_s"].max().sum() / 3600)
    reference = frame[["recording", "window", "start_s", "end_s"]].assign(true_class=frame["class"].astype(np.uint8))
    extra = None
    if "augmented" in selected:
        if not AUGMENTED.exists():
            raise SystemExit(f"{AUGMENTED.relative_to(ROOT)} not found: run python scripts/build_augmented.py")
        augmented = pd.read_parquet(AUGMENTED)
        extra = (augmented[features].to_numpy(dtype=np.float64), augmented["recording"].to_numpy())
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    registry = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "channels": list(CHANNELS),
        "features": features,
        "preprocessing": describe("bandpass"),
        "validation": "GroupKFold by recording, 5 folds; out-of-fold predictions",
        "balanced_training": "random undersampling of each training fold: all class-1 windows + as many class-0 windows",
        "augmented_training": "extra class-1 windows of the training recordings (0.25 s shifts and left-right mirror), from scripts/build_augmented.py",
        "augmented_dataset": str(AUGMENTED.relative_to(ROOT)),
        "balanced_seeds": list(BALANCED_SEEDS),
        "saved_seed": SAVED_SEED,
        "recorded_hours": hours,
        "sklearn_version": sklearn.__version__,
        "models": {},
    }
    previous = json.loads((MODELS / "registry.json").read_text())["models"] if (MODELS / "registry.json").exists() else {}
    for algorithm, name in ALGORITHMS.items():
        for mode in MODES:
            key = f"{algorithm}_{mode}"
            if mode not in selected:
                if key in previous:
                    registry["models"][key] = previous[key]
                continue
            seeds = BALANCED_SEEDS if mode == "balanced" else (None,)
            runs = []
            for seed in seeds:
                proba, fold_of, seconds, training = out_of_fold(algorithm, X, y, groups, seed, extra if mode == "augmented" else None)
                table = reference.assign(fold=fold_of, proba=proba.astype(np.float32))
                threshold = best_threshold(table, annotations)
                runs.append(
                    {
                        "seed": seed, "table": table, "seconds": seconds, "training": training,
                        "best": evaluate(table, annotations, threshold, hours), "half": evaluate(table, annotations, 0.5, hours),
                    }
                )
            saved = runs[0]
            saved["table"].to_parquet(PREDICTIONS / f"{key}.parquet", index=False, compression="zstd")

            # Final model on all recordings, with the same undersampling seed as the saved predictions.
            rows = undersample(y, SAVED_SEED) if mode == "balanced" else np.arange(len(y))
            X_final, y_final = X[rows], y[rows]
            if mode == "augmented":
                X_final, y_final = np.vstack([X_final, extra[0]]), np.r_[y_final, np.ones(len(extra[0]), dtype=y.dtype)]
            started = time.perf_counter()
            final = pipeline(algorithm).fit(X_final, y_final)
            final_seconds = time.perf_counter() - started
            joblib.dump(
                {
                    "model": final, "features": features, "algorithm": algorithm, "mode": mode,
                    "threshold": saved["best"]["threshold"], "preprocessing": describe("bandpass"),
                    "window_seconds": 2.0, "hop_seconds": 1.0, "sklearn_version": sklearn.__version__,
                },
                MODELS / f"{key}.joblib", compress=3,
            )
            entry = {
                "algorithm": algorithm, "algorithm_name": name, "mode": mode, "label": f"{name} · {mode.capitalize()}",
                "predictions": f"outputs/predictions/{key}.parquet", "model_file": f"models/{key}.joblib",
                "best_threshold": saved["best"]["threshold"],
                "training": saved["training"],
                "fit_seconds_per_fold": float(np.mean(saved["seconds"])),
                "final_fit_seconds": final_seconds,
                "at_best_threshold": saved["best"], "at_0.50": saved["half"],
            }
            if mode == "balanced":
                def spread(getter) -> dict:
                    values = [getter(run) for run in runs]
                    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": [float(value) for value in values]}
                entry["seeds"] = {
                    "best_threshold": spread(lambda run: run["best"]["threshold"]),
                    "event_f1_best": spread(lambda run: run["best"]["event"]["f1"]),
                    "event_recall_best": spread(lambda run: run["best"]["event"]["recall"]),
                    "event_precision_best": spread(lambda run: run["best"]["event"]["precision"]),
                    "false_alarms_best": spread(lambda run: run["best"]["event"]["fp"]),
                    "window_f1_best": spread(lambda run: run["best"]["window"]["f1"]),
                    "event_f1_0.50": spread(lambda run: run["half"]["event"]["f1"]),
                    "window_f1_0.50": spread(lambda run: run["half"]["window"]["f1"]),
                    "window_recall_0.50": spread(lambda run: run["half"]["window"]["recall"]),
                    "window_precision_0.50": spread(lambda run: run["half"]["window"]["precision"]),
                }
            registry["models"][key] = entry
            event = saved["best"]["event"]
            print(
                f"{key}: threshold {saved['best']['threshold']:.2f}, event F1 {event['f1']:.3f} "
                f"(found {event['tp']}, missed {event['fn']}, false alarms {event['fp']}), window F1 {saved['best']['window']['f1']:.3f}, "
                f"fit {entry['fit_seconds_per_fold']:.1f} s/fold, model file {(MODELS / f'{key}.joblib').stat().st_size / 1e6:.1f} MB",
                flush=True,
            )
    registry["default_model"] = max(registry["models"], key=lambda key: registry["models"][key]["at_best_threshold"]["event"]["f1"])
    # default=float converts any remaining NumPy scalar.
    (MODELS / "registry.json").write_text(json.dumps(registry, indent=2, default=float) + "\n")
    print(f"Default model (best event F1): {registry['default_model']}")
    print(f"Registry: {MODELS / 'registry.json'}")


if __name__ == "__main__":
    main()
