"""Part X: out-of-fold seizure probabilities per recording and the final Random Forest.

Also writes out-of-fold probabilities for the four models of Parts VIII–IX, so the app can
compare them on recordings they never saw.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import detect_all, event_metrics, sweep_thresholds
from src.preprocessing import describe
from train_models import models as candidate_models


SEED = 42
METADATA = {"recording", "subject", "window", "start_s", "end_s", "class"}
DATASET = ROOT / "data/processed/windows_features_official.parquet"
PREDICTIONS = ROOT / "outputs/predictions_oof.parquet"
MODEL = ROOT / "models/rf_final.joblib"
ALL_PREDICTIONS = ROOT / "outputs/predictions_oof_models.parquet"
COMPARISON = ROOT / "outputs/model_comparison.csv"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"


def model() -> RandomForestClassifier:
    # Same estimator as Parts VIII–IX: default hyperparameters, unbalanced training.
    return RandomForestClassifier(random_state=SEED, n_jobs=-1)


def compare_models(frame: pd.DataFrame, X: np.ndarray, y: np.ndarray, groups: np.ndarray, reference: pd.DataFrame) -> None:
    """Out-of-fold probabilities of every model (unbalanced training) and their best thresholds."""
    annotations = pd.read_csv(ANNOTATIONS)
    hours = frame.groupby("recording")["end_s"].max().sum() / 3600
    tables, rows = [], []
    for name, pipeline in candidate_models().items():
        if name == "SVM":
            # Platt scaling gives the SVM a probability output; the decision boundary is unchanged.
            pipeline.set_params(model__probability=True)
        proba = np.full(len(y), np.nan)
        for train, test in GroupKFold(n_splits=5).split(X, y, groups):
            proba[test] = pipeline.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        table = reference.drop(columns="proba").assign(model=name, proba=proba.astype(np.float32))
        tables.append(table)
        sweep = sweep_thresholds(table, annotations, np.round(np.arange(0.05, 0.955, 0.01), 2))
        threshold = float(sweep.loc[np.isclose(sweep["f1"], sweep["f1"].max()), "threshold"].min())
        detected, matches = detect_all(table, annotations, threshold)
        event = event_metrics(matches, len(detected))
        predicted = proba >= threshold
        tp = int((predicted & (y == 1)).sum()); fp = int((predicted & (y == 0)).sum()); fn = int((~predicted & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        rows.append(
            {
                "model": name, "threshold": threshold,
                "event_f1": event["f1"], "event_recall": event["recall"], "event_precision": event["precision"],
                "found": event["tp"], "missed": event["fn"], "false_alarms": event["fp"], "false_alarms_per_hour": event["fp"] / hours,
                "window_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "window_recall": recall, "window_precision": precision,
            }
        )
        print(f"{name}: threshold {threshold:.2f}, event F1 {event['f1']:.3f} (found {event['tp']}, missed {event['fn']}, false alarms {event['fp']})", flush=True)
    pd.concat(tables, ignore_index=True).to_parquet(ALL_PREDICTIONS, index=False, compression="zstd")
    pd.DataFrame(rows).to_csv(COMPARISON, index=False)
    print(f"All models: {ALL_PREDICTIONS}; comparison: {COMPARISON}")


def main() -> None:
    frame = pd.read_parquet(DATASET)
    features = [column for column in frame.columns if column not in METADATA]
    X = frame[features].to_numpy(dtype=np.float64)
    y = frame["class"].to_numpy(dtype=np.int64)
    groups = frame["recording"].to_numpy()

    proba = np.full(len(y), np.nan)
    fold_of = np.zeros(len(y), dtype=np.int8)
    # Same deterministic folds as scripts/train_models.py.
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        assert not set(groups[train]) & set(groups[test])
        proba[test] = model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        fold_of[test] = fold
        print(f"fold {fold}: {len(set(groups[test]))} recordings, {len(test):,} windows", flush=True)
    assert not np.isnan(proba).any()

    predictions = pd.DataFrame(
        {
            "recording": frame["recording"],
            "window": frame["window"],
            "start_s": frame["start_s"],
            "end_s": frame["end_s"],
            "true_class": frame["class"].astype(np.uint8),
            "proba": proba.astype(np.float32),
            "fold": fold_of,
        }
    )
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS, index=False, compression="zstd")
    print(f"Out-of-fold predictions: {PREDICTIONS} ({len(predictions):,} windows)")

    compare_models(frame, X, y, groups, predictions)

    final = model().fit(X, y)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final,
            "features": features,
            "preprocessing": describe("bandpass"),
            "window_seconds": 2.0,
            "hop_seconds": 1.0,
            "training_recordings": sorted(set(groups)),
            "sklearn_version": sklearn.__version__,
        },
        MODEL,
        compress=3,
    )
    print(f"Final model: {MODEL} ({MODEL.stat().st_size / 1e6:.1f} MB, {len(features)} features, {len(set(groups))} recordings)")


if __name__ == "__main__":
    main()
