"""Part X: out-of-fold seizure probabilities per recording and the final Random Forest."""

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

from src.preprocessing import describe


SEED = 42
METADATA = {"recording", "subject", "window", "start_s", "end_s", "class"}
DATASET = ROOT / "data/processed/windows_features_official.parquet"
PREDICTIONS = ROOT / "outputs/predictions_oof.parquet"
MODEL = ROOT / "models/rf_final.joblib"


def model() -> RandomForestClassifier:
    # Same estimator as Parts VIII–IX: default hyperparameters, unbalanced training.
    return RandomForestClassifier(random_state=SEED, n_jobs=-1)


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
