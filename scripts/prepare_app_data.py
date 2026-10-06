"""Precompute the files read by app.py.

Filtered EEG signals, reference channels (ECG, EMG, SLI: shown in the app, never used as
features), recording metadata, technician events and, when the model registry exists, the
comparison curves of the 8 models. Run after scripts/train_registry.py.
"""

from pathlib import Path
import json
import re
import sys

from matio import load_from_mat
from openpyxl import load_workbook
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import precision_recall_curve

from src.detection import detect_all, event_metrics, sweep_thresholds
from src.io import CHANNELS, FS
from src.preprocessing import bandpass


APP_DATA = ROOT / "data/processed/app"
SIGNALS = APP_DATA / "signals"
REFERENCE = APP_DATA / "reference"
REGISTRY = ROOT / "models/registry.json"
# Raw column names of the non-EEG channels; EMG is spelled two ways and absent from most recordings.
REFERENCE_CHANNELS = {"ECG": "ECG", "EMG1": "EMG1", "EMG_1_": "EMG1", "EMG2": "EMG2", "EMG_2": "EMG2", "SLI": "SLI"}
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
OFFSETS = ROOT / "data/interim/official/offsets.csv"
# First matching pattern wins.
EVENT_CATEGORIES = [
    ("Seizure note", r"absence|abscence|absnce|crise|clonies"),
    ("HPN", r"^hpn"),
    ("SLI", r"^sli"),
    ("Eyes open (YO)", r"^yo\b"),
    ("Eyes closed (YF)", r"^yf\b"),
    ("Artifact", r"artefact|saturation"),
    ("Movement", r"bouge|mvts"),
]


def categorize(text: str) -> str:
    lowered = text.lower()
    for category, pattern in EVENT_CATEGORIES:
        if re.search(pattern, lowered):
            return category
    return "Other"


def repair_accents(text: str) -> str:
    """Accented characters were lost when the headers were exported; restore them.

    Between letters the lost character is "é" in every observed label; standalone it is "à".
    """
    lost = "�+"
    text = re.sub(f"(?<=[A-Z]){lost}(?=[A-Z])", "É", text)
    text = re.sub(f"(?<=[A-Za-z]){lost}(?=[a-z])", "é", text)
    return re.sub(lost, "à", text)


def technician_events(recording: str) -> pd.DataFrame:
    header = load_from_mat(ROOT / f"data/raw/{recording}_0000h.mat", variable_names=["output_h"])["output_h"]
    labels = [repair_accents(str(value)).strip() for value in header["Annotations"]]
    frame = pd.DataFrame({"recording": recording, "time_s": header.index.total_seconds(), "label": labels})
    frame = frame.loc[frame["label"] != ""]
    frame["category"] = frame["label"].map(categorize)
    frame["label"] = frame["label"].str.slice(0, 80)
    return frame


def reference_signals(recording: str) -> pd.DataFrame:
    """ECG, EMG and SLI as recorded, minus their median. A constant channel is dropped."""
    table = load_from_mat(ROOT / f"data/raw/{recording}_0000d.mat", variable_names=["output_d"])["output_d"]
    columns = {}
    for raw_name, name in REFERENCE_CHANNELS.items():
        if raw_name not in table.columns:
            continue
        values = np.concatenate([np.asarray(block).reshape(-1) for block in table[raw_name]]).astype(np.float32)
        if values.max() > values.min():
            columns[name] = values - np.median(values)
    return pd.DataFrame(columns)


def model_curves(annotations: pd.DataFrame) -> None:
    """Threshold sweeps, precision-recall curves and per-recording results of every model."""
    registry = json.loads(REGISTRY.read_text())
    thresholds = np.round(np.arange(0.05, 0.995, 0.01), 2)
    sweeps, curves, per_recording = [], [], []
    for key, entry in registry["models"].items():
        table = pd.read_parquet(ROOT / entry["predictions"])
        truth = table["true_class"].to_numpy() == 1
        proba = table["proba"].to_numpy()
        sweep = sweep_thresholds(table, annotations, thresholds)[["threshold", "detections", "tp", "fn", "fp", "precision", "recall", "f1"]]
        window = []
        for threshold in thresholds:
            predicted = proba >= threshold
            tp, fp, fn = (predicted & truth).sum(), (predicted & ~truth).sum(), (~predicted & truth).sum()
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn)
            window.append((precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0))
        sweep[["window_precision", "window_recall", "window_f1"]] = window
        sweeps.append(sweep.assign(model=key))

        precision, recall, cut = precision_recall_curve(truth, proba)
        curve = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": cut})
        # Keep at most ~300 points per curve: enough to draw it, light to send to the browser.
        curves.append(curve.iloc[:: max(1, len(curve) // 300)].assign(model=key))

        detected, matches = detect_all(table, annotations, entry["best_threshold"])
        counts = detected.groupby("recording").size()
        for recording, group in matches.groupby("recording"):
            metrics = event_metrics(group, int(counts.get(recording, 0)))
            per_recording.append(
                {"model": key, "recording": recording, "found": metrics["tp"], "missed": metrics["fn"], "false_alarms": metrics["fp"], "event_f1": metrics["f1"]}
            )
    pd.concat(sweeps, ignore_index=True).to_parquet(APP_DATA / "model_sweeps.parquet", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(APP_DATA / "model_pr_curves.parquet", index=False)
    pd.DataFrame(per_recording).to_csv(APP_DATA / "model_per_recording.csv", index=False)
    print(f"Comparison curves for {len(registry['models'])} models")


def workbook_rows(path: Path) -> dict[int, dict]:
    sheet = load_workbook(path, data_only=True).active
    rows = {}
    for number in range(2, sheet.max_row + 1):
        if sheet.cell(number, 1).value is None:
            if number > 200:
                break
            continue
        rows[number] = {
            "sex": sheet.cell(number, 2).value,
            "age": sheet.cell(number, 3).value,
            "absence_type": sheet.cell(number, 5).value,
        }
    return rows


def main() -> None:
    SIGNALS.mkdir(parents=True, exist_ok=True)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(ANNOTATIONS)
    methods = pd.read_csv(OFFSETS).set_index("recording")["method"]
    official = workbook_rows(ROOT / "Annotations MAJ.xlsx")
    # Recordings missing from the updated workbook keep the row of the original one.
    original = workbook_rows(ROOT / "data/raw/Annotations.xlsx")
    metadata, events = [], []
    for recording, intervals in annotations.groupby("recording", sort=True):
        frame = pd.read_parquet(ROOT / f"data/processed/official/{recording}.parquet")
        filtered = bandpass(frame.loc[:, CHANNELS].to_numpy(dtype=np.float32))
        output = pd.DataFrame(filtered, columns=CHANNELS)
        output.insert(0, "Time", frame["Time"].to_numpy())
        output["Class"] = frame["Class"].to_numpy()
        output.to_parquet(SIGNALS / f"{recording}.parquet", index=False, compression="zstd")

        reference = reference_signals(recording)
        reference.to_parquet(REFERENCE / f"{recording}.parquet", index=False, compression="zstd")

        source_row = int(intervals["source_row"].iloc[0])
        person = (official if methods[recording] == "official" else original)[source_row]
        metadata.append(
            {
                "recording": recording,
                "sex": person["sex"],
                "age": person["age"],
                "absence_type": person["absence_type"] or "",
                "duration_s": len(frame) / FS,
                "seizures": len(intervals),
                "seizure_seconds": float(intervals["duration_s"].sum()),
                "annotation_source": methods[recording],
                "reference_channels": " ".join(reference.columns),
            }
        )
        events.append(technician_events(recording))
        print(f"{recording}: {len(frame):,} samples filtered, {len(events[-1])} technician events, reference: {list(reference.columns)}", flush=True)
    pd.DataFrame(metadata).to_csv(APP_DATA / "recording_metadata.csv", index=False)
    pd.concat(events, ignore_index=True).to_csv(APP_DATA / "technician_events.csv", index=False)
    if REGISTRY.exists():
        model_curves(annotations)
    else:
        print("models/registry.json not found: run scripts/train_registry.py, then this script again, for the comparison page.")
    print(f"App data: {APP_DATA}")


if __name__ == "__main__":
    main()
