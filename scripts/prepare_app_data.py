"""Precompute the files read by app.py: filtered signals, recording metadata, technician events."""

from pathlib import Path
import re
import sys

from matio import load_from_mat
from openpyxl import load_workbook
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import CHANNELS, FS
from src.preprocessing import bandpass


APP_DATA = ROOT / "data/processed/app"
SIGNALS = APP_DATA / "signals"
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
            }
        )
        events.append(technician_events(recording))
        print(f"{recording}: {len(frame):,} samples filtered, {len(events[-1])} technician events", flush=True)
    pd.DataFrame(metadata).to_csv(APP_DATA / "recording_metadata.csv", index=False)
    pd.concat(events, ignore_index=True).to_csv(APP_DATA / "technician_events.csv", index=False)
    print(f"App data: {APP_DATA}")


if __name__ == "__main__":
    main()
