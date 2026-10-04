"""Convert aligned EEG recordings to Excel and per-recording Parquet files."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import annotations_for_recording, load_annotations, load_recording, recording_ids
from src.labels import add_class_column


RAW = ROOT / "data/raw"
ANNOTATIONS = ROOT / "data/interim/annotations_clean.csv"
EXCLUSIONS = ROOT / "data/interim/exclusions.csv"
EXCEL_DIR = ROOT / "data/excel"
PROCESSED_DIR = ROOT / "data/processed"
EXCEL_MAX_DATA_ROWS = 1_048_575  # One row is reserved for column names.


def write_excel(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp.xlsx")
    workbook = xlsxwriter.Workbook(
        temporary,
        {"constant_memory": True, "use_zip64": True, "strings_to_urls": False},
    )
    try:
        data = frame.to_numpy(copy=False)
        for sheet_index, first in enumerate(range(0, len(frame), EXCEL_MAX_DATA_ROWS), 1):
            last = min(first + EXCEL_MAX_DATA_ROWS, len(frame))
            worksheet = workbook.add_worksheet(f"EEG_{sheet_index}")
            worksheet.write_row(0, 0, frame.columns.tolist())
            for excel_row, values in enumerate(data[first:last], 1):
                worksheet.write_row(excel_row, 0, values.tolist())
            worksheet.freeze_panes(1, 1)
            worksheet.autofilter(0, 0, last - first, len(frame.columns) - 1)
            worksheet.set_column(0, 0, 13)
            worksheet.set_column(1, len(frame.columns) - 2, 12)
            worksheet.set_column(len(frame.columns) - 1, len(frame.columns) - 1, 8)
    finally:
        workbook.close()
    temporary.replace(path)


def convert_recording(recording: str, force: bool = False) -> None:
    excel_path = EXCEL_DIR / f"{recording}.xlsx"
    parquet_path = PROCESSED_DIR / f"{recording}.parquet"
    if not force and excel_path.exists() and parquet_path.exists():
        print(f"{recording}: outputs already exist; skipped", flush=True)
        return
    frame = load_recording(RAW / f"{recording}_0000d.mat")
    annotations = load_annotations(ANNOTATIONS)
    intervals = annotations_for_recording(annotations, recording)
    if intervals.empty:
        raise ValueError(f"{recording}: included recording has no cleaned seizure annotations")
    frame = add_class_column(frame, intervals)

    parquet_temp = parquet_path.with_name(parquet_path.stem + ".tmp.parquet")
    frame.to_parquet(parquet_temp, index=False, engine="pyarrow", compression="zstd")
    parquet_temp.replace(parquet_path)
    write_excel(frame, excel_path)
    print(
        f"{recording}: {len(frame):,} samples, Class 1={int(frame['Class'].sum()):,}, "
        f"{excel_path.name}, {parquet_path.name}",
        flush=True,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--recording", action="append", help="Convert only this recording ID; repeatable")
    parser.add_argument("--force", action="store_true", help="Replace existing conversion outputs")
    args = parser.parse_args()
    EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    excluded = set(pd.read_csv(EXCLUSIONS)["recording"])
    requested = args.recording or recording_ids(RAW)
    selected = [recording for recording in requested if recording not in excluded]
    unknown = set(requested).difference(recording_ids(RAW))
    if unknown:
        raise ValueError(f"Unknown recording IDs: {sorted(unknown)}")
    print(f"Excluded: {sorted(excluded)}", flush=True)
    for recording in selected:
        convert_recording(recording, args.force)


if __name__ == "__main__":
    main()

