"""Plot one seizure ±10 s before and after the band-pass preprocessing."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import CHANNELS, FS, annotations_for_recording
from src.preprocessing import BANDPASS_HZ, BANDPASS_ORDER, bandpass


TRACE = "#2a78d6"
SEIZURE = "#eb6834"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--recording", default="211104B_D")
    parser.add_argument("--seizure", type=int, default=3, help="seizure_id in the official annotations")
    parser.add_argument("--channels", nargs=3, default=["Fp1", "C3", "O1"])
    parser.add_argument("--context", type=float, default=10.0)
    args = parser.parse_args()

    frame = pd.read_parquet(ROOT / f"data/processed/official/{args.recording}.parquet")
    raw = frame.loc[:, CHANNELS].to_numpy(dtype=np.float32)
    # Filter the whole recording, exactly as the dataset build does, then crop.
    filtered = bandpass(raw)
    annotations = pd.read_csv(ROOT / "data/interim/official/annotations_clean.csv")
    seizure = annotations_for_recording(annotations, args.recording).set_index("seizure_id").loc[args.seizure]
    first = max(0, int((seizure.start_s - args.context) * FS))
    last = min(len(frame), int((seizure.end_s + args.context) * FS))
    time = frame["Time"].to_numpy()[first:last]

    figure, axes = plt.subplots(3, 2, figsize=(15, 8), sharex=True, constrained_layout=True)
    for row, channel in enumerate(args.channels):
        index = CHANNELS.index(channel)
        for column, (signal, title) in enumerate([(raw, "Avant filtrage (signal brut)"), (filtered, "Après filtrage passe-bande")]):
            axis = axes[row, column]
            values = signal[first:last, index]
            axis.axvspan(seizure.start_s, seizure.end_s, color=SEIZURE, alpha=0.15, linewidth=0, label="Crise annotée")
            axis.plot(time, values, color=TRACE, linewidth=0.8)
            axis.axhline(0, color="#52514e", linewidth=0.6, linestyle=":")
            axis.set_ylabel(f"{channel}\namplitude (unité non renseignée)")
            axis.text(
                0.01, 0.95, f"moyenne = {abs(values.mean()) if abs(values.mean()) < 0.5 else values.mean():.0f}", transform=axis.transAxes, va="top", fontsize=9, color="#52514e",
            )
            axis.grid(axis="y", color="#e6e5e0", linewidth=0.6)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if row == 0:
                axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Temps depuis le début de l’enregistrement (s)")
    axes[0, 1].legend(loc="upper right", frameon=False)
    low, high = BANDPASS_HZ
    figure.suptitle(
        f"{args.recording}, crise {args.seizure} ({seizure.start_s:g}–{seizure.end_s:g} s) ± {args.context:g} s — "
        f"Butterworth {low:g}–{high:g} Hz, ordre {BANDPASS_ORDER}, phase nulle",
        fontsize=13,
    )
    output = ROOT / "outputs/figures/official/preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.recording}_seizure{args.seizure}_filtering.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
