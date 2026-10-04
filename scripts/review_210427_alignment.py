"""Compare the two defensible clock offsets for low-confidence 210427B_C."""

from pathlib import Path
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from align_clock_offsets import (
    CHANNELS,
    FS,
    interval_score,
    load_signal,
    raw_intervals,
    relative_band_power,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/figures/offset_alignment/210427B_C_candidate_comparison.png"
DATA_OFFSET = 32333.0
HEADER_OFFSET = 31839.0


def main() -> None:
    summary = json.loads((ROOT / "docs/inspection_summary.json").read_text())
    interval = raw_intervals(summary)["210427B_C"][0][0]
    signal = load_signal("210427B_C")
    times, power = relative_band_power(signal)
    prefix = np.r_[0.0, np.cumsum(power, dtype=np.float64)]
    candidates = [("Data-driven", DATA_OFFSET), ("Header anchor", HEADER_OFFSET)]
    results = []
    for label, offset in candidates:
        score, power_ratio, count = interval_score(offset, [interval], times, prefix)
        results.append(
            {
                "label": label,
                "offset": offset,
                "start_s": interval.start_clock - offset,
                "end_s": interval.end_clock - offset,
                "score": score,
                "inside_outside_ratio": power_ratio,
                "window_count": count,
            }
        )

    channel_names = ["EEGFp1", "EEGC3", "EEGO1"]
    channel_indices = [CHANNELS.index(channel) for channel in channel_names]
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    for axis, result in zip(axes, results):
        start = result["start_s"]
        end = result["end_s"]
        left = max(0, int(math.floor((start - 10) * FS)))
        right = min(len(signal), int(math.ceil((end + 10) * FS)))
        elapsed = np.arange(left, right) / FS
        segment = signal[left:right, channel_indices].astype(np.float64)
        scale = np.median(np.std(segment, axis=0))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        vertical_offsets = [6.0, 3.0, 0.0]
        for column, vertical in enumerate(vertical_offsets):
            axis.plot(elapsed, segment[:, column] / scale + vertical, linewidth=0.6)
        axis.axvspan(start, end, color="orange", alpha=0.28)
        axis.set_xlim(max(0, start - 10), min(len(signal) / FS, end + 10))
        axis.set_yticks(vertical_offsets, [value.removeprefix("EEG") for value in channel_names])
        axis.set(
            title=(
                f"{result['label']} offset {result['offset']:.2f} s: interval "
                f"{start:.2f}–{end:.2f} s; score={result['score']:.6f}, "
                f"inside/outside={result['inside_outside_ratio']:.3f}"
            ),
            xlabel="Elapsed time (s)",
            ylabel="Channels (scaled)",
        )
    figure.suptitle("210427B_C: data-driven versus header-anchored alignment", fontsize=15)
    figure.savefig(OUTPUT, dpi=160)
    plt.close(figure)
    result_path = ROOT / "docs/210427_alignment_comparison.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result)
    print(OUTPUT)


if __name__ == "__main__":
    main()
