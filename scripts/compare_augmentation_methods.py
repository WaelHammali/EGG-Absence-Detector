"""Compare every augmentation method against the unbalanced baseline, algorithm by algorithm.

Working/scratch comparison for the staged "test each method alone, then combine the winners"
experiment -- reads models/registry.json (written by scripts/train_registry.py) and prints a
plain table. docs/MODEL_COMPARISON.md (and its generator, scripts/compare_models.py) is the
polished, committed report; it is updated separately, by hand, once a result is worth keeping.

Usage:
    python scripts/compare_augmentation_methods.py
    python scripts/compare_augmentation_methods.py --out outputs/augmentation_methods_comparison.csv
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REGISTRY = ROOT / "models/registry.json"
ALGORITHMS = {"dt": "Decision Tree", "rf": "Random Forest", "knn": "KNN", "svm": "SVM"}


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "outputs/augmentation_methods_comparison.csv"))
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text())
    models = registry["models"]
    rows = []
    for algorithm, name in ALGORITHMS.items():
        baseline_key = f"{algorithm}_unbalanced"
        if baseline_key not in models:
            continue
        baseline = models[baseline_key]["at_best_threshold"]["event"]
        for key, entry in models.items():
            if not key.startswith(f"{algorithm}_augmented"):
                continue
            event = entry["at_best_threshold"]["event"]
            rows.append(
                {
                    "algorithm": name,
                    "mode": entry["mode"],
                    "label": entry["label"],
                    "threshold": entry["best_threshold"],
                    "event_f1": event["f1"],
                    "event_f1_vs_unbalanced": event["f1"] - baseline["f1"],
                    "found": event["tp"],
                    "missed": event["fn"],
                    "false_alarms": event["fp"],
                    "false_alarms_vs_unbalanced": event["fp"] - baseline["fp"],
                    "false_alarms_per_hour": event["false_alarms_per_hour"],
                    "fit_seconds_per_fold": entry["fit_seconds_per_fold"],
                }
            )
        rows.append(
            {
                "algorithm": name, "mode": "unbalanced", "label": models[baseline_key]["label"],
                "threshold": models[baseline_key]["best_threshold"], "event_f1": baseline["f1"],
                "event_f1_vs_unbalanced": 0.0, "found": baseline["tp"], "missed": baseline["fn"],
                "false_alarms": baseline["fp"], "false_alarms_vs_unbalanced": 0,
                "false_alarms_per_hour": baseline["false_alarms_per_hour"],
                "fit_seconds_per_fold": models[baseline_key]["fit_seconds_per_fold"],
            }
        )

    if not rows:
        raise SystemExit("No augmented* models found in models/registry.json yet -- train some first (see scripts/build_augmented.py and scripts/train_registry.py --modes <mode>).")

    table = pd.DataFrame(rows).sort_values(["algorithm", "event_f1"], ascending=[True, False]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    beats_baseline = table.loc[(table["mode"] != "unbalanced") & (table["event_f1_vs_unbalanced"] > 0)]
    if len(beats_baseline):
        print(f"\nBeats the unbalanced baseline ({len(beats_baseline)} rows):")
        print(beats_baseline[["algorithm", "label", "event_f1", "event_f1_vs_unbalanced"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("\nNo augmented mode beats its algorithm's unbalanced baseline yet.")
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
