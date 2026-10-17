"""Blend out-of-fold probabilities of several models and score the blend.

The cheapest way to try to beat a single model (research report, part A7): average the
probabilities of two models that make different kinds of mistakes, window by window. No
retraining. Works on any mix of the 12 registered classical models and external predictions
with the same schema (recording, window, start_s, end_s, true_class, fold, proba) — in
particular the CNN predictions in outputs/predictions/cnn_*.parquet.

Threshold, two ways (no model here was retrained, so there is nothing to choose a threshold
"inside"; both avoid ever using a fold's own predictions to pick its threshold):
- held-out-fold (default, the number to trust): for each outer fold, the threshold is swept on
  the OTHER 4 folds' out-of-fold predictions only, then applied to this fold.
- optimistic: one threshold swept on all out-of-fold predictions at once (reference only, same
  convention as the registered models' own best_threshold).

Usage:
    python scripts/ensemble_predictions.py --models svm_unbalanced rf_unbalanced
    python scripts/ensemble_predictions.py --models svm_unbalanced \
        --extra outputs/predictions/cnn_shallow_augmented.parquet
    python scripts/ensemble_predictions.py --models svm_unbalanced rf_unbalanced --weights 2 1

Outputs: outputs/ensemble_evaluation.csv, docs/ENSEMBLE_RESULTS.md.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.detection import detect_all, event_metrics
from evaluate_nested import szcore
from train_registry import THRESHOLDS


REGISTRY = ROOT / "models/registry.json"
ANNOTATIONS = ROOT / "data/interim/official/annotations_clean.csv"
APP_METADATA = ROOT / "data/processed/app/recording_metadata.csv"
TABLE = ROOT / "outputs/ensemble_evaluation.csv"
REPORT = ROOT / "docs/ENSEMBLE_RESULTS.md"
KEY = ["recording", "window"]


def load_source(name: str, registry: dict) -> tuple[str, pd.DataFrame]:
    """A registered model key, or a path to an external predictions parquet."""
    if name in registry["models"]:
        return registry["models"][name]["label"], pd.read_parquet(ROOT / registry["models"][name]["predictions"])
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"'{name}' is neither a registered model ({', '.join(registry['models'])}) nor an existing file.")
    return path.stem, pd.read_parquet(path)


def blend(sources: list[tuple[str, pd.DataFrame]], weights: list[float]) -> pd.DataFrame:
    base = sources[0][1][["recording", "window", "start_s", "end_s", "true_class", "fold"]]
    total = np.zeros(len(base))
    for (label, table), weight in zip(sources, weights):
        merged = base[KEY].merge(table[[*KEY, "proba"]], on=KEY, how="left")
        if merged["proba"].isna().any():
            raise SystemExit(f"'{label}' is missing {merged['proba'].isna().sum()} windows present in the first source: not the same window grid.")
        total += weight * merged["proba"].to_numpy()
    return base.assign(proba=(total / sum(weights)).astype(np.float32))


def sweep(table: pd.DataFrame, annotations: pd.DataFrame) -> tuple[float, dict]:
    best_threshold, best_event = THRESHOLDS[0], None
    for threshold in THRESHOLDS:
        detected, matches = detect_all(table, annotations, float(threshold))
        event = event_metrics(matches, len(detected))
        if best_event is None or event["f1"] > best_event["f1"]:
            best_threshold, best_event = float(threshold), event
    return best_threshold, best_event


def held_out_fold_score(table: pd.DataFrame, annotations: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    """Each outer fold scored with a threshold swept on the other folds only."""
    thresholds = {}
    for fold in sorted(table["fold"].unique()):
        threshold, _ = sweep(table.loc[table["fold"] != fold], annotations)
        thresholds[int(fold)] = threshold
    binarized = table.copy()
    for fold, threshold in thresholds.items():
        rows = binarized["fold"] == fold
        binarized.loc[rows, "proba"] = np.where(binarized.loc[rows, "proba"] >= threshold, 1.0, -1.0)
    detected, matches = detect_all(binarized, annotations, 0.0)
    return event_metrics(matches, len(detected)), detected, thresholds


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=[], help="Registered model keys to blend (models/registry.json)")
    parser.add_argument("--extra", nargs="*", default=[], help="External predictions.parquet files to blend in (e.g. a CNN run)")
    parser.add_argument("--weights", nargs="*", type=float, help="One weight per source, in the order --models then --extra (default: equal)")
    parser.add_argument("--name", help="Label for the report (default: built from the sources)")
    args = parser.parse_args()
    names = args.models + args.extra
    if len(names) < 2:
        raise SystemExit("Give at least 2 sources with --models and/or --extra.")
    registry = json.loads(REGISTRY.read_text())
    weights = args.weights or [1.0] * len(names)
    if len(weights) != len(names):
        raise SystemExit(f"--weights needs one value per source ({len(names)} sources, {len(weights)} weights).")

    sources = [load_source(name, registry) for name in names]
    blended = blend(sources, weights)
    label = args.name or " + ".join(f"{w:g}×{l}" if w != 1 else l for l, w in zip((s[0] for s in sources), weights))
    print(f"Ensemble: {label} ({len(names)} sources, {len(blended):,} windows)", flush=True)

    annotations = pd.read_csv(ANNOTATIONS)
    durations = pd.read_csv(APP_METADATA).set_index("recording")["duration_s"]
    hours = durations.sum() / 3600

    held_out, detected, fold_thresholds = held_out_fold_score(blended, annotations)
    optimistic_threshold, optimistic = sweep(blended, annotations)
    sz = szcore(detected, annotations, durations)

    print(f"  held-out-fold event F1 {held_out['f1']:.3f} (found {held_out['tp']}/{held_out['tp'] + held_out['fn']}, "
          f"false alarms {held_out['fp']}, {held_out['fp'] / hours:.1f}/h); SzCORE event F1 {sz['event']['f1']:.3f}")
    print(f"  optimistic        event F1 {optimistic['f1']:.3f} at {optimistic_threshold:.2f} "
          f"(found {optimistic['tp']}, false alarms {optimistic['fp']})")

    registry_default = registry["models"][registry["default_model"]]
    baseline = registry_default["at_best_threshold"]["event"]
    rows = [
        {"name": registry["default_model"], "label": registry_default["label"], "kind": "baseline (registry, optimistic)",
         "event_f1": baseline["f1"], "found": baseline["tp"], "missed": baseline["fn"], "false_alarms": baseline["fp"],
         "false_alarms_per_hour": baseline["false_alarms_per_hour"]},
        {"name": "ensemble", "label": label, "kind": "held_out_fold", "event_f1": held_out["f1"], "found": held_out["tp"],
         "missed": held_out["fn"], "false_alarms": held_out["fp"], "false_alarms_per_hour": held_out["fp"] / hours,
         "szcore_event_f1": sz["event"]["f1"], "szcore_sensitivity": sz["event"]["sensitivity"], "fold_thresholds": json.dumps(fold_thresholds)},
        {"name": "ensemble", "label": label, "kind": "optimistic", "event_f1": optimistic["f1"], "found": optimistic["tp"],
         "missed": optimistic["fn"], "false_alarms": optimistic["fp"], "false_alarms_per_hour": optimistic["fp"] / hours, "threshold": optimistic_threshold},
    ]
    table = pd.DataFrame(rows)
    if TABLE.exists():
        # Keep earlier ensembles tried in previous runs, drop any row for this exact combination
        # (so re-running the same --models/--extra replaces it) and the old baseline row (replaced below).
        existing = pd.read_csv(TABLE)
        existing = existing.loc[(existing["kind"] != "baseline (registry, optimistic)") & (existing["label"] != label)]
        table = pd.concat([existing, table], ignore_index=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    write_report()
    print(f"\nTable: {TABLE}\nReport: {REPORT}")


def write_report() -> None:
    table = pd.read_csv(TABLE)
    baseline = table.loc[table["kind"] == "baseline (registry, optimistic)"].iloc[0]
    held_out = table.loc[table["kind"] == "held_out_fold"].sort_values("event_f1", ascending=False)
    lines = [
        "# Ensemble results",
        "",
        "Blends of out-of-fold probabilities from several already-trained models (no retraining): "
        "research report, part A7, \"the cheapest way to get a real gain\". Built with `python scripts/ensemble_predictions.py`.",
        "",
        "- **held-out-fold** (main number): for each outer fold, the threshold is swept on the other 4 folds' predictions only, "
        "never on the fold being scored — valid even though nothing was retrained for this fold.",
        "- **optimistic**: one threshold swept on all out-of-fold predictions at once, kept for reference "
        "(the same convention as the registered models' own best threshold).",
        "",
        "## Results",
        "",
        "| Ensemble | Event F1 (held-out-fold) | Found (of 93) | False alarms | FA / hour | SzCORE event F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| **{baseline['label']}** (single model, baseline) | **{baseline['event_f1']:.3f}** | {baseline['found']} | "
        f"{baseline['false_alarms']} | {baseline['false_alarms_per_hour']:.1f} | — |",
    ]
    for _, row in held_out.iterrows():
        lines.append(
            f"| {row['label']} | {row['event_f1']:.3f} | {row['found']} | {row['false_alarms']} | "
            f"{row['false_alarms_per_hour']:.1f} | {row['szcore_event_f1']:.3f} |"
        )
    best = held_out.iloc[0] if len(held_out) else None
    lines += ["", "## Reading", ""]
    if best is not None:
        gap = baseline["event_f1"] - best["event_f1"]
        if gap > 0.01:
            lines.append(f"- No blend tried yet beats {baseline['label']} ({baseline['event_f1']:.3f}); the best, {best['label']}, reaches {best['event_f1']:.3f}.")
        elif gap > -0.01:
            lines.append(
                f"- **{best['label']}** ties the baseline ({best['event_f1']:.3f} vs {baseline['event_f1']:.3f}), with {best['found']} found "
                f"and {best['false_alarms']} false alarms against the baseline's {baseline['found']} found and {baseline['false_alarms']}. "
                "A tie with a different error profile can still be worth reporting, but is not on its own a reason to switch the default model."
            )
        else:
            lines.append(
                f"- **{best['label']}** beats the baseline by {-gap:.3f} ({best['event_f1']:.3f} vs {baseline['event_f1']:.3f}), "
                f"{best['found']} found and {best['false_alarms']} false alarms against {baseline['found']} and {baseline['false_alarms']}. "
                "Worth proposing as the new default if this holds up (it uses no new data, just two existing models together)."
            )
    lines += [
        "",
        "Reproduction: `python scripts/ensemble_predictions.py --models <key> <key> ... --extra <path.parquet> ...` "
        "(see the script's docstring). Raw numbers: `outputs/ensemble_evaluation.csv`.",
        "",
    ]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
