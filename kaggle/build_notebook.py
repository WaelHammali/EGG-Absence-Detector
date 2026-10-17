"""Build kaggle/absence_seizure_cnn.ipynb from kaggle/train_cnn.py (run locally, not on Kaggle).

Run after editing train_cnn.py, so the notebook's embedded copy never drifts out of sync:
    python kaggle/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "kaggle" / "train_cnn.py").read_text()

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

cells = []

cells.append(md("""# Absence-seizure detection — compact CNNs (GPU)

Trains deep models directly on the cleaned 19-channel EEG (no hand-made features), to compare
with the project's 12 classical models (best so far: **SVM · unbalanced, event F1 0.894**,
86/93 seizures found, 16 false alarms, threshold 0.82).

**Second version of the recipe.** A first run's per-fold thresholds landed at 0.91–0.99 —
probabilities compressed near 1, the same symptom the undersampled classical models showed —
and several folds' "best epoch" was 1–2, a sign the 3-recording validation set was too small
and noisy to trust. This version holds out 5 recordings instead of 3, lowers the seizure class
weight from 8 to 5, and will not stop early before epoch 5. If you already have a first run's
results, keep them — this one is for comparison, not a replacement.

**Before running:** Notebook settings (right panel) → **Accelerator: GPU T4 x2** (or P100) →
**Internet: off** (not needed, keeps the session reproducible) → **Add Input** → your private
dataset `eeg-absence` (built by `scripts/export_kaggle.py` in the project repo).

**What this notebook does**

1. Writes `train_cnn.py` (identical to the one in the project's `kaggle/` folder — keep both in
   sync if you edit either).
2. Quick check (~1 minute) that the pipeline runs end to end.
3. Three full runs, in the order recommended by the research report:
   `shallow` (plain) → `shallow` (augmented) → `eegnet` (augmented).
   Each outer fold also retrains 3 seeds (default); expect roughly 30–90 minutes per run on a
   T4, more on a shared/queued GPU. Stop after the first run if you are short on GPU quota —
   it already gives a fair comparison point.
4. A summary table against the SVM baseline.
5. Zips everything in `/kaggle/working/cnn_outputs` for download (Output tab → Download).

**After downloading:** copy the zip into the project under `outputs/kaggle/`, then run
`python scripts/evaluate_cnn.py` (project repo) to re-score the CNNs exactly like the other
12 models (nested threshold, SzCORE) and add them to `docs/NESTED_EVALUATION.md` /
the app's Comparison page."""))

cells.append(md("## 1. Check the GPU and the dataset"))
cells.append(code("""!nvidia-smi
import pathlib
for p in pathlib.Path("/kaggle/input").rglob("windows.csv"):
    print("dataset found:", p.parent)
"""))

cells.append(md("""## 2. Write `train_cnn.py`

This cell is generated from the project's `kaggle/train_cnn.py` — do not edit it here; edit the
file in the repository and regenerate this notebook, so the two never drift apart."""))

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (["%%writefile train_cnn.py\n"] + SCRIPT.splitlines(keepends=True)),
})

cells.append(md("## 3. Quick check — tiny run on CPU/GPU, a couple of minutes"))
cells.append(code("!python train_cnn.py --quick"))

cells.append(md("""## 4. Full run 1/3 — ShallowConvNet, no augmentation

The plain CNN, to see what raw-signal deep learning gets to on its own before adding anything."""))
cells.append(code("!python train_cnn.py --arch shallow"))

cells.append(md("""## 5. Full run 2/3 — ShallowConvNet, with augmentation

Amplitude scaling, small noise, channel dropout, left-right mirror, small time shift — applied
to **both classes**, so the seizure share stays near 4 % (unlike the first classical-model
augmentation attempt, which only added seizure windows and doubled the false alarms)."""))
cells.append(code("!python train_cnn.py --arch shallow --augment"))

cells.append(md("""## 6. Full run 3/3 — EEGNet, with augmentation

A second architecture for comparison. Skip this cell first if you are short on GPU time; the
two ShallowConvNet runs above are enough for a first comparison."""))
cells.append(code("!python train_cnn.py --arch eegnet --augment"))

cells.append(md("## 7. Summary: event F1 against the SVM baseline"))
cells.append(code("""import json, glob
import pandas as pd

SVM_BASELINE = {"name": "svm_unbalanced (project baseline)", "event_f1": 0.894, "found": 86, "missed": 7, "false_alarms": 16}

rows = [SVM_BASELINE | {"optimistic_f1": 0.894, "window_ap": None}]
for path in sorted(glob.glob("cnn_outputs/*_config.json")):
    c = json.load(open(path))
    rows.append({
        "name": c["name"], "event_f1": c["nested"]["f1"], "found": c["nested"]["found"],
        "missed": c["nested"]["missed"], "false_alarms": c["nested"]["false_alarms"],
        "optimistic_f1": c["optimistic"]["f1"], "window_ap": None,
    })
table = pd.DataFrame(rows).sort_values("event_f1", ascending=False)
print(table.to_string(index=False))
print()
print("event_f1 above uses each outer fold's own threshold (chosen on 3 held-out training")
print("recordings, never on the test fold) — the fair number to compare with the SVM.")
print("optimistic_f1 uses one threshold chosen on all out-of-fold predictions, like the")
print("project's original (slightly optimistic) scores — for reference only.")
"""))

cells.append(md("""## 8. Package everything for download

Download from the **Output** tab on the right (or the Data pane), then copy the zip into the
project repo under `outputs/kaggle/` and run `python scripts/evaluate_cnn.py` there."""))
cells.append(code("""import shutil
shutil.make_archive("/kaggle/working/cnn_outputs", "zip", "/kaggle/working/cnn_outputs")
print("Ready: /kaggle/working/cnn_outputs.zip")
"""))

import uuid
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:02d}-" + uuid.uuid4().hex[:8]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "kaggle" / "absence_seizure_cnn.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print("wrote", out, out.stat().st_size, "bytes")
