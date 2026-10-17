# CNN weights

Compact CNNs trained on the cleaned 19-channel EEG (see `scripts/train_cnn.py`). The model
classes (`ShallowConvNet`, `EEGNet`) are defined in that script.

| Name | Architecture | Augmentation | Event F1 (nested threshold) |
|---|---|---|---:|
| `cnn_shallow_augmented` | ShallowConvNet | yes | 0.815 |
| `cnn_eegnet_augmented` | EEGNet | yes | 0.776 |
| `cnn_shallow_plain` | ShallowConvNet | no | 0.774 |

The SVM baseline reaches 0.894 (optimistic threshold; the CNNs' optimistic scores are
0.813-0.842). Details: `docs/DEEP_LEARNING_RESULTS.md`.

## Files

- `<name>_final.pt`, `<name>_final.onnx`: model trained on all 21 recordings, to use on new
  data. Input: `(batch, 19, 1024)`, 4 s at 256 Hz in the channel order of `src/io.py`
  (`CHANNELS`), band-pass filtered 0.5-40 Hz, then normalized per recording and channel:
  `(x - median) / (1.4826 * MAD)`, clipped at ±20. Output: 2 logits, class 1 = seizure.
  The window is centred on the 2 s window to classify; windows step by 1 s.
- `<name>_config.json`: hyperparameters, per-fold thresholds, scores. `final.threshold` is the
  median of the fold thresholds, a starting point for a decision threshold on new data.
- `<name>_history.json`: training curves.
- `<name>_validation.parquet`: probabilities on the recordings held out inside each training
  fold, from which the fold thresholds were chosen.
- `folds/<name>_fold<k>_seed<s>.pt`: the 15 models per run (5 folds × 3 seeds) that produced
  the out-of-fold predictions in `outputs/predictions/<name>.parquet`.

## Loading

```python
import sys, torch
sys.path.insert(0, "scripts")
from train_cnn import build

model = build("shallow", 1024)  # or "eegnet"
model.load_state_dict(torch.load("models/cnn/cnn_shallow_augmented_final.pt", map_location="cpu"))
model.eval()
```
