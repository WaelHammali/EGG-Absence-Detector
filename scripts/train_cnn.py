"""Train compact CNNs to detect absence seizures from the cleaned EEG (GPU recommended).

Self-contained: needs only numpy, pandas, pyarrow, scikit-learn and PyTorch, so it can be copied
to any GPU machine. Input: the folder written by scripts/export_cnn_data.py. It also holds the
model definitions (ShallowConvNet, EEGNet) needed to load the weights in models/cnn/.

Method (research report, part A8):
- Input: 4 s of the 19 filtered channels (1,024 samples) centred on each 2 s window of the
  official dataset, so the label is exactly the official one (at least 50 % of the central
  2 s is seizure) and the predictions line up with the 12 classical models.
- Normalization per recording and channel: (x - median) / (1.4826 · MAD), clipped at ±20.
- Same 5 outer folds as the classical models. In each outer fold, 5 training recordings
  (with seizures) are held out for early stopping and for choosing the threshold — a first
  run with only 3 gave a noisy, unstable signal (several folds "best epoch" was 1-2, and the
  per-fold thresholds this chose were compressed near 0.9-0.99, the same symptom the
  undersampled classical models showed).
- Real class proportions, weighted cross-entropy (seizure weight 5; a first run at 8 pushed
  probabilities too hard towards the extremes), AdamW + cosine schedule, early stopping on
  validation average precision, never before epoch --min-epochs (5 by default) so training
  does not lock onto a lucky noisy early checkpoint.
- Optional online augmentation of BOTH classes (the seizure share stays at about 4 %): amplitude
  scaling, noise, channel dropout, left-right mirror, time shift (the original five -- in the
  first run this combination was the one change that reliably helped, event F1 +0.04 on
  ShallowConvNet), plus time masking, frequency band-stop and mixup. Each has its own
  --aug-<name> flag so it can be tested alone or combined; --augment is a shorthand for the
  original five together, --aug-all enables everything.
- Several seeds per fold; probabilities averaged.
- Post-processing (smoothing + hysteresis) is not done here: it needs no retraining, so it is
  applied afterwards, automatically, by scripts/evaluate_cnn.py.

Bug fixed in this version: ``event_counts`` (used for every threshold choice and score) used
to index back into the table with the pandas index *labels* returned by
``groupby(...).groups``, which only happen to equal row positions when the table's index is
untouched 0..len(table)-1. That is false for the held-out validation predictions
(``windows.iloc[valid_rows]``, which keeps the original row numbers), so it crashed with
``IndexError: index 7943 is out of bounds for axis 0 with size 4344`` the first time a chosen
validation recording's original row number happened to exceed the validation subset's own
size — which earlier runs avoided only by luck of which recordings were randomly picked. Now
uses a boolean mask on arrays taken directly from the table, which is always correct
regardless of its index. Verified to reproduce the exact same numbers as before everywhere
the old code happened to work, and to no longer crash where it did.

Outputs in --out (default outputs/cnn/):
- predictions/<name>.parquet: out-of-fold probability of every window (same format as
  outputs/predictions/*.parquet of the project);
- predictions/<name>_validation.parquet: probabilities of the held-out training recordings,
  used to choose each fold's threshold without looking at the test fold;
- models/<name>_fold<k>_seed<s>.pt, and models/<name>_final.pt (+ .onnx when possible): weights;
- <name>_config.json and <name>_history.json.

Usage (about 30-40 minutes per run on a T4 GPU):
    python train_cnn.py --data data/cnn --arch shallow
    python train_cnn.py --data data/cnn --arch shallow --augment          # the original five, together
    python train_cnn.py --data data/cnn --arch eegnet --augment
    python train_cnn.py --data data/cnn --arch shallow --aug-noise        # one transform alone
    python train_cnn.py --data data/cnn --arch shallow --aug-time-mask
    python train_cnn.py --data data/cnn --arch shallow --aug-band-stop
    python train_cnn.py --data data/cnn --arch shallow --aug-mixup
    python train_cnn.py --data data/cnn --arch shallow --aug-all          # everything together
Quick check (minutes): python train_cnn.py --data data/cnn --quick
Copy predictions/<name>.parquet to outputs/predictions/ and the models/ and *_config.json files to
models/cnn/, then score with scripts/evaluate_cnn.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch import nn


FS = 256
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
MIRROR = {"Fp1": "Fp2", "F7": "F8", "F3": "F4", "T3": "T4", "C3": "C4", "T5": "T6", "P3": "P4", "O1": "O2"}
MIRROR.update({right: left for left, right in MIRROR.items()})
MIRROR_INDEX = [CHANNELS.index(MIRROR.get(channel, channel)) for channel in CHANNELS]
# Same post-processing as the project (src/detection.py).
MIN_GAP_S = 2.0
MIN_DURATION_S = 2.0
THRESHOLDS = np.round(np.arange(0.05, 0.995, 0.01), 2)
# The original --augment combo, in the exact order augment() applies them.
ORIGINAL_FIVE = ["amplitude", "noise", "channel_dropout", "mirror", "time_shift"]
AUG_FLAGS = ORIGINAL_FIVE + ["time_mask", "band_stop", "mixup"]
AUG_CODES = {
    "amplitude": "amp", "noise": "noise", "channel_dropout": "chdrop", "mirror": "mirror",
    "time_shift": "tshift", "time_mask": "tmask", "band_stop": "bstop", "mixup": "mixup",
}


# ----------------------------------------------------------------------------- arguments


def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Train compact CNNs for absence-seizure detection.")
    parser.add_argument("--data", default="data/cnn", help="Folder with windows.csv and signals/ (written by scripts/export_cnn_data.py)")
    parser.add_argument("--out", default="outputs/cnn")
    parser.add_argument("--arch", choices=["shallow", "eegnet"], default="shallow")
    parser.add_argument(
        "--augment", action="store_true",
        help="Shorthand for the original five transforms together: " + " ".join(f"--aug-{name.replace('_', '-')}" for name in ORIGINAL_FIVE),
    )
    for name in AUG_FLAGS:
        parser.add_argument(f"--aug-{name.replace('_', '-')}", action="store_true", help=f"Online {name.replace('_', ' ')} augmentation alone (both classes alike)")
    parser.add_argument("--aug-all", action="store_true", help="Enable every transform above, original five and new ones")
    parser.add_argument("--window-s", type=float, default=4.0, help="Input length in seconds, centred on each 2 s window")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=5, help="Early stopping cannot trigger before this epoch (avoids locking onto a lucky noisy early checkpoint)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seizure-weight", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--validation-recordings", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="*", help="Outer folds to run (default: all)")
    parser.add_argument("--no-final", action="store_true", help="Skip the final model trained on all recordings")
    parser.add_argument("--quick", action="store_true", help="Tiny run to check that everything works")
    # parse_known_args: also works when run inside a notebook cell.
    args, _ = parser.parse_known_args(argv)
    if args.aug_all:
        for name in AUG_FLAGS:
            setattr(args, f"aug_{name}", True)
    if args.augment:
        for name in ORIGINAL_FIVE:
            setattr(args, f"aug_{name}", True)
    if args.quick:
        args.epochs, args.min_epochs, args.patience, args.seeds, args.folds = 2, 1, 2, 1, [1]
    return args


def active_augmentations(args) -> list[str]:
    """Enabled --aug-* transform names, in the fixed order augment() applies them."""
    return [name for name in AUG_FLAGS if getattr(args, f"aug_{name}")]


def run_name(arch: str, active: list[str]) -> str:
    if not active:
        suffix = "plain"
    elif active == ORIGINAL_FIVE:
        suffix = "augmented"  # same name the original --augment combo has always used
    else:
        suffix = "+".join(AUG_CODES[name] for name in active)
    return f"cnn_{arch}_{suffix}"


def find_data(path: str) -> Path:
    folder = Path(path)
    if not (folder / "windows.csv").exists():
        raise FileNotFoundError(f"{folder / 'windows.csv'} not found: run scripts/export_cnn_data.py or pass --data")
    return folder


# ----------------------------------------------------------------------------- data


def load(data: Path, window_samples: int, quick: bool):
    """All input windows as one float32 array (windows, channels, samples), plus the window table."""
    windows = pd.read_csv(data / "windows.csv")
    annotations = pd.read_csv(data / "annotations.csv")
    if quick:
        keep = windows.loc[windows["fold"] == 1, "recording"].unique().tolist() + windows.loc[windows["fold"] == 2, "recording"].unique().tolist()[:3]
        windows = windows.loc[windows["recording"].isin(keep)].reset_index(drop=True)
    X = np.zeros((len(windows), len(CHANNELS), window_samples), dtype=np.float32)
    half = (window_samples - 2 * FS) // 2  # context added on each side of the 2 s window
    for recording, rows in windows.groupby("recording").groups.items():
        signal = pd.read_parquet(data / f"signals/{recording}.parquet", columns=CHANNELS).to_numpy(np.float32).T
        median = np.median(signal, axis=1, keepdims=True)
        mad = np.median(np.abs(signal - median), axis=1, keepdims=True) * 1.4826
        signal = np.clip((signal - median) / np.maximum(mad, 1e-6), -20, 20)
        padded = np.pad(signal, ((0, 0), (half, half + window_samples)))
        for row in rows:
            start = int(round(windows.at[row, "start_s"] * FS))  # in the padded signal, start - half + half
            X[row] = padded[:, start : start + window_samples]
    return X, windows, annotations


# ----------------------------------------------------------------------------- models


class ShallowConvNet(nn.Module):
    """Schirrmeister et al. 2017: temporal conv → spatial conv → square → mean pool → log."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2, filters: int = 40, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, filters, (1, 25)),
            nn.Conv2d(filters, filters, (n_chans, 1), bias=False),
            nn.BatchNorm2d(filters),
        )
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            size = self._embed(torch.zeros(1, n_chans, n_times)).shape[1]
        self.classifier = nn.Linear(size, n_classes)

    def _embed(self, x):
        x = self.features(x.unsqueeze(1))
        x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self.dropout(self._embed(x)))


class EEGNet(nn.Module):
    """Lawhern et al. 2018 (EEGNet-8,2), temporal kernel of half a second at 256 Hz."""

    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2, f1: int = 8, depth: int = 2, f2: int = 16, kernel: int = 128, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (n_chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8), groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            size = self.block(torch.zeros(1, 1, n_chans, n_times)).flatten(1).shape[1]
        self.classifier = nn.Linear(size, n_classes)

    def forward(self, x):
        return self.classifier(self.block(x.unsqueeze(1)).flatten(1))


def build(arch: str, n_times: int) -> nn.Module:
    return (ShallowConvNet if arch == "shallow" else EEGNet)(len(CHANNELS), n_times)


# ----------------------------------------------------------------------------- augmentation

# Each transform below is split out of what used to be one combined function, so every one can
# be enabled alone (--aug-<name>) or together, for a staged "test each separately, then combine
# the winners" comparison. --augment still applies exactly the original five, in the same order
# as before, with the same per-call RNG consumption -- so old runs stay reproducible.


def aug_amplitude(x: torch.Tensor) -> torch.Tensor:
    """Random 0.6-1.6x gain, one factor per sample."""
    return x * torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(0.6, 1.6)


def aug_noise(x: torch.Tensor) -> torch.Tensor:
    """Small additive Gaussian noise (the signal is in MAD units)."""
    return x + 0.03 * torch.randn_like(x)


def aug_channel_dropout(x: torch.Tensor) -> torch.Tensor:
    """30% of samples get 1 to 3 random channels zeroed."""
    batch, device = x.shape[0], x.device
    drop = torch.rand(batch, device=device) < 0.3
    if not drop.any():
        return x
    count = torch.randint(1, 4, (batch,), device=device)
    ranks = torch.rand(batch, x.shape[1], device=device).argsort(dim=1).argsort(dim=1)
    mask = (ranks < count[:, None]) & drop[:, None]
    return x.masked_fill(mask[:, :, None], 0.0)


def aug_mirror(x: torch.Tensor) -> torch.Tensor:
    """50% of samples get their channels permuted left-right (same pairs as the classical pipeline)."""
    batch, device = x.shape[0], x.device
    flip = torch.rand(batch, device=device) < 0.5
    if not flip.any():
        return x
    x = x.clone()
    x[flip] = x[flip][:, MIRROR_INDEX]
    return x


def aug_time_shift(x: torch.Tensor) -> torch.Tensor:
    """Roll the whole batch by up to +-0.125 s, same shift for every sample."""
    shift = int(torch.randint(-32, 33, (1,)).item())
    return torch.roll(x, shifts=shift, dims=2) if shift else x


def aug_time_mask(x: torch.Tensor, max_fraction: float = 0.25) -> torch.Tensor:
    """Zero one random contiguous time span (up to max_fraction of the window), same span for the batch."""
    samples = x.shape[2]
    span = max(1, int(round(samples * max_fraction)))
    length = int(torch.randint(1, span + 1, (1,)).item())
    start = int(torch.randint(0, samples - length + 1, (1,)).item())
    x = x.clone()
    x[:, :, start : start + length] = 0.0
    return x


def aug_band_stop(x: torch.Tensor, fs: int = FS, max_bandwidth_hz: float = 4.0) -> torch.Tensor:
    """Suppress one random frequency band (1-4 Hz wide), same band for the batch."""
    samples = x.shape[2]
    spectrum = torch.fft.rfft(x, dim=2)
    freqs = torch.fft.rfftfreq(samples, d=1 / fs, device=x.device)
    width = float(torch.empty(1).uniform_(1.0, max_bandwidth_hz).item())
    low = float(torch.empty(1).uniform_(0.5, max(0.5, freqs[-1].item() - width)).item())
    band = (freqs >= low) & (freqs < low + width)
    spectrum = spectrum.clone()
    spectrum[:, :, band] = 0
    return torch.fft.irfft(spectrum, n=samples, dim=2)


def aug_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Linearly blend each sample with another random sample in the batch; the caller mixes the
    loss the same way, using the two label sets and the returned weight."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    permutation = torch.randperm(x.shape[0], device=x.device)
    mixed = lam * x + (1 - lam) * x[permutation]
    return mixed, y, y[permutation], lam


TRANSFORMS = {
    "amplitude": aug_amplitude, "noise": aug_noise, "channel_dropout": aug_channel_dropout,
    "mirror": aug_mirror, "time_shift": aug_time_shift, "time_mask": aug_time_mask, "band_stop": aug_band_stop,
}


def augment(x: torch.Tensor, y: torch.Tensor, active: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """Applies every transform in ``active``, in TRANSFORMS' fixed order, to a training batch
    (both classes alike). Returns (x, y_a, y_b, lam): y_b is None and lam is 1.0 unless mixup is
    active, so the caller can always compute lam * loss(out, y_a) + (1 - lam) * loss(out, y_b)."""
    for name in AUG_FLAGS:
        if name not in active or name == "mixup":
            continue
        x = TRANSFORMS[name](x)
    if "mixup" in active:
        x, y_a, y_b, lam = aug_mixup(x, y)
        return x, y_a, y_b, lam
    return x, y, None, 1.0


# ----------------------------------------------------------------------------- detection (same rules as src/detection.py)


def intervals(start: np.ndarray, end: np.ndarray, positive: np.ndarray) -> list[list[float]]:
    order = np.argsort(start)
    start, end, positive = start[order], end[order], positive[order]
    events: list[list[float]] = []
    for s, e, p in zip(start, end, positive):
        if not p:
            continue
        if events and s - events[-1][1] < MIN_GAP_S:
            events[-1][1] = max(events[-1][1], e)
        else:
            events.append([s, e])
    return [event for event in events if event[1] - event[0] >= MIN_DURATION_S]


def event_counts(table: pd.DataFrame, annotations: pd.DataFrame, threshold) -> dict:
    """Found / missed seizures and false alarms; ``threshold`` is a number or a per-row array
    (already aligned with ``table``'s row order).

    Uses boolean masks over arrays pulled once from ``table``, never the DataFrame's own index
    (fixed bug: the previous version used ``groupby(...).groups``, whose keys are *index
    labels*, as positions into ``.to_numpy()`` — correct only when ``table``'s index happens to
    already be 0..len(table)-1, which is false for any ``.iloc[...]``-sliced subset such as the
    held-out validation predictions, e.g. ``IndexError: index 7943 is out of bounds for axis 0
    with size 4344``).
    """
    positive = table["proba"].to_numpy() >= threshold
    recordings = table["recording"].to_numpy()
    start_values = table["start_s"].to_numpy()
    end_values = table["end_s"].to_numpy()
    tp = fn = fp = detections = 0
    for recording in pd.unique(recordings):
        mask = recordings == recording
        detected = intervals(start_values[mask], end_values[mask], positive[mask])
        real = annotations.loc[annotations["recording"] == recording, ["start_s", "end_s"]].to_numpy()
        used = np.zeros(len(detected), dtype=bool)
        for s, e in real:
            hit = np.array([d[0] < e and d[1] > s for d in detected], dtype=bool)
            if len(detected):
                used |= hit
            if hit.any():
                tp += 1
            else:
                fn += 1
        fp += int((~used).sum())
        detections += len(detected)
    precision = (detections - fp) / detections if detections else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "found": tp, "missed": fn, "false_alarms": fp, "precision": precision, "recall": recall}


def best_threshold(table: pd.DataFrame, annotations: pd.DataFrame) -> float:
    scores = [event_counts(table, annotations, threshold)["f1"] for threshold in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])  # first maximum = lowest threshold


# ----------------------------------------------------------------------------- training


def predict(model: nn.Module, X: np.ndarray, rows: np.ndarray, device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for first in range(0, len(rows), batch):
            xb = torch.from_numpy(X[rows[first : first + batch]]).to(device, non_blocking=True)
            out.append(torch.softmax(model(xb).float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(args, X, y, train_rows, valid_rows, seed: int, device, fixed_epochs: int | None = None):
    """Train one model; early stopping on validation average precision unless ``fixed_epochs``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    active = active_augmentations(args)
    model = build(args.arch, X.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = fixed_epochs or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.seizure_weight], device=device))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best = {"score": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        total = 0.0
        for first in range(0, len(order), args.batch):
            rows = np.sort(order[first : first + args.batch])
            xb = torch.from_numpy(X[rows]).to(device, non_blocking=True)
            yb = torch.from_numpy(y[rows]).to(device, non_blocking=True)
            yb_b, lam = None, 1.0
            if active:
                xb, yb, yb_b, lam = augment(xb, yb, active)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(xb)
                loss = loss_fn(logits, yb) if yb_b is None else lam * loss_fn(logits, yb) + (1 - lam) * loss_fn(logits, yb_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item() * len(rows)
        scheduler.step()
        record = {"epoch": epoch, "loss": total / len(order)}
        if fixed_epochs is None:
            score = average_precision_score(y[valid_rows], predict(model, X, valid_rows, device))
            record["validation_ap"] = float(score)
            if score > best["score"]:
                best = {"score": float(score), "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            elif epoch >= args.min_epochs and epoch - best["epoch"] >= args.patience:
                # min_epochs keeps training past a lucky noisy early peak (the validation set is
                # only a handful of recordings) instead of locking onto it immediately.
                history.append(record)
                break
        history.append(record)
    if fixed_epochs is None:
        model.load_state_dict(best["state"])
    return model, best["epoch"] if fixed_epochs is None else epochs, history


def main(argv=None) -> None:
    args = arguments(argv)
    data = find_data(args.data)
    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    active = active_augmentations(args)
    name = run_name(args.arch, active)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    window_samples = int(round(args.window_s * FS))
    started = time.time()
    X, windows, annotations = load(data, window_samples, args.quick)
    y = windows["true_class"].to_numpy(np.int64)
    print(f"{name} on {device}: {len(windows):,} windows ({y.sum():,} seizure), input {X.shape[1:]} , {time.time() - started:.0f} s to load", flush=True)

    folds = args.folds or sorted(windows["fold"].unique())
    proba = np.full(len(windows), np.nan)
    validation_tables, history, thresholds, best_epochs = [], {}, {}, []
    for fold in folds:
        test_rows = np.flatnonzero(windows["fold"].to_numpy() == fold)
        train_side = windows.loc[windows["fold"] != fold]
        with_seizures = sorted(train_side.loc[train_side["true_class"] == 1, "recording"].unique())
        chosen = np.random.default_rng(fold).choice(with_seizures, size=min(args.validation_recordings, len(with_seizures) - 1), replace=False)
        valid_rows = np.flatnonzero(windows["recording"].isin(chosen).to_numpy())
        train_rows = np.flatnonzero(((windows["fold"] != fold) & ~windows["recording"].isin(chosen)).to_numpy())
        fold_test, fold_valid = [], []
        for seed in range(args.seeds):
            model, epoch, log = train(args, X, y, train_rows, valid_rows, seed=100 * fold + seed, device=device)
            best_epochs.append(epoch)
            history[f"fold{fold}_seed{seed}"] = log
            fold_test.append(predict(model, X, test_rows, device))
            fold_valid.append(predict(model, X, valid_rows, device))
            torch.save(model.state_dict(), out / "models" / f"{name}_fold{fold}_seed{seed}.pt")
            print(f"  fold {fold} seed {seed}: best epoch {epoch}, validation AP {max(r.get('validation_ap', 0) for r in log):.3f}, "
                  f"{time.time() - started:.0f} s", flush=True)
        proba[test_rows] = np.mean(fold_test, axis=0)
        valid = windows.iloc[valid_rows].assign(proba=np.mean(fold_valid, axis=0))
        thresholds[int(fold)] = best_threshold(valid, annotations)
        validation_tables.append(valid)

    done = ~np.isnan(proba)
    predictions = windows.loc[done].assign(proba=proba[done].astype(np.float32))
    predictions.to_parquet(out / "predictions" / f"{name}.parquet", index=False)
    pd.concat(validation_tables).assign(proba=lambda t: t["proba"].astype(np.float32)).to_parquet(out / "predictions" / f"{name}_validation.parquet", index=False)

    # Scores on the out-of-fold predictions (the project re-scores them with scripts/evaluate_cnn.py).
    per_row = predictions["fold"].map(thresholds).to_numpy()
    nested = event_counts(predictions, annotations, per_row)
    best = best_threshold(predictions, annotations)
    optimistic = event_counts(predictions, annotations, best)
    print(f"\n{name}: fold thresholds (chosen on validation recordings) {thresholds}")
    print(f"  nested     event F1 {nested['f1']:.3f}  found {nested['found']}/{nested['found'] + nested['missed']}  false alarms {nested['false_alarms']}")
    print(f"  optimistic event F1 {optimistic['f1']:.3f} at {best:.2f}  found {optimistic['found']}  false alarms {optimistic['false_alarms']}")
    print(f"  window average precision {average_precision_score(predictions['true_class'], predictions['proba']):.3f}")

    config = {
        "name": name, "arch": args.arch,
        "augment": bool(active),  # legacy flag, kept so old readers (and the 3 already-committed runs) still work
        "augmentation": {flag: flag in active for flag in AUG_FLAGS},
        "channels": CHANNELS, "fs": FS, "window_s": args.window_s,
        "window_samples": window_samples, "context": "centred on each 2 s window (1 s step)",
        "normalization": "per recording and channel: (x - median) / (1.4826 * MAD), clipped at ±20",
        "filter": "Butterworth band-pass 0.5-40 Hz, order 4, zero-phase, before normalization",
        "loss": f"cross-entropy, class weights [1, {args.seizure_weight}]", "optimizer": f"AdamW lr {args.lr} wd {args.weight_decay}, cosine",
        "batch": args.batch, "max_epochs": args.epochs, "patience": args.patience, "seeds": args.seeds,
        "fold_thresholds": thresholds, "best_epochs": best_epochs,
        "nested": nested, "optimistic": {**optimistic, "threshold": best}, "seconds": time.time() - started,
    }

    # Final model on all recordings, for use on new data: the median best epoch, no early stopping.
    if not args.no_final and not args.quick:
        epochs = max(1, int(np.median(best_epochs)))
        everything = np.arange(len(windows))
        model, _, log = train(args, X, y, everything, everything[:0], seed=0, device=device, fixed_epochs=epochs)
        history["final"] = log
        torch.save(model.state_dict(), out / "models" / f"{name}_final.pt")
        config["final"] = {"epochs": epochs, "threshold": float(np.median(list(thresholds.values())))}
        try:
            model = model.float().cpu().eval()
            torch.onnx.export(model, torch.zeros(1, len(CHANNELS), window_samples), out / "models" / f"{name}_final.onnx",
                              input_names=["eeg"], output_names=["logits"], dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
        except Exception as error:  # ONNX is optional: the .pt weights are enough.
            print(f"ONNX export skipped: {error}")
    (out / f"{name}_config.json").write_text(json.dumps(config, indent=2, default=float) + "\n")
    (out / f"{name}_history.json").write_text(json.dumps(history, indent=1) + "\n")
    print(f"Saved in {out} ({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
