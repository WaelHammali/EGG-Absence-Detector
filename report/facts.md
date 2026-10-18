# Fact sheet for the project report

Every fact below is tagged with its exact source. Nothing here comes from memory or from
`HANDOFF.md` alone — `HANDOFF.md` and `CLAUDE.md` were re-read fresh on 2026-10-17 before
writing this, since earlier context in this session was stale (the second CNN run had finished
and the repo had been cleaned up since). Where a number has a caveat (optimistic threshold,
partial run, superseded doc), the caveat is stated here so the report inherits it.

---

## 0. Current project state (re-verified 2026-10-17)

- Working tree clean, `git status` shows no uncommitted changes. 29 commits on `main`,
  `057d173` (cleanup + keep CNN weights) is HEAD.
- `HANDOFF.md`, `CLAUDE.md` read in full just now (quoted below where relevant).
- **Two headline numbers exist, and they answer different questions:**
  - Best **single model**: SVM unbalanced, event F1 **0.894** (86/93 found, 16 false alarms,
    threshold 0.82) — `docs/MODEL_COMPARISON.md`, `HANDOFF.md:10-12`. This is also
    `models/registry.json`'s `"default_model": "svm_unbalanced"` (verified by reading the
    JSON key directly). Caveat, stated in `HANDOFF.md:11-12` and in `MODEL_COMPARISON.md`
    itself: the threshold is chosen on the same out-of-fold predictions it is scored on, so
    this number is slightly optimistic.
  - Best **blend**: SVM unbalanced + ShallowConvNet augmented, event F1 **0.901** (90/93
    found, 19 false alarms), threshold chosen on the *other* folds only (held-out-fold
    protocol, not optimistic) — `docs/ENSEMBLE_RESULTS.md`, `HANDOFF.md:13-15`. This beats
    the plain SVM by +0.007, not a tie — `docs/ENSEMBLE_RESULTS.md`'s own prose says "ties
    the baseline," which is leftover wording from an earlier version of that file and is
    wrong; the report must not repeat it. `HANDOFF.md:14-15` itself is honest about the
    gain being small (4 more seizures, 3 more false alarms) and "not tested for noise."
  - **Decision for the report**: present the SVM-unbalanced single model as "the baseline" /
    headline since it is the registry default and the simplest result to defend, and present
    the SVM+CNN blend as "the best result found, with a caveat" — matching how `HANDOFF.md`
    itself frames the two numbers. This mirrors the project's own framing, not an invented one.
- **CNNs are DONE, not pending.** `docs/DEEP_LEARNING_RESULTS.md` and `outputs/cnn_evaluation.csv`
  exist on disk, dated 2026-10-17. Final nested-threshold numbers (`HANDOFF.md:16-18`):
  ShallowConvNet augmented 0.815, EEGNet augmented 0.776 (0.811 after smoothing+hysteresis),
  ShallowConvNet plain 0.774 (unaffected by smoothing in that run). None beats the SVM alone;
  `HANDOFF.md` calls this expected, given 93 seizures across 21 patients.
- The old Kaggle-notebook tooling (`kaggle/` folder, `docs/KAGGLE.md`, `docs/RESEARCH_PROMPT.md`,
  `docs/NESTED_EVALUATION.md`) was removed in the cleanup commit `057d173`. CNN training is now
  `scripts/train_cnn.py` (self-contained, meant to be copied to a GPU machine), export is
  `scripts/export_cnn_data.py` → `data/cnn/` (git-ignored, patient data, never committed).
- **Open / incomplete, confirmed on disk, must be reported as such:**
  - `scripts/evaluate_nested.py` (nested-threshold + SzCORE re-scoring of the 12 classical
    models, for a fair comparison with the CNNs' nested scores) has only been run for 3 of the
    12 models. Verified directly: `outputs/nested/` holds exactly three cache files —
    `knn_balanced.json`, `rf_unbalanced.json`, `svm_augmented.json` — not even the default
    `svm_unbalanced`. No `outputs/nested_evaluation.csv`, no `docs/NESTED_EVALUATION.md` exist
    on disk. `HANDOFF.md:24-27` says the full run takes about an hour and is memory-heavy
    ("killed once for lack of RAM").
  - The neurologist has not yet reviewed the false alarms (`HANDOFF.md:28-29`); some, especially
    in `210504B_C` and `210427B_C`, are suspected to be real unannotated seizures.
  - The CNN loading example in `models/cnn/README.md` was written but never run — no PyTorch
    installed in this environment (`HANDOFF.md:30`, confirmed: `requirements.txt` has no torch
    entry).
  - `data/excel/` (2 GB) is unused by any later step and can be deleted/rebuilt
    (`HANDOFF.md:31-32`).
- **Bug-history citation correction**: `HANDOFF.md:37` and the project's own docs cite commit
  `b026a70` for the `event_counts()` IndexError fix. That hash is **not reachable from the
  current branch** (`git log` does not show it — history was rewritten at some point); it only
  still exists as a dangling object. The actual fix on the current branch is commit `37eb36c`
  ("fix CNN threshold crash"), verified by reading its diff directly. **The report should cite
  `37eb36c`, or describe the bug without a hash**, not copy `b026a70` forward.

---

## 1. Introduction / project identity

- University data-mining project (Tunisia), École Nationale d'Ingénieurs de Tunis —
  `CLAUDE.md:3-8`, confirmed by the slide deck built earlier this session (author: Wael Hammali).
- Goal: detect absence seizures automatically in EEG, event by event, in a way a clinician could
  trust — `CLAUDE.md:5-8`.
- Pipeline, in order (`CLAUDE.md:5-8`): filter signal → window → extract features → train
  classical models (DT/RF/KNN/SVM, 3 training modes) → evaluate detection event by event →
  Dash app to explore results. CNNs trained on a separate GPU machine are compared against the
  classical models, and blends of models are scored too.
- Absence seizures, clinically: generalized (both hemispheres at once), roughly 3 Hz
  spike-and-wave discharges, usually 3–20 seconds long, mostly in children — stated in this
  session's own earlier slide-deck work, traceable to the same domain description used
  throughout `docs/MODEL_COMPARISON.md` and the detection logic (`src/detection.py`).

---

## 2. The raw data

- 22 original recordings, each a pair of raw MATLAB files (`*_0000d.mat` signal + `*_0000h.mat`
  header) plus one shared `Annotations.xlsx` — `docs/DATA_NOTES.md` §1.
- Sampling rate 256 Hz for all 22 recordings, not declared in any header field — inferred from
  256-sample blocks in the decoded timetable — `docs/DATA_NOTES.md` §1-2.
- Full channel montage: 19 EEG channels (10-20 layout) + SLI (pulse marker) + ECG +, in some
  recordings, 2 EMG channels; channel order not consistent file to file, so channels are always
  matched by name, never by column position — `docs/DATA_NOTES.md` §3.
- No amplitude unit documented anywhere (`varUnits` empty on every signal) — reported as
  "unit not confirmed," never labeled µV without proof — `docs/DATA_NOTES.md` §2.
- Total raw data: 8,506,368 samples, 33,228 s = 9 h 13 min 48 s across the 22 recordings; no NaN,
  no infinite values, no fully empty channel — `docs/DATA_NOTES.md` §3.
- `200625A_F` excluded entirely: only 8 of the 19 EEG channels (the 8 common to all 22
  recordings) **and** zero seizure annotations. Missing annotations do not prove "seizure-free" —
  could equally mean the label was never entered — so the recording was dropped rather than
  mislabeled — `docs/DATA_NOTES.md` §3-4, §6, confirmed as the final decision in
  `docs/ANNOTATION_DECISIONS.md`'s "Exclusions" section. Result: 21 recordings remain, each with
  the full 19-channel EEG montage.
- Clock-alignment problem: signal and annotation workbook each start their own clock at 0; raw
  files never recorded an absolute wall-clock start time; seizure times in the workbook are
  Excel time-of-day fractions (e.g. "10:35:06"), not seconds already relative to recording
  start. Official rule: `elapsed = clock_annotation − DebutTrace`, with midnight rollover when
  needed — `docs/ANNOTATION_DECISIONS.md` (header line), `docs/DATA_NOTES.md` §4. Two shortcuts
  were explicitly rejected: treating the Excel fraction as already-relative seconds, and
  anchoring on the first seizure — `docs/DATA_NOTES.md` §4.
- Ambiguous/duplicate annotation rows resolved by scoring each candidate interval against the
  real EEG signal (seizure-likelihood score inside vs. outside the interval) and keeping the
  cleaner separation: `230406B_A` — row 11 (score 0.189095, ratio 2.184) kept over row 19 (score
  0.173201, ratio 2.086); `230515B_G` — row 20 (score 0.354451, ratio 4.076) kept over row 12
  (score 0.336547, ratio 3.996) — `docs/ANNOTATION_DECISIONS.md` "Applied decisions" and
  "Signal scores" tables.
- `210504B_C` CE1: proposed end-time correction to 12:51:06 scored 0.305716 (ratio 2.610),
  retained — `docs/ANNOTATION_DECISIONS.md`.
- `210427B_C`: official `DebutTrace` places the interval at 1031.000–1049.000 s, score 0.306318
  (ratio 2.695), included — `docs/ANNOTATION_DECISIONS.md`. (Note: an older, superseded doc,
  `docs/archive_8ch/DATASET_PARTS_I_VI.md`, had once excluded this recording for "ambiguous
  alignment" under the earlier 8-channel/provisional pipeline — that exclusion does **not**
  apply to the current official dataset and must not be cited as current.)
- `210204B_C` and `210208B_G` are missing from the final, corrected annotation workbook; both
  fall back to an earlier provisional set of intervals rather than being invented or dropped —
  `docs/ANNOTATION_DECISIONS.md` "Applied decisions".
- Final result: **21 recordings, 19 EEG channels each, 93 confirmed seizures** —
  `CLAUDE.md:5-6`, consistent with `docs/DATA_LAYOUT.md`'s dataset table (1,244 seizure windows
  from 93 seizure events) and with `models/registry.json`.

---

## 3. Preprocessing

- Non-EEG channels (SLI, ECG, EMG) are never used as model input features — only the 19 EEG
  channels feed the model (`src/features.py`, `CLAUDE.md`). SLI/ECG/EMG and the technician's HPN
  notes are kept and used only afterward, as an audit tool, to check whether a model's false
  alarms coincide with hyperpnea (a seizure-provocation technique), muscle-artifact bursts, or
  SLI pulses — never to influence the model's decision — `docs/MODEL_COMPARISON.md`
  "Fausses alarmes" section.
- Demographic/metadata columns (sex, age, seizure count, seizure type, the undocumented
  `Intercrit*` markers) are kept as separate audit metadata, never turned into training features
  — `docs/DATA_NOTES.md` §4, §6.
- Filter: 4th-order Butterworth band-pass, 0.5–40 Hz, applied zero-phase
  (`scipy.signal.sosfiltfilt`, forward-backward, not a single causal pass) —
  `src/preprocessing.py:9-26` (`BANDPASS_HZ = (0.5, 40.0)`, `BANDPASS_ORDER = 4`). Computed in
  float64, returned as float32 — `src/preprocessing.py:26`.
  - 0.5 Hz low cutoff: removes slow baseline drift (electrode/skin potential drift, sweat, slow
    movement) without touching the frequency range seizure activity and normal brain rhythms
    live in.
  - 40 Hz high cutoff: removes high-frequency muscle/movement noise and sits below 50 Hz mains
    hum (local mains frequency is 50 Hz, so a 40 Hz cutoff attenuates it).
  - Zero-phase specifically: a normal (causal) filter introduces a time lag; forward-backward
    filtering cancels it, so filtered seizure boundaries land at the same timestamps as the
    original annotated boundaries — docstring, `src/preprocessing.py:21-23` ("seizure
    boundaries are not shifted").
- Windowing: 2-second windows, 50% overlap — `src/segmentation.py:10-11`
  (`WINDOW_SECONDS = 2.0`, `OVERLAP = 0.5`). A window is labeled seizure (class 1) if at least
  50% of its samples are seizure (`positive_fraction=0.5` default, `src/segmentation.py:29`).
- Result: 31,419 windows total, 1,244 seizure windows (3.96%), 30,175 non-seizure (96.04%) —
  `docs/DATA_LAYOUT.md` dataset table, confirmed again in `docs/MODEL_COMPARISON.md`'s training
  distribution table.
- Feature extraction (`src/features.py`, read in full):
  - 8 time-domain feature types: mean, std, variance, min, max, amplitude (peak-to-peak,
    `np.ptp`), RMS, energy (sum of squares) — `src/features.py:9`.
  - 4 frequency bands, each giving an absolute-power and a relative-power feature type (8
    frequency feature types total): delta 0.5–4.0 Hz, theta 4.0–8.0 Hz, alpha 8.0–13.0 Hz, beta
    13.0–30.0 Hz (beta's upper edge inclusive, others exclusive) — `src/features.py:10-15,40-43`.
    Power spectral density via a Hann-windowed real FFT, normalized by `fs * sum(hann²)`,
    non-DC/Nyquist bins doubled for a one-sided spectrum — `src/features.py:22-51`. Relative
    power = absolute band power ÷ total power in 0.5–30 Hz.
  - 16 feature types total (8 time + 8 frequency). Each type gets one column per EEG channel
    (19) plus one cross-channel average column → 20 columns per type.
    **16 × 20 = 320 features per window** — matches `CLAUDE.md`/`docs/DATA_LAYOUT.md`'s "320
    features" exactly, arithmetic confirmed against `extract_features()`,
    `src/features.py:67-72`.
- Why features instead of raw signal for the classical pipeline: classical models need a
  fixed-length input vector, and with only 93 seizures total, hand-crafted features generalize
  far better than asking a model to learn its own representation from raw signal — same
  reasoning the project uses later to explain why the CNN (which does use raw signal) doesn't
  beat the SVM (§7 below).

---

## 4. Classical machine learning

- 4 algorithms × 3 training modes = **12 models** — `CLAUDE.md:6-7`, `scripts/train_registry.py`.
- Pipelines (`scripts/train_models.py`, read in full; `SEED = 42`):
  - **Decision Tree**: `DecisionTreeClassifier(random_state=42)`, no scaler, all other
    hyperparameters at scikit-learn 1.9.1 defaults (`criterion="gini"`, `max_depth=None`,
    `min_samples_split=2`, `min_samples_leaf=1`).
  - **Random Forest**: `RandomForestClassifier(random_state=42, n_jobs=-1)`, no scaler; defaults
    `n_estimators=100`, `criterion="gini"`, `max_depth=None`, `max_features="sqrt"`.
  - **KNN**: `StandardScaler()` → `KNeighborsClassifier()`, all defaults (`n_neighbors=5`,
    `weights="uniform"`, Minkowski distance, p=2).
  - **SVM**: `StandardScaler()` → `SVC(random_state=42)`, defaults `C=1.0`, `kernel="rbf"`,
    `gamma="scale"`; `train_registry.py` additionally sets `model__probability=True` only for
    SVM, enabling Platt scaling so its output can be swept across thresholds (this does not
    change the decision boundary, only adds a calibrated probability on top of it).
  - `METADATA = {"recording", "subject", "window", "start_s", "end_s", "class"}` — the columns
    excluded when building the feature matrix `X` (everything else is a feature).
  - `undersample()` keeps every class-1 (seizure) index plus an equal-size random draw of
    class-0 indices, via `np.random.RandomState(seed)`, restored to original order.
- Three training modes (`scripts/train_registry.py:44`, `docs/MODEL_COMPARISON.md`):
  - **Unbalanced**: each training fold used exactly as-is, real class proportions.
  - **Balanced**: random undersampling of the training fold only (every class-1 window + an
    equal number of random class-0 windows), repeated with 5 seeds (0–4); seed 0 is the one
    used for saved models/predictions.
  - **Augmented**: the full training fold plus synthetic extra seizure windows built only from
    the training recordings' real signal (see §5 below). Test data always keeps real proportions
    and only real, original windows, in every mode.
- Evaluation protocol (`src/detection.py`, read in full):
  - 5-fold `GroupKFold` by recording (same folds project-wide) — a model is never scored on a
    recording it was trained on.
  - A window counts as positive if its probability ≥ threshold. Consecutive positive windows
    merge into a run; runs separated by less than `MIN_GAP_S = 2.0 s` are merged into one
    interval; merged intervals shorter than `MIN_DURATION_S = 2.0 s` are dropped
    (`windows_to_intervals`).
  - Event matching: a real seizure is a true positive if any detected interval overlaps it
    (`start < real_end AND end > real_start`), else a false negative. A detected interval that
    overlaps no real seizure is a false positive (false alarm). Onset/offset error = detected −
    real seconds, averaged only over true positives (`match_events`).
  - `event_metrics`: precision = true detections ÷ (true detections + false positives),
    recall = TP ÷ (TP + FN), F1 = harmonic mean of the two.
  - For each model, the threshold sweep picks the threshold that maximizes event F1 on its own
    out-of-fold predictions (`sweep_thresholds`); this is the "best threshold" reported in
    `docs/MODEL_COMPARISON.md`, and it is explicitly flagged there as slightly optimistic because
    the same predictions are used to choose and to score the threshold. Scores at a fixed
    threshold of 0.50 are also reported, for comparison, without this optimism.
  - **This plain path (no smoothing/hysteresis) is what `train_registry.py` uses for all 12
    models and what `docs/MODEL_COMPARISON.md`'s tables report.** Smoothing+hysteresis
    post-processing (§7 below) is a separate, optional path applied later, only to the CNNs, in
    `scripts/evaluate_cnn.py`.
- **Full 12-model results table, best threshold** (`docs/MODEL_COMPARISON.md`, verbatim numbers —
  use directly in the report's comparison table):

  | Algorithm | Mode | Threshold | Event F1 | Recall | Precision | Found/missed | False alarms | FA/hour |
  |---|---|---:|---:|---:|---:|---:|---:|---:|
  | Decision Tree | unbalanced | 0.05 | 0.290 | 0.978 | 0.170 | 91/2 | 561 | 64.2 |
  | Decision Tree | balanced | 0.05 | 0.148 | 0.989 | 0.080 | 92/1 | 1229 | 140.7 |
  | Decision Tree | augmented | 0.05 | 0.171 | 0.989 | 0.094 | 92/1 | 1007 | 115.3 |
  | Random Forest | unbalanced | 0.67 | 0.867 | 0.935 | 0.807 | 87/6 | 26 | 3.0 |
  | Random Forest | balanced | 0.99 | 0.785 | 0.860 | 0.721 | 80/13 | 39 | 4.5 |
  | Random Forest | augmented | 0.93 | 0.854 | 0.903 | 0.810 | 84/9 | 26 | 3.0 |
  | KNN | unbalanced | 0.61 | 0.828 | 0.925 | 0.750 | 86/7 | 37 | 4.2 |
  | KNN | balanced | 0.81 | 0.519 | 0.968 | 0.355 | 90/3 | 189 | 21.6 |
  | KNN | augmented | 0.81 | 0.692 | 0.968 | 0.539 | 90/3 | 95 | 10.9 |
  | **SVM** | **unbalanced** | **0.82** | **0.894** | 0.925 | 0.866 | **86/93** | **16** | **1.8** |
  | SVM | balanced | 0.99 | 0.821 | 0.860 | 0.785 | 80/13 | 29 | 3.3 |
  | SVM | augmented | 0.98 | 0.862 | 0.968 | 0.778 | 90/3 | 32 | 3.7 |

  The Decision Tree only ever outputs probabilities of 0 or 1, so its threshold sweep has no
  effect — `docs/MODEL_COMPARISON.md`'s note under the 0.50-threshold table.
- Balanced-mode variability across the 5 seeds (mean ± std, `docs/MODEL_COMPARISON.md`): e.g.
  SVM balanced event F1 = 0.797 ± 0.017, Random Forest balanced = 0.808 ± 0.033 — useful to show
  balanced training is not just worse on average, it is also less stable.
- False-alarm interpretation against HPN/SLI/EMG/technician notes (`docs/MODEL_COMPARISON.md`,
  for the default model SVM unbalanced): 7 of 16 false alarms (44%) occur during hyperpnea (a
  provocation technique covering 27.8% of recorded time); 1/16 (6%) during an SLI marker
  interval; 0/1 near an EMG burst; 2/16 (12%) within 10 s of a technician's seizure-related note.
  Real seizures show the same hyperpnea tendency (37/93, 40%), because hyperpnea is used
  precisely to provoke them — the doc is explicit that this cannot by itself prove a false alarm
  is a real missed seizure, only that it's plausible.
- Conclusion stated in the doc itself: SVM unbalanced is the best model (F1 0.894), Random Forest
  unbalanced second (0.867); the gap between the top two corresponds to only a few events and is
  not large enough to call it certain — `docs/MODEL_COMPARISON.md` "Conclusion".

---

## 5. Data augmentation

- Built by `scripts/build_augmented.py` (read in full). Only class-1 (seizure) windows are
  created, from the real filtered signal of the training recordings — never from random
  recordings held out as test.
  - **Shift**: extra 2 s windows starting 0.25 s, 0.5 s, 0.75 s after each regular window start
    (the regular grid has a 1 s hop), kept under the same ≥50%-seizure-coverage rule as the
    official dataset.
  - **Mirror**: every seizure window (regular or shifted) with left and right electrodes
    swapped: Fp1↔Fp2, F7↔F8, F3↔F4, T3↔T4, C3↔C4, T5↔T6, P3↔P4, O1↔O2; Fz, Cz, Pz (midline)
    stay in place — `scripts/build_augmented.py:33-39`.
- Counts (`docs/MODEL_COMPARISON.md`): across the 21 recordings, 1,244 original seizure windows
  + 8,230 added (3,493 shifted, 1,244 mirrored, 3,493 shifted-and-mirrored). Kept in a separate
  file, `data/processed/datasets/augmented_shift_mirror/`, never mixed into the official/test
  dataset — training folds only.
- **Why shift and mirror specifically, and not other augmentation methods.** Both transform real
  recorded signal rather than synthesizing new signal, so nothing physiologically implausible is
  fabricated:
  - Mirror is anatomically justified specifically for absence seizures: they are generalized,
    bilaterally synchronous spike-and-wave discharges, so a left-right-swapped seizure window is
    still a valid seizure pattern — not an invented one.
  - Shift is consistent with a tolerance already built into the official labeling rule (a window
    only needs ≥50% seizure coverage to count as class 1), so nudging the start time by a
    fraction of a second doesn't violate anything the labeling scheme already accepts.
  - Heavier-machinery alternatives (GANs/VAEs to synthesize new EEG, or mixup/cutmix to blend
    unrelated windows) were not attempted in this version: a generative model would need far
    more than 93 seizures to avoid memorizing or producing unrealistic signal, and mixing two
    unrelated seizure segments risks creating a composite that doesn't correspond to any real
    physiological event. (This reasoning is this session's own synthesis of the method's logic,
    grounded in `build_augmented.py`'s approach and the project's stated preference for
    real-signal-only transforms; it is not itself written down in a project doc — the report
    should present it as the rationale inferred from what the code actually does, not quote it
    as a line from a source file.)
- **Result, honestly**: `docs/MODEL_COMPARISON.md`'s own 12-model table shows augmented training
  raised recall for every algorithm (e.g. SVM augmented: 90/93 found vs. unbalanced's 86/93) but
  also raised false alarms sharply (SVM augmented: 32 false alarms vs. unbalanced's 16) and
  lowered or left flat the resulting event F1 (SVM augmented 0.862 vs. unbalanced 0.894; Random
  Forest augmented 0.854 vs. unbalanced 0.867). Augmentation never beat the matching algorithm's
  unbalanced F1 in this table. This matches `HANDOFF.md`'s framing of a documented finding, not
  a failure.

---

## 6. Ensemble

- `scripts/ensemble_predictions.py` (read in full): blends out-of-fold probabilities of ≥2
  already-trained models (classical and/or CNN parquets), with no retraining. CLI:
  `python scripts/ensemble_predictions.py --models <key> <key> [--extra <cnn.parquet>]
  [--weights <w1> <w2> ...]`.
- Two threshold conventions, both reported (`docs/ENSEMBLE_RESULTS.md`):
  - **held-out-fold** (the honest number): for each outer fold, the threshold is swept on the
    other 4 folds' predictions only, never on the fold being scored.
  - **optimistic**: one threshold swept on all out-of-fold predictions at once (same convention
    as the registered models' own "best threshold" numbers), kept only for reference/comparison.
- Results table (`docs/ENSEMBLE_RESULTS.md`, verbatim):

  | Ensemble | Event F1 (held-out-fold) | Found (of 93) | False alarms | FA/hour | SzCORE event F1 |
  |---|---:|---:|---:|---:|---:|
  | SVM·Unbalanced (single model, baseline) | 0.894 | 86 | 16 | 1.8 | — |
  | **SVM·Unbalanced + ShallowConvNet·Augmented** | **0.901** | **90** | **19** | — | **0.900** |
  | SVM·Unbalanced + Random Forest·Unbalanced | 0.894 | 85 | 15 | 1.7 | 0.894 |
  | SVM·Unbalanced + RF·Unbalanced + KNN·Unbalanced | 0.874 | 83 | 18 | 2.1 | 0.881 |
  | SVM·Unbalanced + KNN·Unbalanced | 0.871 | 81 | 15 | 1.7 | 0.872 |
  | 2×SVM·Unbalanced + Random Forest·Unbalanced | 0.849 | 83 | 23 | 2.6 | 0.870 |

  (The SVM+ShallowConvNet row's exact FA/hour was not independently re-derived here; cite
  `docs/ENSEMBLE_RESULTS.md` directly for that figure if the report's table needs it filled in,
  or compute `19 false alarms ÷ total recorded hours` from `models/registry.json`'s
  `recorded_hours` field.)
- **Correction to carry into the report**: `docs/ENSEMBLE_RESULTS.md`'s prose literally says the
  SVM+ShallowConvNet blend "ties the baseline (0.901 vs 0.894)" — this is wrong on its face
  (0.901 > 0.894 is a real gain of +0.007, not a tie) and is leftover wording from an earlier
  draft of that file. The SVM+RF blend (0.894, 85 found, 15 false alarms) is the one that
  actually ties the baseline. The report must state the SVM+ShallowConvNet result as a genuine,
  if modest, improvement — matching `HANDOFF.md:13-15`'s own framing ("small... was not tested
  for noise") — and should note this as an open question (is the gain robust, or within noise?)
  rather than present it as a settled win.

---

## 7. Deep learning

- Training script: `scripts/train_cnn.py` (formerly `kaggle/train_cnn.py`; the `kaggle/`
  directory and its Kaggle-notebook tooling were removed in cleanup commit `057d173` — the
  script is self-contained specifically so it can be copied to any GPU machine, not only
  Kaggle). Model classes live in this same file — `CLAUDE.md:16-18`.
- Export: `scripts/export_cnn_data.py` → `data/cnn/` (git-ignored, real patient data, never
  committed) — `CLAUDE.md:27`.
- Two architectures: **ShallowConvNet** (temporal conv + spatial conv + pooling — a standard,
  compact EEG-decoding baseline) and **EEGNet** (depthwise + separable convolutions, a
  parameter-efficient architecture designed specifically for small EEG datasets). Both are
  intentionally small (few thousand parameters) given only 93 seizures total, to avoid
  overfitting a larger network would risk.
- Input: 4 s of the 19 filtered channels (1,024 samples at 256 Hz), centered on each 2 s window
  of the official dataset, so labels line up exactly with the classical models'. Normalization
  per recording and channel: `(x − median) / (1.4826 · MAD)`, clipped at ±20.
- Same 5 outer `GroupKFold` folds as the classical models. Within each outer fold, training
  recordings are further split so some are held out for early stopping and threshold selection,
  never the test fold. Real class proportions kept; weighted cross-entropy (seizure class
  weighted up) to counter the imbalance; AdamW optimizer with a cosine schedule; early stopping
  on validation average precision. Optional online augmentation (amplitude scaling, small noise,
  channel dropout, left-right mirror, small time shift) applied to **both** classes during
  training — a deliberately different augmentation strategy from the classical pipeline's
  seizure-only shift+mirror (see §5), because it is applied per-batch during CNN training rather
  than pre-built into a static extra dataset.
- Several random seeds per fold; probabilities averaged across seeds.
- **The IndexError bug** (`HANDOFF.md:36-37`, verified directly via `git show 37eb36c`):
  `event_counts()` used to look up pandas `groupby(...).groups` dictionary **keys** (index
  *labels*) and use them directly as array **positions** — correct only when the table's row
  index happened to be untouched `0..len-1`. On a `.iloc[]`-sliced validation subset (a real
  case, once a chosen validation recording's original row number exceeded the validation
  subset's own size), this produced `IndexError: index 7943 is out of bounds for axis 0 with
  size 4344`. The crash depended on which recordings were randomly chosen for validation, so
  earlier runs had worked only by luck. Fix: compute `recordings`/`start_values`/`end_values` as
  plain arrays once, then select by boolean mask instead of index-label lookup — correct
  regardless of the table's index. **Cite this as commit `37eb36c`, not `b026a70`** (see §0 for
  why).
- Scoring: `scripts/evaluate_cnn.py` (read in full; current, confirmed run and complete —
  `docs/DEEP_LEARNING_RESULTS.md` and `outputs/cnn_evaluation.csv` both exist, dated 2026-10-17).
  For each CNN run it computes: nested-threshold event F1 (using each fold's own threshold,
  chosen on held-out training recordings, never the test fold), an optimistic single-threshold
  event F1, a smoothed+hysteresis nested F1, a SzCORE event F1, and window-level average
  precision — then reports the best of plain vs. smoothed per model.
  - Smoothing+hysteresis (`src/detection.py`, also read in full): a separate, optional
    post-processing path with **no retraining** — centered rolling-median smoothing of the
    per-window probabilities (default window 3), then a two-threshold hysteresis rule (a higher
    threshold starts a detected event, a lower one lets it continue). Applied automatically by
    `evaluate_cnn.py` to the CNN predictions; **not** applied to the 12 classical models'
    headline numbers in `docs/MODEL_COMPARISON.md` — that table uses the plain path only. This
    distinction matters for any apples-to-apples comparison the report draws.
- Final results (`docs/DEEP_LEARNING_RESULTS.md`, `HANDOFF.md:16-18`): SVM baseline 0.894 vs.
  ShallowConvNet augmented **0.815** (nested threshold, best of plain/smoothed), EEGNet augmented
  **0.811** (smoothed; 0.776 plain — smoothing helped EEGNet specifically), ShallowConvNet plain
  **0.774**. None beats the SVM alone. The doc's own verdict: expected, given 93 seizures across
  21 patients is too little data for a CNN trained on raw signal to out-learn hand-crafted
  features — stated as a real, documented finding, not a failure.
- 3 real completed CNN runs exist under `models/cnn/`: `cnn_shallow_plain`,
  `cnn_shallow_augmented`, `cnn_eegnet_augmented` — each with its config JSON, training history
  JSON, final `.pt` and `.onnx` weights, per-fold/per-seed weights (`models/cnn/folds/`), and a
  validation-set parquet. No PyTorch is installed locally (confirmed: absent from
  `requirements.txt`), so these were trained on separate GPU hardware and only the
  already-computed predictions/weights were brought back into this repo.

---

## 8. Evaluation (cross-cutting)

- Metric throughout: **event-level F1**, not window-level accuracy or window-level F1 — the
  project is explicit that "always predict no seizure" scores 96% window accuracy, which is
  meaningless here (`docs/MODEL_COMPARISON.md`-style reasoning, echoed in the detection logic's
  entire design around found/missed/false-alarm counts rather than raw accuracy).
  - "Found": any overlap between a detected interval and the true seizure interval.
  - "False alarm": a detected interval with no overlap to any true seizure.
  - FA/hour normalizes false alarms by total recorded hours, since recordings vary in length.
  - Onset/offset error (seconds): detected − real, averaged only over found seizures (positive =
    detected late).
- **Three different threshold-selection protocols are used across the project, and the report
  must keep them clearly labeled, since they are not interchangeable**:
  1. "Best threshold" in `docs/MODEL_COMPARISON.md` (12 classical models) — chosen and scored on
     the same out-of-fold predictions; explicitly flagged as optimistic.
  2. "Held-out-fold" in `docs/ENSEMBLE_RESULTS.md` (ensembles) and the CNN "nested threshold" in
     `docs/DEEP_LEARNING_RESULTS.md` — chosen on other folds/held-out recordings only, not
     optimistic.
  3. A fully nested, SzCORE-compatible re-scoring of the 12 classical models
     (`scripts/evaluate_nested.py`) — the apples-to-apples comparison that would let the SVM be
     compared fairly against the CNNs' nested numbers — is **not yet complete** (3 of 12 models
     cached, no final report). The report's evaluation chapter should say plainly that the
     012-model table's 0.894 and the CNNs' nested 0.815/0.811/0.774 are not fully
     apples-to-apples yet, for exactly this reason, and name this as the single biggest piece of
     unfinished evaluation work.
- Baseline to beat, stated plainly: SVM unbalanced, 0.894. Best result found so far: the
  SVM+ShallowConvNet blend, 0.901 (small gain, not yet stress-tested). Deep learning alone does
  not beat the baseline. This is the honest, current state of "classical vs. deep learning" for
  this project.

---

## 9. The application

- `app.py` (1,357 lines) + `src/app_data.py`, `src/app_figures.py`, `src/app_home.py` (all read
  in full or structurally skimmed). `app.py`'s own module docstring states only precomputed
  files are read; nothing is trained inside the app.
- 4 pages, selected via a `?page=` query parameter (`app.py:64-67`): **Home**, **Visualisation**,
  **Prediction** (legacy alias: `recording`), **Comparison** (legacy alias: `overview`) — the
  legacy aliases imply the app was restructured/renamed at some point, kept for old links.
- `src/app_home.py`: the Home page — explains the method and data preparation in plain language
  for a visitor, including the exact training-mode descriptions used in the UI, e.g. (quoted
  directly, `src/app_home.py:29`): *"Every training window plus extra seizure windows: shifted
  by 0.25 to 0.75 s and mirrored left-right."* for the augmented mode.
- `src/app_data.py`: the data-loading layer; raises a `MissingData` exception if a required
  precomputed file is absent, rather than failing silently.
- `src/app_figures.py` (582 lines): builds every Plotly figure used across the app —
  `confusion_figure`, `curves_figure`, `recording_figure`, `spectrum_figure`,
  `patient_comparison_figure`, `training_distribution_figure`, and others.
- "Calm clinical" redesign, concretely (`src/app_figures.py:20-36`, `THEMES` dict): light theme
  is the default — background `#F5F7FA`, panels `#FFFFFF`, text `#0F172A`, teal accent
  `#0B7285`; a parallel dark theme exists with the same structure. The 19 channels are colored by
  scalp region (frontal/central/temporal/posterior), consistently across both themes.
- Sidebar channel selector groups the 19 channels by scalp region (`app.py:41-45`).
- Comparison page's ranking table supports 12 different sort metrics (`app.py:48-62`, the
  `RANKING` dict): event F1/recall/precision, found/missed/false alarms, FA/hour, window
  F1/recall/precision, onset/offset error, and training time — letting a visitor re-rank the 12
  classical models (and CNNs/blends) by whichever criterion matters to them, not just F1.
- Redesign history, from the git log (`git log --oneline`, read in full): `3801150` first added
  detection + the Dash app; `ef16989` split it into 3 pages; `9fb551d` proposed the redesign
  (mockups in `docs/design/`, read: `REDESIGN_PROPOSAL.md` + 4 mockup PNGs + an HTML mockup);
  `3b6f9bf` implemented the "calm clinical, light by default" redesign; `c494a02` added the Home
  page; `caefbdf` sped the app up. Screenshots of the current app exist in
  `outputs/figures/app/` (light/dark, multiple widths, for Home/Visualisation/Prediction/
  Comparison, plus "before" screenshots from the pre-redesign version and several named UI-state
  captures like `state_overlay_1440.png`, `state_reference_1440.png`) — real images, safe to
  include directly in the report's application chapter.

---

## 10. Problems and lessons learned

Real, verifiable problems hit during the project, each traceable to a source:

1. **Clock alignment with no absolute start time anywhere in the raw files** — had to be solved
   from first principles (`elapsed = clock − DebutTrace` + midnight rollover), with two tempting
   shortcuts explicitly rejected because they would fabricate labels. §2 above.
2. **Conflicting/duplicate annotation rows** for two recordings, resolved by scoring candidates
   against the real signal rather than picking by hand. §2 above.
3. **One recording unusable** (`200625A_F`: fewer channels, no annotations) — excluded rather
   than guessed at. §2 above.
4. **Class imbalance by construction** (3.96% seizure windows) — the central modeling problem;
   tackled three ways (unbalanced/balanced/augmented), with balanced and augmented training both
   turning out to hurt overall F1 despite raising recall — §4, §5 above, a genuine, reported
   negative result, not a bug.
5. **The `event_counts()` IndexError** in the CNN training script — a real crash, traced to a
   pandas index-label-vs-position confusion that only manifested on certain random validation
   splits, fixed in commit `37eb36c`. §7 above.
6. **CNN deep learning did not beat the classical SVM** — not a bug, a finding: with 93 seizures
   across 21 patients, a raw-signal CNN cannot out-learn 320 hand-crafted features. §7 above.
7. **A memory-heavy, long-running evaluation script had to be killed once** —
   `scripts/evaluate_nested.py`'s full run was killed for lack of RAM (`HANDOFF.md:26`), which is
   why only 3 of 12 models have nested scores today — an honest, still-open limitation, not
   something to paper over.
8. **A wording bug in `docs/ENSEMBLE_RESULTS.md` itself** ("ties the baseline" for a result that
   actually beats it by +0.007) — worth a short mention in the lessons chapter as an example of
   why every number in this report was independently re-verified against source files rather
   than copied from existing prose.
9. **Repo history was rewritten at some point**, leaving `HANDOFF.md`'s own bug-fix citation
   (`b026a70`) pointing at a commit no longer reachable from `main` — another small reminder that
   documentation drifts and the underlying commit should be re-checked rather than trusted at
   face value. §0 above.
10. **Unused data left on disk** — `data/excel/` (2 GB) is dead weight, not used by any later
    pipeline step; flagged in `HANDOFF.md` rather than silently left for someone else to
    rediscover.

---

## 11. Conclusion / future work (facts to ground it, not written yet)

Open items as of 2026-10-17, each independently confirmed on disk/in docs above, to be turned
into the report's "future work" list:
- Finish `scripts/evaluate_nested.py` for all 12 classical models (currently 3/12), to get a fair
  nested-threshold comparison against the CNNs' own nested numbers.
- Get a neurologist's review of the false alarms, especially in `210504B_C` and `210427B_C`,
  since some may be real unannotated seizures that would raise every model's measured precision.
- Stress-test whether the SVM+ShallowConvNet blend's +0.007 gain over the plain SVM is robust or
  within noise (`HANDOFF.md` itself says this was "not tested for noise").
- Run and verify the CNN loading example in `models/cnn/README.md` (untested, no local PyTorch).
- Clean up `data/excel/` (2 GB, unused).

---

## 12. Appendix material (repo structure / how to run)

- Repo structure: see `docs/DATA_LAYOUT.md` for the full `data/` tree (verified to match
  `src/paths.py` exactly — same constants: `OFFICIAL`, `OFFICIAL_RAW`, `OFFICIAL_8CH` (archive),
  `PROVISIONAL_8CH` (archive), `AUGMENTED = "augmented_shift_mirror"`). `scripts/` = one script
  per pipeline step, each independently runnable (`CLAUDE.md:15`). `models/` holds
  `registry.json` + the 12 classical `.joblib` models + `models/cnn/` for CNN weights
  (`CLAUDE.md:16-18`).
- Key commands, verbatim from `CLAUDE.md:24-29` (current, re-verified against the actual script
  names on disk today — the old `docs/KAGGLE.md`/`export_kaggle.py` commands from earlier in
  this session's context **no longer exist** and must not appear in the report):
  ```
  python app.py                                                  # run the app
  python scripts/build_dataset.py --annotations official         # rebuild the official dataset
  python scripts/train_registry.py                                # train the 12 classical models
  python scripts/export_cnn_data.py                               # package data for GPU training -> data/cnn/
  python scripts/evaluate_cnn.py                                  # score the CNNs
  python scripts/ensemble_predictions.py --models <key> <key>     # blend already-trained models
  ```
- Exact pinned dependency versions (`requirements.txt`, read in full): numpy 2.5.3, scipy 1.18.1,
  pandas 3.0.6, pyarrow 25.0.1, scikit-learn 1.9.1, joblib 1.6.0, mat-io 1.0.0, h5py 3.16.0,
  openpyxl 3.1.5, XlsxWriter 3.2.9, PyYAML 6.0.3, matplotlib 3.11.2, seaborn 0.13.2,
  timescoring 0.0.7, dash 4.4.1, dash-mantine-components 2.8.0, dash-iconify 0.1.2, plotly 6.9.0,
  plotly-resampler 0.11.1, selenium 4.50.0 (optional, screenshot script only). No `torch` entry —
  CNN training happens on external GPU hardware, confirmed.

---

## Docs inventory — current vs. archived/superseded (do not cite archived ones as current state)

**Current, safe to cite directly for facts/numbers:**
`DATA_LAYOUT.md`, `ANNOTATION_DECISIONS.md`, `MODEL_COMPARISON.md`, `ENSEMBLE_RESULTS.md`
(with the "ties" wording correction noted in §6), `DEEP_LEARNING_RESULTS.md`,
`DATASET_PARTS_I_VI_OFFICIAL.md` / `_OFFICIAL_RAW.md` (19-channel, official annotations).

**Primary-source / investigative, good for narrative detail but checkpoints, not final state:**
`DATA_NOTES.md` (initial raw-data inspection, superseded in its open questions by
`ANNOTATION_DECISIONS.md`'s resolutions, but still the right source for "what the raw files
looked like"), `OFFSET_ALIGNMENT.md`, `TIME_ALIGNMENT.md` (clock-alignment investigation,
explicitly states no files were modified), `CHANNELS_8_VS_19.md` (a deliberate comparison that
led to the 19-channel decision — fine to cite as "we checked, 19ch won," not as ongoing work),
`PART_VII_ANALYSIS_OFFICIAL.md` / `_OFFICIAL_RAW.md` (exploratory feature-correlation analysis on
the current 19ch data — numbers are current, no model trained here).

**Archived / superseded — do NOT present as current project state:**
`docs/archive_8ch/` (`PARTS_VIII_IX.md`, `PARTS_X_XI.md`, `PART_VII_ANALYSIS_OFFICIAL.md` copies)
— an earlier classical-model comparison and detection-methodology demo run on the **8-channel**
data with a different (80/20 random-split) protocol; useful only for narrative ("how we first
explored GroupKFold-vs-random-split leakage, why accuracy is misleading, how detection-from-
probabilities works"), never for final numbers. `DATASET_PARTS_I_VI.md` (not the `_OFFICIAL`
version) — the original 8-channel/provisional-annotation dataset report, which once excluded
`210427B_C`; that exclusion was reversed once the official workbook arrived, so this file's
recording/seizure counts are stale. `docs/design/` (`REDESIGN_PROPOSAL.md` + mockups) — a design
artifact behind the app-redesign commits, good for one narrative paragraph in the app chapter,
not a results doc. Removed entirely (confirmed not on disk): `KAGGLE.md`, `RESEARCH_PROMPT.md`,
`NESTED_EVALUATION.md` — do not reference these paths/filenames in the report at all.

**Figures available on disk (real, not to be invented), useful for the report:**
`outputs/figures/official/models/confusion_matrices_groupkfold.png`,
`.../metrics_by_setting.png`, `.../detection/threshold_sweep.png`,
`.../detection/error_examples.png`, `.../detection/<recording>.png` (per-recording detection
plots, 21 available), `.../preprocessing/211104B_D_seizure3_filtering.png` (a real before/after
filtering example), `.../channels/importance_by_channel.png`,
`.../part_vii/correlation_*.png` (feature-correlation heatmaps),
`outputs/figures/offset_alignment/*_offset_alignment.png` (per-recording clock-alignment plots,
21 available, plus `210427B_C_candidate_comparison.png` for the duplicate-row resolution),
`outputs/figures/app/{home,visualisation,prediction,comparison}_{light,dark}_*.png` (real app
screenshots, multiple pages/themes/widths), `docs/design/mockup_*.png` (redesign mockups, for
an app-chapter "before the redesign" aside).
