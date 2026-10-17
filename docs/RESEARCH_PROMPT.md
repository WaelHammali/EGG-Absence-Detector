# ROLE

You are a senior research engineer in **EEG signal processing, deep learning for biomedical time series, and ML deployment**. You do thorough literature and code research (papers, benchmarks, GitHub repositories, official documentation), you verify that what you cite exists, and you give practical, honest advice adapted to a small dataset and limited hardware.

# MISSION

I have built a classical machine-learning system that detects **absence seizures** in scalp EEG. I want your research on three things:

1. **Deep learning**: the best deep learning model I can train on the **cleaned multichannel EEG signal itself** (no hand-made features) to detect seizures, and how to train and evaluate it so it can fairly replace my 4 classical models.
2. **Data augmentation**: the best augmentation methods for EEG seizure detection that fit my data, given that my first attempt did not help.
3. **Deployment**: what I must do to make my web application ready to be deployed and used by others.

Read the whole context first. All numbers below are measured on my data.

---

# CONTEXT

## 1. Project

- **Type**: university data-mining project (Tunisia).
- **Clinical goal**: find absence seizures automatically in routine EEG recordings, and show the neurologist where the model agrees with the annotations, where it misses a seizure and where it raises a false alarm. Absence seizures are generalized, about 3 Hz spike-and-wave discharges, usually 3–20 s long, mostly in children.
- **Stack**: Python 3.14, NumPy, SciPy, pandas, PyArrow, scikit-learn 1.9, joblib; web app in Dash 4 + dash-mantine-components + Plotly + plotly-resampler.
- **Code**: a Git repository with one script per step and Markdown reports.

## 2. Raw data

- **Files**: 22 EEG recordings in MATLAB format. `<id>_0000d.mat` holds a MATLAB timetable with one column per channel (for example `EEGFp1`); `<id>_0000h.mat` holds a header. One recording is excluded (no annotations, only 8 EEG channels), so **21 recordings are used**.
- **Sampling rate**: 256 Hz.
- **Duration**: 8.7 hours in total (31,440 s). 19 recordings last about 20–30 minutes; two last 52 and 60 minutes.
- **EEG channels**: 19, international 10-20 system, referential montage, in this order: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2. Every used recording has all 19.
- **Other channels**: ECG, EMG (1–2 channels) and a photic-stimulation trigger (SLI), present only in some recordings. They are never model inputs; the app shows them as references.
- **Patients**: 21 recordings, assumed to be 21 different patients (not confirmed). 10 male, 11 female. Ages 5–25 years: 19 patients are 5–15 years old, the other two are 17 and 25. One patient is labelled "atypical absences".
- **Protocol**: routine EEG with **hyperventilation (HPN)** and **intermittent photic stimulation (SLI)**, both used to provoke absences, and eyes open/closed periods.
- **Technician notes**: time-stamped free text, which I categorized: HPN 375, SLI 233, artifact 160, eyes closed 50, movement 48, eyes open 38, seizure note 28, other 76.

## 3. Labels

- **Source**: an Excel workbook from the neurologist with the clock time of each seizure start and end. Aligned to the signal with `start_s = seizure clock time − recording start clock time` (midnight rollover handled). Two recordings missing from the updated workbook use an earlier, carefully aligned version. Special cases (duplicate rows, one corrected end time, one ambiguous recording kept) are documented in a YAML decision file.
- **Seizures**: 93 in total, 1 to 12 per recording (median 3, mean 4.4).
- **Duration**: 2 to 60 s, median 10 s, mean 12.5 s.
- **Total seizure time**: 1,162 s, so **3.7 % of the recorded time is seizure**.
- **Granularity**: per-sample 0/1 labels.
- **Known label problems**:
  - In recording `210427B_C`, several bursts look like absences but only one is annotated, so some "false alarms" are probably real seizures.
  - Clock alignment may be off by about 1 s.
  - Absences provoked by hyperventilation may be under-annotated.

## 4. Cleaning (already done; this is what a deep model will receive)

- **Filter**: zero-phase Butterworth band-pass, 0.5–40 Hz, order 4 (`scipy.signal.sosfiltfilt`), applied per channel to each full recording before any segmentation. It removes large DC offsets and drift and the 50 Hz mains noise, and keeps the 3 Hz rhythm.
- **Stored format**: one Parquet file per recording with `Time`, the 19 channels as `float32` (µV) and the per-sample label `Class`.
- **Not done**: no re-referencing (no average reference, no bipolar montage), no artifact removal (no ICA), no amplitude normalization.
- An unfiltered version of the dataset also exists for comparison.

## 5. Classical pipeline (current system)

1. **Windows**: 2 s (512 samples) with a 1 s step (50 % overlap). A window is class 1 if at least 50 % of its samples are seizure. This gives **31,419 windows; 1,244 are seizure windows (3.96 %)**.
2. **Features**: 16 per channel.
   - Time (8): mean, std, variance, min, max, peak-to-peak amplitude, RMS, energy.
   - Frequency (8, Hann window + FFT): absolute and relative power in delta 0.5–4, theta 4–8, alpha 8–13 and beta 13–30 Hz.
   - With the across-channel average of each feature: **320 features**.
3. **What the feature analysis showed**:
   - The best single features are amplitude, std and RMS, especially at central channels (C3, C4, Cz). Their AUC is about 0.93 across all windows and about 0.98 within a recording, so the amplitude level differs a lot between patients.
   - Relative theta power is lower during seizures.
   - The features are highly redundant: 1,033 feature pairs have |Pearson r| ≥ 0.95.
4. **Algorithms**: Decision Tree, Random Forest, KNN, SVM (RBF kernel; Platt scaling for probabilities). Default scikit-learn hyperparameters; StandardScaler for KNN and SVM.
5. **Validation**: `GroupKFold`, 5 folds, grouped by recording, the same folds everywhere. Every window is predicted by a model that never saw its patient (out-of-fold).
   - A random stratified 80/20 split by window gave window-level F1 about 0.10 higher: leakage between overlapping windows and recording-specific traits.
   - Fold-to-fold variability is large: mean window F1 ranges from 0.25 to 0.59 depending on which recordings are tested.
6. **Event detection**:
   - Windows with probability ≥ threshold are joined into events. Events less than 2 s apart are merged, and events shorter than 2 s are dropped.
   - Matching: a real seizure overlapped by any event is *found* (TP), a seizure with no event is *missed* (FN), and an event overlapping no seizure is a *false alarm* (FP). Event precision = detections that hit a seizure / all detections.
   - Threshold: swept from 0.05 to 0.99; each model keeps the one with the best event F1. This is chosen on the same out-of-fold predictions, so it is slightly optimistic; no nested CV yet.
7. **Imbalance handling**, three training modes, applied to training folds only (test folds always keep the real windows and the real 3.96 %):
   - **Unbalanced**: training folds as recorded.
   - **Balanced**: random undersampling to 50/50, repeated with 5 seeds.
   - **Augmented**: extra seizure windows cut from the real signal: windows shifted by 0.25, 0.5 and 0.75 s inside seizures (still ≥ 50 % seizure), plus the left-right mirror of every seizure window (Fp1↔Fp2, F7↔F8, F3↔F4, T3↔T4, C3↔C4, T5↔T6, P3↔P4, O1↔O2; Fz, Cz, Pz unchanged). That adds 8,230 windows, about 24 % seizure in training.
8. **8 vs 19 channels**: using all 19 channels instead of the original 8 changed the event F1 by only about ±0.01.

## 6. Current results (baseline to beat)

Event-level F1 at each model's best threshold, out-of-fold, all 21 patients:

| Algorithm | Unbalanced | Balanced | Augmented |
|---|---:|---:|---:|
| SVM | **0.894** | 0.821 | 0.862 |
| Random Forest | 0.867 | 0.785 | 0.854 |
| KNN | 0.828 | 0.519 | 0.692 |
| Decision Tree | 0.290 | 0.148 | 0.171 |

- **Best model, SVM unbalanced**: threshold 0.82. It finds 86 of 93 seizures (event recall 0.92) with 16 false alarms (1.8 per hour), event precision 0.87. Window-level F1 0.57. Mean onset error +1.6 s (late), mean offset error −2.0 s (early).
- **SVM augmented**: finds 90 of 93 seizures but raises 32 false alarms.
- **Balanced modes**: best thresholds reach 0.98–0.99, so their probabilities are poorly calibrated on real proportions.
- **Accuracy is not used**: "always no seizure" scores 96 %.

### Where the best model fails

- **Missed seizures (7)**:

  | Recording | Patient | Missed |
  |---|---|---|
  | 190304A_E | 17 years | 1 of 1 |
  | 191113A_D | 25 years | 2 of 2 |
  | 230718A_A | 7 years | 3 of 7 |
  | 210208B_G | 14 years | 1 of 4 |

  The two oldest patients get no detection at all.
- **False alarms (16)**: 7 in `210504B_C` (52-minute recording), 2 each in `210427B_C`, `210914B_A` and `211027B_C`. About 41–46 % of the RF/SVM false alarms fall during hyperventilation, which produces high-amplitude slow activity similar to absences and also provokes real ones.

## 7. The web application (current state)

- **Pages**:
  - *Home*: explains the method.
  - *Visualisation*: EEG viewer for 19 channels with real seizures drawn on the curves, technician notes and reference channels.
  - *Prediction*: one model on one patient, with probability, detections, threshold slider and found/missed/false alarm summary.
  - *Comparison*: the 12 models, with ranking, threshold and precision-recall curves, confusion matrices, a per-patient timeline and per-patient F1.
- **Data**: the app reads only precomputed files: Parquet signals and out-of-fold predictions, a JSON model registry, joblib models (the KNN models are 72–92 MB). Nothing is trained at startup.
- **Runtime**:
  - Flask development server, single process (`python app.py`).
  - The full-resolution figures are cached in a Python dict in server memory, per browser session (plotly-resampler).
  - Fonts and icons are loaded from the internet.
  - Rendering many Dash components was slow; large tables are now sent as one HTML block.
- **Not implemented**: uploading new recordings, running a model on new data, authentication, automated tests, CI, Docker, logging.

## 8. Constraints

- **Small data**: 21 patients, 93 seizures, 8.7 hours. All evaluation must be **patient-independent** (my 5 GroupKFold folds, or leave-one-patient-out), never a random split of windows.
- **Hardware**:
  - Laptop: 16 CPU threads, 15 GB RAM, **no NVIDIA GPU**.
  - Free cloud GPUs (Google Colab, Kaggle) are acceptable.
  - State the hardware each proposal needs.
- **Fair comparison**: same folds, same event-matching rules, same metrics, and ideally a nested threshold choice for both the deep model and the SVM.
  - Main metric: event F1.
  - Also: recall (seizures found of 93), false alarms per hour, onset/offset error, window-level F1.
- **Inputs**: the 19 cleaned EEG channels. Re-referencing or montage changes are allowed if justified. ECG/EMG/SLI only as clearly marked optional extras.
- **Timeline**: a few weeks, one person, solid Python and machine learning basics, beginner in deep learning.

---

# QUESTIONS

## Part A: deep learning model

1. **Ranked candidates** for seizure detection from raw or minimally processed multichannel scalp EEG, focused on **absence or generalized seizures**, **pediatric EEG** and **small datasets**. For each one give:
   - paper (title, year, link) and a reliable code repository (PyTorch, `braindecode`, MNE);
   - input: window length, overlap, normalization (per channel, per recording, robust z-score), montage, raw signal vs spectrogram;
   - output: one label per window, or sequence/per-sample labeling (segmentation, U-Net-like, temporal context across windows); which suits precise onset/offset;
   - reported results on public benchmarks (TUH EEG Seizure Corpus, CHB-MIT, Siena, others), whether they are patient-independent, and any results on absence (ABSZ) or generalized seizures;
   - number of parameters, and training time on CPU vs GPU for about 9 hours of 19-channel, 256 Hz data;
   - whether 93 seizures is enough, and what makes it work (pretraining, strong regularization, small models).

   Consider at least compact CNNs (EEGNet, ShallowConvNet, DeepConvNet), temporal convolutional networks, CNN + LSTM/GRU, transformers, graph neural networks over electrode positions, and **pretrained EEG foundation models** that can be fine-tuned. Verify that each is available and usable today.
2. **Transfer learning**:
   - which public datasets contain absence or generalized seizures (for example the TUH absence class), and their access procedure and licence;
   - how to map their channels and sampling rates to my 19 channels at 256 Hz;
   - whether pretraining then fine-tuning is realistic on my hardware.
3. **Imbalance in deep learning**: weighted cross-entropy, focal loss, balanced batch sampling, or real proportions plus threshold tuning. Explain, given that rebalancing hurt my classical models.
4. **Patient variability**: the oldest patients are always missed and amplitude differs between patients. What normalization or domain-adaptation methods help: per-recording normalization, adaptive batch normalization, test-time adaptation?
5. **Post-processing and threshold**: probability smoothing, hysteresis, minimum duration, merging, and how to choose the threshold with nested cross-validation.
6. **Evaluation standard**: which event-based scoring is used in the literature (for example the TUSZ scoring methods or the SzCORE framework), so that my results are comparable.
7. **Honest expectation**: can a deep model beat an RBF-SVM on 320 amplitude and band-power features with 93 seizures? If unlikely, which hybrid options are worth it (deep embeddings + classical classifier, deep model + handcrafted features, ensembles)?
8. **First recipe**: the single approach I should try first, with model, hyperparameters, input pipeline (how to cut and normalize from my per-recording Parquet files), loss, optimizer, epochs, early stopping inside the training folds, a PyTorch training-loop outline, and the expected training time on my hardware.

## Part B: data augmentation

My signal-level augmentation (sub-second shifts + left-right mirror) raised recall from 86 to 90 seizures found, but doubled false alarms and lowered F1.

1. **Methods proven for EEG or seizure detection**, with references and reported effect sizes, especially with deep models. Consider:
   - time: random crop or shift, amplitude scaling, Gaussian noise, time masking, time reversal, sign flip, smooth time warping;
   - frequency: Fourier-transform surrogates, frequency shift, random band-stop;
   - spatial: channel dropout, hemispheric mirroring, sensor rotation or interpolation;
   - mixing: mixup, cutmix, pasting seizure segments into real background EEG;
   - generative: GANs, VAEs, diffusion. Are they reliable with only 93 seizures?
2. **Fit with absence seizures**: for each method, does it keep the 3 Hz spike-and-wave morphology valid? Give parameter ranges.
3. **Reducing false alarms**: augmenting or mining the **negative** class. For example hard negatives from hyperventilation and artifacts, using the technician's HPN timestamps, or contrastive or self-supervised learning on unlabelled EEG.
4. **Practical use**: libraries that implement these online during training (for example `braindecode.augmentation`), and how to validate them correctly: training folds only, test on real untouched data.

## Part C: deployment readiness

Goal: a neurologist uploads a new recording and gets the detected seizures, a viewer and a report. Give a prioritized checklist with tools and examples:

1. **Inference pipeline**:
   - reading EDF/EDF+/BDF and my MATLAB format;
   - channel-name mapping and missing-channel handling;
   - resampling to 256 Hz, the same filter, the model, post-processing;
   - export to CSV, EDF+ annotations and a PDF report.
2. **Model packaging**:
   - TorchScript or ONNX;
   - fast CPU inference for a 1-hour recording;
   - model and data versioning (for example DVC or MLflow) and a model registry;
   - whether to put inference behind an API (FastAPI) separate from the Dash front end.
3. **Production Dash**:
   - gunicorn configuration (workers, threads);
   - replacing the in-memory per-session figure cache with a shared one (Redis, flask-caching) or another design;
   - background callbacks or job queues for long computations;
   - memory use with long signals.
4. **Packaging and hosting**:
   - Docker image and pinned dependencies;
   - free or low-cost hosting for a student project that can hold about 1 GB of data and a 90 MB model;
   - HTTPS;
   - self-hosted fonts and icons.
5. **Quality**:
   - unit tests (filter, windowing, event matching), callback tests, end-to-end tests;
   - GitHub Actions CI;
   - logging, error monitoring, performance profiling.
6. **Security and privacy**:
   - authentication;
   - anonymizing EDF headers (names, dates) and data retention;
   - GDPR and the Tunisian personal-data law (Organic Law No. 2004-63);
   - how to state clearly that the tool is **not a medical device**, and what certification (EU MDR, FDA software as a medical device) would be required otherwise.
7. **Clinical usability**:
   - viewer features clinicians expect: montage switching (bipolar double-banana, average reference), sensitivity in µV/mm, time base, annotation editing, report generation;
   - how to collect clinician corrections to improve the labels.

---

# OUTPUT FORMAT

- **One section per part (A, B, C)**, recommendations ranked best first. For each: what it is, why it fits my case, expected benefit, effort (hours or days), risk, hardware, and sources.
- **Sources**: a working link for every claim (paper, repository or documentation). Prefer peer-reviewed work from 2018 onward. Say clearly when evidence is weak, not patient-independent, or not about absence seizures.
- **Comparison table** of the top deep learning candidates: input, output type, parameters, CPU feasibility, best reported patient-independent result, code availability.
- **Week-by-week plan for 4 weeks**:
  - what to implement first;
  - how to compare fairly with the SVM baseline (event F1 0.894, 86/93 found, 16 false alarms, 1.8 per hour);
  - which result would justify switching to the deep model.
- **Honest verdict**: if deep learning is unlikely to beat the SVM with this data, say so and explain what would change it (more data, pretraining, label cleaning).
- **Language**: clear and simple, short sentences. Explain advanced deep-learning terms briefly the first time.
