# 8 channels vs 19 channels

Same 21 recordings, same annotations and labels, same windows and the same folds. Only the channels used for the features change: 8 channels (Fp1, Fp2, C3, C4, T3, T4, O1, O2; 144 features) versus the 19 EEG channels of the 10-20 montage (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2; 320 features). ECG, EMG and SLI are excluded in both. `200625A_F` has only 8 EEG channels, but it is excluded for lack of annotations, so every recording used has all 19.

The 8-channel files are kept in `outputs/archive_8ch/` and `docs/archive_8ch/`. Annotation decisions (duplicate rows, corrected end time) are still scored on the original 8 channels, so the labels are identical in both runs.

## Window level — GroupKFold by recording (5 folds), unbalanced training

| Model | F1 8 ch | F1 19 ch | Δ F1 | Recall 8 ch | Recall 19 ch | Precision 8 ch | Precision 19 ch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.460 ± 0.139 | 0.497 ± 0.108 | +0.037 | 0.565 | 0.589 | 0.426 | 0.446 |
| Random Forest | 0.717 ± 0.043 | 0.714 ± 0.044 | -0.003 | 0.672 | 0.663 | 0.789 | 0.792 |
| KNN | 0.680 ± 0.087 | 0.645 ± 0.070 | -0.035 | 0.616 | 0.565 | 0.782 | 0.776 |
| SVM | 0.689 ± 0.028 | 0.709 ± 0.038 | +0.020 | 0.611 | 0.637 | 0.805 | 0.813 |

## Window level — GroupKFold, undersampled (balanced) training

| Model | F1 8 ch | F1 19 ch | Δ F1 | Recall 8 ch | Recall 19 ch | Precision 8 ch | Precision 19 ch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.343 ± 0.195 | 0.287 ± 0.099 | -0.056 | 0.853 | 0.836 | 0.232 | 0.179 |
| Random Forest | 0.502 ± 0.159 | 0.489 ± 0.155 | -0.013 | 0.866 | 0.874 | 0.371 | 0.356 |
| KNN | 0.463 ± 0.147 | 0.471 ± 0.127 | +0.009 | 0.836 | 0.824 | 0.341 | 0.347 |
| SVM | 0.530 ± 0.166 | 0.479 ± 0.173 | -0.052 | 0.854 | 0.857 | 0.409 | 0.359 |

## Window level — random stratified 80/20 split, unbalanced training

| Model | Accuracy 8 / 19 | Precision 8 / 19 | Recall 8 / 19 | F1 8 ch | F1 19 ch | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.975 / 0.974 | 0.686 / 0.671 | 0.667 / 0.655 | 0.676 | 0.663 | -0.014 |
| Random Forest | 0.985 / 0.985 | 0.866 / 0.871 | 0.727 / 0.731 | 0.790 | 0.795 | +0.004 |
| KNN | 0.982 / 0.982 | 0.826 / 0.843 | 0.707 / 0.671 | 0.762 | 0.747 | -0.015 |
| SVM | 0.983 / 0.983 | 0.865 / 0.867 | 0.671 / 0.683 | 0.756 | 0.764 | +0.008 |

## Event level — whole seizures, out-of-fold predictions, each model at its best threshold

| Model | Threshold 8 / 19 | Event F1 8 ch | Event F1 19 ch | Δ | Found 8 / 19 (of 93) | False alarms 8 / 19 |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.05 / 0.05 | 0.306 | 0.290 | -0.016 | 93 / 91 | 563 / 561 |
| Random Forest | 0.68 / 0.67 | 0.857 | 0.867 | +0.010 | 86 / 87 | 26 / 26 |
| KNN | 0.81 / 0.61 | 0.841 | 0.828 | -0.013 | 78 / 86 | 19 / 37 |
| SVM | 0.71 / 0.82 | 0.882 | 0.894 | +0.012 | 86 / 86 | 20 / 16 |

The threshold of each run is chosen on its own out-of-fold predictions, so both event-level columns are slightly optimistic in the same way.

## Training time

Seconds to fit and predict, measured inside `train_models.py`. The two runs were timed separately on the same machine, so differences of a few tenths of a second are noise.

| Model | Random split, unbalanced: 8 ch | 19 ch | Ratio | GroupKFold mean per fold, unbalanced: 8 ch | 19 ch | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 8.5 | 23.0 | ×2.7 | 9.7 | 23.5 | ×2.4 |
| Random Forest | 3.7 | 5.2 | ×1.4 | 3.6 | 5.3 | ×1.5 |
| KNN | 0.6 | 1.2 | ×1.9 | 0.7 | 1.2 | ×1.7 |
| SVM | 4.9 | 10.2 | ×2.1 | 5.0 | 9.7 | ×2.0 |

## Feature importance — final Random Forest trained on all 21 recordings

Impurity-based importances (they sum to 1). When several features carry the same information the importance is split between them, so read these as indications of where the model looks, not as a ranking of medical relevance.

### Top 20 features with 19 channels

| Rank | Feature | Channel | Region | Importance |
|---:|---|---|---|---:|
| 1 | `O1_beta_power` | O1 | Occipital | 0.0409 |
| 2 | `P3_beta_power` | P3 | Parietal | 0.0383 |
| 3 | `O2_beta_power` | O2 | Occipital | 0.0358 |
| 4 | `C3_beta_power` | C3 | Central | 0.0295 |
| 5 | `F3_beta_power` | F3 | Frontal | 0.0262 |
| 6 | `P4_beta_power` | P4 | Parietal | 0.0255 |
| 7 | `Cz_beta_power` | Cz | Central | 0.0248 |
| 8 | `P3_alpha_power` | P3 | Parietal | 0.0211 |
| 9 | `C4_beta_power` | C4 | Central | 0.0192 |
| 10 | `T5_beta_power` | T5 | Temporal | 0.0153 |
| 11 | `Pz_beta_power` | Pz | Parietal | 0.0152 |
| 12 | `T3_beta_power` | T3 | Temporal | 0.0140 |
| 13 | `avg_beta_power` | all (average) | All channels (average) | 0.0130 |
| 14 | `O2_alpha_power` | O2 | Occipital | 0.0123 |
| 15 | `T6_beta_power` | T6 | Temporal | 0.0118 |
| 16 | `Pz_min` | Pz | Parietal | 0.0117 |
| 17 | `T4_beta_power` | T4 | Temporal | 0.0116 |
| 18 | `P4_alpha_power` | P4 | Parietal | 0.0108 |
| 19 | `Pz_alpha_power` | Pz | Parietal | 0.0097 |
| 20 | `Cz_alpha_power` | Cz | Central | 0.0090 |

For reference, the top 10 with 8 channels: `O2_beta_power` (0.056), `O1_beta_power` (0.048), `T4_beta_power` (0.042), `avg_alpha_power` (0.032), `C3_beta_power` (0.030), `avg_beta_power` (0.028), `C3_alpha_power` (0.027), `C4_beta_power` (0.026), `T3_alpha_power` (0.025), `T3_beta_power` (0.022).

### Which regions matter most

| Region | Channels | Share of importance | Share per channel |
|---|---:|---:|---:|
| Frontal | 7 | 26.4% | 3.8% |
| Parietal | 3 | 19.3% | 6.4% |
| Central | 3 | 17.4% | 5.8% |
| Occipital | 2 | 16.6% | 8.3% |
| Temporal | 4 | 15.7% | 3.9% |
| All channels (average) | — | 4.6% | — |

Per channel, the occipital region carries the most importance (8.3% per channel), followed by the parietal region (6.4%); the frontal region carries the least (3.8%). The five most used channels are O1 (9.4%), P3 (7.6%), O2 (7.2%), Cz (6.3%), C3 (6.2%). The 11 channels that were not used before (F7, F3, Fz, F4, F8, Cz, T5, P3, Pz, P4, T6) receive 52.0% of the importance. Features averaged over all channels receive 4.6%.

By type of feature: beta_power 34.4%, alpha_power 11.7%, amplitude 6.9%, min 6.4%, max 5.1%, variance 4.5%, energy 3.9%, std 3.9%.

Figure: [importance by channel](../outputs/figures/official/channels/importance_by_channel.png).

## Summary

- Window level (GroupKFold, unbalanced): F1 changes by Decision Tree +0.037, Random Forest -0.003, KNN -0.035, SVM +0.020.
- Event level: F1 changes by Decision Tree -0.016, Random Forest +0.010, KNN -0.013, SVM +0.012. Best model: SVM with 8 channels (0.882), SVM with 19 channels (0.894).
- These differences come from 5 folds over 21 recordings and 93 seizures. A change of 0.01–0.02 in event F1 is one or two seizures or false alarms and is within fold-to-fold variation; only larger, consistent changes should be read as a real effect.

Reproduction: `python scripts/compare_channel_sets.py` (needs `outputs/archive_8ch/`).
