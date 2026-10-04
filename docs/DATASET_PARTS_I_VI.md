# Dataset report — Parts I–VI

The dataset contains 2 s windows (512 samples at 256 Hz) with a 1 s hop. A window is positive when at least 256 samples are inside a cleaned seizure interval. Features use only Fp1, Fp2, C3, C4, T3, T4, O1 and O2.

`200625A_F` is excluded because it is unannotated. `210427B_C` is excluded because both competing alignments contain convincing spike-and-wave activity, so its single Excel interval cannot be assigned uniquely. The comparison is [here](../outputs/figures/offset_alignment/210427B_C_candidate_comparison.png).

| Recording | Windows | Class 0 | Class 1 | Class 0 % | Class 1 % | Seizures represented / total |
|---|---:|---:|---:|---:|---:|---:|
| 190304A_E | 1574 | 1566 | 8 | 99.492 | 0.508 | 1 / 1 |
| 191113A_D | 1809 | 1791 | 18 | 99.005 | 0.995 | 2 / 2 |
| 210204B_C | 1226 | 1199 | 27 | 97.798 | 2.202 | 6 / 6 |
| 210208B_G | 1199 | 1152 | 47 | 96.080 | 3.920 | 4 / 4 |
| 210406A_F | 1453 | 1445 | 8 | 99.449 | 0.551 | 1 / 1 |
| 210504B_C | 3120 | 2985 | 135 | 95.673 | 4.327 | 12 / 12 |
| 210914B_A | 1204 | 1028 | 176 | 85.382 | 14.618 | 11 / 11 |
| 210928B_B | 1343 | 1286 | 57 | 95.756 | 4.244 | 2 / 2 |
| 211027B_C | 1254 | 1131 | 123 | 90.191 | 9.809 | 8 / 8 |
| 211104B_D | 1209 | 1169 | 40 | 96.691 | 3.309 | 5 / 5 |
| 220106A_A | 1220 | 1181 | 39 | 96.803 | 3.197 | 2 / 2 |
| 220121B_D | 1291 | 1241 | 50 | 96.127 | 3.873 | 3 / 3 |
| 230102B_F | 1224 | 1089 | 135 | 88.971 | 11.029 | 9 / 9 |
| 230105B_H | 3599 | 3561 | 38 | 98.944 | 1.056 | 7 / 7 |
| 230210B_D | 1172 | 1150 | 22 | 98.123 | 1.877 | 1 / 1 |
| 230224B_B | 1427 | 1381 | 46 | 96.776 | 3.224 | 3 / 3 |
| 230406B_A | 1224 | 1176 | 48 | 96.078 | 3.922 | 4 / 4 |
| 230515B_G | 1206 | 1172 | 34 | 97.181 | 2.819 | 2 / 2 |
| 230519A_E | 1215 | 1180 | 35 | 97.119 | 2.881 | 2 / 2 |
| 230718A_A | 1205 | 1138 | 67 | 94.440 | 5.560 | 7 / 7 |
| OVERALL | 30174 | 29021 | 1153 | 96.179 | 3.821 | 92 / 92 |

Feature columns comprise, per channel and averaged across channels: mean, standard deviation, variance, minimum, maximum, amplitude, RMS, energy, absolute Delta/Theta/Alpha/Beta power, and relative Delta/Theta/Alpha/Beta power. Spectral power uses a mean-removed 2 s signal, Hann window and one-sided rFFT periodogram.

Artifacts:

- `data/excel/<recording>.xlsx`: labeled samples, split automatically at 1,048,575 data rows per sheet.
- `data/processed/<recording>.parquet`: fast labeled sample copies.
- `data/processed/windows_features.parquet`: final Parts I–VI feature table.
- `outputs/window_counts.csv` and `outputs/dataset_summary.json`: machine-readable counts and schema.

No model training or train/test splitting has been performed.
