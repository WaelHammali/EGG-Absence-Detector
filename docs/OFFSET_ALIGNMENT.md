# Clock offset alignment

No raw file was modified and no feature dataset was built. `offset_clock` means Excel clock seconds minus elapsed signal seconds; it is therefore the inferred clock time at elapsed 0 s.

## D-file timetable metadata

All 22 d-files have the same relevant structure. The raw MCOS timetable fields are `CustomProps`, `VariableCustomProps`, `versionSavedFrom`, `minCompatibleVersion`, `incompatibilityMsg`, `arrayProps`, `data`, `numDims`, `useVarNamesOrig`, `useDimNamesOrig`, `dimNames`, `dimNamesOrig`, `varNames`, `varNamesOrig`, `numRows`, `numVars`, `varDescriptions`, `varUnits`, `rowTimes`, and `varContinuity`.

`rowTimes` is a MATLAB `duration`, with `millis = [0, 1000, 2000, …]` and format `s`. It is not an absolute `datetime`. No d-file has a `StartTime`, `SampleRate`, or `TimeStep` field. `Description`, `UserData`, table custom properties, variable descriptions and variable units are empty. The full per-file audit is in [dfile_time_metadata.json](dfile_time_metadata.json). The 256 Hz rate comes from 256 samples stored in each one-second channel block.

## Data-driven method and confidence

For each 2 s Hann window at a 0.25 s hop, the seizure score is mean relative 2.5–4 Hz power across Fp1, Fp2, C3, C4, T3, T4, O1 and O2; the denominator is 0.5–30 Hz power. For every feasible offset at 0.25 s resolution, the objective is mean window score inside all shifted seizure intervals minus mean score outside.

The second peak must be at least 5 s from the best peak. Confidence is best score divided by second-best score; ratios below 1.2, or a non-positive best score, are flagged low. This ratio measures offset uniqueness, not clinical certainty.

| Recording | Offset (s) | Start clock | Best score | Second peak: offset / score | Ratio | Header offset | Agreement ≤2 s | Low confidence | Annotation version |
|---|---:|---|---:|---|---:|---:|---|---|---|
| 190304A_E | 36815.00 | 10:13:35 | 0.141320 | 37554.75 / 0.113715 | 1.243 | 36819.00 | yes | no | source row 2 |
| 191113A_D | 37098.25 | 10:18:18.250 | 0.205257 | 37314.00 / 0.120643 | 1.701 | 37101.50 | yes | no | source row 3 |
| 210204B_C | 37166.50 | 10:19:26.500 | 0.363211 | 37252.50 / 0.060683 | 5.985 | — | NA | no | source row 5 |
| 210208B_G | 40256.25 | 11:10:56.250 | 0.226609 | 40567.25 / 0.098038 | 2.311 | 40255.00 | yes | no | source row 6 |
| 210406A_F | 43516.25 | 12:05:16.250 | 0.430939 | 42725.00 / 0.066429 | 6.487 | 43518.00 | yes | no | candidate correction 12:15:58 |
| 210427B_C | 32333.00 | 08:58:53 | 0.342679 | 31629.50 / 0.310537 | 1.104 | 31839.00 | no | yes | source row 8 |
| 210504B_C | 46257.75 | 12:50:57.750 | 0.152127 | 45838.50 / 0.100796 | 1.509 | 45920.50 | yes | no | source row 9 |
| 210914B_A | 30046.75 | 08:20:46.750 | 0.107837 | 30033.75 / 0.050127 | 2.151 | — | NA | no | source row 10 |
| 210928B_B | 32875.00 | 09:07:55 | 0.344282 | 32883.00 / 0.290518 | 1.185 | 32886.50 | yes | yes | source row 11 |
| 211027B_C | 36964.75 | 10:16:04.750 | 0.413519 | 36945.00 / 0.215017 | 1.923 | — | NA | no | source row 12 |
| 211104B_D | 39642.50 | 11:00:42.500 | 0.266461 | 39470.50 / 0.068891 | 3.868 | — | NA | no | source row 13 |
| 220106A_A | 29637.00 | 08:13:57 | 0.170782 | 29627.50 / 0.132170 | 1.292 | 29638.50 | yes | no | source row 16 |
| 220121B_D | 36641.50 | 10:10:41.500 | 0.150103 | 36411.50 / 0.073611 | 2.039 | 36641.00 | yes | no | source row 17 |
| 230102B_F | 43492.25 | 12:04:52.250 | 0.316172 | 43531.50 / 0.029594 | 10.684 | — | NA | no | source row 18 |
| 230105B_H | 52290.50 | 14:31:30.500 | 0.238520 | 52265.50 / 0.151931 | 1.570 | — | NA | no | source row 19 |
| 230210B_D | 36227.75 | 10:03:47.750 | 0.278565 | 36373.50 / 0.195182 | 1.427 | — | NA | no | source row 20 |
| 230224B_B | 32592.25 | 09:03:12.250 | 0.265211 | 32755.00 / 0.061659 | 4.301 | — | NA | no | source row 21 |
| 230406B_A | 30024.00 | 08:20:24 | 0.189095 | 30017.00 / 0.149952 | 1.261 | 29911.50 | yes | no | source row 14 |
| 230515B_G | 44593.75 | 12:23:13.750 | 0.356000 | 44669.50 / 0.180087 | 1.977 | 44596.50 | yes | no | source row 22 |
| 230519A_E | 40773.75 | 11:19:33.750 | 0.175590 | 40852.75 / 0.099129 | 1.771 | 40776.00 | yes | no | source row 23 |
| 230718A_A | 29473.50 | 08:11:13.500 | 0.335014 | 29384.50 / 0.040996 | 8.172 | 29475.50 | yes | no | source row 24 |

Seizure-like header anchors produced candidate offsets for **13** recordings; in **12**, every extracted marker falls inside a shifted Excel interval or within 2 s of a boundary under the independent data optimum. Header extraction includes `absence`, `crise`, `pointe-onde`/`PO`, and centers of explicit `a`/`b` pairs; it excludes unrelated `HPN fin` and export-end events. The displayed header offset is an independent midpoint-match representative; generic event labels can occur anywhere within a seizure, so compatibility is tested against interval ranges rather than requiring that representative point to equal the data optimum. Every header candidate is constrained to the feasible offset interval, which places every Excel seizure inside the recording. Full marker lists, feasible ranges and residuals are in [offset_alignment_details.json](offset_alignment_details.json).

Low-confidence recordings under the stated ratio rule: `210427B_C`, `210928B_B`.

## Annotation decisions

- 210406A_F: suspected 12:15:58 correction retained after positive signal confirmation.
- 210914B_A: 11/11 intervals have above-background 2.5–4 Hz power at the shared alignment.
- 230515B_G: selected source row 22 by the larger constrained data score.
- 200625A_F is absent from both derived CSVs because it is unknown/unannotated, not a confirmed class-0 recording.
- `18` populated `Intercrit*` point events are stored as class 0 in [intercritical_events.csv](../data/interim/intercritical_events.csv). They were not converted to seizure intervals.
- ECG, SLI and EMG are excluded. The fixed EEG order is Fp1, Fp2, C3, C4, T3, T4, O1, O2.

The cleaned derived file contains **93 seizure intervals**. It is [annotations_clean.csv](../data/interim/annotations_clean.csv); inferred starts are in [offsets.csv](../data/interim/offsets.csv).

## Plots

There is one figure per annotated recording in [outputs/figures/offset_alignment](../outputs/figures/offset_alignment). The first panel shows the full objective curve and its two reported peaks. Every following panel shows Fp1, C3 and O1 for one aligned interval with ±10 s context; channels are independently displayed on a shared robust scale and the proposed seizure interval is shaded.

These are alignment diagnostics. A high 3 Hz score supports an offset, but visual morphology and ambiguous annotations still require review before dataset construction.
