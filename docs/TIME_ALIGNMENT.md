# Time alignment investigation

Raw inputs are unchanged. No labels, conversions, dataset or model have been built.

## 1. Exact workbook cells and complete header

Selected records: `191113A_D` (Excel row 3), `210204B_C` (row 5), `210208B_G` (row 6), each with at least two populated intervals. The first three ID rows of the workbook are `190304A-E`, `191113A-D`, ` 200625A-F `; the third has no seizure pairs.

Values below retain the original XLSX XML numeric text; the Python value is shown with `repr`. Excel type is openpyxl `data_type` (`n`: numeric or empty, `s`: string); XML type is shown separately. An absent XML cell is empty, not a recorded zero. All selected cells use number format `General`.

### Excel row 1

| Cell | Column header | Python value | Excel type | XML type | Exact XML value/text |
|---|---|---|---|---|---|
| A1 | None | `None` | n | absent | `None` |
| B1 | 'Sexe' | `'Sexe'` | s | s | `'0'` |
| C1 | 'Age' | `'Age'` | s | s | `'1'` |
| D1 | 'NombreCrises' | `'NombreCrises'` | s | s | `'2'` |
| E1 | 'TypeAbs' | `'TypeAbs'` | s | s | `'3'` |
| F1 | 'DebCE1' | `'DebCE1'` | s | s | `'4'` |
| G1 | 'FinCE1' | `'FinCE1'` | s | s | `'5'` |
| H1 | 'DebCE2' | `'DebCE2'` | s | s | `'6'` |
| I1 | 'FinCE2' | `'FinCE2'` | s | s | `'7'` |
| J1 | 'DebCE3' | `'DebCE3'` | s | s | `'8'` |
| K1 | 'FinCE3' | `'FinCE3'` | s | s | `'9'` |
| L1 | 'DebCE4' | `'DebCE4'` | s | s | `'10'` |
| M1 | 'FinCE4' | `'FinCE4'` | s | s | `'11'` |
| N1 | 'DebutCE5' | `'DebutCE5'` | s | s | `'12'` |
| O1 | 'FinCE5' | `'FinCE5'` | s | s | `'13'` |
| P1 | 'DebCE6' | `'DebCE6'` | s | s | `'14'` |
| Q1 | 'FinCE6' | `'FinCE6'` | s | s | `'15'` |
| R1 | 'DebCE7' | `'DebCE7'` | s | s | `'16'` |
| S1 | 'FinCE7' | `'FinCE7'` | s | s | `'17'` |
| T1 | 'DebCE8' | `'DebCE8'` | s | s | `'18'` |
| U1 | 'FinCE8' | `'FinCE8'` | s | s | `'19'` |
| V1 | 'DebCE9' | `'DebCE9'` | s | s | `'20'` |
| W1 | 'FinCE9' | `'FinCE9'` | s | s | `'21'` |
| X1 | 'DebCE10' | `'DebCE10'` | s | s | `'22'` |
| Y1 | 'FinCE10' | `'FinCE10'` | s | s | `'23'` |
| Z1 | 'DebtCE11' | `'DebtCE11'` | s | s | `'24'` |
| AA1 | 'FinCE11' | `'FinCE11'` | s | s | `'25'` |
| AB1 | 'DebCE12' | `'DebCE12'` | s | s | `'26'` |
| AC1 | 'FinCE12' | `'FinCE12'` | s | s | `'27'` |
| AD1 | 'Intercrit1' | `'Intercrit1'` | s | s | `'28'` |
| AE1 | 'Intercrit2' | `'Intercrit2'` | s | s | `'29'` |
| AF1 | 'Intercrit3' | `'Intercrit3'` | s | s | `'30'` |
| AG1 | 'Intercrit4' | `'Intercrit4'` | s | s | `'31'` |
| AH1 | 'Intercrit5' | `'Intercrit5'` | s | s | `'32'` |
| AI1 | 'Intercrit6' | `'Intercrit6'` | s | s | `'33'` |
| AJ1 | 'Intercrit7' | `'Intercrit7'` | s | s | `'34'` |
### Excel row 3

| Cell | Column header | Python value | Excel type | XML type | Exact XML value/text |
|---|---|---|---|---|---|
| A3 | None | `'191113A-D'` | s | s | `'37'` |
| B3 | 'Sexe' | `'F'` | s | s | `'38'` |
| C3 | 'Age' | `25` | n | n (implicit) | `'25'` |
| D3 | 'NombreCrises' | `2` | n | n (implicit) | `'2'` |
| E3 | 'TypeAbs' | `None` | n | absent | `None` |
| F3 | 'DebCE1' | `0.443900462962963` | n | n (implicit) | `'0.443900462962963'` |
| G3 | 'FinCE1' | `0.444050925925926` | n | n (implicit) | `'0.444050925925926'` |
| H3 | 'DebCE2' | `0.445914351851852` | n | n (implicit) | `'0.445914351851852'` |
| I3 | 'FinCE2' | `0.445972222222222` | n | n (implicit) | `'0.445972222222222'` |
| J3 | 'DebCE3' | `None` | n | absent | `None` |
| K3 | 'FinCE3' | `None` | n | absent | `None` |
| L3 | 'DebCE4' | `None` | n | absent | `None` |
| M3 | 'FinCE4' | `None` | n | absent | `None` |
| N3 | 'DebutCE5' | `None` | n | absent | `None` |
| O3 | 'FinCE5' | `None` | n | absent | `None` |
| P3 | 'DebCE6' | `None` | n | absent | `None` |
| Q3 | 'FinCE6' | `None` | n | absent | `None` |
| R3 | 'DebCE7' | `None` | n | absent | `None` |
| S3 | 'FinCE7' | `None` | n | absent | `None` |
| T3 | 'DebCE8' | `None` | n | absent | `None` |
| U3 | 'FinCE8' | `None` | n | absent | `None` |
| V3 | 'DebCE9' | `None` | n | absent | `None` |
| W3 | 'FinCE9' | `None` | n | absent | `None` |
| X3 | 'DebCE10' | `None` | n | absent | `None` |
| Y3 | 'FinCE10' | `None` | n | absent | `None` |
| Z3 | 'DebtCE11' | `None` | n | absent | `None` |
| AA3 | 'FinCE11' | `None` | n | absent | `None` |
| AB3 | 'DebCE12' | `None` | n | absent | `None` |
| AC3 | 'FinCE12' | `None` | n | absent | `None` |
| AD3 | 'Intercrit1' | `None` | n | absent | `None` |
| AE3 | 'Intercrit2' | `None` | n | absent | `None` |
| AF3 | 'Intercrit3' | `None` | n | absent | `None` |
| AG3 | 'Intercrit4' | `None` | n | absent | `None` |
| AH3 | 'Intercrit5' | `None` | n | absent | `None` |
| AI3 | 'Intercrit6' | `None` | n | absent | `None` |
| AJ3 | 'Intercrit7' | `None` | n | absent | `None` |
### Excel row 5

| Cell | Column header | Python value | Excel type | XML type | Exact XML value/text |
|---|---|---|---|---|---|
| A5 | None | `'210204B-C'` | s | s | `'40'` |
| B5 | 'Sexe' | `'F'` | s | s | `'38'` |
| C5 | 'Age' | `5` | n | n (implicit) | `'5'` |
| D5 | 'NombreCrises' | `6` | n | n (implicit) | `'6'` |
| E5 | 'TypeAbs' | `None` | n | absent | `None` |
| F5 | 'DebCE1' | `0.431828703703704` | n | n (implicit) | `'0.431828703703704'` |
| G5 | 'FinCE1' | `0.431886574074074` | n | n (implicit) | `'0.431886574074074'` |
| H5 | 'DebCE2' | `0.433668981481481` | n | n (implicit) | `'0.433668981481481'` |
| I5 | 'FinCE2' | `0.433726851851852` | n | n (implicit) | `'0.433726851851852'` |
| J5 | 'DebCE3' | `0.43630787037037` | n | n (implicit) | `'0.43630787037037'` |
| K5 | 'FinCE3' | `0.436365740740741` | n | n (implicit) | `'0.436365740740741'` |
| L5 | 'DebCE4' | `0.437905092592593` | n | n (implicit) | `'0.437905092592593'` |
| M5 | 'FinCE4' | `0.437939814814815` | n | n (implicit) | `'0.437939814814815'` |
| N5 | 'DebutCE5' | `0.438900462962963` | n | n (implicit) | `'0.438900462962963'` |
| O5 | 'FinCE5' | `0.438946759259259` | n | n (implicit) | `'0.438946759259259'` |
| P5 | 'DebCE6' | `0.444247685185185` | n | n (implicit) | `'0.444247685185185'` |
| Q5 | 'FinCE6' | `0.444305555555556` | n | n (implicit) | `'0.444305555555556'` |
| R5 | 'DebCE7' | `None` | n | n (implicit) | `None` |
| S5 | 'FinCE7' | `None` | n | n (implicit) | `None` |
| T5 | 'DebCE8' | `None` | n | n (implicit) | `None` |
| U5 | 'FinCE8' | `None` | n | n (implicit) | `None` |
| V5 | 'DebCE9' | `None` | n | n (implicit) | `None` |
| W5 | 'FinCE9' | `None` | n | n (implicit) | `None` |
| X5 | 'DebCE10' | `None` | n | n (implicit) | `None` |
| Y5 | 'FinCE10' | `None` | n | n (implicit) | `None` |
| Z5 | 'DebtCE11' | `None` | n | n (implicit) | `None` |
| AA5 | 'FinCE11' | `None` | n | n (implicit) | `None` |
| AB5 | 'DebCE12' | `None` | n | n (implicit) | `None` |
| AC5 | 'FinCE12' | `None` | n | n (implicit) | `None` |
| AD5 | 'Intercrit1' | `0.433796296296296` | n | n (implicit) | `'0.433796296296296'` |
| AE5 | 'Intercrit2' | `0.43380787037037` | n | n (implicit) | `'0.43380787037037'` |
| AF5 | 'Intercrit3' | `0.433819444444444` | n | n (implicit) | `'0.433819444444444'` |
| AG5 | 'Intercrit4' | `0.434814814814815` | n | n (implicit) | `'0.434814814814815'` |
| AH5 | 'Intercrit5' | `0.434872685185185` | n | n (implicit) | `'0.434872685185185'` |
| AI5 | 'Intercrit6' | `0.438391203703704` | n | n (implicit) | `'0.438391203703704'` |
| AJ5 | 'Intercrit7' | `0.439398148148148` | n | n (implicit) | `'0.439398148148148'` |
### Excel row 6

| Cell | Column header | Python value | Excel type | XML type | Exact XML value/text |
|---|---|---|---|---|---|
| A6 | None | `'210208B-G'` | s | s | `'41'` |
| B6 | 'Sexe' | `'F'` | s | s | `'38'` |
| C6 | 'Age' | `14` | n | n (implicit) | `'14'` |
| D6 | 'NombreCrises' | `4` | n | n (implicit) | `'4'` |
| E6 | 'TypeAbs' | `None` | n | absent | `None` |
| F6 | 'DebCE1' | `0.471678240740741` | n | n (implicit) | `'0.471678240740741'` |
| G6 | 'FinCE1' | `0.471793981481481` | n | n (implicit) | `'0.471793981481481'` |
| H6 | 'DebCE2' | `0.4725` | n | n (implicit) | `'0.4725'` |
| I6 | 'FinCE2' | `0.472581018518519` | n | n (implicit) | `'0.472581018518519'` |
| J6 | 'DebCE3' | `0.475358796296296` | n | n (implicit) | `'0.475358796296296'` |
| K6 | 'FinCE3' | `0.475520833333333` | n | n (implicit) | `'0.475520833333333'` |
| L6 | 'DebCE4' | `0.478402777777778` | n | n (implicit) | `'0.478402777777778'` |
| M6 | 'FinCE4' | `0.478587962962963` | n | n (implicit) | `'0.478587962962963'` |
| N6 | 'DebutCE5' | `None` | n | absent | `None` |
| O6 | 'FinCE5' | `None` | n | absent | `None` |
| P6 | 'DebCE6' | `None` | n | absent | `None` |
| Q6 | 'FinCE6' | `None` | n | absent | `None` |
| R6 | 'DebCE7' | `None` | n | absent | `None` |
| S6 | 'FinCE7' | `None` | n | absent | `None` |
| T6 | 'DebCE8' | `None` | n | absent | `None` |
| U6 | 'FinCE8' | `None` | n | absent | `None` |
| V6 | 'DebCE9' | `None` | n | absent | `None` |
| W6 | 'FinCE9' | `None` | n | absent | `None` |
| X6 | 'DebCE10' | `None` | n | absent | `None` |
| Y6 | 'FinCE10' | `None` | n | absent | `None` |
| Z6 | 'DebtCE11' | `None` | n | absent | `None` |
| AA6 | 'FinCE11' | `None` | n | absent | `None` |
| AB6 | 'DebCE12' | `None` | n | absent | `None` |
| AC6 | 'FinCE12' | `None` | n | absent | `None` |
| AD6 | 'Intercrit1' | `0.470358796296296` | n | n (implicit) | `'0.470358796296296'` |
| AE6 | 'Intercrit2' | `0.47037037037037` | n | n (implicit) | `'0.47037037037037'` |
| AF6 | 'Intercrit3' | `0.471412037037037` | n | n (implicit) | `'0.471412037037037'` |
| AG6 | 'Intercrit4' | `0.471574074074074` | n | n (implicit) | `'0.471574074074074'` |
| AH6 | 'Intercrit5' | `0.473402777777778` | n | n (implicit) | `'0.473402777777778'` |
| AI6 | 'Intercrit6' | `0.473865740740741` | n | n (implicit) | `'0.473865740740741'` |
| AJ6 | 'Intercrit7' | `None` | n | absent | `None` |

### Complete header: `190304A_E_0000h.mat`

Every MCOS field and array value, all 77 decoded events, table attributes, container metadata and the complete binary internal workspace (base64) are in [alignment_header_complete.json](alignment_header_complete.json). No arrays are truncated.

Container description: `MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Fri Sep  5 10:43:29 2025`. The creation time is a MAT-file export time, not the recording acquisition start.

Raw timetable fields: `CustomProps`, `VariableCustomProps`, `versionSavedFrom`, `minCompatibleVersion`, `incompatibilityMsg`, `arrayProps`, `data`, `numDims`, `useVarNamesOrig`, `useDimNamesOrig`, `dimNames`, `dimNamesOrig`, `varNames`, `varNamesOrig`, `numRows`, `numVars`, `varDescriptions`, `varUnits`, `rowTimes`, `varContinuity`.

`arrayProps` contains `Description` (empty), `UserData` (empty), `TableCustomProperties` (empty). `CustomProps` and `VariableCustomProps` are empty. `rowTimes` is a MATLAB `duration` object with relative elapsed seconds, not a `datetime`. The nested `Duration` data are missing (NaT); the other data column is `Annotations`. There is no acquisition date, absolute start time, or time-zone field in this header.

| Onset (s) | Annotations (decoded, exact text) | Duration |
|---:|---|---|
| 0 | D��but | None |
| 1 | <IMPEDANCE> Fp1:22k Fp2:15k F7:31k F3:18k Fz:23k F4:17k F8:26k T3:36k C3:24k Cz:33k C4:90k T4:16k T5:23k P3:20k Pz:60k P4:59k T6:23k O1:26k O2:37k EMG1:250k EMG2:250k SLI:250k ECG:250k | None |
| 238 | HPN 6mn d��but | None |
| 257 | HPN - 00:00:20 | None |
| 277 | HPN - 00:00:40 | None |
| 297 | HPN - 00:01:00 | None |
| 317 | HPN - 00:01:20 | None |
| 337 | HPN - 00:01:40 | None |
| 357 | HPN - 00:02:00 | None |
| 377 | HPN - 00:02:20 | None |
| 386 | ARTEFACT | None |
| 397 | HPN - 00:02:40 | None |
| 417 | HPN - 00:03:00 | None |
| 437 | HPN - 00:03:20 | None |
| 457 | HPN - 00:03:40 | None |
| 477 | HPN - 00:04:00 | None |
| 497 | HPN - 00:04:20 | None |
| 517 | HPN - 00:04:40 | None |
| 537 | HPN - 00:05:00 | None |
| 551 | Stop : 00:05:14 | None |
| 552 | YO | None |
| 581 | YF | None |
| 690 | SLI=1Hz | None |
| 710 | SLI=5Hz | None |
| 730 | SLI=2Hz | None |
| 750 | SLI=10Hz | None |
| 752 | YO | None |
| 757 | YF | None |
| 770 | SLI=12Hz | None |
| 790 | SLI=15Hz | None |
| 811 | SLI=18Hz | None |
| 831 | SLI=23Hz | None |
| 832 | YO | None |
| 839 | YF | None |
| 851 | SLI=28Hz | None |
| 871 | SLI=31Hz | None |
| 872 | YO | None |
| 878 | YF | None |
| 891 | SLI=36Hz | None |
| 911 | SLI=45Hz | None |
| 913 | YO | None |
| 918 | YF | None |
| 931 | SLI=50Hz | None |
| 970 | HPN 6mn d��but | None |
| 972 | BOUGE | None |
| 989 | HPN - 00:00:20 | None |
| 1009 | HPN - 00:00:40 | None |
| 1029 | HPN - 00:01:00 | None |
| 1049 | HPN - 00:01:20 | None |
| 1058 | BOUGE | None |
| 1069 | HPN - 00:01:40 | None |
| 1089 | HPN - 00:02:00 | None |
| 1109 | HPN - 00:02:20 | None |
| 1129 | HPN - 00:02:40 | None |
| 1149 | HPN - 00:03:00 | None |
| 1169 | HPN - 00:03:20 | None |
| 1189 | HPN - 00:03:40 | None |
| 1209 | HPN - 00:04:00 | None |
| 1223 | ARTEFACT | None |
| 1229 | HPN - 00:04:20 | None |
| 1249 | HPN - 00:04:40 | None |
| 1269 | HPN - 00:05:00 | None |
| 1269 | Stop : 00:05:00 | None |
| 1271 | YO | None |
| 1278 | YF | None |
| 1280 | a | None |
| 1292 | crise | None |
| 1301 | b | None |
| 1305 | Annotation | None |
| 1314 | YF | None |
| 1360 | ARTEFACT | None |
| 1386 | YF | None |
| 1398 | YF | None |
| 1410 | ARTEFACT | None |
| 1434 | ARTEFACT | None |
| 1544 | BOUGE | None |
| 1555 | ARTEFACT | None |

## 2. Elapsed-time hypothesis, all recordings

Test: interpret each numeric annotation as a duration in days, so `elapsed_seconds = Excel_value × 86400`, then require every interval to lie in `[0, recording_duration]`. No offset is subtracted. Max time includes every populated seizure endpoint and both versions of a duplicate; `Intercrit*` values are not seizure endpoints.

| Recording | Duration (s) | Maximum annotated elapsed time (s) | Difference max − duration (s) | Consistent? |
|---|---:|---:|---:|---|
| 190304A_E | 1575 | 38113.000000 | 36538.000000 | no |
| 191113A_D | 1810 | 38532.000000 | 36722.000000 | no |
| 200625A_F | 1788 | — | — | no annotations |
| 210204B_C | 1227 | 38388.000000 | 37161.000000 | no |
| 210208B_G | 1200 | 41350.000000 | 40150.000000 | no |
| 210406A_F | 1454 | 54958.000000 | 53504.000000 | no |
| 210427B_C | 1246 | 32679.000000 | 31433.000000 | no |
| 210504B_C | 3121 | 48717.000000 | 45596.000000 | no |
| 210914B_A | 1205 | 31085.000000 | 29880.000000 | no |
| 210928B_B | 1344 | 34190.000000 | 32846.000000 | no |
| 211027B_C | 1255 | 38132.000000 | 36877.000000 | no |
| 211104B_D | 1210 | 40547.000000 | 39337.000000 | no |
| 220106A_A | 1221 | 30805.000000 | 29584.000000 | no |
| 220121B_D | 1292 | 37257.000000 | 35965.000000 | no |
| 230102B_F | 1225 | 44713.000000 | 43488.000000 | no |
| 230105B_H | 3600 | 55185.000000 | 51585.000000 | no |
| 230210B_D | 1173 | 36776.000000 | 35603.000000 | no |
| 230224B_B | 1428 | 33611.000000 | 32183.000000 | no |
| 230406B_A | 1225 | 31101.000000 | 29876.000000 | no |
| 230515B_G | 1207 | 45044.000000 | 43837.000000 | no |
| 230519A_E | 1216 | 41908.000000 | 40692.000000 | no |
| 230718A_A | 1206 | 30584.000000 | 29378.000000 | no |

**0/22 recordings are positively consistent; 21/22 are inconsistent; 1/22 (`200625A_F`) cannot be tested because it has no annotations.** All 21 annotated recordings have every populated seizure interval starting after the recording has already ended. Thus even the first annotated seizures cannot be displayed at their proposed elapsed coordinates.

## 3. Visual/Delta checks under that hypothesis

Three figures include the real EEG Delta-power timeline and two 20 s views centered on the first two annotated seizure midpoints. Orange shading denotes proposed annotations, not validated seizures. The 20 s views contain no signal because their coordinates are outside the acquisition; this is explicitly marked rather than clipping timestamps or fabricating samples.

Delta power is computed on all EEG channels and averaged across channels: 2 s Hann windows, 1 s hop, mean removal per window, one-sided PSD density, sum of bins satisfying `0.5 ≤ f < 4 Hz` times the frequency-bin spacing (0.5 Hz). Units are signal amplitude², since physical units are unavailable. At this resolution 3 Hz is a Fourier bin. Supplementary first-20-s EEG/Delta plots show actual data; they have no seizure overlays because no proposed interval intersects this segment.

**There are no samples or Delta windows inside any proposed interval. Inside/outside Delta ratios and seizure morphology at those times are undefined. The elapsed interpretation is rejected by coverage, not by interpreting an empty panel as absence of spike-and-wave activity.**

- `191113A_D`: proposed intervals 38353–38366 s, 38527–38532 s; recording ends at 1810 s. [Alignment plot](../outputs/figures/alignment/191113A_D_elapsed_alignment.png), [actual first 20 s](../outputs/figures/alignment/191113A_D_actual_first20s.png).
- `210204B_C`: proposed intervals 37310–37315 s, 37469–37474 s; recording ends at 1227 s. [Alignment plot](../outputs/figures/alignment/210204B_C_elapsed_alignment.png), [actual first 20 s](../outputs/figures/alignment/210204B_C_actual_first20s.png).
- `210208B_G`: proposed intervals 40753–40763 s, 40824–40831 s; recording ends at 1200 s. [Alignment plot](../outputs/figures/alignment/210208B_G_elapsed_alignment.png), [actual first 20 s](../outputs/figures/alignment/210208B_G_actual_first20s.png).

## 4. Annotation problems, without corrections

### Exact explanation of 95 pairs versus declared 90

`210914B_A`, Excel row 10, declares 6 but contains **11** pairs. The **five additional pairs are CE7–CE11** (ordinal positions beyond the declared six); this does not establish that those five are invalid.

| Pair | Start cell and exact value | End cell and exact value | Time interpreted as hh:mm:ss | Duration (s) |
|---|---|---|---|---:|
| CE7 | R10: `0.350902777777778` | S10: `0.350960648148148` | 08:25:18 → 08:25:23 | 5 |
| CE8 | T10: `0.351909722222222` | U10: `0.352268518518519` | 08:26:45 → 08:27:16 | 31 |
| CE9 | V10: `0.35369212962963` | W10: `0.354259259259259` | 08:29:19 → 08:30:08 | 49 |
| CE10 | X10: `0.355555555555556` | Y10: `0.355648148148148` | 08:32:00 → 08:32:08 | 8 |
| CE11 | Z10: `0.359641203703704` | AA10: `0.359780092592593` | 08:37:53 → 08:38:05 | 12 |

### Duplicate versions: no rows removed

The earlier “93 after removing the duplicate” was a **hypothetical count**, not a deletion. No duplicates have been removed from any input. `230515B_G` occurs in Excel rows 15 and 22; both declare 2 seizures. Keeping one row would remove 2 pairs (95 → 93), and reduce the sum of declared counts from 90 to 88. These are conflicting versions, not exact duplicates.

| Excel row | Age | Pair | Start raw value | End raw value | Time | Duration (s) |
|---:|---:|---|---|---|---|---:|
| 15 | 9 | CE1 | `0.520208333333333` | `0.52037037037037` | 12:29:06 → 12:29:20 | 14 |
| 15 | 9 | CE2 | `0.520972222222222` | `0.521342592592593` | 12:30:12 → 12:30:44 | 32 |
| 22 | 8 | CE1 | `0.520208333333333` | `0.520358796296296` | 12:29:06 → 12:29:19 | 13 |
| 22 | 8 | CE2 | `0.521099537037037` | `0.521342592592593` | 12:30:23 → 12:30:44 | 21 |

### Interval longer than its entire acquisition

`210406A_F`, Excel row 7, CE1: `F7 = 0.51099537037037`, `G7 = 0.636087962962963`. Converted times: **12:15:50 → 15:15:58**; duration **10 808 s**; recording duration **1454 s**. This cannot fit even after subtracting a constant clock offset. No correction has been applied.

Additional issues: `200625A_F` has missing count and no intervals (not confirmed seizure-free); isolated totals 90 at D25 and 180 at D1048574; blank ID header; spaces around `200625A-F`; inconsistent beginning-column spellings `DebutCE5` and `DebtCE11`; physical units and absolute acquisition start unavailable.

## 5. Channels per recording and intersection

Exact channel names and original order:

- **190304A_E (23 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT3`, `EEGC3`, `EEGCz`, `EEGC4`, `EEGT4`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO1`, `EEGO2`, `EMG1`, `EMG2`, `SLI`, `ECG`
- **191113A_D (23 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT3`, `EEGC3`, `EEGCz`, `EEGC4`, `EEGT4`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO1`, `EEGO2`, `EMG1`, `EMG2`, `SLI`, `ECG`
- **200625A_F (12 channels):** `EEGFp1`, `EEGFp2`, `EEGC3`, `EEGC4`, `EEGO1`, `EEGO2`, `EEGT3`, `EEGT4`, `EMG1`, `EMG2`, `SLI`, `ECG`
- **210204B_C (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **210208B_G (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **210406A_F (23 channels):** `EEGFp2`, `EEGF8`, `EEGT4`, `EEGT6`, `EEGF4`, `EEGC4`, `EEGP4`, `EEGO2`, `EEGFz`, `EEGCz`, `EEGPz`, `EEGFp1`, `EEGF3`, `EEGC3`, `EEGP3`, `EEGO1`, `EEGF7`, `EEGT3`, `EEGT5`, `ECG`, `EMG_1_`, `SLI`, `EMG_2`
- **210427B_C (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **210504B_C (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **210914B_A (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **210928B_B (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **211027B_C (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **211104B_D (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **220106A_A (23 channels):** `EEGFp2`, `EEGF8`, `EEGT4`, `EEGT6`, `EEGF4`, `EEGC4`, `EEGP4`, `EEGO2`, `EEGFz`, `EEGCz`, `EEGPz`, `EEGFp1`, `EEGF3`, `EEGC3`, `EEGP3`, `EEGO1`, `EEGF7`, `EEGT3`, `EEGT5`, `ECG`, `EMG_1_`, `SLI`, `EMG_2`
- **220121B_D (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **230102B_F (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **230105B_H (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **230210B_D (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **230224B_B (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **230406B_A (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **230515B_G (21 channels):** `EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT4`, `EEGC4`, `EEGCz`, `EEGC3`, `EEGT3`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO2`, `EEGO1`, `SLI`, `ECG`
- **230519A_E (23 channels):** `EEGFp2`, `EEGF8`, `EEGT4`, `EEGT6`, `EEGF4`, `EEGC4`, `EEGP4`, `EEGO2`, `EEGFz`, `EEGCz`, `EEGPz`, `EEGFp1`, `EEGF3`, `EEGC3`, `EEGP3`, `EEGO1`, `EEGF7`, `EEGT3`, `EEGT5`, `ECG`, `EMG_1_`, `SLI`, `EMG_2`
- **230718A_A (23 channels):** `EEGFp2`, `EEGF8`, `EEGT4`, `EEGT6`, `EEGF4`, `EEGC4`, `EEGP4`, `EEGO2`, `EEGFz`, `EEGCz`, `EEGPz`, `EEGFp1`, `EEGF3`, `EEGC3`, `EEGP3`, `EEGO1`, `EEGF7`, `EEGT3`, `EEGT5`, `ECG`, `EMG_1_`, `SLI`, `EMG_2`

**Common to all 22 recordings (10 channels):** `ECG`, `EEGC3`, `EEGC4`, `EEGFp1`, `EEGFp2`, `EEGO1`, `EEGO2`, `EEGT3`, `EEGT4`, `SLI`.

**EEG-only intersection (8 channels):** `EEGC3`, `EEGC4`, `EEGFp1`, `EEGFp2`, `EEGO1`, `EEGO2`, `EEGT3`, `EEGT4`.

The common set includes auxiliary `ECG` and `SLI`; EMG names differ or are absent. This list is an inventory, not a channel-selection decision.

## Conclusion and checkpoint

The direct elapsed-time hypothesis fails for every annotated acquisition. It remains plausible that the numbers are clock times, but this test does not establish the missing acquisition offset. The next reliable input would be original recording start metadata or annotation-author clarification. No alignment is accepted, no annotation is fixed, and no dataset has been built. Awaiting user review before further work.
