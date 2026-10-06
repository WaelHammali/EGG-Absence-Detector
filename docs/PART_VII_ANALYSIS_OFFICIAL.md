# Partie VII — Analyse exploratoire (official, signal filtré passe-bande 0,5–40 Hz)

Analyse de 31,419 fenêtres : 30,175 de classe 0 et 1,244 de classe 1. Aucun modèle n’est entraîné dans cette partie.

## Q10 — Comparaison entre les classes

Les boxplots utilisent une échelle logarithmique car l’énergie et les puissances spectrales sont très asymétriques. Les valeurs aberrantes ne sont pas affichées dans les boîtes, mais elles restent présentes dans les calculs.

| Caractéristique | Médiane classe 0 | Médiane classe 1 | Cohen d | AUC séparatrice (globale) | AUC séparatrice (médiane intra-enregistrement) | Plus élevée en classe |
|---|---:|---:|---:|---:|---:|---:|
| avg_energy | 303343 | 9.10238e+06 | 0.131 | 0.926 | 0.974 | 1 |
| avg_delta_power | 298.996 | 11748.8 | 0.088 | 0.916 | 0.959 | 1 |
| avg_theta_power | 91.8935 | 1469.12 | 0.134 | 0.925 | 0.949 | 1 |
| avg_std | 22.8209 | 124.732 | 1.196 | 0.928 | 0.981 | 1 |
| avg_rms | 22.9128 | 124.893 | 1.191 | 0.928 | 0.981 | 1 |
| avg_amplitude | 121.184 | 592.949 | 1.144 | 0.933 | 0.978 | 1 |
| avg_delta_relative_power | 0.637158 | 0.707839 | 0.283 | 0.585 | 0.603 | 1 |
| avg_theta_relative_power | 0.183216 | 0.114708 | -0.530 | 0.667 | 0.718 | 0 |

Le graphique correspondant est [class_comparisons.png](../outputs/figures/official/part_vii/class_comparisons.png). Une AUC séparatrice proche de 1 indique une bonne séparation univariée; 0,5 indique une absence de séparation. L’AUC globale mélange toutes les fenêtres; l’AUC intra-enregistrement est calculée dans chaque enregistrement puis résumée par la médiane, ce qui neutralise les différences de niveau entre enregistrements. « Plus élevée en classe » compare les médianes.

Les distributions sont très asymétriques : un Cohen d faible (calculé sur les moyennes et écarts-types) peut coexister avec une AUC élevée (fondée sur les rangs). L’AUC est donc la mesure de référence ici.

### Caractéristiques les plus discriminantes (AUC globale)

| Rang | Caractéristique | AUC globale | AUC intra-enregistrement | Cohen d | Direction (médianes) |
|---:|---|---:|---:|---:|---|
| 1 | C3_amplitude | 0.938 | 0.978 | 1.316 | crise > hors crise |
| 2 | C4_amplitude | 0.936 | 0.979 | 1.318 | crise > hors crise |
| 3 | Cz_amplitude | 0.935 | 0.980 | 1.420 | crise > hors crise |
| 4 | Cz_variance | 0.934 | 0.977 | 0.238 | crise > hors crise |
| 5 | Cz_std | 0.934 | 0.977 | 1.641 | crise > hors crise |
| 6 | Cz_energy | 0.934 | 0.977 | 0.237 | crise > hors crise |
| 7 | Cz_rms | 0.934 | 0.977 | 1.634 | crise > hors crise |
| 8 | Cz_min | 0.934 | 0.977 | -1.598 | crise < hors crise |
| 9 | C4_variance | 0.934 | 0.977 | 0.186 | crise > hors crise |
| 10 | C4_std | 0.934 | 0.977 | 1.473 | crise > hors crise |
| 11 | C3_std | 0.934 | 0.973 | 1.469 | crise > hors crise |
| 12 | C3_variance | 0.934 | 0.973 | 0.190 | crise > hors crise |
| 13 | C4_energy | 0.934 | 0.977 | 0.185 | crise > hors crise |
| 14 | C4_rms | 0.934 | 0.977 | 1.467 | crise > hors crise |
| 15 | C3_rms | 0.934 | 0.973 | 1.463 | crise > hors crise |

### Caractéristiques les plus discriminantes à l’intérieur des enregistrements

| Rang | Caractéristique | AUC intra-enregistrement | AUC globale |
|---:|---|---:|---:|
| 1 | T5_amplitude | 0.983 | 0.931 |
| 2 | Pz_std | 0.982 | 0.928 |
| 3 | Pz_variance | 0.982 | 0.928 |
| 4 | Pz_rms | 0.982 | 0.928 |
| 5 | Pz_energy | 0.982 | 0.928 |
| 6 | avg_rms | 0.981 | 0.928 |
| 7 | F7_amplitude | 0.981 | 0.923 |
| 8 | avg_std | 0.981 | 0.928 |
| 9 | T3_min | 0.981 | 0.928 |
| 10 | T3_amplitude | 0.980 | 0.931 |

`mean` est quasi nulle après filtrage : son AUC reflète du bruit résiduel, pas une information utile. Elle est conservée car demandée par l’énoncé. L’effet du filtre sur les variables sensibles à l’offset est détaillé dans la dernière section.

Ces résultats mesurent chaque variable séparément et ne constituent pas encore un modèle prédictif.

## Q11 — Corrélations et redondance

Deux matrices sont calculées. Pearson mesure une relation linéaire et est dominée par les valeurs extrêmes (artefacts) de ces distributions asymétriques. Spearman mesure une relation monotone sur les rangs : c’est la mesure retenue pour juger la redondance, car `variance = std²` ou `energy ∝ rms²` sont des relations parfaitement monotones mais non linéaires.

- Paires avec |r de Pearson| ≥ 0,95 : 1033 sur 51040.
- Paires avec |ρ de Spearman| ≥ 0,95 : 445.

Exemples Pearson vs Spearman sur les moyennes inter-canaux :

| Paire | Pearson r | Spearman ρ |
|---|---:|---:|
| avg_std / avg_variance | 0.837 | 0.993 |
| avg_rms / avg_energy | 0.837 | 0.993 |
| avg_std / avg_amplitude | 0.969 | 0.986 |
| avg_theta_power / avg_alpha_power | 0.794 | 0.511 |

### Caractéristiques fortement corrélées (moyennes inter-canaux, |ρ| ≥ 0,90)

| Caractéristique 1 | Caractéristique 2 | Spearman ρ |
|---|---|---:|
| avg_std | avg_rms | 1.000 |
| avg_variance | avg_energy | 1.000 |
| avg_variance | avg_rms | 0.993 |
| avg_std | avg_variance | 0.993 |
| avg_rms | avg_energy | 0.993 |
| avg_std | avg_energy | 0.993 |
| avg_std | avg_amplitude | 0.986 |
| avg_amplitude | avg_rms | 0.986 |
| avg_min | avg_amplitude | -0.978 |
| avg_variance | avg_amplitude | 0.977 |
| avg_amplitude | avg_energy | 0.976 |
| avg_max | avg_amplitude | 0.974 |
| avg_std | avg_min | -0.965 |
| avg_min | avg_rms | -0.965 |
| avg_std | avg_max | 0.963 |
| avg_max | avg_rms | 0.962 |
| avg_variance | avg_min | -0.956 |
| avg_min | avg_energy | -0.956 |
| avg_variance | avg_max | 0.954 |
| avg_max | avg_energy | 0.953 |
| avg_min | avg_max | -0.909 |
| avg_energy | avg_delta_power | 0.905 |
| avg_variance | avg_delta_power | 0.905 |

### Groupes redondants (|ρ| ≥ 0,95, toutes caractéristiques)

Chaque groupe est une composante connexe : ses membres sont reliés par des chaînes de paires avec |ρ| ≥ 0,95. Un seul représentant par groupe suffit avant les modèles sensibles à la colinéarité.

1. (55) `C3_amplitude`, `C3_energy`, `C3_max`, `C3_min`, `C3_rms`, `C3_std`, `C3_variance`, `C4_amplitude`, `C4_energy`, `C4_max`, `C4_min`, `C4_rms`, `C4_std`, `C4_variance`, `Cz_amplitude`, `Cz_energy`, `Cz_max`, `Cz_min`, `Cz_rms`, `Cz_std`, `Cz_variance`, `T3_amplitude`, `T3_energy`, `T3_max`, `T3_min`, `T3_rms`, `T3_std`, `T3_variance`, `T4_amplitude`, `T4_energy`, `T4_max`, `T4_min`, `T4_rms`, `T4_std`, `T4_variance`, `T5_amplitude`, `T5_energy`, `T5_max`, `T5_min`, `T5_rms`, `T5_std`, `T5_variance`, `T6_amplitude`, `T6_energy`, `T6_min`, `T6_rms`, `T6_std`, `T6_variance`, `avg_amplitude`, `avg_energy`, `avg_max`, `avg_min`, `avg_rms`, `avg_std`, `avg_variance` — représentant suggéré : `T5_amplitude` (meilleure AUC intra-enregistrement 0.983).
2. (14) `P3_amplitude`, `P3_energy`, `P3_max`, `P3_min`, `P3_rms`, `P3_std`, `P3_variance`, `P4_amplitude`, `P4_energy`, `P4_max`, `P4_min`, `P4_rms`, `P4_std`, `P4_variance` — représentant suggéré : `P3_std` (meilleure AUC intra-enregistrement 0.979).
3. (7) `Fp1_amplitude`, `Fp1_energy`, `Fp1_max`, `Fp1_min`, `Fp1_rms`, `Fp1_std`, `Fp1_variance` — représentant suggéré : `Fp1_amplitude` (meilleure AUC intra-enregistrement 0.949).
4. (7) `Fp2_amplitude`, `Fp2_energy`, `Fp2_max`, `Fp2_min`, `Fp2_rms`, `Fp2_std`, `Fp2_variance` — représentant suggéré : `Fp2_amplitude` (meilleure AUC intra-enregistrement 0.932).
5. (7) `F7_amplitude`, `F7_energy`, `F7_max`, `F7_min`, `F7_rms`, `F7_std`, `F7_variance` — représentant suggéré : `F7_amplitude` (meilleure AUC intra-enregistrement 0.981).
6. (7) `F3_amplitude`, `F3_energy`, `F3_max`, `F3_min`, `F3_rms`, `F3_std`, `F3_variance` — représentant suggéré : `F3_std` (meilleure AUC intra-enregistrement 0.975).
7. (7) `Fz_amplitude`, `Fz_energy`, `Fz_max`, `Fz_min`, `Fz_rms`, `Fz_std`, `Fz_variance` — représentant suggéré : `Fz_std` (meilleure AUC intra-enregistrement 0.974).
8. (7) `F4_amplitude`, `F4_energy`, `F4_max`, `F4_min`, `F4_rms`, `F4_std`, `F4_variance` — représentant suggéré : `F4_std` (meilleure AUC intra-enregistrement 0.969).
9. (7) `F8_amplitude`, `F8_energy`, `F8_max`, `F8_min`, `F8_rms`, `F8_std`, `F8_variance` — représentant suggéré : `F8_std` (meilleure AUC intra-enregistrement 0.968).
10. (7) `Pz_amplitude`, `Pz_energy`, `Pz_max`, `Pz_min`, `Pz_rms`, `Pz_std`, `Pz_variance` — représentant suggéré : `Pz_std` (meilleure AUC intra-enregistrement 0.982).
11. (7) `O1_amplitude`, `O1_energy`, `O1_max`, `O1_min`, `O1_rms`, `O1_std`, `O1_variance` — représentant suggéré : `O1_std` (meilleure AUC intra-enregistrement 0.975).
12. (7) `O2_amplitude`, `O2_energy`, `O2_max`, `O2_min`, `O2_rms`, `O2_std`, `O2_variance` — représentant suggéré : `O2_energy` (meilleure AUC intra-enregistrement 0.966).
13. (3) `P3_delta_power`, `P4_delta_power`, `Pz_delta_power` — représentant suggéré : `P3_delta_power` (meilleure AUC intra-enregistrement 0.963).
14. (2) `C4_delta_power`, `T4_delta_power` — représentant suggéré : `C4_delta_power` (meilleure AUC intra-enregistrement 0.966).

Figures : corrélations moyennes [Spearman](../outputs/figures/official/part_vii/correlation_average_features_spearman.png) et [Pearson](../outputs/figures/official/part_vii/correlation_average_features.png); matrice complète [Spearman](../outputs/figures/official/part_vii/correlation_all_features_spearman.png) et [Pearson](../outputs/figures/official/part_vii/correlation_all_features.png). Tableaux : `outputs/feature_class_comparison_official.csv`, `outputs/high_correlations_official.csv` (Pearson), `outputs/high_correlations_spearman_official.csv`, `outputs/feature_correlations_official.csv` et `outputs/feature_correlations_spearman_official.csv`.

## Effet du prétraitement (comparaison avec les données non filtrées)

Le filtre passe-bande 0,5–40 Hz retire l’offset continu propre à chaque enregistrement (< 0,5 Hz) et le bruit secteur à 50 Hz, très présent dans le signal brut (pic à 50 Hz 300 à 35 000 fois au-dessus du niveau 30–45 Hz selon les enregistrements). Illustration : [avant/après filtrage](../outputs/figures/official/preprocessing/211104B_D_seizure3_filtering.png). Le rapport non filtré complet reste disponible : [PART_VII_ANALYSIS_OFFICIAL_RAW.md](PART_VII_ANALYSIS_OFFICIAL_RAW.md).

### Écart AUC globale / intra-enregistrement des variables sensibles à l’offset

| Caractéristique | Brut : globale | Brut : intra | Brut : écart | Filtré : globale | Filtré : intra | Filtré : écart |
|---|---:|---:|---:|---:|---:|---:|
| avg_mean | 0.565 | 0.561 | -0.004 | 0.511 | 0.501 | -0.011 |
| avg_min | 0.738 | 0.956 | +0.218 | 0.933 | 0.978 | +0.045 |
| avg_max | 0.567 | 0.960 | +0.393 | 0.929 | 0.970 | +0.041 |
| avg_rms | 0.615 | 0.837 | +0.222 | 0.928 | 0.981 | +0.053 |
| avg_energy | 0.595 | 0.791 | +0.196 | 0.926 | 0.974 | +0.047 |

Écart moyen sur ces cinq variables : +0.205 avant filtrage, +0.035 après. Après filtrage, la moyenne de chaque fenêtre est proche de 0 : `mean` n’apporte plus d’information (elle est conservée car demandée par l’énoncé), `rms` devient presque identique à `std`, et `min`/`max` mesurent l’amplitude des oscillations au lieu de l’offset.

### Caractéristiques les plus discriminantes avant / après filtrage (AUC globale)

| Rang | Brut | AUC | Filtré | AUC |
|---:|---|---:|---|---:|
| 1 | T5_alpha_power | 0.931 | C3_amplitude | 0.938 |
| 2 | Pz_beta_power | 0.930 | C4_amplitude | 0.936 |
| 3 | P3_beta_power | 0.930 | Cz_amplitude | 0.935 |
| 4 | P4_beta_power | 0.930 | Cz_variance | 0.934 |
| 5 | avg_beta_power | 0.930 | Cz_std | 0.934 |
| 6 | C3_beta_power | 0.929 | Cz_energy | 0.934 |
| 7 | F7_alpha_power | 0.929 | Cz_rms | 0.934 |
| 8 | avg_alpha_power | 0.928 | Cz_min | 0.934 |
| 9 | Cz_beta_power | 0.928 | C4_variance | 0.934 |
| 10 | P3_alpha_power | 0.928 | C4_std | 0.934 |

### Groupes redondants avant / après filtrage (|ρ de Spearman| ≥ 0,95)

Brut : 171 paires, 36 groupes (tailles 27, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2). Filtré : 445 paires, 14 groupes (tailles 55, 14, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 2).

| Type de paire | Apparues après filtrage | Disparues après filtrage |
|---|---:|---:|
| amplitude – amplitude (canal vs moyenne) | 1 | 3 |
| amplitude – amplitude (canaux différents) | 1 | 5 |
| amplitude – energy (même canal) | 20 | 0 |
| amplitude – max (même canal) | 19 | 0 |
| amplitude – min (même canal) | 20 | 0 |
| amplitude – rms (même canal) | 20 | 0 |
| amplitude – std (canal vs moyenne) | 1 | 1 |
| amplitude – std (canaux différents) | 0 | 2 |
| amplitude – variance (canal vs moyenne) | 1 | 1 |
| amplitude – variance (canaux différents) | 0 | 2 |
| amplitude – variance (même canal) | 1 | 0 |
| delta_power – delta_power (canaux différents) | 1 | 0 |
| energy – energy (canaux différents) | 8 | 0 |
| energy – max (même canal) | 3 | 0 |
| energy – min (même canal) | 4 | 0 |
| energy – rms (canal vs moyenne) | 5 | 0 |
| energy – rms (canaux différents) | 16 | 0 |
| energy – std (canal vs moyenne) | 5 | 0 |
| energy – std (canaux différents) | 16 | 0 |
| energy – std (même canal) | 20 | 0 |
| energy – variance (canaux différents) | 16 | 0 |
| energy – variance (même canal) | 20 | 0 |
| max – mean (même canal) | 0 | 3 |
| max – rms (même canal) | 3 | 0 |
| max – std (même canal) | 3 | 0 |
| max – variance (même canal) | 3 | 0 |
| mean – min (même canal) | 0 | 4 |
| min – rms (même canal) | 4 | 0 |
| min – std (même canal) | 4 | 0 |
| min – variance (même canal) | 4 | 0 |
| rms – rms (canal vs moyenne) | 5 | 0 |
| rms – rms (canaux différents) | 8 | 0 |
| rms – std (canal vs moyenne) | 10 | 0 |
| rms – std (canaux différents) | 16 | 0 |
| rms – std (même canal) | 20 | 0 |
| rms – variance (canal vs moyenne) | 5 | 0 |
| rms – variance (canaux différents) | 16 | 0 |
| rms – variance (même canal) | 20 | 0 |
| std – std (canal vs moyenne) | 2 | 2 |
| std – std (canaux différents) | 4 | 10 |
| std – variance (canal vs moyenne) | 2 | 2 |
| std – variance (canaux différents) | 8 | 20 |
| variance – variance (canaux différents) | 4 | 10 |
