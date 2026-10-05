# Partie VII — Analyse exploratoire (official, signal filtré passe-bande 0,5–40 Hz)

Analyse de 31,419 fenêtres : 30,175 de classe 0 et 1,244 de classe 1. Aucun modèle n’est entraîné dans cette partie.

## Q10 — Comparaison entre les classes

Les boxplots utilisent une échelle logarithmique car l’énergie et les puissances spectrales sont très asymétriques. Les valeurs aberrantes ne sont pas affichées dans les boîtes, mais elles restent présentes dans les calculs.

| Caractéristique | Médiane classe 0 | Médiane classe 1 | Cohen d | AUC séparatrice (globale) | AUC séparatrice (médiane intra-enregistrement) | Plus élevée en classe |
|---|---:|---:|---:|---:|---:|---:|
| avg_energy | 314282 | 8.55694e+06 | 0.133 | 0.929 | 0.974 | 1 |
| avg_delta_power | 305.394 | 11102.8 | 0.087 | 0.917 | 0.962 | 1 |
| avg_theta_power | 93.3107 | 1392.99 | 0.183 | 0.926 | 0.953 | 1 |
| avg_std | 22.8542 | 118.883 | 1.178 | 0.929 | 0.980 | 1 |
| avg_rms | 22.9313 | 118.986 | 1.173 | 0.929 | 0.980 | 1 |
| avg_amplitude | 121.035 | 566.429 | 1.116 | 0.934 | 0.978 | 1 |
| avg_delta_relative_power | 0.638078 | 0.706001 | 0.242 | 0.574 | 0.562 | 1 |
| avg_theta_relative_power | 0.176881 | 0.114012 | -0.509 | 0.659 | 0.699 | 0 |

Le graphique correspondant est [class_comparisons.png](../outputs/figures/official/part_vii/class_comparisons.png). Une AUC séparatrice proche de 1 indique une bonne séparation univariée; 0,5 indique une absence de séparation. L’AUC globale mélange toutes les fenêtres; l’AUC intra-enregistrement est calculée dans chaque enregistrement puis résumée par la médiane, ce qui neutralise les différences de niveau entre enregistrements. « Plus élevée en classe » compare les médianes.

Les distributions sont très asymétriques : un Cohen d faible (calculé sur les moyennes et écarts-types) peut coexister avec une AUC élevée (fondée sur les rangs). L’AUC est donc la mesure de référence ici.

### Caractéristiques les plus discriminantes (AUC globale)

| Rang | Caractéristique | AUC globale | AUC intra-enregistrement | Cohen d | Direction (médianes) |
|---:|---|---:|---:|---:|---|
| 1 | C3_amplitude | 0.938 | 0.978 | 1.316 | crise > hors crise |
| 2 | C4_amplitude | 0.936 | 0.979 | 1.318 | crise > hors crise |
| 3 | avg_amplitude | 0.934 | 0.978 | 1.116 | crise > hors crise |
| 4 | C4_variance | 0.934 | 0.977 | 0.186 | crise > hors crise |
| 5 | C4_std | 0.934 | 0.977 | 1.473 | crise > hors crise |
| 6 | C3_std | 0.934 | 0.973 | 1.469 | crise > hors crise |
| 7 | C3_variance | 0.934 | 0.973 | 0.190 | crise > hors crise |
| 8 | C4_energy | 0.934 | 0.977 | 0.185 | crise > hors crise |
| 9 | C4_rms | 0.934 | 0.977 | 1.467 | crise > hors crise |
| 10 | C3_rms | 0.934 | 0.973 | 1.463 | crise > hors crise |
| 11 | C3_energy | 0.934 | 0.973 | 0.189 | crise > hors crise |
| 12 | C3_max | 0.934 | 0.973 | 1.030 | crise > hors crise |
| 13 | C3_min | 0.933 | 0.977 | -1.515 | crise < hors crise |
| 14 | avg_min | 0.933 | 0.979 | -1.255 | crise < hors crise |
| 15 | C4_min | 0.933 | 0.978 | -1.462 | crise < hors crise |

### Caractéristiques les plus discriminantes à l’intérieur des enregistrements

| Rang | Caractéristique | AUC intra-enregistrement | AUC globale |
|---:|---|---:|---:|
| 1 | T3_min | 0.981 | 0.928 |
| 2 | avg_std | 0.980 | 0.929 |
| 3 | T3_amplitude | 0.980 | 0.931 |
| 4 | avg_rms | 0.980 | 0.929 |
| 5 | avg_min | 0.979 | 0.933 |
| 6 | T3_variance | 0.979 | 0.926 |
| 7 | T3_std | 0.979 | 0.926 |
| 8 | C4_amplitude | 0.979 | 0.936 |
| 9 | C3_amplitude | 0.978 | 0.938 |
| 10 | T3_energy | 0.978 | 0.926 |

`mean` est quasi nulle après filtrage : son AUC reflète du bruit résiduel, pas une information utile. Elle est conservée car demandée par l’énoncé. L’effet du filtre sur les variables sensibles à l’offset est détaillé dans la dernière section.

Ces résultats mesurent chaque variable séparément et ne constituent pas encore un modèle prédictif.

## Q11 — Corrélations et redondance

Deux matrices sont calculées. Pearson mesure une relation linéaire et est dominée par les valeurs extrêmes (artefacts) de ces distributions asymétriques. Spearman mesure une relation monotone sur les rangs : c’est la mesure retenue pour juger la redondance, car `variance = std²` ou `energy ∝ rms²` sont des relations parfaitement monotones mais non linéaires.

- Paires avec |r de Pearson| ≥ 0,95 : 271 sur 10296.
- Paires avec |ρ de Spearman| ≥ 0,95 : 208.

Exemples Pearson vs Spearman sur les moyennes inter-canaux :

| Paire | Pearson r | Spearman ρ |
|---|---:|---:|
| avg_std / avg_variance | 0.828 | 0.993 |
| avg_rms / avg_energy | 0.828 | 0.993 |
| avg_std / avg_amplitude | 0.968 | 0.986 |
| avg_theta_power / avg_alpha_power | 0.900 | 0.490 |

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
| avg_min | avg_amplitude | -0.977 |
| avg_variance | avg_amplitude | 0.976 |
| avg_amplitude | avg_energy | 0.976 |
| avg_max | avg_amplitude | 0.974 |
| avg_std | avg_min | -0.964 |
| avg_min | avg_rms | -0.964 |
| avg_std | avg_max | 0.963 |
| avg_max | avg_rms | 0.962 |
| avg_variance | avg_min | -0.956 |
| avg_min | avg_energy | -0.955 |
| avg_variance | avg_max | 0.953 |
| avg_max | avg_energy | 0.953 |
| avg_min | avg_max | -0.908 |
| avg_energy | avg_delta_power | 0.903 |
| avg_variance | avg_delta_power | 0.902 |

### Groupes redondants (|ρ| ≥ 0,95, toutes caractéristiques)

Chaque groupe est une composante connexe : ses membres sont reliés par des chaînes de paires avec |ρ| ≥ 0,95. Un seul représentant par groupe suffit avant les modèles sensibles à la colinéarité.

1. (35) `C3_amplitude`, `C3_energy`, `C3_max`, `C3_min`, `C3_rms`, `C3_std`, `C3_variance`, `C4_amplitude`, `C4_energy`, `C4_max`, `C4_min`, `C4_rms`, `C4_std`, `C4_variance`, `T3_amplitude`, `T3_energy`, `T3_max`, `T3_min`, `T3_rms`, `T3_std`, `T3_variance`, `T4_amplitude`, `T4_energy`, `T4_max`, `T4_min`, `T4_rms`, `T4_std`, `T4_variance`, `avg_amplitude`, `avg_energy`, `avg_max`, `avg_min`, `avg_rms`, `avg_std`, `avg_variance` — représentant suggéré : `T3_min` (meilleure AUC intra-enregistrement 0.981).
2. (7) `Fp1_amplitude`, `Fp1_energy`, `Fp1_max`, `Fp1_min`, `Fp1_rms`, `Fp1_std`, `Fp1_variance` — représentant suggéré : `Fp1_amplitude` (meilleure AUC intra-enregistrement 0.949).
3. (7) `Fp2_amplitude`, `Fp2_energy`, `Fp2_max`, `Fp2_min`, `Fp2_rms`, `Fp2_std`, `Fp2_variance` — représentant suggéré : `Fp2_amplitude` (meilleure AUC intra-enregistrement 0.932).
4. (7) `O1_amplitude`, `O1_energy`, `O1_max`, `O1_min`, `O1_rms`, `O1_std`, `O1_variance` — représentant suggéré : `O1_std` (meilleure AUC intra-enregistrement 0.975).
5. (7) `O2_amplitude`, `O2_energy`, `O2_max`, `O2_min`, `O2_rms`, `O2_std`, `O2_variance` — représentant suggéré : `O2_energy` (meilleure AUC intra-enregistrement 0.966).
6. (2) `C4_delta_power`, `T4_delta_power` — représentant suggéré : `C4_delta_power` (meilleure AUC intra-enregistrement 0.966).

Figures : corrélations moyennes [Spearman](../outputs/figures/official/part_vii/correlation_average_features_spearman.png) et [Pearson](../outputs/figures/official/part_vii/correlation_average_features.png); matrice complète [Spearman](../outputs/figures/official/part_vii/correlation_all_features_spearman.png) et [Pearson](../outputs/figures/official/part_vii/correlation_all_features.png). Tableaux : `outputs/feature_class_comparison_official.csv`, `outputs/high_correlations_official.csv` (Pearson), `outputs/high_correlations_spearman_official.csv`, `outputs/feature_correlations_official.csv` et `outputs/feature_correlations_spearman_official.csv`.

## Effet du prétraitement (comparaison avec les données non filtrées)

Le filtre passe-bande 0,5–40 Hz retire l’offset continu propre à chaque enregistrement (< 0,5 Hz) et le bruit secteur à 50 Hz, très présent dans le signal brut (pic à 50 Hz 300 à 35 000 fois au-dessus du niveau 30–45 Hz selon les enregistrements). Illustration : [avant/après filtrage](../outputs/figures/official/preprocessing/211104B_D_seizure3_filtering.png). Le rapport non filtré complet reste disponible : [PART_VII_ANALYSIS_OFFICIAL_RAW.md](PART_VII_ANALYSIS_OFFICIAL_RAW.md).

### Écart AUC globale / intra-enregistrement des variables sensibles à l’offset

| Caractéristique | Brut : globale | Brut : intra | Brut : écart | Filtré : globale | Filtré : intra | Filtré : écart |
|---|---:|---:|---:|---:|---:|---:|
| avg_mean | 0.570 | 0.558 | -0.012 | 0.513 | 0.505 | -0.008 |
| avg_min | 0.727 | 0.957 | +0.230 | 0.933 | 0.979 | +0.046 |
| avg_max | 0.551 | 0.960 | +0.409 | 0.929 | 0.974 | +0.044 |
| avg_rms | 0.581 | 0.812 | +0.231 | 0.929 | 0.980 | +0.052 |
| avg_energy | 0.565 | 0.766 | +0.201 | 0.929 | 0.974 | +0.045 |

Écart moyen sur ces cinq variables : +0.212 avant filtrage, +0.036 après. Après filtrage, la moyenne de chaque fenêtre est proche de 0 : `mean` n’apporte plus d’information (elle est conservée car demandée par l’énoncé), `rms` devient presque identique à `std`, et `min`/`max` mesurent l’amplitude des oscillations au lieu de l’offset.

### Caractéristiques les plus discriminantes avant / après filtrage (AUC globale)

| Rang | Brut | AUC | Filtré | AUC |
|---:|---|---:|---|---:|
| 1 | avg_beta_power | 0.932 | C3_amplitude | 0.938 |
| 2 | C3_beta_power | 0.929 | C4_amplitude | 0.936 |
| 3 | avg_alpha_power | 0.928 | avg_amplitude | 0.934 |
| 4 | C4_beta_power | 0.928 | C4_variance | 0.934 |
| 5 | O1_beta_power | 0.928 | C4_std | 0.934 |
| 6 | T4_beta_power | 0.927 | C3_std | 0.934 |
| 7 | O2_beta_power | 0.927 | C3_variance | 0.934 |
| 8 | avg_theta_power | 0.926 | C4_energy | 0.934 |
| 9 | C3_alpha_power | 0.925 | C4_rms | 0.934 |
| 10 | T3_alpha_power | 0.923 | C3_rms | 0.934 |

### Groupes redondants avant / après filtrage (|ρ de Spearman| ≥ 0,95)

Brut : 50 paires, 16 groupes (tailles 12, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2). Filtré : 208 paires, 6 groupes (tailles 35, 7, 7, 7, 7, 2).

| Type de paire | Apparues après filtrage | Disparues après filtrage |
|---|---:|---:|
| amplitude – amplitude (canal vs moyenne) | 1 | 1 |
| amplitude – amplitude (canaux différents) | 1 | 0 |
| amplitude – energy (même canal) | 9 | 0 |
| amplitude – max (même canal) | 9 | 0 |
| amplitude – min (même canal) | 9 | 0 |
| amplitude – rms (même canal) | 9 | 0 |
| delta_power – delta_power (canaux différents) | 1 | 0 |
| energy – energy (canaux différents) | 3 | 0 |
| energy – max (même canal) | 3 | 0 |
| energy – min (même canal) | 3 | 0 |
| energy – rms (canal vs moyenne) | 3 | 0 |
| energy – rms (canaux différents) | 6 | 0 |
| energy – std (canal vs moyenne) | 3 | 0 |
| energy – std (canaux différents) | 6 | 0 |
| energy – std (même canal) | 9 | 0 |
| energy – variance (canaux différents) | 6 | 0 |
| energy – variance (même canal) | 9 | 0 |
| max – mean (même canal) | 0 | 1 |
| max – rms (même canal) | 3 | 0 |
| max – std (même canal) | 3 | 0 |
| max – variance (même canal) | 3 | 0 |
| mean – min (même canal) | 0 | 1 |
| min – rms (même canal) | 3 | 0 |
| min – std (même canal) | 3 | 0 |
| min – variance (même canal) | 3 | 0 |
| rms – rms (canal vs moyenne) | 3 | 0 |
| rms – rms (canaux différents) | 3 | 0 |
| rms – std (canal vs moyenne) | 6 | 0 |
| rms – std (canaux différents) | 6 | 0 |
| rms – std (même canal) | 9 | 0 |
| rms – variance (canal vs moyenne) | 3 | 0 |
| rms – variance (canaux différents) | 6 | 0 |
| rms – variance (même canal) | 9 | 0 |
| std – std (canal vs moyenne) | 1 | 0 |
| std – std (canaux différents) | 2 | 0 |
| std – variance (canal vs moyenne) | 1 | 1 |
| std – variance (canaux différents) | 4 | 0 |
| variance – variance (canal vs moyenne) | 0 | 1 |
| variance – variance (canaux différents) | 2 | 0 |
