# Partie VII — Analyse exploratoire (official, signal non filtré)

Analyse de 31,419 fenêtres : 30,175 de classe 0 et 1,244 de classe 1. Aucun modèle n’est entraîné dans cette partie.

## Q10 — Comparaison entre les classes

Les boxplots utilisent une échelle logarithmique car l’énergie et les puissances spectrales sont très asymétriques. Les valeurs aberrantes ne sont pas affichées dans les boîtes, mais elles restent présentes dans les calculs.

| Caractéristique | Médiane classe 0 | Médiane classe 1 | Cohen d | AUC séparatrice (globale) | AUC séparatrice (médiane intra-enregistrement) | Plus élevée en classe |
|---|---:|---:|---:|---:|---:|---:|
| avg_energy | 1.09625e+08 | 2.44807e+08 | 0.167 | 0.595 | 0.791 | 1 |
| avg_delta_power | 397.239 | 12904.6 | 0.073 | 0.906 | 0.951 | 1 |
| avg_theta_power | 91.9018 | 1468.97 | 0.134 | 0.925 | 0.949 | 1 |
| avg_std | 57.6341 | 142.985 | 0.668 | 0.824 | 0.956 | 1 |
| avg_rms | 387.148 | 581.641 | 0.324 | 0.615 | 0.837 | 1 |
| avg_amplitude | 256.556 | 682.515 | 0.895 | 0.868 | 0.974 | 1 |
| avg_delta_relative_power | 0.686265 | 0.725451 | 0.155 | 0.541 | 0.543 | 1 |
| avg_theta_relative_power | 0.158894 | 0.106547 | -0.425 | 0.633 | 0.695 | 0 |

Le graphique correspondant est [class_comparisons.png](../outputs/figures/official_raw/part_vii/class_comparisons.png). Une AUC séparatrice proche de 1 indique une bonne séparation univariée; 0,5 indique une absence de séparation. L’AUC globale mélange toutes les fenêtres; l’AUC intra-enregistrement est calculée dans chaque enregistrement puis résumée par la médiane, ce qui neutralise les différences de niveau entre enregistrements. « Plus élevée en classe » compare les médianes.

Les distributions sont très asymétriques : un Cohen d faible (calculé sur les moyennes et écarts-types) peut coexister avec une AUC élevée (fondée sur les rangs). L’AUC est donc la mesure de référence ici.

### Caractéristiques les plus discriminantes (AUC globale)

| Rang | Caractéristique | AUC globale | AUC intra-enregistrement | Cohen d | Direction (médianes) |
|---:|---|---:|---:|---:|---|
| 1 | T5_alpha_power | 0.931 | 0.960 | 0.663 | crise > hors crise |
| 2 | Pz_beta_power | 0.930 | 0.968 | 0.790 | crise > hors crise |
| 3 | P3_beta_power | 0.930 | 0.968 | 0.842 | crise > hors crise |
| 4 | P4_beta_power | 0.930 | 0.964 | 0.767 | crise > hors crise |
| 5 | avg_beta_power | 0.930 | 0.971 | 0.250 | crise > hors crise |
| 6 | C3_beta_power | 0.929 | 0.963 | 0.575 | crise > hors crise |
| 7 | F7_alpha_power | 0.929 | 0.967 | 0.070 | crise > hors crise |
| 8 | avg_alpha_power | 0.928 | 0.948 | 0.351 | crise > hors crise |
| 9 | Cz_beta_power | 0.928 | 0.960 | 0.569 | crise > hors crise |
| 10 | P3_alpha_power | 0.928 | 0.956 | 0.680 | crise > hors crise |
| 11 | T6_alpha_power | 0.928 | 0.957 | 0.619 | crise > hors crise |
| 12 | C4_beta_power | 0.928 | 0.963 | 0.620 | crise > hors crise |
| 13 | O1_beta_power | 0.928 | 0.970 | 0.804 | crise > hors crise |
| 14 | T4_beta_power | 0.927 | 0.965 | 0.650 | crise > hors crise |
| 15 | O2_beta_power | 0.927 | 0.967 | 0.111 | crise > hors crise |

### Caractéristiques les plus discriminantes à l’intérieur des enregistrements

| Rang | Caractéristique | AUC intra-enregistrement | AUC globale |
|---:|---|---:|---:|
| 1 | T5_amplitude | 0.979 | 0.870 |
| 2 | Cz_amplitude | 0.977 | 0.888 |
| 3 | P3_amplitude | 0.976 | 0.879 |
| 4 | F7_amplitude | 0.975 | 0.844 |
| 5 | C4_amplitude | 0.975 | 0.878 |
| 6 | Pz_amplitude | 0.974 | 0.856 |
| 7 | T3_amplitude | 0.974 | 0.873 |
| 8 | avg_amplitude | 0.974 | 0.868 |
| 9 | C3_amplitude | 0.973 | 0.887 |
| 10 | avg_beta_power | 0.971 | 0.930 |

### Avertissement : composante continue (offset DC)

Plusieurs enregistrements ont une moyenne de signal très éloignée de zéro (médiane par enregistrement de `avg_mean` entre environ −830 et +1000), ce qui indique un signal non filtré passe-haut. `mean`, `min`, `max`, `rms` et `energy` sont calculés sans retrait de la moyenne : leur niveau dépend donc de cet offset propre à chaque enregistrement. À l’intérieur d’un enregistrement (offset constant), `min` et `max` suivent encore l’amplitude des pointes-ondes, mais entre enregistrements l’offset domine, d’où l’écart entre AUC globale et AUC intra-enregistrement :

| Caractéristique | AUC globale | AUC intra-enregistrement |
|---|---:|---:|
| avg_mean | 0.565 | 0.561 |
| avg_min | 0.738 | 0.956 |
| avg_max | 0.567 | 0.960 |
| avg_rms | 0.615 | 0.837 |
| avg_energy | 0.595 | 0.791 |

Un modèle évalué sur des enregistrements non vus risque d’utiliser ces variables pour reconnaître l’enregistrement plutôt que la crise. `std`, `variance`, `amplitude` (crête à crête) et les puissances spectrales (calculées après retrait de la moyenne) ne sont pas affectées. Ce constat motive le filtrage passe-bande du jeu de données principal.

Ces résultats mesurent chaque variable séparément et ne constituent pas encore un modèle prédictif.

## Q11 — Corrélations et redondance

Deux matrices sont calculées. Pearson mesure une relation linéaire et est dominée par les valeurs extrêmes (artefacts) de ces distributions asymétriques. Spearman mesure une relation monotone sur les rangs : c’est la mesure retenue pour juger la redondance, car `variance = std²` ou `energy ∝ rms²` sont des relations parfaitement monotones mais non linéaires.

- Paires avec |r de Pearson| ≥ 0,95 : 317 sur 51040.
- Paires avec |ρ de Spearman| ≥ 0,95 : 171.

Exemples Pearson vs Spearman sur les moyennes inter-canaux :

| Paire | Pearson r | Spearman ρ |
|---|---:|---:|
| avg_std / avg_variance | 0.777 | 0.973 |
| avg_rms / avg_energy | 0.902 | 0.984 |
| avg_std / avg_amplitude | 0.955 | 0.971 |
| avg_theta_power / avg_alpha_power | 0.794 | 0.511 |

### Caractéristiques fortement corrélées (moyennes inter-canaux, |ρ| ≥ 0,90)

| Caractéristique 1 | Caractéristique 2 | Spearman ρ |
|---|---|---:|
| avg_rms | avg_energy | 0.984 |
| avg_std | avg_variance | 0.973 |
| avg_std | avg_amplitude | 0.971 |
| avg_variance | avg_amplitude | 0.936 |
| avg_mean | avg_min | 0.931 |
| avg_mean | avg_max | 0.911 |

### Groupes redondants (|ρ| ≥ 0,95, toutes caractéristiques)

Chaque groupe est une composante connexe : ses membres sont reliés par des chaînes de paires avec |ρ| ≥ 0,95. Un seul représentant par groupe suffit avant les modèles sensibles à la colinéarité.

1. (27) `C3_amplitude`, `C3_std`, `C3_variance`, `Cz_amplitude`, `Cz_std`, `Cz_variance`, `P3_amplitude`, `P3_std`, `P3_variance`, `P4_amplitude`, `P4_std`, `P4_variance`, `T3_amplitude`, `T3_std`, `T3_variance`, `T4_amplitude`, `T4_std`, `T4_variance`, `T5_amplitude`, `T5_std`, `T5_variance`, `T6_amplitude`, `T6_std`, `T6_variance`, `avg_amplitude`, `avg_std`, `avg_variance` — représentant suggéré : `T5_amplitude` (meilleure AUC intra-enregistrement 0.979).
2. (6) `F3_amplitude`, `F3_std`, `F3_variance`, `F4_amplitude`, `F4_std`, `F4_variance` — représentant suggéré : `F3_amplitude` (meilleure AUC intra-enregistrement 0.969).
3. (3) `F7_max`, `F7_mean`, `F7_min` — représentant suggéré : `F7_min` (meilleure AUC intra-enregistrement 0.934).
4. (3) `C3_max`, `C3_mean`, `C3_min` — représentant suggéré : `C3_max` (meilleure AUC intra-enregistrement 0.957).
5. (3) `T6_max`, `T6_mean`, `T6_min` — représentant suggéré : `T6_max` (meilleure AUC intra-enregistrement 0.947).
6. (3) `Fp1_amplitude`, `Fp1_std`, `Fp1_variance` — représentant suggéré : `Fp1_amplitude` (meilleure AUC intra-enregistrement 0.911).
7. (3) `Fp2_amplitude`, `Fp2_std`, `Fp2_variance` — représentant suggéré : `Fp2_amplitude` (meilleure AUC intra-enregistrement 0.914).
8. (3) `F7_amplitude`, `F7_std`, `F7_variance` — représentant suggéré : `F7_amplitude` (meilleure AUC intra-enregistrement 0.975).
9. (3) `Fz_amplitude`, `Fz_std`, `Fz_variance` — représentant suggéré : `Fz_amplitude` (meilleure AUC intra-enregistrement 0.948).
10. (3) `F8_amplitude`, `F8_std`, `F8_variance` — représentant suggéré : `F8_amplitude` (meilleure AUC intra-enregistrement 0.949).
11. (3) `C4_amplitude`, `C4_std`, `C4_variance` — représentant suggéré : `C4_amplitude` (meilleure AUC intra-enregistrement 0.975).
12. (3) `Pz_amplitude`, `Pz_std`, `Pz_variance` — représentant suggéré : `Pz_amplitude` (meilleure AUC intra-enregistrement 0.974).
13. (3) `O1_amplitude`, `O1_std`, `O1_variance` — représentant suggéré : `O1_amplitude` (meilleure AUC intra-enregistrement 0.962).
14. (3) `O2_amplitude`, `O2_std`, `O2_variance` — représentant suggéré : `O2_amplitude` (meilleure AUC intra-enregistrement 0.955).
15. (3) `P3_delta_power`, `P4_delta_power`, `Pz_delta_power` — représentant suggéré : `P4_delta_power` (meilleure AUC intra-enregistrement 0.949).
16. (2) `T5_mean`, `T5_min` — représentant suggéré : `T5_min` (meilleure AUC intra-enregistrement 0.943).
17. (2) `Fp1_energy`, `Fp1_rms` — représentant suggéré : `Fp1_rms` (meilleure AUC intra-enregistrement 0.633).
18. (2) `Fp2_energy`, `Fp2_rms` — représentant suggéré : `Fp2_energy` (meilleure AUC intra-enregistrement 0.671).
19. (2) `F7_energy`, `F7_rms` — représentant suggéré : `F7_energy` (meilleure AUC intra-enregistrement 0.639).
20. (2) `F3_energy`, `F3_rms` — représentant suggéré : `F3_energy` (meilleure AUC intra-enregistrement 0.786).
21. (2) `Fz_energy`, `Fz_rms` — représentant suggéré : `Fz_energy` (meilleure AUC intra-enregistrement 0.891).
22. (2) `F4_energy`, `F4_rms` — représentant suggéré : `F4_energy` (meilleure AUC intra-enregistrement 0.724).
23. (2) `F8_energy`, `F8_rms` — représentant suggéré : `F8_energy` (meilleure AUC intra-enregistrement 0.816).
24. (2) `T3_energy`, `T3_rms` — représentant suggéré : `T3_energy` (meilleure AUC intra-enregistrement 0.781).
25. (2) `C3_energy`, `C3_rms` — représentant suggéré : `C3_energy` (meilleure AUC intra-enregistrement 0.711).
26. (2) `Cz_energy`, `Cz_rms` — représentant suggéré : `Cz_energy` (meilleure AUC intra-enregistrement 0.912).
27. (2) `C4_energy`, `C4_rms` — représentant suggéré : `C4_energy` (meilleure AUC intra-enregistrement 0.816).
28. (2) `T4_energy`, `T4_rms` — représentant suggéré : `T4_energy` (meilleure AUC intra-enregistrement 0.912).
29. (2) `T5_energy`, `T5_rms` — représentant suggéré : `T5_energy` (meilleure AUC intra-enregistrement 0.775).
30. (2) `P3_energy`, `P3_rms` — représentant suggéré : `P3_energy` (meilleure AUC intra-enregistrement 0.812).
31. (2) `Pz_energy`, `Pz_rms` — représentant suggéré : `Pz_energy` (meilleure AUC intra-enregistrement 0.929).
32. (2) `P4_energy`, `P4_rms` — représentant suggéré : `P4_energy` (meilleure AUC intra-enregistrement 0.832).
33. (2) `T6_energy`, `T6_rms` — représentant suggéré : `T6_energy` (meilleure AUC intra-enregistrement 0.745).
34. (2) `O1_energy`, `O1_rms` — représentant suggéré : `O1_energy` (meilleure AUC intra-enregistrement 0.831).
35. (2) `O2_energy`, `O2_rms` — représentant suggéré : `O2_energy` (meilleure AUC intra-enregistrement 0.745).
36. (2) `avg_energy`, `avg_rms` — représentant suggéré : `avg_rms` (meilleure AUC intra-enregistrement 0.837).

Figures : corrélations moyennes [Spearman](../outputs/figures/official_raw/part_vii/correlation_average_features_spearman.png) et [Pearson](../outputs/figures/official_raw/part_vii/correlation_average_features.png); matrice complète [Spearman](../outputs/figures/official_raw/part_vii/correlation_all_features_spearman.png) et [Pearson](../outputs/figures/official_raw/part_vii/correlation_all_features.png). Tableaux : `outputs/feature_class_comparison_official_raw.csv`, `outputs/high_correlations_official_raw.csv` (Pearson), `outputs/high_correlations_spearman_official_raw.csv`, `outputs/feature_correlations_official_raw.csv` et `outputs/feature_correlations_spearman_official_raw.csv`.
