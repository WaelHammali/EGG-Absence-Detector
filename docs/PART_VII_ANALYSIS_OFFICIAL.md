# Partie VII — Analyse exploratoire (official)

Analyse de 31,419 fenêtres : 30,175 de classe 0 et 1,244 de classe 1. Aucun modèle n’est entraîné dans cette partie.

## Q10 — Comparaison entre les classes

Les boxplots utilisent une échelle logarithmique car l’énergie et les puissances spectrales sont très asymétriques. Les valeurs aberrantes ne sont pas affichées dans les boîtes, mais elles restent présentes dans les calculs.

| Caractéristique | Médiane classe 0 | Médiane classe 1 | Cohen d | AUC séparatrice (globale) | AUC séparatrice (médiane intra-enregistrement) | Plus élevée en classe |
|---|---:|---:|---:|---:|---:|---:|
| avg_energy | 1.39513e+08 | 1.89985e+08 | 0.044 | 0.565 | 0.766 | 1 |
| avg_delta_power | 403.287 | 12547.9 | 0.067 | 0.907 | 0.955 | 1 |
| avg_theta_power | 93.2955 | 1393.4 | 0.183 | 0.926 | 0.953 | 1 |
| avg_std | 56.5743 | 136.081 | 0.651 | 0.821 | 0.951 | 1 |
| avg_rms | 445.525 | 503.726 | 0.197 | 0.581 | 0.812 | 1 |
| avg_amplitude | 254.235 | 662.675 | 0.867 | 0.865 | 0.972 | 1 |
| avg_delta_relative_power | 0.687682 | 0.721801 | 0.109 | 0.529 | 0.537 | 1 |
| avg_theta_relative_power | 0.152872 | 0.10529 | -0.403 | 0.623 | 0.679 | 0 |

Le graphique correspondant est [class_comparisons.png](../outputs/figures/official/part_vii/class_comparisons.png). Une AUC séparatrice proche de 1 indique une bonne séparation univariée; 0,5 indique une absence de séparation. L’AUC globale mélange toutes les fenêtres; l’AUC intra-enregistrement est calculée dans chaque enregistrement puis résumée par la médiane, ce qui neutralise les différences de niveau entre enregistrements. « Plus élevée en classe » compare les médianes.

Les distributions sont très asymétriques : un Cohen d faible (calculé sur les moyennes et écarts-types) peut coexister avec une AUC élevée (fondée sur les rangs). L’AUC est donc la mesure de référence ici.

### Caractéristiques les plus discriminantes (AUC globale)

| Rang | Caractéristique | AUC globale | AUC intra-enregistrement | Cohen d | Direction (médianes) |
|---:|---|---:|---:|---:|---|
| 1 | avg_beta_power | 0.932 | 0.970 | 0.438 | crise > hors crise |
| 2 | C3_beta_power | 0.929 | 0.963 | 0.575 | crise > hors crise |
| 3 | avg_alpha_power | 0.928 | 0.953 | 0.382 | crise > hors crise |
| 4 | C4_beta_power | 0.928 | 0.963 | 0.620 | crise > hors crise |
| 5 | O1_beta_power | 0.928 | 0.970 | 0.804 | crise > hors crise |
| 6 | T4_beta_power | 0.927 | 0.965 | 0.650 | crise > hors crise |
| 7 | O2_beta_power | 0.927 | 0.967 | 0.111 | crise > hors crise |
| 8 | avg_theta_power | 0.926 | 0.953 | 0.183 | crise > hors crise |
| 9 | C3_alpha_power | 0.925 | 0.959 | 0.540 | crise > hors crise |
| 10 | T3_alpha_power | 0.923 | 0.961 | 0.487 | crise > hors crise |
| 11 | T4_alpha_power | 0.923 | 0.960 | 0.426 | crise > hors crise |
| 12 | C3_theta_power | 0.921 | 0.955 | 0.250 | crise > hors crise |
| 13 | T4_theta_power | 0.921 | 0.951 | 0.197 | crise > hors crise |
| 14 | C4_theta_power | 0.920 | 0.956 | 0.236 | crise > hors crise |
| 15 | T3_beta_power | 0.920 | 0.968 | 0.685 | crise > hors crise |

### Caractéristiques les plus discriminantes à l’intérieur des enregistrements

| Rang | Caractéristique | AUC intra-enregistrement | AUC globale |
|---:|---|---:|---:|
| 1 | C4_amplitude | 0.975 | 0.878 |
| 2 | T3_amplitude | 0.974 | 0.873 |
| 3 | C3_amplitude | 0.973 | 0.887 |
| 4 | avg_amplitude | 0.972 | 0.865 |
| 5 | avg_beta_power | 0.970 | 0.932 |
| 6 | O1_beta_power | 0.970 | 0.928 |
| 7 | T4_amplitude | 0.969 | 0.869 |
| 8 | T3_beta_power | 0.968 | 0.920 |
| 9 | T3_std | 0.967 | 0.832 |
| 10 | T3_variance | 0.967 | 0.832 |

### Avertissement : composante continue (offset DC)

Plusieurs enregistrements ont une moyenne de signal très éloignée de zéro (médiane par enregistrement de `avg_mean` entre environ −830 et +1000), ce qui indique un signal non filtré passe-haut. `mean`, `min`, `max`, `rms` et `energy` sont calculés sans retrait de la moyenne : leur niveau dépend donc de cet offset propre à chaque enregistrement. À l’intérieur d’un enregistrement (offset constant), `min` et `max` suivent encore l’amplitude des pointes-ondes, mais entre enregistrements l’offset domine, d’où l’écart entre AUC globale et AUC intra-enregistrement :

| Caractéristique | AUC globale | AUC intra-enregistrement |
|---|---:|---:|
| avg_mean | 0.570 | 0.558 |
| avg_min | 0.727 | 0.957 |
| avg_max | 0.551 | 0.960 |
| avg_rms | 0.581 | 0.812 |
| avg_energy | 0.565 | 0.766 |

Un modèle évalué sur des enregistrements non vus risque d’utiliser ces variables pour reconnaître l’enregistrement plutôt que la crise. `std`, `variance`, `amplitude` (crête à crête) et les puissances spectrales (calculées après retrait de la moyenne) ne sont pas affectées. À décider avant les modèles : retirer ces variables ou retirer la moyenne de chaque fenêtre.

Ces résultats mesurent chaque variable séparément et ne constituent pas encore un modèle prédictif.

## Q11 — Corrélations et redondance

Deux matrices sont calculées. Pearson mesure une relation linéaire et est dominée par les valeurs extrêmes (artefacts) de ces distributions asymétriques. Spearman mesure une relation monotone sur les rangs : c’est la mesure retenue pour juger la redondance, car `variance = std²` ou `energy ∝ rms²` sont des relations parfaitement monotones mais non linéaires.

- Paires avec |r de Pearson| ≥ 0,95 : 88 sur 10296.
- Paires avec |ρ de Spearman| ≥ 0,95 : 50.

Exemples Pearson vs Spearman sur les moyennes inter-canaux :

| Paire | Pearson r | Spearman ρ |
|---|---:|---:|
| avg_std / avg_variance | 0.768 | 0.993 |
| avg_rms / avg_energy | 0.885 | 0.985 |
| avg_std / avg_amplitude | 0.953 | 0.973 |
| avg_theta_power / avg_alpha_power | 0.900 | 0.490 |

### Caractéristiques fortement corrélées (moyennes inter-canaux, |ρ| ≥ 0,90)

| Caractéristique 1 | Caractéristique 2 | Spearman ρ |
|---|---|---:|
| avg_std | avg_variance | 0.993 |
| avg_rms | avg_energy | 0.985 |
| avg_std | avg_amplitude | 0.973 |
| avg_variance | avg_amplitude | 0.965 |
| avg_mean | avg_max | 0.925 |
| avg_mean | avg_min | 0.917 |

### Groupes redondants (|ρ| ≥ 0,95, toutes caractéristiques)

Chaque groupe est une composante connexe : ses membres sont reliés par des chaînes de paires avec |ρ| ≥ 0,95. Un seul représentant par groupe suffit avant les modèles sensibles à la colinéarité.

1. (12) `C3_amplitude`, `C3_std`, `C3_variance`, `T3_amplitude`, `T3_std`, `T3_variance`, `T4_amplitude`, `T4_std`, `T4_variance`, `avg_amplitude`, `avg_std`, `avg_variance` — représentant suggéré : `T3_amplitude` (meilleure AUC intra-enregistrement 0.974).
2. (3) `C3_max`, `C3_mean`, `C3_min` — représentant suggéré : `C3_max` (meilleure AUC intra-enregistrement 0.957).
3. (3) `Fp1_amplitude`, `Fp1_std`, `Fp1_variance` — représentant suggéré : `Fp1_amplitude` (meilleure AUC intra-enregistrement 0.911).
4. (3) `Fp2_amplitude`, `Fp2_std`, `Fp2_variance` — représentant suggéré : `Fp2_amplitude` (meilleure AUC intra-enregistrement 0.914).
5. (3) `C4_amplitude`, `C4_std`, `C4_variance` — représentant suggéré : `C4_amplitude` (meilleure AUC intra-enregistrement 0.975).
6. (3) `O1_amplitude`, `O1_std`, `O1_variance` — représentant suggéré : `O1_amplitude` (meilleure AUC intra-enregistrement 0.962).
7. (3) `O2_amplitude`, `O2_std`, `O2_variance` — représentant suggéré : `O2_amplitude` (meilleure AUC intra-enregistrement 0.955).
8. (2) `Fp1_energy`, `Fp1_rms` — représentant suggéré : `Fp1_rms` (meilleure AUC intra-enregistrement 0.633).
9. (2) `Fp2_energy`, `Fp2_rms` — représentant suggéré : `Fp2_energy` (meilleure AUC intra-enregistrement 0.671).
10. (2) `C3_energy`, `C3_rms` — représentant suggéré : `C3_energy` (meilleure AUC intra-enregistrement 0.711).
11. (2) `C4_energy`, `C4_rms` — représentant suggéré : `C4_energy` (meilleure AUC intra-enregistrement 0.816).
12. (2) `T3_energy`, `T3_rms` — représentant suggéré : `T3_energy` (meilleure AUC intra-enregistrement 0.781).
13. (2) `T4_energy`, `T4_rms` — représentant suggéré : `T4_energy` (meilleure AUC intra-enregistrement 0.912).
14. (2) `O1_energy`, `O1_rms` — représentant suggéré : `O1_energy` (meilleure AUC intra-enregistrement 0.831).
15. (2) `O2_energy`, `O2_rms` — représentant suggéré : `O2_energy` (meilleure AUC intra-enregistrement 0.745).
16. (2) `avg_energy`, `avg_rms` — représentant suggéré : `avg_rms` (meilleure AUC intra-enregistrement 0.812).

Figures : corrélations moyennes [Spearman](../outputs/figures/official/part_vii/correlation_average_features_spearman.png) et [Pearson](../outputs/figures/official/part_vii/correlation_average_features.png); matrice complète [Spearman](../outputs/figures/official/part_vii/correlation_all_features_spearman.png) et [Pearson](../outputs/figures/official/part_vii/correlation_all_features.png). Tableaux : `outputs/feature_class_comparison_official.csv`, `outputs/high_correlations_official.csv` (Pearson), `outputs/high_correlations_spearman_official.csv`, `outputs/feature_correlations_official.csv` et `outputs/feature_correlations_spearman_official.csv`.
