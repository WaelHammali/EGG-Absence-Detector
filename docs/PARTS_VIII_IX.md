# Parties VIII–IX — Modèles de classification

Jeu de données : `data/processed/windows_features_official.parquet` (annotations officielles, signal filtré passe-bande 0,5–40 Hz, 320 caractéristiques sur 19 canaux EEG). Hyperparamètres par défaut de scikit-learn, sans réglage (prévu en Partie X). Arbre de décision et Random Forest sans normalisation; KNN et SVM avec `StandardScaler` dans le `Pipeline`, donc ajusté sur les seules données d’entraînement. `random_state=42` partout.

## Protocole

1. Split stratifié 80/20 : 25,135 fenêtres d’entraînement (995 crises) et 6,284 fenêtres de test (249 crises, 3.96%).
2. Sous-échantillonnage aléatoire du **train uniquement** : les 995 fenêtres de classe 1 + 995 fenêtres de classe 0 tirées au hasard (1,990 fenêtres). Le test garde les proportions réelles : un test équilibré surestimerait la precision.
3. Comparaison : mêmes modèles entraînés sur le train complet non équilibré.
4. Évaluation honnête : `GroupKFold` à 5 plis par enregistrement, sous-échantillonnage tiré dans chaque pli d’entraînement seulement. La même validation avec un train non équilibré est ajoutée comme référence.

Precision, Recall et F1 concernent la classe 1 (crise). La balanced accuracy (moyenne des rappels des deux classes) est ajoutée pour Q13.

## Split aléatoire — train sous-échantillonné

| Modèle | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Toujours 0 | 0.960 | 0.000 | 0.000 | 0.000 | 0.500 |
| Decision Tree | 0.871 | 0.221 | 0.896 | 0.355 | 0.883 |
| Random Forest | 0.950 | 0.436 | 0.924 | 0.593 | 0.937 |
| KNN | 0.947 | 0.420 | 0.867 | 0.566 | 0.909 |
| SVM | 0.954 | 0.460 | 0.880 | 0.604 | 0.918 |

| Modèle | tn | fp | fn | tp |
|---|---:|---:|---:|---:|
| Toujours 0 | 6,035 | 0 | 249 | 0 |
| Decision Tree | 5,250 | 785 | 26 | 223 |
| Random Forest | 5,738 | 297 | 19 | 230 |
| KNN | 5,737 | 298 | 33 | 216 |
| SVM | 5,778 | 257 | 30 | 219 |

## Split aléatoire — train non équilibré (comparaison)

| Modèle | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Toujours 0 | 0.960 | 0.000 | 0.000 | 0.000 | 0.500 |
| Decision Tree | 0.974 | 0.671 | 0.655 | 0.663 | 0.821 |
| Random Forest | 0.985 | 0.871 | 0.731 | 0.795 | 0.863 |
| KNN | 0.982 | 0.843 | 0.671 | 0.747 | 0.833 |
| SVM | 0.983 | 0.867 | 0.683 | 0.764 | 0.839 |

| Modèle | tn | fp | fn | tp |
|---|---:|---:|---:|---:|
| Toujours 0 | 6,035 | 0 | 249 | 0 |
| Decision Tree | 5,955 | 80 | 86 | 163 |
| Random Forest | 6,008 | 27 | 67 | 182 |
| KNN | 6,004 | 31 | 82 | 167 |
| SVM | 6,009 | 26 | 79 | 170 |

tn/fp/fn/tp : vrais négatifs, faux positifs (fausses alarmes), faux négatifs (crises manquées), vrais positifs. Matrices de confusion : [split aléatoire](../outputs/figures/official/models/confusion_matrices_random_split.png).

## Évaluation par enregistrement — GroupKFold (5 plis)

### Train sous-échantillonné (protocole demandé)

| Modèle | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Toujours 0 | 0.960 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Decision Tree | 0.826 ± 0.069 | 0.179 ± 0.076 | 0.836 ± 0.062 | 0.287 ± 0.099 | 0.831 ± 0.032 |
| Random Forest | 0.923 ± 0.044 | 0.356 ± 0.139 | 0.874 ± 0.077 | 0.489 ± 0.155 | 0.899 ± 0.040 |
| KNN | 0.927 ± 0.026 | 0.347 ± 0.137 | 0.824 ± 0.074 | 0.471 ± 0.127 | 0.878 ± 0.026 |
| SVM | 0.923 ± 0.039 | 0.359 ± 0.190 | 0.857 ± 0.065 | 0.479 ± 0.173 | 0.891 ± 0.023 |

### Train non équilibré (référence)

| Modèle | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Toujours 0 | 0.960 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Decision Tree | 0.953 ± 0.015 | 0.446 ± 0.137 | 0.589 ± 0.033 | 0.497 ± 0.108 | 0.779 ± 0.020 |
| Random Forest | 0.979 ± 0.008 | 0.792 ± 0.126 | 0.663 ± 0.067 | 0.714 ± 0.044 | 0.828 ± 0.033 |
| KNN | 0.976 ± 0.008 | 0.776 ± 0.073 | 0.565 ± 0.113 | 0.645 ± 0.070 | 0.779 ± 0.056 |
| SVM | 0.979 ± 0.008 | 0.813 ± 0.110 | 0.637 ± 0.059 | 0.709 ± 0.038 | 0.816 ± 0.029 |

| Pli | Enregistrements de test | Fenêtres | Crises |
|---:|---|---:|---:|
| 1 | 230102B_F, 230105B_H, 230718A_A | 6,028 | 263 |
| 2 | 210504B_C, 230406B_A, 230515B_G | 5,550 | 230 |
| 3 | 191113A_D, 210204B_C, 211104B_D, 220121B_D, 230210B_D | 6,707 | 168 |
| 4 | 190304A_E, 210208B_G, 210427B_C, 210928B_B, 230519A_E | 6,576 | 168 |
| 5 | 210406A_F, 210914B_A, 211027B_C, 220106A_A, 230224B_B | 6,558 | 415 |

F1 par pli (train sous-échantillonné) :

| Modèle | Pli 1 | Pli 2 | Pli 3 | Pli 4 | Pli 5 |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.233 | 0.392 | 0.156 | 0.277 | 0.377 |
| Random Forest | 0.528 | 0.553 | 0.224 | 0.513 | 0.629 |
| KNN | 0.434 | 0.533 | 0.306 | 0.438 | 0.646 |
| SVM | 0.377 | 0.609 | 0.301 | 0.394 | 0.711 |

Matrices de confusion cumulées sur les 5 plis : [GroupKFold](../outputs/figures/official/models/confusion_matrices_groupkfold.png). Synthèse des quatre configurations : [metrics_by_setting.png](../outputs/figures/official/models/metrics_by_setting.png).

### Split aléatoire vs GroupKFold

| Modèle | F1 aléatoire (équilibré) | F1 GroupKFold (équilibré) | Écart | F1 aléatoire (non équilibré) | F1 GroupKFold (non équilibré) | Écart |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.355 | 0.287 | +0.068 | 0.663 | 0.497 | +0.166 |
| Random Forest | 0.593 | 0.489 | +0.103 | 0.795 | 0.714 | +0.080 |
| KNN | 0.566 | 0.471 | +0.095 | 0.747 | 0.645 | +0.102 |
| SVM | 0.604 | 0.479 | +0.126 | 0.764 | 0.709 | +0.055 |

Le split aléatoire est optimiste dans les deux cas : F1 supérieur de +0.098 en moyenne avec le train équilibré, et de +0.101 avec le train non équilibré. Deux raisons :

- **Fuite entre fenêtres voisines.** Avec 50 % de recouvrement, deux fenêtres consécutives partagent une seconde de signal, et les fenêtres d’une même crise se ressemblent. Un split aléatoire place ces quasi-copies à la fois dans le train et dans le test.
- **Spécificités de l’enregistrement.** Niveau d’amplitude de fond, électrodes, bruit : le modèle apprend aussi ce qui caractérise chaque enregistrement, ce qui ne se généralise pas à un nouveau patient.

Hors arbre de décision, l’écart est du même ordre avec ou sans équilibrage (+0.055 à +0.126). Il est le plus fort pour l’arbre de décision non équilibré (+0.166) : un arbre unique et profond mémorise les fenêtres voisines, donc profite le plus de la fuite. GroupKFold teste sur des enregistrements jamais vus, ce qui correspond à l’usage réel. Les identifiants patients n’étant pas confirmés, un même patient peut toutefois apparaître dans deux enregistrements : même GroupKFold peut rester légèrement optimiste.

La variabilité entre plis est l’autre enseignement : le F1 moyen des quatre modèles va de 0.246 (pli 3 : 191113A_D, 210204B_C, 211104B_D, 220121B_D, 230210B_D) à 0.591 (pli 5 : 210406A_F, 210914B_A, 211027B_C, 220106A_A, 230224B_B). Un seul split aléatoire ne montre pas cette dépendance aux enregistrements testés.

## Q12 — Pourquoi l’accuracy est trompeuse

La classe 1 ne représente que 3.96% des fenêtres de test. Le modèle « Toujours 0 », qui ne détecte aucune crise, obtient une accuracy de **0.960**, avec un recall, une precision et un F1 de 0. Il dépasse même en accuracy l’arbre de décision entraîné sur le train équilibré (0.871), qui détecte pourtant 90% des crises. L’accuracy est dominée par la classe majoritaire et ne dit rien des crises manquées.

## Q13 — Quelle métrique privilégier

- **Recall (sensibilité)** : part des fenêtres de crise détectées. C’est la priorité clinique : une crise manquée est l’erreur la plus coûteuse.
- **Precision** : part des alarmes qui sont de vraies crises. Elle mesure le coût des fausses alarmes, nombreuses dès que la classe 0 domine.
- **F1** : moyenne harmonique des deux, nulle si l’un est nul. C’est la métrique principale pour comparer les modèles, toujours accompagnée du recall et de la precision.
- La **balanced accuracy** corrige aussi le déséquilibre (0.500 pour « Toujours 0 », soit le hasard), mais elle est peu sensible aux fausses alarmes : quelques centaines de faux positifs pèsent peu face à plus de 6 000 vraies fenêtres de classe 0.

En évaluation par enregistrement, le meilleur F1 est obtenu par **Random Forest** avec le train équilibré (0.489 ± 0.155) et par **Random Forest** avec le train non équilibré (0.714 ± 0.044).

## Effet de l’équilibrage (compromis recall / precision)

Split aléatoire :

| Modèle | Recall non équilibré | Recall équilibré | Δ recall | Precision non équilibrée | Precision équilibrée | Δ precision | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.655 | 0.896 | +0.241 | 0.671 | 0.221 | -0.450 | -0.308 |
| Random Forest | 0.731 | 0.924 | +0.193 | 0.871 | 0.436 | -0.434 | -0.202 |
| KNN | 0.671 | 0.867 | +0.197 | 0.843 | 0.420 | -0.423 | -0.181 |
| SVM | 0.683 | 0.880 | +0.197 | 0.867 | 0.460 | -0.407 | -0.160 |

Le sous-échantillonnage fait passer la classe 1 de 3.96% à 50 % du train. Le modèle apprend une frontière plus favorable aux crises : le recall augmente en moyenne de +0.207, mais la precision chute de -0.429, et le F1 baisse de -0.213. La cause : le modèle s’entraîne avec 50 % de crises mais est testé avec 4%. Un faible taux de fausses alarmes sur la classe 0, très nombreuse, produit alors davantage de faux positifs que de vrais positifs pour chacun des quatre modèles. Le sous-échantillonnage jette aussi 92% des fenêtres d’entraînement, et donc de l’information sur la variabilité de l’activité normale.

En GroupKFold, la même tendance se confirme : Δ recall moyen +0.234, Δ F1 moyen -0.210.

Le choix dépend donc du coût des erreurs. Pour un outil d’aide à la lecture d’EEG, manquer une crise est généralement plus grave qu’une fausse alarme qu’un neurologue peut écarter : c’est l’argument pour l’équilibrage. Si l’on optimise le F1, le train non équilibré est meilleur ici. Le seuil de décision peut aussi être déplacé après l’entraînement pour choisir un compromis recall / precision; ce réglage relève de la Partie X.

Résultats bruts : `outputs/models_random_split.csv`, `outputs/models_groupkfold_folds.csv`, `outputs/models_groupkfold_summary.csv` et `outputs/models_summary.json`. Reproduction : `python scripts/train_models.py`.
