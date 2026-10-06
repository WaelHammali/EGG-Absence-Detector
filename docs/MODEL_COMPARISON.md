# Comparaison des 8 modèles

8 modèles = 4 algorithmes (Decision Tree, Random Forest, KNN, SVM) × 2 modes d’entraînement, sur le jeu officiel filtré à 19 canaux EEG (320 caractéristiques). Hyperparamètres par défaut de scikit-learn; `StandardScaler` pour KNN et SVM.

- **Non équilibré** : chaque pli d’entraînement est utilisé tel quel.
- **Équilibré** : sous-échantillonnage aléatoire du pli d’entraînement seulement : toutes les fenêtres de classe 1 + autant de fenêtres de classe 0 tirées au hasard (50/50). Répété avec 5 graines (0, 1, 2, 3, 4); la graine 0 sert aux prédictions et aux modèles sauvegardés.
- **Test** : toujours les proportions réelles. `GroupKFold` à 5 plis par enregistrement (mêmes plis partout) : chaque fenêtre est prédite par un modèle qui n’a jamais vu son enregistrement.
- **Seuil** : pour chaque modèle, le seuil qui maximise le F1 événement sur ses prédictions hors-pli; les résultats à 0,50 sont donnés aussi. Ce seuil est choisi et évalué sur les mêmes prédictions, donc les scores au meilleur seuil sont légèrement optimistes.

Modèle par défaut (meilleur F1 événement) : **SVM, non équilibré**.

## Entraînement : distribution des classes

| Mode | Fenêtres d’entraînement par pli (moyenne) | Classe 0 | Classe 1 | Part de classe 1 |
|---|---:|---:|---:|---:|
| Non équilibré | 25,135 | 24,140 | 995 | 4.0% |
| Équilibré | 1,990 | 995 | 995 | 50.0% |

## Les 8 modèles au meilleur seuil de chacun

| Algorithme | Mode | Seuil | Fenêtres : precision | recall | F1 | Événements : F1 | recall | precision | Trouvées / manquées | Fausses alarmes | FA / heure | Erreur début (s) | Erreur fin (s) | Temps d’entraînement (s/pli) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | non équilibré | 0.05 | 0.429 | 0.590 | 0.497 | 0.290 | 0.978 | 0.170 | 91 / 2 | 561 | 64.2 | +0.43 | -0.09 | 23.49 |
| Decision Tree | équilibré | 0.05 | 0.191 | 0.825 | 0.311 | 0.148 | 0.989 | 0.080 | 92 / 1 | 1229 | 140.7 | -5.28 | +6.00 | 0.93 |
| Random Forest | non équilibré | 0.67 | 0.844 | 0.518 | 0.642 | 0.867 | 0.935 | 0.807 | 87 / 6 | 26 | 3.0 | +0.90 | -1.46 | 5.68 |
| Random Forest | équilibré | 0.99 | 0.778 | 0.343 | 0.476 | 0.785 | 0.860 | 0.721 | 80 / 13 | 39 | 4.5 | +1.94 | -2.03 | 0.32 |
| KNN | non équilibré | 0.61 | 0.848 | 0.420 | 0.562 | 0.828 | 0.925 | 0.750 | 86 / 7 | 37 | 4.2 | +1.21 | -2.08 | 0.13 |
| KNN | équilibré | 0.81 | 0.609 | 0.699 | 0.650 | 0.519 | 0.968 | 0.355 | 90 / 3 | 189 | 21.6 | +0.10 | +0.12 | 0.01 |
| SVM | non équilibré | 0.82 | 0.891 | 0.421 | 0.572 | **0.894** | 0.925 | 0.866 | 86 / 7 | 16 | 1.8 | +1.64 | -1.99 | 49.01 |
| SVM | équilibré | 0.99 | 0.741 | 0.314 | 0.441 | 0.821 | 0.860 | 0.785 | 80 / 13 | 29 | 3.3 | +1.49 | -1.57 | 0.81 |

Fenêtres : fenêtres de 2 s. Événements : une crise est trouvée si au moins une détection la chevauche; une fausse alarme est une détection hors de toute crise annotée. Erreur de début / de fin : détecté − réel, moyenne sur les crises trouvées (positif = en retard). Temps d’entraînement : ajustement du modèle sur un pli; le mode équilibré est plus rapide car il s’entraîne sur environ 12 fois moins de fenêtres. Pour KNN, l’ajustement est presque instantané : son coût est à la prédiction, non mesurée ici.

## Les 8 modèles au seuil 0,50

| Algorithme | Mode | Seuil | Fenêtres : precision | recall | F1 | Événements : F1 | recall | precision | Trouvées / manquées | Fausses alarmes | FA / heure | Erreur début (s) | Erreur fin (s) | Temps d’entraînement (s/pli) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | non équilibré | 0.50 | 0.429 | 0.590 | 0.497 | 0.290 | 0.978 | 0.170 | 91 / 2 | 561 | 64.2 | +0.43 | -0.09 | 23.49 |
| Decision Tree | équilibré | 0.50 | 0.191 | 0.825 | 0.311 | 0.148 | 0.989 | 0.080 | 92 / 1 | 1229 | 140.7 | -5.28 | +6.00 | 0.93 |
| Random Forest | non équilibré | 0.50 | 0.793 | 0.654 | 0.717 | 0.832 | 0.978 | 0.723 | 91 / 2 | 41 | 4.7 | +0.54 | -0.54 | 5.68 |
| Random Forest | équilibré | 0.50 | 0.304 | 0.872 | 0.451 | 0.274 | 1.000 | 0.159 | 93 / 0 | 524 | 60.0 | -2.49 | +5.56 | 0.32 |
| KNN | non équilibré | 0.50 | 0.787 | 0.556 | 0.652 | 0.754 | 0.968 | 0.618 | 90 / 3 | 68 | 7.8 | +0.71 | -1.12 | 0.13 |
| KNN | équilibré | 0.50 | 0.324 | 0.824 | 0.465 | 0.224 | 0.989 | 0.126 | 92 / 1 | 679 | 77.7 | -1.48 | +1.89 | 0.01 |
| SVM | non équilibré | 0.50 | 0.833 | 0.588 | 0.689 | **0.857** | 0.957 | 0.775 | 89 / 4 | 31 | 3.5 | +0.73 | -0.88 | 49.01 |
| SVM | équilibré | 0.50 | 0.300 | 0.858 | 0.444 | 0.285 | 0.989 | 0.167 | 92 / 1 | 495 | 56.7 | -2.18 | +3.98 | 0.81 |

L’arbre de décision ne produit que des probabilités 0 ou 1 : son résultat ne dépend pas du seuil.

## Mode équilibré : variabilité sur 5 graines

Moyenne ± écart-type selon le tirage des fenêtres de classe 0.

| Algorithme | Meilleur seuil | F1 événement (meilleur seuil) | Recall événement | Precision événement | Fausses alarmes | F1 fenêtre (meilleur seuil) | F1 événement à 0,50 | F1 fenêtre à 0,50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.05 ± 0.00 | 0.130 ± 0.011 | 0.996 ± 0.006 | 0.069 ± 0.007 | 1377.0 ± 102.6 | 0.295 ± 0.012 | 0.130 ± 0.011 | 0.295 ± 0.012 |
| Random Forest | 0.99 ± 0.01 | 0.808 ± 0.033 | 0.886 ± 0.045 | 0.743 ± 0.034 | 35.8 ± 5.1 | 0.529 ± 0.058 | 0.246 ± 0.024 | 0.447 ± 0.015 |
| KNN | 0.81 ± 0.00 | 0.510 ± 0.018 | 0.974 ± 0.006 | 0.346 ± 0.016 | 196.6 ± 11.3 | 0.650 ± 0.006 | 0.214 ± 0.013 | 0.435 ± 0.019 |
| SVM | 0.99 ± 0.00 | 0.797 ± 0.017 | 0.886 ± 0.033 | 0.727 ± 0.042 | 42.2 ± 11.0 | 0.504 ± 0.044 | 0.260 ± 0.015 | 0.409 ± 0.027 |

## Fausses alarmes : coïncidence avec SLI, EMG et hyperpnée

ECG, EMG et SLI ne sont jamais utilisés comme caractéristiques. Ils servent ici, avec les notes du technicien, à interpréter les fausses alarmes de chaque modèle à son meilleur seuil.

- **Hyperpnée (HPN)** : périodes reconstruites à partir des notes « HPN » (notes espacées de moins de 60 s, plus 60 s après la dernière). Elles couvrent 27.8% du temps enregistré.
- **SLI (notes)** : chaque note « SLI=x Hz » ouvre un palier jusqu’à la note suivante (20 s au plus). Les paliers couvrent 14.8% du temps enregistré. « 2–4 Hz » : palier dont la fréquence est entre 2 et 4 Hz.
- **SLI (canal)** : le canal SLI est un marqueur d’impulsions. Il n’est exploitable que dans 9 enregistrements sur 21 (constant dans les autres); la colonne compte les fausses alarmes de ces enregistrements pendant lesquelles il pulse.
- **Bouffée EMG** : RMS de l’EMG pendant l’événement ≥ 3 fois sa valeur typique. L’EMG n’existe que dans 6 enregistrements; la colonne ne porte que sur eux.
- **Note de crise** : note « absence », « crise » ou « clonies » du technicien à ±10 s.

| Modèle | Fausses alarmes | Pendant HPN | Pendant SLI (notes) | dont SLI 2–4 Hz | SLI actif (canal) | Bouffée EMG | Note de crise à ±10 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree, non équilibré | 561 | 175/561 (31%) | 90/561 (16%) | 11/561 (2%) | 47/316 (15%) | 1/195 (1%) | 10/561 (2%) |
| Decision Tree, équilibré | 1229 | 354/1229 (29%) | 194/1229 (16%) | 21/1229 (2%) | 80/594 (13%) | 1/340 (0%) | 6/1229 (0%) |
| Random Forest, non équilibré | 26 | 12/26 (46%) | 2/26 (8%) | 1/26 (4%) | 1/16 (6%) | 0/3 (0%) | 5/26 (19%) |
| Random Forest, équilibré | 39 | 17/39 (44%) | 5/39 (13%) | 1/39 (3%) | 3/31 (10%) | 0/1 (0%) | 5/39 (13%) |
| KNN, non équilibré | 37 | 10/37 (27%) | 3/37 (8%) | 2/37 (5%) | 1/14 (7%) | 0/4 (0%) | 2/37 (5%) |
| KNN, équilibré | 189 | 55/189 (29%) | 19/189 (10%) | 5/189 (3%) | 5/94 (5%) | 1/28 (4%) | 3/189 (2%) |
| SVM, non équilibré | 16 | 7/16 (44%) | 1/16 (6%) | 1/16 (6%) | 3/9 (33%) | 0/1 (0%) | 2/16 (12%) |
| SVM, équilibré | 29 | 12/29 (41%) | 5/29 (17%) | 2/29 (7%) | 2/14 (14%) | 0/1 (0%) | 2/29 (7%) |
| *Crises réelles (référence)* | 93 | 37/93 (40%) | 10/93 (11%) | 1/93 (1%) | 4/31 (13%) | 0/15 (0%) | 23/93 (25%) |

Lecture pour le modèle par défaut (SVM, non équilibré) : 7/16 (44%) de ses fausses alarmes tombent pendant l’hyperpnée, qui ne représente que 28% du temps; 1/16 (6%) pendant la SLI (15% du temps), dont 1 pendant un palier à 2–4 Hz; bouffée EMG : 0/1 (0%). Les crises réelles suivent la même tendance : 37/93 (40%) surviennent pendant l’hyperpnée, qui sert justement à les provoquer. Une fausse alarme pendant HPN ou SLI peut donc être une décharge réelle non annotée aussi bien qu’une erreur du modèle; ces chiffres ne permettent pas de trancher.

## Conclusion

- **Meilleur modèle** : SVM non équilibré (F1 événement 0.894, 86 crises trouvées sur 93, 16 fausses alarmes), devant Random Forest non équilibré (0.867). L’écart entre les premiers correspond à quelques événements : il n’est pas assez grand pour les départager avec certitude.
- **Effet de l’équilibrage au meilleur seuil** (non équilibré → équilibré) : Decision Tree : F1 événement 0.290 → 0.148, fausses alarmes 561 → 1229, seuil 0.05 → 0.05; Random Forest : F1 événement 0.867 → 0.785, fausses alarmes 26 → 39, seuil 0.67 → 0.99; KNN : F1 événement 0.828 → 0.519, fausses alarmes 37 → 189, seuil 0.61 → 0.81; SVM : F1 événement 0.894 → 0.821, fausses alarmes 16 → 29, seuil 0.82 → 0.99.
- **Au seuil fixe 0,50**, l’équilibrage change surtout le compromis : recall fenêtre +0.248 en moyenne, precision fenêtre -0.431, et 12.1 fois plus de fausses alarmes pour Random Forest, KNN et SVM réunis. Un modèle entraîné à 50 % de crises surestime la probabilité de crise quand il est testé sur 4 %.
- **Le seuil ne compense qu’en partie l’équilibrage** : les modèles équilibrés demandent un seuil bien plus élevé (Random Forest 0.99, KNN 0.81, SVM 0.99) et restent en dessous de leur version non équilibrée. Pour Random Forest et SVM, le meilleur seuil atteint la limite haute explorée (0,99) : presque toutes leurs probabilités utiles sont comprimées près de 1, ce qui rend le réglage fragile. Ici, entraîner sur les proportions réelles et choisir le seuil donne de meilleurs résultats que rééquilibrer.
- **Arbre de décision** : inutilisable pour la détection temporelle dans les deux modes (561 et 1229 fausses alarmes), car il ne donne que 0 ou 1.
- **Limites** : 21 enregistrements, 93 crises, hyperparamètres par défaut, identité des patients non confirmée. Les différences de 0,01 à 0,02 en F1 événement sont dans la variabilité entre plis et entre graines.

Reproduction : `python scripts/train_registry.py`, `python scripts/prepare_app_data.py`, `python scripts/compare_models.py`. Détails par modèle : `models/registry.json`.
