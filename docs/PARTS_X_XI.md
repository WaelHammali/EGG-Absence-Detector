# Parties X–XI — Détection temporelle des crises et analyse des erreurs

Modèle : Random Forest (hyperparamètres par défaut, entraînement non équilibré) sur le jeu officiel filtré. Toutes les probabilités utilisées ici sont **hors-pli** : `GroupKFold` à 5 plis par enregistrement (mêmes plis qu’en Partie IX), donc chaque fenêtre est prédite par un modèle qui n’a jamais vu son enregistrement. Le modèle final, entraîné sur les 21 enregistrements, est sauvegardé dans `models/rf_final.joblib` avec la liste des 144 caractéristiques; il sert aux futurs enregistrements, pas à l’évaluation.

## Partie X — Des fenêtres aux intervalles

Règles (`src/detection.py`) :

1. une fenêtre est positive si sa probabilité ≥ seuil;
2. les fenêtres positives consécutives forment un intervalle, du début de la première à la fin de la dernière;
3. deux intervalles séparés de moins de 2 s sont fusionnés;
4. les événements de moins de 2 s sont supprimés. Une fenêtre isolée dure exactement 2 s : elle est donc conservée, et cette règle n’élimine rien avec des fenêtres de 2 s.

Comparaison avec les annotations, au niveau événement : une crise réelle est **trouvée (TP)** si au moins une détection la chevauche, sinon **manquée (FN)**; une détection qui ne chevauche aucune crise réelle est une **fausse alarme (FP)**. Recall = crises trouvées / crises réelles. Precision = détections chevauchant une crise / détections. Erreur de début = début détecté − début réel (négatif : détecté en avance); erreur de fin = fin détectée − fin réelle.

### Choix du seuil

Le seuil qui maximise le F1 événement sur les prédictions hors-pli est **0.68** (balayage de 0.05 à 0.95 par pas de 0,01) :

| Seuil | Détections | Crises trouvées | Manquées | Fausses alarmes | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 250 | 93 | 0 | 147 | 0.412 | 1.000 | 0.584 |
| 0.40 | 172 | 91 | 2 | 73 | 0.576 | 0.978 | 0.725 |
| 0.50 | 146 | 90 | 3 | 40 | 0.726 | 0.968 | 0.830 |
| 0.60 | 141 | 88 | 5 | 36 | 0.745 | 0.946 | 0.833 |
| **0.68** | 129 | 86 | 7 | 26 | 0.798 | 0.925 | **0.857** |
| 0.70 | 131 | 84 | 9 | 28 | 0.786 | 0.903 | 0.841 |
| 0.80 | 120 | 77 | 16 | 18 | 0.850 | 0.828 | 0.839 |
| 0.90 | 79 | 60 | 33 | 7 | 0.911 | 0.645 | 0.756 |

Le maximum est peu marqué : le F1 reste à moins de 0,02 du maximum pour tous les seuils de 0.64 à 0.85 (quelques événements d’écart sur 93 crises). Dans ce plateau, le seuil règle surtout le compromis : à 0,50, 90 crises trouvées et 40 fausses alarmes (F1 0.830); à 0.68, 86 trouvées et 26 fausses alarmes (F1 0.857). Le seuil est choisi et évalué sur les mêmes prédictions hors-pli : le F1 annoncé est donc légèrement optimiste. Figure : [threshold_sweep.png](../outputs/figures/official/detection/threshold_sweep.png).

### Résultats au seuil 0.68

- Crises réelles : 93; trouvées : **86**; manquées : **7**.
- Détections : 129; fausses alarmes : **26** (3.0 par heure d’enregistrement).
- Precision 0.798, recall 0.925, **F1 0.857**.
- Erreur de début : moyenne +1.14 s, médiane absolue 1.0 s. Erreur de fin : moyenne -1.19 s, médiane absolue 1.0 s.

La résolution temporelle est limitée par le pas de 1 s des fenêtres. Une fenêtre est étiquetée crise dès qu’elle contient 1 s de crise : un détecteur parfait commencerait donc jusqu’à 1 s avant le début réel et finirait jusqu’à 1 s après la fin réelle. En pratique, les détections commencent plutôt en retard et finissent plutôt en avance : les fenêtres de bord, à moitié normales, reçoivent une probabilité plus faible et passent sous ce seuil élevé.

| Enregistrement | Crises réelles | Trouvées | Manquées | Détections | Fausses alarmes |
|---|---:|---:|---:|---:|---:|
| 190304A_E | 1 | 0 | 1 | 0 | 0 |
| 191113A_D | 2 | 0 | 2 | 0 | 0 |
| 210204B_C | 6 | 6 | 0 | 6 | 0 |
| 210208B_G | 4 | 4 | 0 | 4 | 0 |
| 210406A_F | 1 | 1 | 0 | 1 | 0 |
| 210427B_C | 1 | 1 | 0 | 10 | 8 |
| 210504B_C | 12 | 12 | 0 | 20 | 8 |
| 210914B_A | 11 | 11 | 0 | 21 | 3 |
| 210928B_B | 2 | 2 | 0 | 3 | 0 |
| 211027B_C | 8 | 8 | 0 | 11 | 2 |
| 211104B_D | 5 | 5 | 0 | 5 | 0 |
| 220106A_A | 2 | 2 | 0 | 3 | 0 |
| 220121B_D | 3 | 3 | 0 | 4 | 1 |
| 230102B_F | 9 | 9 | 0 | 11 | 0 |
| 230105B_H | 7 | 7 | 0 | 7 | 0 |
| 230210B_D | 1 | 1 | 0 | 5 | 2 |
| 230224B_B | 3 | 3 | 0 | 4 | 0 |
| 230406B_A | 4 | 4 | 0 | 6 | 1 |
| 230515B_G | 2 | 2 | 0 | 2 | 0 |
| 230519A_E | 2 | 2 | 0 | 2 | 0 |
| 230718A_A | 7 | 3 | 4 | 4 | 1 |

### Q14 — Intervalles détectés et intervalles réels

Liste complète : `outputs/detected_intervals.csv` (une ligne par intervalle réel ou détecté) et `outputs/detected_intervals.txt`. Une figure par enregistrement (signal C3 filtré, crises réelles, détections, probabilité et seuil) : `outputs/figures/official/detection/<enregistrement>.png`. Extrait :

```text
210504B_C
  Crise détectée 1 : 1.0 s → 8.0 s
  Crise détectée 2 : 730.0 s → 747.0 s
  Crise détectée 3 : 1041.0 s → 1045.0 s
  Crise détectée 4 : 1126.0 s → 1130.0 s
  Crise détectée 5 : 1217.0 s → 1220.0 s
  Crise détectée 6 : 1281.0 s → 1283.0 s  (fausse alarme)
  Crise détectée 7 : 1344.0 s → 1346.0 s
  Crise détectée 8 : 1375.0 s → 1386.0 s
  Crise détectée 9 : 1460.0 s → 1473.0 s
  Crise détectée 10 : 1700.0 s → 1703.0 s
  Crise détectée 11 : 1860.0 s → 1862.0 s  (fausse alarme)
  Crise détectée 12 : 1873.0 s → 1875.0 s
  Crise détectée 13 : 1940.0 s → 1943.0 s  (fausse alarme)
  Crise détectée 14 : 1975.0 s → 1982.0 s
  Crise détectée 15 : 2249.0 s → 2269.0 s  (fausse alarme)
  Crise détectée 16 : 2274.0 s → 2276.0 s  (fausse alarme)
  Crise détectée 17 : 2400.0 s → 2411.0 s
  Crise détectée 18 : 2837.0 s → 2856.0 s  (fausse alarme)
  Crise détectée 19 : 2858.0 s → 2862.0 s  (fausse alarme)
  Crise détectée 20 : 3003.0 s → 3011.0 s  (fausse alarme)
  Crise réelle 1 : 0.0 s → 8.0 s
  Crise réelle 2 : 731.0 s → 747.0 s
  Crise réelle 3 : 1041.0 s → 1047.0 s
  Crise réelle 4 : 1126.0 s → 1129.0 s
  Crise réelle 5 : 1217.0 s → 1220.0 s
  Crise réelle 6 : 1344.0 s → 1346.0 s
  Crise réelle 7 : 1375.0 s → 1386.0 s
  Crise réelle 8 : 1460.0 s → 1472.0 s
  Crise réelle 9 : 1700.0 s → 1703.0 s
  Crise réelle 10 : 1873.0 s → 1876.0 s
  Crise réelle 11 : 1975.0 s → 1983.0 s
  Crise réelle 12 : 2399.0 s → 2459.0 s

190304A_E
  Aucune crise détectée
  Crise réelle 1 : 1291.0 s → 1298.0 s  (manquée)

210427B_C
  Crise détectée 1 : 277.0 s → 281.0 s  (fausse alarme)
  Crise détectée 2 : 287.0 s → 293.0 s  (fausse alarme)
  Crise détectée 3 : 327.0 s → 332.0 s  (fausse alarme)
  Crise détectée 4 : 345.0 s → 348.0 s  (fausse alarme)
  Crise détectée 5 : 817.0 s → 821.0 s  (fausse alarme)
  Crise détectée 6 : 825.0 s → 828.0 s  (fausse alarme)
  Crise détectée 7 : 832.0 s → 835.0 s  (fausse alarme)
  Crise détectée 8 : 890.0 s → 895.0 s  (fausse alarme)
  Crise détectée 9 : 1031.0 s → 1044.0 s
  Crise détectée 10 : 1047.0 s → 1050.0 s
  Crise réelle 1 : 1031.0 s → 1049.0 s

```

CLI : `python scripts/detect.py --recording 190304A_E [--threshold 0.5]`.

## Partie XI — Analyse des erreurs

### Comptages au seuil 0.68

| Niveau | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Fenêtres (2 s) | 658 | 114 | 586 | 30,061 |
| Événements | 86 | 26 | 7 | non défini |

Au niveau fenêtre : precision 0.852, recall 0.529, F1 0.653. Au niveau événement, les vrais négatifs ne sont pas définis : il n’existe pas de liste d’« événements non-crise » à compter. On rapporte à la place le nombre de fausses alarmes par heure. Le recall événement est bien plus élevé que le recall fenêtre : il suffit qu’une partie de la crise soit détectée pour la trouver, alors que les fenêtres de bord (mi-crise, mi-normal) sont souvent manquées.

### Exemples

Figure : [error_examples.png](../outputs/figures/official/detection/error_examples.png) — Fp1, C3, O1 filtrés et probabilité, ±10 s autour de l’événement.

- **TP — crise détectée** : 210504B_C, 1375.0 s → 1386.0 s.
- **FN — crise manquée** : 191113A_D, 1254.0 s → 1267.0 s.
- **FP — fausse alarme** : 211027B_C, 424.0 s → 439.0 s.
- **TN — activité normale** : 210504B_C, 1000.0 s → 1010.0 s.

### Crises manquées (FN) : 7

| Enregistrement | Âge | Début (s) | Fin (s) | Durée (s) | Probabilité max dans la crise | Notes du technicien à ±10 s |
|---|---:|---:|---:|---:|---:|---|
| 190304A_E | 17 | 1291.0 | 1298.0 | 7.0 | 0.34 | crise; b; Annotation |
| 191113A_D | 25 | 1254.0 | 1267.0 | 13.0 | 0.38 | HPN - 00:01:00; ARTEFACT; crise??; HPN - 00:01:20 |
| 191113A_D | 25 | 1428.0 | 1433.0 | 5.0 | 0.43 | Début d'export EEG-VIDEO; a; ABSNCE? NE répond pas à l appe; YO; b |
| 230718A_A | 7 | 218.0 | 222.0 | 4.0 | 0.50 | — |
| 230718A_A | 7 | 302.0 | 313.0 | 11.0 | 0.57 | — |
| 230718A_A | 7 | 430.0 | 441.0 | 11.0 | 0.63 | absence |
| 230718A_A | 7 | 1103.0 | 1110.0 | 7.0 | 0.61 | absence |

Causes probables :

- **Patients plus âgés, tracé différent.** 3 des 7 crises manquées viennent de `190304A_E` (17 ans) et `191113A_D` (25 ans), les deux seuls patients de plus de 15 ans; les 19 autres ont entre 5 et 15 ans. Aucune de leurs 3 crises n’est trouvée, même au seuil 0,50. Le modèle, évalué hors-pli, n’a alors vu que des enfants à l’entraînement. Les notes du technicien y sont en outre hésitantes (« crise?? », « ABSNCE? »).
- **Probabilité juste sous le seuil.** Les autres crises manquées ont une probabilité maximale proche du seuil : elles sont trouvées avec un seuil plus bas, au prix de fausses alarmes supplémentaires.
- **Amplitude plus faible.** Dans l’exemple manqué de `191113A_D`, la décharge est visible sur C3 mais avec une amplitude deux à trois fois plus faible que dans l’exemple de crise détectée, et elle se distingue à peine du fond sur O1; or les caractéristiques dominantes du modèle sont des mesures d’amplitude.

### Fausses alarmes (FP) : 26

- Durée : médiane 4 s; 12 sur 26 durent 3 s ou moins (une ou deux fenêtres).
- 5 sur 26 ont une note « absence / crise / clonies » du technicien à ±10 s.
- 1 sur 26 ont une note d’artefact, de saturation ou de mouvement à ±10 s.
- 15 sur 26 tombent pendant l’hyperpnée (HPN) ou dans la minute qui suit.
- Enregistrements les plus touchés : `210427B_C` (8), `210504B_C` (8), `210914B_A` (3).

| Enregistrement | Début (s) | Fin (s) | Probabilité max | Notes du technicien à ±10 s |
|---|---:|---:|---:|---|
| 211027B_C | 424.0 | 439.0 | 1.00 | SLI=3Hz; SLI=5Hz |
| 211027B_C | 834.0 | 844.0 | 1.00 | — |
| 210504B_C | 2837.0 | 2856.0 | 0.98 | HPN - 00:02:00; HPN - 00:02:20 |
| 210914B_A | 904.0 | 908.0 | 0.96 | — |
| 210914B_A | 916.0 | 919.0 | 0.94 | — |
| 210504B_C | 3003.0 | 3011.0 | 0.91 | — |
| 210427B_C | 832.0 | 835.0 | 0.88 | clonies palpébrales; absence; HPN - 00:00:40; b |
| 210427B_C | 277.0 | 281.0 | 0.87 | HPN - 00:00:40; absence |
| 210427B_C | 287.0 | 293.0 | 0.86 | HPN - 00:00:40; absence; HPN - 00:01:00 |
| 210427B_C | 345.0 | 348.0 | 0.86 | HPN - 00:01:40; HPN - 00:02:00 |
| 230210B_D | 398.0 | 402.0 | 0.86 | HPN - 00:02:20; HPN - 00:02:40 |
| 210504B_C | 1940.0 | 1943.0 | 0.86 | — |

Causes probables :

- **Crises probablement réelles mais non annotées.** Dans `210427B_C`, le fichier Excel ne contient qu’une crise (1031–1049 s), mais le tracé montre cinq bouffées de même aspect et de même amplitude (vers 280–293, 327–348, 817–835, 890–901 et 1031–1049 s; voir [210427B_C.png](../outputs/figures/official/detection/210427B_C.png)). Le technicien a noté « absence » à 290 s, 821 s et 831 s, et le modèle, qui n’a jamais vu cet enregistrement, détecte les cinq. Ces 8 « fausses alarmes » sont vraisemblablement de vraies crises absentes du fichier : la precision réelle est sous-estimée. À confirmer avec le professeur.
- **Décharges pendant les épreuves d’activation.** La fausse alarme la plus probable (`211027B_C`, 424–439 s, probabilité 1,00, pendant la stimulation lumineuse SLI) montre 15 s de décharge rythmique ample sur Fp1, C3 et O1 (figure des exemples). 15 fausses alarmes sur 26 surviennent pendant l’hyperpnée ou juste après. SLI et hyperpnée sont justement utilisées pour provoquer des décharges : une partie de ces événements peut être de l’activité pointe-onde que l’annotateur n’a pas retenue comme crise clinique. L’hyperpnée provoque aussi un ralentissement ample du tracé, proche en amplitude d’une crise.
- **Fragments d’une ou deux fenêtres.** 12 fausses alarmes sur 26 durent 3 s ou moins. Une durée minimale supérieure à 2 s les supprimerait, au risque de manquer les crises très courtes.
- **Artefacts et mouvements : rôle limité ici.** Seulement 1 fausse(s) alarme(s) sur 26 se trouve(nt) près d’une note d’artefact ou de mouvement.

Ces causes sont des hypothèses tirées des notes du technicien et de l’aspect du tracé; elles ne remplacent pas une relecture par un neurologue.

Résultats : `outputs/detection_summary.json`, `outputs/detection_threshold_sweep.csv`, `outputs/event_matches.csv`. Reproduction : `python scripts/predict_oof.py`, `python scripts/prepare_app_data.py`, `python scripts/evaluate_detection.py`.
