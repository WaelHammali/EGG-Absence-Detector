# DATA_NOTES — Inspection des données (Étape 0)

Inspection réalisée le 4 octobre 2026. Aucun fichier de `data/raw/` n’a été modifié. La conversion, la création des labels et l’apprentissage attendent la confirmation demandée.

## 1. Méthode et inventaire

- 22 enregistrements, 22 paires `*_0000d.mat` / `*_0000h.mat`, `Annotations.xlsx` et deux scripts Octave. Une arborescence `.venv` existe également dans `data/raw/` : elle est ignorée et conservée.
- Le script d’inspection fourni a été exécuté : il échouait sur un objet MCOS lors du calcul de `nanmin`. Il a été adapté pour décoder les timetables et auditer les 22 paires une par une.
- Reproduction : `.venv/bin/python scripts/inspect_data.py`. Résultats détaillés : [inspection_summary.json](inspection_summary.json), comprenant les premières valeurs et les statistiques de chaque canal, les événements des headers et les cellules d’annotation utiles.
- Lecture essayée dans l’ordre demandé : SciPy ouvre le conteneur MATLAB mais renvoie `MatlabOpaque`; le premier d-file et le premier h-file ne sont pas HDF5 (v7.3). `mat-io==1.0.0` décode correctement les 44 timetables. Cette dépendance a été ajoutée à `requirements.txt`.

## 2. Structure réelle des signaux et des headers

Variables applicatives : **`output_d`** pour chaque signal, **`output_h`** pour chaque header. Le premier couple est un MAT-file MATLAB 5.0, plateforme PCWIN64. SciPy expose aussi `__header__`, `__version__`, `__globals__` et `__function_workspace__`; le dernier contient les données internes MCOS et ne représente pas un canal. Aucune variable applicative `None` n’a été trouvée.

`output_d` est un **timetable de blocs**, pas directement une matrice échantillons × canaux. Son index `RecordTime` est un `TimedeltaIndex`, partant de 0 s avec un pas exactement égal à 1 s. Chaque cellule de canal contient un tableau numérique **(256, 1)** en `float64`. Tous les blocs ont 256 valeurs, tous les canaux ont le même nombre d’échantillons et tous les index sont réguliers sur les 22 enregistrements.

Orientation après dépliage : **échantillons × canaux**. Pour le bloc b et son échantillon k, `Time = RecordTime[b] + k / 256`. Fréquence mesurée : **256 Hz pour tous les enregistrements**, déduite des 256 échantillons par seconde; elle n’est pas déclarée dans un champ de header. La durée de couverture est N / 256; le dernier échantillon est à durée − 1/256 s. Une fenêtre de 2 s contiendra donc 512 échantillons, avec un pas de 256 à 50 % de recouvrement.

**Exemple `190304A_E` :** timetable (1575, 23), index 0 à 1574 s, matrice dépliée (403200, 23), couverture 1575 s, dernier échantillon 1574.99609375 s.

`output_h` est un timetable d’événements : index `Onset` en secondes depuis le début, colonnes `Annotations` (texte) et `Duration` (durée). Le premier header a 77 lignes; les headers ont entre 7 et 104 événements. **Toutes les valeurs `Duration` sont NaT dans les 22 headers.** Les textes indiquent notamment impédances, HPN, SLI, artefacts, mouvements et parfois « absence » ou « crise ». Ce ne sont pas des métadonnées complètes d’acquisition et ils ne fournissent pas systématiquement les deux bornes d’une crise. Certains accents sont décodés en caractères de remplacement.

**Unités : non renseignées.** Les attributs `varUnits` sont vides sur tous les signaux. Sur le premier couple, `varDescriptions`, `UserData` et `Description` sont également vides. Il ne faut pas présenter l’amplitude comme µV sans confirmation de la source. Jusqu’à confirmation, les figures doivent indiquer « amplitude (unité non renseignée) ». Les valeurs d’impédance en k présentes dans les commentaires ne sont pas l’unité du signal.

## 3. Canaux, durées et valeurs manquantes

Montage complet : 19 canaux EEG (`EEGFp1`, `EEGFp2`, `EEGF7`, `EEGF3`, `EEGFz`, `EEGF4`, `EEGF8`, `EEGT3`, `EEGC3`, `EEGCz`, `EEGC4`, `EEGT4`, `EEGT5`, `EEGP3`, `EEGPz`, `EEGP4`, `EEGT6`, `EEGO1`, `EEGO2`) plus `SLI`, `ECG` et, pour certains enregistrements, deux canaux EMG. Les EMG se nomment `EMG1` / `EMG2` ou `EMG_1_` / `EMG_2`. L’ordre des colonnes varie : l’association doit utiliser les noms, jamais la position.

`200625A_F` a seulement 8 EEG : `EEGFp1`, `EEGFp2`, `EEGC3`, `EEGC4`, `EEGO1`, `EEGO2`, `EEGT3`, `EEGT4`, plus `EMG1`, `EMG2`, `SLI`, `ECG`. Ces huit EEG constituent l’intersection commune aux 22 enregistrements. Les autres ont tous les 19 EEG.

| Enregistrement | Blocs (s) | Échantillons | EEG / total canaux | Durée (s) | SLI constant | Paires de crises renseignées |
|---|---:|---:|---:|---:|---|---:|
| 190304A_E | 1575 | 403200 | 19 / 23 | 1575 | non | 1 |
| 191113A_D | 1810 | 463360 | 19 / 23 | 1810 | non | 2 |
| 200625A_F | 1788 | 457728 | 8 / 12 | 1788 | non | 0 |
| 210204B_C | 1227 | 314112 | 19 / 21 | 1227 | non | 6 |
| 210208B_G | 1200 | 307200 | 19 / 21 | 1200 | non | 4 |
| 210406A_F | 1454 | 372224 | 19 / 23 | 1454 | non | 1 |
| 210427B_C | 1246 | 318976 | 19 / 21 | 1246 | non | 1 |
| 210504B_C | 3121 | 798976 | 19 / 21 | 3121 | non | 12 |
| 210914B_A | 1205 | 308480 | 19 / 21 | 1205 | oui | 11 |
| 210928B_B | 1344 | 344064 | 19 / 21 | 1344 | oui | 2 |
| 211027B_C | 1255 | 321280 | 19 / 21 | 1255 | oui | 8 |
| 211104B_D | 1210 | 309760 | 19 / 21 | 1210 | oui | 5 |
| 220106A_A | 1221 | 312576 | 19 / 23 | 1221 | non | 2 |
| 220121B_D | 1292 | 330752 | 19 / 21 | 1292 | oui | 3 |
| 230102B_F | 1225 | 313600 | 19 / 21 | 1225 | oui | 9 |
| 230105B_H | 3600 | 921600 | 19 / 21 | 3600 | oui | 7 |
| 230210B_D | 1173 | 300288 | 19 / 21 | 1173 | oui | 1 |
| 230224B_B | 1428 | 365568 | 19 / 21 | 1428 | oui | 3 |
| 230406B_A | 1225 | 313600 | 19 / 21 | 1225 | oui | 4 |
| 230515B_G | 1207 | 308992 | 19 / 21 | 1207 | oui | 2/2 |
| 230519A_E | 1216 | 311296 | 19 / 23 | 1216 | non | 2 |
| 230718A_A | 1206 | 308736 | 19 / 23 | 1206 | oui | 7 |

**Total : 8 506 368 échantillons temporels, 33 228 s (9 h 13 min 48 s).** Aucun NaN, aucune valeur infinie et aucun canal vide dans les données numériques dépliées. Aucun canal EEG, ECG ou EMG entièrement constant. `SLI` est entièrement constant dans 12 enregistrements, indiqués ci-dessus. Un plateau local dans un canal ne signifie pas qu’il est constant sur toute la session.

## 4. Annotations : liaison, format et anomalies

Classeur : une feuille `Feuil1`, dimensions déclarées **1 048 574 × 36**, mais seulement **23 lignes avec un identifiant** (lignes Excel 2 à 24), correspondant à **22 identifiants distincts**. Il faut lire les cellules réellement renseignées et ignorer les lignes sans ID; une lecture naïve par pandas produit plus d’un million de lignes presque vides.

La première colonne n’a pas de titre et contient l’ID, par exemple `190304A-E`. Liaison aux fichiers : retirer les espaces externes puis remplacer `-` par `_`; ` 200625A-F ` devient `200625A_F`. Tous les IDs correspondent à une paire de fichiers.

Colonnes descriptives : `Sexe`, `Age`, `NombreCrises`, `TypeAbs`. Puis 12 couples de bornes, de `DebCE1` / `FinCE1` à `DebCE12` / `FinCE12`. Les noms du début varient : `DebutCE5` et `DebtCE11` sont des variantes à prendre en compte. Enfin `Intercrit1` à `Intercrit7` contiennent des points isolés, dont le sens précis n’est pas documenté; ils ne constituent pas des intervalles positifs. Les métadonnées démographiques ne doivent pas devenir des caractéristiques de fenêtre.

**Hypothèse de format : les bornes sont des fractions de jour Excel représentant l’heure de la journée.** Elles sont numériques avec format « General », pas des cellules explicitement formatées comme heures. Multiplier par 86400 donne des secondes de la journée, presque exactement entières; les différences donnent des durées de 2 à 60 s pour tous les couples sauf une anomalie. Ce ne sont ni des indices d’échantillons ni des secondes directement relatives au début.

**Problème d’alignement non résolu :** les signaux et headers commencent à 0 s, mais aucune heure absolue de début d’acquisition n’a été trouvée dans les données décodées. La formule nécessaire est `début_relatif = début_Excel × 86400 − heure_début_enregistrement_en_secondes` (et idem pour la fin). Le passage éventuel de minuit doit être traité si nécessaire. Soustraire la première crise, prendre modulo la durée ou traiter directement les fractions comme des secondes créerait des labels artificiels.

Exemple `190304A_E` : l’intervalle Excel vaut **10:35:06 → 10:35:13**, soit 7 s. Le header contient « a » à 1280 s, « crise » à 1292 s et « b » à 1301 s. Ces marqueurs ne définissent pas explicitement le même intervalle : il serait injustifié d’en déduire automatiquement une heure de départ précise.

### Comptage des crises : plusieurs valeurs incompatibles

- Somme des `NombreCrises` renseignés sur les 23 lignes : **90**, identique au total de la ligne 25. Un autre total isolé **180** existe à la ligne 1 048 574 et ne correspond pas aux paires renseignées.
- Nombre de couples début/fin effectivement présents : **95**, tous complets, avec fin > début. Cela comprend les deux versions du doublon.
- En conservant une seule des deux lignes de `230515B_G`, il reste **93 couples sur 22 enregistrements**; le choix de version change les bornes, pas le nombre.
- Ces nombres décrivent le fichier brut. **Le nombre validé de crises exploitables ne peut pas encore être établi**, à cause des anomalies et de l’absence d’alignement absolu.

### Cas à résoudre avant les labels

1. **`230515B_G`, lignes 15 et 22 : annotations conflictuelles.** Ligne 15 : âge 9, crises 12:29:06 → 12:29:20 (14 s), 12:30:12 → 12:30:44 (32 s). Ligne 22 : âge 8, crises 12:29:06 → 12:29:19 (13 s), 12:30:23 → 12:30:44 (21 s). Aucun choix silencieux ni addition de ces deux versions.
2. **`210406A_F`, ligne 7 : 12:15:50 → 15:15:58, soit 10 808 s**, alors que le signal couvre seulement 1454 s. Une erreur de saisie est probable; la correction possible de l’heure de fin ne doit pas être inventée. Le header indique « absence » à 636 s, sans durée.
3. **`210914B_A`, ligne 10 : `NombreCrises = 6`, mais 11 couples renseignés**, avec `TypeAbs = atypiques`. Il faut confirmer si les 11 couples sont bien des crises de la cible demandée.
4. **`200625A_F`, ligne 4 : aucun couple, `NombreCrises` vide.** Cela ne prouve pas qu’il s’agit d’un contrôle sans crise; cela peut signifier une annotation manquante. Proposition : l’exclure de l’apprentissage supervisé jusqu’à confirmation.

## 5. Identifiants et évaluation par groupes

Le préfixe à six chiffres est compatible avec une date YYMMDD, suivi de A/B et d’un suffixe A à H. **La signification de A/B et du suffixe n’est pas confirmée.** Les données démographiques montrent que le suffixe seul ne peut pas être utilisé comme identifiant de sujet : `190304A_E` est M, 17 ans, tandis que `230519A_E` est M, 6 ans; des enregistrements en `_C` ont aussi des sexes différents. Il faut une table d’identité patient explicite pour faire une validation leave-one-subject-out. En attendant, `Recording` est le groupe disponible; une validation par enregistrement limite la fuite des fenêtres voisines mais ne garantit pas la séparation des patients.

## 6. Nettoyage proposé, sans suppression effectuée

**Aucune colonne n’a encore été retirée et aucune conversion n’a été faite.** Journal des retraits proposés :

- Garder uniquement les colonnes `EEG*` pour les caractéristiques EEG; retirer `ECG`, `SLI` et les variantes EMG des entrées du modèle. Ces canaux auxiliaires restent documentés dans l’audit. Cela retire aussi les 12 `SLI` constants.
- Ne pas ajouter `Annotations`, `Duration`, `Onset`, `Sexe`, `Age`, `NombreCrises`, `TypeAbs` ou `Intercrit*` aux caractéristiques. Garder les métadonnées et annotations séparément.
- Représenter `Time` en secondes, avec assez de précision pour conserver le pas 1/256 s; convertir les amplitudes en `float32` lors du traitement ultérieur. Aucun remplacement de NaN n’est nécessaire pour les signaux inspectés.
- Décider du montage commun avant les modèles : les huit EEG communs donnent un espace de caractéristiques fixe pour les 22 enregistrements; si `200625A_F` est exclu comme non annoté, les 21 autres possèdent les 19 EEG communs. Ne pas remplir les canaux absents par des valeurs arbitraires.
- Chaque enregistrement actuel a moins de 1 048 575 échantillons : il tient dans une feuille Excel avec une ligne d’en-tête (maximum observé : 921 600). La conversion devra tout de même gérer plusieurs feuilles pour les futurs fichiers plus longs.

## 7. Décisions attendues au checkpoint

Avant la suite, confirmer l’inspection et fournir, si possible : les heures de début absolues des 22 acquisitions (ou les fichiers EDF/metadata d’origine), l’unité d’amplitude, la bonne version du doublon `230515B_G`, la correction de `210406A_F`, le statut des 11 intervalles de `210914B_A` et le statut sans-crise/non-annoté de `200625A_F`.

À défaut, les cas non résolus devront être explicitement exclus ou signalés; un décalage temporel ne peut pas être déduit de façon fiable du seul tableau Excel. L’inspection seule est terminée; le pipeline et le notebook de modélisation attendent la confirmation.
