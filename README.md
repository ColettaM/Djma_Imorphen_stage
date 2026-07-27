# Contexte phytosanitaire regional par BSV

## Objectif

Ce projet constitue la phase Vision/BSV du pipeline DJMA pour la tomate.

Il combine deux sources :

- un modele CNN ResNet50 qui predit une classe phytosanitaire a partir d'une
  image de tomate ;
- un contexte phytosanitaire regional extrait des Bulletins de Sante du Vegetal
  (BSV).

Le moteur ne cherche pas a etablir une verite biologique absolue. Il repond a
une question plus precise :

```text
La prediction CNN est-elle coherente avec les bioagresseurs signales dans les
BSV de la meme region, a une date proche ?
```

La sortie finale du moteur est volontairement simple :

```text
Coherent
Non_corrobore
Indetermine
```

Les informations fines, comme le risque, la meteo ou le type de preuve, sont
conservees dans des colonnes separees.

Ce README documente le protocole, les entrees fixes, les controles et les
artefacts produits. Les effectifs et metriques calcules pendant un run ne sont
pas recopies ici : ils restent dans les CSV, les notebooks et les journaux afin
d'eviter qu'une execution ulterieure rende la documentation obsolete.

## Structure

```text
phythosanitaire/
|-- README.md
|-- config/
|   `-- pipeline.json
|-- docs/
|   `-- phase3_historique.md
|-- data/
|   |-- raw/
|   |   |-- referentiel_cnn_bsv.csv
|   |   `-- tomato_final/
|   `-- processed/
|       |-- bsv_bulletins.csv
|       |-- bsv_observations.csv
|       |-- synthese_tomate.csv
|       |-- synthese_bioagresseurs_tomate_par_region.csv
|       |-- periodes_bioagresseurs_tomate_par_region.csv
|       |-- periodes_bioagresseurs_tomate_par_region_annee.csv
|       |-- bsv_compatibility_phase3.csv
|       `-- phase3/
|           |-- images_unlabeled/
|           |-- ground_truth.csv
|           |-- phase3_test_manifest_masked.csv
|           |-- phase3_test_with_metadata.csv
|           |-- phase3_scenario_assignments.csv
|           `-- phase3_bsv_metadata_coverage.csv
|-- models/
|   `-- best_resnet50_tomato.pth
|-- notebooks/
|   |-- phase3_analysis.ipynb
|   `-- phase3_visualisation.ipynb
|-- outputs/
|   |-- extracted_text/
|   |-- tables/
|   |-- logs/
|   `-- phase3/
|       |-- cnn_topk_predictions.json
|       |-- requete_phase3.json
|       `-- analysis/
`-- src/
    |-- scrape_bsv.py
    |-- prepare_corpus.py
    |-- extract_text.py
    |-- extract_bulletins.py
    |-- extract_observations.py
    |-- extract_tomato_bioagressor_periods.py
    |-- build_manifest_images.py
    |-- add_bsv_compatible_metadata.py
    |-- inference_cnn.py
    |-- build_phase3_query.py
    |-- moteur_compatibilite.py
    |-- phase3_prepare_analysis.py
    |-- phase3_evaluate.py
    `-- run_pipeline.py
```

## Corpus BSV

Le corpus couvre trois regions :

- Grand Est ;
- Pays de la Loire ;
- Provence-Alpes-Cote d'Azur.

Les BSV sont d'abord prepares, extraits en texte/tableaux, puis structures en
deux tables principales :

```text
data/processed/bsv_bulletins.csv
data/processed/bsv_observations.csv
```

## Tables principales

### bsv_bulletins.csv

Table de niveau bulletin.

Colonnes principales :

```text
bulletin_id
region
date_bulletin
template_id
nature_meteo
meteo_extraction_status
meteo_extraction_method
meteo_extraction_score
periode_prevision_meteo
prevision_meteo
```

`bulletin_id` est stable et unique.

Une correction est integree dans `extract_bulletins.py` pour les cas ou l'annee
du nom de fichier contredit l'annee du dossier source. Exemple traite :

```text
pdl/maraichage/2024/BSV_maraichage_flash_no_01_20230118.txt
```

Ce fichier est corrige en bulletin 2024 car le dossier et le texte indiquent
2024.

### bsv_observations.csv

Table de niveau observation phytosanitaire.

Colonnes :

```text
bulletin_id
registre
culture
bioagresseur
type
departements
localisations
observation
indicateur_quantitatif
evolution
analyse_risque
niveau_risque_observation
facteurs_favorables
source_information
```

### synthese_tomate.csv

`extract_observations.py` produit maintenant aussi :

```text
data/processed/synthese_tomate.csv
```

Elle est construite ainsi :

```text
bsv_observations.csv
→ selection culture tomate ou solanacees
→ jointure avec bsv_bulletins.csv via bulletin_id
→ ajout region, date, annee, semaine, meteo
```

Les lignes `solanacees` sont conservees, car Grand Est regroupe souvent tomate,
aubergine et poivron sous ce libelle.

## Bioagresseurs tomate et periodes BSV

Le script :

```bash
python src/extract_tomato_bioagressor_periods.py
```

lit :

```text
data/processed/synthese_tomate.csv
```

et produit :

```text
data/processed/synthese_bioagresseurs_tomate_par_region.csv
data/processed/periodes_bioagresseurs_tomate_par_region.csv
data/processed/periodes_bioagresseurs_tomate_par_region_annee.csv
```

Le fichier le plus important pour la suite est :

```text
periodes_bioagresseurs_tomate_par_region_annee.csv
```

Il sert a verifier qu'un bioagresseur est bien couvert par le corpus BSV et a
attribuer des dates/regions compatibles aux images.

## Referentiel CNN-BSV

Le fichier :

```text
data/raw/referentiel_cnn_bsv.csv
```

relie les classes du dataset image aux termes BSV :

```text
classe_dataset
cultures_concernees
nom_fr
agent
type
synonymes_bsv
hotes
```

Exemples :

```text
tomato_late_blight -> Mildiou -> Phytophthora infestans
tomato_early_blight -> Alternaria
tomato_leaf_mold -> Cladosporiose
tomato_spider_mites_two_spotted_spider_mite -> Acariens
```

## Phase 3 image

### 1. Anonymisation des images

```bash
python src/build_manifest_images.py
```

Ce script lit :

```text
data/raw/tomato_final/test/
```

et produit :

```text
data/processed/phase3/images_unlabeled/
data/processed/phase3/ground_truth.csv
data/processed/phase3/phase3_test_manifest_masked.csv
```

Les images sont renommees :

```text
obs_000001.jpg
obs_000002.jpg
...
```

La classe reelle reste uniquement dans `ground_truth.csv`.

### 2. Construction experimentale des contextes BSV

```bash
python src/add_bsv_compatible_metadata.py
```

Ce script :

1. lit le manifest anonymise ;
2. recupere la vraie classe depuis `ground_truth.csv` ;
3. tire un scenario avant toute utilisation des predictions CNN ;
4. utilise la vraie classe uniquement pour rechercher un contexte conforme ;
5. conserve les 3 761 images, y compris lorsque le scenario tire est impossible ;
6. separe le manifest aveugle de la table experimentale d'evaluation.

Les proportions par defaut sont 40 % `favorable`, 40 % `non_favorable` et
20 % `insuffisant`, avec la graine 2026. Ces proportions constituent un choix
de construction semi-synthetique. Elles ne mesurent pas la frequence reelle de
ces situations sur le terrain. Les quotas entiers sont calcules de maniere
deterministe pour conserver exactement toutes les images d'entree.

La verite terrain est utilisee uniquement hors ligne pour construire et
evaluer ces scenarios experimentaux. Elle n'intervient jamais dans le
fonctionnement du CNN ni du moteur BSV. En production, la date, la region et
les bulletins disponibles proviendraient directement de l'observation reelle.
Les quotas 40/40/20 ne doivent donc pas etre interpretes comme des frequences
attendues en conditions de terrain.

Definitions :

- `favorable` : bulletin regional recent, periode litterature valide et agent
  de la vraie classe mentionne ;
- `non_favorable` : bulletin regional recent et informatif pour la tomate ou
  un hote associe, mais agent de la vraie classe absent ;
- `insuffisant` : aucun bulletin recent, periode invalide ou classe non couverte.

Si aucun contexte conforme n'existe, le scenario demande reste enregistre,
`scenario_realisable` vaut `False` et un contexte insuffisant explicite est
attribue. Aucune image n'est supprimee silencieusement.

Sorties :

```text
data/processed/phase3/phase3_test_with_metadata.csv
data/processed/phase3/phase3_scenario_assignments.csv
data/processed/phase3/phase3_bsv_metadata_coverage.csv
outputs/logs/bsv_compatible_metadata.log
```

Le script doit conserver les 3 761 images d'entree. Il mesure ensuite, pour le
run courant, combien de scenarios demandes sont effectivement realisables.
Un scenario `favorable` ou `non_favorable` peut etre irrealisable lorsque la
classe reelle n'est pas couverte par le corpus BSV ou lorsqu'aucun bulletin ne
permet de construire le contexte demande. Ces effectifs sont des resultats du
run et sont donc consultables dans les CSV, les notebooks et les logs, pas dans
ce README.

Dans l'entonnoir, ces observations sont comptabilisees en priorite comme
« classe non couverte ». Elles ne sont pas recomptees comme « scenario non
realisable », ce qui explique que cette seconde exclusion puisse etre nulle
dans le jeu couvert.

Le manifest aveugle `phase3_test_with_metadata.csv` contient uniquement :

```text
observation_id
image_path
label_masked
date_observation
region
departement
```

La table reservee a l'evaluation `phase3_scenario_assignments.csv` contient :

```text
observation_id
true_label
scenario_assigne
scenario_realise
scenario_realisable
raison_non_realisation
cause_insuffisance
classe_couverte_bsv
context_id
bulletin_id_assigne
date_observation
region
departement
seed
```

`cause_insuffisance` decrit le contexte voulu lors de la construction et prend
les valeurs `classe_non_couverte_bsv`, `sans_bulletin` ou
`hors_periode_litterature`. Il ne doit pas etre confondu avec `niveau_preuve`,
qui est produit ensuite par le moteur aveugle. Leur concordance est controlee
de facon bloquante lorsque le Top-1 CNN correspond a la vraie classe ; lorsque
le CNN se trompe, l'ecart est conserve comme diagnostic descriptif.

## Inference CNN

Execution normale avec reutilisation securisee par SHA-256 :

```bash
python src/inference_cnn.py
```

Reinference complete forcee, uniquement si les images, le modele ou le
pretraitement CNN ont change :

```bash
python src/inference_cnn.py --no-reuse
```

Le CNN prend directement les images anonymisees depuis :

```text
data/processed/phase3/images_unlabeled/
```

Le fichier `phase3_test_with_metadata.csv` sert seulement a recuperer les
metadonnees associees a chaque image.

Le script verifie que :

- les 3 761 identifiants sont uniques ;
- chaque chemin du manifest correspond a une image existante ;
- toutes les lignes portent `label_masked=True` ;
- la sortie couvre exactement le manifest.

Une reprise est possible a partir d'une sortie existante, mais une prediction
n'est reutilisee que si l'empreinte SHA-256 de l'ancienne image et celle de
l'image courante sont identiques. L'option `--no-reuse` force une inference
homogene de tout le jeu.

Sortie :

```text
outputs/phase3/cnn_topk_predictions.json
```

La vraie classe, le scenario demande, le scenario realise et son statut de
realisation restent dans la table experimentale separee. Ils ne sont transmis
ni au CNN, ni au moteur, ni au futur juge LLM.

## Construction des requetes Phase 3

```bash
python src/build_phase3_query.py --top-k 3
```

Ce script transforme :

```text
outputs/phase3/cnn_topk_predictions.json
```

en :

```text
outputs/phase3/requete_phase3.json
```

Il conserve uniquement les informations utilisables en situation reelle :

```text
image_id
date_observation
region
top_k
```

Tous les champs experimentaux de `phase3_scenario_assignments.csv` sont
volontairement exclus de `requete_phase3.json`.

Exemple de requete moteur :

```json
{
  "image_id": "obs_000001",
  "date_observation": "2024-07-10",
  "region": "Grand Est",
  "top_k": [
    {
      "classe": "tomato_late_blight",
      "score": 0.82
    }
  ]
}
```

Le moteur ne recoit donc pas la classe reelle de l'image. Il deduit lui-meme
l'agent biologique, les synonymes BSV et les hotes associes a partir de chaque
classe CNN predite et du referentiel.

## Moteur de compatibilite BSV

```bash
python src/moteur_compatibilite.py \
  --query-file outputs/phase3/requete_phase3.json \
  --output-file data/processed/bsv_compatibility_phase3.csv
```

Le moteur :

1. prend une image, une region, une date et une prediction CNN ;
2. retrouve dans le referentiel l'agent biologique, les synonymes et les hotes
   associes de la classe predite ;
3. verifie si la date est dans la periode operationnelle litterature du
   bioagresseur predit, sauf pour `tomato_healthy` qui n'est pas un
   bioagresseur ;
4. cherche le bulletin BSV regional le plus recent avant la date d'observation ;
5. limite la recherche a un bulletin de moins de 14 jours ;
6. cherche si l'agent predit est mentionne dans les observations BSV ;
7. distingue un signal direct tomate d'un signal sur hote associe ;
8. extrait le niveau de risque quand il existe ;
9. garde la meteo comme information descriptive ;
10. produit une decision simplifiee.

La verification de periode n'utilise pas la fenetre exacte de l'annee evaluee.
Le moteur utilise des fenetres operationnelles issues de la litterature pour la
tomate en plein champ en France metropolitaine. Les fenetres BSV annuelles
restent utiles pour construire et controler le jeu experimental, mais elles ne
sont pas utilisees comme oracle dans la decision.

Exemples de fenetres plausibles :

```text
tomato_late_blight                         avril-octobre
tomato_early_blight                        avril-octobre
tomato_leaf_mold                           avril-octobre
tomato_septoria_leaf_spot                  mai-octobre
tomato_spider_mites_two_spotted_spider_mite avril-decembre
tomato_verticulium_wilt                    avril-septembre
tomato_leaf_blight                         non definissable
tomato_healthy                             sans fenetre litterature
```

### Donnees utilisees

```text
data/processed/bsv_bulletins.csv
data/processed/bsv_observations.csv
data/raw/referentiel_cnn_bsv.csv
outputs/phase3/requete_phase3.json
```

### Sortie du moteur

```text
data/processed/bsv_compatibility_phase3.csv
```

Colonnes :

```text
image_id
date_observation
region
classe_cnn_predite
score_cnn
agent_referentiel
agent_present_bulletin
niveau_preuve
niveau_risque
meteo_status
meteo_favorable
compatibilite
decision_reason
```

### Statuts finaux

Le moteur ne produit plus que trois statuts :

```text
Coherent
Non_corrobore
Indetermine
```

`Coherent` signifie que la prediction CNN est compatible avec le contexte BSV.

`Non_corrobore` signifie que le bulletin contient des signaux tomate ou hotes
associes, mais ne corrobore pas cette prediction.

`Indetermine` signifie que le moteur ne peut pas conclure : hors periode
operationnelle, pas de bulletin recent, classe non couverte, absence de signal
exploitable.

Les anciens statuts `Coherent_renforce` et `Partiel` ne sont plus utilises. Leur
information est conservee dans :

```text
niveau_preuve
niveau_risque
decision_reason
```

### Niveau de preuve

`niveau_preuve` peut valoir :

```text
direct
hote_associe
absent
non_couvert
hors_periode_litterature
sans_bulletin
```

Exemple :

```text
compatibilite = Coherent
niveau_preuve = hote_associe
decision_reason = signal present uniquement via un hote associe
```

### Cas hote associe

`hote_associe` signifie que l'agent predit n'est pas signale directement sur
tomate dans le bulletin retenu, mais qu'il est signale sur une culture hote
compatible avec le referentiel.

Exemple :

```text
classe_cnn_predite = tomato_late_blight
agent_referentiel = Phytophthora infestans
culture BSV = pomme de terre
bioagresseur BSV = mildiou
```

Dans ce cas, le moteur ne pretend pas que la tomate est atteinte dans le BSV.
Il considere seulement qu'il existe un signal phytosanitaire regional associe.
La decision reste `Coherent`, mais avec une preuve plus faible que `direct` :

```text
compatibilite = Coherent
niveau_preuve = hote_associe
agent_present_bulletin = True
```

### Meteo

La meteo est conservee dans :

```text
meteo_status
meteo_favorable
```

Actuellement, la meteo reste informative et ne modifie pas la decision finale.

### Lecture de la sortie moteur

Le moteur produit une ligne pour chaque prediction CNN effectivement evaluee
parmi le Top-K. Le nombre de lignes de verdicts peut donc etre superieur au
nombre d'images : une image peut contribuer plusieurs predictions classees.

Les distributions de `compatibilite` et de `niveau_preuve` sont des resultats
du run. Elles sont publiees dans les CSV d'analyse, les notebooks et le journal
du pipeline, mais ne sont pas figees dans ce README.

La valeur `non_couvert` decrit la couverture de la **classe predite** sur une
ligne Top-K. Elle ne doit pas etre confondue avec la couverture de la vraie
classe de l'image, utilisee uniquement hors ligne pour construire les
populations d'evaluation.

`phase3_insufficiency_consistency.csv` ne justifie pas a lui seul cette
distribution complete. Il controle une question plus restreinte, au niveau
Top-1 : la cause d'insuffisance voulue lors de la construction correspond-elle
au `niveau_preuve` produit par le moteur ? Une analyse exhaustive des lignes
Top-K doit distinguer le rang, le nombre d'images concernees, le nombre de
predictions `non_couvert` par image et la couverture de la classe predite par
rapport a celle de la classe reelle.

## Preparation des populations d'analyse Phase 3

```bash
python src/phase3_prepare_analysis.py
```

Ce script prend la requete aveugle comme source d'autorite pour le Top-1. Il
verifie les 3 761 identifiants, l'ordre des predictions, les scores, les
contextes, les verdicts et l'absence de champs experimentaux dans la requete.
Toute anomalie bloque les exports.

Sorties figees dans `outputs/phase3/analysis/` :

```text
phase3_analysis_full.csv
phase3_analysis_covered.csv
phase3_analysis_primary.csv
phase3_selection_summary.csv
phase3_scenario_summary.csv
phase3_contingency_top1.csv
phase3_integrity_checks.csv
phase3_modeling_feasibility.csv
phase3_separation_diagnostics.csv
phase3_insufficiency_consistency.csv
```

Les trois populations sont construites comme suit :

- le **jeu complet** contient les 3 761 images d'entree ;
- le **jeu couvert** contient les images dont la vraie classe appartient au
  referentiel couvert par le corpus BSV ;
- le **jeu principal** applique au jeu couvert les filtres de realisabilite du
  scenario et de decidabilite du verdict Top-1.

Le jeu principal est defini exactement par :

```text
classe reelle couverte par le corpus BSV
ET scenario_realisable = True
ET verdict du moteur Top-1 dans {Coherent, Non_corrobore}
```

Autrement dit, il rassemble les images appartenant a une classe couverte,
associees a un scenario realisable et pour lesquelles le moteur produit un
verdict Top-1 decidable.

Dans l'entonnoir, les classes non couvertes sont retirees en premier au titre
de la couverture structurelle. Une meme image n'est donc pas recomptee comme
un scenario non realisable. Les echecs de construction du contexte sont
ensuite distingues des verdicts `Indetermine`, qui correspondent a une
abstention produite par le moteur. Les effectifs de chaque etape sont calcules
a chaque run dans `phase3_selection_summary.csv`.

Deux notebooks sont maintenus :

- `notebooks/phase3_analysis.ipynb` lit les CSV figes produits par
  `phase3_prepare_analysis.py` et presente l'integrite, les populations,
  l'entonnoir, les scenarios, la contingence et la faisabilite statistique ;
- `notebooks/phase3_visualisation.ipynb` complete cette analyse avec une vue
  descriptive du corpus, des artefacts images, des predictions CNN, de la
  requete et des verdicts du moteur. Il consulte certains artefacts bruts pour
  ces illustrations descriptives et lit aussi les sorties d'analyse afin de
  rester rattache au meme run.

Les notebooks ne reconstruisent aucune jointure sensible pour l'analyse
statistique. Les statistiques principales utilisent les CSV figes par
`phase3_prepare_analysis.py`. Les lectures d'artefacts bruts du notebook de
visualisation sont reservees a des descriptions clairement separees de cette
analyse statistique.

Les jointures sensibles et les controles bloquants restent centralises dans
`phase3_prepare_analysis.py`. Le controle
`cause_insuffisance`/`niveau_preuve` est bloquant lorsque le Top-1 CNN est la
vraie classe et descriptif lorsque le CNN est incorrect.

## Orchestrateur du pipeline

La configuration partagee se trouve dans `config/pipeline.json`. Elle fixe
une seule source de verite pour l'effectif attendu, la graine, le Top-K et
l'age maximal d'un bulletin (14 jours).

Le chemin recommande, sans scraping, est :

```bash
python src/run_pipeline.py
```

Cette commande part par defaut de la phase B et execute les phases B a E.

Le scraping n'est lance que si la phase A est explicitement demandee :

```bash
python src/run_pipeline.py --from-phase A
```

Exemples :

```bash
# Afficher toutes les commandes sans les executer
python src/run_pipeline.py --from-phase A --dry-run

# Refaire les contextes, le moteur et les analyses avec les donnees BSV existantes
python src/run_pipeline.py --from-phase C

# Refaire uniquement les analyses et les notebooks
python src/run_pipeline.py --from-phase E

# Executer uniquement les tests prealables de la phase B
python src/run_pipeline.py --from-phase B --to-phase B

# Forcer une reinference CNN complete
python src/run_pipeline.py --from-phase D --force-cnn

# Ne pas executer les notebooks
python src/run_pipeline.py --skip-notebooks
```

L'orchestrateur s'arrete au premier echec et produit un journal texte ainsi
qu'un rapport JSON horodates dans `outputs/logs`. Si les images anonymisees
existent deja, leur reconstruction est ignoree et leur integrite est controlee ;
aucun dossier n'est supprime automatiquement.

Le suivi lisible du dernier run est toujours disponible dans :

```text
outputs/logs/pipeline.log
```

Ce journal est alimente en direct. Il explique chaque phase, reprend les sorties
des scripts, commente les validations et termine par un bilan. Les versions
horodatees et le rapport structure `pipeline.json` sont conserves dans le
meme repertoire.

## Pipeline complet

L'orchestrateur applique l'ordre suivant :

```text
A - Donnees BSV
    scrape_bsv
    prepare_corpus
    extract_text
    extract_bulletins
    extract_observations
    extract_tomato_bioagressor_periods

B - Prerequis sur fixtures
    tests du moteur
    tests de construction des scenarios
    tests de la requete aveugle, du CNN et de l'analyse

C - Preparation des images
    build_manifest_images
    controle du manifeste et de la verite separee
    add_bsv_compatible_metadata
    controle des scenarios, causes et effectifs

D - Predictions CNN et moteur BSV
    inference_cnn
    controle SHA-256 et couverture CNN
    build_phase3_query
    controle du schema aveugle
    moteur_compatibilite
    controle de la couverture des verdicts

E - Analyse et visualisation
    phase3_prepare_analysis
    controle cause_insuffisance / niveau_preuve
    phase3_analysis.ipynb
    phase3_visualisation.ipynb
    suite complete de tests
```

Equivalent manuel :

```bash
python src/scrape_bsv.py
python src/prepare_corpus.py
python src/extract_text.py
python src/extract_bulletins.py
python src/extract_observations.py

python src/extract_tomato_bioagressor_periods.py
python src/build_manifest_images.py
python src/add_bsv_compatible_metadata.py

python src/inference_cnn.py
python src/build_phase3_query.py --top-k 3
python src/moteur_compatibilite.py --query-file outputs/phase3/requete_phase3.json --output-file data/processed/bsv_compatibility_phase3.csv
python src/phase3_prepare_analysis.py
jupyter nbconvert --to notebook --execute --inplace notebooks/phase3_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/phase3_visualisation.ipynb
python -m unittest discover -s tests
```

## Logs utiles

```text
outputs/logs/bulletins.log
outputs/logs/observations.log
outputs/logs/bsv_compatible_metadata.log
outputs/logs/inference_cnn.log
outputs/logs/pipeline.log
outputs/logs/pipeline.json
```

## Remarques methodologiques

- Les BSV ne sont pas une verite terrain image.
- Le moteur donne une plausibilite contextuelle regionale et temporelle.
- Les proportions 40/40/20 sont imposees par le plan experimental et ne sont
  pas des frequences estimees en conditions reelles.
- Le jeu complet mesure la couverture structurelle. Le jeu principal exige une
  classe reelle couverte, un scenario realisable et un verdict du moteur Top-1
  `Coherent` ou `Non_corrobore`.
- Le moteur transforme une date hors fenetre operationnelle en
  `hors_periode_litterature`.
- `Non_corrobore` signifie absent du bulletin pertinent, pas biologiquement
  absent du terrain.
- Le niveau de risque est conserve mais ne remplace pas la decision principale.
- La meteo est gardee comme contexte informatif, sans regle agroclimatique forte.
