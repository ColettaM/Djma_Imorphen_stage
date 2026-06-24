# Contexte phytosanitaire régional par Bulletins de Santé du Végétal (BSV)

## Objectif

Cette étude vise à enrichir les prédictions issues d'un modèle de vision par ordinateur pour le diagnostic des bioagresseurs de la tomate à l'aide d'un contexte phytosanitaire régional extrait automatiquement des Bulletins de Santé du Végétal (BSV).

L'approche repose sur la combinaison de deux sources d'information :

* un modèle de classification d'images produisant une prédiction de bioagresseur ;
* les observations phytosanitaires régionales publiées dans les BSV.

L'objectif n'est pas de confirmer ou d'infirmer un diagnostic, mais d'évaluer sa cohérence avec le contexte sanitaire observé dans une région donnée à une période donnée.

## Structure du projet

```text
phythosanitaire/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── grand_est_2025/
│   │   ├── pays_de_la_loire_2025/
│   │   └── referentiel_cnn_bsv.csv
│   └── processed/
├── notebooks/
├── outputs/
│   ├── extracted_text/
│   ├── tables/
│   └── logs/
└── src/
```

## Données

### BSV bruts

Les Bulletins de Santé du Végétal sont stockés dans :

```text
data/raw/grand_est_2025/
data/raw/pays_de_la_loire_2025/
```

Ces deux régions sont utilisées dans une première phase exploratoire car elles présentent des formats de BSV différents. Cela permet de tester la robustesse du futur pipeline d'extraction.

### Référentiel CNN-BSV

Le fichier suivant contient la correspondance entre les classes du modèle de vision et le vocabulaire phytosanitaire utilisé dans les BSV :

```text
data/raw/referentiel_cnn_bsv.csv
```

Il contient notamment les colonnes suivantes :

```text
classe_dataset
cultures_concernées
nom_fr
agent
type
synonymes_bsv
hotes
```

L'agent biologique est utilisé comme clé principale de comparaison afin de limiter les ambiguïtés liées aux noms français des bioagresseurs.

## Modèle de données

L'information extraite des BSV est organisée en plusieurs tables afin de respecter la granularité réelle des bulletins.

### 1. Niveau bulletin

Table cible :

```text
data/processed/bsv_bulletins.csv
```

Une ligne correspond à un bulletin.

Colonnes :

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

Le champ `bulletin_id` doit être stable et unique. Il est construit à partir de la région, de l'année et du numéro du bulletin.

La colonne `periode_prevision_meteo` désigne la période temporelle couverte par la prévision météorologique associée au bulletin. Elle est plus informative qu'un simple numéro de semaine éditorial.

`nature_meteo` distingue une prévision, un bilan/tendance et une météo absente.
`meteo_extraction_status` indique si le texte est `lisible`, `image_seule`,
`absente` ou `trop_longue`. Seul un texte lisible peut être interprété comme
favorable, défavorable ou neutre par le moteur de compatibilité.

`meteo_extraction_method` conserve la provenance de la sélection (`general:*`
ou `fallback:*`). `meteo_extraction_score` conserve le score heuristique du
candidat général retenu ; il vaut `0` lorsqu'un fallback de profil est utilisé.

`periode_prevision_meteo` n'est renseignée que lorsque la période future est
identifiable. Elle reste vide pour les bilans PDL 2023, sauf plage de dates
explicitement présente dans le texte.

L'extraction météo est hybride : le script détecte et score d'abord des blocs
candidats à partir des titres, sources et mots-clés météorologiques. Le
`template_id` documente le format et n'active un fallback spécifique que si
l'extraction générale est insuffisante ou qu'un profil connu doit invalider un
faux positif.

### 2. Niveau observation phytosanitaire

Table cible :

```text
data/processed/bsv_observations.csv
```

Une ligne correspond à une information phytosanitaire extraite d'une section culture ou d'une section de surveillance.

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

Le champ `registre` permet de distinguer la nature de l'information :

```text
observation_parcelle
curseur_risque
veille_reglementaire
```

Par défaut, le moteur de comparaison utilisera les lignes de type `observation_parcelle`. Les lignes de veille réglementaire sont conservées pour la traçabilité mais ne doivent pas être interprétées comme des présences observées.

### 3. Risque synthétique

Table cible :

```text
data/processed/bsv_risque_synthetique.csv
```

Cette table contient les curseurs ou niveaux de risque fournis à un niveau agrégé dans les BSV, lorsqu'ils ne correspondent pas directement à une observation précise.

Colonnes :

```text
bulletin_id
groupe_culture
categorie
niveau_1_5
niveau_normalise
```

Exemple :

```text
Solanacées
Maladies
4
élevé
```

Cette séparation évite de dupliquer artificiellement un risque global sur chaque observation individuelle.

### 4. Résolution biologique

La résolution vers un agent biologique n'est pas effectuée directement au moment de l'extraction du PDF.

Elle est réalisée dans une étape séparée, déterministe et auditable, à partir du référentiel :

```text
(bioagresseur, culture)
→ referentiel_cnn_bsv.csv
→ agent
→ classe_dataset
```

Les couples non résolus doivent être conservés dans un fichier de log afin d'identifier :

* les synonymes manquants dans le référentiel ;
* les bioagresseurs hors périmètre des 12 classes CNN ;
* les cas ambigus nécessitant une validation experte.

## Étapes du projet

1. Construire le référentiel de correspondance entre les classes CNN et les bioagresseurs décrits dans les BSV.
2. Réaliser une analyse exploratoire manuelle sur plusieurs BSV de différentes régions.
3. Identifier les informations stables à extraire : culture, bioagresseur, observations, risque, évolution, localisation et registre de l'information.
4. Extraire automatiquement le texte complet des PDF et les tableaux détectables afin de créer un cache local exploitable.
5. Extraire les métadonnées de niveau bulletin.
6. Extraire les observations phytosanitaires à partir des tableaux CSV et du texte brut.
7. Extraire les risques synthétiques lorsque les bulletins fournissent des curseurs globaux.
8. Résoudre les bioagresseurs extraits vers les agents biologiques du référentiel.
9. Comparer les prédictions du modèle de vision avec le contexte phytosanitaire régional.
10. Produire un indicateur de cohérence du diagnostic.

## Installation

Créer et activer l'environnement virtuel :

```bash
python -m venv .bsv
source .bsv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Étape 1 — Extraction texte et tableaux

La première étape de code consiste à extraire :

* le texte brut complet de chaque PDF ;
* les tableaux détectés dans chaque page.

Objectif :

```text
PDF BSV
→ extraction texte complète
→ extraction des tableaux
→ cache local exploitable
→ extraction ciblée des sections utiles
→ structuration des informations
```

Sorties :

```text
outputs/extracted_text/
outputs/tables/
outputs/logs/extraction_log.csv
```

Le texte brut sert principalement à extraire les métadonnées, les prévisions météorologiques, les analyses de risque et les facteurs favorables.

Les tableaux CSV servent prioritairement à extraire les observations en parcelles :

```text
Ravageurs / Maladies
Localisation
Cultures
Observations
Evolution
```

## Étape 2 — Extraction niveau bulletin

Script :

```text
src/extract_bulletins.py
```

Sortie :

```text
data/processed/bsv_bulletins.csv
```

Champs extraits :

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

## Étape 3 — Extraction niveau observation

Script :

```text
src/extract_observations.py
```

Stratégie :

1. Charger `bsv_bulletins.csv` comme registre des identifiants et templates.
2. Extraire PDL 2023 depuis le texte narratif en séparant les deux colonnes.
3. Enrichir PDL 2023 avec les tableaux de la même page seulement lorsqu'ils mentionnent la culture ou le bioagresseur.
4. Extraire les tableaux PDL 2024-2025 contenant `Cultures` et `Observations`.
5. Extraire les sous-sections narratives `Observations` des BSV Grand Est 2023-2025.
6. Nettoyer les cellules, supprimer les lignes parasites et séparer les observations multi-cultures.
7. Vérifier que chaque observation est liée à un `bulletin_id` connu.
8. Produire `data/processed/bsv_observations.csv` et le journal d'extraction.

Le bulletin `grand_est_sans_observation` est ignoré explicitement. Le BSV Flash
PDL 2023 utilise un parseur dédié, car il est structuré par registre puis par
culture plutôt que par titres individuels de bioagresseurs.

## Ordre des scripts prévu

```text
src/extract_text.py
src/extract_bulletins.py
src/extract_observations.py
src/extract_risks.py
src/resolve_agents.py
```

## Preparation du corpus avant extraction

Avant de lancer l'extraction, classifier les PDF, calculer leur MD5 et exclure
les bilans, catalogues et doublons exacts sans supprimer les fichiers sources :

```bash
python src/prepare_corpus.py
```

Le script produit :

```text
data/processed/bsv_corpus_manifest.csv
data/processed/bsv_corpus_exploitable.txt
```

Le manifeste conserve `type_document`, les indicateurs de doublons et
`corpus_exploitable`. Les bulletins classiques et les flashs sont conserves.

Par defaut, `src/extract_text.py` traite uniquement les chemins presents dans
la liste exploitable. L'option `--all-pdfs` permet exceptionnellement de
parcourir tous les PDF.

Ordre de lancement :

```bash
python src/prepare_corpus.py
python src/extract_text.py
```
