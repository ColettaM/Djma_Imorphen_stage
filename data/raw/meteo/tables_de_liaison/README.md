# Tables de liaison – Phase 2 (contexte climatique)

Ce dossier contient les tables de référence utilisées par le pipeline de la phase 2 afin de relier les informations extraites des images, des fiches utilisateurs et des exports applicatifs.

## Fichiers attendus

### gardens.csv

Informations relatives aux jardins déclarés par les utilisateurs.

Utilisé pour :

* récupérer les coordonnées géographiques ;
* identifier le type d'emplacement (jardin, balcon, serre, intérieur, etc.) ;
* déterminer si une observation est exploitable pour l'analyse climatique.

### plant_diseases.csv

Référentiel des maladies des plantes.

Utilisé pour :

* résoudre les identifiants de maladies ;
* harmoniser les noms de bioagresseurs ;
* produire les analyses de cohérence entre les images et les fiches.

### plant_pests.csv

Référentiel des ravageurs.

Utilisé pour :

* résoudre les identifiants de ravageurs ;
* harmoniser les libellés utilisés dans les exports DJMA.

### product_pages.csv

Référentiel des pages produits et contenus associés.

Utilisé pour :

* récupérer les informations descriptives liées aux bioagresseurs ;
* enrichir certaines étapes d'analyse et de validation.

## Utilisation dans le pipeline

Ces tables sont utilisées par plusieurs scripts de la phase 2 :

```text
src/meteo/
├── match_images.py
├── extract_exif.py
├── build_phase2_context.py
├── aggregate_to_bulletins.py
├── fetch_weather.py
├── weather_features.py
└── diagnose_bulletin_consistency.py
```

## Remarque

Les fichiers présents dans ce dossier correspondent à des tables de référence utilisées lors du stage. Selon le contexte de diffusion du dépôt (privé ou public), certaines tables peuvent être remplacées par des versions anonymisées ou ne pas être distribuées.
