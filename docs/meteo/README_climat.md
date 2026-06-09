# Climat — Pipeline phase 2

Module dédié à la récupération du contexte météo pour chaque bulletin
utilisateur, en complément de la phase 1 (CNN + LLM + NLI sur l'image).

## Pourquoi cette phase ?

La phase 1 produit des prédictions de pathologie à partir de l'image seule.
Certaines erreurs très confiantes ne peuvent être levées qu'avec un signal
externe : un mildiou est très improbable après deux semaines de sécheresse,
une attaque d'araignées rouges est très probable après une vague de chaleur
sèche.

La phase 2 fournit ce signal contextuel sous deux formes :
- un **score par label candidat** : `favorable / neutre / défavorable`
- un **scénario environnemental global** lisible : `chaud_et_sec`,
  `humide_et_pluvieux`, `doux_et_humide`, etc.

## Données d'entrée

| Source                  | Apport                                                |
| ----------------------- | ----------------------------------------------------- |
| `pest-diagnostic/`      | EXIF → date d'observation                             |
| `manifest_images.csv`   | pivot : `id_bulletin`, `filename`, `Garden ID`, `nom_espece`, `nom_bioagresseur`, `Status`, `Pest Type`, etc. |
| `gardens.csv`           | `id` → lat/lon, postal_code, sunshine_type, plant_location |

Le `manifest_images.csv` agit comme **table pivot** : il contient déjà le
lien `filename ↔ id_bulletin` ainsi que les libellés en clair pour
espèce et bioagresseur.

### Schéma de manifest_images.csv

```
id_bulletin, is_test, filename, image_hash, extension,
bioagresseur_filename, nom_bioagresseur,
id_espece_filename, ID Espèce, nom_espece,
ID BioAgresseur, Pest Type, Parties inféctées, Symptoms Description,
Status, Type, User ID, Garden ID, Garden Plant ID,
mtime, size_bytes, path, espece_match, bio_name_match
```

Convention de nommage des images :

```
<nom_bioagresseur>---<id_espece>---<id_bulletin>---<hashHex>

Araignée rouge---25---3---48e008d924394391d77c1479
       ↑         ↑   ↑              ↑
nom_bioagresseur ID  id_bulletin   image_hash
                Espèce
```

`build_phase2_context.py` n'utilise pas cette convention pour la jointure
(le manifest_images.csv suffit), mais effectue un **sanity check** au
démarrage : pour chaque filename, il vérifie que les 4 segments
correspondent bien aux colonnes du manifest. Si la cohérence tombe sous
95 % sur l'un des contrôles, l'audit le signale explicitement.

### Renommage appliqué dans `build_phase2_context.py`

```python
"id_bulletin"        → "bulletin_id"
"Garden ID"          → "garden_id"
"ID Espèce"          → "espece_id"
"ID BioAgresseur"    → "bioagresseur_id"
"Status"             → "status"
"Pest Type"          → "pest_type"
"Parties inféctées"  → "parties_infectees"
"User ID"            → "user_id"
"Garden Plant ID"    → "garden_plant_id"
```

### Architecture des jointures

Deux jointures simples, sans heuristique :

1. `manifest_images ⨝ manifest_exif` sur `filename` → ajoute la date EXIF
2. `⨝ garden` sur `garden_id` → ajoute lat/lon et contexte jardin

## Ordre d'exécution

```
1. extract_exif.py            → outputs/manifest_exif.csv          # EXIF de toutes les images (date)
2. build_phase2_context.py    → outputs/phase2_context.csv         # Jointure (image-centric)
3. aggregate_to_bulletins.py  → outputs/phase2_bulletins.csv       # Pivot bulletin-centric
4. fetch_weather.py           → outputs/weather_raw/*.parquet      # Open-Meteo Historical
5. weather_features.py        → outputs/phase2_features.csv        # Agrégation J-15 → J-1
6. weather_rules.py           → scénario + scores par pathologie
```

### Granularité : pourquoi le pivot bulletin ?

Une pathologie est diagnostiquée **par bulletin** (= une observation
utilisateur), pas image par image. La météo est un signal contextuel
de l'observation, pas du pixel. Si un bulletin contient 3 photos prises
au même endroit le même jour, on ne veut interroger l'API météo qu'**une
seule fois** et lui associer un scénario unique.

L'étape 3 (`aggregate_to_bulletins.py`) fait ce pivot et calcule :

| Champ                  | Stratégie d'agrégation                              |
| ---------------------- | --------------------------------------------------- |
| `n_images`             | Compte                                              |
| `datetime_observation` | Médiane des dates EXIF                              |
| `garden_id`, `lat/lon` | Valeur unique (mode si divergence + flag d'audit)   |
| `nom_bioagresseur`     | Identique par construction                          |
| `phase2_ready`         | True si **au moins une** image est ready            |

Les scripts en aval (`fetch_weather.py`, `weather_features.py`)
acceptent un argument `--granularity image|bulletin`. Par défaut,
ils tournent en mode bulletin.

## Étape 1 — `extract_exif.py`

Extrait pour chaque image :
- date EXIF (`DateTimeOriginal` priorité, fallback `DateTime`)
- `file_modification_date` (proxy si EXIF absent)
- modèle de téléphone, software (audit qualité)
- coordonnées GPS si présentes (rares en pratique)

**Question critique :** Combien d'images ont une vraie date EXIF
exploitable ? Si < 50 %, on bascule sur `file_modification_date`.

## Étape 2 — `build_phase2_context.py`

Joint les trois sources et produit `phase2_context.csv`.

### Décodage des id

```python
PLANT_LOCATION_MAP = {1: "INSIDE", 2: "WINDOW", 3: "BALCONY",
                      4: "GARDEN", 5: "GREENHOUSE"}
SUNSHINE_TYPE_MAP = {1: "SUN", 2: "HALF_SHADOW", 3: "SHADOW"}
```

### Drapeaux de qualité — sémantique outdoor strict vs loose

Deux niveaux d'éligibilité, selon que la météo extérieure est pleinement
pertinente ou seulement faiblement pertinente :

| Catégorie          | Plant locations                                 | Pertinence météo extérieure |
| ------------------ | ----------------------------------------------- | --------------------------- |
| `outdoor` (strict) | `GARDEN`, `BALCONY`, `GREENHOUSE`               | Forte à moyenne             |
| `outdoor_loose`    | `GARDEN`, `BALCONY`, `GREENHOUSE`, **WINDOW**   | + cas faible (fenetre)      |

Une plante derrière une vitre (`WINDOW`) vit dans le climat intérieur :
la pluie ne la mouille pas, la température suit le chauffage du logement.
Mais l'utilisateur a renseigné une localisation précise, et certaines
fenêtres sont ouvertes ; on conserve donc ces bulletins dans la version
*loose* pour l'audit et les analyses exploratoires.

| Flag                  | Définition                                                  |
| --------------------- | ----------------------------------------------------------- |
| `outdoor`             | `plant_location ∈ {GARDEN, BALCONY, GREENHOUSE}`            |
| `outdoor_loose`       | `plant_location ∈ {GARDEN, BALCONY, GREENHOUSE, WINDOW}`    |
| `phase2_ready`        | image + date EXIF + coordonnées + `outdoor` (strict)        |
| `phase2_ready_loose`  | image + date EXIF + coordonnées + `outdoor_loose`           |
| `weather_relevance`   | `high` (GARDEN) / `medium` (BALCONY/GREENHOUSE) / `low` (WINDOW/INSIDE) |

**Règle d'usage :**
- **Calibration des règles agronomiques et fusion finale** → `phase2_ready`
- **Statistiques d'audit, exploration, monitoring** → `phase2_ready_loose`


## Étape 2bis — `aggregate_to_bulletins.py`

Transforme la table `phase2_context.csv`, centrée image, en une table
`phase2_bulletins.csv`, centrée bulletin.

Cette étape est nécessaire car le diagnostic et la météo doivent être
interprétés au niveau de l'observation utilisateur, c'est-à-dire du bulletin,
et non au niveau de chaque image. Un même bulletin peut contenir plusieurs
photos prises au même endroit et au même moment ; il ne faut donc pas
interroger l'API météo une fois par image.

### Entrée

```text
outputs/phase2_context.csv
```

## Étape 3a — `fetch_weather.py`

Interroge l'API **Open-Meteo Historical Weather** (gratuite, sans clé).
Architecture HTTP empruntée au pipeline climatique d'un collègue, pour la
robustesse en production :

- Session `requests` avec `urllib3.Retry` sur 429 / 5xx (backoff
  exponentiel × 2, 4 tentatives, respect du `Retry-After`)
- Timeouts (connect=20s, read=60s) séparés
- Erreurs permanentes (4xx hors 429) → pas de retry
- Cache parquet par `(lat, lon, date)` arrondis à 4 décimales

```bash
# Test sur 50 bulletins
python fetch_weather.py --limit 50

# Run complet (par défaut, mode loose pour récupérer les WINDOW
# en audit ; les bulletins WINDOW ne seront pas utilisés dans la
# fusion finale mais on a les données disponibles)
python fetch_weather.py

# Mode strict (outdoor au sens strict : GARDEN, BALCONY, GREENHOUSE)
python fetch_weather.py --ready-mode strict
```

### Variables récupérées

15 variables quotidiennes sur la fenêtre J-15 → J-1 :

| Catégorie    | Variables                                                              |
| ------------ | ---------------------------------------------------------------------- |
| Température  | tmean, tmax, tmin, apparent_temperature_mean                           |
| Pluie        | precipitation_sum, rain_sum                                            |
| Humidité     | rh_mean / rh_min / rh_max, **dew_point_2m_mean**, **vpd_max**          |
| Vent         | wind_speed_max, wind_gusts_max                                         |
| Énergie      | shortwave_radiation_sum, et0_fao_evapotranspiration                    |
| Sol          | soil_moisture_0_to_7cm_mean                                            |

| Variable                      | Signification                               |
| ----------------------------- | ------------------------------------------- |
| `temperature_2m_mean`         | Température moyenne à 2 m                   |
| `temperature_2m_max`          | Température maximale à 2 m                  |
| `temperature_2m_min`          | Température minimale à 2 m                  |
| `apparent_temperature_mean`   | Température ressentie moyenne               |
| `precipitation_sum`           | Cumul total des précipitations              |
| `rain_sum`                    | Cumul de pluie                              |
| `relative_humidity_2m_mean`   | Humidité relative moyenne à 2 m             |
| `relative_humidity_2m_max`    | Humidité relative maximale à 2 m            |
| `relative_humidity_2m_min`    | Humidité relative minimale à 2 m            |
| `dew_point_2m_mean`           | Point de rosée moyen à 2 m                  |
| `vapour_pressure_deficit_max` | Déficit maximal de pression de vapeur (VPD) |
| `wind_speed_10m_max`          | Vitesse maximale du vent à 10 m             |
| `wind_gusts_10m_max`          | Rafales maximales de vent à 10 m            |
| `shortwave_radiation_sum`     | Rayonnement solaire global cumulé           |
| `et0_fao_evapotranspiration`  | Évapotranspiration de référence FAO ET0     |
| `soil_moisture_0_to_7cm_mean` | Humidité moyenne du sol entre 0 et 7 cm     |

> **Pourquoi VPD et dew point ?** Le VPD (déficit de pression de vapeur)
> élevé indique un stress hydrique foliaire qui favorise les acariens. Un
> point de rosée élevé et un VPD bas signalent au contraire des conditions
> de condensation foliaire propices au mildiou et à la cladosporiose.

## Étape 3b — `weather_features.py`

Agrège les séries journalières en vecteur synthétique :

| Catégorie    | Features 15 jours                                              |
| ------------ | -------------------------------------------------------------- |
| Pluie        | `rain_sum`, `rain_days`, `heavy_rain_days`, `dry_days`, `max_dry_streak` |
| Humidité     | `humidity_mean/max/min`, `high_humidity_days` (≥85 %)         |
| Vapeur       | `dew_point_mean`, `vpd_mean`, `vpd_max`                       |
| Température  | `temp_mean/max/min`, `hot_days` (>28 °C), `cold_days` (<10 °C), `heatwave_events` (≥3 j ≥30 °C) |
| Vent         | `wind_max`, `gust_max`                                         |
| Bilan        | `et0_sum`, `water_balance` (rain − ET0), `radiation_sum`       |
| Sol          | `soil_moisture_mean`, `soil_moisture_min`                      |

Les helpers `longest_streak()` et `count_events_with_min_length()` sont
repris du pipeline du collègue.

## Étape 4 — `weather_rules.py`

Deux niveaux de sortie :

### 1. Score par pathologie

Six labels couverts pour la tomate :
- mildiou (`tomato_late_blight`)
- alternariose (`tomato_early_blight`)
- araignée rouge (`tomato_spider_mites_*`)
- cladosporiose (`tomato_leaf_mold`)
- septoriose (`tomato_septoria_leaf_spot`)
- tache bactérienne (`tomato_bacterial_spot`)

Ratio de conditions favorables → verdict
`favorable / neutre / defavorable`.

### 2. Scénario environnemental global

Inspiré de `classify_environmental_scenario()` du pipeline du collègue,
adapté à la fenêtre courte 15 jours :

```
chaud_et_sec | humide_et_pluvieux | doux_et_humide | sec | chaud
| pluvieux | modere
```

Plus une fonction `build_summary_text()` qui génère un résumé en français
prêt à intégrer dans un journal de bord ou la sortie utilisateur.

> Les seuils actuels sont des ordres de grandeur. La calibration
> agronomique reste à effectuer avec David et la banque experte.

## Métriques d'audit à reporter

| Métrique                       | Objectif                           |
| ------------------------------ | ---------------------------------- |
| % images avec EXIF date        | Audit qualité des données          |
| % bulletins phase2_ready       | Couverture utilisable              |
| Diversité des téléphones       | Représentativité                   |
| Cohérence dates EXIF / upload  | Détection d'horloges mal réglées   |
| Distribution `weather_relevance` | Répartition outdoor / indoor     |
| Distribution scénarios         | Répartition saisonnière des cas    |

## Dépendances

```
pip install pandas pillow requests pyarrow urllib3
```

## Statut

- [x] Squelette des cinq scripts
- [x] Adaptation aux noms de colonnes réels (`Garden ID`, `Status`, etc.)
- [x] Session HTTP robuste reprise du collègue
- [x] Variables enrichies (VPD, dew point, soil moisture)
- [x] Scénario environnemental global
- [x] Architecture simplifiée via `manifest_images.csv` (pivot direct)
- [x] Support des libellés en clair (`nom_espece`, `nom_bioagresseur`)
- [x] Sémantique outdoor strict vs loose pour WINDOW
- [ ] Vérifier l'encodage de `gardens.csv` (Rezé → RezÃ©)
- [ ] Calibrer les seuils de `weather_rules.py` avec David
- [ ] Brancher la sortie sur le module de fusion (CNN + LLM + météo)

## Remerciements

L'architecture HTTP (session avec retries) et plusieurs idées de feature
engineering (longest_streak, count_events_with_min_length, scénarios
environnementaux) sont reprises du pipeline climatique d'un collègue de
DJMA, adaptées à la fenêtre courte J-15 → J-1.
