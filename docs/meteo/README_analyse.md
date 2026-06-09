# Analyse qualité des fiches_utilisateurs — focus `PEST_IDENTIFICATION`

Outils d'exploration des fiches utilisateurs remontés via la fonctionnalité de diagnostic
photo + texte de l'application, focalisée sur le tunnel `PEST_IDENTIFICATION`
(identification assistée), seul producteur d'observations multimodales complètes.

## Décisions méthodologiques

### Périmètre d'analyse (axes 1 à 3)

L'application propose deux fonctionnalités distinctes :
- `PEST_IDENTIFICATION` — tunnel guidé : symptômes → suggestions de diagnostic à valider
- `PEST_REPORT` — déclaration directe : l'utilisateur connaît déjà le bioagresseur

**Seuls les `PEST_IDENTIFICATION` sont retenus pour l'analyse de qualité** (axes 1 à 3),
parce qu'eux seuls sollicitent l'ajout d'images, la description textuelle, la sélection
de symptômes et un diagnostic confirmé/non confirmé. Une annexe rappelle les
statistiques de base sur les `PEST_REPORT`.

### Sous-datasets exploitables (axe 4)

Pour la construction des sous-datasets exploitables, une **inclusion sélective** des
`PEST_REPORT` est appliquée : les bulletins PR qui remplissent les mêmes critères
techniques que les PI (par exemple : image + texte exploitable) sont réintégrés dans
les sous-datasets correspondants. Cette stratégie permet de maximiser le volume de
données réutilisables sans polluer les statistiques de qualité de la collecte.

> ⚠ **Attention sur le sous-dataset `confirmed_only`** : `Status=CONFIRMED` n'a pas
> la même sémantique selon le tunnel. Pour PI, c'est une validation utilisateur d'une
> suggestion algorithmique. Pour PR, c'est l'état par défaut (déclaration directe).
> Le gain apparent sur ce sous-dataset reflète cette différence et **ne signifie pas**
> une amélioration de la fiabilité d'annotation.

### Fiches de test

Le dev a confirmé l'existence de fiches utilisateurs créés par l'équipe pour valider la
fonctionnalité, identifiables car leur `ID` n'apparaît pas dans Metabase mais leurs
images persistent sur Scaleway. Une **inspection visuelle** a confirmé que ces images
sont de même nature que celles produites par les utilisateurs (plantes réelles,
conditions de terrain).

**Choix méthodologique :** ces images sont **incluses** dans le pool exploitable pour
maximiser le volume disponible. Le flag `is_test=True` reste disponible dans
`manifest_images.csv` pour permettre des analyses ciblées si nécessaire.


## Setup (WSL Ubuntu)

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r docs/requirements.txt
```

> **Note WSL :** si le dossier `pest-diagnostic` est sur un point de montage Windows
> (`/mnt/c/...`), garder toujours le flag `--copy` lors du `match_images.py --reorganize` :
> les liens symboliques peuvent ne pas fonctionner sur ce type de montage.

## Workflow

### ⚠ Ordre d'exécution important

**Toujours lancer `match_images.py` AVANT `analyse_bulletins.py` (ou le notebook).**

`analyse_bulletins.py` dépend du fichier `manifest_bulletins.csv` produit par
`match_images.py` pour intégrer le nombre d'images par bulletin et calculer les
sous-datasets (`image_only`, `image_text`, `complet`, etc.). Si le manifest est
absent, le script tourne quand même mais les indicateurs liés aux images sont à
zéro et le tableau de synthèse devient trompeur.

### 1. Matching images ↔ bulletins

```bash
python match_images.py --reorganize --copy
```

Sorties :
- `manifest_images.csv` — ligne par image, avec `is_test`
- `manifest_bulletins.csv` — ligne par bulletin avec `n_images` global et
  `n_images_test` pour traçabilité
- `bulletins_enrichis_avec_noms.csv`
- `matching_report.txt` — couverture globale + ventilation par Type +
  cohérence filename ↔ référentiels
- `by_bulletin/<ID>/<image>` — toutes les images sont placées dans le dossier
  de leur bulletin (les images de test sont identifiables via `is_test` dans
  `manifest_images.csv`)

### 2. Analyse de qualité (focus PI)

```bash
# Mode interactif (recommandé)
jupyter notebook analyse_bulletins.ipynb

# OU mode CLI (produit rapport texte + figures + CSV)
python analyse_bulletins.py
```

Le notebook et le script chargent automatiquement `manifest_bulletins.csv`
pour intégrer le nombre d'images par bulletin et identifier les bulletins
COMPLETS.

Sorties principales :
- `PI_bulletins_enrichis.csv` — bulletins PI avec toutes les colonnes calculées
- `PI_exploitables_texte.csv` — sous-ensemble exploitable côté texte
- `PI_complets.csv` — image + texte + diagnostic confirmé (PI seul)
- `PI_dataset_summary.csv` — synthèse des sous-datasets (avec colonnes étendues
  pour la version PI+PR sélectifs)

Sorties étendues (inclusion sélective des PR) :
- `EXT_image_text.csv`
- `EXT_complet.csv`
- `EXT_weak_labels.csv`
- `EXT_confirmed_only.csv`

Autres :
- `rapport_analyse_PI.txt` — rapport texte complet (script CLI uniquement)
- `figures/*.png` — figures des axes 1, 2 et 4

## Sémantique métier (champs bruts du CSV)

| Champ | Valeurs | Sens |
|---|---|---|
| `Status` | `CONFIRMED` | utilisateur a validé un diagnostic suggéré (PI) ou état par défaut de la déclaration (PR) |
| `Status` | `NOT_CONFIRMED` | utilisateur a quitté sans valider (PI uniquement) |
| `Type` | `PEST_IDENTIFICATION` | parcours guidé symptômes → suggestions |
| `Type` | `PEST_REPORT` | déclaration directe (pas d'aide à l'identification) |

> **Note** : un 3ᵉ statut « demande d'aide » a été mentionné par le développeur mais
> n'apparaît pas dans l'export actuel. Cette modalité éventuelle reste à clarifier.

## Indicateurs calculés par les scripts

| Indicateur | Définition |
|---|---|
| `has_image` | au moins une image rattachée au bulletin |
| `desc_filled` | description textuelle non vide après normalisation |
| `parties_filled` | au moins une partie infectée renseignée |
| `avec_diagnostic` | bioagresseur identifié + Status CONFIRMED |
| `exploitable_texte` | espèce + parties + description renseignées |
| `complet` | exploitable_texte + au moins 1 image + avec_diagnostic |
| `weak_label` | image + texte exploitable, sans diagnostic confirmé |

Les bulletins `complet` constituent le sous-ensemble idéal pour un dataset
d'entraînement multimodal annoté. Les `weak_label` constituent un pool plus
large utilisable via pseudo-étiquetage ou annotation experte priorisée.

## Normalisation du vocabulaire (axe 2)

L'analyse du vocabulaire dominant dans les descriptions utilise une normalisation
de niveau 2 :
- Suppression des accents et de la casse pour la clé de regroupement
- Retrait du `s` final si pluriel (avec table de cas spéciaux pour les mots qui
  se terminent naturellement par `s` au singulier : `sans`, `sous`, `puis`,
  `mois`, `fois`, etc.)
- Le libellé d'affichage est la forme la plus fréquente effectivement observée

Ainsi, `tâches`, `tâche`, `taches`, `tache` sont fusionnés en un seul groupe
dont le libellé sera la variante la plus fréquente, avec un comptage cumulé
sur l'ensemble des variantes.
