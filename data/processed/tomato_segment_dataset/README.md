# tomato_segment_dataset

Dataset segmenté utilisé pour les expériences de classification après isolation des feuilles par SAM3.

## Objectif

Ce dataset a été construit afin d'évaluer l'impact de la suppression du contexte visuel (fond, autres feuilles, environnement) sur les performances de classification des maladies de la tomate.

L'hypothèse testée était qu'un classifieur entraîné uniquement sur la feuille segmentée pourrait mieux distinguer certaines maladies foliaires difficiles.

## Origine des données

Le dataset est généré à partir du dataset :

```text
data/processed/tomato_final/
```

qui constitue le jeu de données de référence utilisé dans les expériences de classification.

## Construction

La segmentation est réalisée avec :

```bash
python src/segmentation_tomato/01_segment_final.py
```

Pour chaque image du dataset source, le pipeline produit :

* un masque binaire ;
* une image segmentée (crop) ;
* une image de visualisation (overlay) ;
* des métadonnées de segmentation.

## Configuration retenue

La configuration finale est issue de l'étude d'ablation réalisée durant le stage.

Paramètres conservés :

* prompt principal : `leaves`
* sliding window : activé
* fallback : `diseased leaves` puis `plant`

Cette configuration a permis d'obtenir une couverture complète du dataset avec un taux de segmentation valide de 100 %.
Malgrès la présence de masques partiels.

## Structure attendue

```text
tomato_segment_dataset/
├── train/
├── val/
└── test/
```

La structure des classes est identique à celle de `tomato_final`.

## Utilisation

Le dataset segmenté est utilisé pour les expériences de classification suivantes :

```bash
python src/classification_tomato/03_train_resnet50_tomato_sam3.py
```

Les performances obtenues sont comparées à celles obtenues sur les images complètes afin d'évaluer l'apport réel de la segmentation.

## Résultat principal

Les expériences ont montré que la suppression du contexte par segmentation SAM3 n'améliore pas les performances globales du classifieur.

Le pipeline basé sur les images complètes obtient des performances légèrement supérieures au pipeline basé sur les feuilles segmentées.

## Remarque

Le contenu complet du dataset n'est pas versionné dans Git en raison de sa taille. Il peut être régénéré à partir de `tomato_final` et du pipeline de segmentation fourni dans ce dépôt.
