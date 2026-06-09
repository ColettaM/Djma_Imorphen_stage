# tomato_final

Dataset consolidé utilisé pour les expériences de classification et de segmentation des maladies de la tomate.

## Origine des données

Le dataset est construit à partir de trois jeux de données publics :

* TomatoDataset
* Crop_Pest_and_Disease_Detection
* Plant_leaf_diseases_without_augmentation

Les archives d'origine doivent être placées dans :

```text
data/raw/classification_tomato/
├── TomatoDataset.zip
├── Crop_Pest_and_Disease_Detection.zip
└── Plant_leaf_diseases_without_augmentation.zip
```

## Construction

Le dataset est généré à l'aide du script :

```bash
python src/classification_tomato/01_build_tomato_dataset.py
```

Le pipeline réalise notamment :

* extraction des archives ;
* normalisation des noms de classes ;
* suppression des doublons ;
* vérification des images ;
* harmonisation de la structure ;
* création des ensembles train / validation / test.

## Structure attendue

```text
tomato_final/
├── train/
├── val/
└── test/
```

Chaque sous-dossier contient les classes de maladies de la tomate utilisées pour l'entraînement et l'évaluation des modèles.

## Utilisation

Ce dataset sert d'entrée aux expériences de classification :

```bash
python src/classification_tomato/02_train_resnet50_tomato.py
```

et aux expériences de segmentation SAM3 :

```bash
python src/segmentation_tomato/01_segment_final.py
```

## Remarque

Le contenu complet du dataset n'est pas versionné dans Git en raison de sa taille. Il peut être reconstruit à partir des données brutes publiques et du script de préparation fourni dans ce dépôt.
