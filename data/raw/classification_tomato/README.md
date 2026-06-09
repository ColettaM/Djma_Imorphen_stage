# Données brutes — classification tomate

Ce dossier contient les trois archives publiques utilisées pour construire le dataset `tomato_final`.

Archives attendues :

- `TomatoDataset.zip`
- `Crop_Pest_and_Disease_Detection.zip`
- `Plant_leaf_diseases_without_augmentation.zip`

Ces fichiers ne sont pas versionnés dans Git car ils sont volumineux et téléchargeables publiquement.

Le script suivant permet de construire le dataset final :

```bash
python src/classification_tomato/01_build_tomato_dataset.py
```
