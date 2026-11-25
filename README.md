# Stage DJMA – Outils IA pour l'Assistance au Jardinage
Ce dépôt contient les travaux développés durant mon stage de Master 2 Data Science en collaboration avec **Dr. Jonquille & Mr. Ail (DJMA)** et l'**Université d'Angers**.
L'objectif du projet est d'explorer et de développer des outils basés sur les données pour améliorer le diagnostic des pathologies des plantes et les recommandations de jardinage dans l'application mobile DJMA.

---
## Aperçu du Projet
Le projet combine plusieurs composantes de data science et d'apprentissage automatique :
- collecte et analyse de jeux de données publics liés aux maladies des plantes
- validation sémantique des diagnostics par Inférence en Langage Naturel (NLI)
- classification et segmentation d’images de plantes
- mise en relation des prédictions du modèle avec les bulletins de santé du végétal (BSV)
---
## Composantes Principales
### 1. Collecte de Jeux de Données sur les Maladies des Plantes
Collecte et analyse de jeux de données publics relatifs aux maladies des plantes, aux ravageurs et aux carences nutritionnelles.
L'objectif est d'identifier des jeux de données adaptés à l'entraînement de modèles d'apprentissage profond pour la détection des maladies des plantes.
### 2. Validation Sémantique par NLI
Exploration de modèles d'Inférence en Langage Naturel (NLI) pour valider les diagnostics de maladies des plantes en comparant :
- des descriptions expertes des maladies des plantes
- des descriptions générées par des modèles d'IA
Pipeline de validation proposé :
```text
Image → Modèle de Vision → Description → Validation NLI → Décision
```
### 3. Vision par ordinateur
Test et évaluation d’approches de classification et de segmentation d’images pour le diagnostic des maladies des plantes, à partir de jeux de données publics.

---

## Structure du Dépôt

```text
Djma_internship/
├── src/       : code principal
├── notebooks/ : exploration et tests
├── tests/     : tests unitaires
├── docs/      : documentation
├── data/      : données (raw, données brutes et processed, données transformées)
├── outputs/   : sorties (images, métriques)
├── ...
└── README.md
```
---

## Utilisation

- **Classification et segmentation.** Consulte les [notebooks de classification](notebooks/classification_tomato/) et les [scripts d’exécution](scripts/).
- **Validation sémantique.** Consulte les [notebooks NLI](notebooks/nli/).
- **Phase 3 BSV.** Le moteur confronte les prédictions du modèle aux informations des bulletins de santé du végétal. La [documentation de la phase 3](docs/moteur/phase3_moteur.md) détaille les dépendances, les données et les fichiers du modèle nécessaires.

Depuis la racine, après installation des dépendances de la phase 3, tu peux simuler les étapes avec :

```bash
python src/moteur/run_pipeline.py --dry-run
```

Les [notebooks de la phase 3](notebooks/moteur/) servent à analyser les sorties du moteur.

---

## Installation

```bash
# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Auteur
**Coletta Moutsiha**
Master 2 Data Science – Université d'Angers
## Organisation Associée
**Dr. Jonquille & Mr. Ail (DJMA)**
