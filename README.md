# Stage DJMA – Outils IA pour l'Assistance au Jardinage
Ce dépôt contient les travaux développés durant mon stage de Master 2 Data Science en collaboration avec **Dr. Jonquille & Mr. Ail (DJMA)** et l'**Université d'Angers**.
L'objectif du projet est d'explorer et de développer des outils basés sur les données pour améliorer le diagnostic des pathologies des plantes et les recommandations de jardinage dans l'application mobile DJMA.

---
## Aperçu du Projet
Le projet combine plusieurs composantes de data science et d'apprentissage automatique :
- collecte et analyse de jeux de données publics liés aux maladies des plantes
- validation sémantique des diagnostics par Inférence en Langage Naturel (NLI)
- développement de modèles de vision par ordinateur pour la détection des maladies des plantes
- intégration de données environnementales (sol, météo, connaissances botaniques)
- exploration d'agents conversationnels pour l'assistance au jardinage
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
​```text
Image → Modèle de Vision → Description → Validation NLI → Décision
​```
### 3. Vision par Ordinateur pour la Détection des Maladies des Plantes
Test et évaluation de modèles d'apprentissage profond pour la détection des maladies des plantes à partir d'images, en utilisant des jeux de données publics.
Les tâches possibles comprennent :
- classification d'images
- détection d'objets
- segmentation d'images
### 4. Données sur le Sol et l'Environnement
Collecte et analyse de données locales sur le sol et l'environnement pour proposer des conseils de jardinage adaptés à chaque zone géographique.
### 5. Assistant de Jardinage Conversationnel
Exploration d'agents conversationnels capables de fournir des conseils de jardinage en s'appuyant sur les connaissances botaniques, les données environnementales et les modèles d'IA.

---

## Structure du Dépôt

```text
Djma_internship/
├── src/       : code principal
├── notebooks/ : exploration et tests
├── tests/     : tests unitaires
├── docs/      : documentation
├── data/      : données (raw, données brutes et processed, données transformées)
├── results/   : sorties (images, métriques)
├── ...
└── README.md
```
---

## Utilisation

À compléter au fur et à mesure du développement.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Auteur
**Coletta Moutsiha**
Master 2 Data Science – Université d'Angers
## Organisation Associée
**Dr. Jonquille & Mr. Ail (DJMA)**
