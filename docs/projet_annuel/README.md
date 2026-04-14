# Validation sémantique des diagnostics de pathologies végétales par le NLI

Ce dépôt contient les travaux réalisés dans le cadre de mon projet annuel de Master 2 Data Science, en collaboration avec **Dr. Jonquille & Mr. Ail (DJMA)**. L'objectif du projet est d'étudier le potentiel de l'**Inférence en Langage Naturel (NLI)** comme mécanisme de **validation sémantique** pour contrôler la cohérence entre :
- des **références expertes** décrivant les maladies des plantes,
- et des **descriptions générées automatiquement** par des modèles multimodaux à partir d'images.

L'idée centrale est la suivante :  **la prédiction visuelle seule** ne garantit pas toujours un diagnostic fiable. Le NLI est donc utilisé ici comme une **couche de contrôle qualité** dans un futur pipeline multimodal.

---

## Contexte

DJMA veut développer de nouvelles fonctionnalités pour leur application mobile dédiée au jardinage. Dans ce cadre, l'une des pistes à l'étude est l'identification automatique des maladies, ravageurs et carences à partir de photos envoyées par les utilisateurs.
Cependant, plusieurs défis se posent :
- des symptômes visuellement similaires peuvent correspondre à des maladies différentes ;
- les modèles multimodaux peuvent produire des descriptions vagues, partielles ou incorrectes ;
- les images des jeux de données publics diffèrent souvent des photos réelles des utilisateurs ;
- le vocabulaire spécialisé de la phytopathologie diffère du langage courant utilisé dans les corpus NLI standards.

Ce projet explore donc l'utilisation du NLI pour vérifier la cohérence sémantique entre la sortie d'un modèle et une référence experte.

---

## Objectifs du Projet

Les principaux objectifs sont :

1. **Analyser les contraintes des données** pour le diagnostic des maladies des plantes
   (maladies, ravageurs, carences, espèces cibles, tâches de vision).
2. **Construire un pipeline NLI** basé sur **DistilBERT + LoRA**.
3. **Valider la méthodologie sur le corpus SNLI**, le benchmark standard du NLI.
4. **Appliquer le pipeline à un cas exploratoire DJMA** en comparant des références expertes à des descriptions générées par **GPT-5** et **Moondream**.
5. **Transformer la sortie NLI en une décision opérationnelle**
   (validation, incertitude, rejet, information incomplète).

---

## Méthodologie

Le projet suit trois étapes principales :

1. **Validation méthodologique sur SNLI**
   DistilBERT est affiné pour la tâche NLI en utilisant LoRA afin de réduire le nombre de paramètres entraînables.

2. **Construction des paires textuelles DJMA**
   Les références expertes sont comparées aux descriptions générées par des modèles multimodaux (GPT-5 et Moondream).

3. **Validation sémantique par NLI**
   Le modèle prédit si la description générée est :
   - une implication (cohérente avec la référence experte),
   - neutre (information insuffisante),
   - une contradiction (incohérente).

---

## Principaux Résultats

### Sur SNLI

Le pipeline DistilBERT + LoRA atteint de bonnes performances sur le benchmark NLI standard :

- Accuracy ≈ 0.89
- F1 macro ≈ 0.89

Cela valide la faisabilité méthodologique de l'approche.

### Sur DJMA

L'application au cas DJMA reste exploratoire en raison de la taille limitée de l'échantillon, mais met en évidence plusieurs points :

- GPT-5 produit généralement des descriptions plus cohérentes que Moondream ;
- les carences restent particulièrement difficiles à identifier ;
- le NLI permet de distinguer les cas :
  - validés,
  - incertains,
  - incomplets,
  - rejetés.

---

## Limitations

Plusieurs limitations sont à noter :

- **Décalage de domaine linguistique** entre le vocabulaire SNLI et celui de la phytopathologie
- **Décalage de domaine visuel** entre les jeux de données publics et les images réelles des utilisateurs
- Taille réduite de l'échantillon DJMA
- Bruit possible introduit par la traduction automatique

---

## Perspectives

Les extensions possibles de ce travail comprennent :

- la construction d'un jeu de données NLI spécifique à la phytopathologie
- l'enrichissement du jeu de données annoté DJMA
- le test d'autres architectures (ex. : DeBERTa, modèles multilingues)
- l'intégration du NLI dans un pipeline de diagnostic multimodal complet

---

## Contributions

Ce travail démontre que le NLI peut servir de **mécanisme de validation sémantique** dans un futur système multimodal de diagnostic des maladies des plantes. La contribution principale n'est pas un diagnostic automatique final, mais une **preuve de concept méthodologique** pour :

- filtrer les descriptions générées,
- détecter les incohérences,
- améliorer la fiabilité d'un pipeline image + texte.

---

## Structure (module NLI)

```text
notebooks/
data/raw/nli/
docs/projet_annuel/
```

---

## Données

- `Inference_LLM.xlsx` : base d'inference des modeles
- `bioravageurs.csv` : base experte
- `public_database_collected.xlsx` : recensement datasets publics

---

## Mots-clés

NLI, DistilBERT, LoRA, Phytopathologie, Vision par Ordinateur, IA Multimodale, Validation Sémantique

---

## Auteur

**Coletta Moutsiha**
Master 2 Data Science — Université d'Angers
