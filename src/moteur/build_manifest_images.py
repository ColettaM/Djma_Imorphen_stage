"""
build_manifest_images.py

Construit le jeu de données d'entrée de la Phase 3 à partir du jeu de test.

Objectif
--------
La phase 3 cherche à reproduire le cas réel d'un utilisateur qui envoie une photo
sans connaître le diagnostic. Les images du jeu de test sont donc anonymisées afin
que leur classe ne puisse plus être déduite du chemin d'accès.

Le script réalise automatiquement trois opérations :

1. Copie chaque image du dossier test sous un nom anonyme
   (obs_000001.jpg, obs_000002.jpg, ...)
   → data/processed/phase3/images_unlabeled/

2. Génère le fichier ground_truth.csv contenant la vérité terrain.
   Ce fichier est conservé uniquement pour l'évaluation finale et ne doit
   jamais être utilisé pendant l'inférence.
   → data/processed/phase3/ground_truth.csv

3. Génère le manifest utilisé par les scripts de la phase 3.
   Celui-ci ne contient que les chemins des images anonymisées.
   → data/processed/phase3/phase3_test_manifest_masked.csv

Le script est prévu pour être exécuté une seule fois.
S'il détecte que le dossier images_unlabeled existe déjà, il s'arrête afin
d'éviter d'écraser un jeu d'images déjà anonymisé.
"""

import shutil
import pandas as pd
from pathlib import Path

# Chemins des données

ROOT_TEST = Path("data/raw/tomato_final/test")
UNLABELED = Path("data/processed/moteur/phase3/images_unlabeled")
PHASE3_DIR = Path("data/processed/moteur/phase3")
OUT_TRUTH = PHASE3_DIR / "ground_truth.csv"
OUT_MANIFEST = PHASE3_DIR / "phase3_test_manifest_masked.csv"

# Extensions d'images acceptées
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Sécurité

# On refuse d'écraser un précédent jeu anonymisé.
# L'utilisateur devra supprimer explicitement le dossier pour relancer
# l'anonymisation.
if UNLABELED.exists():
    raise FileExistsError(
        f"{UNLABELED} existe déjà.\n"
        "Supprimez-le manuellement si vous souhaitez refaire l'anonymisation :\n"
        f"  rm -rf {UNLABELED}"
    )

# Création des dossiers de sortie
UNLABELED.mkdir(parents=True, exist_ok=True)
PHASE3_DIR.mkdir(parents=True, exist_ok=True)

# Parcours du jeu de test

# Liste des images à traiter.
# Chaque élément est un couple :
#   (nom_de_la_classe, chemin_vers_l_image)
image_files = []

for class_dir in sorted(ROOT_TEST.iterdir()):
    # Ignore les éventuels fichiers présents dans test/
    if not class_dir.is_dir():
        continue

    # Parcours des images de la classe
    for img_path in sorted(class_dir.iterdir()):
        # Conservation uniquement des formats d'image supportés
        if img_path.suffix.lower() in VALID_EXT:
            image_files.append((class_dir.name, img_path))

print(f"Images trouvées : {len(image_files)}")

# Construction des deux tables

# Table de vérité terrain
truth_rows = []

# Manifest utilisé pendant la phase 3
manifest_rows = []

# Chaque image reçoit un identifiant unique de la forme :
# obs_000001, obs_000002, ...
for idx, (true_label, src_path) in enumerate(image_files, start=1):
    obs_id = f"obs_{idx:06d}"

    # Nouveau nom totalement indépendant de la classe
    anon_name = f"{obs_id}.jpg"

    # Chemin de destination de l'image anonymisée
    dst_path = UNLABELED / anon_name

    # Copie de l'image en conservant ses métadonnées
    shutil.copy2(src_path, dst_path)

    # Vérité terrain (non utilisée pendant l'inférence)
    truth_rows.append(
        {
            "observation_id": obs_id,
            "true_label": true_label,
            "original_path": str(src_path),
        }
    )

    # Manifest utilisé par les scripts de la phase 3
    # Aucun label n'apparaît dans ce fichier.
    manifest_rows.append(
        {
            "observation_id": obs_id,
            "image_path": str(dst_path),
            "label_masked": True,
        }
    )

# Export des fichiers CSV

df_truth = pd.DataFrame(truth_rows)
df_manifest = pd.DataFrame(manifest_rows)

df_truth.to_csv(OUT_TRUTH, index=False)
df_manifest.to_csv(OUT_MANIFEST, index=False)

# Résumé de l'exécution

print(f"Images anonymisées : {len(df_manifest)}")
print(
    f"Classes            : {df_truth['true_label'].nunique()} "
    f"→ {sorted(df_truth['true_label'].unique())}"
)
print(f"ground_truth.csv   : {OUT_TRUTH}")
print(f"manifest_masked.csv: {OUT_MANIFEST}")
