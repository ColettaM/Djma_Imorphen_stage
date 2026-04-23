"""
build_tomato_dataset.py
=======================

Pipeline complet de construction d'un dataset de classification de feuilles
de tomate à partir de plusieurs sources zippées hétérogènes.

Étapes :
    1.  Extraction des archives .zip
    2.  Normalisation des noms de classes + fusion inter-datasets
        (incluant les alias connus, ex. tomato_leaf_curl → tomato_yellow_leaf_curl_virus)
    3.  Suppression des images corrompues
    4.  Suppression des doublons exacts (hash MD5)
    5.  Construction d'un DataFrame (filepath, class_name, label)
    6.  Split stratifié différencié :
          - classes >= THRESHOLD images  -> 70/15/15
          - classes <  THRESHOLD images  -> 50/25/25
    7.  Downsampling du train (MAX_PER_CLASS)
    8.  Augmentation ciblée des classes minoritaires du train
    9.  Calcul des class weights (sklearn balanced)
    10. Export : CSV finaux + arborescence train/val/test prête pour ImageFolder
    11. Vérification d'intégrité (chemins, chevauchements, cohérence)
    +   Archive tomato_final.zip (optionnelle)

Usage :
    python build_tomato_dataset.py --data-dir ./dataset
    python build_tomato_dataset.py --data-dir ./data --deep-verify
    python build_tomato_dataset.py --data-dir ./data --no-archive
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Configuration & logging
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tomato")


@dataclass
class PipelineConfig:
    """Paramètres globaux du pipeline."""

    data_dir: Path
    threshold: int = 1000  # seuil split 70/15/15 vs 50/25/25
    max_per_class: int = 2000  # cap pour le downsampling du train
    aug_threshold: int = 800  # classes à augmenter si train < aug_threshold
    aug_target: int = 800  # effectif train cible après augmentation
    random_state: int = 42

    # Détection fond noir
    black_bg_border_fraction: float = 0.1  # 10% de bordure analysée
    black_bg_dark_threshold: int = 30  # pixel sombre si max(R,G,B) < 30
    black_bg_dark_ratio: float = 0.85  # 85% de bordure sombre = fond noir
    black_bg_dirname: str = "black_background_removed"

    extract_dirname: str = "extracted"
    merged_dirname: str = "tomato_only"
    augmented_dirname: str = "augmented_train"
    final_dirname: str = "tomato_final"

    # champs dérivés, remplis dans __post_init__
    extract_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    aug_dir: Path = field(init=False)
    final_dir: Path = field(init=False)
    black_bg_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).resolve()
        self.extract_dir = self.data_dir / self.extract_dirname
        self.output_dir = self.data_dir / self.merged_dirname
        self.aug_dir = self.output_dir / self.augmented_dirname
        self.final_dir = self.output_dir / self.final_dirname
        self.black_bg_dir = self.data_dir / self.black_bg_dirname


# ──────────────────────────────────────────────────────────────────────────────
# Étape 1 : extraction des zips
# ──────────────────────────────────────────────────────────────────────────────


def extract_zips(cfg: PipelineConfig) -> None:
    """Dézippe tous les .zip présents dans data_dir vers extract_dir/<nom_zip>/."""
    cfg.extract_dir.mkdir(parents=True, exist_ok=True)

    zips = [f for f in cfg.data_dir.iterdir() if f.suffix.lower() == ".zip"]
    if not zips:
        log.warning(
            "Aucun .zip trouvé dans %s — on suppose déjà extrait.", cfg.data_dir
        )
        return

    for zpath in zips:
        out = cfg.extract_dir / zpath.stem
        if out.exists() and any(out.iterdir()):
            log.info("Déjà extrait : %s", out.name)
            continue
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out)
        log.info("Dézippé : %s → %s", zpath.name, out)


# ──────────────────────────────────────────────────────────────────────────────
# Étape 2 : normalisation + fusion inter-datasets
# ──────────────────────────────────────────────────────────────────────────────


def normalize_class_name(folder_name: str) -> Optional[str]:
    """
    Normalise un nom de dossier de classe.

    Règles :
    - lowercase, ascii, remplacement des séparateurs par '_'
    - n'accepte que les classes commençant par 'tomato' ou 'tomatoes'
    - supprime les préfixes répétés ('tomato_tomato_...')
    - retourne None si la classe n'est pas une classe tomate
    """
    name = folder_name.strip().lower()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    if not (name.startswith("tomato") or name.startswith("tomatoes")):
        return None

    parts = name.split("_")
    if parts[0] not in {"tomato", "tomatoes"}:
        return None
    parts = parts[1:]
    while parts and parts[0] in {"tomato", "tomatoes"}:
        parts = parts[1:]

    if not parts:
        return "tomato_unknown"
    return "tomato_" + "_".join(parts)


def merge_datasets(cfg: PipelineConfig) -> None:
    """
    Parcourt tous les datasets extraits, normalise les noms de classes tomate,
    et copie les images dans output_dir/<classe_normalisée>/.
    Gère les conflits de noms de fichiers identiques entre datasets.
    """
    if cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # class_file_map[class_name][fname_lower] = [(dataset_name, full_path, original_file)]
    class_file_map: dict[str, dict[str, list[tuple[str, Path, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    class_sources: dict[str, set[str]] = defaultdict(set)
    raw_to_norm: dict[str, str] = {}

    for dataset_dir in cfg.extract_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        for root, _, files in os.walk(dataset_dir):
            folder_name = os.path.basename(root)
            if not any(f.lower().endswith(IMAGE_EXTS) for f in files):
                continue

            norm = normalize_class_name(folder_name)
            if norm is None:
                continue

            raw_to_norm[folder_name] = norm
            class_sources[norm].add(dataset_name)

            for fname in files:
                if fname.lower().endswith(IMAGE_EXTS):
                    fpath = Path(root) / fname
                    class_file_map[norm][fname.lower()].append(
                        (dataset_name, fpath, fname)
                    )

    # Log : correspondance brut → normalisé
    log.info("Classes normalisées : %d", len(raw_to_norm))
    multi = {c: s for c, s in class_sources.items() if len(s) > 1}
    if multi:
        log.info("Classes présentes dans plusieurs datasets :")
        for c, srcs in sorted(multi.items()):
            log.info("  %s ← %s", c, sorted(srcs))

    # Copie finale avec désambiguïsation des conflits
    total_copied = 0
    total_conflicts = 0
    for class_name, file_dict in class_file_map.items():
        class_out = cfg.output_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        for _, entries in file_dict.items():
            if len(entries) == 1:
                _, src, orig = entries[0]
                shutil.copy2(src, class_out / orig)
                total_copied += 1
            else:
                total_conflicts += 1
                for dataset_name, src, orig in entries:
                    new_name = f"{dataset_name}__{orig}"
                    shutil.copy2(src, class_out / new_name)
                    total_copied += 1

    log.info(
        "Copie terminée : %d fichiers, %d conflits résolus",
        total_copied,
        total_conflicts,
    )

    _merge_redundant_classes(cfg)


def _merge_redundant_classes(cfg: PipelineConfig) -> None:
    """
    Fusions manuelles connues : tomato_leaf_curl → tomato_yellow_leaf_curl_virus.
    À étendre ici si d'autres alias sont identifiés.
    """
    aliases = {
        "tomato_leaf_curl": "tomato_yellow_leaf_curl_virus",
    }
    for src_cls, dst_cls in aliases.items():
        src = cfg.output_dir / src_cls
        dst = cfg.output_dir / dst_cls
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)

        for f in src.iterdir():
            if not f.is_file():
                continue
            target = dst / f.name
            if target.exists():
                i = 1
                while (dst / f"{f.stem}__merged{i}{f.suffix}").exists():
                    i += 1
                target = dst / f"{f.stem}__merged{i}{f.suffix}"
            shutil.move(str(f), target)

        if not any(src.iterdir()):
            src.rmdir()
            log.info("Fusion : %s → %s", src_cls, dst_cls)


# ──────────────────────────────────────────────────────────────────────────────
# Étape 3 : nettoyage (corrompues + doublons)
# ──────────────────────────────────────────────────────────────────────────────


def remove_corrupted_images(cfg: PipelineConfig) -> int:
    """Vérifie chaque image et supprime celles qui sont illisibles."""
    bad: list[Path] = []
    for class_dir in sorted(cfg.output_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    im.load()
            except Exception:
                bad.append(f)

    for p in bad:
        try:
            p.unlink()
        except OSError as e:
            log.warning("Suppression impossible : %s (%s)", p, e)

    log.info("Images corrompues supprimées : %d", len(bad))
    return len(bad)


def _file_md5(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def remove_exact_duplicates(cfg: PipelineConfig) -> int:
    """Supprime les doublons exacts (même MD5), en gardant un représentant par groupe."""
    hashes: dict[str, list[Path]] = defaultdict(list)

    for class_dir in sorted(cfg.output_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                try:
                    hashes[_file_md5(f)].append(f)
                except OSError:
                    pass

    removed = 0
    for paths in hashes.values():
        if len(paths) > 1:
            for p in paths[1:]:
                try:
                    p.unlink()
                    removed += 1
                except OSError as e:
                    log.warning("Suppression impossible : %s (%s)", p, e)

    log.info("Doublons exacts supprimés : %d", removed)
    return removed


# ──────────────────────────────────────────────────────────────────────────────
# Étape 4: détection et déplacement des images à fond noir
# ──────────────────────────────────────────────────────────────────────────────


def _has_black_background(
    image_path: Path,
    border_fraction: float = 0.1,
    dark_threshold: int = 30,
    dark_ratio_threshold: float = 0.85,
) -> bool:
    """
    Détecte un fond noir en n'analysant QUE les bordures de l'image.
    Une image est classée "fond noir" si la fraction de pixels sombres
    sur la bordure dépasse dark_ratio_threshold.

    Pixel "sombre" = max(R, G, B) < dark_threshold.
    Utiliser max() (pas la moyenne) évite les faux positifs sur les pixels
    de couleur saturée foncée (bleu nuit, vert profond, etc.).
    """
    with Image.open(image_path) as im:
        arr = np.array(im.convert("RGB"))
    h, w = arr.shape[:2]
    bw = max(1, int(min(h, w) * border_fraction))

    borders = np.concatenate(
        [
            arr[:bw, :, :].reshape(-1, 3),
            arr[-bw:, :, :].reshape(-1, 3),
            arr[:, :bw, :].reshape(-1, 3),
            arr[:, -bw:, :].reshape(-1, 3),
        ]
    )
    dark_mask = borders.max(axis=1) < dark_threshold
    return dark_mask.mean() >= dark_ratio_threshold


def remove_black_background_images(cfg: PipelineConfig) -> int:
    """
    Parcourt tomato_only/, détecte les images à fond noir, et les déplace
    vers <data_dir>/black_background_removed/<classe>/ en préservant
    l'arborescence par classe. Les images déplacées ne sont PAS supprimées
    (audit possible, réintégration si faux positifs).

    Retourne le nombre d'images déplacées.
    """
    if cfg.black_bg_dir.exists():
        shutil.rmtree(cfg.black_bg_dir)
    cfg.black_bg_dir.mkdir(parents=True, exist_ok=True)

    moved_by_class: dict[str, int] = {}
    errors = 0

    for class_dir in sorted(cfg.output_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name in {cfg.augmented_dirname, cfg.final_dirname}:
            continue

        class_out = cfg.black_bg_dir / class_dir.name
        moved_here = 0

        for f in class_dir.iterdir():
            if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
                continue
            try:
                if _has_black_background(
                    f,
                    border_fraction=cfg.black_bg_border_fraction,
                    dark_threshold=cfg.black_bg_dark_threshold,
                    dark_ratio_threshold=cfg.black_bg_dark_ratio,
                ):
                    class_out.mkdir(parents=True, exist_ok=True)
                    dst = class_out / f.name
                    if dst.exists():  # éviter écrasement
                        i = 1
                        while (class_out / f"{f.stem}__{i}{f.suffix}").exists():
                            i += 1
                        dst = class_out / f"{f.stem}__{i}{f.suffix}"
                    shutil.move(str(f), dst)
                    moved_here += 1
            except Exception as e:
                log.warning("Erreur détection fond noir %s : %s", f, e)
                errors += 1

        if moved_here > 0:
            moved_by_class[class_dir.name] = moved_here

    total = sum(moved_by_class.values())
    log.info("Images à fond noir déplacées : %d (erreurs : %d)", total, errors)
    if moved_by_class:
        for cls, n in sorted(moved_by_class.items(), key=lambda x: -x[1]):
            log.info("  %-45s → %d", cls, n)
        log.info("Destination : %s", cfg.black_bg_dir)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Étape 5 : DataFrame + LabelEncoder
# ──────────────────────────────────────────────────────────────────────────────


def build_dataframe(cfg: PipelineConfig) -> tuple[pd.DataFrame, LabelEncoder]:
    """Construit le DataFrame (filepath, class_name, label) à partir de output_dir."""
    rows = []
    for class_dir in sorted(cfg.output_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        # Exclusion des dossiers techniques (augmented_train, tomato_final)
        if class_dir.name in {cfg.augmented_dirname, cfg.final_dirname}:
            continue
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                rows.append({"filepath": str(f), "class_name": class_dir.name})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Aucune image trouvée dans {cfg.output_dir}")

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["class_name"])

    log.info("Dataset : %d images, %d classes", len(df), len(le.classes_))
    return df, le


# ──────────────────────────────────────────────────────────────────────────────
# Étape 6 : split stratifié différencié
# ──────────────────────────────────────────────────────────────────────────────


def stratified_split(
    df: pd.DataFrame, cfg: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split stratifié avec régime différencié selon le volume de la classe.
    - n >= threshold : 70/15/15
    - n <  threshold : 50/25/25 (pour fiabilité stat sur classes rares)
    """
    train_parts, val_parts, test_parts = [], [], []

    for class_name, group in df.groupby("class_name"):
        n = len(group)
        if n >= cfg.threshold:
            train_g, temp_g = train_test_split(
                group, test_size=0.30, random_state=cfg.random_state
            )
            val_g, test_g = train_test_split(
                temp_g, test_size=0.50, random_state=cfg.random_state
            )
            regime = "70/15/15"
        else:
            train_g, temp_g = train_test_split(
                group, test_size=0.50, random_state=cfg.random_state
            )
            val_g, test_g = train_test_split(
                temp_g, test_size=0.50, random_state=cfg.random_state
            )
            regime = "50/25/25"

        train_parts.append(train_g)
        val_parts.append(val_g)
        test_parts.append(test_g)

        log.info(
            "  %-45s [%s] train=%4d val=%3d test=%3d",
            class_name,
            regime,
            len(train_g),
            len(val_g),
            len(test_g),
        )

    def _concat_shuffle(parts):
        return (
            pd.concat(parts)
            .sample(frac=1, random_state=cfg.random_state)
            .reset_index(drop=True)
        )

    train_df = _concat_shuffle(train_parts)
    val_df = _concat_shuffle(val_parts)
    test_df = _concat_shuffle(test_parts)

    log.info(
        "Split total : train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# Étape 7 : downsampling du train
# ──────────────────────────────────────────────────────────────────────────────


def downsample_train(train_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Réduit chaque classe à max_per_class échantillons maximum dans le train."""
    balanced_parts = []
    for class_name, group in train_df.groupby("class_name"):
        if len(group) > cfg.max_per_class:
            group = group.sample(n=cfg.max_per_class, random_state=cfg.random_state)
            log.info("  downsampled %-45s → %d", class_name, cfg.max_per_class)
        balanced_parts.append(group)

    out = (
        pd.concat(balanced_parts)
        .sample(frac=1, random_state=cfg.random_state)
        .reset_index(drop=True)
    )
    log.info("Train après downsampling : %d", len(out))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Étape 8 : augmentation des classes minoritaires
# ──────────────────────────────────────────────────────────────────────────────


def _build_aug_transform() -> transforms.Compose:
    """Augmentations adaptées aux feuilles de tomate (invariances biologiques)."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
            ),
        ]
    )


def augment_minority_classes(
    train_df: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    """
    Génère de nouvelles images augmentées pour chaque classe dont l'effectif
    dans le train est inférieur à aug_threshold, jusqu'à atteindre aug_target.
    """
    if cfg.aug_dir.exists():
        shutil.rmtree(cfg.aug_dir)
    cfg.aug_dir.mkdir(parents=True, exist_ok=True)

    counts = train_df["class_name"].value_counts().to_dict()
    minority = [c for c, n in counts.items() if n < cfg.aug_threshold]

    if not minority:
        log.info("Aucune classe minoritaire sous le seuil (%d).", cfg.aug_threshold)
        return train_df

    aug_transform = _build_aug_transform()
    rng = random.Random(cfg.random_state)
    new_rows = []

    for cls in minority:
        cls_df = train_df[train_df["class_name"] == cls]
        current = len(cls_df)
        needed = cfg.aug_target - current
        if needed <= 0:
            continue

        out_dir = cfg.aug_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        src_paths = cls_df["filepath"].tolist()
        label = int(cls_df["label"].iloc[0])

        log.info(
            "  augmentation %-45s %d → %d (+%d)", cls, current, cfg.aug_target, needed
        )

        generated = 0
        while generated < needed:
            src = rng.choice(src_paths)
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    aug = aug_transform(im)
                save_path = out_dir / f"aug_{generated:04d}.jpg"
                aug.save(save_path, format="JPEG", quality=92)
                new_rows.append(
                    {
                        "filepath": str(save_path),
                        "class_name": cls,
                        "label": label,
                    }
                )
                generated += 1
            except Exception as e:
                log.warning("Échec augmentation %s : %s", src, e)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        train_final = pd.concat([train_df, new_df], ignore_index=True)
    else:
        train_final = train_df

    log.info("Train après augmentation : %d", len(train_final))
    return train_final


# ──────────────────────────────────────────────────────────────────────────────
# Étape 9 : class weights
# ──────────────────────────────────────────────────────────────────────────────


def compute_weights(train_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Calcule les class weights équilibrés à partir du train final."""
    classes = np.array(sorted(train_df["label"].unique()))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["label"]
    )
    idx_to_class = (
        train_df[["label", "class_name"]]
        .drop_duplicates()
        .set_index("label")["class_name"]
        .to_dict()
    )
    weights_df = (
        pd.DataFrame(
            {
                "label": classes,
                "class_name": [idx_to_class[i] for i in classes],
                "weight": weights,
            }
        )
        .sort_values("label")
        .reset_index(drop=True)
    )

    out_path = cfg.output_dir / "class_weights.csv"
    weights_df.to_csv(out_path, index=False)
    log.info("Class weights sauvegardés : %s", out_path)
    return weights_df


# ──────────────────────────────────────────────────────────────────────────────
# Étape 10 : export arborescence finale train/val/test
# ──────────────────────────────────────────────────────────────────────────────


def export_final_tree(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> None:
    """
    Écrit l'arborescence finale tomato_final/{train,val,test}/<classe>/*.jpg,
    directement consommable par torchvision.datasets.ImageFolder.
    """
    if cfg.final_dir.exists():
        shutil.rmtree(cfg.final_dir)
    cfg.final_dir.mkdir(parents=True, exist_ok=True)

    def _copy_split(df_split: pd.DataFrame, split_name: str) -> None:
        split_dir = cfg.final_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df_split.iterrows():
            src = Path(row["filepath"])
            dst_dir = split_dir / row["class_name"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.exists():
                i = 1
                while (dst_dir / f"{src.stem}__{i}{src.suffix}").exists():
                    i += 1
                dst = dst_dir / f"{src.stem}__{i}{src.suffix}"
            shutil.copy2(src, dst)

    _copy_split(train_df, "train")
    _copy_split(val_df, "val")
    _copy_split(test_df, "test")
    log.info("Arborescence finale écrite : %s", cfg.final_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Étape 11 : vérification finale d'intégrité
# ──────────────────────────────────────────────────────────────────────────────


def verify_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    deep_check: bool = False,
) -> dict:
    """
    Vérifie la cohérence du dataset final avant livraison.

    Contrôles :
    - Chaque filepath des CSV existe sur disque
    - Aucune image n'est corrompue (verify + load complet)
    - Aucun chevauchement entre train / val / test
    - Chaque image de tomato_final/ a bien un pendant dans les CSV (taille cohérente)
    - Mapping label ↔ class_name stable

    deep_check=True force un re-verify PIL de chaque image (lent mais exhaustif).
    Par défaut, seul l'existence du chemin est vérifiée ; un re-scan approfondi
    serait redondant avec remove_corrupted_images, mais reste utile si le
    dataset a été manipulé entre-temps.
    """
    report = {
        "missing_paths": [],
        "corrupted": [],
        "overlaps": {},
        "split_vs_tree_mismatch": {},
        "label_inconsistencies": [],
    }

    all_splits = {"train": train_df, "val": val_df, "test": test_df}

    # 1) Chemins existants + images décodables
    for split_name, df_split in all_splits.items():
        for fp in df_split["filepath"]:
            path = Path(fp)
            if not path.exists():
                report["missing_paths"].append((split_name, fp))
                continue
            if deep_check:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        im.load()
                except Exception as e:
                    report["corrupted"].append((split_name, fp, str(e)))

    log.info("  chemins manquants : %d", len(report["missing_paths"]))
    if deep_check:
        log.info("  images corrompues : %d", len(report["corrupted"]))

    # 2) Aucun chevauchement entre splits (filepaths uniques)
    splits_sets = {name: set(df["filepath"]) for name, df in all_splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = splits_sets[a] & splits_sets[b]
        if inter:
            report["overlaps"][f"{a} ∩ {b}"] = len(inter)
    if report["overlaps"]:
        log.warning("  chevauchements détectés : %s", report["overlaps"])
    else:
        log.info("  aucun chevauchement train/val/test")

    # 3) Cohérence entre CSV et arborescence tomato_final/
    if cfg.final_dir.exists():
        for split_name in ["train", "val", "test"]:
            split_dir = cfg.final_dir / split_name
            if not split_dir.exists():
                continue
            tree_count = sum(
                1
                for cls_dir in split_dir.iterdir()
                if cls_dir.is_dir()
                for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
            csv_count = len(all_splits[split_name])
            if tree_count != csv_count:
                report["split_vs_tree_mismatch"][split_name] = {
                    "csv": csv_count,
                    "tree": tree_count,
                }
        if report["split_vs_tree_mismatch"]:
            log.warning(
                "  incohérence CSV/arborescence : %s", report["split_vs_tree_mismatch"]
            )
        else:
            log.info("  CSV et arborescence cohérents")

    # 4) Mapping label ↔ class_name stable (chaque label ↔ 1 seul class_name)
    for split_name, df_split in all_splits.items():
        if "label" not in df_split.columns:
            continue
        check = df_split.groupby("label")["class_name"].nunique()
        bad = check[check > 1]
        if not bad.empty:
            report["label_inconsistencies"].append((split_name, bad.to_dict()))
    if report["label_inconsistencies"]:
        log.warning(
            "  incohérences label↔class_name : %s", report["label_inconsistencies"]
        )
    else:
        log.info("  mapping label↔class_name stable")

    # Verdict global
    n_issues = (
        len(report["missing_paths"])
        + len(report["corrupted"])
        + len(report["overlaps"])
        + len(report["split_vs_tree_mismatch"])
        + len(report["label_inconsistencies"])
    )
    if n_issues == 0:
        log.info("  Dataset VALIDE — aucune anomalie détectée")
    else:
        log.warning("  Dataset contient %d anomalie(s) — voir rapport", n_issues)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Étape 12 : archive zip de tomato_final
# ──────────────────────────────────────────────────────────────────────────────


def zip_final_dataset(cfg: PipelineConfig) -> Path:
    """Crée une archive tomato_final.zip du dossier final, prête à être transférée."""
    if not cfg.final_dir.exists():
        log.warning("Pas d'arborescence finale à zipper : %s", cfg.final_dir)
        return None
    zip_base = cfg.output_dir / "tomato_final"
    archive_path = Path(
        shutil.make_archive(str(zip_base), "zip", root_dir=str(cfg.final_dir))
    )
    log.info(
        "Archive créée : %s (%.1f MB)",
        archive_path,
        archive_path.stat().st_size / (1024**2),
    )
    return archive_path


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde des CSV de split
# ──────────────────────────────────────────────────────────────────────────────


def save_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> None:
    train_df.to_csv(cfg.output_dir / "train_split_final.csv", index=False)
    val_df.to_csv(cfg.output_dir / "val_split_final.csv", index=False)
    test_df.to_csv(cfg.output_dir / "test_split_final.csv", index=False)
    log.info("CSV de split sauvegardés dans %s", cfg.output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    cfg: PipelineConfig, deep_verify: bool = False, make_archive: bool = True
) -> dict:
    log.info("=" * 72)
    log.info("Pipeline de construction du dataset tomate")
    log.info("  data_dir      : %s", cfg.data_dir)
    log.info("  threshold     : %d", cfg.threshold)
    log.info("  max_per_class : %d", cfg.max_per_class)
    log.info("  aug_threshold : %d", cfg.aug_threshold)
    log.info("  aug_target    : %d", cfg.aug_target)
    log.info("=" * 72)

    log.info("[1/12] Extraction des archives zip")
    extract_zips(cfg)

    log.info("[2/12] Normalisation et fusion inter-datasets")
    merge_datasets(cfg)

    log.info("[3/12] Nettoyage : images corrompues")
    remove_corrupted_images(cfg)

    log.info("[4/12] Nettoyage : doublons exacts")
    remove_exact_duplicates(cfg)

    log.info("[5/12] Retrait des images à fond noir")
    remove_black_background_images(cfg)

    log.info("[6/12] Construction du DataFrame")
    df, _ = build_dataframe(cfg)

    log.info("[7/12] Split stratifié différencié")
    train_df, val_df, test_df = stratified_split(df, cfg)
    save_split_csvs(train_df, val_df, test_df, cfg)

    log.info("[8/12] Downsampling du train")
    train_df = downsample_train(train_df, cfg)

    log.info("[9/12] Augmentation des classes minoritaires")
    train_df = augment_minority_classes(train_df, cfg)

    log.info("[10/12] Calcul des class weights")
    compute_weights(train_df, cfg)

    log.info("[11/12] Export final train/val/test (CSV + arborescence)")
    save_split_csvs(train_df, val_df, test_df, cfg)  # écrase avec le train final
    export_final_tree(train_df, val_df, test_df, cfg)

    log.info("[12/12] Vérification finale d'intégrité")
    report = verify_integrity(train_df, val_df, test_df, cfg, deep_check=deep_verify)

    if make_archive:
        log.info("Étape optionnelle : création de l'archive zip")
        zip_final_dataset(cfg)

    log.info("=" * 72)
    log.info("Pipeline terminé.")
    log.info("=" * 72)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construit un dataset de classification de feuilles de tomate."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dossier racine contenant les .zip bruts et les sous-dossiers extraits.",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="Seuil de split 70/15/15 vs 50/25/25 (défaut: 1000).",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        help="Plafond train par classe (défaut: 2000).",
    )
    p.add_argument(
        "--aug-threshold",
        type=int,
        default=800,
        help="Seuil en deçà duquel augmenter (défaut: 800).",
    )
    p.add_argument(
        "--aug-target",
        type=int,
        default=800,
        help="Effectif train cible après augmentation (défaut: 800).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--deep-verify",
        action="store_true",
        help="Re-vérifie PIL chaque image à la fin (lent mais exhaustif).",
    )
    p.add_argument(
        "--no-archive",
        action="store_true",
        help="Ne génère pas l'archive tomato_final.zip en fin de pipeline.",
    )
    p.add_argument(
        "--keep-black-bg",
        action="store_true",
        help="Ne pas retirer les images à fond noir (les garde dans tomato_only/).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PipelineConfig(
        data_dir=args.data_dir,
        threshold=args.threshold,
        max_per_class=args.max_per_class,
        aug_threshold=args.aug_threshold,
        aug_target=args.aug_target,
        random_state=args.random_state,
    )
    try:
        run_pipeline(
            cfg, deep_verify=args.deep_verify, make_archive=not args.no_archive
        )
    except Exception:
        log.exception("Échec du pipeline")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
