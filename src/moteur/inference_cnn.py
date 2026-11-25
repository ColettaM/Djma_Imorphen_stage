"""Inference CNN aveugle de la Phase 3, avec reprise sure des predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


TOP_K = 3
SCORE_MIN = 0.05
BATCH = 64
METADATA_CSV = Path("data/processed/moteur/phase3/phase3_test_with_metadata.csv")
MODEL_PATH = Path("models/best_resnet50_tomato.pth")
CLASSES_PATH = Path("models/class_names.json")
OUT_JSON = Path("outputs/moteur/phase3/cnn_topk_predictions.json")
LOG_FILE = Path("outputs/moteur/logs/inference_cnn.log")

REQUIRED_COLUMNS = {
    "observation_id",
    "image_path",
    "label_masked",
    "date_observation",
    "region",
    "departement",
}
OUTPUT_KEYS = {
    "observation_id",
    "image_path",
    "date_observation",
    "region",
    "departement",
    "predictions",
    "inference_ok",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def charger_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")
    if manifest["observation_id"].duplicated().any():
        raise ValueError("observation_id doit etre unique dans le manifest")
    if not manifest["label_masked"].fillna(False).astype(bool).all():
        raise ValueError("Toutes les images doivent avoir label_masked=True")
    missing_files = [
        value for value in manifest["image_path"] if not Path(str(value)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} images du manifest sont introuvables"
        )
    return manifest


def charger_predictions_existantes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("La sortie CNN existante doit contenir une liste")
    records: dict[str, dict[str, Any]] = {}
    for record in payload:
        observation_id = str(record.get("observation_id", ""))
        if not observation_id or observation_id in records:
            raise ValueError(
                "Identifiant vide ou duplique dans la sortie CNN existante"
            )
        records[observation_id] = record
    return records


def record_reutilisable(
    metadata: pd.Series,
    ancien: dict[str, Any] | None,
    hash_cache: dict[Path, str],
) -> dict[str, Any] | None:
    """Reutilise uniquement une prediction issue exactement des memes pixels."""

    if not ancien or not ancien.get("inference_ok") or not ancien.get("predictions"):
        return None
    ancien_path = Path(str(ancien.get("image_path", "")))
    nouveau_path = Path(str(metadata["image_path"]))
    if not ancien_path.is_file() or not nouveau_path.is_file():
        return None
    for path in (ancien_path, nouveau_path):
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
    if hash_cache[ancien_path] != hash_cache[nouveau_path]:
        return None
    return construire_record(metadata, ancien["predictions"], inference_ok=True)


def construire_record(
    metadata: pd.Series,
    predictions: list[dict[str, Any]],
    inference_ok: bool,
) -> dict[str, Any]:
    record = {
        "observation_id": str(metadata["observation_id"]),
        "image_path": str(metadata["image_path"]),
        "date_observation": str(metadata["date_observation"]),
        "region": str(metadata["region"]),
        "departement": str(metadata["departement"]),
        "predictions": predictions,
        "inference_ok": bool(inference_ok),
    }
    assert set(record) == OUTPUT_KEYS
    return record


def inferer_images(
    rows: list[pd.Series],
    *,
    top_k: int,
    score_min: float,
    batch_size: int,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from PIL import Image, ImageFile
    from torchvision import models, transforms
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(class_names))
    )
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Modele charge sur %s - %s classes", device, len(class_names))

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    results: dict[str, dict[str, Any]] = {}
    for start in tqdm(range(0, len(rows), batch_size), desc="Inference CNN"):
        batch_rows = rows[start : start + batch_size]
        tensors = []
        valid_mask = []
        for row in batch_rows:
            try:
                with Image.open(Path(str(row["image_path"]))) as image:
                    tensors.append(transform(image.convert("RGB")))
                valid_mask.append(True)
            except Exception as exc:
                logger.warning("Image illisible: %s - %s", row["image_path"], exc)
                tensors.append(torch.zeros(3, 224, 224))
                valid_mask.append(False)

        with torch.no_grad():
            probabilities = functional.softmax(
                model(torch.stack(tensors).to(device)), dim=1
            ).cpu()
        scores, indices = torch.topk(
            probabilities, k=min(top_k, len(class_names)), dim=1
        )

        for index, row in enumerate(batch_rows):
            predictions = []
            if valid_mask[index]:
                for rank in range(scores.shape[1]):
                    score = float(scores[index][rank])
                    if score >= score_min:
                        predictions.append(
                            {
                                "rank": len(predictions) + 1,
                                "label": class_names[int(indices[index][rank])],
                                "score": round(score, 4),
                            }
                        )
            record = construire_record(row, predictions, valid_mask[index])
            results[record["observation_id"]] = record
    return results


def ecrire_json_atomique(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-file", type=Path, default=OUT_JSON)
    parser.add_argument("--reuse-file", type=Path, default=OUT_JSON)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--score-min", type=float, default=SCORE_MIN)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--no-reuse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("inference_cnn")
    if args.top_k < 1 or args.batch_size < 1 or args.score_min < 0:
        raise ValueError("Parametres top-k, batch-size ou score-min invalides")

    manifest = charger_manifest(args.metadata_file)
    anciens = {} if args.no_reuse else charger_predictions_existantes(args.reuse_file)
    hash_cache: dict[Path, str] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    a_inferer: list[pd.Series] = []
    for _, row in manifest.iterrows():
        reused = record_reutilisable(
            row, anciens.get(str(row["observation_id"])), hash_cache
        )
        if reused is None:
            a_inferer.append(row)
        else:
            records_by_id[reused["observation_id"]] = reused

    logger.info("Images du manifest: %s", len(manifest))
    logger.info(
        "Predictions reutilisees apres controle SHA-256: %s", len(records_by_id)
    )
    logger.info("Images a inferer: %s", len(a_inferer))
    if a_inferer:
        records_by_id.update(
            inferer_images(
                a_inferer,
                top_k=args.top_k,
                score_min=args.score_min,
                batch_size=args.batch_size,
                logger=logger,
            )
        )

    ordered = [records_by_id[str(obs_id)] for obs_id in manifest["observation_id"]]
    if len(ordered) != len(manifest) or len(
        {row["observation_id"] for row in ordered}
    ) != len(manifest):
        raise AssertionError("La sortie CNN ne couvre pas exactement le manifest")
    ecrire_json_atomique(args.output_file, ordered)
    logger.info(
        "Sortie complete ecrite: %s lignes vers %s", len(ordered), args.output_file
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
