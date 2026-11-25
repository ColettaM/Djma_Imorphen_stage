"""
build_phase3_query.py

Construit le fichier de requetes du moteur BSV a partir des predictions CNN.

Par defaut, le Top-3 est conserve. La sortie utilise un schema ferme qui exclut
la verite terrain, le scenario experimental et son statut de realisation.

Execution :
    python src/build_phase3_query.py
    python src/build_phase3_query.py --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = (
    PROJECT_ROOT / "outputs" / "moteur" / "phase3" / "cnn_topk_predictions.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "moteur" / "phase3" / "requete_phase3.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build moteur_compatibilite.py queries from CNN predictions."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"CNN predictions JSON (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Query JSON written for the compatibility engine (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of predictions to keep per image (default: 3).",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("CNN predictions JSON must contain a list of records")
    return payload


def build_queries(
    records: list[dict[str, Any]], top_k: int = 3
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")

    queries: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not record.get("inference_ok"):
            continue

        predictions = record.get("predictions", [])
        if not predictions:
            continue

        sanitized_predictions = []
        for prediction in predictions[:top_k]:
            classe = prediction.get("classe", prediction.get("label", ""))
            score = prediction.get("score")
            if not classe or score is None:
                raise ValueError(f"Prediction invalide pour l'image #{index}")
            sanitized_predictions.append({"classe": str(classe), "score": float(score)})

        query = {
            "image_id": record.get(
                "image_id", record.get("observation_id", f"obs_{index:06d}")
            ),
            "date_observation": record.get("date_observation", ""),
            "region": record.get("region", ""),
            "top_k": sanitized_predictions,
        }
        queries.append(query)

    return queries


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()

    try:
        records = load_predictions(args.input_file.resolve())
        queries = build_queries(records, top_k=args.top_k)
        write_json(args.output_file.resolve(), queries)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Summary: {len(queries)} requetes ecrites "
        f"avec Top-{args.top_k} vers {args.output_file.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
