"""
phase3_evaluate.py

Evaluation finale de la Phase 3 Vision / BSV.

Ce script est la seule etape ou la verite terrain est revelee. Il joint les
sorties du moteur de compatibilite avec ground_truth.csv, calcule les metriques
CNN Top-1 sur le sous-ensemble decidable par le BSV, puis croise la correction
du CNN avec le statut de compatibilite BSV.

Entrees par defaut
------------------
    data/processed/bsv_compatibility_phase3.csv
    data/processed/phase3/ground_truth.csv

Sorties par defaut
------------------
    outputs/phase3/evaluation/phase3_top1_decidable.csv
    outputs/phase3/evaluation/phase3_metrics_summary.csv
    outputs/phase3/evaluation/phase3_metrics_by_region.csv
    outputs/phase3/evaluation/phase3_status_by_region.csv
    outputs/phase3/evaluation/phase3_confusion_matrix.csv
    outputs/phase3/evaluation/phase3_bsv_contribution.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COMPATIBILITY_FILE = (
    PROJECT_ROOT / "data" / "processed" / "moteur" / "bsv_compatibility_phase3.csv"
)
DEFAULT_GROUND_TRUTH_FILE = (
    PROJECT_ROOT / "data" / "processed" / "moteur" / "phase3" / "ground_truth.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "moteur" / "phase3" / "evaluation"

DECIDABLE_STATUSES = {
    "Coherent",
    "Non_corrobore",
}

COHERENT_STATUSES = {
    "Coherent",
}

CONTRIBUTION_LABELS = {
    (True, "coherent"): "BSV confirme un bon diagnostic",
    (True, "non_corrobore"): "BSV n'apporte rien mais ne nuit pas",
    (False, "coherent"): "BSV valide a tort - cas critique",
    (False, "non_corrobore"): "BSV aurait pu alerter",
    (True, "hote_associe"): "BSV confirme via un hote associe",
    (
        False,
        "hote_associe",
    ): "BSV fournit un signal hote associe sur un diagnostic faux",
}


@dataclass(frozen=True)
class EvaluationMetrics:
    total_rows: int
    total_images: int
    joined_images: int
    decidable_images: int
    accuracy_top1: float
    macro_f1_top1: float


def parse_args() -> argparse.Namespace:
    """Lit les chemins d'entree et le dossier de sortie depuis la CLI."""

    parser = argparse.ArgumentParser(
        description="Evaluate Phase 3 CNN predictions after revealing ground truth."
    )
    parser.add_argument(
        "--compatibility-file",
        type=Path,
        default=DEFAULT_COMPATIBILITY_FILE,
        help=f"CSV produced by moteur_compatibilite.py (default: {DEFAULT_COMPATIBILITY_FILE})",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_FILE,
        help=f"Phase 3 ground truth CSV (default: {DEFAULT_GROUND_TRUTH_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for evaluation outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Charge un CSV en liste de dictionnaires avec gestion du BOM UTF-8."""

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Ecrit un CSV de sortie de facon atomique avec les colonnes demandees."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def require_columns(
    rows: list[dict[str, str]],
    required: set[str],
    source_name: str,
) -> None:
    """Verifie que le fichier charge contient les colonnes obligatoires."""

    fieldnames = set(rows[0]) if rows else set()
    missing = required - fieldnames
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {source_name}: {', '.join(sorted(missing))}"
        )


def parse_score(value: str) -> float:
    """Convertit un score CNN en nombre, avec repli a 0 si la valeur est vide."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_top1_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Retient une seule prediction Top-1 par image, via rank ou score maximum."""

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        image_id = row.get("image_id", "").strip()
        if not image_id:
            continue
        grouped.setdefault(image_id, []).append(row)

    top1: list[dict[str, str]] = []
    for image_id in sorted(grouped):
        candidates = grouped[image_id]
        if "rank" in candidates[0] and any(row.get("rank") for row in candidates):
            selected = min(
                candidates,
                key=lambda row: int(row.get("rank") or "999999"),
            )
        else:
            selected = max(
                candidates,
                key=lambda row: parse_score(row.get("score_cnn", "")),
            )
        top1.append(selected)

    return top1


def join_ground_truth(
    top1_rows: list[dict[str, str]],
    truth_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Revele la verite terrain et ajoute les champs d'evaluation Top-1."""

    truth_by_id = {
        row["observation_id"].strip(): row
        for row in truth_rows
        if row.get("observation_id", "").strip()
    }

    joined: list[dict[str, Any]] = []
    for row in top1_rows:
        image_id = row.get("image_id", "").strip()
        truth = truth_by_id.get(image_id)
        if truth is None:
            continue

        true_label = truth.get("true_label", "").strip()
        predicted_label = row.get("classe_cnn_predite", "").strip()
        cnn_correct = bool(true_label and predicted_label == true_label)

        joined.append(
            {
                **row,
                "observation_id": image_id,
                "true_label": true_label,
                "original_path": truth.get("original_path", ""),
                "top1_label": predicted_label,
                "top1_score": row.get("score_cnn", ""),
                "cnn_correct": cnn_correct,
                "bsv_decidable": row.get("compatibilite", "") in DECIDABLE_STATUSES,
                "bsv_contribution_case": contribution_case(
                    cnn_correct,
                    row.get("compatibilite", ""),
                    row.get("niveau_preuve", ""),
                ),
            }
        )

    return joined


def contribution_family(compatibility: str, evidence_level: str = "") -> str:
    """Regroupe les statuts BSV en familles utiles pour l'analyse scientifique."""

    if compatibility in COHERENT_STATUSES and evidence_level == "hote_associe":
        return "hote_associe"
    if compatibility in COHERENT_STATUSES:
        return "coherent"
    if compatibility == "Non_corrobore":
        return "non_corrobore"
    return "non_decidable"


def contribution_case(
    cnn_correct: bool,
    compatibility: str,
    evidence_level: str = "",
) -> str:
    """Traduit le croisement CNN/BSV en interpretation lisible."""

    family = contribution_family(compatibility, evidence_level)
    return CONTRIBUTION_LABELS.get((cnn_correct, family), "BSV non decidable")


def accuracy(rows: list[dict[str, Any]]) -> float:
    """Calcule l'accuracy Top-1 sur les lignes evaluables."""

    if not rows:
        return 0.0
    return sum(bool(row["cnn_correct"]) for row in rows) / len(rows)


def macro_f1(rows: list[dict[str, Any]]) -> float:
    """Calcule le F1 macro sans dependance externe comme scikit-learn."""

    labels = sorted(
        {row["true_label"] for row in rows if row.get("true_label")}
        | {row["top1_label"] for row in rows if row.get("top1_label")}
    )
    if not labels:
        return 0.0

    scores: list[float] = []
    for label in labels:
        tp = sum(
            row["true_label"] == label and row["top1_label"] == label for row in rows
        )
        fp = sum(
            row["true_label"] != label and row["top1_label"] == label for row in rows
        )
        fn = sum(
            row["true_label"] == label and row["top1_label"] != label for row in rows
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        scores.append(f1)

    return sum(scores) / len(scores)


def confusion_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Construit la matrice de confusion au format lignes CSV."""

    labels = sorted(
        {row["true_label"] for row in rows} | {row["top1_label"] for row in rows}
    )
    counts = Counter((row["true_label"], row["top1_label"]) for row in rows)

    matrix_rows: list[dict[str, Any]] = []
    for true_label in labels:
        matrix_row: dict[str, Any] = {"true_label": true_label}
        for predicted_label in labels:
            matrix_row[predicted_label] = counts[(true_label, predicted_label)]
        matrix_rows.append(matrix_row)

    return matrix_rows


def contribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrege les cas CNN correct/faux croises avec le statut BSV."""

    counts = Counter(
        (
            "correct" if row["cnn_correct"] else "incorrect",
            contribution_family(
                row.get("compatibilite", ""),
                row.get("niveau_preuve", ""),
            ),
            row["bsv_contribution_case"],
        )
        for row in rows
    )
    total = len(rows)

    output: list[dict[str, Any]] = []
    for (cnn_status, bsv_status, interpretation), count in sorted(counts.items()):
        output.append(
            {
                "cnn_status": cnn_status,
                "bsv_status": bsv_status,
                "interpretation": interpretation,
                "count": count,
                "share": round(count / total, 6) if total else 0.0,
            }
        )
    return output


def metrics_by_region_rows(
    top1_joined_rows: list[dict[str, Any]],
    decidable_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calcule les compteurs et métriques Top-1 décidables par région."""

    joined_by_region: dict[str, list[dict[str, Any]]] = {}
    decidable_by_region: dict[str, list[dict[str, Any]]] = {}

    for row in top1_joined_rows:
        region = row.get("region", "") or "(vide)"
        joined_by_region.setdefault(region, []).append(row)
    for row in decidable_rows:
        region = row.get("region", "") or "(vide)"
        decidable_by_region.setdefault(region, []).append(row)

    output: list[dict[str, Any]] = []
    for region in sorted(joined_by_region):
        region_decidable = decidable_by_region.get(region, [])
        output.append(
            {
                "region": region,
                "joined_images": len(joined_by_region[region]),
                "decidable_images": len(region_decidable),
                "decidable_share": round(
                    len(region_decidable) / len(joined_by_region[region]),
                    6,
                )
                if joined_by_region[region]
                else 0.0,
                "accuracy_top1": round(accuracy(region_decidable), 6),
                "macro_f1_top1": round(macro_f1(region_decidable), 6),
            }
        )
    return output


def status_by_region_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compte les statuts de compatibilité BSV par région."""

    counts = Counter(
        (
            row.get("region", "") or "(vide)",
            row.get("compatibilite", "") or "(vide)",
        )
        for row in rows
    )
    totals = Counter(row.get("region", "") or "(vide)" for row in rows)

    output: list[dict[str, Any]] = []
    for (region, compatibility), count in sorted(counts.items()):
        output.append(
            {
                "region": region,
                "compatibilite": compatibility,
                "count": count,
                "share": round(count / totals[region], 6) if totals[region] else 0.0,
            }
        )
    return output


def build_metrics(
    compatibility_rows: list[dict[str, str]],
    top1_joined_rows: list[dict[str, Any]],
    decidable_rows: list[dict[str, Any]],
) -> EvaluationMetrics:
    """Prepare les compteurs globaux et les metriques finales."""

    return EvaluationMetrics(
        total_rows=len(compatibility_rows),
        total_images=len({row.get("image_id", "") for row in compatibility_rows}),
        joined_images=len(top1_joined_rows),
        decidable_images=len(decidable_rows),
        accuracy_top1=accuracy(decidable_rows),
        macro_f1_top1=macro_f1(decidable_rows),
    )


def evaluate(
    compatibility_file: Path,
    ground_truth_file: Path,
    output_dir: Path,
) -> EvaluationMetrics:
    """Execute l'evaluation complete et ecrit tous les fichiers de sortie."""

    compatibility_rows = read_csv(compatibility_file)
    truth_rows = read_csv(ground_truth_file)

    require_columns(
        compatibility_rows,
        {
            "image_id",
            "classe_cnn_predite",
            "score_cnn",
            "compatibilite",
            "niveau_preuve",
        },
        str(compatibility_file),
    )
    require_columns(
        truth_rows,
        {"observation_id", "true_label"},
        str(ground_truth_file),
    )

    top1_rows = select_top1_rows(compatibility_rows)
    joined_rows = join_ground_truth(top1_rows, truth_rows)
    decidable_rows = [
        row for row in joined_rows if row.get("compatibilite") in DECIDABLE_STATUSES
    ]

    metrics = build_metrics(compatibility_rows, joined_rows, decidable_rows)

    top1_fields = [
        "observation_id",
        "true_label",
        "top1_label",
        "top1_score",
        "cnn_correct",
        "compatibilite",
        "niveau_preuve",
        "niveau_risque",
        "agent_referentiel",
        "agent_present_bulletin",
        "bsv_decidable",
        "bsv_contribution_case",
        "date_observation",
        "region",
        "decision_reason",
        "original_path",
    ]
    write_csv(
        output_dir / "phase3_top1_decidable.csv",
        decidable_rows,
        top1_fields,
    )

    write_csv(
        output_dir / "phase3_metrics_summary.csv",
        [
            {
                "total_compatibility_rows": metrics.total_rows,
                "total_images": metrics.total_images,
                "joined_images": metrics.joined_images,
                "decidable_images": metrics.decidable_images,
                "accuracy_top1": round(metrics.accuracy_top1, 6),
                "macro_f1_top1": round(metrics.macro_f1_top1, 6),
            }
        ],
        [
            "total_compatibility_rows",
            "total_images",
            "joined_images",
            "decidable_images",
            "accuracy_top1",
            "macro_f1_top1",
        ],
    )

    write_csv(
        output_dir / "phase3_metrics_by_region.csv",
        metrics_by_region_rows(joined_rows, decidable_rows),
        [
            "region",
            "joined_images",
            "decidable_images",
            "decidable_share",
            "accuracy_top1",
            "macro_f1_top1",
        ],
    )

    write_csv(
        output_dir / "phase3_status_by_region.csv",
        status_by_region_rows(compatibility_rows),
        ["region", "compatibilite", "count", "share"],
    )

    matrix = confusion_matrix_rows(decidable_rows)
    matrix_fields = list(matrix[0]) if matrix else ["true_label"]
    write_csv(
        output_dir / "phase3_confusion_matrix.csv",
        matrix,
        matrix_fields,
    )

    write_csv(
        output_dir / "phase3_bsv_contribution.csv",
        contribution_rows(decidable_rows),
        ["cnn_status", "bsv_status", "interpretation", "count", "share"],
    )

    return metrics


def main() -> int:
    """Point d'entree CLI avec affichage d'un resume court."""

    args = parse_args()

    try:
        metrics = evaluate(
            args.compatibility_file.resolve(),
            args.ground_truth_file.resolve(),
            args.output_dir.resolve(),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "Summary: "
        f"{metrics.decidable_images}/{metrics.joined_images} images decidables, "
        f"accuracy={metrics.accuracy_top1:.4f}, "
        f"macro_f1={metrics.macro_f1_top1:.4f}."
    )
    print(f"Output dir: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
