"""
Extraction du texte complet des Bulletins de Santé du Végétal (BSV).

Objectif :
    - Parcourir récursivement les PDF présents dans data/raw/
    - Extraire le texte de chaque page  (PyMuPDF  → .txt)
    - Extraire les tableaux de chaque page (pdfplumber → .csv)
    - Conserver un journal d'exécution (log)

Sorties :
    outputs/extracted_text/   ← texte brut (.txt, un par bulletin)
    outputs/tables/           ← tableaux (.csv, un par tableau par page)
    outputs/logs/extraction_log.csv

Usage :
    python src/extract_text.py
    python src/extract_text.py --force
    python src/extract_text.py --no-tables
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Chemins du projet

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR  = PROJECT_ROOT / "data" / "raw" / "bsv_pdf"
DEFAULT_CORPUS_LIST = (
    PROJECT_ROOT / "data" / "processed" / "bsv_corpus_exploitable.txt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "extracted_text"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_LOG_FILE   = PROJECT_ROOT / "outputs" / "logs" / "extraction.log"


# Résumé de l'exécution

@dataclass
class ExtractionSummary:
    """Stocke les compteurs globaux de l'extraction."""

    extracted: int = 0  # Nombre de PDF effectivement extraits
    skipped:   int = 0  # Nombre de PDF ignorés car déjà extraits
    failed:    int = 0  # Nombre de PDF en erreur
    tables:    int = 0  # Nombre total de tableaux CSV écrits


# Arguments de ligne de commande

def parse_args() -> argparse.Namespace:
    """
    Lit les options fournies au lancement du script.

    Exemple :
        python src/extract_text.py --force
        python src/extract_text.py --no-tables
    """

    parser = argparse.ArgumentParser(
        description="Extraction texte + tableaux des PDF BSV"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Répertoire contenant les PDF"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie des fichiers texte"
    )

    parser.add_argument(
        "--corpus-list",
        type=Path,
        default=DEFAULT_CORPUS_LIST,
        help="Liste des chemins relatifs des PDF exploitables"
    )

    parser.add_argument(
        "--all-pdfs",
        action="store_true",
        help="Ignorer la liste du corpus et parcourir tous les PDF"
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Répertoire de sortie des tableaux CSV"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Réécrire les fichiers déjà extraits"
    )

    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Désactiver l'extraction des tableaux"
    )

    return parser.parse_args()


# Découverte des PDF

def discover_pdfs(
    input_dir: Path,
    corpus_list: Path | None = None,
) -> list[Path]:
    """
    Recherche tous les fichiers PDF dans le dossier d'entrée et ses sous-dossiers.

    Exemple :
        data/raw/grand_est_2025/*.pdf
        data/raw/pays_de_la_loire_2025/*.pdf
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {input_dir}")

    input_dir = input_dir.resolve()

    if corpus_list is None:
        return sorted(
            p for p in input_dir.rglob("*.pdf")
            if p.is_file()
        )

    if not corpus_list.exists():
        raise FileNotFoundError(
            f"Liste du corpus introuvable : {corpus_list}. "
            "Lancer d'abord src/prepare_corpus.py ou utiliser --all-pdfs."
        )

    pdf_paths: list[Path] = []
    missing_paths: list[str] = []
    seen: set[Path] = set()

    for raw_line in corpus_list.read_text(encoding="utf-8-sig").splitlines():
        relative_path = raw_line.strip()
        if not relative_path or relative_path.startswith("#"):
            continue

        pdf_path = (input_dir / Path(relative_path)).resolve()
        try:
            pdf_path.relative_to(input_dir)
        except ValueError as exc:
            raise ValueError(
                f"Chemin hors du corpus dans {corpus_list} : {relative_path}"
            ) from exc

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Fichier non PDF dans {corpus_list} : {relative_path}"
            )
        if not pdf_path.is_file():
            missing_paths.append(relative_path)
            continue
        if pdf_path not in seen:
            seen.add(pdf_path)
            pdf_paths.append(pdf_path)

    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        suffix = "..." if len(missing_paths) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing_paths)} PDF de la liste sont introuvables : "
            f"{preview}{suffix}"
        )

    return sorted(pdf_paths)


# Construction des chemins de sortie

def txt_path_for(
    pdf_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Construit le chemin du fichier texte associé à un PDF.

    Exemple :
        data/raw/grand_est_2025/BSV01.pdf
        → outputs/extracted_text/grand_est_2025/BSV01.txt
    """

    return (
        output_dir / pdf_path.relative_to(input_dir)
    ).with_suffix(".txt")


def table_dir_for(
    pdf_path: Path,
    input_dir: Path,
    tables_dir: Path,
) -> Path:
    """
    Construit le dossier de sortie des tableaux associés à un PDF.

    Exemple :
        data/raw/grand_est_2025/BSV01.pdf
        → outputs/tables/grand_est_2025/BSV01/
    """

    return tables_dir / pdf_path.relative_to(input_dir).with_suffix("")


# Extraction texte avec PyMuPDF

def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """
    Extrait le texte complet d'un PDF avec PyMuPDF.

    Chaque page est précédée d'un marqueur :
        --- PAGE 1 ---
        --- PAGE 2 ---

    Ces marqueurs serviront ensuite à retrouver les sections par page.
    """

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF est requis pour extraire le texte. "
            "Installer les dependances avec pip install -r requirements.txt."
        ) from exc

    page_texts = []

    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("PDF protégé par mot de passe")

        nb_pages = len(document)

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()

            page_texts.append(
                f"--- PAGE {page_number} ---\n{text}"
            )

    return "\n\n".join(page_texts), nb_pages


# Extraction des tableaux avec pdfplumber

def extract_pdf_tables(
    pdf_path:   Path,
    input_dir:  Path,
    tables_dir: Path,
    force:      bool = False,
) -> int:
    """
    Extrait les tableaux détectés par pdfplumber et les sauvegarde en CSV.

    Un fichier CSV est créé pour chaque tableau détecté.

    Exemple :
        outputs/tables/grand_est_2025/BSV01/
            page_03_table_1.csv
            page_04_table_1.csv

    Retour :
        nombre de tableaux effectivement écrits.
    """

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber est requis pour extraire les tableaux. "
            "Installer les dependances avec pip install -r requirements.txt."
        ) from exc

    bulletin_table_dir = table_dir_for(
        pdf_path,
        input_dir,
        tables_dir,
    )

    n_tables_written = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table_idx, table in enumerate(tables, start=1):
                csv_path = (
                    bulletin_table_dir
                    / f"page_{page_num:02d}_table_{table_idx}.csv"
                )

                # Si le tableau existe déjà, on ne le réécrit pas sauf si --force est utilisé.
                if csv_path.exists() and not force:
                    continue

                csv_path.parent.mkdir(parents=True, exist_ok=True)

                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

                    for row in table:
                        # Les cellules None sont remplacées par une chaîne vide
                        # pour éviter les valeurs nulles dans les CSV.
                        writer.writerow(
                            ["" if cell is None else cell for cell in row]
                        )

                n_tables_written += 1

    return n_tables_written


# Journalisation

def write_log(log_file: Path, row: dict) -> None:
    """
    Ajoute une ligne au fichier de log.

    Le log permet de garder une trace :
        - des PDF extraits ;
        - des PDF ignorés ;
        - des erreurs ;
        - du nombre de pages ;
        - du nombre de tableaux extraits.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_file.exists()

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "pdf",
                "status",
                "pages",
                "tables",
                "message",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# Extraction globale du corpus

def extract_all(
    input_dir:   Path,
    output_dir:  Path,
    tables_dir:  Path,
    force:       bool = False,
    with_tables: bool = True,
    corpus_list: Path | None = None,
) -> ExtractionSummary:
    """
    Parcourt tous les PDF du corpus et lance les extractions.

    Pour chaque PDF :
        1. extraction du texte complet en .txt ;
        2. extraction optionnelle des tableaux en .csv ;
        3. écriture d'une ligne dans le journal d'exécution.

    Si le fichier .txt existe déjà, le PDF est ignoré sauf si --force est utilisé.
    """

    summary = ExtractionSummary()

    pdf_paths = discover_pdfs(input_dir, corpus_list)

    if not pdf_paths:
        print("Aucun PDF trouvé.")
        return summary

    for pdf_path in pdf_paths:
        destination = txt_path_for(
            pdf_path,
            input_dir,
            output_dir,
        )

        # Si le texte a déjà été extrait, on ignore le PDF sauf demande explicite.
        if destination.exists() and not force:
            summary.skipped += 1

            write_log(
                DEFAULT_LOG_FILE,
                {
                    "timestamp": datetime.now().isoformat(),
                    "pdf": str(pdf_path),
                    "status": "SKIPPED",
                    "pages": "",
                    "tables": "",
                    "message": "Déjà extrait",
                },
            )

            print(f"SKIP  {pdf_path.relative_to(input_dir)}")
            continue

        try:
            # 1. Extraction du texte brut.
            text, nb_pages = extract_pdf_text(pdf_path)

            destination.parent.mkdir(parents=True, exist_ok=True)

            destination.write_text(
                text,
                encoding="utf-8",
            )

            # 2. Extraction optionnelle des tableaux.
            n_tables = 0

            if with_tables:
                n_tables = extract_pdf_tables(
                    pdf_path,
                    input_dir,
                    tables_dir,
                    force=force,
                )

                summary.tables += n_tables

            summary.extracted += 1

            # 3. Écriture du log de succès.
            write_log(
                DEFAULT_LOG_FILE,
                {
                    "timestamp": datetime.now().isoformat(),
                    "pdf": str(pdf_path),
                    "status": "OK",
                    "pages": nb_pages,
                    "tables": n_tables,
                    "message": "",
                },
            )

            print(
                f"OK    {pdf_path.relative_to(input_dir)} "
                f"({nb_pages} pages, {n_tables} tableaux)"
            )

        except Exception as exc:
            summary.failed += 1

            # Log d'erreur pour conserver la trace du PDF problématique.
            write_log(
                DEFAULT_LOG_FILE,
                {
                    "timestamp": datetime.now().isoformat(),
                    "pdf": str(pdf_path),
                    "status": "ERROR",
                    "pages": "",
                    "tables": "",
                    "message": str(exc),
                },
            )

            print(
                f"ERROR {pdf_path.relative_to(input_dir)} : {exc}",
                file=sys.stderr,
            )

    return summary


# Main

def main() -> int:
    """
    Point d'entrée du programme.

    Rôle :
        - lire les arguments ;
        - lancer l'extraction ;
        - afficher le résumé final.
    """

    args = parse_args()

    try:
        summary = extract_all(
            input_dir=args.input_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            tables_dir=args.tables_dir.resolve(),
            force=args.force,
            with_tables=not args.no_tables,
            corpus_list=(
                None if args.all_pdfs else args.corpus_list.resolve()
            ),
        )

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nRésumé : {summary.extracted} extraits, "
        f"{summary.skipped} ignorés, "
        f"{summary.failed} erreurs, "
        f"{summary.tables} tableaux écrits."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
