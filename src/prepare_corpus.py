"""
Classer et vérifier les fichiers PDF BSV téléchargés avant l'extraction du texte.

Ce script réalise un audit du corpus de Bulletins de Santé du Végétal (BSV)
avant les étapes d'extraction et d'analyse.

Fonctionnalités principales :
- Parcours récursif de tous les PDF téléchargés.
- Identification du type de document (bulletin, flash, bilan, catalogue, autre).
- Extraction automatique du numéro de bulletin et de la date présents dans le nom du fichier.
- Calcul d'une empreinte MD5 pour détecter les doublons exacts.
- Signalement des doublons potentiels partageant la même date et le même numéro.
- Génération d'un manifeste CSV décrivant l'ensemble du corpus.
- Création d'une liste des fichiers considérés comme exploitables.
- Copie optionnelle du corpus exploitable dans un répertoire dédié.

Aucun fichier n'est supprimé : les documents exclus sont uniquement signalés
dans le manifeste.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "bsv_pdf"
DEFAULT_MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "bsv_corpus_manifest.csv"
DEFAULT_EXPLOITABLE_LIST = PROJECT_ROOT / "data" / "processed" / "bsv_corpus_exploitable.txt"

MANIFEST_COLUMNS = [
    "relative_path",
    "source",
    "filiere",
    "annee_dossier",
    "filename",
    "size_bytes",
    "md5",
    "type_document",
    "numero_bulletin",
    "date_nom_fichier",
    "duplicate_exact",
    "duplicate_of",
    "potential_duplicate_same_number_date",
    "same_date_multiple_numbers",
    "corpus_exploitable",
    "exclusion_reason",
]


def parse_args() -> argparse.Namespace:
    """ 
      Définition des arguments de ligne de commande :
       dossier contenant les PDF BSV téléchargés ; fichier manifeste de sortie ;
       liste des PDF exploitables ; dossier optionnel de copie du corpus exploitable.
    """
    parser = argparse.ArgumentParser(
        description="Classify, hash and filter downloaded BSV PDFs without deleting them."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument(
        "--exploitable-list",
        type=Path,
        default=DEFAULT_EXPLOITABLE_LIST,
    )
    parser.add_argument(
        "--materialize-dir",
        type=Path,
        default=None,
        help="Optional directory where exploitable PDFs are copied with their relative paths.",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    """ Normalise une chaîne de caractères """
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def classify_document(filename: str) -> str:
    """ Détermine le type de document à partir de son nom """
    key = normalize(Path(filename).stem)

    if "catalogue" in key and "formation" in key:
        return "catalogue"
    if "bilan" in key:
        return "bilan"
    if "flash" in key:
        return "flash"
    if "bsv" in key:
        return "bulletin"
    return "autre"


def extract_date_from_filename(filename: str) -> str:
    """ Recherche une date au format AAAAMMJJ dans le nom du fichier."""
    match = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", filename)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def extract_bulletin_number(filename: str) -> str:
    """ Recherche le numéro du bulletin dans le nom du fichier. """
    key = normalize(Path(filename).stem)
    patterns = (
        r"(?:^|_)bsv0*(\d+)(?:_|$)",
        r"(?:^|_)no_?0*(\d+)(?:_|$)",
        r"(?:^|_)n_?0*(\d+)(?:_|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, key)
        if match:
            return str(int(match.group(1)))
    return ""


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """ Calcule l'empreinte MD5 d'un fichier PDF. """
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_pdfs(input_dir: Path) -> list[Path]:
    """ Recherche récursivement tous les fichiers PDF présents. """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(input_dir.rglob("*.pdf"), key=lambda path: str(path).casefold())


def path_metadata(path: Path, input_dir: Path) -> tuple[str, str, str]:
    """ Récupère les métadonnées de classement à partir de l'arborescence. """
    parts = path.relative_to(input_dir).parts
    source = parts[0] if len(parts) >= 1 else ""
    filiere = parts[1] if len(parts) >= 2 else ""
    annee = parts[2] if len(parts) >= 3 else ""
    return source, filiere, annee


def build_manifest(input_dir: Path) -> list[dict[str, str]]:
    """ Construit la table descriptive du corpus. """
    rows: list[dict[str, str]] = []

    for path in discover_pdfs(input_dir):
        source, filiere, annee = path_metadata(path, input_dir)
        document_type = classify_document(path.name)
        rows.append(
            {
                "relative_path": path.relative_to(input_dir).as_posix(),
                "source": source,
                "filiere": filiere,
                "annee_dossier": annee,
                "filename": path.name,
                "size_bytes": str(path.stat().st_size),
                "md5": file_md5(path),
                "type_document": document_type,
                "numero_bulletin": extract_bulletin_number(path.name),
                "date_nom_fichier": extract_date_from_filename(path.name),
                "duplicate_exact": "False",
                "duplicate_of": "",
                "potential_duplicate_same_number_date": "False",
                "same_date_multiple_numbers": "False",
                "corpus_exploitable": str(document_type in {"bulletin", "flash"}),
                "exclusion_reason": (
                    "" if document_type in {"bulletin", "flash"} else document_type
                ),
            }
        )

    mark_duplicates(rows)
    return rows


def mark_duplicates(rows: list[dict[str, str]]) -> None:
    """ Détection des différentes formes de doublons. """

    by_md5: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_identity: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    by_date: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_md5[row["md5"]].append(row)
        if row["numero_bulletin"] and row["date_nom_fichier"]:
            identity = (
                row["source"],
                row["filiere"],
                row["annee_dossier"],
                row["numero_bulletin"],
                row["date_nom_fichier"],
            )
            by_identity[identity].append(row)
            date_key = (
                row["source"],
                row["filiere"],
                row["annee_dossier"],
                row["date_nom_fichier"],
            )
            by_date[date_key].append(row)

    for group in by_md5.values():
        if len(group) < 2:
            continue
        canonical = min(group, key=lambda row: row["relative_path"])
        for row in group:
            if row is canonical:
                continue
            row["duplicate_exact"] = "True"
            row["duplicate_of"] = canonical["relative_path"]
            row["corpus_exploitable"] = "False"
            row["exclusion_reason"] = "duplicate_exact"

    for group in by_identity.values():
        if len(group) > 1:
            for row in group:
                row["potential_duplicate_same_number_date"] = "True"

    for group in by_date.values():
        if len({row["numero_bulletin"] for row in group}) > 1:
            for row in group:
                row["same_date_multiple_numbers"] = "True"


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """ Écrit le manifeste complet au format CSV. """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_exploitable_list(path: Path, rows: list[dict[str, str]]) -> None:
    """ Génère un fichier texte contenant uniquement les chemins """

    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row["relative_path"]
        for row in rows
        if row["corpus_exploitable"] == "True"
    ]
    path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")


def materialize_corpus(
    input_dir: Path,
    destination: Path,
    rows: list[dict[str, str]],
) -> int:
    """ Copie le corpus exploitable dans un répertoire dédié. """

    copied = 0
    for row in rows:
        if row["corpus_exploitable"] != "True":
            continue
        source = input_dir / Path(row["relative_path"])
        target = destination / Path(row["relative_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or file_md5(target) != row["md5"]:
            shutil.copy2(source, target)
            copied += 1
    return copied


def main() -> int:
    """ Fonction principale du script. """
    
    args = parse_args()
    rows = build_manifest(args.input_dir.resolve())
    write_manifest(args.manifest_file.resolve(), rows)
    write_exploitable_list(args.exploitable_list.resolve(), rows)

    copied = 0
    if args.materialize_dir is not None:
        copied = materialize_corpus(
            args.input_dir.resolve(),
            args.materialize_dir.resolve(),
            rows,
        )

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["type_document"]] += 1

    exploitable = sum(row["corpus_exploitable"] == "True" for row in rows)
    duplicates = sum(row["duplicate_exact"] == "True" for row in rows)
    print(f"PDF total: {len(rows)}")
    for document_type in ("bulletin", "flash", "bilan", "catalogue", "autre"):
        print(f"{document_type}: {counts[document_type]}")
    print(f"Doublons exacts exclus: {duplicates}")
    print(f"Corpus exploitable: {exploitable}")
    print(f"Manifest: {args.manifest_file.resolve()}")
    print(f"Liste: {args.exploitable_list.resolve()}")
    if args.materialize_dir is not None:
        print(f"Fichiers copiés: {copied} vers {args.materialize_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
