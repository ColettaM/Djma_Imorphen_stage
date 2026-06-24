"""Extraction des observations phytosanitaires depuis les caches BSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import extract_bulletins

# Chemins du projet

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_TEXT_DIR = PROJECT_ROOT / "outputs" / "extracted_text"
DEFAULT_BULLETINS_FILE = PROJECT_ROOT / "data" / "processed" / "bsv_bulletins.csv"
DEFAULT_TYPES_FILE = PROJECT_ROOT / "data" / "raw" / "referentiel_types.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bsv_observations.csv"
DEFAULT_LOG_FILE = PROJECT_ROOT / "outputs" / "logs" / "observations.log"

# Colonnes de sortie

OBSERVATION_COLUMNS = [
    "bulletin_id",
    "registre",
    "culture",
    "bioagresseur",
    "type",
    "departements",
    "localisations",
    "observation",
    "indicateur_quantitatif",
    "evolution",
    "analyse_risque",
    "niveau_risque_observation",
    "facteurs_favorables",
    "source_information",
]

# Titres et termes utiles

GRAND_EST_CULTURES = {
    "ASPERGE",
    "ASPERGES",
    "CHOU",
    "CHOUX",
    "LAITUE",
    "OIGNON",
    "OMBELLIFERES",
    "POMME DE TERRE",
    "SOLANACEES ET CUCURBITACEES SOUS ABRI",
}

PDL_2023_CULTURE_HEADINGS = {
    "ALLIACEES": "alliacees",
    "ALLIUMS": "alliacees",
    "ASPERGE": "asperge",
    "ASPERGES": "asperge",
    "BRASSICACEES": "brassicacees",
    "CAROTTE CELERI": "carotte celeri",
    "CAROTTE CELERI FENOUIL": "carotte celeri fenouil",
    "CAROTTE CELERI PANAIS": "carotte celeri panais",
    "CAROTTE FENOUIL": "carotte fenouil",
    "CAROTTE PANAIS": "carotte panais",
    "CUCURBITACEES": "cucurbitacees",
    "FRAISIER": "fraisier",
    "FRAISIERS": "fraisier",
    "SALADE EPINARD MACHE": "salade epinard mache",
    "SALADE MACHE EPINARD": "salade mache epinard",
    "SOLANACEES": "solanacees",
}

PDL_2023_SECTION_ENDS = {
    "A SURVEILLER",
    "DEVENEZ OBSERVATEUR",
    "FICHE FOCUS",
    "NOTE NATIONALE BIODIVERSITE",
    "RESEAU A R B R E LE",
}

PDL_2023_HEADING_REJECT_TERMS = {
    "ABONNEMENT",
    "ANALYSE DU RISQUE",
    "CREDIT PHOTO",
    "DEGATS DE",
    "FOCUS",
    "GESTION DU RISQUE",
    "GALERIE",
    "INDIQUE",
    "MODELE",
    "OBSERVE",
    "OBSERVATIONS VISUELLES",
    "PERIODE DE RISQUE",
    "SEUIL INDICATIF",
    "SOURCE",
    "SIGNALE",
}

PDL_2023_INVALID_BIO_HEADINGS = {
    "AUXILIAIRE",
    "AUXILIAIRES",
    "CDDL",
    "COCCINELLE",
    "COCCINELLES",
    "COOPERATIVE DE NOIRMOUTIER",
    "DEGATS",
    "GEL",
    "GRELE",
    "GDM",
    "SYRPHE",
    "SYRPHES",
}

PDL_2023_DISEASE_TERMS = {
    "ALTERNARIA",
    "ALTERNARIOSE",
    "BACTERIOSE",
    "BOTRYTIS",
    "CLADOSPORIOSE",
    "FUSARIOSE",
    "MILDIOU",
    "MYCOSPHAERELLA",
    "OIDIUM",
    "PHOMA",
    "PYTHIUM",
    "PHYTOPHTORA",
    "RHIZOCTONE",
    "RHIZOCTONIA",
    "ROUILLE",
    "SCLEROTINIA",
    "SLEROTINIA",
    "SEPTORIOSE",
    "STEMPHYLIUM",
    "THIELAVIOPSIS",
    "VIROSE",
}

PDL_2023_PEST_TERMS = {
    "ACARIEN",
    "ALEURODE",
    "ALTISE",
    "CHENILLE",
    "CLOPORTE",
    "CRIOCERE",
    "DORYPHORE",
    "DROSOPHILA",
    "ESCARGOT",
    "FORFICULE",
    "GASTEROPODE",
    "LIMACE",
    "MINEUSE",
    "MOUCHE",
    "NOCTUELLE",
    "PIERIDE",
    "PENTHALEUS",
    "PYRALE",
    "PUCERON",
    "PUNAISE",
    "LYGUS",
    "SESAMIE",
    "TAUPIN",
    "TEIGNE",
    "TENTHREDE",
    "THRIPS",
    "TUTA",
    "XENOSTRONGYLUS",
}

PARASITE_TERMS = {
    "ANALYSE DU RISQUE",
    "GESTION DU RISQUE",
    "GESTION ALTERNATIVE DU RISQUE",
    "OBSERVATIONS EN PARCELLES",
    "SEUIL INDICATIF DE RISQUE",
}

INVALID_CULTURE_TERMS = {
    "MILDIOU",
    "OIDIUM",
    "PUCERONS",
    "THRIPS",
    "ALEURODES",
    "BOTRYTIS",
    "CLADOSPORIOSE",
    "TUTA ABSOLUTA",
    "MALADIES",
    "RAVAGEURS",
}

INVALID_BIOAGRESSOR_TERMS = {
    "MALADIES",
    "RAVAGEURS",
    "OBSERVATIONS",
    "ANALYSE DU RISQUE",
    "GESTION DU RISQUE",
}

CULTURE_ALIASES = {
    # Tomate
    "TOMATE": "tomate",
    "TOMATES": "tomate",
    # Aubergine
    "AUBERGINE": "aubergine",
    "AUBERGINES": "aubergine",
    # Poivron
    "POIVRON": "poivron",
    "POIVRONS": "poivron",
    # Asperge
    "ASPERGE": "asperge",
    "ASPERGES": "asperge",
    # Pomme de terre
    "POMME DE TERRE": "pomme de terre",
    "POMMES DE TERRE": "pomme de terre",
    "POMMES DE TERRE PRIMEUR": "pomme de terre",
    # Oignon
    "OIGNON": "oignon",
    "OIGNONS": "oignon",
    "OIGNONS DE PRINTEMPS": "oignon",
    # Échalote
    "ECHALOTE": "echalote",
    "ECHALOTES": "echalote",
    # Salade
    "SALADE": "salade",
    "SALADES": "salade",
    "LAITUE": "salade",
    "LAITUES": "salade",
    # Jeunes pousses
    "JEUNES POUSSES": "jeunes pousses",
    "JEUNES POUSSE": "jeunes pousses",
    "JEUNES POUSSES D EPINARDS": "jeunes pousses",
    # Épinard
    "EPINARD": "epinard",
    "EPINARDS": "epinard",
    # Chou
    "CHOU": "chou",
    "CHOUX": "chou",
    # Navet
    "NAVET": "navet",
    "NAVETS": "navet",
    # Radis
    "RADIS": "radis",
    # Mâche
    "MACHE": "mache",
    "MACHES": "mache",
    # Roquette
    "ROQUETTE": "roquette",
    "ROQUETTES": "roquette",
    # Apiacées
    "CAROTTE": "carotte",
    "CAROTTES": "carotte",
    "CELERI": "celeri",
    "CELERIS": "celeri",
    "FENOUIL": "fenouil",
    "FENOUILS": "fenouil",
    "PANAIS": "panais",
    # Fraisier
    "FRAISIER": "fraisier",
    "FRAISIERS": "fraisier",
    "FRAISE": "fraisier",
    "FRAISES": "fraisier",
    # Cucurbitacées
    "COURGETTE": "courgette",
    "COURGETTES": "courgette",
    "CONCOMBRE": "concombre",
    "CONCOMBRES": "concombre",
    "MELON": "melon",
    "MELONS": "melon",
    # Catégorie Grand Est
    "SOLANACEES ET CUCURBITACEES SOUS ABRI": "solanacees_cucurbitacees_sous_abri",
}


DEPARTMENT_RE = re.compile(
    r"(?<!\d)(?:08|10|44|49|51|52|53|54|55|57|67|68|72|79|85|88)(?!\d)"
)

QUANTITATIVE_RE = re.compile(
    r"\b\d+(?:[,.]\d+)?\s*%|"
    r"\b\d+(?:[,.]\d+)?\s*(?:captures?|individus?|plants?|plantes?|pi[eé]geages?|pi[eè]ges?)\b",
    flags=re.IGNORECASE,
)


RISK_LEVEL_PATTERNS = [
    ("tres_fort", r"\bRISQUE\b[^.;:]{0,45}\b(?:TRES\s+FORT|TRES\s+ELEVE)\b"),
    (
        "fort",
        r"\b(?:RISQUE|NIVEAU\s+DE\s+RISQUE)\b[^.;:]{0,45}\b(?:FORT|ELEVE|IMPORTANT)\b",
    ),
    (
        "moyen_a_fort",
        r"\bRISQUE\b[^.;:]{0,45}\b(?:MOYEN\s+A\s+FORT|MOYEN\s+A\s+ELEVE|ASSEZ\s+FORT)\b",
    ),
    (
        "faible_a_moyen",
        r"\bRISQUE\b[^.;:]{0,45}\b(?:FAIBLE\s+A\s+MOYEN|MOYEN\s+A\s+FAIBLE|ASSEZ\s+FAIBLE)\b",
    ),
    ("moyen", r"\bRISQUE\b[^.;:]{0,45}\bMOYEN\b"),
    ("faible", r"\bRISQUE\b[^.;:]{0,45}\b(?:FAIBLE|LIMITE|TRES\s+FAIBLE)\b"),
    ("nul", r"\b(?:PAS\s+DE\s+RISQUE|RISQUE\b[^.;:]{0,45}\bNUL)\b"),
]

RISK_ANALYSIS_MARKERS = {
    "ANALYSE DU RISQUE",
    "ANALYSE DE RISQUE",
    "ANALYSE DES RISQUES",
}

RISK_ANALYSIS_END_MARKERS = {
    "GESTION DU RISQUE",
    "GESTION ALTERNATIVE DU RISQUE",
    "METHODES ALTERNATIVES",
    "PROPHYLAXIE",
}

FAVORABLE_TERMS = {
    "FAVORABLE",
    "FAVORABLES",
    "DEFAVORABLE",
    "DEFAVORABLES",
    "HUMIDE",
    "HUMIDES",
    "HYGROMETRIE",
    "PLUIE",
    "PLUIES",
    "ROSEE",
    "TEMPERATURE",
    "TEMPERATURES",
    "CHAUD",
    "CHAUDE",
    "FRAIS",
    "FRAICHE",
    "VENT",
    "SEC",
    "SECHES",
}


# Structures de données


@dataclass(frozen=True)
class Observation:
    bulletin_id: str
    registre: str
    culture: str
    bioagresseur: str
    type: str
    departements: str
    localisations: str
    observation: str
    indicateur_quantitatif: str
    evolution: str
    analyse_risque: str
    niveau_risque_observation: str
    facteurs_favorables: str
    source_information: str


@dataclass
class ExtractionSummary:
    observations: int = 0
    table_files_used: int = 0
    text_files_used: int = 0
    text_files_skipped: int = 0
    skipped_tables: int = 0
    failed: int = 0


# Arguments CLI


def parse_args() -> argparse.Namespace:
    """Lit les chemins d'entrée et de sortie depuis la ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Extraction des observations phytosanitaires BSV."
    )

    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument(
        "--bulletins-file",
        type=Path,
        default=DEFAULT_BULLETINS_FILE,
    )
    parser.add_argument("--types-file", type=Path, default=DEFAULT_TYPES_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)

    return parser.parse_args()


# Nettoyage et normalisation


def ascii_key(value: str) -> str:
    """Supprime les accents."""

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize(value: str) -> str:
    """Normalise une chaîne pour comparaison robuste."""

    value = ascii_key(value).upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_cell(value: str) -> str:
    """Nettoie une cellule CSV ou une ligne de texte."""

    value = value.replace("\ufeff", "")
    value = re.sub(r"(\w)-\s+(\w)", r"\1\2", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_culture_name(culture: str) -> str:
    """Normalise un nom de culture."""

    key = normalize(culture)

    if key in CULTURE_ALIASES:
        return CULTURE_ALIASES[key]

    return culture.lower().strip()


def split_and_normalize_cultures(culture: str) -> list[str]:
    """
    Sépare les cultures multiples et les normalise.

    Exemple :
        "Tomates, Aubergines, Poivrons"
        -> ["tomate", "aubergine", "poivron"]
    """

    culture = clean_cell(culture)

    if not culture:
        return []

    if len(culture.split()) > 10:
        return []

    parts = re.split(r",|;| et ", culture, flags=re.IGNORECASE)

    results: list[str] = []

    for part in parts:
        part = clean_cell(part)
        key = normalize(part)

        if not part:
            continue

        if key in INVALID_CULTURE_TERMS:
            continue

        results.append(normalize_culture_name(part))

    return list(dict.fromkeys(results))


def culture_terms(culture_norm: str) -> set[str]:
    """Retourne les formes textuelles permettant de reconnaître une culture."""

    terms = {normalize(culture_norm)}

    for alias, normalized in CULTURE_ALIASES.items():
        if normalized == culture_norm:
            terms.add(normalize(alias))

    return {term for term in terms if term}


def split_observation_units(observation_text: str) -> list[str]:
    """Découpe une observation en unités courtes sans dépendre d'une ponctuation parfaite."""

    text = clean_cell(observation_text)

    if not text:
        return []

    units = re.split(
        r";|\n|\.|\s+(?=\d+(?:[,.]\d+)?\s*%)",
        text,
    )

    return [
        clean_cell(unit.strip(" ,")) for unit in units if clean_cell(unit.strip(" ,"))
    ]


def split_observation_by_culture(
    observation_text: str,
    cultures: list[str],
) -> dict[str, str]:
    """
    Répartit une observation multi-cultures en une observation par culture.

    Règle V1 :
        - une unité mentionnant une culture est rattachée à cette culture ;
        - une unité sans culture explicite prolonge la dernière culture détectée ;
        - si aucune culture spécifique n'est trouvée, le texte complet est conservé
          pour chaque culture.
    """

    cultures = list(dict.fromkeys(cultures))

    if not cultures:
        return {}

    observation_text = clean_cell(observation_text)
    units = split_observation_units(observation_text)
    terms_by_culture = {culture: culture_terms(culture) for culture in cultures}
    matched_units: dict[str, list[str]] = {culture: [] for culture in cultures}
    generic_units: list[str] = []
    last_targets: list[str] = []

    for unit in units:
        unit_norm = normalize(unit)
        targets = [
            culture
            for culture, terms in terms_by_culture.items()
            if any(re.search(rf"\b{re.escape(term)}\b", unit_norm) for term in terms)
        ]

        if targets:
            for target in targets:
                matched_units[target].append(unit)
            last_targets = targets
            continue

        if last_targets:
            for target in last_targets:
                matched_units[target].append(unit)
        else:
            generic_units.append(unit)

    has_specific_match = any(matched_units.values())

    if not has_specific_match:
        return {culture: observation_text for culture in cultures}

    for culture in cultures:
        if generic_units:
            matched_units[culture] = generic_units + matched_units[culture]

    return {
        culture: clean_cell("; ".join(units_for_culture))
        for culture, units_for_culture in matched_units.items()
        if units_for_culture
    }


def is_invalid_bioagresseur(bioagresseur: str) -> bool:
    """Détecte les faux bioagresseurs issus de tableaux mal extraits."""

    key = normalize(bioagresseur)

    return not key or key in INVALID_BIOAGRESSOR_TERMS or len(bioagresseur.split()) > 8


def is_parasite(value: str) -> bool:
    """Détecte les lignes qui ne sont pas des observations."""

    normalized = normalize(value)
    return not normalized or any(term in normalized for term in PARASITE_TERMS)


# Référentiel des types


def load_type_referential(types_file: Path) -> dict[str, str]:
    """
    Charge data/raw/referentiel_types.csv.

    Format attendu minimal :
        bioagresseur,type

    Exemple :
        pucerons,ravageur
        mildiou,maladie
        oidium,maladie
    """

    if not types_file.is_file():
        raise FileNotFoundError(f"Référentiel des types introuvable : {types_file}")

    last_error: UnicodeDecodeError | None = None

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            type_map: dict[str, str] = {}

            with types_file.open(encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)

                for row in reader:
                    terme = (
                        row.get("terme")
                        or row.get("bioagresseur")
                        or row.get("nom")
                        or ""
                    )

                    type_value = row.get("type") or ""

                    terme = clean_cell(terme)
                    type_value = clean_cell(type_value).lower()

                    if terme and type_value:
                        type_map[normalize(terme)] = type_value

            return type_map

        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return {}


def resolve_type(
    bioagresseur: str,
    type_map: dict[str, str] | None = None,
    default: str = "",
) -> str:
    """
    Résout le type d'un bioagresseur à partir du référentiel.
    Si aucune correspondance n'est trouvée, retourne default.
    """

    if type_map is None:
        type_map = {}

    key = normalize(bioagresseur)

    if key in type_map:
        return type_map[key]

    for ref_key, ref_type in type_map.items():
        if ref_key in key or key in ref_key:
            return ref_type

    return default


def load_bulletin_registry(
    bulletins_file: Path,
) -> dict[str, dict[str, str]]:
    """Load bulletin metadata used to route and validate observations."""

    if not bulletins_file.is_file():
        raise FileNotFoundError(f"Fichier bulletins introuvable : {bulletins_file}")

    with bulletins_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"bulletin_id", "region", "template_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Colonnes manquantes dans bsv_bulletins.csv : "
                + ", ".join(sorted(missing))
            )

        registry: dict[str, dict[str, str]] = {}
        for row in reader:
            bulletin_id = clean_cell(row.get("bulletin_id", ""))
            if not bulletin_id:
                continue
            if bulletin_id in registry:
                raise ValueError(
                    f"bulletin_id dupliqué dans le registre : {bulletin_id}"
                )
            registry[bulletin_id] = row

    return registry


def canonical_region_slug(region_dir: str) -> str:
    """Normalize legacy and current regional directory names."""

    detected = re.sub(r"_20\d{2}$", "", region_dir.lower())
    return extract_bulletins.REGION_ALIASES.get(detected, detected)


# Extraction simple


def extract_departments(*values: str) -> str:
    """Extrait les départements mentionnés."""

    departments: set[str] = set()

    for value in values:
        departments.update(DEPARTMENT_RE.findall(value))

    return ";".join(sorted(departments))


def extract_quantitative_indicators(value: str) -> str:
    """Extrait les indicateurs quantitatifs."""

    indicators = [
        clean_cell(match.group(0)) for match in QUANTITATIVE_RE.finditer(value)
    ]

    return "; ".join(dict.fromkeys(indicators))


def extract_risk_level(*values: str) -> str:
    """Extrait un niveau de risque textuel normalisé."""

    text = normalize(" ".join(value for value in values if value))

    if not text:
        return ""

    if re.search(r"\bPAS\s+DE\s+RISQUE\b", text):
        return "nul"

    if re.search(r"\bRISQUE\b[^.;:]{0,45}\bNUL\b", text):
        return "nul"

    for level, pattern in RISK_LEVEL_PATTERNS:
        if re.search(pattern, text):
            return level

    if "RISQUE" in text and any(term in text for term in {"AUGMENTE", "HAUSSE"}):
        return "en_hausse"

    if "RISQUE" in text and any(term in text for term in {"DIMINUE", "BAISSE"}):
        return "en_baisse"

    if "RISQUE" in text and any(
        term in text for term in {"STABLE", "MAINTIENT", "PERSISTE"}
    ):
        return "stable"

    return ""


def extract_favorable_factors(*values: str) -> str:
    """Extrait les phrases décrivant les facteurs favorables ou défavorables."""

    factors: list[str] = []

    for value in values:
        for unit in split_observation_units(value):
            unit_norm = normalize(unit)
            if any(
                re.search(rf"\b{re.escape(term)}\b", unit_norm)
                for term in FAVORABLE_TERMS
            ):
                factors.append(unit)

    return "; ".join(dict.fromkeys(factors))


def has_risk_signal(value: str) -> bool:
    """Indique si un texte contient une information interprétable de risque."""

    value_norm = normalize(value)
    if not value_norm:
        return False

    return (
        "RISQUE" in value_norm
        or any(
            re.search(rf"\b{re.escape(term)}\b", value_norm) for term in FAVORABLE_TERMS
        )
        or any(re.search(pattern, value_norm) for _, pattern in RISK_LEVEL_PATTERNS)
    )


def build_risk_fields(*values: str) -> tuple[str, str, str]:
    """Construit analyse, niveau et facteurs à partir d'un ou plusieurs textes."""

    analysis = clean_cell(
        " ".join(value for value in values if value and has_risk_signal(value))
    )
    analysis = clean_cell(re.sub(r"\bBSV\s+\d+\s+[–-]\s+P\.\d+\b", "", analysis))
    if not analysis:
        return "", "", ""

    return (
        analysis,
        extract_risk_level(analysis),
        extract_favorable_factors(analysis),
    )


def extract_risk_analysis_from_text(value: str) -> str:
    """Extrait un bloc 'Analyse du risque' depuis du texte brut ou du CSV aplati."""

    lines = [clean_cell(line) for line in value.splitlines()]
    if len(lines) <= 1:
        lines = [clean_cell(part) for part in re.split(r"\s{2,}|,", value)]

    collecting = False
    buffer: list[str] = []

    for line in lines:
        line_norm = normalize(line)
        if not line_norm:
            continue

        if line_norm in RISK_ANALYSIS_MARKERS:
            collecting = True
            continue

        if collecting and line_norm in RISK_ANALYSIS_END_MARKERS:
            break

        if collecting:
            buffer.append(line)

    return clean_cell(" ".join(buffer))


def extract_pdl_page_risk_analysis(table_path: Path) -> str:
    """Récupère le bloc d'analyse de risque d'une table située sur la même page."""

    match = re.match(r"(page_\d+)_table_", table_path.name)
    if not match:
        return ""

    page_prefix = match.group(1)
    analyses: list[str] = []

    for sibling in sorted(table_path.parent.glob(f"{page_prefix}_table_*.csv")):
        try:
            rows = read_csv_rows(sibling)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

        table_text = "\n".join(
            " ".join(cell for cell in row if cell) for row in rows if any(row)
        )
        analysis = extract_risk_analysis_from_text(table_text)
        if analysis:
            analyses.append(analysis)

    return " ".join(dict.fromkeys(analyses))


def bulletin_id_from_stem(region_dir: str, bulletin_stem: str) -> str:
    """Reconstruit le bulletin_id à partir du dossier région et du nom du bulletin."""

    region_slug = canonical_region_slug(region_dir)

    number = extract_bulletins.extract_bulletin_number(bulletin_stem)
    bulletin_date = extract_bulletins.date_from_filename(bulletin_stem)

    if bulletin_date is None:
        year_match = re.search(r"_(20\d{2})$", bulletin_stem) or re.search(
            r"_(20\d{2})$", region_dir
        )
        year = int(year_match.group(1)) if year_match else "unknown"
    else:
        year = bulletin_date.year

    return extract_bulletins.build_bulletin_id(
        region_slug,
        year,
        number,
        bulletin_stem,
    )


def find_column(headers: list[str], *needles: str) -> int | None:
    """Trouve une colonne à partir de mots-clés normalisés."""

    normalized_headers = [normalize(header) for header in headers]

    for index, header in enumerate(normalized_headers):
        if all(needle in header for needle in needles):
            return index

    return None


def read_csv_rows(path: Path) -> list[list[str]]:
    """Lit un CSV de tableau extrait par pdfplumber."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [[clean_cell(cell) for cell in row] for row in csv.reader(handle)]


def table_metadata(path: Path, tables_dir: Path) -> tuple[str, str]:
    """Récupère le dossier région et le nom du bulletin à partir du chemin du tableau."""

    relative = path.relative_to(tables_dir)

    if len(relative.parts) < 3:
        raise ValueError("table path does not include region and bulletin folders")

    return relative.parts[0], relative.parts[-2]


# Extraction Pays de la Loire depuis les tableaux CSV


def extract_pays_de_la_loire_table(
    table_path: Path,
    tables_dir: Path,
    type_map: dict[str, str] | None = None,
) -> list[Observation]:
    """Extrait les observations depuis les tableaux Pays de la Loire."""

    if type_map is None:
        type_map = {}

    rows = read_csv_rows(table_path)

    if len(rows) < 2:
        return []

    headers = rows[0]
    header_key = normalize(" ".join(headers))

    if "OBSERVATIONS" not in header_key or "CULTURES" not in header_key:
        return []

    first_header = normalize(headers[0])

    if "RAVAGEURS" in first_header:
        observation_type = "ravageur"
    elif "MALADIES" in first_header:
        observation_type = "maladie"
    else:
        return []

    bio_col = 0
    loc_col = find_column(headers, "LOCALISATION")
    culture_col = find_column(headers, "CULTURES")
    observation_col = find_column(headers, "OBSERVATIONS")
    evolution_col = find_column(headers, "EVOLUTION")

    if culture_col is None or observation_col is None:
        return []

    region_dir, bulletin_stem = table_metadata(table_path, tables_dir)
    bulletin_id = bulletin_id_from_stem(region_dir, bulletin_stem)
    page_risk_analysis = extract_pdl_page_risk_analysis(table_path)

    observations: list[Observation] = []

    for row in rows[1:]:
        row = row + [""] * (len(headers) - len(row))

        bioagresseur = clean_cell(row[bio_col])
        culture = clean_cell(row[culture_col])
        observation_text = clean_cell(row[observation_col])
        localisations = clean_cell(row[loc_col]) if loc_col is not None else ""
        evolution = clean_cell(row[evolution_col]) if evolution_col is not None else ""

        if (
            is_parasite(bioagresseur)
            or is_invalid_bioagresseur(bioagresseur)
            or not culture
            or is_parasite(observation_text)
        ):
            continue

        cultures = split_and_normalize_cultures(culture)

        if not cultures:
            continue

        obs_by_culture = split_observation_by_culture(
            observation_text,
            cultures,
        )

        for culture_norm, observation_for_culture in obs_by_culture.items():
            analyse_risque, niveau_risque, facteurs_favorables = build_risk_fields(
                observation_for_culture,
                evolution,
                page_risk_analysis,
            )

            observations.append(
                Observation(
                    bulletin_id=bulletin_id,
                    registre="observation_parcelle",
                    culture=culture_norm,
                    bioagresseur=bioagresseur,
                    type=resolve_type(bioagresseur, type_map, observation_type),
                    departements=extract_departments(
                        localisations, observation_for_culture
                    ),
                    localisations=localisations,
                    observation=observation_for_culture,
                    indicateur_quantitatif=extract_quantitative_indicators(
                        observation_for_culture
                    ),
                    evolution=evolution,
                    analyse_risque=analyse_risque,
                    niveau_risque_observation=niveau_risque,
                    facteurs_favorables=facteurs_favorables,
                    source_information=str(table_path.relative_to(PROJECT_ROOT)),
                )
            )

    return observations


# Extraction narrative Pays de la Loire 2023


def pdl_2023_culture_heading(line: str) -> str:
    """Return the normalized culture group represented by a section heading."""

    heading = normalize(line)
    heading = re.sub(r"^(?:SUITE\s+)", "", heading).strip()
    return PDL_2023_CULTURE_HEADINGS.get(heading, "")


def split_pdl_2023_columns(line: str) -> tuple[str, str]:
    """Split a layout-preserving line into its left and right PDF columns."""

    if not line.strip():
        return "", ""

    leading_spaces = len(line) - len(line.lstrip(" "))
    if leading_spaces >= 45:
        return "", line.strip()

    candidates = [
        match
        for match in re.finditer(r" {4,}", line)
        if 48 <= match.end() <= 170
        and line[: match.start()].strip()
        and line[match.end() :].strip()
    ]
    if candidates:
        separator = min(candidates, key=lambda match: abs(match.end() - 62))
        return (
            line[: separator.start()].strip(),
            line[separator.end() :].strip(),
        )

    return line.strip(), ""


def detect_cultures_in_text(value: str) -> list[str]:
    """Detect explicit crop names in a heading or observation paragraph."""

    value_norm = normalize(value)
    detected: list[str] = []

    for alias in sorted(CULTURE_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", value_norm):
            detected.append(CULTURE_ALIASES[alias])

    return list(dict.fromkeys(detected))


def clean_pdl_2023_bioagressor_heading(value: str) -> str:
    """Remove host-crop parentheses while retaining scientific names."""

    value = clean_cell(value.strip(" •-"))

    def remove_host_parenthesis(match: re.Match[str]) -> str:
        return "" if detect_cultures_in_text(match.group(1)) else match.group(0)

    value = re.sub(r"\(([^()]*)\)", remove_host_parenthesis, value)
    return clean_cell(value.strip(" ,-"))


def is_pdl_2023_bioagressor_heading(
    value: str,
    type_map: dict[str, str],
) -> bool:
    """Identify short bioaggressor headings in a narrative PDL 2023 column."""

    value = clean_cell(value)
    heading = normalize(value)

    if not heading or len(value) > 85 or len(heading.split()) > 11:
        return False
    if not value[:1].isupper():
        return False
    if value.endswith(".") or re.search(r"\b\d+(?:[,.]\d+)?\s*%", value):
        return False
    if DEPARTMENT_RE.search(value):
        return False
    if pdl_2023_culture_heading(value) or heading in PDL_2023_SECTION_ENDS:
        return False
    if heading in PDL_2023_INVALID_BIO_HEADINGS:
        return False
    if "RAVAGEUR" in heading or "MALADIE" in heading:
        return False
    if any(term in heading for term in PDL_2023_HEADING_REJECT_TERMS):
        return False
    if re.match(
        r"^(?:A|AU|AUX|AVEC|AUCUN|AUCUNE|CE|CES|CETTE|DANS|DE|DES|EN|IL|LA|LE|LES|ON|POUR|SUR)\b",
        heading,
    ):
        return False
    if set(heading.split()) & {
        "CONDITIONS",
        "CULTURE",
        "CULTURES",
        "PARCELLE",
        "PARCELLES",
        "PLANT",
        "PLANTS",
        "PRESSION",
        "SEMAINE",
        "SIGNALE",
    }:
        return False

    cleaned = normalize(clean_pdl_2023_bioagressor_heading(value))
    if cleaned in type_map:
        return True
    if any(
        ref_term in cleaned or cleaned in ref_term
        for ref_term in type_map
        if len(ref_term) >= 5
    ):
        return True

    if detect_cultures_in_text(value):
        return False

    return bool(value[:1].isupper() and len(heading.split()) <= 6)


def resolve_pdl_2023_type(
    bioagresseur: str,
    type_map: dict[str, str],
    default: str = "",
) -> str:
    """Resolve common PDL narrative terms missing from the base referential."""

    resolved = resolve_type(bioagresseur, type_map)
    if resolved:
        return resolved

    heading = normalize(bioagresseur)
    if any(term in heading for term in PDL_2023_DISEASE_TERMS):
        return "maladie"
    if any(term in heading for term in PDL_2023_PEST_TERMS):
        return "ravageur"
    return default


def pdl_2023_page_lines(text: str) -> list[tuple[int, list[tuple[int, str]]]]:
    """Split text into pages while retaining original global line numbers."""

    pages: list[tuple[int, list[tuple[int, str]]]] = []
    page_number = 1
    current: list[tuple[int, str]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        marker = re.fullmatch(r"--- PAGE (\d+) ---", line.strip())
        if marker:
            if current:
                pages.append((page_number, current))
            page_number = int(marker.group(1))
            current = []
            continue
        current.append((line_number, line))

    if current:
        pages.append((page_number, current))

    return pages


def pdl_2023_supporting_tables(
    text_path: Path,
    text_dir: Path,
    tables_dir: Path,
) -> dict[int, list[tuple[Path, str]]]:
    """Load non-empty table text indexed by PDF page for optional enrichment."""

    relative = text_path.relative_to(text_dir)
    bulletin_dir = tables_dir / relative.with_suffix("")
    tables_by_page: dict[int, list[tuple[Path, str]]] = {}

    if not bulletin_dir.is_dir():
        return tables_by_page

    for table_path in sorted(bulletin_dir.glob("page_*_table_*.csv")):
        match = re.match(r"page_(\d+)_table_", table_path.name)
        if not match:
            continue
        rows = read_csv_rows(table_path)
        table_text = clean_cell(" ".join(cell for row in rows for cell in row if cell))
        table_norm = normalize(table_text)
        if not table_text or any(
            noise in table_norm
            for noise in ("ABONNEMENT BSV", "BILAN METEOROLOGIQUE", "PLUVIOMETRIE")
        ):
            continue
        tables_by_page.setdefault(int(match.group(1)), []).append(
            (table_path, table_text)
        )

    return tables_by_page


def matching_pdl_2023_table_support(
    page_tables: list[tuple[Path, str]],
    bioagresseur: str,
    cultures: list[str],
) -> list[tuple[Path, str]]:
    """Keep only same-page tables mentioning the observation context."""

    bio_norm = normalize(bioagresseur)
    culture_norms = {normalize(culture) for culture in cultures}
    matches: list[tuple[Path, str]] = []

    for path, table_text in page_tables:
        table_norm = normalize(table_text)
        bio_match = bio_norm and bio_norm in table_norm
        culture_match = any(culture in table_norm for culture in culture_norms)
        if bio_match or culture_match:
            matches.append((path, table_text))

    return matches


def extract_pdl_2023_text(
    text_path: Path,
    text_dir: Path,
    tables_dir: Path,
    type_map: dict[str, str] | None = None,
) -> list[Observation]:
    """Extract PDL 2023 narrative observations and enrich from same-page tables."""

    if type_map is None:
        type_map = {}

    relative = text_path.relative_to(text_dir)
    region_dir = relative.parts[0]
    bulletin_id = bulletin_id_from_stem(region_dir, text_path.stem)
    text = text_path.read_text(encoding="utf-8")
    supporting_tables = pdl_2023_supporting_tables(
        text_path,
        text_dir,
        tables_dir,
    )
    observations: list[Observation] = []

    for page_number, page_lines in pdl_2023_page_lines(text):
        section_starts: list[tuple[int, str]] = []
        for index, (_, line) in enumerate(page_lines):
            culture_group = pdl_2023_culture_heading(line)
            if culture_group:
                section_starts.append((index, culture_group))

        for section_position, (start, culture_group) in enumerate(section_starts):
            end = (
                section_starts[section_position + 1][0]
                if section_position + 1 < len(section_starts)
                else len(page_lines)
            )
            segment = page_lines[start + 1 : end]
            if any(normalize(line) in PDL_2023_SECTION_ENDS for _, line in segment):
                first_end = next(
                    index
                    for index, (_, line) in enumerate(segment)
                    if normalize(line) in PDL_2023_SECTION_ENDS
                )
                segment = segment[:first_end]

            segment_norm = normalize(" ".join(line for _, line in segment))
            if "RAVAGEUR" not in segment_norm and "MALADIE" not in segment_norm:
                continue
            if "RAVAGEUR" in segment_norm and "MALADIE" not in segment_norm:
                default_type = "ravageur"
            elif "MALADIE" in segment_norm and "RAVAGEUR" not in segment_norm:
                default_type = "maladie"
            else:
                default_type = ""

            columns: list[list[tuple[int, str]]] = [[], []]
            for line_number, line in segment:
                left, right = split_pdl_2023_columns(line)
                columns[0].append((line_number, left))
                columns[1].append((line_number, right))

            for column in columns:
                current_bioagresseur = ""
                current_start_line = 0
                buffer: list[str] = []

                def flush() -> None:
                    nonlocal current_bioagresseur, current_start_line, buffer

                    observation_text = clean_cell(" ".join(buffer))
                    bioagresseur = clean_pdl_2023_bioagressor_heading(
                        current_bioagresseur
                    )
                    if (
                        not bioagresseur
                        or not observation_text
                        or len(observation_text.split()) < 4
                        or is_invalid_bioagresseur(bioagresseur)
                    ):
                        current_bioagresseur = ""
                        buffer = []
                        return

                    cultures = detect_cultures_in_text(
                        f"{current_bioagresseur} {observation_text}"
                    )
                    if not cultures:
                        cultures = [culture_group]

                    obs_by_culture = split_observation_by_culture(
                        observation_text,
                        cultures,
                    )
                    for culture, observation_for_culture in obs_by_culture.items():
                        support = matching_pdl_2023_table_support(
                            supporting_tables.get(page_number, []),
                            bioagresseur,
                            [culture],
                        )
                        support_text = " ".join(text for _, text in support)
                        source = (
                            f"{text_path.relative_to(PROJECT_ROOT)}"
                            f"#L{current_start_line}"
                        )
                        if support:
                            source += ";" + ";".join(
                                str(path.relative_to(PROJECT_ROOT))
                                for path, _ in support
                            )
                        analyse_risque, niveau_risque, facteurs_favorables = (
                            build_risk_fields(
                                observation_for_culture,
                                support_text,
                            )
                        )

                        observations.append(
                            Observation(
                                bulletin_id=bulletin_id,
                                registre="observation_parcelle",
                                culture=culture,
                                bioagresseur=bioagresseur,
                                type=resolve_pdl_2023_type(
                                    bioagresseur,
                                    type_map,
                                    default_type,
                                ),
                                departements=extract_departments(
                                    observation_for_culture,
                                    support_text,
                                ),
                                localisations="",
                                observation=observation_for_culture,
                                indicateur_quantitatif=extract_quantitative_indicators(
                                    f"{observation_for_culture} {support_text}"
                                ),
                                evolution="",
                                analyse_risque=analyse_risque,
                                niveau_risque_observation=niveau_risque,
                                facteurs_favorables=facteurs_favorables,
                                source_information=source,
                            )
                        )
                    current_bioagresseur = ""
                    buffer = []

                for line_number, value in column:
                    value = clean_cell(value)
                    if not value or value.startswith("•"):
                        continue
                    if is_pdl_2023_bioagressor_heading(value, type_map):
                        flush()
                        current_bioagresseur = value
                        current_start_line = line_number
                        buffer = []
                    elif current_bioagresseur:
                        if "CREDIT PHOTO" not in normalize(value):
                            buffer.append(value)

                flush()

    return observations


def split_pdl_2023_culture_prefix(value: str) -> tuple[str, str]:
    """Split a leading culture heading from text on the same extracted line."""

    value = clean_cell(value)
    ascii_value = ascii_key(value).upper()

    for heading in sorted(PDL_2023_CULTURE_HEADINGS, key=len, reverse=True):
        words = heading.split()
        pattern = r"^\W*" + r"\W*".join(re.escape(word) for word in words)
        match = re.match(pattern, ascii_value)
        if not match:
            continue
        return (
            PDL_2023_CULTURE_HEADINGS[heading],
            clean_cell(value[match.end() :].lstrip(" :-")),
        )

    return "", value


PDL_2023_FLASH_EXTRA_TERMS = {
    "BOTRYTIS": ("Botrytis", "maladie"),
    "CRIOCERES": ("Criocere", "ravageur"),
    "MOUCHE DES SEMIS": ("Mouche des semis", "ravageur"),
    "MOUCHES DES SEMIS": ("Mouche des semis", "ravageur"),
    "ROUILLE": ("Rouille", "maladie"),
    "MILDIOU": ("Mildiou", "maladie"),
    "OIDIUM": ("Oidium", "maladie"),
    "PUCERON": ("Puceron", "ravageur"),
    "SEPTORIOSE": ("Septoriose", "maladie"),
    "THRIPS": ("Thrips", "ravageur"),
    "VIROSE": ("Virose", "maladie"),
    "VIROSES": ("Virose", "maladie"),
    "XENOSTRONGYLUS": ("Xenostrongylus", "ravageur"),
}


def detect_bioaggressors_in_flash_block(
    text: str,
    type_map: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Return (display name, type, matched normalized term) for a Flash block."""

    text_norm = normalize(text)
    candidates: list[tuple[str, str, str]] = []

    for term, type_value in sorted(
        type_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if len(term) < 5 or not re.search(rf"\b{re.escape(term)}\b", text_norm):
            continue
        if any(term in existing[2] or existing[2] in term for existing in candidates):
            continue
        candidates.append((term.lower(), type_value, term))

    for term, (display, type_value) in PDL_2023_FLASH_EXTRA_TERMS.items():
        if term not in text_norm:
            continue
        if any(term in existing[2] or existing[2] in term for existing in candidates):
            continue
        candidates.append((display, type_value, term))

    return candidates


def extract_pdl_2023_flash_text(
    text_path: Path,
    text_dir: Path,
    type_map: dict[str, str] | None = None,
) -> list[Observation]:
    """Extract culture blocks from the PDL 2023 Flash grouped by register."""

    if type_map is None:
        type_map = {}

    relative = text_path.relative_to(text_dir)
    bulletin_id = bulletin_id_from_stem(relative.parts[0], text_path.stem)
    observations: list[Observation] = []

    for page_number, page_lines in pdl_2023_page_lines(
        text_path.read_text(encoding="utf-8")
    ):
        if page_number not in {2, 3}:
            continue

        register = ""
        columns: list[list[tuple[int, str, str]]] = [[], []]
        for line_number, line in page_lines:
            line_norm = normalize(line)
            if line_norm in {"RAVAGEURS", "SUITE RAVAGEURS"}:
                register = "ravageur"
                continue
            if line_norm == "MALADIES":
                register = "maladie"
                continue

            left, right = split_pdl_2023_columns(line)
            columns[0].append((line_number, left, register))
            columns[1].append((line_number, right, register))

        for column in columns:
            current_culture = ""
            current_register = ""
            current_line = 0
            buffer: list[str] = []

            def flush_flash_block() -> None:
                nonlocal current_culture, current_register, current_line, buffer

                block_text = clean_cell(" ".join(buffer))
                if not current_culture or len(block_text.split()) < 5:
                    buffer = []
                    return

                for (
                    bioagresseur,
                    type_value,
                    matched_term,
                ) in detect_bioaggressors_in_flash_block(
                    block_text,
                    type_map,
                ):
                    units = split_observation_units(block_text)
                    matching_units = [
                        unit for unit in units if matched_term in normalize(unit)
                    ]
                    observation_text = clean_cell(
                        "; ".join(matching_units) if matching_units else block_text
                    )
                    cultures = detect_cultures_in_text(observation_text)
                    if not cultures:
                        cultures = [current_culture]

                    for (
                        culture,
                        observation_for_culture,
                    ) in split_observation_by_culture(
                        observation_text,
                        cultures,
                    ).items():
                        analyse_risque, niveau_risque, facteurs_favorables = (
                            build_risk_fields(
                                observation_for_culture,
                            )
                        )

                        observations.append(
                            Observation(
                                bulletin_id=bulletin_id,
                                registre="observation_parcelle",
                                culture=culture,
                                bioagresseur=bioagresseur,
                                type=type_value or current_register,
                                departements=extract_departments(
                                    observation_for_culture
                                ),
                                localisations="",
                                observation=observation_for_culture,
                                indicateur_quantitatif=extract_quantitative_indicators(
                                    observation_for_culture
                                ),
                                evolution="",
                                analyse_risque=analyse_risque,
                                niveau_risque_observation=niveau_risque,
                                facteurs_favorables=facteurs_favorables,
                                source_information=(
                                    f"{text_path.relative_to(PROJECT_ROOT)}"
                                    f"#L{current_line}"
                                ),
                            )
                        )

                buffer = []

            for line_number, value, line_register in column:
                if (
                    line_register
                    and line_register != current_register
                    and current_culture
                ):
                    flush_flash_block()
                    current_culture = ""
                if line_register:
                    current_register = line_register

                culture, remainder = split_pdl_2023_culture_prefix(value)
                if culture:
                    flush_flash_block()
                    current_culture = culture
                    current_line = line_number
                    buffer = [remainder] if remainder else []
                elif current_culture and value:
                    buffer.append(value)

            flush_flash_block()

    return observations


# Extraction Grand Est depuis le texte


def is_grand_est_bioagressor_heading(line: str) -> str | None:
    """Détecte un titre de bioagresseur du type '2 Pucerons' ou '3 Mildiou'."""

    match = re.match(r"^\s*\d+\s+(.+?)\s*$", line)

    if not match:
        return None

    title = clean_cell(match.group(1))
    normalized = normalize(title)

    if normalized in {
        "STADES PHENOLOGIQUES",
        "AUTRES BIOAGRESSEURS",
        "MALADIES",
        "RAVAGEURS",
    }:
        return None

    if len(title) > 80:
        return None

    return title


def line_is_subsection(line: str, label: str) -> bool:
    """Détecte une sous-section du type 'a. Observations'."""

    return re.match(rf"^\s*[a-d]\.\s*{label}\b", line, flags=re.IGNORECASE) is not None


def flush_grand_est_observation(
    observations: list[Observation],
    bulletin_id: str,
    culture: str,
    bioagresseur: str,
    buffer: list[str],
    analysis_buffer: list[str],
    text_path: Path,
    line_number: int,
    type_map: dict[str, str] | None = None,
) -> None:
    """Ajoute une observation Grand Est lorsque le bloc 'Observations' est terminé."""

    if type_map is None:
        type_map = {}

    observation_text = clean_cell(" ".join(buffer))
    analysis_text = clean_cell(" ".join(analysis_buffer))

    if not observation_text or is_parasite(observation_text):
        return

    if is_invalid_bioagresseur(bioagresseur):
        return

    cultures = split_and_normalize_cultures(culture)

    if not cultures:
        return

    analyse_risque, niveau_risque, facteurs_favorables = build_risk_fields(
        analysis_text,
        observation_text if not analysis_text else "",
    )

    for culture_norm in cultures:
        observations.append(
            Observation(
                bulletin_id=bulletin_id,
                registre="observation_parcelle",
                culture=culture_norm,
                bioagresseur=bioagresseur,
                type=resolve_type(bioagresseur, type_map),
                departements=extract_departments(observation_text),
                localisations="",
                observation=observation_text,
                indicateur_quantitatif=extract_quantitative_indicators(
                    observation_text
                ),
                evolution="",
                analyse_risque=analyse_risque,
                niveau_risque_observation=niveau_risque,
                facteurs_favorables=facteurs_favorables,
                source_information=f"{text_path.relative_to(PROJECT_ROOT)}#L{line_number}",
            )
        )


def extract_grand_est_text(
    text_path: Path,
    text_dir: Path,
    type_map: dict[str, str] | None = None,
) -> list[Observation]:
    """Extrait les observations Grand Est depuis les sections narratives du texte."""

    if type_map is None:
        type_map = {}

    relative = text_path.relative_to(text_dir)
    region_dir = relative.parts[0]

    bulletin_id = bulletin_id_from_stem(region_dir, text_path.stem)
    lines = text_path.read_text(encoding="utf-8").splitlines()

    observations: list[Observation] = []

    current_culture = ""
    current_bioagresseur = ""
    current_section = ""
    observation_buffer: list[str] = []
    analysis_buffer: list[str] = []
    start_line = 0

    def flush_current() -> None:
        nonlocal observation_buffer, analysis_buffer, current_section

        if not observation_buffer and not analysis_buffer:
            return

        flush_grand_est_observation(
            observations,
            bulletin_id,
            current_culture,
            current_bioagresseur,
            observation_buffer,
            analysis_buffer,
            text_path,
            start_line,
            type_map,
        )
        observation_buffer = []
        analysis_buffer = []
        current_section = ""

    for index, raw_line in enumerate(lines, start=1):
        line = clean_cell(raw_line)
        heading = normalize(line)

        if heading in GRAND_EST_CULTURES:
            flush_current()

            current_culture = line.title()
            current_bioagresseur = ""
            current_section = ""
            continue

        next_bioagresseur = is_grand_est_bioagressor_heading(line)

        if next_bioagresseur:
            flush_current()

            current_bioagresseur = next_bioagresseur
            current_section = ""
            continue

        if line_is_subsection(line, "Observations"):
            current_section = (
                "observations" if current_culture and current_bioagresseur else ""
            )
            observation_buffer = []
            start_line = index
            continue

        if line_is_subsection(line, "Analyse"):
            current_section = (
                "analyse" if current_culture and current_bioagresseur else ""
            )
            analysis_buffer = []
            continue

        if current_section == "analyse" and any(
            marker in heading for marker in RISK_ANALYSIS_END_MARKERS
        ):
            flush_current()
            continue

        if (
            line_is_subsection(line, "Seuil")
            or line_is_subsection(line, "Gestion")
            or re.fullmatch(r"--- PAGE \d+ ---", line)
        ):
            if current_section == "analyse" or line_is_subsection(line, "Gestion"):
                flush_current()
            current_section = ""
            continue

        if current_section == "observations" and line:
            observation_buffer.append(line)
        elif current_section == "analyse" and line:
            analysis_buffer.append(line)

    flush_current()

    return observations


# Découverte des fichiers


def discover_csv_files(tables_dir: Path) -> list[Path]:
    """Liste tous les CSV de tableaux."""

    if not tables_dir.is_dir():
        raise FileNotFoundError(f"Tables directory does not exist: {tables_dir}")

    return sorted(tables_dir.rglob("*.csv"), key=lambda path: str(path).casefold())


def discover_observation_texts(text_dir: Path) -> list[Path]:
    """List all regional texts that may require narrative extraction."""

    if not text_dir.is_dir():
        raise FileNotFoundError(f"Text directory does not exist: {text_dir}")

    return sorted(
        text_dir.rglob("*.txt"),
        key=lambda path: str(path.relative_to(text_dir)).casefold(),
    )


def discover_grand_est_texts(text_dir: Path) -> list[Path]:
    """List Grand Est texts across legacy and current directory layouts."""

    return [
        path
        for path in discover_observation_texts(text_dir)
        if canonical_region_slug(path.relative_to(text_dir).parts[0]) == "grandest"
    ]


# Écriture CSV


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Écrit un CSV de manière sécurisée."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


# Extraction globale


def deduplicate_observations(observations: list[Observation]) -> list[Observation]:
    """Supprime les doublons stricts d'observations."""

    deduped = {
        (
            obs.bulletin_id,
            obs.culture,
            obs.bioagresseur,
            obs.observation,
        ): obs
        for obs in observations
    }

    return list(deduped.values())


def extract_all(
    tables_dir: Path,
    text_dir: Path,
    bulletins_file: Path,
    types_file: Path,
    output_file: Path,
    log_file: Path,
) -> ExtractionSummary:
    """Extrait toutes les observations du corpus."""

    summary = ExtractionSummary()
    observations: list[Observation] = []
    logs: list[dict[str, str]] = []

    bulletin_registry = load_bulletin_registry(bulletins_file)
    type_map = load_type_referential(types_file)

    for table_path in discover_csv_files(tables_dir):
        relative = str(table_path.relative_to(tables_dir))

        try:
            region_dir, bulletin_stem = table_metadata(table_path, tables_dir)
            region_slug = canonical_region_slug(region_dir)
            extracted: list[Observation] = []

            if region_slug == "pdl":
                bulletin_id = bulletin_id_from_stem(
                    region_dir,
                    bulletin_stem,
                )
                bulletin = bulletin_registry.get(bulletin_id)
                if bulletin is None:
                    raise ValueError(
                        f"bulletin absent de bsv_bulletins.csv : {bulletin_id}"
                    )

                extracted = extract_pays_de_la_loire_table(
                    table_path,
                    tables_dir,
                    type_map,
                )

            if extracted:
                observations.extend(extracted)
                summary.table_files_used += 1

                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "source_file": relative,
                        "status": "OK",
                        "rows": str(len(extracted)),
                        "message": "table observations extracted",
                    }
                )
            else:
                summary.skipped_tables += 1

        except Exception as exc:
            summary.failed += 1

            logs.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source_file": relative,
                    "status": "ERROR",
                    "rows": "0",
                    "message": str(exc),
                }
            )

            print(f"ERROR {relative}: {exc}", file=sys.stderr)

    for text_path in discover_observation_texts(text_dir):
        relative = str(text_path.relative_to(text_dir))

        try:
            relative_path = text_path.relative_to(text_dir)
            region_dir = relative_path.parts[0]
            region_slug = canonical_region_slug(region_dir)
            bulletin_id = bulletin_id_from_stem(
                region_dir,
                text_path.stem,
            )
            bulletin = bulletin_registry.get(bulletin_id)

            if bulletin is None:
                raise ValueError(
                    f"bulletin absent de bsv_bulletins.csv : {bulletin_id}"
                )

            template_id = bulletin.get("template_id", "")

            if region_slug == "pdl":
                if template_id == extract_bulletins.TEMPLATE_PDL_2023:
                    if "flash" in text_path.stem.lower():
                        extracted = extract_pdl_2023_flash_text(
                            text_path,
                            text_dir,
                            type_map,
                        )
                    else:
                        extracted = extract_pdl_2023_text(
                            text_path,
                            text_dir,
                            tables_dir,
                            type_map,
                        )
                    if extracted:
                        observations.extend(extracted)
                        summary.text_files_used += 1
                        logs.append(
                            {
                                "timestamp": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "source_file": relative,
                                "status": "OK",
                                "rows": str(len(extracted)),
                                "message": "PDL 2023 narrative observations extracted",
                            }
                        )
                    else:
                        summary.text_files_skipped += 1
                        logs.append(
                            {
                                "timestamp": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "source_file": relative,
                                "status": "SKIPPED",
                                "rows": "0",
                                "message": "no PDL 2023 narrative observation found",
                            }
                        )
                continue

            if region_slug != "grandest":
                continue

            if template_id == extract_bulletins.TEMPLATE_GRAND_EST_SANS_OBSERVATION:
                summary.text_files_skipped += 1
                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "source_file": relative,
                        "status": "SKIPPED",
                        "rows": "0",
                        "message": "bulletin explicitly has no observations",
                    }
                )
                continue

            extracted = extract_grand_est_text(
                text_path,
                text_dir,
                type_map,
            )

            if extracted:
                observations.extend(extracted)
                summary.text_files_used += 1

                logs.append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "source_file": relative,
                        "status": "OK",
                        "rows": str(len(extracted)),
                        "message": "text observations extracted",
                    }
                )

        except Exception as exc:
            summary.failed += 1

            logs.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source_file": relative,
                    "status": "ERROR",
                    "rows": "0",
                    "message": str(exc),
                }
            )

            print(f"ERROR {relative}: {exc}", file=sys.stderr)

    observations = deduplicate_observations(observations)

    unknown_ids = sorted(
        {row.bulletin_id for row in observations} - set(bulletin_registry)
    )
    if unknown_ids:
        raise ValueError(
            "observations liées à des bulletins inconnus : "
            + ", ".join(unknown_ids[:10])
        )

    observations.sort(
        key=lambda item: (
            item.bulletin_id,
            item.culture,
            item.bioagresseur,
            item.source_information,
        )
    )

    summary.observations = len(observations)

    write_csv(output_file, [asdict(row) for row in observations], OBSERVATION_COLUMNS)

    write_csv(
        log_file,
        logs,
        ["timestamp", "source_file", "status", "rows", "message"],
    )

    return summary


# Main


def main() -> int:
    """Point d'entrée du script."""

    args = parse_args()

    try:
        summary = extract_all(
            args.tables_dir.resolve(),
            args.text_dir.resolve(),
            args.bulletins_file.resolve(),
            args.types_file.resolve(),
            args.output_file.resolve(),
            args.log_file.resolve(),
        )

    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "Summary: "
        f"{summary.observations} observations, "
        f"{summary.table_files_used} table files used, "
        f"{summary.text_files_used} text files used, "
        f"{summary.text_files_skipped} text files skipped, "
        f"{summary.skipped_tables} tables skipped, "
        f"{summary.failed} errors."
    )

    print(f"CSV: {args.output_file.resolve()}")
    print(f"Log: {args.log_file.resolve()}")

    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
