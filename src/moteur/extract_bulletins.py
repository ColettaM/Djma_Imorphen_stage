"""Extraction des métadonnées de niveau bulletin à partir des fichiers texte BSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


# Chemins par défaut du projet

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "moteur" / "extracted_text"
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "moteur" / "bsv_bulletins.csv"
)
DEFAULT_LOG_FILE = PROJECT_ROOT / "outputs" / "moteur" / "logs" / "bulletins.log"


# Dictionnaires de normalisation

MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}

REGION_NAMES = {
    "grandest": "Grand Est",
    "paca": "Provence-Alpes-Côte d'Azur",
    "pdl": "Pays de la Loire",
}

REGION_ALIASES = {
    "grand_est": "grandest",
    "grandest": "grandest",
    "paca": "paca",
    "pays_de_la_loire": "pdl",
    "pdl": "pdl",
}

TEMPLATE_PDL_2023 = "pdl_2023_bilan"
TEMPLATE_PDL_2024 = "pdl_2024_prevision"
TEMPLATE_PDL_2025 = "pdl_2025_prevision_risque"
TEMPLATE_GRAND_EST = "grand_est_standard"
TEMPLATE_GRAND_EST_SANS_OBSERVATION = "grand_est_sans_observation"
TEMPLATE_PACA_MARAICHAGE = "paca_maraichage"

# Titres qui marquent généralement la fin de la section météo dans les BSV Grand Est.
GRAND_EST_SECTION_ENDS = {
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

# Titres qui marquent la fin de la section météo dans les BSV Pays de la Loire.
PAYS_DE_LA_LOIRE_SECTION_ENDS = {
    "EVALUATION DES RISQUES",
    "ALLIACEES",
    "ALLIUMS",
    "ASPERGE",
    "ASPERGES",
    "BRASSICACEES",
    "CAROTTE",
    "CAROTTES",
    "CUCURBITACEES",
    "FRAISIERS",
    "MALADIES",
    "RAVAGEURS",
    "SALADE MACHE EPINARD",
    "SALADES",
}

MAX_WEATHER_WORDS = 350

WEATHER_START_PATTERNS = (
    "PREVISION METEOROLOGIQUE",
    "PREVISIONS METEOROLOGIQUES",
    "BILAN METEOROLOGIQUE",
    "DONNEES METEO",
    "METEO",
)

WEATHER_CONTEXT_TERMS = {
    "CHAUD",
    "CHALEUR",
    "CONDITIONS CLIMATIQUES",
    "DEFAVORABLE",
    "ENSOLEILLE",
    "FAVORABLE",
    "FRAIS",
    "FROID",
    "GELEE",
    "HUMIDE",
    "HUMIDITE",
    "HYGROMETRIE",
    "MALADIES",
    "ORAGE",
    "PLUIE",
    "PLUIES",
    "PRECIPITATION",
    "PRECIPITATIONS",
    "RAVAGEURS",
    "SEC",
    "SECHERESSE",
    "TEMPERATURE",
    "TEMPERATURES",
    "VENT",
}

WEATHER_FAVORABILITY_TERMS = {
    "CONDITIONS CLIMATIQUES",
    "DEFAVORABLE",
    "FAVORABLE",
    "PROPICE",
}

WEATHER_CULTURE_TERMS = {
    "AIL",
    "ASPERGE",
    "AUBERGINE",
    "CAROTTE",
    "CHOU",
    "CONCOMBRE",
    "COURGETTE",
    "FRAISIER",
    "LAITUE",
    "OIGNON",
    "POIREAU",
    "POIVRON",
    "POMME DE TERRE",
    "SALADE",
    "TOMATE",
}

GENERAL_WEATHER_SECTION_ENDS = (
    GRAND_EST_SECTION_ENDS
    | PAYS_DE_LA_LOIRE_SECTION_ENDS
    | {"NOTE BIODIVERSITE", "NOTE NATIONALE BIODIVERSITE"}
)

INFORMATIVE_WEATHER_TERMS = {
    "CHAUD",
    "CHALEUR",
    "ENSOLEILLE",
    "FROID",
    "FRAIS",
    "GELEE",
    "HUMIDE",
    "HUMIDITE",
    "ORAGE",
    "PLUIE",
    "PLUIES",
    "PRECIPITATION",
    "PRECIPITATIONS",
    "SEC",
    "SECHERESSE",
    "TEMPERATURE",
    "TEMPERATURES",
    "VENT",
}


# Structures de données


@dataclass(frozen=True)
class Bulletin:
    """Représente une ligne de la table bsv_bulletins.csv."""

    bulletin_id: str
    region: str
    date_bulletin: str
    template_id: str
    nature_meteo: str
    meteo_extraction_status: str
    meteo_extraction_method: str
    meteo_extraction_score: int
    periode_prevision_meteo: str
    prevision_meteo: str


@dataclass(frozen=True)
class WeatherExtraction:
    """Weather text and metadata produced by the hybrid extractor."""

    text: str
    nature: str
    status: str
    method: str = "general"
    score: int = 0


@dataclass(frozen=True)
class WeatherCandidate:
    """A bounded text block that may contain usable weather information."""

    text: str
    source: str
    page_number: int
    nature_hint: str = ""
    score: int = 0


@dataclass
class ExtractionSummary:
    """Compteurs de suivi de l'extraction."""

    extracted: int = 0
    warnings: int = 0
    failed: int = 0


# Lecture des arguments de la ligne de commande


def parse_args() -> argparse.Namespace:
    """Lit les chemins d'entrée, de sortie et de log depuis la ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Extract bulletin-level metadata from cached BSV text files."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing extracted text files (default: {DEFAULT_INPUT_DIR})",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"CSV file to create (default: {DEFAULT_OUTPUT_FILE})",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Extraction log file (default: {DEFAULT_LOG_FILE})",
    )

    return parser.parse_args()


# Fonctions de normalisation du texte


def ascii_key(value: str) -> str:
    """
    Supprime les accents d'une chaîne de caractères.

    Exemple :
        "Prévision météorologique" -> "Prevision meteorologique"
    """

    normalized = unicodedata.normalize("NFKD", value)

    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalized_heading(value: str) -> str:
    """
    Normalise un titre de section pour faciliter la comparaison.

    Exemple :
        "Prévision   Météorologique" -> "PREVISION METEOROLOGIQUE"
    """

    value = ascii_key(value).upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    return re.sub(r"\s+", " ", value).strip()


# Extraction de la région


def region_slug_from_path(text_path: Path, input_dir: Path) -> str:
    """
    Déduit la région à partir du dossier parent du fichier texte.

    Exemple :
        outputs/extracted_text/pays_de_la_loire_2025/xxx.txt
        -> pays_de_la_loire
    """

    relative_path = text_path.relative_to(input_dir)

    if len(relative_path.parts) < 2:
        raise ValueError("text file is not stored in a regional subdirectory")

    detected = re.sub(
        r"_20\d{2}$",
        "",
        relative_path.parts[0].lower(),
    )

    return REGION_ALIASES.get(detected, detected)


# Extraction du numéro, de la date et de la semaine


def extract_bulletin_number(filename: str) -> int | None:
    """
    Extrait le numéro du bulletin depuis le nom de fichier.

    Exemples :
        BSV17_LEG_GE_S32_2025 -> 17
        11_20250430_BSV-maraichage-n°11 -> 11
    """

    patterns = (
        r"^BSV0*(\d+)_",
        r"(?:^|_)NO_?0*(\d+)(?:_|$)",
        r"^0*(\d+)_\d{8}_BSV",
        r"\bN[°º]?\s*0*(\d+)\b",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            filename,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def date_from_filename(filename: str) -> date | None:
    """
    Extrait la date du bulletin depuis le nom de fichier si elle est présente.

    Exemple :
        11_20250430_BSV-maraichage-n°11 -> 2025-04-30
    """

    match = re.search(
        r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)",
        filename,
    )

    if match:
        try:
            return date(*(int(part) for part in match.groups()))
        except ValueError:
            return None

    match = re.search(
        r"(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)",
        filename,
    )
    if match:
        day, month, year = (int(part) for part in match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None

    searchable = ascii_key(filename).lower()
    month_names = "|".join(MONTHS)
    match = re.search(
        rf"(?:^|[_\s-])du[_\s-]+(\d{{1,2}})[_\s-]+({month_names})"
        rf"[_\s-]+(20\d{{2}})(?:$|[_\s.-])",
        searchable,
    )
    if not match:
        return None

    day, month_name, year = match.groups()
    try:
        return date(int(year), MONTHS[month_name], int(day))
    except ValueError:
        return None


def date_from_text(text: str) -> date | None:
    """
    Extrait la date du bulletin depuis le texte si elle n'est pas dans le nom.

    Exemple recherché :
        30 avril 2025
    """

    searchable = ascii_key(text[:5000]).lower()
    month_names = "|".join(MONTHS)

    match = re.search(
        rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b",
        searchable,
    )

    if not match:
        return None

    day, month_name, year = match.groups()

    try:
        return date(
            int(year),
            MONTHS[month_name],
            int(day),
        )
    except ValueError:
        return None


def date_from_grand_est_header(text: str) -> date | None:
    """
    Extrait la date officielle des BSV Grand Est.

    Exemple :
        BSV n°7 – 1er juin 2023
    """

    searchable = ascii_key(text[:4000]).lower()
    month_names = "|".join(MONTHS)

    match = re.search(
        rf"bsv\s*n\s*[°º]?\s*\d+\s*[–—-]\s*"
        rf"(\d{{1,2}})(?:er)?\s+({month_names})\s+(20\d{{2}})",
        searchable,
    )

    if not match:
        return None

    day, month_name, year = match.groups()

    try:
        return date(int(year), MONTHS[month_name], int(day))
    except ValueError:
        return None


def forecast_period(
    bulletin_date: date,
    region_slug: str,
) -> str:
    """
    Calcule la période de prévision météorologique sur sept jours.

    Pays de la Loire :
        du jour du bulletin à J+6.

    Grand Est :
        du lendemain du bulletin à J+7.
    """

    region_slug = REGION_ALIASES.get(region_slug, region_slug)

    if region_slug == "pdl":
        start_date = bulletin_date
    elif region_slug == "grandest":
        start_date = bulletin_date + timedelta(days=1)
    else:
        raise ValueError(
            f"forecast period rule is not defined for region: {region_slug}"
        )

    end_date = start_date + timedelta(days=6)

    return f"{start_date:%d/%m/%Y}-" f"{end_date:%d/%m/%Y}"


def year_from_path(text_path: Path, input_dir: Path) -> int | None:
    """Return the first four-digit year stored in the relative path."""

    for part in text_path.relative_to(input_dir).parts:
        if re.fullmatch(r"20\d{2}", part):
            return int(part)

    return None


def align_date_with_path_year(
    filename_date: date | None,
    path_year: int | None,
) -> tuple[date | None, str | None]:
    """
    Corrige les dates de nom de fichier dont l'annee contredit le dossier source.

    Certains fichiers officiels peuvent porter une date mal encodee dans le nom
    tout en etant ranges dans le bon dossier annuel. Exemple observe :
    pdl/maraichage/2024/BSV_maraichage_flash_no_01_20230118.txt correspond au
    flash du 18/01/2024. Sans correction, il collisionne avec le flash 2023.
    """

    if filename_date is None or path_year is None:
        return filename_date, None

    if filename_date.year == path_year:
        return filename_date, None

    try:
        corrected = filename_date.replace(year=path_year)
    except ValueError:
        return filename_date, (
            f"filename year {filename_date.year} differs from path year "
            f"{path_year}; date not corrected"
        )

    return corrected, (
        f"filename year {filename_date.year} corrected to path year {path_year}"
    )


def detect_template_id(
    region_slug: str,
    year: int | None,
    text: str,
) -> str:
    """Identify the extraction profile used by a bulletin."""

    region_slug = REGION_ALIASES.get(region_slug, region_slug)
    normalized_text = normalized_heading(text)

    if region_slug == "pdl":
        if year == 2023:
            return TEMPLATE_PDL_2023
        if year == 2024:
            return TEMPLATE_PDL_2024
        if year == 2025:
            return TEMPLATE_PDL_2025

    if region_slug == "grandest":
        has_weather = "DONNEES METEO" in normalized_text
        no_observations = "IL N Y A PAS EU DE PARCELLES OBSERVEES" in normalized_text
        if not has_weather and no_observations:
            return TEMPLATE_GRAND_EST_SANS_OBSERVATION
        return TEMPLATE_GRAND_EST

    if region_slug == "paca":
        return TEMPLATE_PACA_MARAICHAGE

    return "inconnu"


def build_bulletin_id(
    region_slug: str,
    year: int | str,
    number: int | None,
    filename: str,
) -> str:
    """Build a stable ID while keeping flash bulletins distinct."""

    region_slug = REGION_ALIASES.get(region_slug, region_slug)
    number_key = f"{number:02d}" if number is not None else filename
    normalized_filename = normalized_heading(filename)
    if "ALERTE" in normalized_filename:
        document_kind = "alerte"
    elif "FLASH" in normalized_filename:
        document_kind = "flash"
    else:
        document_kind = "bsv"

    if region_slug == "paca":
        bulletin_date = date_from_filename(filename)
        if bulletin_date is not None:
            return (
                f"{region_slug}_{year}_{document_kind}{number_key}_"
                f"{bulletin_date:%Y%m%d}"
            )

    return f"{region_slug}_{year}_{document_kind}{number_key}"


# Nettoyage des sections extraites


def clean_section(lines: list[str]) -> str:
    """
    Nettoie un bloc de lignes extrait d'une section du bulletin.

    La fonction retire :
        - les lignes vides ;
        - les marqueurs de page ;
        - les en-têtes récurrents ;
        - les numéros de page ;
        - les coupures de mots liées aux retours ligne.
    """

    kept: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line or re.fullmatch(r"--- PAGE \d+ ---", line):
            continue

        if re.match(r"^Page \d+\b", line, flags=re.IGNORECASE):
            continue

        heading = normalized_heading(line)
        compact_heading = heading.replace(" ", "")

        if re.fullmatch(r"BSV \d+ P \d+", heading):
            continue

        if compact_heading.startswith(
            (
                "PAYSDELALOIRE",
                "BSVMARAICHAGE",
                "RESEAUDESURVEILLANCE",
            )
        ):
            continue

        kept.append(line)

    text = " ".join(kept)

    # Supprime les coupures typographiques : "développe- ment" -> "développement".
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    return re.sub(r"\s+", " ", text).strip()


# Extraction de la prévision météo - Pays de la Loire


def split_pages(text: str) -> list[tuple[int, list[str]]]:
    """Split extracted text into numbered pages while preserving line order."""

    pages: list[tuple[int, list[str]]] = []
    page_number = 1
    current: list[str] = []

    for line in text.splitlines():
        marker = re.fullmatch(r"--- PAGE (\d+) ---", line.strip())
        if marker:
            if current:
                pages.append((page_number, current))
            page_number = int(marker.group(1))
            current = []
            continue
        current.append(line)

    if current:
        pages.append((page_number, current))

    return pages


def matched_weather_start(heading: str) -> str:
    """Return the weather heading pattern found in a plausible short title."""

    for pattern in WEATHER_START_PATTERNS:
        if heading == pattern:
            return pattern
        if pattern != "METEO" and pattern in heading and len(heading) <= 100:
            return pattern
    return ""


def weather_nature_hint(pattern: str) -> str:
    """Map the retained candidate's own heading to a weather nature."""

    if "BILAN" in pattern:
        return "bilan_tendance"
    if "PREVISION" in pattern or pattern == "DONNEES METEO":
        return "prevision"
    return ""


def is_general_weather_end(line: str, heading: str) -> bool:
    """Detect a strong section or footer boundary after a weather block."""

    compact = heading.replace(" ", "")
    if heading in GENERAL_WEATHER_SECTION_ENDS:
        return True
    if re.fullmatch(r"BSV \d+ P \d+", heading):
        return True
    return compact.startswith(
        (
            "ABONNEMENTBSV",
            "ACCEDERAU",
            "REMERCIEMENTS",
        )
    )


def paragraph_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Return non-empty paragraph ranges as half-open line indexes."""

    ranges: list[tuple[int, int]] = []
    start: int | None = None

    for index, line in enumerate(lines + [""]):
        if line.strip() and start is None:
            start = index
        elif not line.strip() and start is not None:
            ranges.append((start, index))
            start = None

    return ranges


def weather_term_count(text: str) -> int:
    """Count distinct meteorological concepts in a candidate."""

    heading = normalized_heading(text)
    return sum(
        bool(re.search(rf"\b{re.escape(term)}\b", heading))
        for term in WEATHER_CONTEXT_TERMS
    )


def score_weather_candidate(candidate: WeatherCandidate) -> WeatherCandidate:
    """Score a candidate using content, source and bounded-length signals."""

    text = candidate.text
    heading = normalized_heading(text)
    word_count = len(text.split())
    weather_terms = weather_term_count(text)
    favorable_terms = sum(term in heading for term in WEATHER_FAVORABILITY_TERMS)
    culture_terms = sum(
        bool(re.search(rf"\b{re.escape(term)}S?\b", heading))
        for term in WEATHER_CULTURE_TERMS
    )

    score = min(weather_terms * 2, 20) + favorable_terms * 2

    if candidate.source.startswith("heading:"):
        score += 8
    elif candidate.source == "weather_source":
        score += 4
    elif candidate.source == "keyword_paragraph":
        score += 2

    if "PREVISION" in heading or "7 JOURS" in heading:
        score += 3
    if "SOURCE METEO" in heading or "SOURCE WEATHER MEASURES" in heading:
        score += 2

    if 20 <= word_count <= 220:
        score += 5
    elif 8 <= word_count <= MAX_WEATHER_WORDS:
        score += 2
    elif word_count < 5:
        score -= 8
    else:
        score -= 15

    if culture_terms > 3:
        score -= (culture_terms - 3) * 2
    if "SOMMAIRE" in heading or "CLIQUEZ SUR LE SOMMAIRE" in heading:
        score -= 10
    if "ABONNEMENT BSV" in heading:
        score -= 8

    return WeatherCandidate(
        text=candidate.text,
        source=candidate.source,
        page_number=candidate.page_number,
        nature_hint=candidate.nature_hint,
        score=score,
    )


def find_weather_candidates(text: str) -> list[WeatherCandidate]:
    """Find bounded candidate blocks from headings, sources and paragraphs."""

    candidates: list[WeatherCandidate] = []

    for page_number, lines in split_pages(text):
        headings = [normalized_heading(line) for line in lines]

        for index, heading in enumerate(headings):
            pattern = matched_weather_start(heading)
            if pattern:
                end = len(lines)
                for candidate_end in range(index + 1, len(lines)):
                    if is_general_weather_end(
                        lines[candidate_end],
                        headings[candidate_end],
                    ):
                        end = candidate_end
                        break
                block = clean_section(lines[index + 1 : end])
                if block:
                    candidates.append(
                        WeatherCandidate(
                            text=block,
                            source=f"heading:{pattern}",
                            page_number=page_number,
                            nature_hint=weather_nature_hint(pattern),
                        )
                    )

            compact = heading.replace(" ", "")
            if "SOURCEWEATHERMEASURES" in compact or "SOURCEMETEOFRANCE" in compact:
                preceding_pattern = ""
                for preceding_heading in reversed(headings[: index + 1]):
                    preceding_pattern = matched_weather_start(preceding_heading)
                    if preceding_pattern:
                        break
                end = len(lines)
                for candidate_end in range(index + 1, len(lines)):
                    if is_general_weather_end(
                        lines[candidate_end],
                        headings[candidate_end],
                    ):
                        end = candidate_end
                        break
                block = clean_section(lines[index:end])
                if block:
                    candidates.append(
                        WeatherCandidate(
                            text=block,
                            source="weather_source",
                            page_number=page_number,
                            nature_hint=weather_nature_hint(preceding_pattern),
                        )
                    )

        for start, end in paragraph_ranges(lines):
            if any(
                matched_weather_start(headings[index]) for index in range(start, end)
            ):
                continue
            block = clean_section(lines[start:end])
            if weather_term_count(block) >= 2:
                candidates.append(
                    WeatherCandidate(
                        text=block,
                        source="keyword_paragraph",
                        page_number=page_number,
                    )
                )

    unique: dict[str, WeatherCandidate] = {}
    for candidate in candidates:
        key = normalized_heading(candidate.text)
        if key and key not in unique:
            unique[key] = score_weather_candidate(candidate)

    structured = {
        key: candidate
        for key, candidate in unique.items()
        if candidate.source.startswith("heading:")
        or candidate.source == "weather_source"
    }
    if structured:
        unique = structured

    return sorted(
        unique.values(),
        key=lambda candidate: (
            candidate.score,
            weather_term_count(candidate.text),
            -abs(len(candidate.text.split()) - 100),
        ),
        reverse=True,
    )


def detect_weather_nature(text: str, nature_hint: str = "") -> str:
    """Describe the retained block without inspecting the whole bulletin."""

    if nature_hint:
        return nature_hint

    headings = [normalized_heading(line) for line in text.splitlines()]
    if any("BILAN METEOROLOGIQUE" in heading for heading in headings):
        return "bilan_tendance"
    if any(
        "PREVISION METEOROLOGIQUE" in heading
        or "PREVISIONS METEOROLOGIQUES" in heading
        or heading == "DONNEES METEO"
        for heading in headings
    ):
        return "prevision"
    return "observation"


def extract_weather_general(text: str) -> WeatherExtraction:
    """Extract weather by selecting the strongest bounded candidate."""

    candidates = find_weather_candidates(text)
    if not candidates or candidates[0].score < 4:
        return WeatherExtraction(
            text="",
            nature="absent",
            status="absente",
            method="general",
            score=candidates[0].score if candidates else 0,
        )

    best = candidates[0]
    return WeatherExtraction(
        text=best.text,
        nature=detect_weather_nature(best.text, best.nature_hint),
        status=weather_status(best.text),
        method=f"general:{best.source}",
        score=best.score,
    )


def extract_pdl_2023_weather(text: str) -> str:
    """
    Extrait le contexte météo qualitatif des BSV PDL 2023.

    En 2023, la météo est répartie entre :
    - un bloc d'actualités sanitaire en haut de page ;
    - un commentaire bilan/tendance près du tableau météo.

    Le tableau chiffré complet n'est pas conservé pour limiter le bruit.
    """

    page_one = text.split("--- PAGE 2 ---", maxsplit=1)[0]
    lines = page_one.splitlines()
    headings = [normalized_heading(line) for line in lines]

    fragments: list[str] = []

    # Bloc 1 : actualités météo/sanitaire en haut de page
    actualites_lines = []
    for line, heading in zip(lines, headings):
        h = heading
        if any(
            term in h
            for term in (
                "LA METEO",
                "CONDITIONS CLIMATIQUES",
                "PROPICE",
                "FAVORABLE",
                "DEVELOPPEMENT DES MALADIES",
                "DEVELOPPEMENT DES PUCERONS",
                "ATTENTION AU DEVELOPPEMENT",
            )
        ):
            actualites_lines.append(line)

    actualites = clean_section(actualites_lines)
    if actualites:
        fragments.append(f"[ACTUALITES] {actualites}")

    # Bloc 2 : commentaire météo bilan/tendance
    start_candidates = [
        i
        for i, h in enumerate(headings)
        if h.startswith("LA SEMAINE")
        or "LA FIN DE SEMAINE" in h
        or "LES TEMPERATURES" in h
    ]

    if start_candidates:
        start = start_candidates[0]
        end = len(lines)

        for i in range(start + 1, len(lines)):
            compact = headings[i].replace(" ", "")
            if (
                compact.startswith("ACCEDERAU")
                or compact.startswith("SITEDELA")
                or compact.startswith("SURVEILLANCE")
                or compact.startswith("BIOLOGIQUE")
                or compact.startswith("TERRITOIRE")
            ):
                end = i
                break

        bilan_tendance = clean_section(lines[start:end])
        if bilan_tendance:
            fragments.append(f"[BILAN_TENDANCE] {bilan_tendance}")

    return " ".join(fragments).strip()


def extract_pays_de_la_loire_weather(
    text: str,
    template_id: str = TEMPLATE_PDL_2025,
) -> str:
    """
    Extrait la section météo des BSV Pays de la Loire.

    Début :
        PREVISION METEOROLOGIQUE

    Fin :
        EVALUATION DES RISQUES
    """

    lines = text.splitlines()
    headings = [normalized_heading(line) for line in lines]

    try:
        start = headings.index("PREVISION METEOROLOGIQUE") + 1
    except ValueError:
        return ""

    end = len(lines)

    for index in range(start, len(lines)):
        if re.fullmatch(r"--- PAGE \d+ ---", lines[index].strip()):
            end = index
            break
        if (
            template_id == TEMPLATE_PDL_2025
            and headings[index] == "EVALUATION DES RISQUES"
        ):
            end = index
            break
        if (
            template_id == TEMPLATE_PDL_2024
            and headings[index] in PAYS_DE_LA_LOIRE_SECTION_ENDS
        ):
            end = index
            break

    return clean_section(lines[start:end])


# Extraction de la prévision météo - Grand Est


def extract_grand_est_weather(text: str) -> str:
    """
    Extrait la section météo des BSV Grand Est.

    Les BSV Grand Est peuvent contenir plusieurs occurrences de "DONNEES METEO".
    On retient la dernière occurrence, généralement celle située avant les sections cultures.
    """

    lines = text.splitlines()
    headings = [normalized_heading(line) for line in lines]

    occurrences = [
        index for index, heading in enumerate(headings) if heading == "DONNEES METEO"
    ]

    if not occurrences:
        return ""

    start = occurrences[-1] + 1
    end = len(lines)

    for index in range(start, len(lines)):
        if headings[index] in GRAND_EST_SECTION_ENDS:
            end = index
            break

    return clean_section(lines[start:end])


def weather_text_is_informative(text: str) -> bool:
    """Return whether extracted text contains actual weather conditions."""

    heading = normalized_heading(text)
    return any(
        re.search(rf"\b{re.escape(term)}\b", heading)
        for term in INFORMATIVE_WEATHER_TERMS
    )


def weather_status(text: str, template_id: str = "") -> str:
    """Classify whether weather text can be interpreted downstream."""

    if not text.strip():
        return "absente"
    if len(text.split()) > MAX_WEATHER_WORDS:
        return "trop_longue"
    if template_id in {
        TEMPLATE_PDL_2023,
        TEMPLATE_PDL_2024,
        TEMPLATE_PDL_2025,
    }:
        return "lisible"
    if not weather_text_is_informative(text):
        return "image_seule"
    return "lisible"


def fallback_by_template(text: str, template_id: str) -> WeatherExtraction:
    """Apply a known profile only when general extraction is insufficient."""

    if template_id == TEMPLATE_PDL_2023:
        weather = extract_pdl_2023_weather(text)
        nature = "bilan_tendance"
    elif template_id in {TEMPLATE_PDL_2024, TEMPLATE_PDL_2025}:
        weather = extract_pays_de_la_loire_weather(text, template_id)
        nature = "prevision"
    elif template_id == TEMPLATE_GRAND_EST:
        weather = extract_grand_est_weather(text)
        nature = "prevision"
    elif template_id == TEMPLATE_GRAND_EST_SANS_OBSERVATION:
        weather = ""
        nature = "absent"
    elif template_id == TEMPLATE_PACA_MARAICHAGE:
        weather = ""
        nature = "absent"
    else:
        weather = ""
        nature = "absent"

    return WeatherExtraction(
        text=weather,
        nature=nature,
        status=weather_status(weather, template_id),
        method=f"fallback:{template_id}",
    )


def extract_weather(text: str, template_id: str) -> WeatherExtraction:
    """Use general extraction first and template rules only as safeguards."""

    if template_id == TEMPLATE_PACA_MARAICHAGE:
        return fallback_by_template(text, template_id)

    general = extract_weather_general(text)
    if template_id == TEMPLATE_GRAND_EST_SANS_OBSERVATION:
        return fallback_by_template(text, template_id)
    if general.status == "lisible":
        return general

    fallback = fallback_by_template(text, template_id)
    status_rank = {
        "absente": 0,
        "trop_longue": 1,
        "image_seule": 2,
        "lisible": 3,
    }
    if status_rank.get(fallback.status, 0) > status_rank.get(general.status, 0):
        return fallback

    return general


def extract_explicit_forecast_period(text: str) -> str:
    """Extract an explicitly written numeric date range from weather text."""

    match = re.search(
        r"(?<!\d)(\d{1,2})[/.](\d{1,2})[/.](20\d{2})"
        r"\s*(?:-|AU|A)\s*"
        r"(\d{1,2})[/.](\d{1,2})[/.](20\d{2})(?!\d)",
        ascii_key(text).upper(),
    )
    if not match:
        return ""

    start_day, start_month, start_year, end_day, end_month, end_year = (
        int(value) for value in match.groups()
    )
    try:
        start = date(start_year, start_month, start_day)
        end = date(end_year, end_month, end_day)
    except ValueError:
        return ""

    if end < start:
        return ""

    return f"{start:%d/%m/%Y}-{end:%d/%m/%Y}"


def extract_bulletin(
    text_path: Path,
    input_dir: Path,
) -> tuple[Bulletin, list[str]]:
    """
    Extrait les métadonnées d'un fichier texte BSV.

    Champs extraits :
        - bulletin_id ;
        - région ;
        - date du bulletin ;
        - periode_prevision_meteo ;
        - prévision météo.
    """

    text = text_path.read_text(encoding="utf-8")

    region_slug = region_slug_from_path(
        text_path,
        input_dir,
    )

    region = REGION_NAMES.get(
        region_slug,
        region_slug.replace("_", " ").title(),
    )

    number = extract_bulletin_number(text_path.stem)
    path_year = year_from_path(text_path, input_dir)
    filename_date, date_correction_warning = align_date_with_path_year(
        date_from_filename(text_path.stem),
        path_year,
    )

    if region_slug == "grandest":
        bulletin_date = (
            filename_date or date_from_grand_est_header(text) or date_from_text(text)
        )
    else:
        bulletin_date = filename_date or date_from_text(text)

    detected_year = bulletin_date.year if bulletin_date is not None else path_year
    template_id = detect_template_id(
        region_slug,
        detected_year,
        text,
    )
    weather = extract_weather(text, template_id)

    warnings: list[str] = []

    if number is None:
        warnings.append("bulletin number not found")

    if date_correction_warning:
        warnings.append(date_correction_warning)

    if template_id == "inconnu":
        warnings.append("weather template not identified")

    if bulletin_date is None:
        warnings.append("bulletin date not found")
        periode_prevision_meteo = ""
    elif weather.nature == "prevision" and weather.status != "absente":
        try:
            periode_prevision_meteo = forecast_period(
                bulletin_date,
                region_slug,
            )
        except ValueError as exc:
            warnings.append(str(exc))
            periode_prevision_meteo = ""
    elif weather.nature in {"bilan", "bilan_tendance"}:
        periode_prevision_meteo = extract_explicit_forecast_period(weather.text)
    else:
        periode_prevision_meteo = ""

    if weather.status == "absente" and template_id not in {
        TEMPLATE_GRAND_EST_SANS_OBSERVATION,
        TEMPLATE_PACA_MARAICHAGE,
    }:
        warnings.append("weather section not found")
    elif weather.status == "image_seule":
        warnings.append("weather text is image-only")
    elif weather.status == "trop_longue":
        warnings.append("weather section is suspiciously long")

    year = detected_year if detected_year is not None else "unknown"
    bulletin_id = build_bulletin_id(
        region_slug,
        year,
        number,
        text_path.stem,
    )

    return (
        Bulletin(
            bulletin_id=bulletin_id,
            region=region,
            date_bulletin=bulletin_date.isoformat() if bulletin_date else "",
            template_id=template_id,
            nature_meteo=weather.nature,
            meteo_extraction_status=weather.status,
            meteo_extraction_method=weather.method,
            meteo_extraction_score=weather.score,
            periode_prevision_meteo=periode_prevision_meteo,
            prevision_meteo=weather.text,
        ),
        warnings,
    )


# Découverte des fichiers texte


def discover_text_files(input_dir: Path) -> list[Path]:
    """
    Recherche tous les fichiers .txt générés à l'étape précédente.
    """

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    return sorted(
        input_dir.rglob("*.txt"),
        key=lambda path: str(path.relative_to(input_dir)).casefold(),
    )


# Écriture CSV sécurisée


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """
    Écrit un fichier CSV de manière sécurisée.

    Le fichier est d'abord écrit dans un fichier temporaire,
    puis remplacé une fois l'écriture terminée.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


# Extraction complète du corpus


def extract_all(
    input_dir: Path,
    output_file: Path,
    log_file: Path,
) -> ExtractionSummary:
    """
    Extrait les métadonnées de tous les fichiers texte du corpus.

    Produit :
        - data/processed/bsv_bulletins.csv
        - outputs/logs/bulletins_log.csv
    """

    summary = ExtractionSummary()
    bulletins: list[Bulletin] = []
    logs: list[dict[str, str]] = []

    for text_path in discover_text_files(input_dir):
        relative_path = str(text_path.relative_to(input_dir))

        try:
            bulletin, warnings = extract_bulletin(
                text_path,
                input_dir,
            )

            bulletins.append(bulletin)

            summary.extracted += 1
            summary.warnings += len(warnings)

            logs.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "text_file": relative_path,
                    "status": "WARNING" if warnings else "OK",
                    "message": "; ".join(warnings),
                }
            )

        except Exception as exc:
            summary.failed += 1

            logs.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "text_file": relative_path,
                    "status": "ERROR",
                    "message": str(exc),
                }
            )

            print(
                f"ERROR {relative_path}: {exc}",
                file=sys.stderr,
            )

    bulletins.sort(
        key=lambda item: (
            item.region,
            item.date_bulletin,
            item.bulletin_id,
        )
    )

    ids = [bulletin.bulletin_id for bulletin in bulletins]

    if len(ids) != len(set(ids)):
        raise ValueError("duplicate bulletin_id values detected")

    write_csv(
        output_file,
        [asdict(bulletin) for bulletin in bulletins],
        list(Bulletin.__dataclass_fields__),
    )

    write_csv(
        log_file,
        logs,
        [
            "timestamp",
            "text_file",
            "status",
            "message",
        ],
    )

    return summary


# Point d'entrée du script


def main() -> int:
    """
    Lance l'extraction des métadonnées des bulletins.
    """

    args = parse_args()

    try:
        summary = extract_all(
            args.input_dir.resolve(),
            args.output_file.resolve(),
            args.log_file.resolve(),
        )

    except (FileNotFoundError, ValueError) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "Summary: "
        f"{summary.extracted} bulletins, "
        f"{summary.warnings} warnings, "
        f"{summary.failed} errors."
    )

    print(f"CSV: {args.output_file.resolve()}")
    print(f"Log: {args.log_file.resolve()}")

    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
