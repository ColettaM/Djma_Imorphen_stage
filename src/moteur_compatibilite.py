"""Evaluate CNN predictions against regional BSV phytosanitary context.

Commande :
python src/moteur_compatibilite.py --query-file requete.json

Le moteur retourne notamment : Accord / Partiel / Non_corrobore / Indetermine
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BULLETINS_FILE = PROJECT_ROOT / "data" / "processed" / "bsv_bulletins.csv"
DEFAULT_OBSERVATIONS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "bsv_observations.csv"
)
DEFAULT_REFERENTIAL_FILE = PROJECT_ROOT / "data" / "raw" / "referentiel_cnn_bsv.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bsv_compatibility.csv"

RESOLVED_STATUSES = {"resolved_direct", "resolved_hote_associe", "resolu"}
DIRECT_STATUSES = {"resolved_direct", "resolu"}
HOST_STATUSES = {"resolved_hote_associe"}

MAX_BULLETIN_AGE_DAYS = 7

TOMATO_CULTURE = "tomate"

SEASONAL_WEEK_WINDOWS = {
    "pays de la loire": (13, 40),
    "pdl": (13, 40),
    "grand est": (17, 38),
    "grandest": (17, 38),
}

METEO_FAVORABLE = {
    "humide",
    "humidite",
    "hygrometrie",
    "pluie",
    "pluies",
    "rosee",
    "brouillard",
}
METEO_DEFAVORABLE = {
    "sec",
    "seche",
    "chaud",
    "chaude",
    "ensoleille",
    "ensoleillement",
    "vent fort",
    "vents forts",
}

OUTPUT_COLUMNS = [
    "image_id",
    "date_observation",
    "region",
    "classe_cnn",
    "score_cnn",
    "agent_referentiel",
    "bulletin_id_retenu",
    "date_bulletin_retenu",
    "fenetre_saisonniere_debut",
    "fenetre_saisonniere_fin",
    "agent_present_bulletin",
    "resolution_status_bulletin",
    "niveau_risque",
    "meteo_status",
    "meteo_favorable",
    "compatibilite",
    "decision_reason",
]


@dataclass(frozen=True)
class Prediction:
    classe: str
    score: float


@dataclass(frozen=True)
class CompatibilityQuery:
    image_id: str
    date_observation: date
    region: str
    top_k: list[Prediction]


@dataclass
class EngineData:
    bulletins: list[dict[str, Any]]
    observations: list[dict[str, str]]
    referential_by_class: dict[str, dict[str, str]]
    bulletins_by_region: dict[str, list[dict[str, Any]]]
    observations_by_bulletin: dict[str, list[dict[str, str]]]
    covered_classes: set[str]
    covered_agents: set[str]


@dataclass(frozen=True)
class ReferentialEntry:
    classe_dataset: str
    cultures_concernees: frozenset[str]
    nom_fr: str
    agent: str
    synonymes: frozenset[str]
    hotes: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Top-K CNN predictions against BSV context."
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        required=True,
        help="JSON object or list of objects containing image_id, date_observation, region and top_k.",
    )
    parser.add_argument("--bulletins-file", type=Path, default=DEFAULT_BULLETINS_FILE)
    parser.add_argument(
        "--observations-file",
        type=Path,
        default=DEFAULT_OBSERVATIONS_FILE,
    )
    parser.add_argument(
        "--referential-file",
        type=Path,
        default=DEFAULT_REFERENTIAL_FILE,
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def ascii_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize(value: str) -> str:
    value = ascii_key(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid ISO date: {value!r}") from exc


def read_dicts_auto(path: Path, delimiter: str | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                if delimiter is None:
                    sample = handle.read(4096)
                    handle.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                        reader = csv.DictReader(handle, dialect=dialect)
                    except csv.Error:
                        header = sample.splitlines()[0] if sample else ""
                        fallback_delimiter = ";" if header.count(";") > header.count(",") else ","
                        reader = csv.DictReader(handle, delimiter=fallback_delimiter)
                else:
                    reader = csv.DictReader(handle, delimiter=delimiter)
                return [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def get_column(row: dict[str, str], normalized_name: str) -> str:
    target_name = normalize(normalized_name)
    for key, value in row.items():
        if normalize(key) == target_name:
            return value or ""
    return ""


def split_terms(value: str) -> set[str]:
    return {
        normalize(term)
        for term in re.split(r"[;|,]", value or "")
        if normalize(term)
    }


def load_referential_entries(rows: list[dict[str, str]]) -> list[ReferentialEntry]:
    entries: list[ReferentialEntry] = []

    for row in rows:
        classe_dataset = get_column(row, "classe_dataset").strip()
        nom_fr = get_column(row, "nom_fr").strip()
        agent = get_column(row, "agent").strip()
        synonymes = split_terms(get_column(row, "synonymes_bsv"))
        if nom_fr:
            synonymes.add(normalize(nom_fr))

        if not classe_dataset or not synonymes:
            continue

        entries.append(
            ReferentialEntry(
                classe_dataset=classe_dataset,
                cultures_concernees=frozenset(split_terms(get_column(row, "cultures_concernees"))),
                nom_fr=nom_fr,
                agent=agent,
                synonymes=frozenset(synonymes),
                hotes=frozenset(split_terms(get_column(row, "hotes"))),
            )
        )

    return entries


def term_matches_text(term: str, text_norm: str) -> bool:
    if not term or not text_norm:
        return False
    return (
        re.search(rf"\b{re.escape(term)}\b", text_norm) is not None
        or re.search(rf"\b{re.escape(text_norm)}\b", term) is not None
    )


def resolve_observation(
    observation: dict[str, str],
    referential: list[ReferentialEntry],
) -> tuple[str, str, str, str]:
    culture_norm = normalize(observation.get("culture", ""))
    bio_norm = normalize(observation.get("bioagresseur", ""))

    matching_entries = [
        entry
        for entry in referential
        if any(term_matches_text(term, bio_norm) for term in entry.synonymes)
    ]

    if not matching_entries:
        return "", "", "non_resolu", "aucune correspondance bioagresseur dans le referentiel"

    direct_matches = [
        entry
        for entry in matching_entries
        if culture_norm in entry.cultures_concernees
    ]
    if direct_matches:
        entry = direct_matches[0]
        return entry.agent, entry.classe_dataset, "resolved_direct", ""

    host_matches = [
        entry
        for entry in matching_entries
        if culture_norm in entry.hotes
    ]
    if host_matches:
        entry = host_matches[0]
        return entry.agent, entry.classe_dataset, "resolved_hote_associe", ""

    return (
        "",
        "",
        "hors_perimetre_cnn",
        "bioagresseur connu mais culture/hote non compatible avec le referentiel CNN-BSV",
    )


def enrich_observations_with_resolution(
    observations: list[dict[str, str]],
    referential: list[ReferentialEntry],
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []

    for observation in observations:
        row = dict(observation)
        if not row.get("resolution_status"):
            agent, classe_dataset, status, comment = resolve_observation(row, referential)
            row["agent_resolu"] = agent
            row["classe_dataset"] = classe_dataset
            row["resolution_status"] = status
            row["resolution_comment"] = comment
        enriched.append(row)

    return enriched


def load_queries(path: Path) -> list[CompatibilityQuery]:
    if not path.is_file():
        raise FileNotFoundError(f"Query file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    items = payload if isinstance(payload, list) else [payload]
    queries: list[CompatibilityQuery] = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Query #{index} must be a JSON object")

        predictions: list[Prediction] = []
        for prediction in item.get("top_k", []):
            if isinstance(prediction, dict):
                classe = prediction.get("classe", prediction.get("class", ""))
                score = prediction.get("score")
            elif isinstance(prediction, (list, tuple)) and len(prediction) == 2:
                classe, score = prediction
            else:
                raise ValueError(f"Invalid Top-K prediction in query #{index}: {prediction!r}")

            if not classe or score is None:
                raise ValueError(f"Prediction requires classe and score in query #{index}")
            predictions.append(Prediction(str(classe), float(score)))

        if not predictions:
            raise ValueError(f"Query #{index} has an empty top_k")

        region = str(item.get("region", "")).strip()
        if not region:
            raise ValueError(f"Query #{index} requires region")

        queries.append(
            CompatibilityQuery(
                image_id=str(item.get("image_id", f"image_{index}")),
                date_observation=parse_date(str(item.get("date_observation", ""))),
                region=region,
                top_k=predictions,
            )
        )

    return queries


def seasonal_window_for(region: str, year: int) -> tuple[date, date] | None:
    """Return the fixed BSV vegetable season used by the compatibility engine."""

    week_window = SEASONAL_WEEK_WINDOWS.get(normalize(region))
    if week_window is None:
        return None

    start_week, end_week = week_window
    start_day = date.fromisocalendar(year, start_week, 1)
    end_day = date.fromisocalendar(year, end_week, 7)
    return start_day, end_day


def is_in_season(region: str, observation_date: date) -> tuple[bool, tuple[date, date] | None]:
    window = seasonal_window_for(region, observation_date.year)
    return bool(window and window[0] <= observation_date <= window[1]), window


def charger_donnees(
    bulletins_file: Path = DEFAULT_BULLETINS_FILE,
    observations_file: Path = DEFAULT_OBSERVATIONS_FILE,
    referential_file: Path = DEFAULT_REFERENTIAL_FILE,
) -> EngineData:
    bulletins_raw = read_dicts_auto(bulletins_file)
    observations_raw = read_dicts_auto(observations_file)
    referential_rows = read_dicts_auto(referential_file, delimiter=";")
    referential_entries = load_referential_entries(referential_rows)
    observations = enrich_observations_with_resolution(
        observations_raw,
        referential_entries,
    )

    bulletins: list[dict[str, Any]] = []
    bulletins_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bulletins_raw:
        prepared: dict[str, Any] = dict(row)
        prepared["date_bulletin_parsed"] = parse_date(row.get("date_bulletin", ""))
        bulletins.append(prepared)
        bulletins_by_region[normalize(row.get("region", ""))].append(prepared)

    for rows in bulletins_by_region.values():
        rows.sort(key=lambda item: item["date_bulletin_parsed"])

    observations_by_bulletin: dict[str, list[dict[str, str]]] = defaultdict(list)
    covered_classes: set[str] = set()
    covered_agents: set[str] = set()
    for row in observations:
        observations_by_bulletin[row.get("bulletin_id", "")].append(row)
        if row.get("resolution_status", "") in RESOLVED_STATUSES:
            if row.get("classe_dataset", "").strip():
                covered_classes.add(row["classe_dataset"].strip())
            if row.get("agent_resolu", "").strip():
                covered_agents.add(normalize(row["agent_resolu"]))

    referential_by_class: dict[str, dict[str, str]] = {}
    for row in referential_rows:
        classe = row.get("classe_dataset", "").strip()
        if classe:
            referential_by_class[classe] = row

    return EngineData(
        bulletins=bulletins,
        observations=observations,
        referential_by_class=referential_by_class,
        bulletins_by_region=dict(bulletins_by_region),
        observations_by_bulletin=dict(observations_by_bulletin),
        covered_classes=covered_classes,
        covered_agents=covered_agents,
    )


def trouver_bulletin_retenu(
    bulletins_by_region: dict[str, list[dict[str, Any]]],
    region: str,
    date_observation: date,
    max_age_days: int = MAX_BULLETIN_AGE_DAYS,
) -> dict[str, Any] | None:
    candidates = bulletins_by_region.get(normalize(region), [])
    eligible = [
        bulletin
        for bulletin in candidates
        if bulletin["date_bulletin_parsed"] <= date_observation
    ]
    if not eligible:
        return None

    selected = eligible[-1]
    if date_observation - selected["date_bulletin_parsed"] > timedelta(days=max_age_days):
        return None

    return selected


def contains_any(text_norm: str, terms: set[str]) -> bool:
    return any(
        re.search(rf"\b{re.escape(term)}\b", text_norm)
        for term in terms
    )


def extraire_meteo_favorable(
    prevision_meteo: str,
    classe_dataset: str,
    meteo_extraction_status: str = "lisible",
) -> tuple[bool | None, str]:
    """Return a simple auditable weather modulation signal."""

    if normalize(meteo_extraction_status) != "lisible":
        return None, "non_utilisee"

    if not (prevision_meteo or "").strip():
        return None, "non_utilisee"

    text_norm = normalize(prevision_meteo)
    has_favorable = contains_any(text_norm, METEO_FAVORABLE)
    has_defavorable = contains_any(text_norm, METEO_DEFAVORABLE)

    if has_favorable and not has_defavorable:
        return True, "favorable"
    if has_defavorable and not has_favorable:
        return False, "defavorable"

    return None, "neutre"


def risk_from_text(text: str) -> str | None:
    text_norm = normalize(text)
    if not text_norm:
        return None

    if text_norm in {
        "tres fort",
        "fort",
        "moyen a fort",
        "moyen",
        "faible a moyen",
        "faible",
        "nul",
        "en hausse",
        "en baisse",
        "stable",
    }:
        return text_norm.replace(" ", "_")

    high_patterns = (
        r"\brisque\b.{0,35}\btres fort\b",
        r"\brisque\b.{0,35}\btres eleve\b",
        r"\brisque\b.{0,35}\beleve\b",
        r"\brisque\b.{0,35}\bfort\b",
    )
    medium_patterns = (
        r"\brisque\b.{0,35}\bmoyen\b",
        r"\brisque\b.{0,35}\bmodere\b",
    )
    low_patterns = (
        r"\brisque\b.{0,35}\bfaible\b",
        r"\brisque\b.{0,35}\bnul\b",
        r"\bpas de risque\b",
    )

    if any(re.search(pattern, text_norm) for pattern in high_patterns):
        return "fort"
    if any(re.search(pattern, text_norm) for pattern in medium_patterns):
        return "moyen"
    if re.search(r"\brisque\b.*\b(?:augmente|hausse)\b", text_norm):
        return "en_hausse"
    if re.search(r"\brisque\b.*\b(?:diminue|baisse)\b", text_norm):
        return "en_baisse"
    if re.search(r"\brisque\b.*\b(?:stable|maintient|persiste)\b", text_norm):
        return "stable"
    if re.search(r"\b(?:risque nul|pas de risque)\b", text_norm):
        return "nul"
    if any(re.search(pattern, text_norm) for pattern in low_patterns):
        return "faible"
    return None


def extraire_niveau_risque(rows: list[dict[str, str]]) -> str | None:
    """Extract the strongest explicit risk level from matched observation rows."""

    levels: list[str] = []
    for row in rows:
        for field in (
            "niveau_risque_observation",
            "analyse_risque",
            "observation",
            "evolution",
        ):
            level = risk_from_text(row.get(field, ""))
            if level:
                levels.append(level)

    for level in (
        "tres_fort",
        "fort",
        "moyen_a_fort",
        "moyen",
        "faible_a_moyen",
        "faible",
        "nul",
        "en_hausse",
        "stable",
        "en_baisse",
    ):
        if level in levels:
            return level
    return None


def chercher_dans_bsv_resolved(
    rows: list[dict[str, str]],
    classe_dataset: str,
    agent: str,
) -> list[dict[str, str]]:
    agent_norm = normalize(agent)
    return [
        row
        for row in rows
        if row.get("classe_dataset", "").strip() == classe_dataset
        or (
            agent_norm
            and normalize(row.get("agent_resolu", "")) == agent_norm
            and row.get("resolution_status", "") in RESOLVED_STATUSES
        )
    ]


def est_signal_tomate_compatible(row: dict[str, str]) -> bool:
    """Return True for direct tomato signals or associated-host CNN tomato signals."""

    return row.get("resolution_status", "") in RESOLVED_STATUSES and bool(
        row.get("classe_dataset", "").strip()
    )


def est_observation_tomate_directe(row: dict[str, str]) -> bool:
    return normalize(row.get("culture", "")) == TOMATO_CULTURE


def appliquer_regle_decision(
    *,
    bulletin_present: bool,
    dans_fenetre_saisonniere: bool,
    agent_couvert_corpus: bool,
    agent_rows_bulletin: list[dict[str, str]],
    direct_tomato_rows_bulletin: list[dict[str, str]],
    tomato_compatible_rows_bulletin: list[dict[str, str]],
    niveau_risque: str | None,
    meteo_status: str,
) -> tuple[str, str]:
    if not dans_fenetre_saisonniere:
        return "Indetermine", "date hors saison regionale"

    if not bulletin_present:
        return "Indetermine", "aucun bulletin disponible dans les 7 jours precedant la date"

    if not agent_couvert_corpus:
        return "Indetermine", "classe CNN jamais couverte dans le corpus resolu"

    if agent_rows_bulletin:
        if niveau_risque is None:
            base_status = "Coherent"
            reason = "signal direct/associe present, niveau de risque absent"
        else:
            base_status = ""
            reason = ""

        statuses = {row.get("resolution_status", "") for row in agent_rows_bulletin}
        has_direct = bool(statuses & DIRECT_STATUSES)
        has_host_only = bool(statuses & HOST_STATUSES) and not has_direct

        if has_host_only:
            return "Partiel", "signal present uniquement via un hote associe"

        if has_direct and not base_status:
            if niveau_risque in {"tres_fort", "fort", "moyen_a_fort"}:
                base_status = "Coherent_renforce"
                reason = "signal direct avec risque fort ou tres fort"
            elif niveau_risque == "moyen":
                base_status = "Coherent"
                reason = "signal direct avec risque moyen"
            elif niveau_risque in {"faible_a_moyen", "faible", "nul", "en_baisse"}:
                base_status = "Partiel"
                reason = "signal direct mais risque faible, nul ou en baisse"
            elif niveau_risque in {"en_hausse", "stable"}:
                base_status = "Coherent"
                reason = "signal direct avec risque en hausse ou stable"
            else:
                base_status = "Coherent"
                reason = "signal direct present"

        if meteo_status == "favorable" and base_status == "Coherent":
            return "Coherent_renforce", reason + "; meteo favorable"
        if meteo_status == "favorable" and base_status == "Partiel":
            return "Coherent", reason + "; meteo favorable"
        if meteo_status == "defavorable" and base_status == "Coherent_renforce":
            return "Coherent", reason + "; meteo defavorable"
        if meteo_status == "defavorable" and base_status == "Coherent":
            return "Partiel", reason + "; meteo defavorable"

        return base_status or "Indetermine", reason or "combinaison de signaux non couverte"

    if not direct_tomato_rows_bulletin and not tomato_compatible_rows_bulletin:
        return "Indetermine", "aucun signal tomate direct ou hote associe dans le bulletin retenu"

    return (
        "Non_corrobore",
        "classe absente du bulletin retenu alors que des signaux tomate ou hotes associes existent",
    )


def evaluer_requete(query: CompatibilityQuery, data: EngineData) -> list[dict[str, Any]]:
    bulletin = trouver_bulletin_retenu(
        data.bulletins_by_region,
        query.region,
        query.date_observation,
    )
    in_window, window = is_in_season(query.region, query.date_observation)

    bulletin_rows = (
        data.observations_by_bulletin.get(bulletin["bulletin_id"], [])
        if bulletin
        else []
    )
    direct_tomato_rows = [
        row
        for row in bulletin_rows
        if est_observation_tomate_directe(row)
    ]
    tomato_compatible_rows = [
        row
        for row in bulletin_rows
        if est_signal_tomate_compatible(row)
    ]
    output_rows: list[dict[str, Any]] = []

    for prediction in query.top_k:
        ref = data.referential_by_class.get(prediction.classe, {})
        agent = get_column(ref, "agent") if ref else ""
        agent_rows = chercher_dans_bsv_resolved(
            bulletin_rows,
            prediction.classe,
            agent,
        )
        niveau_risque = extraire_niveau_risque(agent_rows) if agent_rows else None
        meteo_bool, meteo_status = extraire_meteo_favorable(
            bulletin.get("prevision_meteo", "") if bulletin else "",
            prediction.classe,
            bulletin.get("meteo_extraction_status", "lisible") if bulletin else "absente",
        )
        agent_couvert = (
            prediction.classe in data.covered_classes
            or bool(agent and normalize(agent) in data.covered_agents)
        )

        compatibility, reason = appliquer_regle_decision(
            bulletin_present=bulletin is not None,
            dans_fenetre_saisonniere=in_window,
            agent_couvert_corpus=agent_couvert,
            agent_rows_bulletin=agent_rows,
            direct_tomato_rows_bulletin=direct_tomato_rows,
            tomato_compatible_rows_bulletin=tomato_compatible_rows,
            niveau_risque=niveau_risque,
            meteo_status=meteo_status,
        )

        statuses = sorted(
            {row.get("resolution_status", "") for row in agent_rows if row.get("resolution_status", "")}
        )
        output_rows.append(
            {
                "image_id": query.image_id,
                "date_observation": query.date_observation.isoformat(),
                "region": query.region,
                "classe_cnn": prediction.classe,
                "score_cnn": prediction.score,
                "agent_referentiel": agent,
                "bulletin_id_retenu": bulletin.get("bulletin_id", "") if bulletin else "",
                "date_bulletin_retenu": (
                    bulletin["date_bulletin_parsed"].isoformat() if bulletin else ""
                ),
                "fenetre_saisonniere_debut": window[0].isoformat() if window else "",
                "fenetre_saisonniere_fin": window[1].isoformat() if window else "",
                "agent_present_bulletin": bool(agent_rows),
                "resolution_status_bulletin": ";".join(statuses),
                "niveau_risque": niveau_risque or "",
                "meteo_status": meteo_status,
                "meteo_favorable": "" if meteo_bool is None else meteo_bool,
                "compatibilite": compatibility,
                "decision_reason": reason,
            }
        )

    return output_rows


def main() -> int:
    args = parse_args()
    try:
        data = charger_donnees(
            args.bulletins_file.resolve(),
            args.observations_file.resolve(),
            args.referential_file.resolve(),
        )
        queries = load_queries(args.query_file.resolve())
        rows = [
            row
            for query in queries
            for row in evaluer_requete(query, data)
        ]
        write_csv(args.output_file.resolve(), rows)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Summary: {len(queries)} image query(ies), "
        f"{len(rows)} Top-K compatibility row(s)."
    )
    print(f"CSV: {args.output_file.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
