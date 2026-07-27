"""Genere les syntheses et periodes des bioagresseurs tomate depuis les BSV.

Entree:
    data/processed/synthese_tomate.csv

Sorties par defaut:
    data/processed/synthese_bioagresseurs_tomate_par_region.csv
    data/processed/periodes_bioagresseurs_tomate_par_region.csv
    data/processed/periodes_bioagresseurs_tomate_par_region_annee.csv

Les periodes sont calculees a partir des dates de bulletin. Le fichier annuel
regroupe par region, annee, type et bioagresseur afin d'eviter qu'une periode
ne couvre artificiellement plusieurs annees.

python src/extract_tomato_bioagressor_periods.py
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fichiers utilises par defaut quand le script est lance sans argument.
DEFAULT_INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "moteur" / "synthese_tomate.csv"
)
DEFAULT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "moteur"
    / "synthese_bioagresseurs_tomate_par_region.csv"
)
DEFAULT_PERIODS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "moteur"
    / "periodes_bioagresseurs_tomate_par_region.csv"
)
DEFAULT_ANNUAL_PERIODS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "moteur"
    / "periodes_bioagresseurs_tomate_par_region_annee.csv"
)

SUMMARY_COLUMNS = ["Région", "Maladies", "Ravageurs", "Carences / désordres"]
PERIOD_COLUMNS = [
    "region",
    "type",
    "bioagresseur",
    "premiere_date_bulletin",
    "derniere_date_bulletin",
    "periode_large",
    "mois_observes",
    "semaines_observees",
    "annees_observees",
    "nb_bulletins",
    "bulletins",
]
ANNUAL_PERIOD_COLUMNS = [
    "region",
    "annee",
    "type",
    "bioagresseur",
    "premiere_date_bulletin",
    "derniere_date_bulletin",
    "periode_large",
    "mois_observes",
    "semaines_observees",
    "nb_bulletins",
    "bulletins",
]

FRENCH_MONTHS = [
    "",
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]

VALID_TYPES = {"maladie", "ravageur", "carence"}


@dataclass(frozen=True)
class Event:
    """Mention elementaire d'un bioagresseur dans un bulletin BSV."""

    region: str
    annee: int
    type: str
    bioagresseur: str
    date_bulletin: date
    bulletin_id: str


TEXT_DETECTIONS_GRAND_EST = [
    ("Mildiou", "maladie", r"mildiou"),
    ("Botrytis", "maladie", r"botrytis"),
    ("Cladosporiose", "maladie", r"cladosporiose"),
    ("Oïdium", "maladie", r"oidium"),
    ("Fusariose", "maladie", r"fusariose"),
    ("Verticilliose", "maladie", r"verticilliose"),
    ("Nécrose apicale des fruits", "carence", r"necrose apicale des fruits"),
    ("Pucerons", "ravageur", r"pucerons?"),
    ("Thrips", "ravageur", r"thrips"),
    ("Acariens", "ravageur", r"acariens?|tetranyques?"),
    ("Aleurodes", "ravageur", r"aleurodes?"),
    ("Cochenilles", "ravageur", r"cochenilles?"),
    ("Drosophile", "ravageur", r"drosophile"),
    ("Altises", "ravageur", r"altises?"),
    ("Fourmis", "ravageur", r"fourmis"),
]


def strip_accents(value: str) -> str:
    """Retire les accents pour faciliter les comparaisons textuelles."""

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def norm_text(value: str | None) -> str:
    """Normalise un texte pour les recherches: minuscules, sans accents."""

    value = strip_accents(value or "").lower()
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(name: str | None) -> str | None:
    """Regroupe les variantes d'un meme bioagresseur sous un libelle unique."""

    if not name or not name.strip():
        return None

    raw = re.sub(r"\s+", " ", name.strip())
    raw_ascii = norm_text(raw)

    # Correspondances exactes les plus frequentes dans les extractions BSV.
    exact_mapping = {
        "oidium": "Oïdium",
        "botrytis": "Botrytis",
        "septoriose": "Septoriose",
        "sclerotinia": "Sclérotiniose",
        "sclerotiniose": "Sclérotiniose",
        "virus tofbv (blotch)": "Virus ToFBV (Blotch)",
        "virus tswv": "TSWV",
        "puceron": "Pucerons",
        "pucerons": "Pucerons",
        "aleurode": "Aleurodes",
        "aleurodes": "Aleurodes",
        "acariens tetranyques": "Acariens",
        "acariens": "Acariens",
        "acariose bronzee": "Acariose bronzée",
        "noctuelle": "Noctuelles",
        "noctuelle defoliatrice": "Noctuelles",
        "noctuelles defoliatrices": "Noctuelles",
        "noctuelles sur feuilles/fruits": "Noctuelles",
        "mineuse": "Mineuses",
        "mineuses": "Mineuses",
        "cochenille": "Cochenilles",
        "cochenilles": "Cochenilles",
        "altises": "Altises",
        "thrips": "Thrips",
        "necrose apicale des fruits": "Nécrose apicale des fruits",
    }
    if raw_ascii in exact_mapping:
        return exact_mapping[raw_ascii]

    # Les punaises sont nommees de plusieurs facons selon les regions/annees.
    if re.fullmatch(
        r"(punaise \(nezara viridula\)|punaise nezara viridula|"
        r"nezara viridula \(punaise\)|nezara virudula|punaises nezara|"
        r"punaises nezara viridula|punaise nezara|punaises nezara)",
        raw_ascii,
    ):
        return "Punaises Nezara"

    # Nesidiocoris peut etre note seul, comme punaise, ou avec son ancien nom.
    if re.fullmatch(
        r"(punaise nesidiocoris|punaises nesidiocoris|"
        r"punaises nesidiocoris tenuis|nesidiocoris tenuis.*)",
        raw_ascii,
    ):
        return "Punaises Nesidiocoris"

    if raw_ascii == "thrips et acarien":
        return "Thrips; Acariens"

    return raw


def split_bioagressors(name: str | None) -> list[str]:
    """Separe les libelles composes, par exemple 'Mildiou et Oidium'."""

    normalized = normalize_name(name)
    if not normalized:
        return []

    results: list[str] = []
    for part in re.split(r"\s+et\s+|;|,", normalized, flags=re.IGNORECASE):
        item = normalize_name(part)
        if not item:
            continue
        # Ces libelles ne sont pas des bioagresseurs exploitables.
        if norm_text(item) in {
            "autres ravageurs",
            "stade phenologique",
            "adventices",
        }:
            continue
        if item not in results:
            results.append(item)
    return results


def is_tomato_related(row: dict[str, str]) -> bool:
    """Garde les lignes qui concernent explicitement la tomate/solanacees."""

    text = " ".join(
        [
            row.get("culture", ""),
            row.get("bioagresseur", ""),
            row.get("observation", ""),
            row.get("analyse_risque", ""),
        ]
    )
    text_norm = norm_text(text)
    return "tomate" in text_norm or "solanace" in text_norm


def near_tomato(text: str, pattern: str, max_chars: int = 90) -> bool:
    """Verifie qu'un terme est proche du mot tomate dans le paragraphe."""

    text_norm = norm_text(text)
    regex = rf"({pattern}).{{0,{max_chars}}}tomate|tomate.{{0,{max_chars}}}({pattern})"
    return re.search(regex, text_norm, flags=re.IGNORECASE) is not None


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def collect_events(rows: list[dict[str, str]]) -> list[Event]:
    """Transforme les lignes de synthese en mentions datees de bioagresseurs."""

    events: list[Event] = []

    for row in rows:
        if not row.get("date_bulletin") or not is_tomato_related(row):
            continue

        bulletin_date = parse_date(row["date_bulletin"])
        row_type = norm_text(row.get("type", ""))

        # Cas standard: le bioagresseur est deja structure dans la colonne dediee.
        if row_type in VALID_TYPES:
            for bioagresseur in split_bioagressors(row.get("bioagresseur")):
                event_type = row_type
                if norm_text(bioagresseur) == "acariose bronzee":
                    # L'acariose bronzee est un degat d'acariens: on la range
                    # avec les ravageurs meme si elle est parfois typee maladie.
                    event_type = "ravageur"
                events.append(
                    Event(
                        region=row["region"],
                        annee=bulletin_date.year,
                        type=event_type,
                        bioagresseur=bioagresseur,
                        date_bulletin=bulletin_date,
                        bulletin_id=row["bulletin_id"],
                    )
                )

        # Grand Est range parfois les maladies tomate dans des paragraphes
        # "solanacees" dont le libelle structure est un groupe de ravageurs.
        # On ajoute donc une detection texte prudente, limitee aux termes
        # situes pres du mot "tomate" pour eviter les mentions concombre/aubergine.
        if row.get("region") == "Grand Est":
            text = " ".join(
                [
                    row.get("culture", ""),
                    row.get("bioagresseur", ""),
                    row.get("observation", ""),
                    row.get("analyse_risque", ""),
                ]
            )
            for bioagresseur, event_type, pattern in TEXT_DETECTIONS_GRAND_EST:
                if near_tomato(text, pattern):
                    events.append(
                        Event(
                            region=row["region"],
                            annee=bulletin_date.year,
                            type=event_type,
                            bioagresseur=bioagresseur,
                            date_bulletin=bulletin_date,
                            bulletin_id=row["bulletin_id"],
                        )
                    )

    return sorted(
        set(events),
        key=lambda e: (
            e.region,
            e.annee,
            e.type,
            e.bioagresseur,
            e.date_bulletin,
            e.bulletin_id,
        ),
    )


def format_period(first: date, last: date, include_year_start: bool = False) -> str:
    """Formate la periode large a partir des dates extremes observees."""

    if first == last:
        return first.strftime("%d/%m/%Y")
    if include_year_start:
        return (
            f"{FRENCH_MONTHS[first.month]} {first.year} "
            f"a {FRENCH_MONTHS[last.month]} {last.year}"
        )
    return f"{FRENCH_MONTHS[first.month]} a {FRENCH_MONTHS[last.month]} {last.year}"


def build_period_rows(
    events: list[Event],
    by_year: bool,
) -> list[dict[str, str | int]]:
    """Agrege les mentions en periodes globales ou en periodes par annee."""

    grouped: dict[tuple[object, ...], list[Event]] = defaultdict(list)
    for event in events:
        if by_year:
            key = (event.region, event.annee, event.type, event.bioagresseur)
        else:
            key = (event.region, event.type, event.bioagresseur)
        grouped[key].append(event)

    output_rows: list[dict[str, str | int]] = []
    for key, group in grouped.items():
        group = sorted(group, key=lambda event: event.date_bulletin)

        # Les bornes de periode sont les dates du premier et du dernier bulletin
        # ou le bioagresseur est mentionne dans le groupe considere.
        first = group[0].date_bulletin
        last = group[-1].date_bulletin
        months = sorted(
            {(event.date_bulletin.year, event.date_bulletin.month) for event in group}
        )
        weeks = sorted(
            {
                f"{event.date_bulletin.year}-S{event.date_bulletin.isocalendar().week:02d}"
                for event in group
            }
        )
        years = sorted({event.date_bulletin.year for event in group})
        bulletins = sorted({event.bulletin_id for event in group})

        if by_year:
            region, annee, event_type, bioagresseur = key
        else:
            region, event_type, bioagresseur = key
            annee = None

        row: dict[str, str | int] = {
            "region": str(region),
            "type": str(event_type),
            "bioagresseur": str(bioagresseur),
            "premiere_date_bulletin": first.isoformat(),
            "derniere_date_bulletin": last.isoformat(),
            "periode_large": format_period(first, last, include_year_start=not by_year),
            "mois_observes": "; ".join(
                f"{FRENCH_MONTHS[month]} {year}" for year, month in months
            ),
            "semaines_observees": "; ".join(weeks),
            "nb_bulletins": len(bulletins),
            "bulletins": "; ".join(bulletins),
        }
        if by_year:
            row["annee"] = int(annee)  # type: ignore[arg-type]
        else:
            row["annees_observees"] = "; ".join(str(year) for year in years)
        output_rows.append(row)

    if by_year:
        return sorted(
            output_rows,
            key=lambda row: (
                str(row["region"]),
                int(row["annee"]),
                str(row["type"]),
                str(row["bioagresseur"]),
            ),
        )
    return sorted(
        output_rows,
        key=lambda row: (
            str(row["region"]),
            str(row["type"]),
            str(row["bioagresseur"]),
        ),
    )


def build_summary_rows(events: list[Event]) -> list[dict[str, str]]:
    """Construit le tableau simple: Region, Maladies, Ravageurs, Carences."""

    grouped: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"maladie": set(), "ravageur": set(), "carence": set()}
    )
    for event in events:
        grouped[event.region][event.type].add(event.bioagresseur)

    rows: list[dict[str, str]] = []
    for region in sorted(grouped):
        rows.append(
            {
                "Région": region,
                "Maladies": "; ".join(sorted(grouped[region]["maladie"])),
                "Ravageurs": "; ".join(sorted(grouped[region]["ravageur"])),
                "Carences / désordres": "; ".join(sorted(grouped[region]["carence"])),
            }
        )
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    """Lit un CSV UTF-8 avec BOM eventuel, pratique pour les fichiers Excel."""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str | int]], columns: list[str]) -> None:
    """Ecrit en UTF-8 avec BOM pour une ouverture correcte dans Excel Windows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tomato bioagressor summary, global periods and annual "
            "periods from BSV synthesis data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input tomato synthesis CSV (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_FILE,
        help=f"Summary CSV output (default: {DEFAULT_SUMMARY_FILE})",
    )
    parser.add_argument(
        "--periods-output",
        type=Path,
        default=DEFAULT_PERIODS_FILE,
        help=f"Global period CSV output (default: {DEFAULT_PERIODS_FILE})",
    )
    parser.add_argument(
        "--annual-output",
        type=Path,
        default=DEFAULT_ANNUAL_PERIODS_FILE,
        help=f"Annual period CSV output (default: {DEFAULT_ANNUAL_PERIODS_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    events = collect_events(rows)

    summary_rows = build_summary_rows(events)
    period_rows = build_period_rows(events, by_year=False)
    annual_rows = build_period_rows(events, by_year=True)

    write_csv(args.summary_output, summary_rows, SUMMARY_COLUMNS)
    write_csv(args.periods_output, period_rows, PERIOD_COLUMNS)
    write_csv(args.annual_output, annual_rows, ANNUAL_PERIOD_COLUMNS)

    print(f"Summary file: {args.summary_output} ({len(summary_rows)} rows)")
    print(f"Global periods file: {args.periods_output} ({len(period_rows)} rows)")
    print(f"Annual periods file: {args.annual_output} ({len(annual_rows)} rows)")


if __name__ == "__main__":
    main()
