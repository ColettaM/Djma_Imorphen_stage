"""
scrape_bsv.py
=============
Téléchargement automatisé des Bulletins de Santé du Végétal (BSV Maraîchage)
depuis les Chambres d'Agriculture Pays de la Loire, Grand Est et la DRAAF PACA.

Usage
-----
    python scrape_bsv.py                        # toutes sources, 2023-2025
    python scrape_bsv.py --annees 2025          # une seule année
    python scrape_bsv.py --source pdl           # source unique
    python scrape_bsv.py --source paca          # source DRAAF PACA
    python scrape_bsv.py --filiere maraichage   # filière unique (PDL)
    python scrape_bsv.py --dry-run              # liste les URLs sans télécharger

"""

import argparse
import logging
import time
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.retry import Retry

# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "moteur" / "bsv_pdf"
DEFAULT_LOG_FILE = PROJECT_ROOT / "outputs" / "moteur" / "logs" / "scrape_bsv.log"
LOGGER = logging.getLogger("scrape_bsv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

PAUSE_BETWEEN_DOWNLOADS = 1.5  # secondes entre chaque téléchargement

# SOURCES — structure déclarative
# Chaque source est un dict :
#   source_id   : identifiant court
#   region      : nom lisible
#   filieres    : liste de filières, chacune avec :
#       filiere_id  : identifiant court (pour l'option --filiere)
#       nom         : nom lisible
#       url_pattern : URL de la page d'archives avec {annee} à remplacer
#       year_index_url : page listant les archives par année
#       base_url    : domaine pour construire les URLs relatives

SOURCES = [
    {
        "source_id": "pdl",
        "region": "Pays de la Loire",
        "base_url": "https://pays-de-la-loire.chambres-agriculture.fr",
        "filieres": [
            {
                "filiere_id": "maraichage",
                "nom": "Maraîchage",
                "url_pattern": (
                    "https://pays-de-la-loire.chambres-agriculture.fr"
                    "/sinformer/ressources-et-documentation"
                    "/bulletins-techniques-et-de-preconisation"
                    "/bulletins-vegetal/bsvpdl"
                    "/bsv-maraichage-archives-{annee}"
                ),
            },
        ],
    },
    {
        "source_id": "grandest",
        "region": "Grand Est",
        "base_url": "https://grandest.chambres-agriculture.fr",
        # Archives annuelles BSV légumes Grand Est.
        # Format observé : bsv-legumes-grand-est-archives-{annee}
        "filieres": [
            {
                "filiere_id": "legumes_maraichage",
                "nom": "Légumes - Maraîchage",
                "url_pattern": (
                    "https://grandest.chambres-agriculture.fr"
                    "/agritheque/ressources-documentation"
                    "/bulletins-techniques"
                    "/bulletin-de-sante-du-vegetal"
                    "/bsv-legumes-grand-est-archives-{annee}"
                ),
            },
        ],
    },
    {
        "source_id": "paca",
        "region": "Provence-Alpes-Côte d'Azur",
        "base_url": "https://draaf.paca.agriculture.gouv.fr",
        "verify_tls": False,
        "filieres": [
            {
                "filiere_id": "maraichage",
                "nom": "Cultures Maraîchères",
                "year_index_url": (
                    "https://draaf.paca.agriculture.gouv.fr/"
                    "cultures-maraicheres-r273.html"
                ),
                "link_strategy": "paca_article_pages",
            },
        ],
    },
]

ALL_FILIERE_IDS = sorted(
    {filiere["filiere_id"] for source in SOURCES for filiere in source["filieres"]}
)


def configure_logging(log_file: Path) -> None:
    """Configure un journal UTF-8 horodaté pour une exécution du scraper."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    for existing_handler in list(LOGGER.handlers):
        existing_handler.close()
        LOGGER.removeHandler(existing_handler)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(handler)


def create_session() -> requests.Session:
    """Crée une session HTTP avec retries et backoff pour les erreurs temporaires."""

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def is_valid_pdf_file(path: Path) -> bool:
    """Vérifie qu'un fichier existe, n'est pas vide et contient une signature PDF."""

    if not path.is_file() or path.stat().st_size < 5:
        return False

    with path.open("rb") as handle:
        return b"%PDF-" in handle.read(1024)


# COLLECTE DES LIENS PDF


def fetch_page(url: str, session: requests.Session) -> BeautifulSoup | None:
    """Récupère une page HTML et retourne un objet BeautifulSoup."""
    LOGGER.info("FETCH_PAGE | url=%s", url)
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 404:
            print(f"    [404] page inexistante : {url}")
            LOGGER.warning("PAGE_404 | url=%s", url)
            return None
        resp.raise_for_status()
        LOGGER.info("PAGE_OK | url=%s | status=%s", url, resp.status_code)
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as e:
        print(f"    [ERREUR] impossible de récupérer {url} : {e}")
        LOGGER.error("PAGE_ERROR | url=%s | error=%s", url, e)
        return None


def extract_pdf_links(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """
    Extrait tous les liens vers des fichiers .pdf dans la page.
    Retourne une liste de (nom_fichier, url_absolue).
    """
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        abs_url = urljoin(base_url, href)
        parsed_url = urlparse(abs_url)

        if not parsed_url.path.lower().endswith(".pdf"):
            continue

        filename = Path(unquote(parsed_url.path)).name
        if filename:
            links.append((filename, abs_url))
    # Dédoublonnage en conservant l'ordre
    seen = set()
    unique = []
    for item in links:
        if item[1] not in seen:
            seen.add(item[1])
            unique.append(item)
    return unique


def deduplicate_links(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Déduplique une liste de liens en conservant l'ordre."""

    seen = set()
    unique = []
    for item in links:
        if item[1] not in seen:
            seen.add(item[1])
            unique.append(item)
    return unique


def extract_year_urls(soup: BeautifulSoup, base_url: str) -> dict[int, str]:
    """Extrait les URLs d'archives annuelles d'une page index."""

    year_urls = {}
    for tag in soup.find_all("a", href=True):
        text = tag.get_text(" ", strip=True)
        for year in range(2000, 2100):
            if str(year) in text:
                year_urls[year] = urljoin(base_url, tag["href"])
                break
    return year_urls


def resolve_archive_url(
    source: dict,
    filiere: dict,
    annee: int,
    session: requests.Session,
) -> str | None:
    """Résout l'URL d'archive annuelle selon la configuration de la filière."""

    if "url_pattern" in filiere:
        return filiere["url_pattern"].format(annee=annee)

    if "year_index_url" not in filiere:
        LOGGER.error(
            "ARCHIVE_URL_UNCONFIGURED | source=%s | filiere=%s | annee=%s",
            source["source_id"],
            filiere["filiere_id"],
            annee,
        )
        return None

    cache = filiere.setdefault("_resolved_year_urls", {})
    if not cache:
        soup = fetch_page(filiere["year_index_url"], session)
        if soup is None:
            return None
        cache.update(extract_year_urls(soup, source["base_url"]))

    url = cache.get(annee)
    if url is None:
        print(f"    [ABSENT] aucune page d'archive trouvée pour {annee}")
        LOGGER.warning(
            "ARCHIVE_URL_NOT_FOUND | source=%s | filiere=%s | annee=%s | index=%s",
            source["source_id"],
            filiere["filiere_id"],
            annee,
            filiere["year_index_url"],
        )
    return url


def extract_paca_article_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extrait les pages d'articles PACA portant les PDF des bulletins."""

    links = []
    for tag in soup.find_all("a", href=True):
        text = tag.get_text(" ", strip=True).casefold()
        if "bulletin" not in text and "bsv" not in text:
            continue

        abs_url = urljoin(base_url, tag["href"])
        path = urlparse(abs_url).path.lower()
        if path.endswith(".html") and "-a" in Path(path).stem:
            links.append(abs_url)
    return list(dict.fromkeys(links))


def extract_pagination_links(
    soup: BeautifulSoup,
    base_url: str,
    current_url: str,
) -> list[str]:
    """Extrait les liens de pagination d'une page d'archives."""

    current = urlparse(current_url)
    links = []
    for tag in soup.find_all("a", href=True):
        abs_url = urljoin(base_url, tag["href"])
        parsed = urlparse(abs_url)
        if (
            parsed.scheme == current.scheme
            and parsed.netloc == current.netloc
            and parsed.path == current.path
            and "debut_listearticles=" in parsed.query
        ):
            links.append(abs_url)
    return list(dict.fromkeys(links))


def collect_pdf_links(
    source: dict,
    filiere: dict,
    archive_url: str,
    archive_soup: BeautifulSoup,
    session: requests.Session,
) -> list[tuple[str, str]]:
    """Collecte les PDF d'une page d'archive, avec stratégie spécifique si besoin."""

    if filiere.get("link_strategy") != "paca_article_pages":
        return extract_pdf_links(archive_soup, source["base_url"])

    pdf_links = []
    article_urls = []
    pages_to_visit = [archive_url]
    visited_pages = set()
    soups_by_url = {archive_url: archive_soup}

    while pages_to_visit:
        page_url = pages_to_visit.pop(0)
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)

        soup = soups_by_url.get(page_url)
        if soup is None:
            soup = fetch_page(page_url, session)
            if soup is None:
                continue

        pdf_links.extend(extract_pdf_links(soup, source["base_url"]))
        article_urls.extend(extract_paca_article_links(soup, source["base_url"]))

        for next_url in extract_pagination_links(soup, source["base_url"], page_url):
            if next_url not in visited_pages and next_url not in pages_to_visit:
                pages_to_visit.append(next_url)

    for article_url in dict.fromkeys(article_urls):
        article_soup = fetch_page(article_url, session)
        if article_soup is None:
            continue
        pdf_links.extend(extract_pdf_links(article_soup, source["base_url"]))

    return deduplicate_links(pdf_links)


# TÉLÉCHARGEMENT


def download_pdf(
    url: str,
    dest_path: Path,
    session: requests.Session,
    dry_run: bool = False,
) -> str:
    """
    Télécharge un PDF vers dest_path.
    Retourne l'un des statuts : downloaded, skipped, dry_run, error.
    """
    if is_valid_pdf_file(dest_path):
        print(f"    [DÉJÀ PRÉSENT] {dest_path.name}")
        LOGGER.info("PDF_SKIPPED | url=%s | path=%s", url, dest_path)
        return "skipped"

    if dest_path.exists():
        print(f"    [INVALIDE] nouveau téléchargement : {dest_path.name}")
        LOGGER.warning("PDF_INVALID_EXISTING | url=%s | path=%s", url, dest_path)

    if dry_run:
        print(f"    [DRY-RUN] {url}")
        LOGGER.info("PDF_DRY_RUN | url=%s | path=%s", url, dest_path)
        return "dry_run"

    temporary_path = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.unlink(missing_ok=True)

        with session.get(url, timeout=(10, 60), stream=True) as resp:
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/json" in content_type:
                raise ValueError(f"réponse non PDF ({content_type})")

            with temporary_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        handle.write(chunk)

        if not is_valid_pdf_file(temporary_path):
            raise ValueError("signature PDF absente ou fichier vide")

        temporary_path.replace(dest_path)
        print(f"    [OK] {dest_path.name}")
        LOGGER.info(
            "PDF_DOWNLOADED | url=%s | path=%s | bytes=%s",
            url,
            dest_path,
            dest_path.stat().st_size,
        )
        return "downloaded"
    except (requests.RequestException, OSError, ValueError) as e:
        temporary_path.unlink(missing_ok=True)
        print(f"    [ERREUR TÉLÉCHARGEMENT] {dest_path.name} : {e}")
        LOGGER.error("PDF_ERROR | url=%s | path=%s | error=%s", url, dest_path, e)
        return "error"


# PIPELINE PRINCIPAL


def run(
    annees: list[int],
    source_ids: list[str],
    filiere_ids: list[str] | None,
    dry_run: bool,
    output_root: Path,
) -> None:
    LOGGER.info(
        "RUN_START | annees=%s | sources=%s | filieres=%s | dry_run=%s | output=%s",
        annees,
        source_ids,
        filiere_ids or "toutes",
        dry_run,
        output_root,
    )

    total_found = 0
    total_downloaded = 0
    total_skipped = 0
    total_dry_run = 0
    total_errors = 0

    with create_session() as session:
        for source in SOURCES:
            if source["source_id"] not in source_ids:
                continue

            session.verify = source.get("verify_tls", True)
            if not session.verify:
                requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

            print(f"\n{'='*60}")
            print(f"SOURCE : {source['region']}  ({source['source_id']})")
            print(f"{'='*60}")

            for filiere in source["filieres"]:
                if filiere_ids and filiere["filiere_id"] not in filiere_ids:
                    continue

                print(f"\n  Filière : {filiere['nom']}")

                for annee in annees:
                    url = resolve_archive_url(source, filiere, annee, session)
                    if url is None:
                        continue

                    print(f"\n  Année {annee} -> {url}")

                    soup = fetch_page(url, session)
                    if soup is None:
                        continue

                    pdf_links = collect_pdf_links(source, filiere, url, soup, session)
                    if not pdf_links:
                        print("    Aucun PDF trouvé sur cette page.")
                        LOGGER.warning(
                            "NO_PDF_FOUND | source=%s | filiere=%s | annee=%s | url=%s",
                            source["source_id"],
                            filiere["filiere_id"],
                            annee,
                            url,
                        )
                        continue

                    print(f"    {len(pdf_links)} PDF(s) détecté(s)")
                    LOGGER.info(
                        "PDF_LINKS_FOUND | source=%s | filiere=%s | annee=%s | count=%s",
                        source["source_id"],
                        filiere["filiere_id"],
                        annee,
                        len(pdf_links),
                    )
                    total_found += len(pdf_links)

                    for filename, pdf_url in pdf_links:
                        dest = (
                            output_root
                            / source["source_id"]
                            / filiere["filiere_id"]
                            / str(annee)
                            / filename
                        )

                        status = download_pdf(
                            pdf_url,
                            dest,
                            session,
                            dry_run=dry_run,
                        )
                        if status == "downloaded":
                            total_downloaded += 1
                        elif status == "skipped":
                            total_skipped += 1
                        elif status == "dry_run":
                            total_dry_run += 1
                        else:
                            total_errors += 1

                        if status in {"downloaded", "error"}:
                            time.sleep(PAUSE_BETWEEN_DOWNLOADS)

    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"  PDFs détectés    : {total_found}")
    print(f"  Téléchargés      : {total_downloaded}")
    print(f"  Déjà présents    : {total_skipped}")
    print(f"  Dry-run          : {total_dry_run}")
    print(f"  Erreurs          : {total_errors}")
    if dry_run:
        print("  (dry-run : aucun fichier écrit)")

    LOGGER.info(
        "RUN_END | found=%s | downloaded=%s | skipped=%s | dry_run=%s | errors=%s",
        total_found,
        total_downloaded,
        total_skipped,
        total_dry_run,
        total_errors,
    )


# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge les BSV Maraîchage depuis les Chambres d'Agriculture."
    )
    parser.add_argument(
        "--annees",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        metavar="ANNEE",
        help="Années à collecter (défaut : 2023 2024 2025)",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        dest="source_ids",
        default=[s["source_id"] for s in SOURCES],
        choices=[s["source_id"] for s in SOURCES],
        help="Source(s) à scraper (défaut : toutes)",
    )
    parser.add_argument(
        "--filiere",
        nargs="+",
        dest="filiere_ids",
        default=None,
        choices=ALL_FILIERE_IDS,
        help=(
            "Filière(s) à scraper, ex : maraichage legumes_maraichage. "
            "Par défaut toutes les filières de chaque source."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Répertoire de sortie (défaut : {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Fichier journal (défaut : {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Liste les URLs sans télécharger.",
    )
    args = parser.parse_args()

    if args.filiere_ids:
        available_for_sources = {
            filiere["filiere_id"]
            for source in SOURCES
            if source["source_id"] in args.source_ids
            for filiere in source["filieres"]
        }
        incompatible = set(args.filiere_ids) - available_for_sources
        if incompatible:
            parser.error(
                "filière(s) incompatible(s) avec les sources sélectionnées : "
                + ", ".join(sorted(incompatible))
            )

    return args


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_file.resolve())

    print("BSV Scraper — Chambres d'Agriculture")
    print(f"  Années  : {args.annees}")
    print(f"  Sources : {args.source_ids}")
    print(f"  Filières: {args.filiere_ids or 'toutes'}")
    print(f"  Sortie  : {args.output.resolve()}")
    print(f"  Log     : {args.log_file.resolve()}")
    if args.dry_run:
        print("  MODE    : dry-run (pas de téléchargement)")

    run(
        annees=args.annees,
        source_ids=args.source_ids,
        filiere_ids=args.filiere_ids,
        dry_run=args.dry_run,
        output_root=args.output,
    )
