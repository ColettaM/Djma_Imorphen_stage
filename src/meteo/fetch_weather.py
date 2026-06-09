"""
fetch_weather.py
----------------
Étape 3a : récupération des séries météo historiques.

Utilise l'API Open-Meteo Historical Weather (gratuite, sans clé) pour
récupérer les 15 jours précédant chaque date EXIF.

Architecture HTTP :
  - Session avec retry automatique sur 429 / 5xx (backoff exponentiel)
  - Gestion explicite du Retry-After
  - Timeouts (connect, read) séparés
  - Erreurs permanentes (4xx hors 429) ne déclenchent pas de retry

Cache local en parquet par (lat, lon, date) pour ne pas re-télécharger.

Variables demandées :
  - tmean / tmax / tmin / apparent_temperature_mean
  - precipitation_sum / rain_sum
  - shortwave_radiation_sum / et0_fao_evapotranspiration
  - relative_humidity_2m_mean / min / max
  - dew_point_2m_mean
  - vapour_pressure_deficit_max  (clé pour cladosporiose et oïdium)
  - wind_speed_10m_max / wind_gusts_10m_max
  - soil_moisture_0_to_7cm_mean  (sécheresse de surface)

Produit : outputs/weather_raw/{lat_lon_date}.parquet
          outputs/weather_fetch_log.csv

Utilisation :
    python fetch_weather.py
    python fetch_weather.py --limit 50            # mode test
    python fetch_weather.py --ready-mode strict   # outdoor uniquement
    python fetch_weather.py --granularity image   # une requetepar image
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---- Configuration ----------------------------------------------------------
# Définition des chemins d'entrée/sortie utilisés par le script.
# Les chemins sont relatifs au dossier où se trouve fetch_weather.py.


# Table bulletin-centric produite par aggregate_to_bulletins.py.
# C'est la source recommandée : une ligne = un bulletin utilisateur.
PHASE2_BULLETINS = Path(__file__).parent / "outputs" / "phase2_bulletins.csv"

# Table image-centric produite par build_phase2_context.py.
# Utilisée seulement si --granularity image.
PHASE2_CTX = Path(__file__).parent / "outputs" / "phase2_context.csv"

# Dossier de cache local.
# Chaque requête météo réussie est stockée en parquet pour éviter
# de rappeler l'API inutilement.
CACHE_DIR = Path(__file__).parent / "outputs" / "weather_raw"

PARQUET_DIR = CACHE_DIR / "parquet"
JSON_DIR = CACHE_DIR / "json"

# Fichier journal listant, pour chaque observation, si la météo était
# déjà en cache, téléchargée, ou en échec.
LOG_CSV = Path(__file__).parent / "outputs" / "weather_fetch_log.csv"

# Endpoint de l'API historique Open-Meteo.
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Fuseau horaire utilisé pour aligner les dates météo sur la France.
TIMEZONE = "Europe/Paris"

# Timeouts HTTP séparés :
# - 20 secondes pour établir la connexion
# - 60 secondes pour lire la réponse
REQUEST_TIMEOUT = (20, 60)

# Pause volontaire entre deux appels réussis pour éviter d'envoyer
# trop de requêtes trop rapidement.
SLEEP_BETWEEN_CALLS_SEC = 0.15

# Variables journalières demandées à Open-Meteo.
# Elles couvrent température, pluie, humidité, vent, rayonnement,
# évapotranspiration, VPD et humidité du sol.
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "dew_point_2m_mean",
    "vapour_pressure_deficit_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "soil_moisture_0_to_7cm_mean",
]

# Taille de la fenêtre météo :
# pour une observation à J, on récupère J-15 jusqu'à J-1.
WINDOW_DAYS = 15  # J-15 → J-1
 

# ---- Session HTTP robuste ---------------------------------------------------

def build_session() -> requests.Session:
    """Session avec retry automatique sur 429 et 5xx."""

    # Création d'une session persistante. 
    # Cela permet de réutiliser les connexions HTTP entre plusieurs appels.
    session = requests.Session()

    # Configuration des retries automatiques gérés par urllib3.
    # Les erreurs temporaires ciblées sont :
    # - 429 : trop de requêtes
    # - 5xx : erreur serveur temporaire
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    # Adaptateur HTTP avec pool de connexions et stratégie de retry.
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)

    # Application de l'adaptateur aux appels HTTPS et HTTP.
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# Session globale utilisée par toutes les requêtes.
SESSION = build_session()


# ---- Helpers ----------------------------------------------------------------

def cache_key(lat: float, lon: float, date) -> str:
    """Clé de cache stable. lat/lon arrondis à 4 décimales (~10 m)."""

    # La clé identifie une requête météo par :
    # - latitude arrondie
    # - longitude arrondie
    # - date d'observation
    #
    # Exemple :
    # 47.1727_-1.5528_2026-04-18
    return f"{round(lat, 4)}_{round(lon, 4)}_{date.date().isoformat()}"


def fetch_one(lat: float, lon: float, end_date) -> pd.DataFrame | None:
    """
    Interroge l'API pour la fenêtre [end_date - WINDOW, end_date - 1].

    Stratégie de retry à deux niveaux :
      1. urllib3.Retry (transparent, géré par la session) sur 429/5xx
      2. Boucle explicite ici pour les Timeout / ConnectionError
    """

    # Date de début : 15 jours avant la date d'observation.
    start = (end_date - timedelta(days=WINDOW_DAYS)).date().isoformat()

    # Date de fin : veille de la date d'observation.
    # On évite J0 pour ne pas inclure la météo du jour même après
    # la prise de photo.
    end = (end_date - timedelta(days=1)).date().isoformat()

    # Paramètres envoyés à l'API Open-Meteo.
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    # Nombre maximal de tentatives explicites en plus des retries
    # déjà gérés par la session.
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            # Appel GET vers Open-Meteo avec timeouts contrôlés.
            response = SESSION.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

            # Erreurs permanentes : requête invalide, accès interdit,
            # ressource absente, etc.
            # Dans ces cas, refaire la même requête ne sert à rien.
            if response.status_code in (400, 401, 403, 404):
                print(f"  [HTTP {response.status_code}] permanent : "
                      f"lat={lat}, lon={lon}, période={start}→{end}")
                return None

            # Cas spécifique du rate limiting.
            # Même si urllib3 peut déjà gérer Retry-After, cette logique
            # ajoute un contrôle explicite au niveau applicatif.
            if response.status_code == 429:
                if attempt == max_attempts:
                    print(f"  [429] abandonné après {max_attempts} essais")
                    return None

                retry_after = response.headers.get("Retry-After")

                # Si Retry-After est fourni par le serveur, on le respecte.
                # Sinon, on applique un backoff progressif avec un peu
                # d'aléatoire pour éviter de relancer toutes les requêtes
                # exactement au même moment.
                wait = (float(retry_after) if retry_after and retry_after.isdigit()
                        else min(20 * attempt, 90) + random.uniform(1, 3))

                print(f"  [429] retry {attempt}/{max_attempts} dans {wait:.1f}s")
                time.sleep(wait)
                continue

            # Pour les autres codes d'erreur HTTP, déclenche une exception.
            response.raise_for_status()
        
            # Conversion JSON -> dict Python
            payload = response.json()

            # Chemin du fichier JSON
            json_path = JSON_DIR / f"{cache_key(lat, lon, end_date)}.json"

            # Sauvegarde
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)


            # Si la clé "daily" est absente, la réponse n'est pas exploitable.
            if "daily" not in payload:
                return None

            # Transformation des données journalières en DataFrame.
            df = pd.DataFrame(payload["daily"])

            # Vérification minimale : il faut une colonne time.
            if df.empty or "time" not in df.columns:
                return None

            # Conversion de la colonne date.
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

            # Nettoyage des lignes sans date valide, puis tri chronologique.
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

            return df

        except requests.exceptions.Timeout:
            # Timeout : l'API ou le réseau n'a pas répondu assez vite.
            # On retente quelques fois avec attente progressive.
            if attempt == max_attempts:
                print(f"  [Timeout] abandonné lat={lat}, lon={lon}")
                return None

            wait = min(8 * attempt, 30) + random.uniform(0.5, 2)
            print(f"  [Timeout] retry {attempt}/{max_attempts} dans {wait:.1f}s")
            time.sleep(wait)

        except requests.exceptions.ConnectionError as exc:
            # Erreur de connexion : réseau instable, DNS, coupure temporaire.
            if attempt == max_attempts:
                print(f"  [Connection] abandonné : {exc}")
                return None

            wait = min(6 * attempt, 25) + random.uniform(0.5, 2)
            print(f"  [Connection] retry {attempt}/{max_attempts} dans {wait:.1f}s")
            time.sleep(wait)

        except requests.exceptions.RequestException as exc:
            # Toute autre erreur requests non récupérable proprement.
            print(f"  [Request error] {exc}")
            return None

    return None


# ---- Main -------------------------------------------------------------------

def main() -> int:
    # Déclaration des arguments disponibles en ligne de commande.
    parser = argparse.ArgumentParser()

    # Permet de tester le script sur un petit nombre d'observations.
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite le nombre d'observations (test).")

    # Choix du mode d'éligibilité :
    # - strict : seulement les observations outdoor pertinentes
    # - loose : inclut aussi WINDOW pour audit/exploration
    parser.add_argument("--ready-mode", choices=["strict", "loose"], default="loose",
                        help="strict = phase2_ready (outdoor uniquement), "
                             "loose = phase2_ready_loose.")

    # Choix de la granularité :
    # - bulletin : recommandé, une requête météo par observation utilisateur
    # - image : mode historique, une requête météo par image
    parser.add_argument("--granularity", choices=["bulletin", "image"], default="bulletin",
                        help="bulletin = 1 fetch par bulletin (recommandé), "
                             "image = 1 fetch par image (legacy).")

    args = parser.parse_args()

    # Sélection du fichier source selon la granularité.
    # En mode bulletin, on lit phase2_bulletins.csv.
    # En mode image, on lit phase2_context.csv.
    if args.granularity == "bulletin":
        src = PHASE2_BULLETINS
        date_col = "datetime_observation"
        id_col = "bulletin_id"
    else:
        src = PHASE2_CTX
        date_col = "datetime_original"
        id_col = "filename"

    # Vérification de l'existence du fichier d'entrée.
    # Si le fichier manque, on indique quel script lancer avant.
    if not src.exists():
        print(f"[ERREUR] {src} manquant.", file=sys.stderr)
        if args.granularity == "bulletin":
            print("→ Lance aggregate_to_bulletins.py d'abord.", file=sys.stderr)
        else:
            print("→ Lance build_phase2_context.py d'abord.", file=sys.stderr)
        return 1

   
    # Création des dossiers de sortie
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    # Lecture du CSV source avec parsing de la colonne date.
    df = pd.read_csv(src, parse_dates=[date_col])

    # Sélection du flag d'éligibilité selon le mode demandé.
    flag_col = "phase2_ready" if args.ready_mode == "strict" else "phase2_ready_loose"

    # Conservation uniquement des observations prêtes pour la phase météo.
    #df_ready = df[df[flag_col]].copy()
    ready_mask = (
        df[flag_col]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df_ready = df[ready_mask].copy()

    # Si --limit est fourni, on limite le nombre de lignes pour un test rapide.
    if args.limit:
        df_ready = df_ready.head(args.limit)

    # Affichage des paramètres effectifs du run.
    print(f"[INFO] Granularité : {args.granularity}")
    print(f"[INFO] Source      : {src.name}")
    print(f"[INFO] Mode        : {args.ready_mode}")
    print(f"[INFO] À traiter   : {len(df_ready)}")

    # Initialisation du journal et des compteurs d'audit.
    log_records = []
    n_cached = n_fetched = n_failed = 0

    # Parcours des observations prêtes.
    for i, row in enumerate(df_ready.itertuples(), 1):
        # Récupération de la date d'observation depuis la colonne choisie.
        date_value = getattr(row, date_col)

        # Construction de la clé de cache propre à lat/lon/date.
        key = cache_key(row.latitude, row.longitude, date_value)

        # Chemin du fichier parquet correspondant à cette requête météo.
        cache_file = PARQUET_DIR / f"{key}.parquet"

        if cache_file.exists():
            # Si le fichier existe déjà, on ne rappelle pas l'API.
            n_cached += 1
            status = "cached"
        else:
            # Sinon, on interroge Open-Meteo.
            wx = fetch_one(row.latitude, row.longitude, date_value)

            if wx is None or wx.empty:
                # Échec ou réponse vide.
                n_failed += 1
                status = "failed"
            else:
                # Succès : stockage de la série météo en parquet.
                wx.to_parquet(cache_file, index=False)
                n_fetched += 1
                status = "fetched"

                # Petite pause après un téléchargement effectif.
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        # Ajout d'une ligne dans le log d'exécution.
        log_records.append({
            "id": getattr(row, id_col, None),
            "granularity": args.granularity,
            "cache_key": key,
            "status": status,
        })

        # Affichage périodique de l'avancement tous les 50 éléments.
        if i % 50 == 0:
            print(f"  {i}/{len(df_ready)} (cache={n_cached}, "
                  f"fetched={n_fetched}, failed={n_failed})")

    # Écriture du journal final.
    pd.DataFrame(log_records).to_csv(LOG_CSV, index=False)

    # Audit synthétique de fin de run.
    print("\n" + "=" * 60)
    print("AUDIT FETCH")
    print("=" * 60)
    print(f"Granularité    : {args.granularity}")
    print(f"En cache       : {n_cached}")
    print(f"Téléchargés    : {n_fetched}")
    print(f"Échecs         : {n_failed}")
    print(f"Log            : {LOG_CSV}")
    print(f"Cache          : {CACHE_DIR}")

    return 0


# Point d'entrée du script.
# Permet d'exécuter main() uniquement si le fichier est lancé directement.
if __name__ == "__main__":
    sys.exit(main())