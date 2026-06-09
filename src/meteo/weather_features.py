"""
weather_features.py
-------------------
Étape 3b : agrégation des séries météo brutes en features synthétiques.

Pour chaque bulletin avec données météo en cache, calcule un vecteur de
features sur la fenêtre J-15 → J-1.

Helpers réutilisés du pipeline du collègue :
  - longest_streak()           : plus longue série consécutive de True
  - count_events_with_min_len  : nb d'épisodes dépassant une longueur
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---- Configuration ----------------------------------------------------------

PHASE2_BULLETINS = Path(__file__).parent / "outputs" / "phase2_bulletins.csv"
PHASE2_CTX = Path(__file__).parent / "outputs" / "phase2_context.csv"
CACHE_DIR = Path(__file__).parent / "outputs" / "weather_raw"
OUTPUT_BULLETINS = Path(__file__).parent / "outputs" / "phase2_bulletins_features.csv"
OUTPUT_LEGACY = Path(__file__).parent / "outputs" / "phase2_features.csv"

WINDOW_DAYS = 15
HOT_THRESHOLD = 28.0
COLD_THRESHOLD = 10.0
RAIN_THRESHOLD = 1.0  # mm/jour
HEATWAVE_THRESHOLD = 30.0
HEATWAVE_MIN_DAYS = 3


# ---- Helpers (repris/adaptés du pipeline du collègue) -----------------------

def longest_streak(boolean_series: pd.Series) -> int:
    """Plus longue série consécutive de True."""
    if boolean_series.empty:
        return 0
    s = boolean_series.fillna(False).astype(int)
    groups = (s != s.shift()).cumsum()
    streaks = s.groupby(groups).sum()
    return int(streaks.max()) if not streaks.empty else 0


def count_events_with_min_length(boolean_series: pd.Series, min_length: int = 3) -> int:
    """Nb d'épisodes consécutifs de True ≥ min_length."""
    if boolean_series.empty:
        return 0
    s = boolean_series.fillna(False).astype(int)
    groups = (s != s.shift()).cumsum()
    streaks = s.groupby(groups).sum()
    return int((streaks >= min_length).sum())


def cache_key(lat: float, lon: float, date) -> str:
    return f"{round(lat, 4)}_{round(lon, 4)}_{date.date().isoformat()}"


# ---- Calcul des features ----------------------------------------------------

def compute_features(wx: pd.DataFrame) -> dict:
    """Vecteur de features à partir d'un DataFrame quotidien Open-Meteo."""
    f: dict = {}

    # ---- Pluie -----------------------------------------------------------
    rain = wx["precipitation_sum"].fillna(0)
    f["rain_sum_15d"] = round(float(rain.sum()), 2)
    f["rain_max_day_15d"] = round(float(rain.max()), 2)
    f["rain_days_15d"] = int((rain >= RAIN_THRESHOLD).sum())
    f["heavy_rain_days_15d"] = int((rain >= 20).sum())
    dry_mask = rain < RAIN_THRESHOLD
    f["dry_days_15d"] = int(dry_mask.sum())
    f["max_dry_streak_15d"] = longest_streak(dry_mask)

    # ---- Humidité --------------------------------------------------------
    rh = wx.get("relative_humidity_2m_mean")
    f["humidity_mean_15d"] = round(float(rh.mean()), 1) if rh is not None and rh.notna().any() else None
    f["humidity_max_15d"] = round(float(rh.max()), 1) if rh is not None and rh.notna().any() else None
    rh_min = wx.get("relative_humidity_2m_min")
    f["humidity_min_15d"] = round(float(rh_min.min()), 1) if rh_min is not None and rh_min.notna().any() else None

    # Nb de jours d'humidité haute (≥ 85 %), pertinent pour la cladosporiose
    if rh is not None:
        f["high_humidity_days_15d"] = int((rh >= 85).sum())
    else:
        f["high_humidity_days_15d"] = None

    # ---- Point de rosée et VPD -----------------------------------------
    # Point de rosée élevé → condensation foliaire → conditions de mildiou
    dew = wx.get("dew_point_2m_mean")
    f["dew_point_mean_15d"] = round(float(dew.mean()), 2) if dew is not None and dew.notna().any() else None

    # VPD élevé → stress hydrique foliaire, conditions favorables aux acariens
    vpd = wx.get("vapour_pressure_deficit_max")
    f["vpd_max_15d"] = round(float(vpd.max()), 3) if vpd is not None and vpd.notna().any() else None
    f["vpd_mean_15d"] = round(float(vpd.mean()), 3) if vpd is not None and vpd.notna().any() else None

    # ---- Température ----------------------------------------------------
    tmean = wx.get("temperature_2m_mean")
    tmax = wx.get("temperature_2m_max")
    tmin = wx.get("temperature_2m_min")
    f["temp_mean_15d"] = round(float(tmean.mean()), 2) if tmean is not None else None
    f["temp_max_15d"] = round(float(tmax.max()), 2) if tmax is not None else None
    f["temp_min_15d"] = round(float(tmin.min()), 2) if tmin is not None else None
    f["hot_days_15d"] = int((tmax > HOT_THRESHOLD).sum()) if tmax is not None else None
    f["cold_days_15d"] = int((tmin < COLD_THRESHOLD).sum()) if tmin is not None else None

    # Évènements canicule (≥3 jours consécutifs > 30 °C)
    if tmax is not None:
        f["heatwave_events_15d"] = count_events_with_min_length(
            tmax >= HEATWAVE_THRESHOLD, HEATWAVE_MIN_DAYS
        )
    else:
        f["heatwave_events_15d"] = None

    # ---- Vent -----------------------------------------------------------
    wind = wx.get("wind_speed_10m_max")
    f["wind_max_15d"] = round(float(wind.max()), 1) if wind is not None else None
    gust = wx.get("wind_gusts_10m_max")
    f["gust_max_15d"] = round(float(gust.max()), 1) if gust is not None else None

    # ---- Bilan hydrique et rayonnement ----------------------------------
    et0 = wx.get("et0_fao_evapotranspiration")
    if et0 is not None:
        f["et0_sum_15d"] = round(float(et0.sum()), 2)
        f["water_balance_15d"] = round(float(rain.sum() - et0.sum()), 2)
    else:
        f["et0_sum_15d"] = None
        f["water_balance_15d"] = None

    rad = wx.get("shortwave_radiation_sum")
    f["radiation_sum_15d"] = round(float(rad.sum()), 2) if rad is not None else None

    # ---- Humidité du sol (proxy stress sécheresse racinaire) -----------
    soil = wx.get("soil_moisture_0_to_7cm_mean")
    if soil is not None and soil.notna().any():
        f["soil_moisture_mean_15d"] = round(float(soil.mean()), 4)
        f["soil_moisture_min_15d"] = round(float(soil.min()), 4)
    else:
        f["soil_moisture_mean_15d"] = None
        f["soil_moisture_min_15d"] = None

    return f


# ---- Main -------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--granularity", choices=["bulletin", "image"], default="bulletin",
                        help="bulletin = lit phase2_bulletins.csv, "
                             "image = lit phase2_context.csv (legacy).")
    args = parser.parse_args()

    if args.granularity == "bulletin":
        src = PHASE2_BULLETINS
        date_col = "datetime_observation"
        out = OUTPUT_BULLETINS
    else:
        src = PHASE2_CTX
        date_col = "datetime_original"
        out = OUTPUT_LEGACY

    if not src.exists():
        print(f"[ERREUR] {src} manquant.", file=sys.stderr)
        return 1
    if not CACHE_DIR.exists():
        print(f"[ERREUR] {CACHE_DIR} manquant. Lance fetch_weather.py.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(src, parse_dates=[date_col])

    feature_rows = []
    n_ok = n_missing = 0

    for row in df.itertuples():
        date_value = getattr(row, date_col)
        if pd.isna(row.latitude) or pd.isna(date_value):
            feature_rows.append({})
            n_missing += 1
            continue

        key = cache_key(row.latitude, row.longitude, date_value)
        cache_file = CACHE_DIR / f"{key}.parquet"
        if not cache_file.exists():
            feature_rows.append({})
            n_missing += 1
            continue

        wx = pd.read_parquet(cache_file)
        feature_rows.append(compute_features(wx))
        n_ok += 1

    df_feat = pd.DataFrame(feature_rows)
    df_full = pd.concat([df.reset_index(drop=True), df_feat], axis=1)
    df_full.to_csv(out, index=False)

    print("=" * 60)
    print("AUDIT FEATURES")
    print("=" * 60)
    print(f"Granularité           : {args.granularity}")
    print(f"Observations traitées : {len(df_full)}")
    print(f"  avec features météo : {n_ok}")
    print(f"  sans cache          : {n_missing}")
    print(f"\nFichier écrit         : {out}")

    valid = df_full.dropna(subset=["rain_sum_15d"])
    if not valid.empty:
        id_col = "bulletin_id" if args.granularity == "bulletin" else "filename"
        cols = [id_col, "rain_sum_15d", "humidity_mean_15d",
                "vpd_max_15d", "temp_mean_15d", "hot_days_15d",
                "soil_moisture_mean_15d"]
        cols = [c for c in cols if c in valid.columns]
        print("\nAperçu features (5 premières lignes) :")
        print(valid[cols].head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
