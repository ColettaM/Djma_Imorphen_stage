"""
aggregate_to_bulletins.py
-------------------------
Étape 2bis : produit phase2_bulletins.csv (bulletin-centric).

Architecture :
  manifest_bulletins.csv (table de faits officielle DJMA)
        │
        │  LEFT JOIN sur bulletin_id
        ▼
  phase2_context.csv (image-centric, agrégé par bulletin)
        │
        ▼
  phase2_bulletins.csv

manifest_bulletins.csv est la source de vérité pour :
  - l'identité du bulletin (Status, Type, Pest Type, IDs)
  - les compteurs officiels (n_images, n_images_test)
  - le lien jardin (Garden ID, Garden Plant ID, User ID)

phase2_context.csv apporte les agrégats calculés :
  - datetime_observation (médiane EXIF des images du bulletin)
  - latitude, longitude (depuis garden.csv)
  - plant_location, sunshine_type, weather_relevance
  - phase2_ready / phase2_ready_loose

Produit : outputs/phase2_bulletins.csv
          outputs/phase2_bulletins_audit.txt
"""

from __future__ import annotations

import sys
from pathlib import Path 

import pandas as pd


ROOT = Path("/mnt/c/Users/LARIS-1/Desktop/DJMA/app") # <-- à adapter selon votre environnement
MANIFEST_BULLETINS = ROOT / "analyse_descriptive" / "manifest" / "manifest_bulletins.csv"
PHASE2_CTX = Path(__file__).parent / "outputs" / "phase2_context.csv"

OUTPUT = Path(__file__).parent / "outputs" / "phase2_bulletins.csv"
OUTPUT_AUDIT = Path(__file__).parent / "outputs" / "phase2_bulletins_audit.txt"

RENAME_BULLETINS = {
    "ID": "bulletin_id",
    "Status": "status",
    "Type": "type",
    "Pest Type": "pest_type",
    "Parties inféctées": "parties_infectees",
    "Symptoms Description": "symptoms_description",
    "ID Espèce": "espece_id",
    "ID Variété": "variete_id",
    "ID BioAgresseur": "bioagresseur_id",
    "User ID": "user_id",
    "Garden ID": "garden_id",
    "Garden Plant ID": "garden_plant_id",
}


# ---- Helpers ----------------------------------------------------------------

def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = df.columns.str.strip('"').str.strip()
            return df
        except UnicodeDecodeError as exc:
            last_err = exc
    raise RuntimeError(f"Impossible de lire {path} : {last_err}")


def take_unique_or_mode(series: pd.Series) -> object:
    """Retourne la valeur unique non-nulle ; si plusieurs, prend le mode."""
    unique = series.dropna().unique()
    if len(unique) == 0:
        return None
    if len(unique) == 1:
        return unique[0]
    return series.dropna().mode().iloc[0]


def aggregate_images_by_bulletin(df_ctx: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège phase2_context.csv (image-centric) par bulletin.
    Calcule UNIQUEMENT les champs qui ne peuvent venir que des images :
    date EXIF, cohérence intra-bulletin, drapeaux phase2_ready.
    """
    rows = []
    for bid, group in df_ctx.groupby("bulletin_id", dropna=False):
        out = {"bulletin_id": bid}

        # Compteurs calculés depuis les images réelles
        out["n_images_found"] = len(group)
        if "datetime_original" in group.columns:
            out["n_images_with_exif"] = group["datetime_original"].notna().sum()

        # Date d'observation : médiane des dates EXIF
        if "datetime_original" in group.columns:
            dates = pd.to_datetime(group["datetime_original"], errors="coerce").dropna()
            if not dates.empty:
                sorted_dates = dates.sort_values()
                out["datetime_observation"] = sorted_dates.iloc[len(sorted_dates) // 2]
                out["date_span_minutes"] = round(
                    (dates.max() - dates.min()).total_seconds() / 60, 1
                )
            else:
                out["datetime_observation"] = pd.NaT
                out["date_span_minutes"] = None

        # Localisation (vient de garden.csv via build_phase2_context)
        for col in ["latitude", "longitude", "postal_code", "city", "country"]:
            if col in group.columns:
                out[col] = take_unique_or_mode(group[col])
                out[f"{col}_n_distinct"] = group[col].dropna().nunique()

        # Contexte jardin
        for col in ["plant_location_id", "plant_location",
                    "sunshine_type_id", "sunshine_type",
                    "weather_relevance", "outdoor"]:
            if col in group.columns:
                out[col] = take_unique_or_mode(group[col])

        # Drapeaux : ready si AU MOINS UNE image l'est
        if "phase2_ready" in group.columns:
            out["phase2_ready"] = bool(group["phase2_ready"].any())
        if "phase2_ready_loose" in group.columns:
            out["phase2_ready_loose"] = bool(group["phase2_ready_loose"].any())

        # Liste des fichiers (debug/audit)
        if "filename" in group.columns:
            out["filenames"] = ";".join(group["filename"].dropna().astype(str))

        rows.append(out)

    return pd.DataFrame(rows)


# ---- Main -------------------------------------------------------------------

def main() -> int:
    for path, label in [
        (MANIFEST_BULLETINS, "manifest_bulletins.csv"),
        (PHASE2_CTX, "phase2_context.csv (étape 2)"),
    ]:
        if not path.exists():
            print(f"[ERREUR] {label} manquant : {path}", file=sys.stderr)
            return 1

    print(f"[INFO] Manifest bulletins : {MANIFEST_BULLETINS.name}")
    print(f"[INFO] Phase2 context     : {PHASE2_CTX.name}")

    df_bull = read_csv_robust(MANIFEST_BULLETINS)
    df_ctx = pd.read_csv(PHASE2_CTX, parse_dates=["datetime_original"])

    print(f"\n[INFO] manifest_bulletins : {len(df_bull):>5} bulletins")
    print(f"[INFO] phase2_context     : {len(df_ctx):>5} images")

    df_bull = df_bull.rename(columns={k: v for k, v in RENAME_BULLETINS.items()
                                       if k in df_bull.columns})

    print(f"\n[DEBUG] Colonnes manifest_bulletins après renommage :")
    print(f"        {list(df_bull.columns)}")

    if "bulletin_id" not in df_bull.columns:
        print("[ERREUR] Colonne ID/bulletin_id introuvable dans manifest_bulletins",
              file=sys.stderr)
        return 1

    keep_bull = [
        "bulletin_id", "status", "type",
        "nom_espece", "nom_bioagresseur", "pest_type",
        "parties_infectees", "symptoms_description",
        "n_images", "n_images_test",
        "espece_id", "variete_id", "bioagresseur_id",
        "user_id", "garden_id", "garden_plant_id",
    ]
    keep_bull = [c for c in keep_bull if c in df_bull.columns]
    df_bull = df_bull[keep_bull]

    df_agg = aggregate_images_by_bulletin(df_ctx)
    print(f"\n[INFO] Bulletins agrégés depuis images : {len(df_agg)}")

    # LEFT JOIN : on garde tous les bulletins officiels
    df = df_bull.merge(df_agg, on="bulletin_id", how="left")
    print(f"[INFO] Bulletins après LEFT JOIN        : {len(df)}")

    # Sanity : cohérence n_images officiel vs trouvé
    if "n_images" in df.columns and "n_images_found" in df.columns:
        df["n_images_match"] = (df["n_images"] == df["n_images_found"])
        n_mismatch = (~df["n_images_match"] & df["n_images_found"].notna()).sum()
        print(f"\n[INFO] n_images officiel ≠ trouvé : {n_mismatch} bulletins")
        if n_mismatch > 0:
            print("       (images manquantes sur disque ou manifest_images incomplet)")

    if "status" in df.columns:
        df["status_confirmed"] = (df["status"] == "CONFIRMED")

    final_cols = [
        "bulletin_id",
        "status", "status_confirmed", "type",
        "nom_espece", "espece_id", "variete_id",
        "nom_bioagresseur", "bioagresseur_id", "pest_type",
        "parties_infectees", "symptoms_description",
        "n_images", "n_images_test",
        "n_images_found", "n_images_with_exif", "n_images_match",
        "filenames",
        "datetime_observation", "date_span_minutes",
        "user_id", "garden_id", "garden_plant_id",
        "latitude", "longitude",
        "postal_code", "city", "country",
        "plant_location_id", "plant_location",
        "sunshine_type_id", "sunshine_type",
        "weather_relevance", "outdoor",
        "latitude_n_distinct", "garden_id_n_distinct",
        "phase2_ready", "phase2_ready_loose",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df_out = df[final_cols].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT, index=False)
    print(f"\n[OK] {OUTPUT} ({len(df_out)} bulletins, {len(final_cols)} colonnes)")

    # ---- Audit -----------------------------------------------------------
    audit_lines = []
    add = audit_lines.append
    add("=" * 60)
    add("AUDIT phase2_bulletins")
    add("=" * 60)
    add(f"Bulletins officiels DJMA          : {len(df_out)}")
    add(f"  avec images trouvées            : {df_out['n_images_found'].notna().sum()}")
    add(f"  avec date EXIF (médiane)        : {df_out['datetime_observation'].notna().sum()}")
    add(f"  avec coordonnées GPS            : {df_out['latitude'].notna().sum()}")
    if "outdoor" in df_out.columns:
        add(f"  outdoor                         : {df_out['outdoor'].sum()}")
    if "phase2_ready" in df_out.columns:
        add(f"  phase2_ready                    : {df_out['phase2_ready'].sum()}")
        add(f"  phase2_ready_loose              : {df_out['phase2_ready_loose'].sum()}")
    if "status_confirmed" in df_out.columns:
        add(f"  status CONFIRMED                : {df_out['status_confirmed'].sum()}")

    if "n_images" in df_out.columns:
        add("")
        add("Distribution n_images officiel :")
        add(df_out["n_images"].value_counts().sort_index().to_string())
        add(f"  bulletins multi-images          : {(df_out['n_images'] > 1).sum()}")

    if "n_images_match" in df_out.columns:
        n_mismatch = (~df_out["n_images_match"] & df_out["n_images_found"].notna()).sum()
        add("")
        add(f"Cohérence n_images officiel ↔ trouvé : "
            f"{n_mismatch} bulletins divergents")

    if "date_span_minutes" in df_out.columns:
        spans = df_out["date_span_minutes"].dropna()
        if not spans.empty and len(spans) > 0:
            add("")
            add("Écart temporel intra-bulletin (minutes) :")
            add(f"  médiane                       : {spans.median():.1f}")
            add(f"  max                           : {spans.max():.1f}")
            add(f"  bulletins avec écart > 1h     : {(spans > 60).sum()}")
            add(f"  bulletins avec écart > 24h    : {(spans > 1440).sum()}")

    add("")
    add("Top 15 bioagresseurs (au niveau bulletin) :")
    add(df_out["nom_bioagresseur"].value_counts(dropna=False)
        .head(15).to_string())
    add("")
    add("Distribution status :")
    add(df_out["status"].value_counts(dropna=False).to_string())

    audit_text = "\n".join(audit_lines)
    print("\n" + audit_text)
    OUTPUT_AUDIT.write_text(audit_text, encoding="utf-8")
    print(f"\n[OK] Audit écrit : {OUTPUT_AUDIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
