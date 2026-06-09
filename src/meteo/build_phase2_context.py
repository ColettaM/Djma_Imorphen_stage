"""
build_phase2_context.py
-----------------------
Étape 2 du pipeline phase 2 : fusion des sources et décodage des id.

Architecture simplifiée grâce à `manifest_images.csv` qui contient déjà
le lien `id_bulletin ↔ filename` ainsi que toutes les métadonnées plante
résolues (nom_espece, nom_bioagresseur en clair, etc.).

Sources d'entrée :
  - manifest_images.csv  : pivot (filename, id_bulletin, Garden ID,
                           nom_espece, nom_bioagresseur, Status, ...)
  - manifest_exif.csv    : produit par extract_exif.py (date EXIF)
  - gardens.csv          : id → lat/lon, postal_code, expo, location

Produit : outputs/phase2_context.csv
          outputs/phase2_audit.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---- Configuration ----------------------------------------------------------


ROOT = Path("/mnt/c/Users/LARIS-1/Desktop/DJMA/app") # <-- à adapter selon votre environnement
EXIF_MANIFEST = Path(__file__).parent / "outputs" / "manifest_exif.csv"
MANIFEST_IMAGES = ROOT / "analyse_descriptive" / "manifest" / "manifest_images.csv"
GARDEN_CSV = ROOT / "analyse_descriptive" / "tables_de_liaison" / "gardens.csv"

OUTPUT = Path(__file__).parent / "outputs" / "phase2_context.csv"
OUTPUT_AUDIT = Path(__file__).parent / "outputs" / "phase2_audit.txt"


# ---- Tables de décodage -----------------------------------------------------

PLANT_LOCATION_MAP = {
    1: "INSIDE",
    2: "WINDOW",
    3: "BALCONY",
    4: "GARDEN",
    5: "GREENHOUSE",
}

SUNSHINE_TYPE_MAP = {
    1: "SUN",
    2: "HALF_SHADOW",
    3: "SHADOW",
}

WEATHER_RELEVANCE = {
    "GARDEN": "high",
    "BALCONY": "medium",
    "GREENHOUSE": "medium",
    "WINDOW": "low",
    "INSIDE": "low",
}

# Outdoor strict : la plante est pleinement exposée à la météo extérieure.
# Utilisé pour phase2_ready (calibration des règles agronomiques).
OUTDOOR_LOCATIONS = {"BALCONY", "GARDEN", "GREENHOUSE"}

# Outdoor étendu : ajoute WINDOW. Utilisé pour phase2_ready_loose
# (analyses exploratoires). Une plante derrière une vitre vit dans le
# climat intérieur, donc la météo extérieure y a une faible pertinence,
# mais on garde ces bulletins pour les statistiques d'audit.
OUTDOOR_LOCATIONS_LOOSE = OUTDOOR_LOCATIONS | {"WINDOW"}

RENAME_MANIFEST = {
    "id_bulletin": "bulletin_id",
    "ID Espèce": "espece_id",
    "ID BioAgresseur": "bioagresseur_id",
    "Pest Type": "pest_type",
    "Parties inféctées": "parties_infectees",
    "Symptoms Description": "symptoms_description",
    "Status": "status",
    "Type": "type",
    "User ID": "user_id",
    "Garden ID": "garden_id",
    "Garden Plant ID": "garden_plant_id",
}


# ---- Helpers ----------------------------------------------------------------

def read_csv_robust(path: Path) -> pd.DataFrame:
    """Charge un CSV en testant plusieurs encodages courants."""
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = df.columns.str.strip('"').str.strip()
            return df
        except UnicodeDecodeError as exc:
            last_err = exc
    raise RuntimeError(f"Impossible de lire {path} : {last_err}")


def sanity_check_filename_convention(df: pd.DataFrame) -> dict:
    """
    Vérifie que les filenames suivent la convention :
        <nom_bioagresseur>---<id_espece>---<id_bulletin>---<hashHex>

    Cette fonction sert d'audit défensif sur la cohérence du manifest_images.
    Elle ne corrige rien — elle signale les divergences.

    IMPORTANT : pour chaque contrôle, on n'évalue que les lignes où la
    colonne cible est non-nulle. Sinon les bulletins NOT_CONFIRMED (qui
    ont nom_bioagresseur = NaN) feraient chuter artificiellement le taux.
    """
    sample = df.dropna(subset=["filename"]).head(20000).copy()
    if sample.empty:
        return {"checked": 0, "well_formed": 0, "errors": [], "details": {}}

    split = sample["filename"].str.split("---", expand=True)
    if split.shape[1] < 4:
        return {
            "checked": len(sample),
            "well_formed": 0,
            "errors": ["Aucun filename ne contient les 4 segments attendus"],
            "details": {},
        }

    seg_bio, seg_esp, seg_bul, seg_hash = split[0], split[1], split[2], split[3]
    well_formed = split[3].notna().sum()

    errors = []
    details = {}

    # 1. segment[1] ↔ espece_id (ignorer les NaN)
    if "espece_id" in sample.columns:
        mask = sample["espece_id"].notna()
        if mask.any():
            seg_num = pd.to_numeric(seg_esp[mask], errors="coerce")
            match_rate = (seg_num == sample.loc[mask, "espece_id"]).mean()
            details["espece_id"] = (match_rate, mask.sum())
            if match_rate < 0.95:
                errors.append(
                    f"segment[1] ↔ espece_id : {100*match_rate:.1f} % "
                    f"sur {mask.sum()} lignes non-nulles"
                )

    # 2. segment[2] ↔ bulletin_id
    if "bulletin_id" in sample.columns:
        mask = sample["bulletin_id"].notna()
        if mask.any():
            seg_num = pd.to_numeric(seg_bul[mask], errors="coerce")
            match_rate = (seg_num == sample.loc[mask, "bulletin_id"]).mean()
            details["bulletin_id"] = (match_rate, mask.sum())
            if match_rate < 0.95:
                errors.append(
                    f"segment[2] ↔ bulletin_id : {100*match_rate:.1f} % "
                    f"sur {mask.sum()} lignes non-nulles"
                )

    # 3. segment[3] ↔ image_hash
    if "image_hash" in sample.columns:
        mask = sample["image_hash"].notna()
        if mask.any():
            match_rate = (seg_hash[mask] == sample.loc[mask, "image_hash"]).mean()
            details["image_hash"] = (match_rate, mask.sum())
            if match_rate < 0.95:
                errors.append(
                    f"segment[3] ↔ image_hash : {100*match_rate:.1f} % "
                    f"sur {mask.sum()} lignes non-nulles"
                )

    # 4. segment[0] ↔ nom_bioagresseur (NaN exclus, comparaison normalisée)
    if "nom_bioagresseur" in sample.columns:
        mask = sample["nom_bioagresseur"].notna()
        if mask.any():
            left = seg_bio[mask].astype(str).str.strip().str.lower()
            right = sample.loc[mask, "nom_bioagresseur"].astype(str).str.strip().str.lower()
            match_rate = (left == right).mean()
            details["nom_bioagresseur"] = (match_rate, mask.sum())
            if match_rate < 0.95:
                errors.append(
                    f"segment[0] ↔ nom_bioagresseur : {100*match_rate:.1f} % "
                    f"sur {mask.sum()} lignes non-nulles"
                )

    return {
        "checked": len(sample),
        "well_formed": int(well_formed),
        "errors": errors,
        "details": details,
    }


# ---- Main -------------------------------------------------------------------

def main() -> int:
    for path, label in [
        (EXIF_MANIFEST, "manifest_exif.csv (étape 1)"),
        (MANIFEST_IMAGES, "manifest_images.csv"),
        (GARDEN_CSV, "gardens.csv"),
    ]:
        if not path.exists():
            print(f"[ERREUR] {label} manquant : {path}", file=sys.stderr)
            if path == EXIF_MANIFEST:
                print("→ Lance extract_exif.py d'abord.", file=sys.stderr)
            return 1

    print(f"[INFO] EXIF        : {EXIF_MANIFEST.name}")
    print(f"[INFO] Manifest    : {MANIFEST_IMAGES.name}")
    print(f"[INFO] Garden      : {GARDEN_CSV.name}")

    df_exif = pd.read_csv(EXIF_MANIFEST, parse_dates=["datetime_original"])
    df_man = read_csv_robust(MANIFEST_IMAGES)
    df_gard = read_csv_robust(GARDEN_CSV)

    print(f"\n[INFO] EXIF       : {len(df_exif):>6} lignes")
    print(f"[INFO] Manifest   : {len(df_man):>6} lignes")
    print(f"[INFO] Gardens    : {len(df_gard):>6} lignes")

    df_man = df_man.rename(columns={k: v for k, v in RENAME_MANIFEST.items()
                                     if k in df_man.columns})
    df_gard = df_gard.rename(columns={"id": "garden_id"})

    print(f"\n[DEBUG] Colonnes manifest après renommage :")
    print(f"        {list(df_man.columns)}")

    required = {"bulletin_id", "filename", "garden_id"}
    missing = required - set(df_man.columns)
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans manifest_images : {missing}",
              file=sys.stderr)
        return 1

    if "is_test" in df_man.columns:
        n_test = (df_man["is_test"] == True).sum()  # noqa: E712
        print(f"\n[INFO] Images is_test=True : {n_test} (gardées dans le run)")

    # ---- Sanity check de la convention de nommage -------------------------
    # Convention attendue : <nom_bioagresseur>---<id_espece>---<id_bulletin>---<hashHex>
    print("\n[INFO] Sanity check sur la convention de nommage...")
    sc = sanity_check_filename_convention(df_man)
    print(f"  filenames vérifiés        : {sc['checked']}")
    print(f"  bien formés (4 segments)  : {sc['well_formed']}")
    for col, (rate, n) in sc.get("details", {}).items():
        symbol = "✓" if rate >= 0.95 else "⚠"
        print(f"  {symbol} segment ↔ {col:18s} : {100*rate:5.1f} % ({n} lignes non-nulles)")
    if not sc["errors"]:
        print("  ✓ convention respectée à ≥ 95 % sur tous les contrôles")

    # ---- Jointure 1 : manifest ⨝ EXIF sur filename ------------------------
    df = df_man.merge(df_exif, on="filename", how="left")
    n_with_exif = df["datetime_original"].notna().sum()
    print(f"\n[INFO] Images avec EXIF apparié : {n_with_exif}/{len(df)} "
          f"({100*n_with_exif/len(df):.1f} %)")

    # ---- Jointure 2 : ⨝ Garden sur garden_id ------------------------------
    keep_garden = ["garden_id", "postal_code", "city", "country",
                   "latitude", "longitude",
                   "sunshine_type_id", "plant_location_id"]
    keep_garden = [c for c in keep_garden if c in df_gard.columns]
    df = df.merge(df_gard[keep_garden], on="garden_id", how="left")

    n_with_garden = df["latitude"].notna().sum()
    print(f"[INFO] Images avec Garden apparié : {n_with_garden}/{len(df)} "
          f"({100*n_with_garden/len(df):.1f} %)")

    # ---- Décodage des id --------------------------------------------------
    df["plant_location"] = df["plant_location_id"].map(PLANT_LOCATION_MAP)
    df["sunshine_type"] = df["sunshine_type_id"].map(SUNSHINE_TYPE_MAP)
    df["weather_relevance"] = df["plant_location"].map(WEATHER_RELEVANCE)

    # ---- Drapeaux outdoor (strict vs loose) -------------------------------
    # outdoor       : GARDEN, BALCONY, GREENHOUSE (météo pleinement pertinente)
    # outdoor_loose : + WINDOW (météo faiblement pertinente, mais conservé
    #                 pour l'audit et les analyses exploratoires)
    df["outdoor"] = df["plant_location"].isin(OUTDOOR_LOCATIONS)
    df["outdoor_loose"] = df["plant_location"].isin(OUTDOOR_LOCATIONS_LOOSE)

    # ---- Drapeaux phase2_ready --------------------------------------------
    has_image = df["filename"].notna()
    has_exif_date = df["datetime_original"].notna()
    has_coords = df["latitude"].notna() & df["longitude"].notna()

    # phase2_ready (strict) : pour la calibration des règles agronomiques.
    # phase2_ready_loose    : pour les analyses exploratoires, inclut WINDOW.
    df["phase2_ready"] = has_image & has_exif_date & has_coords & df["outdoor"]
    df["phase2_ready_loose"] = has_image & has_exif_date & has_coords & df["outdoor_loose"]

    if "status" in df.columns:
        df["status_confirmed"] = (df["status"] == "CONFIRMED")

    # ---- Sélection finale --------------------------------------------------
    final_cols = [
        # Identifiants
        "bulletin_id", "filename", "image_hash",
        # Date d'observation
        "datetime_original", "datetime_source", "mtime",
        # Localisation
        "garden_id", "user_id", "latitude", "longitude",
        "postal_code", "city", "country",
        # Contexte jardin
        "plant_location_id", "plant_location",
        "sunshine_type_id", "sunshine_type",
        "weather_relevance", "outdoor", "outdoor_loose",
        # Métadonnées plante (déjà résolues dans le manifest)
        "espece_id", "nom_espece",
        "bioagresseur_id", "nom_bioagresseur",
        "pest_type", "parties_infectees", "symptoms_description",
        "status", "type",
        # Audit qualité
        "make", "model", "size_bytes", "is_test",
        # Drapeaux
        "phase2_ready_loose", "phase2_ready", "status_confirmed",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df_out = df[final_cols].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT, index=False)
    print(f"\n[OK] {OUTPUT} ({len(df_out)} lignes, {len(final_cols)} colonnes)")

    # ---- Audit -------------------------------------------------------------
    audit_lines = []
    add = audit_lines.append
    add("=" * 60)
    add("AUDIT phase2_context (granularité : image)")
    add("=" * 60)
    add(f"Images totales                : {len(df_out)}")
    if "bulletin_id" in df_out.columns:
        add(f"  bulletins distincts couverts : {df_out['bulletin_id'].nunique()}")
    add(f"  avec date EXIF              : {df_out['datetime_original'].notna().sum()}")
    add(f"  avec coordonnées GPS        : {df_out['latitude'].notna().sum()}")
    add(f"  outdoor strict              : {df_out['outdoor'].sum()}")
    add(f"  outdoor loose (+WINDOW)     : {df_out['outdoor_loose'].sum()}")
    add(f"  phase2_ready (strict)       : {df_out['phase2_ready'].sum()}")
    add(f"  phase2_ready_loose          : {df_out['phase2_ready_loose'].sum()}")
    if "status_confirmed" in df_out.columns:
        add(f"  status CONFIRMED            : {df_out['status_confirmed'].sum()}")
    add("")
    add("Sanity check convention de nommage :")
    add(f"  filenames vérifiés          : {sc['checked']}")
    add(f"  bien formés (4 segments)    : {sc['well_formed']}")
    if sc["errors"]:
        add("  divergences :")
        for err in sc["errors"]:
            add(f"    - {err}")
    else:
        add("  ✓ convention respectée à ≥ 95 %")
    add("")
    add("Répartition plant_location (par image) :")
    add(df_out["plant_location"].value_counts(dropna=False).to_string())
    add("")
    add("Répartition sunshine_type (par image) :")
    add(df_out["sunshine_type"].value_counts(dropna=False).to_string())
    add("")
    if "nom_bioagresseur" in df_out.columns:
        add("Top 15 bioagresseurs (par image) :")
        add(df_out["nom_bioagresseur"].value_counts(dropna=False)
            .head(15).to_string())
        add("")
    if "nom_espece" in df_out.columns:
        add("Top 15 espèces (par image) :")
        add(df_out["nom_espece"].value_counts(dropna=False)
            .head(15).to_string())
        add("")
    if "model" in df_out.columns:
        add("Diversité téléphones (top 10, par image) :")
        add(df_out["model"].value_counts(dropna=False).head(10).to_string())
        add("")
    if "status" in df_out.columns:
        add("Distribution status (par image) :")
        add(df_out["status"].value_counts(dropna=False).to_string())

    audit_text = "\n".join(audit_lines)
    print("\n" + audit_text)
    OUTPUT_AUDIT.write_text(audit_text, encoding="utf-8")
    print(f"\n[OK] Audit écrit : {OUTPUT_AUDIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
