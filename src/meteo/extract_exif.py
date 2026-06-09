"""
extract_exif.py
---------------
Étape 1 du pipeline phase 2 : extraction EXIF globale.

Parcourt le dossier pest-diagnostic et extrait pour chaque image :
- la date EXIF (DateTimeOriginal de préférence, sinon DateTime)
- le fabricant et le modèle du téléphone (audit qualité)
- la taille du fichier
- la présence éventuelle de coordonnées GPS

Produit : outputs/manifest_exif.csv

Utilisation :
    python extract_exif.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS, GPSTAGS

# ---- Configuration ----------------------------------------------------------

ROOT = Path(
    "/mnt/c/Users/LARIS-1/Desktop/DJMA/app"
)  # <-- à adapter selon votre environnement
IMG_DIR = ROOT / "analyse_descriptive" / "pest-diagnostic"
OUTPUT = Path(__file__).parent / "outputs" / "manifest_exif.csv"

EXIF_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f")


# ---- Helpers ----------------------------------------------------------------


def parse_exif_date(value: str | None) -> datetime | None:
    """Tente plusieurs formats EXIF (avec et sans sous-secondes)."""
    if not value:
        return None
    for fmt in EXIF_DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def dms_to_decimal(dms, ref) -> float | None:
    try:
        deg, mn, sec = (float(x) for x in dms)
        decimal = deg + mn / 60 + sec / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, ValueError):
        return None


def extract_gps(exif: dict) -> tuple[float | None, float | None]:
    gps_raw = exif.get("GPSInfo")
    if not gps_raw:
        return None, None
    gps = {GPSTAGS.get(t, t): v for t, v in gps_raw.items()}
    lat = dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    lon = dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    return lat, lon


def extract_one(img_path: Path) -> dict:
    record = {
        "filename": img_path.name,
        "file_size_kb": round(img_path.stat().st_size / 1024, 1),
        "file_modification_date": datetime.fromtimestamp(
            img_path.stat().st_mtime
        ).isoformat(),
        "datetime_original": None,
        "datetime_source": None,
        "make": None,
        "model": None,
        "software": None,
        "gps_latitude": None,
        "gps_longitude": None,
        "image_width": None,
        "image_height": None,
        "exif_status": "ok",
    }

    try:
        with Image.open(img_path) as img:
            record["image_width"], record["image_height"] = img.size
            exif_raw = img._getexif()

            if not exif_raw:
                record["exif_status"] = "no_exif"
                return record

            exif = {
                TAGS.get(tag_id, tag_id): value for tag_id, value in exif_raw.items()
            }

            dt = parse_exif_date(exif.get("DateTimeOriginal"))
            if dt:
                record["datetime_source"] = "DateTimeOriginal"
            else:
                dt = parse_exif_date(exif.get("DateTime"))
                if dt:
                    record["datetime_source"] = "DateTime"

            record["datetime_original"] = dt.isoformat() if dt else None
            record["make"] = exif.get("Make")
            record["model"] = exif.get("Model")
            record["software"] = exif.get("Software")
            lat, lon = extract_gps(exif)
            record["gps_latitude"] = lat
            record["gps_longitude"] = lon

            if dt is None:
                record["exif_status"] = "no_date"

    except UnidentifiedImageError:
        record["exif_status"] = "unreadable"
    except Exception as exc:
        record["exif_status"] = f"error: {type(exc).__name__}"

    return record


# ---- Main -------------------------------------------------------------------


def main() -> int:
    if not IMG_DIR.exists():
        print(f"[ERREUR] Dossier introuvable : {IMG_DIR}", file=sys.stderr)
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in IMG_DIR.iterdir() if p.is_file())
    print(f"[INFO] {len(images)} fichiers détectés dans {IMG_DIR}")

    fieldnames = [
        "filename",
        "file_size_kb",
        "file_modification_date",
        "datetime_original",
        "datetime_source",
        "make",
        "model",
        "software",
        "gps_latitude",
        "gps_longitude",
        "image_width",
        "image_height",
        "exif_status",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        stats = {"ok": 0, "no_date": 0, "no_exif": 0, "unreadable": 0, "errors": 0}
        for i, img_path in enumerate(images, 1):
            record = extract_one(img_path)
            writer.writerow(record)

            status = record["exif_status"]
            if status == "ok":
                stats["ok"] += 1
            elif status == "no_date":
                stats["no_date"] += 1
            elif status == "no_exif":
                stats["no_exif"] += 1
            elif status == "unreadable":
                stats["unreadable"] += 1
            else:
                stats["errors"] += 1

            if i % 500 == 0:
                print(f"  {i}/{len(images)} traitées")

    total = len(images)
    print("\n" + "=" * 60)
    print("AUDIT EXIF")
    print("=" * 60)
    print(f"Total images          : {total}")
    print(f"  date EXIF valide    : {stats['ok']:>6} ({100*stats['ok']/total:.1f} %)")
    print(
        f"  pas de date EXIF    : {stats['no_date']:>6} ({100*stats['no_date']/total:.1f} %)"
    )
    print(
        f"  aucun bloc EXIF     : {stats['no_exif']:>6} ({100*stats['no_exif']/total:.1f} %)"
    )
    print(f"  fichiers illisibles : {stats['unreadable']:>6}")
    print(f"  autres erreurs      : {stats['errors']:>6}")
    print(f"\nManifest écrit       : {OUTPUT}")

    if stats["ok"] / total < 0.5:
        print("\n[ATTENTION] Moins de 50 % des images ont une date EXIF exploitable.")
        print("Plan B : la colonne file_modification_date est dispo comme proxy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
