"""
Croise les images du dossier pest-diagnostic avec le CSV des bulletins de Metabase
et résout les noms d'espèces et de bioagresseurs via les tables de liaison.

Schéma de nommage des fichiers :
    <NomBioagresseur>---<ID_Espèce>---<ID_Bulletin>---<hashHex>[.ext]

Tables de liaison attendues dans le dossier `tables_de_liaison/` :
    - product_pages.csv   (id, name) : espèces de plantes
    - plant_diseases.csv  (id, name) : maladies (utilisé si Pest Type = DISEASE)
    - plant_pests.csv     (id, name) : insectes/ravageurs (si Pest Type = INSECT)

NB : les espaces d'IDs maladies et insectes sont SÉPARÉS — c'est `Pest Type`
qui détermine la table à interroger.

Usage :
    python match_images.py
    python match_images.py --reorganize
    python match_images.py --reorganize --copy
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

FILENAME_PATTERN = re.compile(
    r"^(?P<bioagresseur>.+?)---"
    r"(?P<id_espece>\d+)---"
    r"(?P<id_bulletin>\d+)---"
    r"(?P<hash>[0-9a-f]+)"
    r"(?P<ext>\.[A-Za-z0-9]+)?$"
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_filename(filename: str) -> dict | None:
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    return {
        "filename": filename,
        "bioagresseur_filename": m["bioagresseur"].strip(),
        "id_espece_filename": int(m["id_espece"]),
        "id_bulletin": int(m["id_bulletin"]),
        "image_hash": m["hash"],
        "extension": m["ext"] or "",
    }


def scan_images(image_dir: Path) -> tuple[list[dict], list[str]]:
    parsed: list[dict] = []
    unparsed: list[str] = []
    for f in sorted(image_dir.iterdir()):
        if not f.is_file():
            continue
        info = parse_filename(f.name)
        if info is None:
            unparsed.append(f.name)
            continue
        info["path"] = str(f.resolve())
        info["size_bytes"] = f.stat().st_size
        info["mtime"] = datetime.fromtimestamp(f.stat().st_mtime).isoformat(
            timespec="seconds"
        )
        parsed.append(info)
    return parsed, unparsed


# --------------------------------------------------------------------------- #
# Chargement CSV + tables
# --------------------------------------------------------------------------- #


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ID Variété" in df.columns and df["ID Variété"].dtype == object:
        df["ID Variété"] = (
            df["ID Variété"].astype(str).str.replace(",", "", regex=False)
        )
    id_cols = [
        "ID",
        "ID Espèce",
        "ID Variété",
        "ID BioAgresseur",
        "User ID",
        "Garden ID",
        "Garden Plant ID",
    ]
    for c in id_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def load_lookup_tables(tables_dir: Path) -> tuple[dict, dict, dict]:
    """Charge les 3 tables → dict {id: name}."""
    tables = {}
    for fname, key in [
        ("product_pages.csv", "especes"),
        ("plant_diseases.csv", "diseases"),
        ("plant_pests.csv", "pests"),
    ]:
        path = tables_dir / fname
        if not path.exists():
            sys.exit(f"Table de liaison manquante : {path}")
        t = pd.read_csv(path)
        tables[key] = dict(zip(t["id"].astype(int), t["name"]))
    return tables["especes"], tables["diseases"], tables["pests"]


def resolve_bioagresseur_name(
    row: pd.Series, diseases: dict, pests: dict
) -> str | None:
    """Résout l'ID BioAgresseur via Pest Type."""
    if pd.isna(row["ID BioAgresseur"]) or pd.isna(row["Pest Type"]):
        return None
    bid = int(row["ID BioAgresseur"])
    if row["Pest Type"] == "DISEASE":
        return diseases.get(bid)
    if row["Pest Type"] == "INSECT":
        return pests.get(bid)
    return None


def enrich_with_names(
    df: pd.DataFrame, especes: dict, diseases: dict, pests: dict
) -> pd.DataFrame:
    """Ajoute nom_espece et nom_bioagresseur au DataFrame des bulletins."""
    df = df.copy()
    df["nom_espece"] = df["ID Espèce"].map(
        lambda x: especes.get(int(x)) if pd.notna(x) else None
    )
    df["nom_bioagresseur"] = df.apply(
        lambda r: resolve_bioagresseur_name(r, diseases, pests), axis=1
    )
    return df


# --------------------------------------------------------------------------- #
# Matching et exports
# --------------------------------------------------------------------------- #


def build_manifest_images(
    images: list[dict], df_enriched: pd.DataFrame
) -> pd.DataFrame:
    df_images = pd.DataFrame(images)
    if df_images.empty:
        return pd.DataFrame()

    df_csv = df_enriched.copy()
    df_csv["ID"] = df_csv["ID"].astype("Int64")

    merged = df_images.merge(
        df_csv,
        left_on="id_bulletin",
        right_on="ID",
        how="left",
        suffixes=("", "_csv"),
    )

    # Cohérence ID Espèce filename ↔ CSV
    merged["espece_match"] = (
        merged["id_espece_filename"].astype("Int64") == merged["ID Espèce"]
    )

    # Une image est marquée "de test" si son id_bulletin n'a pas trouvé de
    # correspondance dans le CSV des bulletins. Le dev a confirmé que les
    # diagnostics de test sont filtrés côté Metabase mais leurs images restent
    # dans le bucket Scaleway.
    merged["is_test"] = merged["ID"].isna()

    # Cohérence nom bioagresseur filename ↔ référentiel (insensible à la casse)
    def names_match(row):
        if (
            pd.isna(row["nom_bioagresseur"])
            or row["bioagresseur_filename"] == "NonConfirme"
        ):
            return None
        a = str(row["bioagresseur_filename"]).lower().strip()
        b = str(row["nom_bioagresseur"]).lower().strip()
        # Match exact OU l'un commence par l'autre (Fusarium/Fusariose)
        return a == b or a.startswith(b[:5]) or b.startswith(a[:5])

    merged["bio_name_match"] = merged.apply(names_match, axis=1)

    cols = [
        "id_bulletin",
        "is_test",
        "filename",
        "image_hash",
        "extension",
        "bioagresseur_filename",
        "nom_bioagresseur",
        "id_espece_filename",
        "ID Espèce",
        "nom_espece",
        "ID BioAgresseur",
        "Pest Type",
        "Parties inféctées",
        "Symptoms Description",
        "Status",
        "Type",
        "User ID",
        "Garden ID",
        "Garden Plant ID",
        "mtime",
        "size_bytes",
        "path",
        "espece_match",
        "bio_name_match",
    ]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].sort_values(["id_bulletin", "mtime"]).reset_index(drop=True)


def build_manifest_bulletins(
    manifest_images: pd.DataFrame, df_enriched: pd.DataFrame
) -> pd.DataFrame:
    if manifest_images.empty:
        agg = pd.DataFrame(
            columns=["id_bulletin", "n_images", "images", "n_images_test"]
        )
    else:
        # On compte TOUTES les images (utilisateurs + test). Le flag is_test
        # reste disponible au niveau de chaque image dans manifest_images.csv
        # pour permettre une analyse ciblée si nécessaire.
        agg = (
            manifest_images.groupby("id_bulletin", dropna=True)
            .agg(
                n_images=("filename", "count"),
                images=("filename", lambda s: " | ".join(s)),
                n_images_test=("is_test", "sum"),
            )
            .reset_index()
        )
        agg["n_images_test"] = agg["n_images_test"].astype(int)

    out = df_enriched.merge(agg, left_on="ID", right_on="id_bulletin", how="left")
    out["n_images"] = out["n_images"].fillna(0).astype(int)
    out["n_images_test"] = out["n_images_test"].fillna(0).astype(int)
    out["images"] = out["images"].fillna("")
    out = out.drop(columns=["id_bulletin"], errors="ignore")

    # Réorganiser pour mettre les colonnes lisibles en tête
    front = [
        "ID",
        "Status",
        "Type",
        "nom_espece",
        "nom_bioagresseur",
        "Pest Type",
        "Parties inféctées",
        "Symptoms Description",
        "n_images",
        "n_images_test",
    ]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


# --------------------------------------------------------------------------- #
# Réorganisation
# --------------------------------------------------------------------------- #


def reorganize(
    manifest_images: pd.DataFrame, image_dir: Path, out_dir: Path, copy: bool = False
) -> None:
    """Crée out_dir/<id_bulletin>/<filename> pour TOUTES les images.

    Les images de test (is_test=True) sont placées dans le même dossier que
    les images utilisateurs car on a fait le choix méthodologique de les
    inclure dans le pool exploitable. Le flag reste disponible dans
    manifest_images.csv pour les analyses ciblées.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_user = 0
    n_test = 0
    for _, row in manifest_images.iterrows():
        if pd.isna(row["id_bulletin"]):
            continue
        bid = int(row["id_bulletin"])
        target_dir = out_dir / str(bid)
        target_dir.mkdir(parents=True, exist_ok=True)
        src = image_dir / row["filename"]
        dst = target_dir / row["filename"]
        if dst.exists() or dst.is_symlink():
            continue
        if copy:
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        if row.get("is_test", False):
            n_test += 1
        else:
            n_user += 1
    mode = "copiées" if copy else "liées (symlink)"
    print(f"  → {n_user + n_test} images {mode} dans {out_dir}/")
    print(
        f"    dont {n_user} utilisateurs et {n_test} de test (flaggées is_test dans le manifest)"
    )


# --------------------------------------------------------------------------- #
# Rapport
# --------------------------------------------------------------------------- #


def write_report(
    manifest_images: pd.DataFrame,
    manifest_bulletins: pd.DataFrame,
    unparsed: list[str],
    df: pd.DataFrame,
    report_path: Path,
) -> None:
    lines: list[str] = []
    add = lines.append
    bar = "=" * 70

    add(bar)
    add("  RAPPORT DE MATCHING IMAGES <-> BULLETINS")
    add(bar)
    add(f'Date           : {datetime.now().isoformat(timespec="seconds")}')
    add(f"Bulletins CSV  : {len(df)}")
    add(f'  - dont PEST_IDENTIFICATION : {(df["Type"] == "PEST_IDENTIFICATION").sum()}')
    add(f'  - dont PEST_REPORT         : {(df["Type"] == "PEST_REPORT").sum()}')
    add(f"Images totales : {len(manifest_images) + len(unparsed)}")
    add(f"  - parsées    : {len(manifest_images)}")
    add(f"  - non parsées: {len(unparsed)}")

    # Mention neutre des images de test (incluses, mais flaggées pour traçabilité)
    if not manifest_images.empty and "is_test" in manifest_images.columns:
        n_test = int(manifest_images["is_test"].sum())
        n_user = int((~manifest_images["is_test"]).sum())
        add(f"  - utilisateurs : {n_user}")
        add(
            f"  - de test (incluses, flaggées is_test=True dans le manifest) : {n_test}"
        )
        if n_test:
            test_ids = sorted(
                int(x)
                for x in manifest_images.loc[manifest_images["is_test"], "id_bulletin"]
                .dropna()
                .unique()
            )
            add(f"    Bulletins de test concernés : {test_ids}")

    if unparsed:
        add("")
        add("Fichiers non parsables (à inspecter) :")
        for f in unparsed:
            add(f"  - {f}")

    add("")
    add("-" * 70)
    add("COUVERTURE IMAGES <-> BULLETINS (toutes images incluses)")
    add("-" * 70)
    if not manifest_images.empty:
        n_couverts = manifest_bulletins["n_images"].gt(0).sum()
        n_images_total = int(manifest_bulletins["n_images"].sum())
        add(
            f"Bulletins avec au moins 1 image  : {n_couverts} / {len(df)} "
            f"({n_couverts / len(df) * 100:.1f} %)"
        )
        add(f"  -> ces bulletins concentrent  : {n_images_total} images")
        add(
            f'Bulletins sans image (orphelins) : {(manifest_bulletins["n_images"] == 0).sum()}'
        )

        # Couverture par Type, en distinguant bulletins vs images
        for t in ["PEST_IDENTIFICATION", "PEST_REPORT"]:
            sub = manifest_bulletins[manifest_bulletins["Type"] == t]
            n_cov = int((sub["n_images"] > 0).sum())
            n_img = int(sub["n_images"].sum())
            if len(sub):
                add(
                    f"  {t:21s} : {n_cov} / {len(sub)} bulletins avec image "
                    f"({n_cov/len(sub)*100:.1f} %) | {n_img} images au total"
                )

        add("")
        add("Distribution du nombre d'images par bulletin (couverts uniquement) :")
        dist = (
            manifest_bulletins.loc[manifest_bulletins["n_images"] > 0, "n_images"]
            .value_counts()
            .sort_index()
        )
        for nn, c in dist.items():
            add(f"  {nn} image(s) -> {c} bulletin(s)  [soit {nn*c} images]")

    add("")
    add("-" * 70)
    add("COHÉRENCE FILENAME <-> RÉFÉRENTIELS (toutes images incluses)")
    add("-" * 70)
    if not manifest_images.empty:
        if "espece_match" in manifest_images.columns:
            n_match = int(manifest_images["espece_match"].fillna(False).sum())
            n_total = int(manifest_images["espece_match"].notna().sum())
            add(f"ID Espèce filename <-> CSV : {n_match} / {n_total} OK")

        if "bio_name_match" in manifest_images.columns:
            n_match = int(manifest_images["bio_name_match"].fillna(False).sum())
            n_total = int(manifest_images["bio_name_match"].notna().sum())
            add(
                f"Nom bioagresseur filename <-> référentiel (fuzzy) : {n_match} / {n_total} OK"
            )
            mismatches = manifest_images.loc[
                ~manifest_images["bio_name_match"],
                # manifest_images["bio_name_match"] == False,
                ["id_bulletin", "bioagresseur_filename", "nom_bioagresseur"],
            ]
            if not mismatches.empty:
                add("  Différences relevées :")
                for _, r in mismatches.drop_duplicates().iterrows():
                    add(
                        f"    bulletin {r['id_bulletin']}: "
                        f"filename='{r['bioagresseur_filename']}' vs réf='{r['nom_bioagresseur']}'"
                    )

    add("")
    add("-" * 70)
    add("SOUS-ENSEMBLES — focus PEST_IDENTIFICATION (analyse principale)")
    add("-" * 70)
    if not manifest_bulletins.empty:
        df_pi = manifest_bulletins[manifest_bulletins["Type"] == "PEST_IDENTIFICATION"]
        with_img = df_pi[df_pi["n_images"] > 0]
        text_filled = with_img["Symptoms Description"].notna() & (
            with_img["Symptoms Description"].astype(str).str.strip() != ""
        )
        diag_ok = with_img["nom_bioagresseur"].notna()
        complets = with_img[diag_ok & text_filled]
        n_img_avec_image = int(with_img["n_images"].sum())
        n_img_diag = int(with_img.loc[diag_ok, "n_images"].sum())
        n_img_texte = int(with_img.loc[text_filled, "n_images"].sum())
        n_img_complets = int(complets["n_images"].sum())
        add(f"PI total                                        : {len(df_pi)} bulletins")
        add(
            f"  ... avec image                                : {len(with_img)} bulletins "
            f"({n_img_avec_image} images)"
        )
        add(
            f"  ... avec image + diagnostic confirmé          : {int(diag_ok.sum())} bulletins "
            f"({n_img_diag} images)"
        )
        add(
            f"  ... avec image + description texte            : {int(text_filled.sum())} bulletins "
            f"({n_img_texte} images)"
        )
        add(
            f"  ... COMPLETS (image + texte + diagnostic)     : {len(complets)} bulletins "
            f"({n_img_complets} images)"
        )

    add("")
    add("-" * 70)
    add("SOUS-ENSEMBLES — annexe PEST_REPORT (hors analyse principale)")
    add("-" * 70)
    if not manifest_bulletins.empty:
        df_pr = manifest_bulletins[manifest_bulletins["Type"] == "PEST_REPORT"]
        with_img = df_pr[df_pr["n_images"] > 0]
        n_img_pr = int(with_img["n_images"].sum())
        add(f"PR total                       : {len(df_pr)} bulletins")
        add(
            f"  ... avec au moins 1 image    : {len(with_img)} bulletin(s) "
            f"({n_img_pr} images)"
        )
        if len(with_img):
            for _, r in with_img.iterrows():
                add(
                    f"      bulletin {int(r['ID'])} ({r.get('nom_espece', '?')} / "
                    f"{r.get('nom_bioagresseur', '?')}) : {int(r['n_images'])} images"
                )
        add(
            f'  ... avec bioagresseur déclaré: {df_pr["nom_bioagresseur"].notna().sum()} / {len(df_pr)}'
        )

    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nRapport écrit dans : {report_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Chemin du CSV Metabase (auto-détecté si absent)",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("pest-diagnostic"),
        help="Dossier des images (défaut: ./pest-diagnostic)",
    )
    p.add_argument(
        "--tables",
        type=Path,
        default=Path("tables_de_liaison"),
        help="Dossier des tables de liaison (défaut: ./tables_de_liaison)",
    )
    p.add_argument(
        "--reorganize", action="store_true", help="Crée by_bulletin/<ID>/<image>"
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Avec --reorganize: copier au lieu de symlink",
    )
    p.add_argument(
        "--out", type=Path, default=Path("."), help="Dossier de sortie (défaut: .)"
    )
    args = p.parse_args()

    if args.csv is None:
        candidates = sorted(Path(".").glob("rapports_de_diagnostiques_*.csv"))
        if not candidates:
            sys.exit(
                "Aucun CSV 'rapports_de_diagnostiques_*.csv' trouvé. "
                "Passe --csv <chemin>."
            )
        args.csv = candidates[-1]

    if not args.images.is_dir():
        sys.exit(f"Dossier d'images introuvable : {args.images}")
    if not args.tables.is_dir():
        sys.exit(f"Dossier des tables introuvable : {args.tables}")

    print(f"CSV    : {args.csv}")
    print(f"Images : {args.images}")
    print(f"Tables : {args.tables}")

    df = load_csv(args.csv)
    especes, diseases, pests = load_lookup_tables(args.tables)
    df_enriched = enrich_with_names(df, especes, diseases, pests)
    parsed, unparsed = scan_images(args.images)

    print(f"\nBulletins         : {len(df)}")
    print(f"Espèces référentiel: {len(especes)}")
    print(f"Maladies            : {len(diseases)}")
    print(f"Insectes            : {len(pests)}")
    print(f"Images parsées      : {len(parsed)}")
    print(f"Images non parsées  : {len(unparsed)}")

    manifest_images = build_manifest_images(parsed, df_enriched)
    manifest_bulletins = build_manifest_bulletins(manifest_images, df_enriched)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_images.to_csv(args.out / "manifest_images.csv", index=False)
    manifest_bulletins.to_csv(args.out / "manifest_bulletins.csv", index=False)
    df_enriched.to_csv(args.out / "bulletins_enrichis_avec_noms.csv", index=False)
    print("\nExports écrits :")
    print(f'  - {args.out / "manifest_images.csv"}')
    print(f'  - {args.out / "manifest_bulletins.csv"}')
    print(f'  - {args.out / "bulletins_enrichis_avec_noms.csv"}')

    if args.reorganize:
        print("\nRéorganisation par bulletin...")
        reorganize(
            manifest_images, args.images, args.out / "by_bulletin", copy=args.copy
        )

    print()
    write_report(
        manifest_images,
        manifest_bulletins,
        unparsed,
        df,
        args.out / "matching_report.txt",
    )


if __name__ == "__main__":
    main()
