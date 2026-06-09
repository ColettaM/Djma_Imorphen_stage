"""
diagnose_bulletin_consistency.py
---------------------------------
Avant de passer image-centric → bulletin-centric, on vérifie que les
images d'un même bulletin partagent bien :
  - le même garden_id
  - une date EXIF cohérente (écart max < quelques heures)

Si oui : agrégation triviale (prendre la 1re image, ou la médiane).
Si non : il faudra une stratégie de désambiguïsation explicite.

Utilisation :
    python diagnose_bulletin_consistency.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CTX = Path(__file__).parent / "outputs" / "phase2_context.csv"


def main() -> None:
    if not CTX.exists():
        print(f"[ERREUR] {CTX} manquant. Lance build_phase2_context.py.")
        return

    df = pd.read_csv(CTX, parse_dates=["datetime_original"])
    print(f"[INFO] {len(df)} lignes (granularité image)")
    print(f"[INFO] {df['bulletin_id'].nunique()} bulletins distincts")

    # ---- Distribution images / bulletin ----------------------------------
    print("\n[1] Distribution du nombre d'images par bulletin")
    counts = df.groupby("bulletin_id").size()
    print(counts.value_counts().sort_index().to_string())
    print(f"  multi-images : {(counts > 1).sum()} bulletins")

    multi = counts[counts > 1].index
    if len(multi) == 0:
        print("\n[OK] Tous les bulletins ont 1 seule image — pas d'enjeu.")
        return

    df_multi = df[df["bulletin_id"].isin(multi)]

    # ---- Cohérence garden_id intra-bulletin -------------------------------
    print("\n[2] Cohérence garden_id par bulletin (multi-images uniquement)")
    by_bull = df_multi.groupby("bulletin_id")
    garden_unique = by_bull["garden_id"].nunique(dropna=False)
    n_inconsistent_garden = (garden_unique > 1).sum()
    print(f"  bulletins multi-images avec garden_id divergent : "
          f"{n_inconsistent_garden}/{len(multi)}")
    if n_inconsistent_garden > 0:
        bad = garden_unique[garden_unique > 1].index
        print(f"  exemples : {list(bad[:5])}")

    # ---- Cohérence date EXIF intra-bulletin -------------------------------
    print("\n[3] Cohérence date EXIF par bulletin (multi-images uniquement)")
    valid = df_multi.dropna(subset=["datetime_original"])
    if valid.empty:
        print("  pas de date EXIF dans les multi-images")
    else:
        by_bull_date = valid.groupby("bulletin_id")["datetime_original"]
        spans = (by_bull_date.max() - by_bull_date.min()).dt.total_seconds() / 60
        print(f"  écart médian (min)  : {spans.median():.1f}")
        print(f"  écart max    (min)  : {spans.max():.1f}")
        print(f"  bulletins avec écart > 1h : {(spans > 60).sum()}")
        print(f"  bulletins avec écart > 24h : {(spans > 1440).sum()}")

    # ---- Verdict ---------------------------------------------------------
    print("\n[VERDICT]")
    if n_inconsistent_garden == 0 and (valid.groupby("bulletin_id")
                                       ["datetime_original"]
                                       .agg(lambda s: s.max() - s.min())
                                       .dt.total_seconds().max() < 3600):
        print("  ✓ Agrégation triviale possible :")
        print("    - garden_id constant par bulletin")
        print("    - dates EXIF dans la même heure")
        print("    → on peut prendre la 1re image (ou la médiane) sans risque")
    else:
        print("  ⚠ Cas hétérogènes détectés — stratégie d'agrégation à définir")


if __name__ == "__main__":
    main()
