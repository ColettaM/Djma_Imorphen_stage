"""
Analyse qualité des bulletins de diagnostic — focus PEST_IDENTIFICATION.

Objectif
--------
Caractériser la qualité réelle des bulletins remontés via le tunnel guidé
PEST_IDENTIFICATION, seul parcours produisant des observations multimodales :
images + description libre + parties infectées + validation/non-validation d'un diagnostic.

Les PEST_REPORT sont conservés uniquement en annexe, car ils correspondent à une
déclaration directe de bioagresseur et ne produisent pas le même niveau d'information.

Usage :
    python analyse_bulletins.py [chemin_csv]

Entrées attendues :
    - rapports_de_diagnostiques_*.csv
    - tables_de_liaison/product_pages.csv
    - tables_de_liaison/plant_diseases.csv
    - tables_de_liaison/plant_pests.csv
    - manifest_bulletins.csv      (optionnel, produit par match_images.py)
    - manifest_images.csv         (optionnel, produit par match_images.py)

Sorties :
    - PI_bulletins_enrichis.csv
    - PI_exploitables_texte.csv
    - PI_complets.csv
    - PI_image_only.csv
    - PI_text_only.csv
    - PI_image_text.csv
    - PI_confirmed_only.csv
    - PI_weak_labels.csv
    - PI_text_quality.csv
    - PI_dataset_summary.csv
    - rapport_analyse_PI.txt
    - figures/*.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #


def find_csv() -> Path:
    """Trouve automatiquement le dernier export Metabase si aucun chemin n'est fourni."""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            sys.exit(f"Fichier introuvable : {p}")
        return p

    candidates = sorted(Path(".").glob("rapports_de_diagnostiques_*.csv"))
    if not candidates:
        sys.exit("Aucun fichier 'rapports_de_diagnostiques_*.csv' trouvé.")
    return candidates[-1]


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV brut et normalise les colonnes d'identifiants."""
    df = pd.read_csv(csv_path)

    # Certains exports contiennent des virgules dans ID Variété.
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


def load_lookups(tables_dir: Path) -> tuple[dict, dict, dict]:
    """Charge les tables de liaison espèces / maladies / ravageurs."""
    if not tables_dir.is_dir():
        sys.exit(f"Dossier de tables introuvable : {tables_dir}")

    required = ["product_pages.csv", "plant_diseases.csv", "plant_pests.csv"]
    for fname in required:
        if not (tables_dir / fname).exists():
            sys.exit(f"Table de liaison manquante : {tables_dir / fname}")

    e = pd.read_csv(tables_dir / "product_pages.csv")
    d = pd.read_csv(tables_dir / "plant_diseases.csv")
    p = pd.read_csv(tables_dir / "plant_pests.csv")

    return (
        dict(zip(e["id"].astype(int), e["name"])),
        dict(zip(d["id"].astype(int), d["name"])),
        dict(zip(p["id"].astype(int), p["name"])),
    )


def load_manifest_images(path: Path = Path("manifest_images.csv")) -> pd.DataFrame:
    """Charge le manifest image si disponible, sinon retourne un DataFrame vide."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Enrichissement
# --------------------------------------------------------------------------- #


def normalize_text_value(s: object) -> str:
    """Normalise une valeur texte sans modifier le CSV source."""
    if pd.isna(s):
        return ""
    return str(s).strip()


def has_text(s: object) -> bool:
    """Indique si une chaîne est réellement renseignée."""
    return normalize_text_value(s) != ""


def resolve_bioagresseur_name(
    row: pd.Series, diseases: dict, pests: dict
) -> str | None:
    """Résout le nom du bioagresseur via ID BioAgresseur + Pest Type."""
    if pd.isna(row.get("ID BioAgresseur")) or pd.isna(row.get("Pest Type")):
        return None

    bid = int(row["ID BioAgresseur"])
    if row["Pest Type"] == "DISEASE":
        return diseases.get(bid)
    if row["Pest Type"] == "INSECT":
        return pests.get(bid)
    return None


def enrich(
    df: pd.DataFrame, especes: dict, diseases: dict, pests: dict
) -> pd.DataFrame:
    """Ajoute les colonnes métier et les indicateurs d'exploitabilité."""
    df = df.copy()

    df["nom_espece"] = df["ID Espèce"].map(
        lambda x: especes.get(int(x)) if pd.notna(x) else None
    )
    df["nom_bioagresseur"] = df.apply(
        lambda r: resolve_bioagresseur_name(r, diseases, pests), axis=1
    )

    # Texte libre
    df["description_clean"] = df["Symptoms Description"].map(normalize_text_value)
    df["desc_filled"] = df["description_clean"].str.len() > 0
    df["desc_len_chars"] = df["description_clean"].str.len()
    df["desc_len_words"] = df["description_clean"].apply(
        lambda s: len(s.split()) if s else 0
    )

    # Parties infectées
    df["parties_clean"] = df["Parties inféctées"].map(normalize_text_value)
    df["parties_filled"] = df["parties_clean"].str.len() > 0

    # Croisement avec manifest_bulletins.csv, produit par match_images.py
    manifest_path = Path("manifest_bulletins.csv")
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        n_images = dict(zip(manifest["ID"], manifest["n_images"]))
        n_images_test = dict(zip(manifest["ID"], manifest.get("n_images_test", 0)))
        images = dict(zip(manifest["ID"], manifest.get("images", "")))
        df["n_images"] = df["ID"].map(
            lambda x: n_images.get(int(x), 0) if pd.notna(x) else 0
        )
        df["n_images_test"] = df["ID"].map(
            lambda x: n_images_test.get(int(x), 0) if pd.notna(x) else 0
        )
        df["images"] = df["ID"].map(
            lambda x: images.get(int(x), "") if pd.notna(x) else ""
        )
    else:
        df["n_images"] = 0
        df["n_images_test"] = 0
        df["images"] = ""

    df["has_image"] = df["n_images"] > 0

    # Diagnostic fort = diagnostic validé par l'utilisateur.
    df["avec_diagnostic"] = df["nom_bioagresseur"].notna() & (
        df["Status"] == "CONFIRMED"
    )

    # Texte exploitable = espèce + parties + description.
    df["exploitable_texte"] = (
        df["nom_espece"].notna() & df["parties_filled"] & df["desc_filled"]
    )

    # Complet = cas idéal pour dataset multimodal supervisé.
    df["complet"] = df["has_image"] & df["exploitable_texte"] & df["avec_diagnostic"]

    # Sous-datasets utiles pour la suite.
    df["image_only"] = df["has_image"] & ~df["exploitable_texte"]
    df["text_only"] = df["exploitable_texte"] & ~df["has_image"]
    df["image_text"] = df["has_image"] & df["exploitable_texte"]
    df["confirmed_only"] = df["avec_diagnostic"]
    df["weak_label"] = df["image_text"] & ~df["avec_diagnostic"]

    # Qualité texte simple pour prioriser l'inspection.
    df["text_quality"] = "missing"
    df.loc[df["desc_filled"] & (df["desc_len_words"] < 3), "text_quality"] = "too_short"
    df.loc[df["desc_len_words"].between(3, 9), "text_quality"] = "short"
    df.loc[df["desc_len_words"].between(10, 29), "text_quality"] = "medium"
    df.loc[df["desc_len_words"] >= 30, "text_quality"] = "long"

    return df


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #


def pct(n: int | float, d: int | float) -> float:
    return (n / d * 100) if d else 0.0


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def value_counts_with_pct(s: pd.Series) -> pd.DataFrame:
    counts = s.value_counts(dropna=False)
    out = counts.rename("n").reset_index()
    out.columns = ["valeur", "n"]
    out["pct"] = out["n"] / out["n"].sum() * 100
    return out


def extract_parties(df: pd.DataFrame) -> pd.Series:
    """Explose les parties infectées pour compter les modalités."""
    return (
        df.loc[df["parties_filled"], "parties_clean"]
        .str.split(r",\s*")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )


def _normalize_token(token: str, keep_as_is: set) -> str:
    """Retire les accents, met en minuscule, retire le 's' final si pluriel.

    Le résultat sert de CLÉ de regroupement uniquement. Le libellé d'affichage
    reste la forme la plus fréquente effectivement observée.
    """
    import unicodedata

    # 1. Casse + accents
    t = "".join(
        c
        for c in unicodedata.normalize("NFD", token.lower())
        if unicodedata.category(c) != "Mn"
    )
    # 2. Cas spéciaux protégés (ne pas dépluraliser)
    if t in keep_as_is:
        return t
    # 3. Pluriel naïf : on retire le 's' final si le mot fait au moins 4 lettres
    if t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def tokenize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Tokenisation simple + normalisation pour repérer le vocabulaire dominant.

    Normalisation niveau 2 : accents + casse + pluriel naïf, avec une table
    de cas spéciaux pour protéger les mots qui finissent par 's' au singulier.

    Retourne un DataFrame avec colonnes :
        - 'mot' : libellé d'affichage (forme la plus fréquente observée)
        - 'n'   : nombre d'occurrences (toutes variantes confondues)
    """
    text = " ".join(df.loc[df["desc_filled"], "description_clean"].astype(str)).lower()
    tokens = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ'-]{3,}", text)

    # Stopwords minimaux FR, volontairement simples pour garder les mots symptôme
    stopwords = {
        "les",
        "des",
        "une",
        "sur",
        "dans",
        "avec",
        "pour",
        "qui",
        "que",
        "aux",
        "est",
        "sont",
        "pas",
        "plus",
        "très",
        "tout",
        "tous",
        "elles",
        "elle",
        "feuille",
        "feuilles",
        "plante",
        "plants",
        "plant",
        "mon",
        "mes",
        "ses",
        "leurs",
        "cela",
        "comme",
        "avoir",
        "semble",
        "voit",
        "voir",
        "jardin",
    }
    tokens = [t for t in tokens if t not in stopwords]

    # Cas spéciaux : mots se terminant par 's' au singulier (à protéger)
    # 'vers' inclus car en contexte agricole c'est le pluriel de 'ver' (l'animal),
    # mais on choisit de le garder comme tel pour éviter la confusion avec 'ver'
    # (préposition rare). Au pire ça regroupe sur 'vers' qui est cohérent.
    keep_as_is = {
        "sans",
        "sous",
        "puis",
        "depuis",
        "mois",
        "fois",
        "pays",
        "corps",
        "bas",
        "gris",
        "frais",
        "jus",
        "os",
        "noix",
        "lieu",
        "lors",
    }

    # Groupement par clé normalisée, en gardant la forme la plus fréquente
    # comme libellé d'affichage.
    from collections import Counter, defaultdict

    groups = defaultdict(Counter)  # clé normalisée -> Counter des formes brutes
    for tok in tokens:
        key = _normalize_token(tok, keep_as_is)
        groups[key][tok] += 1

    rows = []
    for key, forms in groups.items():
        total = sum(forms.values())
        # Forme d'affichage = la plus fréquente parmi les variantes observées
        display = forms.most_common(1)[0][0]
        rows.append({"mot": display, "n": total})

    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def build_dataset_summary(
    df_pi: pd.DataFrame, df_full: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Construit une synthèse des sous-datasets utiles pour la suite.

    Si df_full est fourni, ajoute pour chaque sous-dataset des colonnes
    n_bulletins_ext et n_images_ext correspondant à la version étendue
    qui inclut les bulletins PEST_REPORT remplissant les mêmes critères
    techniques (inclusion sélective).
    """

    def n_b(df, mask_col):
        """Nombre de bulletins du sous-ensemble."""
        return int(df[mask_col].sum())

    def n_i(df, mask_col):
        """Nombre d'images cumulées du sous-ensemble."""
        return int(df.loc[df[mask_col], "n_images"].sum())

    rows = [
        {
            "subset": "PI_total",
            "definition": "Tous les bulletins du tunnel PEST_IDENTIFICATION",
            "n_bulletins": len(df_pi),
            "n_images": int(df_pi["n_images"].sum()),
            "usage": "Population principale de l’analyse qualité",
        },
        {
            "subset": "image_only",
            "definition": "Bulletins avec image mais sans texte exploitable",
            "n_bulletins": n_b(df_pi, "image_only"),
            "n_images": n_i(df_pi, "image_only"),
            "usage": "Vision seule / inspection qualité image",
        },
        {
            "subset": "text_only",
            "definition": "Bulletins avec texte exploitable mais sans image",
            "n_bulletins": n_b(df_pi, "text_only"),
            "n_images": n_i(df_pi, "text_only"),
            "usage": "NLP, extraction de symptômes, enrichissement base experte",
        },
        {
            "subset": "image_text",
            "definition": "Bulletins avec image + texte exploitable",
            "n_bulletins": n_b(df_pi, "image_text"),
            "n_images": n_i(df_pi, "image_text"),
            "usage": "Dataset multimodal non nécessairement annoté",
        },
        {
            "subset": "confirmed_only",
            "definition": "Bulletins avec diagnostic confirmé",
            "n_bulletins": n_b(df_pi, "confirmed_only"),
            "n_images": n_i(df_pi, "confirmed_only"),
            "usage": "Supervision forte mais volume faible",
        },
        {
            "subset": "complet",
            "definition": "Image + texte exploitable + diagnostic confirmé",
            "n_bulletins": n_b(df_pi, "complet"),
            "n_images": n_i(df_pi, "complet"),
            "usage": "Dataset multimodal supervisé idéal",
        },
        {
            "subset": "weak_labels",
            "definition": "Image + texte exploitable sans diagnostic confirmé",
            "n_bulletins": n_b(df_pi, "weak_label"),
            "n_images": n_i(df_pi, "weak_label"),
            "usage": "Pseudo-labeling, retrieval, NLI, annotation humaine priorisée",
        },
    ]

    out = pd.DataFrame(rows)
    out["pct_bulletins_PI"] = out["n_bulletins"].apply(lambda n: pct(n, len(df_pi)))

    # ───── Version étendue (inclusion sélective des PR) ─────
    # Les axes 1-4 restent calculés sur PI seul (les deux tunnels ont des
    # finalités fonctionnelles différentes, mélanger leurs taux de complétude
    # serait trompeur). Mais pour les sous-datasets exploitables (axe 5), on
    # inclut les bulletins PR qui remplissent les mêmes critères techniques :
    # un bulletin PR avec image + texte exploitable est une donnée multimodale
    # valide, peu importe son tunnel d'origine.
    if df_full is not None:
        n_bulletins_ext = []
        n_images_ext = []
        for r in rows:
            subset = r["subset"]
            if subset == "PI_total":
                # PI_total reste PI uniquement par définition
                n_bulletins_ext.append(r["n_bulletins"])
                n_images_ext.append(r["n_images"])
                continue
            # Mapping subset -> colonne de masque dans df_full
            mask_col = {
                "image_only": "image_only",
                "text_only": "text_only",
                "image_text": "image_text",
                "confirmed_only": "confirmed_only",
                "complet": "complet",
                "weak_labels": "weak_label",
            }[subset]
            n_bulletins_ext.append(n_b(df_full, mask_col))
            n_images_ext.append(n_i(df_full, mask_col))

        out["n_bulletins_ext"] = n_bulletins_ext
        out["n_images_ext"] = n_images_ext
        out["delta_bulletins"] = out["n_bulletins_ext"] - out["n_bulletins"]

    return out


def report(df_full: pd.DataFrame) -> str:
    """Affiche et retourne le rapport texte de l'analyse."""
    lines: list[str] = []

    def add(x: str = "") -> None:
        lines.append(x)
        print(x)

    df_pi = df_full[df_full["Type"] == "PEST_IDENTIFICATION"].copy()
    df_pr = df_full[df_full["Type"] == "PEST_REPORT"].copy()
    N = len(df_pi)

    # ------------------------------------------------------------------ #
    # Cadrage
    # ------------------------------------------------------------------ #
    section("CADRAGE")
    add(f"Total bulletins CSV                  : {len(df_full)}")
    add(f"PEST_IDENTIFICATION analysés         : {N}")
    add(f"PEST_REPORT en annexe                : {len(df_pr)}")
    add("Choix méthodologique : l'analyse qualité se concentre sur PI.")
    add("PR est conservé en annexe car il s'agit d'une déclaration directe.")

    # ------------------------------------------------------------------ #
    # Axe 1 — Complétude
    # ------------------------------------------------------------------ #
    section("AXE 1 — COMPLÉTUDE DES CHAMPS")
    fields = [
        ("Espèce résolue", df_pi["nom_espece"].notna()),
        ("Parties infectées renseignées", df_pi["parties_filled"]),
        ("Description renseignée", df_pi["desc_filled"]),
        ("Image associée", df_pi["has_image"]),
        ("Diagnostic confirmé", df_pi["avec_diagnostic"]),
        ("Texte exploitable", df_pi["exploitable_texte"]),
        ("Complet image + texte + diagnostic", df_pi["complet"]),
    ]
    for label, mask in fields:
        n = int(mask.sum())
        add(f"{label:38s}: {n:3d} / {N} ({pct(n, N):5.1f} %)")

    add("")
    add("Combinaisons principales :")
    combos = [
        ("image seule", df_pi["image_only"]),
        ("texte seul exploitable", df_pi["text_only"]),
        ("image + texte exploitable", df_pi["image_text"]),
        ("image + texte + diagnostic confirmé", df_pi["complet"]),
        ("image + texte sans diagnostic confirmé", df_pi["weak_label"]),
    ]
    for label, mask in combos:
        n_b = int(mask.sum())
        n_i = int(df_pi.loc[mask, "n_images"].sum())
        add(f"{label:42s}: {n_b:3d} bulletins | {n_i:3d} images")

    # ------------------------------------------------------------------ #
    # Axe 2 — Texte
    # ------------------------------------------------------------------ #
    section("AXE 2 — ANALYSE DU TEXTE UTILISATEUR")
    n_desc = int(df_pi["desc_filled"].sum())
    add(
        f"Descriptions renseignées              : {n_desc} / {N} ({pct(n_desc, N):.1f} %)"
    )
    if n_desc:
        stats = (
            df_pi.loc[df_pi["desc_filled"], ["desc_len_chars", "desc_len_words"]]
            .describe()
            .round(1)
        )
        add("")
        add("Longueur des descriptions :")
        add(stats.to_string())

        add("")
        add("Qualité texte simple :")
        tq = value_counts_with_pct(df_pi["text_quality"])
        add(tq.to_string(index=False))

        tokens = tokenize_descriptions(df_pi)
        if not tokens.empty:
            add("")
            add("Top 20 mots fréquents (normalisés : accents/casse + pluriel naïf) :")
            add(tokens.head(20).to_string(index=False))

    # NB : l'analyse des images en tant que telles (qualité photographique,
    # cadrage, contenu visuel, etc.) n'est pas du ressort de ce script. Le
    # comptage et la traçabilité des images sont effectués par match_images.py
    # et résumés dans matching_report.txt. Le présent script se concentre sur
    # la complétude, le texte et les diagnostics.

    # ------------------------------------------------------------------ #
    # Axe 3 — Diagnostics confirmés (anciennement Axe 4)
    # ------------------------------------------------------------------ #
    section("AXE 3 — ANALYSE DES DIAGNOSTICS CONFIRMÉS")
    df_conf = df_pi[df_pi["avec_diagnostic"]].copy()
    add(
        f"PI avec diagnostic confirmé            : {len(df_conf)} / {N} ({pct(len(df_conf), N):.1f} %)"
    )
    if not df_conf.empty:
        add("")
        add("Répartition Pest Type :")
        add(value_counts_with_pct(df_conf["Pest Type"]).to_string(index=False))

        add("")
        add("Bioagresseurs confirmés :")
        diag = (
            df_conf.groupby(["Pest Type", "nom_bioagresseur"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        add(diag.to_string(index=False))

        add("")
        add("Espèces concernées par diagnostics confirmés :")
        add(df_conf["nom_espece"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    # Axe 4 — Sous-datasets (anciennement Axe 5)
    # ------------------------------------------------------------------ #
    section("AXE 4 — SOUS-DATASETS POUR LA SUITE")
    add("Versions PI (analyse principale) et PI+PR (inclusion sélective)")
    add("voir Annexe pour la justification méthodologique.")
    add("")
    add("⚠ Attention sur confirmed_only : 'CONFIRMED' n'a pas la même sémantique")
    add("  - Pour PI : l'utilisateur a validé une suggestion algorithmique")
    add("  - Pour PR : état par défaut (déclaration directe par l'utilisateur)")
    add("  Le gain apparent sur ce sous-dataset reflète cette différence.")
    add("")
    summary = build_dataset_summary(df_pi, df_full=df_full)
    add(summary.to_string(index=False))

    # Mise en évidence des PR réintégrés
    if "delta_bulletins" in summary.columns:
        gains = summary[summary["delta_bulletins"] > 0]
        if not gains.empty:
            add("")
            add("Gains apportés par l'inclusion sélective des PR :")
            for _, r in gains.iterrows():
                add(
                    f"  + {r['subset']:15s} : +{int(r['delta_bulletins'])} bulletin(s) "
                    f"(PI:{int(r['n_bulletins'])} → PI+PR:{int(r['n_bulletins_ext'])})"
                )

    # ------------------------------------------------------------------ #
    # Annexe PR
    # ------------------------------------------------------------------ #
    section("ANNEXE — PEST_REPORT")
    add(f"PEST_REPORT total                      : {len(df_pr)}")
    if not df_pr.empty:
        add(
            f"PR avec image                          : {int(df_pr['has_image'].sum())} / {len(df_pr)}"
        )
        add(
            f"PR avec bioagresseur déclaré           : {int(df_pr['nom_bioagresseur'].notna().sum())} / {len(df_pr)}"
        )
        add("")
        add("Bioagresseurs déclarés PR :")
        pr_diag = (
            df_pr.groupby(["Pest Type", "nom_bioagresseur"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        add(pr_diag.to_string(index=False))

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def add_bar_labels(ax, total: int | None = None) -> None:
    """Ajoute les effectifs et pourcentages sur les barres."""
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height):
            continue
        if total:
            label = f"{int(height)}\n({height/total*100:.1f}%)"
        else:
            label = f"{int(height)}"
        ax.annotate(
            label,
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_all(df_full: pd.DataFrame, out_dir: Path) -> None:
    """Génère les figures principales des 5 axes."""
    out_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = df_full[df_full["Type"] == "PEST_IDENTIFICATION"].copy()
    N = len(df)

    # Axe 1 — complétude
    completeness = pd.Series(
        {
            "Espèce": df["nom_espece"].notna().sum(),
            "Parties": df["parties_filled"].sum(),
            "Description": df["desc_filled"].sum(),
            "Image": df["has_image"].sum(),
            "Diagnostic confirmé": df["avec_diagnostic"].sum(),
            "Texte exploitable": df["exploitable_texte"].sum(),
            "Complet": df["complet"].sum(),
        }
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    completeness.plot.bar(ax=ax, color="steelblue")
    ax.set_title(f"Axe 1 — Complétude des champs PI (N={N})")
    ax.set_ylabel("Nombre de bulletins")
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    add_bar_labels(ax, total=N)
    fig.tight_layout()
    fig.savefig(out_dir / "01_completude_PI.png", dpi=140)
    plt.close(fig)

    # Axe 2 — qualité texte
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["missing", "too_short", "short", "medium", "long"]
    counts = df["text_quality"].value_counts().reindex(order).fillna(0)
    counts.plot.bar(ax=ax, color="mediumseagreen")
    ax.set_title("Axe 2 — Qualité simple des descriptions")
    ax.set_ylabel("Nombre de bulletins")
    ax.tick_params(axis="x", rotation=0)
    add_bar_labels(ax, total=N)
    fig.tight_layout()
    fig.savefig(out_dir / "02_qualite_texte_PI.png", dpi=140)
    plt.close(fig)

    # Axe 2 — longueur des descriptions
    if df["desc_filled"].any():
        fig, ax = plt.subplots(figsize=(8, 4))
        df.loc[df["desc_filled"], "desc_len_words"].plot.hist(ax=ax, bins=15)
        ax.set_title("Axe 2 — Longueur des descriptions renseignées")
        ax.set_xlabel("Nombre de mots")
        fig.tight_layout()
        fig.savefig(out_dir / "03_longueur_descriptions_PI.png", dpi=140)
        plt.close(fig)

    # NB : pas de figure dédiée aux images dans ce script — c'est le rôle
    # de match_images.py / matching_report.txt.

    # Axe 4 — diagnostics (anciennement Axe 4 dans la numérotation précédente)
    df_conf = df[df["avec_diagnostic"]].dropna(subset=["nom_bioagresseur"])
    if not df_conf.empty:
        diag = (
            df_conf.groupby(["Pest Type", "nom_bioagresseur"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(11, 4))
        colors = (
            diag["Pest Type"]
            .map({"DISEASE": "crimson", "INSECT": "darkorange"})
            .fillna("gray")
        )
        ax.bar(diag["nom_bioagresseur"], diag["n"], color=colors)
        ax.set_title("Axe 4 — Bioagresseurs confirmés PI")
        ax.set_ylabel("Nombre de bulletins")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        add_bar_labels(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "05_bioagresseurs_confirmes_PI.png", dpi=140)
        plt.close(fig)

    # Axe 5 — sous-datasets (PI seul vs étendue PI+PR pour comparaison)
    summary = build_dataset_summary(df, df_full=df_full)
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = summary[summary["subset"] != "PI_total"].set_index("subset")
    if "n_bulletins_ext" in plot_df.columns:
        # Double barre PI / étendue
        plot_df[["n_bulletins", "n_bulletins_ext"]].plot.bar(
            ax=ax, color=["mediumpurple", "darkviolet"], width=0.75
        )
        ax.legend(["PI seul", "PI + PR sélectifs"], loc="upper right")
    else:
        plot_df["n_bulletins"].plot.bar(ax=ax, color="mediumpurple")
    ax.set_title("Axe 5 — Sous-datasets disponibles pour la suite")
    ax.set_ylabel("Nombre de bulletins")
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_dir / "06_sous_datasets_PI.png", dpi=140)
    plt.close(fig)

    # Espèces
    if df["nom_espece"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 5))
        df["nom_espece"].value_counts().head(15).plot.bar(ax=ax, color="teal")
        ax.set_title(f"Top 15 espèces — PI (N={N})")
        ax.set_ylabel("Nombre de bulletins")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        add_bar_labels(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "07_especes_PI.png", dpi=140)
        plt.close(fig)

    print(f"\nFigures écrites dans : {out_dir.resolve()}")


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #


def export(df_full: pd.DataFrame) -> None:
    """Exporte les fichiers CSV utiles pour la suite."""
    df = df_full[df_full["Type"] == "PEST_IDENTIFICATION"].copy()

    df.to_csv("PI_bulletins_enrichis.csv", index=False)
    df[df["exploitable_texte"]].to_csv("PI_exploitables_texte.csv", index=False)
    df[df["complet"]].to_csv("PI_complets.csv", index=False)

    # Sous-datasets de travail (PI seul)
    df[df["image_only"]].to_csv("PI_image_only.csv", index=False)
    df[df["text_only"]].to_csv("PI_text_only.csv", index=False)
    df[df["image_text"]].to_csv("PI_image_text.csv", index=False)
    df[df["confirmed_only"]].to_csv("PI_confirmed_only.csv", index=False)
    df[df["weak_label"]].to_csv("PI_weak_labels.csv", index=False)

    # Fichiers de synthèse
    text_cols = [
        "ID",
        "Status",
        "Type",
        "nom_espece",
        "parties_clean",
        "description_clean",
        "desc_len_chars",
        "desc_len_words",
        "text_quality",
        "has_image",
        "n_images",
        "avec_diagnostic",
        "nom_bioagresseur",
        "Pest Type",
        "exploitable_texte",
        "image_text",
        "weak_label",
        "complet",
    ]
    text_cols = [c for c in text_cols if c in df.columns]
    df[text_cols].to_csv("PI_text_quality.csv", index=False)

    summary = build_dataset_summary(df, df_full=df_full)
    summary.to_csv("PI_dataset_summary.csv", index=False)

    # ───── Sous-datasets ÉTENDUS (PI + PR sélectifs) ─────
    # Inclusion sélective des PR remplissant les mêmes critères techniques :
    # on garde la rigueur de l'analyse séparée pour les axes 1-4 mais on
    # utilise le maximum de données pour les sous-datasets exploitables.
    df_ext = df_full.copy()
    df_ext[df_ext["image_text"]].to_csv("EXT_image_text.csv", index=False)
    df_ext[df_ext["complet"]].to_csv("EXT_complet.csv", index=False)
    df_ext[df_ext["weak_label"]].to_csv("EXT_weak_labels.csv", index=False)
    df_ext[df_ext["confirmed_only"]].to_csv("EXT_confirmed_only.csv", index=False)

    print("\nExports écrits :")
    pi_files = [
        "PI_bulletins_enrichis.csv",
        "PI_exploitables_texte.csv",
        "PI_complets.csv",
        "PI_image_only.csv",
        "PI_text_only.csv",
        "PI_image_text.csv",
        "PI_confirmed_only.csv",
        "PI_weak_labels.csv",
        "PI_text_quality.csv",
        "PI_dataset_summary.csv",
    ]
    ext_files = [
        "EXT_image_text.csv",
        "EXT_complet.csv",
        "EXT_weak_labels.csv",
        "EXT_confirmed_only.csv",
    ]
    print("  Sous-datasets PI seul :")
    for fname in pi_files:
        print(f"    - {fname}")
    print("  Sous-datasets étendus (PI + PR sélectifs) :")
    for fname in ext_files:
        print(f"    - {fname}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    csv_path = find_csv()
    print(f"Lecture CSV : {csv_path}")

    df = load_csv(csv_path)
    especes, diseases, pests = load_lookups(Path("tables_de_liaison"))
    df = enrich(df, especes, diseases, pests)

    report_text = report(df)
    Path("rapport_analyse_PI.txt").write_text(report_text, encoding="utf-8")
    print("\nRapport écrit : rapport_analyse_PI.txt")

    plot_all(df, Path("figures"))
    export(df)


if __name__ == "__main__":
    main()
