"""Construit les contextes experimentaux BSV du pipeline.

Les proportions de scenarios sont un choix de construction semi-synthetique ;
elles ne representent pas leur frequence naturelle. Le scenario est tire avant
toute utilisation des predictions CNN. La vraie classe sert ensuite uniquement
a rechercher un contexte conforme au scenario tire.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import moteur_compatibilite as moteur
from pipeline_config import RANDOM_SEED


PHASE3_DIR = Path("data/processed/moteur/phase3")
MANIFEST_CSV = PHASE3_DIR / "phase3_test_manifest_masked.csv"
GROUND_TRUTH_CSV = PHASE3_DIR / "ground_truth.csv"
OUT_METADATA_CSV = PHASE3_DIR / "phase3_test_with_metadata.csv"
OUT_ASSIGNMENTS_CSV = PHASE3_DIR / "phase3_scenario_assignments.csv"
OUT_COVERAGE_CSV = PHASE3_DIR / "phase3_bsv_metadata_coverage.csv"
DEFAULT_LOG_FILE = Path("outputs/moteur/logs/bsv_compatible_metadata.log")

SANS_BULLETIN_RATE_COVERED = 0.18
DEFAULT_PROPORTIONS = {
    "favorable": 0.40,
    "non_favorable": 0.40,
    "insuffisant": 0.20,
}
SCENARIOS = tuple(DEFAULT_PROPORTIONS)

DEPARTEMENTS_PAR_REGION = {
    "Grand Est": "Bas-Rhin",
    "Pays de la Loire": "Maine-et-Loire",
    "Provence-Alpes-Cote d'Azur": "Bouches-du-Rhone",
    "Provence-Alpes-CÃ´te d'Azur": "Bouches-du-Rhone",
}

# Fichier aveugle utilise par l'inference. Il ne contient ni verite terrain,
# ni scenario experimental, ni statut de realisation.
BLIND_COLUMNS = [
    "observation_id",
    "image_path",
    "label_masked",
    "date_observation",
    "region",
    "departement",
]

ASSIGNMENT_COLUMNS = [
    "observation_id",
    "true_label",
    "scenario_assigne",
    "scenario_realise",
    "scenario_realisable",
    "raison_non_realisation",
    "cause_insuffisance",
    "classe_couverte_bsv",
    "context_id",
    "bulletin_id_assigne",
    "date_observation",
    "region",
    "departement",
    "seed",
]


@dataclass(frozen=True)
class ContexteBSV:
    scenario: str
    region: str
    date_observation: date
    bulletin_id: str

    @property
    def context_id(self) -> str:
        bulletin = self.bulletin_id or "sans_bulletin"
        return f"{self.region}|{self.date_observation.isoformat()}|{bulletin}"


def valider_proportions(proportions: dict[str, float]) -> None:
    if set(proportions) != set(SCENARIOS):
        raise ValueError(f"Les scenarios attendus sont: {', '.join(SCENARIOS)}")
    if any(value < 0 for value in proportions.values()):
        raise ValueError("Les proportions doivent etre positives")
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("La somme des proportions doit etre egale a 1")


def calculer_quotas(effectif: int, proportions: dict[str, float]) -> dict[str, int]:
    """Calcule des quotas entiers dont la somme vaut exactement effectif."""

    valider_proportions(proportions)
    valeurs = {nom: effectif * proportion for nom, proportion in proportions.items()}
    quotas = {nom: int(valeur) for nom, valeur in valeurs.items()}
    reste = effectif - sum(quotas.values())
    priorite = sorted(
        SCENARIOS,
        key=lambda nom: (valeurs[nom] - quotas[nom], -SCENARIOS.index(nom)),
        reverse=True,
    )
    for nom in priorite[:reste]:
        quotas[nom] += 1
    return quotas


def tirer_scenarios(
    effectif: int,
    seed: int = RANDOM_SEED,
    proportions: dict[str, float] | None = None,
) -> list[str]:
    """Tire les scenarios sans consulter classe, prediction ou score CNN."""

    proportions = proportions or DEFAULT_PROPORTIONS
    quotas = calculer_quotas(effectif, proportions)
    scenarios = [nom for nom in SCENARIOS for _ in range(quotas[nom])]
    random.Random(seed).shuffle(scenarios)
    return scenarios


def classe_est_couverte(classe: str, data: moteur.EngineData) -> bool:
    if classe == "tomato_healthy":
        return False
    ref = data.referential_by_class.get(classe, {})
    agent = moteur.get_column(ref, "agent") if ref else ""
    return classe in data.covered_classes or bool(
        agent and moteur.normalize(agent) in data.covered_agents
    )


def construire_catalogue_contextes(
    data: moteur.EngineData,
) -> dict[str, dict[str, list[ContexteBSV]]]:
    """Indexe les bulletins que le moteur jugera favorables/non favorables."""

    catalogue: dict[str, dict[str, list[ContexteBSV]]] = defaultdict(
        lambda: {"favorable": [], "non_favorable": []}
    )
    classes = sorted(data.referential_by_class)

    for bulletin in data.bulletins:
        bulletin_id = str(bulletin["bulletin_id"])
        region = str(bulletin["region"])
        jour = bulletin["date_bulletin_parsed"]
        # Plusieurs numeros peuvent partager une date. Le moteur retient le
        # dernier dans son index regional : les autres ne sont pas assignables.
        bulletin_retenu = moteur.trouver_bulletin_retenu(
            data.bulletins_by_region, region, jour
        )
        if not bulletin_retenu or bulletin_retenu["bulletin_id"] != bulletin_id:
            continue
        rows = data.observations_by_bulletin.get(bulletin_id, [])
        tomate_directe = [
            row for row in rows if moteur.est_observation_tomate_directe(row)
        ]
        tomate_compatible = [
            row for row in rows if moteur.est_signal_tomate_compatible(row)
        ]
        bulletin_informatif = bool(tomate_directe or tomate_compatible)

        for classe in classes:
            if classe == "tomato_healthy" or not classe_est_couverte(classe, data):
                continue
            dans_periode, _, _ = moteur.expected_window_for_prediction(jour, classe)
            if not dans_periode:
                continue

            ref = data.referential_by_class.get(classe, {})
            agent = moteur.get_column(ref, "agent") if ref else ""
            agent_rows = moteur.chercher_dans_bsv_resolved(rows, classe, agent)
            contexte = ContexteBSV(
                scenario="favorable" if agent_rows else "non_favorable",
                region=region,
                date_observation=jour,
                bulletin_id=bulletin_id,
            )
            if agent_rows:
                catalogue[classe]["favorable"].append(contexte)
            elif bulletin_informatif:
                catalogue[classe]["non_favorable"].append(contexte)

    return dict(catalogue)


def construire_catalogue_insuffisance(
    data: moteur.EngineData,
) -> dict[str, dict[str, list[ContexteBSV]]]:
    """Construit des contextes dont la cause d'insuffisance est controlee."""

    catalogue: dict[str, dict[str, list[ContexteBSV]]] = defaultdict(
        lambda: {"sans_bulletin": [], "hors_periode_litterature": []}
    )
    regions = sorted({str(row["region"]) for row in data.bulletins})
    years = sorted({row["date_bulletin_parsed"].year for row in data.bulletins})
    classes_couvertes = sorted(
        classe
        for classe in data.referential_by_class
        if classe_est_couverte(classe, data)
    )
    for classe in classes_couvertes:
        for region in regions:
            for year in years:
                jour = date(year, 1, 1)
                while jour.year == year:
                    dans_periode, _, _ = moteur.expected_window_for_prediction(
                        jour, classe
                    )
                    bulletin = moteur.trouver_bulletin_retenu(
                        data.bulletins_by_region,
                        region,
                        jour,
                        max_age_days=moteur.MAX_BULLETIN_AGE_DAYS,
                    )
                    if dans_periode and bulletin is None:
                        catalogue[classe]["sans_bulletin"].append(
                            ContexteBSV("insuffisant", region, jour, "")
                        )
                    elif not dans_periode:
                        catalogue[classe]["hors_periode_litterature"].append(
                            ContexteBSV("insuffisant", region, jour, "")
                        )
                    jour += timedelta(days=1)
    return dict(catalogue)


def contexte_non_couvert(data: moteur.EngineData, rng: random.Random) -> ContexteBSV:
    """Utilise un bulletin reel afin que le moteur atteigne le test non_couvert."""

    bulletin = rng.choice(data.bulletins)
    return ContexteBSV(
        "insuffisant",
        str(bulletin["region"]),
        bulletin["date_bulletin_parsed"],
        str(bulletin["bulletin_id"]),
    )


def contexte_sans_bulletin_generique(
    premiers_bulletins: dict[str, date], rng: random.Random
) -> ContexteBSV:
    region = rng.choice(sorted(premiers_bulletins))
    return ContexteBSV(
        "insuffisant", region, premiers_bulletins[region] - timedelta(days=1), ""
    )


def construire_lignes_metadata(
    df_source: pd.DataFrame,
    data: moteur.EngineData,
    seed: int = RANDOM_SEED,
    proportions: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Produit une ligne auditable pour chaque image, sans suppression."""

    required = {"observation_id", "image_path", "label_masked", "true_label"}
    missing = required - set(df_source.columns)
    if missing:
        raise ValueError(f"Colonnes source manquantes: {', '.join(sorted(missing))}")
    if df_source["observation_id"].duplicated().any():
        raise ValueError("observation_id doit etre unique")

    rng = random.Random(seed)
    scenarios = tirer_scenarios(len(df_source), seed, proportions)
    catalogue = construire_catalogue_contextes(data)
    catalogue_insuffisance = construire_catalogue_insuffisance(data)
    premiers_bulletins: dict[str, date] = {}
    for bulletin in data.bulletins:
        region = str(bulletin["region"])
        jour = bulletin["date_bulletin_parsed"]
        premiers_bulletins[region] = min(jour, premiers_bulletins.get(region, jour))
    if not premiers_bulletins:
        raise ValueError("Aucune region disponible dans les bulletins BSV")

    # Parmi les insuffisants des classes couvertes, on conserve le partage
    # historique 18/82 entre sans bulletin et hors periode. L'affectation est
    # tiree avec une graine dediee et ne consulte jamais le CNN.
    covered_insufficient_positions = [
        position
        for position, ((_, row), scenario) in enumerate(
            zip(df_source.iterrows(), scenarios)
        )
        if scenario == "insuffisant"
        and classe_est_couverte(str(row["true_label"]), data)
    ]
    n_sans_bulletin = round(
        len(covered_insufficient_positions) * SANS_BULLETIN_RATE_COVERED
    )
    causes_couvertes = ["sans_bulletin"] * n_sans_bulletin + [
        "hors_periode_litterature"
    ] * (len(covered_insufficient_positions) - n_sans_bulletin)
    random.Random(seed + 1).shuffle(causes_couvertes)
    cause_by_position = dict(zip(covered_insufficient_positions, causes_couvertes))
    regions_sans = sorted(premiers_bulletins)
    rng_regions = random.Random(seed + 2)
    rng_regions.shuffle(regions_sans)
    sans_region_index = 0

    lignes: list[dict[str, Any]] = []
    for position, ((_, row), scenario_assigne) in enumerate(
        zip(df_source.iterrows(), scenarios)
    ):
        classe = str(row["true_label"])
        couverte = classe_est_couverte(classe, data)
        candidats = catalogue.get(classe, {}).get(scenario_assigne, [])
        cause_insuffisance = ""

        if scenario_assigne == "insuffisant":
            if couverte:
                cause_insuffisance = cause_by_position[position]
                candidats_insuffisants = catalogue_insuffisance[classe][
                    cause_insuffisance
                ]
                if cause_insuffisance == "sans_bulletin":
                    region_cible = regions_sans[sans_region_index % len(regions_sans)]
                    sans_region_index += 1
                    candidats_region = [
                        item
                        for item in candidats_insuffisants
                        if item.region == region_cible
                    ]
                    candidats_insuffisants = candidats_region or candidats_insuffisants
                if not candidats_insuffisants:
                    raise ValueError(
                        f"Aucun contexte {cause_insuffisance} pour {classe}"
                    )
                contexte = rng.choice(candidats_insuffisants)
            elif classe == "tomato_healthy":
                cause_insuffisance = "sans_bulletin"
                contexte = contexte_sans_bulletin_generique(premiers_bulletins, rng)
            else:
                cause_insuffisance = "classe_non_couverte_bsv"
                contexte = contexte_non_couvert(data, rng)
            realisable = True
            raison = ""
        elif candidats:
            contexte = rng.choice(candidats)
            realisable = True
            raison = ""
        else:
            if not couverte and classe != "tomato_healthy":
                cause_insuffisance = "classe_non_couverte_bsv"
                contexte = contexte_non_couvert(data, rng)
            elif classe == "tomato_healthy":
                cause_insuffisance = "sans_bulletin"
                contexte = contexte_sans_bulletin_generique(premiers_bulletins, rng)
            else:
                cause_insuffisance = "hors_periode_litterature"
                candidats_insuffisants = catalogue_insuffisance[classe][
                    cause_insuffisance
                ]
                if not candidats_insuffisants:
                    raise ValueError(f"Aucun contexte hors periode pour {classe}")
                contexte = rng.choice(candidats_insuffisants)
            realisable = False
            raison = (
                "classe_non_couverte_bsv"
                if not couverte
                else f"aucun_contexte_{scenario_assigne}_disponible"
            )

        ligne = row.to_dict()
        ligne.update(
            {
                "scenario_assigne": scenario_assigne,
                "scenario_realise": contexte.scenario,
                "scenario_realisable": realisable,
                "raison_non_realisation": raison,
                "cause_insuffisance": cause_insuffisance,
                "classe_couverte_bsv": couverte,
                "context_id": contexte.context_id,
                "bulletin_id_assigne": contexte.bulletin_id,
                "date_observation": contexte.date_observation.isoformat(),
                "region": contexte.region,
                "departement": DEPARTEMENTS_PAR_REGION.get(contexte.region, ""),
                "seed": seed,
            }
        )
        lignes.append(ligne)

    resultat = pd.DataFrame(lignes)
    if len(resultat) != len(df_source):
        raise AssertionError("Des images ont ete perdues pendant la construction")
    return resultat


def construire_table_couverture(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby(
            [
                "true_label",
                "classe_couverte_bsv",
                "scenario_assigne",
                "scenario_realise",
            ],
            dropna=False,
        )
        .agg(
            nb_images=("observation_id", "size"),
            nb_realisables=("scenario_realisable", "sum"),
        )
        .reset_index()
    )


def configurer_log(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("add_bsv_compatible_metadata")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--favorable-rate", type=float, default=0.40)
    parser.add_argument("--non-favorable-rate", type=float, default=0.40)
    parser.add_argument("--insufficient-rate", type=float, default=0.20)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configurer_log(args.log_file)
    proportions = {
        "favorable": args.favorable_rate,
        "non_favorable": args.non_favorable_rate,
        "insuffisant": args.insufficient_rate,
    }
    valider_proportions(proportions)

    manifest = pd.read_csv(MANIFEST_CSV)
    ground_truth = pd.read_csv(GROUND_TRUTH_CSV)
    if len(manifest) != len(ground_truth):
        raise ValueError("Manifest et ground truth n'ont pas le meme effectif")
    source = manifest.merge(
        ground_truth, on="observation_id", how="left", validate="one_to_one"
    )
    if source["true_label"].isna().any():
        raise ValueError("Certaines images n'ont pas de verite terrain")

    data = moteur.charger_donnees()
    assignments = construire_lignes_metadata(source, data, args.seed, proportions)
    blind = assignments[BLIND_COLUMNS].copy()
    audit = assignments[ASSIGNMENT_COLUMNS].copy()
    coverage = construire_table_couverture(assignments)

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    blind.to_csv(OUT_METADATA_CSV, index=False, encoding="utf-8-sig")
    audit.to_csv(OUT_ASSIGNMENTS_CSV, index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT_COVERAGE_CSV, index=False, encoding="utf-8-sig")

    logger.info("Images conservees: %s/%s", len(blind), len(source))
    logger.info("Scenarios demandes: %s", dict(Counter(audit["scenario_assigne"])))
    logger.info("Scenarios realises: %s", dict(Counter(audit["scenario_realise"])))
    logger.info("Scenarios realisables: %s", int(audit["scenario_realisable"].sum()))
    logger.info("Seed: %s", args.seed)
    logger.info("Manifest aveugle: %s", OUT_METADATA_CSV)
    logger.info("Table experimentale: %s", OUT_ASSIGNMENTS_CSV)


if __name__ == "__main__":
    main()
