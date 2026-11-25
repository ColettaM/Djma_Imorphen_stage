"""Prepare les donnees et realise les analyses statistiques de la Phase 3.

Objectif de l'analyse
---------------------
Le script cherche principalement a savoir si le verdict du moteur apporte
une information sur la justesse du Top-1 qui n'est pas deja contenue dans le
score de confiance du CNN. Il ne refait ni l'inference CNN ni le raisonnement
du moteur : il rapproche leurs sorties de la table experimentale et construit
une ligne d'analyse par image.

Le Top-1 est defini exclusivement par le premier element de ``top_k`` dans la
requete aveugle. Des controles bloquants verifient notamment les identifiants,
l'ordre et les scores des predictions, l'absence de fuite de verite terrain,
les contextes, les verdicts et les valeurs indispensables. Toute anomalie
d'integrite leve une exception avant l'ecriture des resultats.

Tests et analyses realises
--------------------------
1. Description des populations et de leur selection :
   jeu complet, classes couvertes, scenarios realisables et jeu principal.
2. Association brute entre justesse CNN et verdict du moteur :
   tableau 2 x 2 et test du khi-deux, ou test exact de Fisher lorsque les
   effectifs attendus sont trop faibles. Cette analyse ne tient pas compte du
   score CNN.
3. Deux regressions logistiques emboitees sur les memes observations :

   M0 : justesse ~ logit(score CNN)
   M1 : justesse ~ logit(score CNN) + verdict du moteur

   Les scores sont bornes dans [1e-6, 1 - 1e-6] avant leur transformation
   logit. Le test du rapport de vraisemblance (LRT) compare M0 et M1 sous
   l'hypothese conventionnelle d'independance des observations.
4. Test de Wald avec covariance robuste groupee par bulletin :
   il teste le coefficient du verdict en tenant compte de la dependance
   possible entre les images rattachees au meme bulletin. Si un bulletin est
   absent, ``context_id`` sert uniquement de groupe de repli. Ce test differe
   du LRT : il corrige l'inference sur le coefficient, mais ne remplace pas les
   log-vraisemblances du LRT par des erreurs standards groupees.
5. Odds ratio ajuste du verdict :
   exp(beta_verdict), avec intervalle conventionnel dans la table des
   coefficients, intervalle robuste groupe dans la table de comparaison et
   intervalle percentile obtenu par bootstrap groupe.
6. Bootstrap groupe par bulletin :
   les bulletins entiers sont tires avec remise, puis M0 et M1 sont reajustes.
   Il fournit la distribution empirique du coefficient du verdict, de son odds
   ratio et du LRT, ainsi qu'un IC percentile a 95 % de l'odds ratio.
7. Validation croisee groupee par bulletin :
   un bulletin ne peut jamais etre partage entre apprentissage et validation.
   M0 et M1 utilisent exactement les memes plis. Les log-loss et scores de
   Brier mesurent la qualite predictive hors echantillon ; une difference
   M1 - M0 negative favorise M1.
8. Diagnostic de forme fonctionnelle :
   les taux et log-odds empiriques de justesse sont resumes par quantiles du
   score afin d'evaluer la linearite supposee avec le logit du score.
9. Diagnostics de faisabilite, separation et coherence des scenarios
   insuffisants.

Principales sorties produites
-----------------------------
``phase3_analysis_full.csv``
    Toutes les images, avant selection analytique.
``phase3_analysis_covered.csv``
    Images dont la vraie classe est couverte par les BSV.
``phase3_analysis_primary.csv``
    Population utilisee pour les tests : classe couverte, scenario realisable
    et verdict Top-1 decidable.
``phase3_selection_summary.csv`` et ``phase3_scenario_summary.csv``
    Entonnoir de selection et realisation des scenarios.
``phase3_contingency_top1.csv``
    Association brute justesse-verdict, sans ajustement sur le score CNN.
``phase3_logistic_coefficients.csv``
    Coefficients, erreurs standards, odds ratios et IC conventionnels de M0
    et M1.
``phase3_logistic_comparison.csv``
    Resultat central : LRT M0 contre M1, OR ajuste du verdict et test de Wald
    avec intervalle robuste groupe par bulletin.
``phase3_logistic_predictions.csv``
    Variables transformees et probabilites ajustees par M0 et M1 pour chaque
    observation.
``phase3_grouped_bootstrap.csv``
    Une ligne par replication reussie avec coefficient, OR et LRT.
``phase3_grouped_bootstrap_summary.csv``
    IC percentile groupe de l'OR, probabilite bilaterale empirique et
    diagnostics de convergence du bootstrap.
``phase3_predicted_class_diagnostics.csv``
    Effectifs, erreurs et repartition des verdicts parmi les erreurs pour
    chaque classe predite. Les cellules nulles ou inferieures a cinq y sont
    signalees explicitement.
``phase3_class_adjusted_coefficients.csv``
    Coefficients de MC et M2, qui controlent tous deux la classe predite.
``phase3_class_adjusted_comparison.csv``
    Comparaisons MC-M2 (apport du verdict apres score et classe) et M1-M2
    (apport global de la classe), avec indication de la methode standard ou
    de Firth effectivement utilisee.
``phase3_firth_diagnostics.csv``
    Raisons objectives qui declenchent, ou non, le recours a Firth. Aucune
    interaction verdict-classe n'est estimee.
``phase3_grouped_cross_validation.csv``
    Log-loss et Brier par pli et par modele.
``phase3_grouped_cross_validation_summary.csv``
    Moyennes hors echantillon et differences M1 - M0.
``phase3_score_linearity.csv``
    Diagnostic descriptif de linearite par classes de score.
``phase3_integrity_checks.csv``, ``phase3_modeling_feasibility.csv`` et
``phase3_separation_diagnostics.csv``
    Controles d'integrite et limites statistiques a examiner avant
    interpretation.
``phase3_insufficiency_consistency.csv``
    Coherence entre la construction des scenarios insuffisants et le niveau
    de preuve effectivement renvoye par le moteur.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, chi2_contingency, fisher_exact, norm

from pipeline_config import EXPECTED_IMAGES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Entrees produites par les phases precedentes et repertoire des sorties
# figees. Ces chemins peuvent etre remplaces par les options de la CLI.
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT / "data/processed/moteur/phase3/phase3_scenario_assignments.csv"
)
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs/moteur/phase3/cnn_topk_predictions.json"
DEFAULT_QUERY = PROJECT_ROOT / "outputs/moteur/phase3/requete_phase3.json"
DEFAULT_VERDICTS = PROJECT_ROOT / "data/processed/moteur/bsv_compatibility_phase3.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/moteur/phase3/analysis"
DEFAULT_LOG_FILE = DEFAULT_OUTPUT_DIR / "phase3_prepare_analysis.log"

# Schema ferme de la requete aveugle. Toute cle supplementaire est refusee pour
# eviter une fuite de verite terrain ou d'information experimentale.
QUERY_KEYS = {"image_id", "date_observation", "region", "top_k"}
QUERY_PREDICTION_KEYS = {"classe", "score"}
FORBIDDEN_ENGINE_FIELDS = {
    "true_label",
    "scenario_assigne",
    "scenario_realise",
    "scenario_realisable",
    "raison_non_realisation",
    "cause_insuffisance",
    "classe_couverte_bsv",
    "context_id",
    "bulletin_id_assigne",
    "seed",
}
DECIDABLE_VERDICTS = {"Coherent", "Non_corrobore"}
SCORE_EPSILON = 1e-6

# Une valeur manquante dans l'une de ces colonnes rendrait l'analyse Top-1
# ambigue ou impossible ; le script s'arrete donc avant tout export.
CRITICAL_COLUMNS = [
    "image_id",
    "true_label",
    "scenario_assigne",
    "scenario_realise",
    "scenario_realisable",
    "classe_couverte_bsv",
    "context_id",
    "classe_cnn_top1",
    "score_cnn_top1",
    "verdict_bsv_top1",
    "niveau_preuve",
    "date_observation",
    "region",
]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    """Charge un fichier JSON et impose une liste d'objets comme racine."""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} doit contenir une liste JSON")
    return payload


def check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    detail: str,
) -> None:
    """Enregistre un controle d'integrite et interrompt le run s'il echoue."""

    checks.append({"check": name, "passed": bool(condition), "detail": detail})
    if not condition:
        raise ValueError(f"Controle bloquant echoue [{name}]: {detail}")


def validate_query_blinding(
    query: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    """Verifie que la requete moteur respecte strictement le schema aveugle."""

    # L'union detecte une cle interdite meme si elle n'apparait que sur une
    # seule image ou une seule prediction du Top-K.
    keys = set().union(*(set(row) for row in query)) if query else set()
    prediction_keys = (
        set().union(*(set(pred) for row in query for pred in row.get("top_k", [])))
        if query
        else set()
    )
    check(checks, "query_schema_closed", keys == QUERY_KEYS, f"cles={sorted(keys)}")
    check(
        checks,
        "query_prediction_schema_closed",
        prediction_keys == QUERY_PREDICTION_KEYS,
        f"cles={sorted(prediction_keys)}",
    )
    check(
        checks,
        "query_has_no_experimental_fields",
        not bool(keys & FORBIDDEN_ENGINE_FIELDS),
        f"champs_interdits={sorted(keys & FORBIDDEN_ENGINE_FIELDS)}",
    )
    empty = sum(not row.get("top_k") for row in query)
    check(checks, "query_topk_non_empty", empty == 0, f"top_k_vides={empty}")


def top1_from_query(query: list[dict[str, Any]]) -> pd.DataFrame:
    """Extrait le Top-1 selon l'ordre declare dans la requete aveugle.

    Aucun maximum de score n'est recalcule : le premier element de ``top_k``
    est la source d'autorite, conformement au protocole.
    """

    rows = []
    for position, item in enumerate(query, start=1):
        top1 = item["top_k"][0]
        rows.append(
            {
                "image_id": str(item["image_id"]),
                "date_observation_query": str(item["date_observation"]),
                "region_query": str(item["region"]),
                "classe_cnn_top1": str(top1["classe"]),
                "score_cnn_top1": float(top1["score"]),
                "query_position": position,
            }
        )
    return pd.DataFrame(rows)


def validate_prediction_coherence(
    query: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    """Compare la requete aux sorties CNN, classe, score et ordre compris."""

    prediction_map = {str(row.get("observation_id", "")): row for row in predictions}
    check(
        checks,
        "prediction_ids_unique",
        len(prediction_map) == len(predictions) and "" not in prediction_map,
        f"lignes={len(predictions)}, ids_uniques={len(prediction_map)}",
    )
    mismatches = 0
    failures = 0
    for item in query:
        record = prediction_map.get(str(item["image_id"]))
        if record is None:
            mismatches += 1
            continue
        failures += int(not record.get("inference_ok"))
        # On compare seulement les rangs effectivement transmis au moteur.
        declared = record.get("predictions", [])[: len(item["top_k"])]
        if len(declared) != len(item["top_k"]):
            mismatches += 1
            continue
        for left, right in zip(item["top_k"], declared):
            if str(left["classe"]) != str(right.get("label", right.get("classe", ""))):
                mismatches += 1
                break
            if not np.isclose(float(left["score"]), float(right["score"]), atol=1e-9):
                mismatches += 1
                break
    check(
        checks, "predictions_match_query_order", mismatches == 0, f"ecarts={mismatches}"
    )
    check(checks, "all_inferences_successful", failures == 0, f"echecs={failures}")


def prepare_top1_table(
    assignments: pd.DataFrame,
    predictions: list[dict[str, Any]],
    query: list[dict[str, Any]],
    verdicts: pd.DataFrame,
    *,
    expected_images: int = EXPECTED_IMAGES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit la table analytique Top-1 et execute les controles bloquants.

    La sortie ``full`` possede exactement une ligne par image. La seconde
    sortie contient la trace des controles passes avant sa construction.
    """

    checks: list[dict[str, Any]] = []
    validate_query_blinding(query, checks)
    validate_prediction_coherence(query, predictions, checks)

    top1 = top1_from_query(query)
    assignment_ids = set(assignments["observation_id"].astype(str))
    query_ids = set(top1["image_id"].astype(str))
    verdict_ids = set(verdicts["image_id"].astype(str))
    prediction_ids = {str(row["observation_id"]) for row in predictions}
    check(
        checks,
        "expected_image_count",
        len(query_ids) == expected_images,
        f"n={len(query_ids)}",
    )
    check(
        checks,
        "assignments_unique",
        not assignments["observation_id"].duplicated().any(),
        f"n={len(assignments)}",
    )
    check(
        checks, "top1_unique", not top1["image_id"].duplicated().any(), f"n={len(top1)}"
    )
    check(
        checks,
        "identifier_sets_identical",
        assignment_ids == query_ids == verdict_ids == prediction_ids,
        (
            f"assignments={len(assignment_ids)}, query={len(query_ids)}, "
            f"verdicts={len(verdict_ids)}, predictions={len(prediction_ids)}"
        ),
    )

    # Une paire image/classe doit correspondre a un seul verdict moteur, sinon
    # la jointure avec le Top-1 serait ambigue.
    verdict_key_counts = verdicts.groupby(["image_id", "classe_cnn_predite"]).size()
    duplicate_keys = int((verdict_key_counts > 1).sum())
    check(
        checks,
        "verdict_image_class_unique",
        duplicate_keys == 0,
        f"doublons={duplicate_keys}",
    )

    # Jointure du Top-1 declare avec le verdict portant sur la meme classe.
    top1_verdict = top1.merge(
        verdicts,
        left_on=["image_id", "classe_cnn_top1"],
        right_on=["image_id", "classe_cnn_predite"],
        how="left",
        validate="one_to_one",
    )
    missing_verdicts = int(top1_verdict["compatibilite"].isna().sum())
    check(
        checks,
        "one_top1_verdict_per_image",
        missing_verdicts == 0,
        f"manquants={missing_verdicts}",
    )
    score_mismatch = int(
        (
            ~np.isclose(
                top1_verdict["score_cnn_top1"], top1_verdict["score_cnn"], atol=1e-9
            )
        ).sum()
    )
    check(
        checks,
        "top1_scores_match_engine",
        score_mismatch == 0,
        f"ecarts={score_mismatch}",
    )
    context_mismatch = int(
        (
            (top1_verdict["date_observation_query"] != top1_verdict["date_observation"])
            | (top1_verdict["region_query"] != top1_verdict["region"])
        ).sum()
    )
    check(
        checks,
        "query_context_matches_engine",
        context_mismatch == 0,
        f"ecarts={context_mismatch}",
    )

    # La table d'assignation est la seule source autorisee pour la verite
    # terrain et les variables experimentales.
    assignment_view = assignments.rename(
        columns={"observation_id": "image_id", "bulletin_id_assigne": "bulletin_id"}
    )
    full = assignment_view.merge(
        top1_verdict, on="image_id", how="left", validate="one_to_one"
    )
    assignment_context_mismatch = int(
        (
            (full["date_observation_x"] != full["date_observation_query"])
            | (full["region_x"] != full["region_query"])
        ).sum()
    )
    check(
        checks,
        "assignment_context_matches_query",
        assignment_context_mismatch == 0,
        f"ecarts={assignment_context_mismatch}",
    )

    full = full.rename(
        columns={
            "date_observation_x": "date_observation",
            "region_x": "region",
            "compatibilite": "verdict_bsv_top1",
        }
    )
    full = full.drop(
        columns=[
            "date_observation_y",
            "region_y",
            "date_observation_query",
            "region_query",
            "classe_cnn_predite",
            "score_cnn",
        ],
        errors="ignore",
    )
    # La verite terrain n'est revelee qu'ici, apres le CNN et le moteur.
    full["cnn_correct"] = full["classe_cnn_top1"] == full["true_label"]
    insufficient = full["scenario_realise"] == "insuffisant"
    missing_causes = int(
        full.loc[insufficient, "cause_insuffisance"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    check(
        checks,
        "insufficient_scenarios_have_construction_cause",
        missing_causes == 0,
        f"manquants={missing_causes}",
    )
    unexpected_causes = int(
        full.loc[~insufficient, "cause_insuffisance"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    check(
        checks,
        "non_insufficient_scenarios_have_no_construction_cause",
        unexpected_causes == 0,
        f"causes_inattendues={unexpected_causes}",
    )
    expected_proof = {
        "classe_non_couverte_bsv": "non_couvert",
        "sans_bulletin": "sans_bulletin",
        "hors_periode_litterature": "hors_periode_litterature",
    }
    # La cause construite et la preuve moteur ne doivent obligatoirement
    # coincider que si le moteur evalue l'agent reel, donc si le CNN est juste.
    strict = insufficient & full["cnn_correct"]
    expected = full.loc[strict, "cause_insuffisance"].map(expected_proof)
    unknown_causes = int(expected.isna().sum())
    check(
        checks,
        "insufficiency_construction_causes_known",
        unknown_causes == 0,
        f"causes_inconnues={unknown_causes}",
    )
    cause_mismatches = int((expected != full.loc[strict, "niveau_preuve"]).sum())
    check(
        checks,
        "insufficiency_cause_matches_engine_when_cnn_correct",
        cause_mismatches == 0,
        f"ecarts={cause_mismatches}, cas_controles={int(strict.sum())}",
    )
    missing_critical = full[CRITICAL_COLUMNS].isna().sum()
    missing_critical = {
        column: int(value) for column, value in missing_critical.items() if value
    }
    check(
        checks,
        "no_critical_missing_values",
        not missing_critical,
        str(missing_critical),
    )
    check(
        checks,
        "full_table_one_row_per_image",
        len(full) == expected_images,
        f"n={len(full)}",
    )
    return full, pd.DataFrame(checks)


def scenario_summary(full: pd.DataFrame) -> pd.DataFrame:
    """Resume demande, realisabilite et echecs par scenario et vraie classe."""

    rows: list[dict[str, Any]] = []
    scopes = [("global", "Toutes classes", full)] + [
        ("classe", classe, group)
        for classe, group in full.groupby("true_label", sort=True)
    ]
    for scope, scope_value, frame in scopes:
        for scenario in ("favorable", "non_favorable", "insuffisant"):
            selected = frame[frame["scenario_assigne"] == scenario]
            failures = selected.loc[
                ~selected["scenario_realisable"], "raison_non_realisation"
            ]
            causes = Counter(failures.fillna("non_precisee").astype(str))
            requested = len(selected)
            realizable = int(selected["scenario_realisable"].sum())
            rows.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    "scenario_assigne": scenario,
                    "nb_demande": requested,
                    "nb_realisable": realizable,
                    "taux_realisation": realizable / requested if requested else np.nan,
                    "causes_echec": "; ".join(
                        f"{key}:{value}" for key, value in sorted(causes.items())
                    ),
                }
            )
    return pd.DataFrame(rows)


def insufficiency_consistency(full: pd.DataFrame) -> pd.DataFrame:
    """Compare l'intention de construction et le diagnostic effectif du moteur.

    L'egalite n'est attendue que lorsque le Top-1 est la vraie classe. Lorsque
    le CNN se trompe, le moteur evalue un autre agent et l'ecart est descriptif.
    """

    selected = full[full["scenario_realise"] == "insuffisant"].copy()
    expected_proof = {
        "classe_non_couverte_bsv": "non_couvert",
        "sans_bulletin": "sans_bulletin",
        "hors_periode_litterature": "hors_periode_litterature",
    }
    selected["niveau_preuve_attendu"] = selected["cause_insuffisance"].map(
        expected_proof
    )
    selected["cause_coherente_moteur"] = (
        selected["niveau_preuve_attendu"] == selected["niveau_preuve"]
    )
    summary = (
        selected.groupby(
            [
                "cause_insuffisance",
                "niveau_preuve_attendu",
                "niveau_preuve",
                "cnn_correct",
                "cause_coherente_moteur",
            ],
            dropna=False,
        )
        .size()
        .rename("nb_images")
        .reset_index()
    )
    summary["statut_controle"] = np.where(
        summary["cnn_correct"], "bloquant", "descriptif_cnn_incorrect"
    )
    return summary


def build_populations(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit le jeu couvert puis le jeu principal.

    ``covered`` exige une vraie classe couverte par les BSV.
    ``primary`` exige en plus un scenario realisable et un verdict Top-1
    decidable. Le jeu complet reste la table ``full``.
    """

    covered = full[full["classe_couverte_bsv"]].copy()
    primary = covered[
        covered["scenario_realisable"]
        & covered["verdict_bsv_top1"].isin(DECIDABLE_VERDICTS)
    ].copy()
    return covered, primary


def selection_summary(
    full: pd.DataFrame, covered: pd.DataFrame, primary: pd.DataFrame
) -> pd.DataFrame:
    """Documente l'entonnoir sans compter deux fois une meme exclusion."""

    # Les exclusions structurelles de classe ont deja ete retirees dans
    # ``covered``. Les lignes suivantes mesurent donc les echecs de contexte,
    # puis les abstentions du moteur sur les images restantes.
    covered_realisable = covered[covered["scenario_realisable"]]
    indeterminate = covered_realisable[
        ~covered_realisable["verdict_bsv_top1"].isin(DECIDABLE_VERDICTS)
    ]
    other_indeterminate = ~indeterminate["niveau_preuve"].isin(
        {"sans_bulletin", "hors_periode_litterature"}
    )
    other_causes = Counter(
        indeterminate.loc[other_indeterminate, "niveau_preuve"].fillna("manquant")
    )
    other_detail = "; ".join(
        f"{key}:{value}" for key, value in sorted(other_causes.items())
    )
    rows = [
        ("jeu_complet", len(full), "Toutes les images", ""),
        (
            "exclusion_classe_non_couverte",
            int((~full["classe_couverte_bsv"]).sum()),
            "Classe reelle non couverte",
            "",
        ),
        ("jeu_couvert", len(covered), "Classe reelle couverte", ""),
        (
            "exclusion_scenario_non_realisable",
            int((~covered["scenario_realisable"]).sum()),
            "Echec de construction du contexte",
            "",
        ),
        (
            "exclusion_sans_bulletin",
            int((indeterminate["niveau_preuve"] == "sans_bulletin").sum()),
            "Abstention du moteur apres construction",
            "sans_bulletin",
        ),
        (
            "exclusion_hors_periode_litterature",
            int((indeterminate["niveau_preuve"] == "hors_periode_litterature").sum()),
            "Abstention du moteur apres construction",
            "hors_periode_litterature",
        ),
        (
            "exclusion_autre_indetermine",
            int(other_indeterminate.sum()),
            "Autre abstention du moteur apres construction",
            other_detail,
        ),
        ("jeu_principal", len(primary), "Couvert, realisable et decidable", ""),
    ]
    return pd.DataFrame(rows, columns=["etape", "nb_images", "definition", "detail"])


def contingency_table(primary: pd.DataFrame) -> pd.DataFrame:
    """Teste l'association exploratoire entre justesse CNN et verdict moteur.

    La table 2 x 2 croise ``cnn_correct`` avec ``Coherent``/
    ``Non_corrobore`` sur le jeu principal. Le khi-deux est utilise si tous
    les effectifs attendus sont au moins egaux a 5 ; sinon le test exact de
    Fisher est applique. Cette analyse ne mesure pas encore l'apport
    incremental du BSV au-dela du score CNN.
    """

    # Effectifs observes avec ordre explicite des lignes et colonnes.
    observed = pd.crosstab(primary["cnn_correct"], primary["verdict_bsv_top1"]).reindex(
        index=[True, False], columns=["Coherent", "Non_corrobore"], fill_value=0
    )
    if observed.to_numpy().sum() == 0:
        raise ValueError("Le jeu principal est vide")
    # Premier appel : calcul des effectifs attendus sous l'hypothese
    # d'independance. La correction de Yates est volontairement desactivee.
    _, _, _, expected = chi2_contingency(observed.to_numpy(), correction=False)
    if (expected >= 5).all():
        statistic, p_value, _, _ = chi2_contingency(
            observed.to_numpy(), correction=False
        )
        test_name = "chi2"
    else:
        statistic, p_value = fisher_exact(observed.to_numpy())
        test_name = "fisher_exact"
    # Le resultat global du test est repete sur les quatre cellules afin que le
    # CSV reste autonome et contienne observes, attendus et resultat du test.
    rows = []
    for row_index, correct in enumerate([True, False]):
        for column_index, verdict in enumerate(["Coherent", "Non_corrobore"]):
            rows.append(
                {
                    "cnn_correct": correct,
                    "verdict_bsv_top1": verdict,
                    "observed": int(observed.loc[correct, verdict]),
                    "expected": float(expected[row_index, column_index]),
                    "test": test_name,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def separation_diagnostics(primary: pd.DataFrame) -> pd.DataFrame:
    """Repere separation et quasi-separation pour chaque modalite candidate."""

    rows = []
    for predictor in (
        "classe_cnn_top1",
        "verdict_bsv_top1",
        "niveau_preuve",
        "niveau_risque",
    ):
        values = primary[predictor].fillna("manquant").astype(str)
        counts = pd.crosstab(values, primary["cnn_correct"]).reindex(
            columns=[True, False], fill_value=0
        )
        for level, count in counts.iterrows():
            correct = int(count[True])
            incorrect = int(count[False])
            rows.append(
                {
                    "predictor": predictor,
                    "level": level,
                    "cnn_correct": correct,
                    "cnn_incorrect": incorrect,
                    "separation": correct == 0 or incorrect == 0,
                    "quasi_separation": min(correct, incorrect) < 5,
                }
            )
    return pd.DataFrame(rows)


def modeling_feasibility(
    primary: pd.DataFrame,
    contingency: pd.DataFrame,
    separation: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """Produit les diagnostics descriptifs avant les modeles M0 et M1."""

    correct = int(primary["cnn_correct"].sum())
    incorrect = len(primary) - correct
    levels = {
        "classe_cnn_top1": primary["classe_cnn_top1"].nunique(),
        "verdict_bsv_top1": primary["verdict_bsv_top1"].nunique(),
        "niveau_preuve": primary["niveau_preuve"].fillna("manquant").nunique(),
        "niveau_risque": primary["niveau_risque"].fillna("manquant").nunique(),
    }
    # Approximation prudente : intercept + indicatrices des variables
    # categorielles. Elle sert uniquement au diagnostic preliminaire EPV.
    parameter_count = 1 + sum(max(count - 1, 0) for count in levels.values())
    epv = min(correct, incorrect) / parameter_count if parameter_count else np.nan
    # Le bootstrap sera groupe par bulletin ; context_id sert de repli lorsque
    # aucun bulletin n'est associe a la ligne.
    group_key = primary["bulletin_id"].fillna("").astype(str)
    group_key = group_key.where(group_key.ne(""), primary["context_id"].astype(str))
    group_sizes = group_key.value_counts()
    observed = contingency["observed"]
    expected = contingency["expected"]
    global_scenarios = scenarios[scenarios["scope"] == "global"].set_index(
        "scenario_assigne"
    )
    realization_gap = abs(
        float(global_scenarios.loc["favorable", "taux_realisation"])
        - float(global_scenarios.loc["non_favorable", "taux_realisation"])
    )
    separation_levels = int(separation["separation"].sum())
    quasi_levels = int(separation["quasi_separation"].sum())
    class_counts = primary["true_label"].value_counts()
    verdict_counts = primary["verdict_bsv_top1"].value_counts()
    metrics = [
        ("n_primary", len(primary), "ok" if len(primary) else "bloquant"),
        ("cnn_correct", correct, "ok" if correct >= 20 else "attention"),
        ("cnn_incorrect", incorrect, "ok" if incorrect >= 20 else "attention"),
        ("provisional_parameter_count", parameter_count, "informatif"),
        ("events_per_parameter", epv, "ok" if epv >= 10 else "attention"),
        (
            "min_observed_contingency_cell",
            int(observed.min()),
            "ok" if observed.min() >= 5 else "attention",
        ),
        (
            "min_expected_contingency_cell",
            float(expected.min()),
            "ok" if expected.min() >= 5 else "attention",
        ),
        (
            "quasi_separation_detected",
            bool((observed < 5).any()),
            "attention" if (observed < 5).any() else "ok",
        ),
        (
            "categorical_separation_levels",
            separation_levels,
            "attention" if separation_levels else "ok",
        ),
        (
            "categorical_quasi_separation_levels",
            quasi_levels,
            "attention" if quasi_levels else "ok",
        ),
        ("predicted_classes", levels["classe_cnn_top1"], "informatif"),
        ("true_class_min_size", int(class_counts.min()), "informatif"),
        ("true_class_max_size", int(class_counts.max()), "informatif"),
        ("verdict_coherent", int(verdict_counts.get("Coherent", 0)), "informatif"),
        (
            "verdict_non_corrobore",
            int(verdict_counts.get("Non_corrobore", 0)),
            "informatif",
        ),
        (
            "favorable_non_favorable_realization_gap",
            realization_gap,
            "ok" if realization_gap <= 0.05 else "attention",
        ),
        ("distinct_bulletins", int(primary["bulletin_id"].nunique()), "informatif"),
        ("distinct_contexts", int(primary["context_id"].nunique()), "informatif"),
        (
            "bootstrap_groups",
            int(group_sizes.size),
            "ok" if group_sizes.size >= 20 else "attention",
        ),
        ("bootstrap_group_min_size", int(group_sizes.min()), "informatif"),
        ("bootstrap_group_median_size", float(group_sizes.median()), "informatif"),
        ("bootstrap_group_max_size", int(group_sizes.max()), "informatif"),
        ("score_min", float(primary["score_cnn_top1"].min()), "informatif"),
        ("score_q25", float(primary["score_cnn_top1"].quantile(0.25)), "informatif"),
        ("score_median", float(primary["score_cnn_top1"].median()), "informatif"),
        ("score_q75", float(primary["score_cnn_top1"].quantile(0.75)), "informatif"),
        ("score_max", float(primary["score_cnn_top1"].max()), "informatif"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "status"])


def prepare_logistic_data(
    primary: pd.DataFrame, *, epsilon: float = SCORE_EPSILON
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Code Y, V et logit(score) sur une population commune complete."""

    required = ["cnn_correct", "score_cnn_top1", "verdict_bsv_top1", "bulletin_id"]
    missing_columns = [column for column in required if column not in primary]
    if missing_columns:
        raise ValueError(f"Colonnes absentes pour la modelisation: {missing_columns}")
    optional = ["context_id"] if "context_id" in primary else []
    data = primary.loc[:, required + optional].copy()
    analytic_columns = ["cnn_correct", "score_cnn_top1", "verdict_bsv_top1"]
    if data[analytic_columns].isna().any().any():
        raise ValueError(
            "M0 et M1 exigent les memes observations sans valeur manquante "
            f"(manquants={data[analytic_columns].isna().sum().to_dict()})"
        )
    group_key = data["bulletin_id"].fillna("").astype(str).str.strip()
    if group_key.eq("").any():
        if "context_id" not in data:
            raise ValueError("bulletin_id manquant et aucun context_id de repli")
        contexts = data["context_id"].fillna("").astype(str).str.strip()
        group_key = group_key.where(group_key.ne(""), "context:" + contexts)
    if group_key.eq("context:").any():
        raise ValueError("bulletin_id et context_id sont tous deux manquants")
    data["modeling_group"] = group_key
    scores = pd.to_numeric(data["score_cnn_top1"], errors="coerce").to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("score_cnn_top1 doit etre fini et compris dans [0, 1]")
    verdict = data["verdict_bsv_top1"].map({"Coherent": 1.0, "Non_corrobore": 0.0})
    if verdict.isna().any():
        raise ValueError("verdict_bsv_top1 contient une modalite non decidable")
    y = data["cnn_correct"].astype(int).to_numpy(float)
    if np.unique(y).size != 2:
        raise ValueError(
            "La regression logistique exige les deux modalites de cnn_correct"
        )
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    score_logit = np.log(clipped / (1.0 - clipped))
    x0 = np.column_stack([np.ones(len(data)), score_logit])
    x1 = np.column_stack([np.ones(len(data)), score_logit, verdict.to_numpy(float)])
    data["score_cnn_borne"] = clipped
    data["logit_score_cnn"] = score_logit
    data["verdict_coherent"] = verdict.to_numpy(int)
    return data, y, x0, x1


def _fit_logistic(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Ajuste un logit par maximum de vraisemblance sans penalisation."""

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta
        loss = float(np.logaddexp(0.0, linear).sum() - y @ linear)
        gradient = x.T @ (expit(linear) - y)
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        method="BFGS",
        jac=True,
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    # BFGS peut signaler une perte de precision alors que le gradient et les
    # coefficients sont parfaitement exploitables. On controle donc les
    # quantites statistiques, et non le seul drapeau de l'optimiseur.
    beta = np.asarray(result.x, dtype=float)
    gradient = x.T @ (expit(x @ beta) - y)
    if (
        not np.isfinite(beta).all()
        or not np.isfinite(result.fun)
        or np.max(np.abs(gradient)) > 1e-4
    ):
        raise ValueError(f"Echec de convergence du logit: {result.message}")
    probabilities = expit(x @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = x.T @ (x * weights[:, None])
    covariance = np.linalg.pinv(information)
    return {
        "beta": beta,
        "log_likelihood": -float(result.fun),
        "probabilities": probabilities,
        "covariance": covariance,
        "information": information,
    }


def _cluster_covariance(
    y: np.ndarray, x: np.ndarray, fit: dict[str, Any], groups: np.ndarray
) -> np.ndarray:
    """Calcule la covariance sandwich avec correction petits echantillons."""

    unique_groups = pd.unique(groups)
    group_count = len(unique_groups)
    if group_count < 2:
        raise ValueError("Au moins deux bulletins sont requis pour le Wald groupe")
    residuals = y - fit["probabilities"]
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in unique_groups:
        score = x[groups == group].T @ residuals[groups == group]
        meat += np.outer(score, score)
    bread = np.linalg.pinv(fit["information"])
    n, parameters = x.shape
    correction = (group_count / (group_count - 1)) * ((n - 1) / (n - parameters))
    return correction * bread @ meat @ bread


def nested_logistic_analysis(
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ajuste M0/M1 et teste l'apport du verdict au-dela du score CNN."""

    data, y, x0, x1 = prepare_logistic_data(primary)
    fit0 = _fit_logistic(y, x0)
    fit1 = _fit_logistic(y, x1)
    groups = data["modeling_group"].to_numpy()
    cluster_covariance = _cluster_covariance(y, x1, fit1, groups)
    names = ["intercept", "logit_score_cnn", "verdict_coherent"]
    rows = []
    for model, names_model, fit in (
        ("M0", names[:2], fit0),
        ("M1", names, fit1),
    ):
        standard_errors = np.sqrt(np.diag(fit["covariance"]))
        for index, term in enumerate(names_model):
            coefficient = float(fit["beta"][index])
            rows.append(
                {
                    "model": model,
                    "term": term,
                    "coefficient": coefficient,
                    "standard_error": float(standard_errors[index]),
                    "odds_ratio": float(np.exp(np.clip(coefficient, -700, 700))),
                    "ci95_low": float(
                        np.exp(
                            np.clip(
                                coefficient - 1.96 * standard_errors[index], -700, 700
                            )
                        )
                    ),
                    "ci95_high": float(
                        np.exp(
                            np.clip(
                                coefficient + 1.96 * standard_errors[index], -700, 700
                            )
                        )
                    ),
                }
            )
    verdict_index = 2
    cluster_se = float(np.sqrt(cluster_covariance[verdict_index, verdict_index]))
    verdict_beta = float(fit1["beta"][verdict_index])
    z_cluster = verdict_beta / cluster_se
    lr = 2.0 * (fit1["log_likelihood"] - fit0["log_likelihood"])
    comparison = pd.DataFrame(
        [
            {
                "n_observations_m0": len(y),
                "n_observations_m1": len(y),
                "n_correct": int(y.sum()),
                "n_incorrect": int(len(y) - y.sum()),
                "n_bulletins": int(pd.Series(groups).nunique()),
                "log_likelihood_m0": fit0["log_likelihood"],
                "log_likelihood_m1": fit1["log_likelihood"],
                "lr_statistic": lr,
                "lr_df": 1,
                "lr_p_value": float(chi2.sf(max(lr, 0.0), 1)),
                "verdict_coefficient": verdict_beta,
                "adjusted_odds_ratio": float(np.exp(verdict_beta)),
                "cluster_standard_error": cluster_se,
                "cluster_ci95_or_low": float(np.exp(verdict_beta - 1.96 * cluster_se)),
                "cluster_ci95_or_high": float(np.exp(verdict_beta + 1.96 * cluster_se)),
                "cluster_wald_z": z_cluster,
                "cluster_wald_p_value": float(2.0 * norm.sf(abs(z_cluster))),
            }
        ]
    )
    predictions = data.copy()
    predictions["cnn_correct"] = y.astype(int)
    predictions["probability_m0"] = fit0["probabilities"]
    predictions["probability_m1"] = fit1["probabilities"]
    return pd.DataFrame(rows), comparison, predictions


def predicted_class_diagnostics(primary: pd.DataFrame) -> pd.DataFrame:
    """Diagnostique les erreurs et cellules rares par classe CNN predite.

    Les classes a surveiller sont determinees par leur nombre d'erreurs, puis
    par la repartition Coherent/Non_corrobore au sein de ces erreurs. Le
    diagnostic ne se limite donc pas a une table classe-verdict globale.
    """

    rows = []
    for predicted_class, frame in primary.groupby("classe_cnn_top1", sort=True):
        errors = frame.loc[~frame["cnn_correct"].astype(bool)]
        error_verdicts = errors["verdict_bsv_top1"].value_counts()
        correct = int(frame["cnn_correct"].sum())
        incorrect = len(frame) - correct
        coherent_errors = int(error_verdicts.get("Coherent", 0))
        non_corrobore_errors = int(error_verdicts.get("Non_corrobore", 0))
        cells = [correct, incorrect, coherent_errors, non_corrobore_errors]
        rows.append(
            {
                "classe_cnn_top1": predicted_class,
                "n_observations": len(frame),
                "cnn_correct": correct,
                "cnn_incorrect": incorrect,
                "error_verdict_coherent": coherent_errors,
                "error_verdict_non_corrobore": non_corrobore_errors,
                "zero_outcome_cell": correct == 0 or incorrect == 0,
                "zero_error_verdict_cell": coherent_errors == 0
                or non_corrobore_errors == 0,
                "min_diagnostic_cell": min(cells),
                "near_zero_cell": min(cells) < 5,
                "class_to_monitor": incorrect < 10 or min(cells) < 5,
            }
        )
    return pd.DataFrame(rows)


def _fit_firth_logistic(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Ajuste un logit de Firth par vraisemblance penalisee de Jeffreys."""

    def penalized_objective(beta: np.ndarray) -> float:
        linear = x @ beta
        log_likelihood = float(y @ linear - np.logaddexp(0.0, linear).sum())
        probability = expit(linear)
        information = x.T @ (x * (probability * (1.0 - probability))[:, None])
        sign, log_determinant = np.linalg.slogdet(information)
        if sign <= 0 or not np.isfinite(log_determinant):
            return 1e100
        return -(log_likelihood + 0.5 * log_determinant)

    result = minimize(
        penalized_objective,
        np.zeros(x.shape[1]),
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 2000},
    )
    beta = np.asarray(result.x, dtype=float)
    probability = expit(x @ beta)
    information = x.T @ (x * (probability * (1.0 - probability))[:, None])
    if not np.isfinite(beta).all() or not np.isfinite(result.fun):
        raise ValueError(f"Echec du logit de Firth: {result.message}")
    return {
        "beta": beta,
        "log_likelihood": -float(result.fun),
        "probabilities": probability,
        "covariance": np.linalg.pinv(information),
        "information": information,
    }


def class_adjusted_sensitivity(
    primary: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare MC a M2, sans interaction verdict-classe.

    MC controle le logit du score et la classe predite. M2 ajoute seulement le
    verdict. Si une classe ne contient qu'une seule issue, Firth est applique
    aux deux modeles de cette comparaison afin de traiter la separation.
    """

    data, y, _, x1 = prepare_logistic_data(primary)
    diagnostics = (
        predicted_class_diagnostics(primary) if diagnostics is None else diagnostics
    )
    classes = primary["classe_cnn_top1"].astype(str)
    dummy = pd.get_dummies(classes, prefix="classe", drop_first=True, dtype=float)
    score_logit = data["logit_score_cnn"].to_numpy(float)
    verdict = data["verdict_coherent"].to_numpy(float)
    xc = np.column_stack([np.ones(len(data)), score_logit, dummy.to_numpy(float)])
    x2 = np.column_stack([xc, verdict])
    names_c = ["intercept", "logit_score_cnn", *dummy.columns.tolist()]
    names_2 = [*names_c, "verdict_coherent"]

    separation = bool(diagnostics["zero_outcome_cell"].any())
    fit_function = _fit_firth_logistic if separation else _fit_logistic
    method = "firth" if separation else "maximum_likelihood"
    fit_c = fit_function(y, xc)
    fit_2 = fit_function(y, x2)
    # M1 est reutilise pour la comparaison descriptive M1-M2. En presence de
    # Firth, on l'ajuste avec la meme penalisation afin que les criteres soient
    # comparables.
    fit_1 = fit_function(y, x1)

    coefficient_rows = []
    for model_name, names, fit in (("MC", names_c, fit_c), ("M2", names_2, fit_2)):
        standard_errors = np.sqrt(np.diag(fit["covariance"]))
        for index, term in enumerate(names):
            coefficient = float(fit["beta"][index])
            coefficient_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "term": term,
                    "coefficient": coefficient,
                    "standard_error": float(standard_errors[index]),
                    "odds_ratio": float(np.exp(np.clip(coefficient, -700, 700))),
                    "ci95_low": float(
                        np.exp(
                            np.clip(
                                coefficient - 1.96 * standard_errors[index], -700, 700
                            )
                        )
                    ),
                    "ci95_high": float(
                        np.exp(
                            np.clip(
                                coefficient + 1.96 * standard_errors[index], -700, 700
                            )
                        )
                    ),
                }
            )

    lr_verdict = 2.0 * (fit_2["log_likelihood"] - fit_c["log_likelihood"])
    class_df = dummy.shape[1]
    lr_class = 2.0 * (fit_2["log_likelihood"] - fit_1["log_likelihood"])
    comparison = pd.DataFrame(
        [
            {
                "comparison": "MC_vs_M2",
                "tested_addition": "verdict",
                "method": method,
                "n_observations": len(y),
                "degrees_of_freedom": 1,
                "lr_statistic": max(lr_verdict, 0.0),
                "p_value": float(chi2.sf(max(lr_verdict, 0.0), 1)),
                "verdict_coefficient_m2": float(fit_2["beta"][-1]),
                "adjusted_verdict_or_m2": float(np.exp(fit_2["beta"][-1])),
            },
            {
                "comparison": "M1_vs_M2",
                "tested_addition": "predicted_class_global",
                "method": method,
                "n_observations": len(y),
                "degrees_of_freedom": class_df,
                "lr_statistic": max(lr_class, 0.0),
                "p_value": float(chi2.sf(max(lr_class, 0.0), class_df)),
                "verdict_coefficient_m2": float(fit_2["beta"][-1]),
                "adjusted_verdict_or_m2": float(np.exp(fit_2["beta"][-1])),
            },
        ]
    )
    trigger_classes = diagnostics.loc[
        diagnostics["zero_outcome_cell"], "classe_cnn_top1"
    ].tolist()
    firth_diagnostics = pd.DataFrame(
        [
            {
                "firth_required": separation,
                "reason": (
                    "separation_by_predicted_class"
                    if separation
                    else "no_observed_separation"
                ),
                "trigger_classes": ";".join(trigger_classes),
                "estimation_method_MC_M2": method,
                "verdict_class_interaction_fitted": False,
                "interaction_reason": "76 CNN errors: interaction intentionally excluded",
                "max_abs_coefficient_m2": float(np.max(np.abs(fit_2["beta"]))),
                "max_standard_error_m2": float(
                    np.max(np.sqrt(np.diag(fit_2["covariance"])))
                ),
                "min_predicted_probability_m2": float(fit_2["probabilities"].min()),
                "max_predicted_probability_m2": float(fit_2["probabilities"].max()),
            }
        ]
    )
    return pd.DataFrame(coefficient_rows), comparison, firth_diagnostics


def score_linearity_diagnostic(
    primary: pd.DataFrame, *, bins: int = 10
) -> pd.DataFrame:
    """Resume les log-odds empiriques de justesse par quantiles du score."""

    data, y, _, _ = prepare_logistic_data(primary)
    data["cnn_correct"] = y.astype(int)
    data["score_bin"] = pd.qcut(
        data["score_cnn_borne"],
        q=min(bins, data["score_cnn_borne"].nunique()),
        duplicates="drop",
    )
    rows = []
    for interval, group in data.groupby("score_bin", observed=True):
        correct = int(group["cnn_correct"].sum())
        n = len(group)
        # Correction de Haldane-Anscombe pour garder des log-odds finis.
        empirical_log_odds = float(np.log((correct + 0.5) / (n - correct + 0.5)))
        rows.append(
            {
                "score_interval": str(interval),
                "n": n,
                "cnn_correct": correct,
                "cnn_incorrect": n - correct,
                "mean_score": float(group["score_cnn_borne"].mean()),
                "mean_logit_score": float(group["logit_score_cnn"].mean()),
                "empirical_correct_rate": correct / n,
                "empirical_log_odds_correct": empirical_log_odds,
            }
        )
    return pd.DataFrame(rows)


def grouped_bulletin_bootstrap(
    primary: pd.DataFrame,
    *,
    iterations: int = 1000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reechantillonne les bulletins entiers et reajuste M0 et M1.

    Le bootstrap est non parametrique et groupe : a chaque replication, autant
    de groupes qu'observes sont tires avec remise. Toutes les observations
    appartenant a un groupe tire sont conservees ensemble. La distribution du
    coefficient du verdict fournit l'IC percentile de l'odds ratio. La
    distribution du LRT est exportee a titre descriptif ; elle ne constitue
    pas un LRT sous l'hypothese nulle.
    """

    if iterations < 1:
        raise ValueError("bootstrap_iterations doit etre strictement positif")
    data, y, x0, x1 = prepare_logistic_data(primary)
    groups = data["modeling_group"].to_numpy()
    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Au moins deux bulletins sont requis pour le bootstrap groupe")
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    failed = 0
    attempts = 0
    max_attempts = max(iterations * 3, iterations + 10)
    while len(rows) < iterations and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sampled_indices = np.concatenate(
            [indices_by_group[group] for group in sampled_groups]
        )
        sampled_y = y[sampled_indices]
        # Une replication sans les deux issues ne permet pas d'ajuster un
        # logit interpretable ; elle est remplacee et comptabilisee en echec.
        if np.unique(sampled_y).size != 2:
            failed += 1
            continue
        try:
            fit0 = _fit_logistic(sampled_y, x0[sampled_indices])
            fit1 = _fit_logistic(sampled_y, x1[sampled_indices])
        except ValueError:
            failed += 1
            continue
        coefficient = float(fit1["beta"][2])
        lr = 2.0 * (fit1["log_likelihood"] - fit0["log_likelihood"])
        rows.append(
            {
                "replication": len(rows) + 1,
                "n_sampled_observations": len(sampled_indices),
                "n_distinct_sampled_groups": int(pd.Series(sampled_groups).nunique()),
                "verdict_coefficient": coefficient,
                "adjusted_odds_ratio": float(np.exp(np.clip(coefficient, -700, 700))),
                "lr_statistic": float(max(lr, 0.0)),
            }
        )
    if len(rows) < iterations:
        raise ValueError(
            "Bootstrap groupe incomplet apres "
            f"{attempts} tentatives: {len(rows)}/{iterations} replications"
        )
    bootstrap = pd.DataFrame(rows)
    coefficients = bootstrap["verdict_coefficient"].to_numpy()
    odds_ratios = bootstrap["adjusted_odds_ratio"].to_numpy()
    # Approximation bilaterale centree sur zero, avec correction +1 pour
    # eviter une p-valeur exactement nulle dans un nombre fini de tirages.
    below = int((coefficients <= 0).sum())
    above = int((coefficients >= 0).sum())
    empirical_p = min(1.0, 2.0 * (min(below, above) + 1) / (iterations + 1))
    summary = pd.DataFrame(
        [
            {
                "requested_replications": iterations,
                "successful_replications": len(bootstrap),
                "failed_attempts": failed,
                "total_attempts": attempts,
                "seed": seed,
                "n_groups": len(unique_groups),
                "median_verdict_coefficient": float(np.median(coefficients)),
                "median_adjusted_odds_ratio": float(np.median(odds_ratios)),
                "bootstrap_ci95_or_low": float(np.quantile(odds_ratios, 0.025)),
                "bootstrap_ci95_or_high": float(np.quantile(odds_ratios, 0.975)),
                "bootstrap_two_sided_p_value": float(empirical_p),
                "median_lr_statistic": float(np.median(bootstrap["lr_statistic"])),
            }
        ]
    )
    return bootstrap, summary


def grouped_cross_validation(
    primary: pd.DataFrame, *, folds: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare M0/M1 hors echantillon sans partager un bulletin entre plis."""

    data, y, x0, x1 = prepare_logistic_data(primary)
    groups = data["modeling_group"]
    unique = groups.value_counts().sort_values(ascending=False)
    fold_count = min(folds, len(unique))
    if fold_count < 2:
        raise ValueError(
            "Au moins deux bulletins sont requis pour la validation croisee"
        )
    loads = np.zeros(fold_count, dtype=int)
    group_to_fold: dict[str, int] = {}
    for group, size in unique.items():
        fold = int(np.argmin(loads))
        group_to_fold[group] = fold
        loads[fold] += int(size)
    assignments = groups.map(group_to_fold).to_numpy()
    rows = []
    for fold in range(fold_count):
        test = assignments == fold
        train = ~test
        if np.unique(y[train]).size != 2:
            raise ValueError(
                f"Le pli {fold + 1} ne contient pas les deux issues a l'entrainement"
            )
        fit0 = _fit_logistic(y[train], x0[train])
        fit1 = _fit_logistic(y[train], x1[train])
        probabilities = {
            "M0": np.clip(
                expit(x0[test] @ fit0["beta"]), SCORE_EPSILON, 1 - SCORE_EPSILON
            ),
            "M1": np.clip(
                expit(x1[test] @ fit1["beta"]), SCORE_EPSILON, 1 - SCORE_EPSILON
            ),
        }
        for model, probability in probabilities.items():
            outcome = y[test]
            log_loss = -float(
                np.mean(
                    outcome * np.log(probability)
                    + (1 - outcome) * np.log(1 - probability)
                )
            )
            rows.append(
                {
                    "fold": fold + 1,
                    "model": model,
                    "n_test": int(test.sum()),
                    "n_bulletins_test": int(groups[test].nunique()),
                    "log_loss": log_loss,
                    "brier_score": float(np.mean((outcome - probability) ** 2)),
                }
            )
    results = pd.DataFrame(rows)
    paired = results.pivot(
        index="fold", columns="model", values=["log_loss", "brier_score"]
    )
    summary = pd.DataFrame(
        [
            {
                "folds": fold_count,
                "n_observations": len(y),
                "n_bulletins": len(unique),
                "mean_log_loss_m0": float(paired["log_loss"]["M0"].mean()),
                "mean_log_loss_m1": float(paired["log_loss"]["M1"].mean()),
                "delta_log_loss_m1_minus_m0": float(
                    (paired["log_loss"]["M1"] - paired["log_loss"]["M0"]).mean()
                ),
                "mean_brier_m0": float(paired["brier_score"]["M0"].mean()),
                "mean_brier_m1": float(paired["brier_score"]["M1"].mean()),
                "delta_brier_m1_minus_m0": float(
                    (paired["brier_score"]["M1"] - paired["brier_score"]["M0"]).mean()
                ),
            }
        ]
    )
    return results, summary


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Ecrit un CSV de maniere atomique via un fichier temporaire."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def configure_run_logger(log_file: Path) -> tuple[logging.Logger, Path]:
    """Cree le journal courant et une copie horodatee propre a l'execution."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = log_file.with_name(f"{log_file.stem}_{timestamp}{log_file.suffix}")
    logger = logging.getLogger("phase3_prepare_analysis")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for path, mode in ((log_file, "w"), (timestamped, "w")):
        handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, timestamped


def log_frame_summary(
    logger: logging.Logger, name: str, frame: pd.DataFrame, *, details: bool = False
) -> None:
    """Journalise dimensions, colonnes et, si demande, toutes les lignes."""

    logger.info(
        "TABLE %-42s lignes=%d colonnes=%d",
        name,
        len(frame),
        len(frame.columns),
    )
    logger.info("COLONNES %-39s %s", name, ", ".join(map(str, frame.columns)))
    if details and not frame.empty:
        for line in frame.to_string(index=False).splitlines():
            logger.info("DETAIL %-41s %s", name, line)


def parse_args() -> argparse.Namespace:
    """Declare les entrees et le repertoire de sortie configurables."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Journal courant. Par defaut: " "<output-dir>/phase3_prepare_analysis.log"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Execute la preparation complete puis publie les artefacts figes."""

    args = parse_args()
    log_file = args.log_file or (args.output_dir / DEFAULT_LOG_FILE.name)
    logger, timestamped_log = configure_run_logger(log_file)
    started = time.perf_counter()
    try:
        logger.info("=" * 88)
        logger.info("DEBUT PHASE3_PREPARE_ANALYSIS")
        logger.info("assignments=%s", args.assignments.resolve())
        logger.info("predictions=%s", args.predictions.resolve())
        logger.info("query=%s", args.query.resolve())
        logger.info("verdicts=%s", args.verdicts.resolve())
        logger.info("output_dir=%s", args.output_dir.resolve())
        logger.info("expected_images=%d", args.expected_images)
        logger.info("bootstrap_iterations=%d", args.bootstrap_iterations)
        logger.info("bootstrap_seed=%d", args.bootstrap_seed)

        assignments = pd.read_csv(args.assignments)
        predictions = load_json_list(args.predictions)
        query = load_json_list(args.query)
        verdicts = pd.read_csv(args.verdicts)
        log_frame_summary(logger, "assignments", assignments)
        logger.info("TABLE %-42s lignes=%d", "predictions_json", len(predictions))
        logger.info("TABLE %-42s lignes=%d", "query_json", len(query))
        log_frame_summary(logger, "verdicts", verdicts)

        # 1. Jointure Top-1 et controles bloquants.
        logger.info("ETAPE 1 | Jointure Top-1 et controles d'integrite")
        full, integrity = prepare_top1_table(
            assignments,
            predictions,
            query,
            verdicts,
            expected_images=args.expected_images,
        )
        log_frame_summary(logger, "integrity_checks", integrity, details=True)

        # 2. Populations et tableaux descriptifs/statistiques.
        logger.info("ETAPE 2 | Populations, diagnostics et modeles")
        covered, primary = build_populations(full)
        scenarios = scenario_summary(full)
        selection = selection_summary(full, covered, primary)
        contingency = contingency_table(primary)
        separation = separation_diagnostics(primary)
        feasibility = modeling_feasibility(primary, contingency, separation, scenarios)
        insufficiency = insufficiency_consistency(full)
        coefficients, logistic_comparison, logistic_predictions = (
            nested_logistic_analysis(primary)
        )
        class_diagnostics = predicted_class_diagnostics(primary)
        class_coefficients, class_comparison, firth_diagnostics = (
            class_adjusted_sensitivity(primary, class_diagnostics)
        )
        linearity = score_linearity_diagnostic(primary)
        bootstrap, bootstrap_summary = grouped_bulletin_bootstrap(
            primary,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        cross_validation, cross_validation_summary = grouped_cross_validation(primary)

        logger.info(
            "POPULATIONS | complet=%d couvert=%d principal=%d",
            len(full),
            len(covered),
            len(primary),
        )
        for name, frame in (
            ("selection_summary", selection),
            ("contingency_top1", contingency),
            ("modeling_feasibility", feasibility),
            ("logistic_comparison_M0_M1", logistic_comparison),
            ("predicted_class_diagnostics", class_diagnostics),
            ("class_adjusted_comparison", class_comparison),
            ("firth_diagnostics", firth_diagnostics),
            ("bootstrap_summary", bootstrap_summary),
            ("cross_validation_summary", cross_validation_summary),
        ):
            log_frame_summary(logger, name, frame, details=True)

        # 3. Exports qui deviennent les sources d'autorite des notebooks.
        logger.info("ETAPE 3 | Ecriture de tous les artefacts CSV")
        outputs = {
            "phase3_analysis_full.csv": full,
            "phase3_analysis_covered.csv": covered,
            "phase3_analysis_primary.csv": primary,
            "phase3_selection_summary.csv": selection,
            "phase3_scenario_summary.csv": scenarios,
            "phase3_contingency_top1.csv": contingency,
            "phase3_integrity_checks.csv": integrity,
            "phase3_modeling_feasibility.csv": feasibility,
            "phase3_separation_diagnostics.csv": separation,
            "phase3_insufficiency_consistency.csv": insufficiency,
            "phase3_logistic_coefficients.csv": coefficients,
            "phase3_logistic_comparison.csv": logistic_comparison,
            "phase3_logistic_predictions.csv": logistic_predictions,
            "phase3_predicted_class_diagnostics.csv": class_diagnostics,
            "phase3_class_adjusted_coefficients.csv": class_coefficients,
            "phase3_class_adjusted_comparison.csv": class_comparison,
            "phase3_firth_diagnostics.csv": firth_diagnostics,
            "phase3_score_linearity.csv": linearity,
            "phase3_grouped_bootstrap.csv": bootstrap,
            "phase3_grouped_bootstrap_summary.csv": bootstrap_summary,
            "phase3_grouped_cross_validation.csv": cross_validation,
            "phase3_grouped_cross_validation_summary.csv": cross_validation_summary,
        }
        for filename, frame in outputs.items():
            output_path = args.output_dir / filename
            write_csv(output_path, frame)
            logger.info(
                "SORTIE | %-52s lignes=%d colonnes=%d octets=%d",
                output_path.resolve(),
                len(frame),
                len(frame.columns),
                output_path.stat().st_size,
            )

        elapsed = time.perf_counter() - started
        logger.info("STATUT FINAL | SUCCESS")
        logger.info("DUREE | %.3f secondes", elapsed)
        logger.info("JOURNAL COURANT | %s", log_file.resolve())
        logger.info("JOURNAL HORODATE | %s", timestamped_log.resolve())
        logger.info("FIN PHASE3_PREPARE_ANALYSIS")
        logger.info("=" * 88)

        print(f"Jeu complet   : {len(full)}")
        print(f"Jeu couvert   : {len(covered)}")
        print(f"Jeu principal : {len(primary)}")
        print(f"Artefacts     : {args.output_dir}")
        print(f"Journal       : {log_file}")
        print(f"Journal run   : {timestamped_log}")
        return 0
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception("STATUT FINAL | ECHEC apres %.3f secondes", elapsed)
        logger.info("JOURNAL COURANT | %s", log_file.resolve())
        logger.info("JOURNAL HORODATE | %s", timestamped_log.resolve())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
