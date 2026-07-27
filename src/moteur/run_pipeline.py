"""Orchestre le pipeline phase 3-Phytosanitaire, de la collecte BSV aux notebooks de Phase 3.

Quelle commande lancer ?
------------------------

Cas normal, sans telecharger de nouveau les BSV :

    python src/run_pipeline.py

Cette commande execute les phases B a E : tests prealables, construction des
contextes, CNN, requete aveugle, moteur, analyses, notebooks et tests finaux.

Si seuls l'analyse ou les notebooks ont change :

    python src/run_pipeline.py --from-phase E

Si le moteur, les scenarios ou leur construction ont change :

    python src/run_pipeline.py --from-phase C

Si le corpus BSV doit etre re-telecharge et entierement reconstruit :

    python src/run_pipeline.py --from-phase A

La phase A lance reellement le scraping. Pour reconstruire le corpus a partir
des PDF deja presents sans relancer le scraping :

    python src/run_pipeline.py --from-phase A --skip-scrape

Pour afficher les commandes sans rien executer :

    python src/run_pipeline.py --from-phase A --dry-run

Pour forcer une reinference CNN complete, operation potentiellement couteuse :

    python src/run_pipeline.py --from-phase D --force-cnn

Comportement de securite
-----------------------

Le scraping n'est lance que si la phase de depart est explicitement A. Par
defaut, les predictions CNN sont reutilisees uniquement apres verification des
empreintes SHA-256. Si les images anonymisees existent deja, elles ne sont ni
supprimees ni ecrasees : leur manifeste est controle puis l'etape est ignoree.
Le pipeline s'arrete au premier echec bloquant.

Chaque run produit un journal texte et un rapport JSON horodates dans
``outputs/logs``. Le journal courant est aussi disponible sous le nom stable
``outputs/logs/pipeline.log`` : il est alimente en direct et commente le role
de chaque phase, les sorties, les validations et le bilan final. La
configuration partagee (effectif attendu, graine, Top-K et fenetre BSV de 14
jours) se trouve dans ``config/pipeline.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from pipeline_config import DEFAULT_CONFIG_FILE, load_pipeline_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ORDER = "ABCDE"
ANALYSIS_DIR = PROJECT_ROOT / "outputs/moteur/phase3/analysis"
PHASE3_DIR = PROJECT_ROOT / "data/processed/moteur/phase3"


class PipelineError(RuntimeError):
    pass


@dataclass
class StepResult:
    step: str
    label: str
    command: list[str]
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    detail: str = ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PipelineError(f"Fichier attendu absent: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        raise PipelineError(f"Fichier attendu absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise PipelineError(f"Une liste JSON est attendue dans {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def validate_nonempty_files(paths: Iterable[Path]) -> str:
    missing = [
        str(path) for path in paths if not path.is_file() or path.stat().st_size == 0
    ]
    require(not missing, f"Fichiers absents ou vides: {missing}")
    return f"{len(list(paths))} fichiers presents et non vides"


def validate_manifest(expected_images: int) -> str:
    manifest = read_csv_rows(PHASE3_DIR / "phase3_test_manifest_masked.csv")
    truth = read_csv_rows(PHASE3_DIR / "ground_truth.csv")
    require(
        len(manifest) == expected_images,
        f"Manifest: {len(manifest)} != {expected_images}",
    )
    require(len(truth) == expected_images, f"Verite: {len(truth)} != {expected_images}")
    ids = [row.get("observation_id", "") for row in manifest]
    truth_ids = [row.get("observation_id", "") for row in truth]
    require(
        len(set(ids)) == expected_images and "" not in ids, "IDs manifest invalides"
    )
    require(set(ids) == set(truth_ids), "IDs manifest et verite differents")
    forbidden = {"true_label", "scenario_assigne", "scenario_realisable"}
    require(
        not (set(manifest[0]) & forbidden), "Fuite de verite dans le manifest aveugle"
    )
    require(
        all(row.get("label_masked", "").lower() == "true" for row in manifest),
        "label_masked invalide",
    )
    require(
        all((PROJECT_ROOT / row["image_path"]).is_file() for row in manifest),
        "Image anonymisee absente",
    )
    return f"{expected_images} images uniques, verite separee"


def validate_assignments(expected_images: int, seed: int) -> str:
    blind = read_csv_rows(PHASE3_DIR / "phase3_test_with_metadata.csv")
    assignments = read_csv_rows(PHASE3_DIR / "phase3_scenario_assignments.csv")
    require(
        len(blind) == len(assignments) == expected_images,
        "Effectif metadata/assignments incorrect",
    )
    blind_keys = set(blind[0])
    require(
        blind_keys
        == {
            "observation_id",
            "image_path",
            "label_masked",
            "date_observation",
            "region",
            "departement",
        },
        f"Schema aveugle incorrect: {sorted(blind_keys)}",
    )
    required = {
        "true_label",
        "scenario_assigne",
        "scenario_realise",
        "scenario_realisable",
        "cause_insuffisance",
        "classe_couverte_bsv",
        "context_id",
        "seed",
    }
    require(
        required <= set(assignments[0]),
        f"Champs experimentaux manquants: {sorted(required - set(assignments[0]))}",
    )
    require(
        {row["scenario_assigne"] for row in assignments}
        == {"favorable", "non_favorable", "insuffisant"},
        "Scenarios demandes incomplets",
    )
    require(all(int(row["seed"]) == seed for row in assignments), "Graine incoherente")
    insufficient = [
        row for row in assignments if row["scenario_realise"] == "insuffisant"
    ]
    allowed_causes = {
        "classe_non_couverte_bsv",
        "sans_bulletin",
        "hors_periode_litterature",
    }
    require(
        all(row["cause_insuffisance"] in allowed_causes for row in insufficient),
        "Cause d'insuffisance invalide",
    )
    return f"{expected_images} lignes, 3 scenarios, {len(insufficient)} contextes insuffisants"


def validate_predictions(expected_images: int) -> str:
    rows = read_json_list(
        PROJECT_ROOT / "outputs/moteur/phase3/cnn_topk_predictions.json"
    )
    ids = [str(row.get("observation_id", "")) for row in rows]
    require(
        len(rows) == expected_images and len(set(ids)) == expected_images,
        "Couverture CNN incorrecte",
    )
    failures = sum(
        not row.get("inference_ok") or not row.get("predictions") for row in rows
    )
    require(failures == 0, f"Inférences CNN invalides: {failures}")
    return f"{expected_images} predictions CNN valides"


def validate_query(expected_images: int, top_k: int) -> str:
    rows = read_json_list(PROJECT_ROOT / "outputs/moteur/phase3/requete_phase3.json")
    require(len(rows) == expected_images, f"Requetes: {len(rows)} != {expected_images}")
    require(
        all(
            set(row) == {"image_id", "date_observation", "region", "top_k"}
            for row in rows
        ),
        "Schema requete non ferme",
    )
    require(
        all(0 < len(row["top_k"]) <= top_k for row in rows), "Top-K vide ou trop long"
    )
    require(
        all(set(pred) == {"classe", "score"} for row in rows for pred in row["top_k"]),
        "Schema prediction moteur incorrect",
    )
    return f"{expected_images} requetes aveugles Top-{top_k}"


def validate_engine(expected_images: int) -> str:
    rows = read_csv_rows(
        PROJECT_ROOT / "data/processed/moteur/bsv_compatibility_phase3.csv"
    )
    image_ids = {row.get("image_id", "") for row in rows}
    require(
        len(image_ids) == expected_images and "" not in image_ids,
        "Couverture moteur incorrecte",
    )
    require(
        all(
            row.get("compatibilite") in {"Coherent", "Non_corrobore", "Indetermine"}
            for row in rows
        ),
        "Verdict moteur invalide",
    )
    return f"{expected_images} images avec sortie moteur, {len(rows)} verdicts Top-K"


def validate_analysis(expected_images: int) -> str:
    integrity = read_csv_rows(ANALYSIS_DIR / "phase3_integrity_checks.csv")
    failed = [row for row in integrity if row.get("passed", "").lower() != "true"]
    require(
        not failed,
        f"Controles d'integrite echoues: {[row.get('check') for row in failed]}",
    )
    full = read_csv_rows(ANALYSIS_DIR / "phase3_analysis_full.csv")
    require(
        len(full) == expected_images, f"Jeu complet: {len(full)} != {expected_images}"
    )
    consistency = read_csv_rows(ANALYSIS_DIR / "phase3_insufficiency_consistency.csv")
    strict_mismatches = sum(
        int(row["nb_images"])
        for row in consistency
        if row.get("cnn_correct", "").lower() == "true"
        and row.get("cause_coherente_moteur", "").lower() != "true"
    )
    controlled = sum(
        int(row["nb_images"])
        for row in consistency
        if row.get("cnn_correct", "").lower() == "true"
    )
    require(
        strict_mismatches == 0,
        f"Cause insuffisance/niveau preuve: {strict_mismatches} ecarts",
    )
    return f"{len(integrity)} controles OK; cause/preuve: 0 ecart sur {controlled} cas CNN corrects"


def validate_notebook(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors = [
        output
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    require(not errors, f"Notebook avec {len(errors)} sortie(s) d'erreur: {path.name}")
    return f"{path.name}: {len(notebook.get('cells', []))} cellules, 0 erreur"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--from-phase", choices=list(PHASE_ORDER), default="B")
    parser.add_argument("--to-phase", choices=list(PHASE_ORDER), default="E")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--force-cnn", action="store_true")
    parser.add_argument("--skip-notebooks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if PHASE_ORDER.index(args.from_phase) > PHASE_ORDER.index(args.to_phase):
        raise SystemExit("--from-phase doit preceder ou egaler --to-phase")
    config = load_pipeline_config(args.config.resolve())
    expected = args.expected_images or config["expected_images"]
    seed = args.seed or config["seed"]
    top_k = args.top_k or config["top_k"]
    require(expected > 0 and seed > 0 and top_k > 0, "Parametres numeriques invalides")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "outputs/moteur/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{timestamp}.log"
    latest_log_path = log_dir / "pipeline.log"
    report_path = log_dir / f"pipeline_{timestamp}.json"
    latest_report_path = log_dir / "pipeline.json"
    log_path.write_text("", encoding="utf-8")
    latest_log_path.write_text("", encoding="utf-8")
    results: list[StepResult] = []
    python = sys.executable

    def selected(phase: str) -> bool:
        return (
            PHASE_ORDER.index(args.from_phase)
            <= PHASE_ORDER.index(phase)
            <= PHASE_ORDER.index(args.to_phase)
        )

    def emit(message: str) -> None:
        print(message, flush=True)
        for destination in (log_path, latest_log_path):
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    phase_comments = {
        "A": "Donnees BSV : collecte, preparation et extraction du corpus phytosanitaire.",
        "B": "Prerequis : tests sur fixtures avant toute regeneration des artefacts.",
        "C": "Images : controle du manifeste puis attribution reproductible des contextes BSV.",
        "D": "Prediction : CNN aveugle, requete fermee puis verdicts du moteur BSV.",
        "E": "Analyse : populations figees, coherence cause/preuve, notebooks et tests finaux.",
    }

    def announce_phase(phase: str) -> None:
        emit("")
        emit(f"=== PHASE {phase} ===")
        emit(f"COMMENTAIRE: {phase_comments[phase]}")

    def run(
        step: str,
        label: str,
        command: list[str],
        validation: Callable[[], str] | None = None,
        *,
        skip: bool = False,
    ) -> None:
        start = datetime.now()
        clock = time.monotonic()
        if skip:
            reason = (
                "artefact protege deja present"
                if step == "C1"
                else "option de ligne de commande"
            )
            result = StepResult(
                step,
                label,
                command,
                "SKIPPED",
                start.isoformat(),
                datetime.now().isoformat(),
                0.0,
                reason,
            )
            results.append(result)
            emit(f"[{step}] {label:<42} SKIPPED - {reason}")
            return
        emit(f"[{step}] {label} ...")
        if args.dry_run:
            result = StepResult(
                step,
                label,
                command,
                "DRY_RUN",
                start.isoformat(),
                datetime.now().isoformat(),
                0.0,
                "commande non executee",
            )
            results.append(result)
            emit("      " + subprocess.list2cmdline(command))
            return
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            emit("      " + line.rstrip())
        return_code = process.wait()
        if return_code:
            duration = time.monotonic() - clock
            results.append(
                StepResult(
                    step,
                    label,
                    command,
                    "FAILED",
                    start.isoformat(),
                    datetime.now().isoformat(),
                    round(duration, 3),
                    f"code retour {return_code}",
                )
            )
            raise PipelineError(f"{step} a retourne le code {return_code}")
        try:
            detail = validation() if validation else ""
        except Exception as exc:
            duration = time.monotonic() - clock
            results.append(
                StepResult(
                    step,
                    label,
                    command,
                    "FAILED",
                    start.isoformat(),
                    datetime.now().isoformat(),
                    round(duration, 3),
                    str(exc),
                )
            )
            raise
        duration = time.monotonic() - clock
        results.append(
            StepResult(
                step,
                label,
                command,
                "SUCCESS",
                start.isoformat(),
                datetime.now().isoformat(),
                round(duration, 3),
                detail,
            )
        )
        emit(f"[{step}] {label:<42} SUCCESS {duration:7.2f}s {detail}")

    def record_validation(step: str, label: str, validation: Callable[[], str]) -> None:
        start = datetime.now()
        clock = time.monotonic()
        emit(f"[{step}] {label} ...")
        if args.dry_run:
            results.append(
                StepResult(
                    step,
                    label,
                    [],
                    "DRY_RUN",
                    start.isoformat(),
                    datetime.now().isoformat(),
                    0.0,
                    "validation non executee",
                )
            )
            return
        try:
            detail = validation()
        except Exception as exc:
            duration = time.monotonic() - clock
            results.append(
                StepResult(
                    step,
                    label,
                    [],
                    "FAILED",
                    start.isoformat(),
                    datetime.now().isoformat(),
                    round(duration, 3),
                    str(exc),
                )
            )
            raise
        duration = time.monotonic() - clock
        results.append(
            StepResult(
                step,
                label,
                [],
                "SUCCESS",
                start.isoformat(),
                datetime.now().isoformat(),
                round(duration, 3),
                detail,
            )
        )
        emit(f"[{step}] {label:<42} SUCCESS {duration:7.2f}s {detail}")

    status = "SUCCESS"
    error = ""
    emit("PIPELINE - DEBUT DU RUN")
    emit(f"Perimetre: phases {args.from_phase} a {args.to_phase}")
    emit(
        "Configuration: "
        f"expected_images={expected}, seed={seed}, top_k={top_k}, "
        f"max_bulletin_age_days={config['max_bulletin_age_days']}"
    )
    emit(f"Mode: {'simulation sans execution' if args.dry_run else 'execution reelle'}")
    try:
        if selected("A"):
            announce_phase("A")
            run(
                "A1",
                "Scraping BSV",
                [python, "src/moteur/scrape_bsv.py"],
                skip=args.skip_scrape,
            )
            run("A2", "Preparation du corpus", [python, "src/moteur/prepare_corpus.py"])
            run("A3", "Extraction des textes", [python, "src/moteur/extract_text.py"])
            run(
                "A4",
                "Extraction des bulletins",
                [python, "src/moteur/extract_bulletins.py"],
            )
            run(
                "A5",
                "Extraction des observations",
                [python, "src/moteur/extract_observations.py"],
            )
            run(
                "A6",
                "Extraction des periodes tomate",
                [python, "src/moteur/extract_tomato_bioagressor_periods.py"],
            )
            record_validation(
                "A-check",
                "Validation des donnees BSV",
                lambda: validate_nonempty_files(
                    [
                        PROJECT_ROOT / "data/processed/moteur/bsv_bulletins.csv",
                        PROJECT_ROOT / "data/processed/moteur/bsv_observations.csv",
                        PROJECT_ROOT
                        / "data/processed/moteur/periodes_bioagresseurs_tomate_par_region.csv",
                        PROJECT_ROOT
                        / "data/processed/moteur/periodes_bioagresseurs_tomate_par_region_annee.csv",
                    ]
                ),
            )

        if selected("B"):
            announce_phase("B")
            run(
                "B1",
                "Tests du moteur",
                [python, "-m", "unittest", "tests.moteur.test_moteur_compatibilite"],
            )
            run(
                "B2",
                "Tests de construction des scenarios",
                [
                    python,
                    "-m",
                    "unittest",
                    "tests.moteur.test_add_bsv_compatible_metadata",
                ],
            )
            run(
                "B3",
                "Tests requete, CNN et analyse",
                [
                    python,
                    "-m",
                    "unittest",
                    "tests.moteur.test_build_phase3_query",
                    "tests.moteur.test_inference_cnn",
                    "tests.moteur.test_phase3_prepare_analysis",
                ],
            )

        if selected("C"):
            announce_phase("C")
            images_exist = (PHASE3_DIR / "images_unlabeled").is_dir()
            run(
                "C1",
                "Construction du manifeste images",
                [python, "src/moteur/build_manifest_images.py"],
                skip=images_exist,
            )
            record_validation(
                "C1-check",
                "Validation du manifeste images",
                lambda: validate_manifest(expected),
            )
            run(
                "C2",
                "Construction des contextes BSV",
                [
                    python,
                    "src/moteur/add_bsv_compatible_metadata.py",
                    "--seed",
                    str(seed),
                ],
                lambda: validate_assignments(expected, seed),
            )

        if selected("D"):
            announce_phase("D")
            inference = [python, "src/moteur/inference_cnn.py", "--top-k", str(top_k)]
            if args.force_cnn:
                inference.append("--no-reuse")
            run(
                "D1", "Inference CNN", inference, lambda: validate_predictions(expected)
            )
            run(
                "D2",
                "Construction de la requete aveugle",
                [python, "src/moteur/build_phase3_query.py", "--top-k", str(top_k)],
                lambda: validate_query(expected, top_k),
            )
            run(
                "D3",
                "Moteur de compatibilite",
                [
                    python,
                    "src/moteur/moteur_compatibilite.py",
                    "--query-file",
                    "outputs/moteur/phase3/requete_phase3.json",
                    "--output-file",
                    "data/processed/moteur/bsv_compatibility_phase3.csv",
                ],
                lambda: validate_engine(expected),
            )

        if selected("E"):
            announce_phase("E")
            run(
                "E1",
                "Preparation des analyses Phase 3",
                [
                    python,
                    "src/moteur/phase3_prepare_analysis.py",
                    "--expected-images",
                    str(expected),
                ],
                lambda: validate_analysis(expected),
            )
            run(
                "E2",
                "Notebook d'analyse",
                [
                    python,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "notebooks/moteur/phase3_analysis.ipynb",
                    "--ExecutePreprocessor.timeout=180",
                ],
                lambda: validate_notebook(
                    PROJECT_ROOT / "notebooks/moteur/phase3_analysis.ipynb"
                ),
                skip=args.skip_notebooks,
            )
            run(
                "E3",
                "Notebook descriptif",
                [
                    python,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "notebooks/moteur/phase3_visualisation.ipynb",
                    "--ExecutePreprocessor.timeout=180",
                ],
                lambda: validate_notebook(
                    PROJECT_ROOT / "notebooks/moteur/phase3_visualisation.ipynb"
                ),
                skip=args.skip_notebooks,
            )
            run(
                "E4",
                "Suite complete de tests",
                [python, "-m", "unittest", "discover", "-s", "tests/moteur"],
            )
    except Exception as exc:
        status = "FAILED"
        error = str(exc)
        emit(f"PIPELINE FAILED: {error}")
    finally:
        report = {
            "pipeline": "phase3_phytosanitaire",
            "status": status,
            "started_from_phase": args.from_phase,
            "ended_at_phase": args.to_phase,
            "dry_run": args.dry_run,
            "configuration": {
                **config,
                "expected_images": expected,
                "seed": seed,
                "top_k": top_k,
            },
            "error": error,
            "steps": [asdict(result) for result in results],
            "log_file": str(log_path),
            "latest_log_file": str(latest_log_path),
        }
        report_text = json.dumps(report, ensure_ascii=False, indent=2)
        report_path.write_text(report_text, encoding="utf-8")
        latest_report_path.write_text(report_text, encoding="utf-8")
        counts = {
            state: sum(result.status == state for result in results)
            for state in ("SUCCESS", "FAILED", "SKIPPED", "DRY_RUN")
        }
        emit("")
        emit("=== BILAN FINAL ===")
        emit(f"Statut global: {status}")
        emit(
            "Etapes: "
            f"{counts['SUCCESS']} reussies, {counts['FAILED']} echouees, "
            f"{counts['SKIPPED']} ignorees, {counts['DRY_RUN']} simulees"
        )
        if error:
            emit(f"COMMENTAIRE D'ERREUR: {error}")
        else:
            emit(
                "COMMENTAIRE: toutes les etapes demandees ont termine sans anomalie bloquante."
            )
        emit(f"Journal horodate: {log_path}")
        emit(f"Journal courant : {latest_log_path}")
        emit(f"Rapport JSON: {report_path}")
    return 0 if status == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
