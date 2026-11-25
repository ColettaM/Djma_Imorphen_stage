from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import phase3_prepare_analysis as prepare


class PrepareAnalysisTests(unittest.TestCase):
    def fixtures(self):
        assignments = pd.DataFrame(
            [
                {
                    "observation_id": "a",
                    "true_label": "class_a",
                    "scenario_assigne": "favorable",
                    "scenario_realise": "favorable",
                    "scenario_realisable": True,
                    "raison_non_realisation": "",
                    "cause_insuffisance": "",
                    "classe_couverte_bsv": True,
                    "context_id": "ctx_a",
                    "bulletin_id_assigne": "b1",
                    "date_observation": "2025-07-01",
                    "region": "R",
                    "departement": "D",
                    "seed": 2026,
                },
                {
                    "observation_id": "b",
                    "true_label": "class_b",
                    "scenario_assigne": "non_favorable",
                    "scenario_realise": "non_favorable",
                    "scenario_realisable": True,
                    "raison_non_realisation": "",
                    "cause_insuffisance": "",
                    "classe_couverte_bsv": True,
                    "context_id": "ctx_b",
                    "bulletin_id_assigne": "b2",
                    "date_observation": "2025-07-02",
                    "region": "R",
                    "departement": "D",
                    "seed": 2026,
                },
            ]
        )
        predictions = [
            {
                "observation_id": image,
                "inference_ok": True,
                "predictions": [
                    {"rank": 1, "label": first, "score": 0.4},
                    {"rank": 2, "label": second, "score": 0.9},
                ],
            }
            for image, first, second in (
                ("a", "class_a", "class_b"),
                ("b", "class_a", "class_b"),
            )
        ]
        query = [
            {
                "image_id": image,
                "date_observation": day,
                "region": "R",
                "top_k": [
                    {"classe": first, "score": 0.4},
                    {"classe": second, "score": 0.9},
                ],
            }
            for image, day, first, second in (
                ("a", "2025-07-01", "class_a", "class_b"),
                ("b", "2025-07-02", "class_a", "class_b"),
            )
        ]
        verdicts = pd.DataFrame(
            [
                {
                    "image_id": image,
                    "date_observation": day,
                    "region": "R",
                    "classe_cnn_predite": classe,
                    "score_cnn": score,
                    "compatibilite": verdict,
                    "niveau_preuve": "direct" if verdict == "Coherent" else "absent",
                    "niveau_risque": "moyen",
                }
                for image, day, classe, score, verdict in (
                    ("a", "2025-07-01", "class_a", 0.4, "Coherent"),
                    ("a", "2025-07-01", "class_b", 0.9, "Non_corrobore"),
                    ("b", "2025-07-02", "class_a", 0.4, "Non_corrobore"),
                    ("b", "2025-07-02", "class_b", 0.9, "Coherent"),
                )
            ]
        )
        return assignments, predictions, query, verdicts

    def test_top1_follows_declared_order_not_maximum_score(self) -> None:
        full, checks = prepare.prepare_top1_table(*self.fixtures(), expected_images=2)

        self.assertEqual(full["classe_cnn_top1"].tolist(), ["class_a", "class_a"])
        self.assertEqual(full["score_cnn_top1"].tolist(), [0.4, 0.4])
        self.assertTrue(checks["passed"].all())

    def test_builds_covered_and_primary_populations(self) -> None:
        full, _ = prepare.prepare_top1_table(*self.fixtures(), expected_images=2)
        covered, primary = prepare.build_populations(full)

        self.assertEqual(len(covered), 2)
        self.assertEqual(len(primary), 2)
        self.assertEqual(primary["cnn_correct"].tolist(), [True, False])

    def test_rejects_experimental_field_in_engine_query(self) -> None:
        assignments, predictions, query, verdicts = self.fixtures()
        query[0]["scenario_assigne"] = "favorable"

        with self.assertRaisesRegex(ValueError, "query_schema_closed"):
            prepare.prepare_top1_table(
                assignments, predictions, query, verdicts, expected_images=2
            )

    def test_separation_diagnostics_flags_zero_outcome_level(self) -> None:
        full, _ = prepare.prepare_top1_table(*self.fixtures(), expected_images=2)
        _, primary = prepare.build_populations(full)

        diagnostics = prepare.separation_diagnostics(primary)

        self.assertTrue(diagnostics["separation"].any())

    def test_nested_logistic_analysis_detects_incremental_verdict(self) -> None:
        rows = []
        for group in range(20):
            for index in range(20):
                score = 0.25 + 0.02 * (index % 10)
                coherent = index % 2 == 0
                correct = coherent or index % 7 != 0
                rows.append(
                    {
                        "cnn_correct": correct,
                        "score_cnn_top1": score,
                        "verdict_bsv_top1": "Coherent" if coherent else "Non_corrobore",
                        "bulletin_id": f"b{group}",
                    }
                )
        primary = pd.DataFrame(rows)

        coefficients, comparison, predictions = prepare.nested_logistic_analysis(
            primary
        )

        self.assertEqual(int(comparison.loc[0, "n_observations_m0"]), len(primary))
        self.assertEqual(int(comparison.loc[0, "n_observations_m1"]), len(primary))
        self.assertGreater(comparison.loc[0, "adjusted_odds_ratio"], 1.0)
        self.assertEqual(len(predictions), len(primary))
        self.assertIn("verdict_coherent", coefficients["term"].tolist())

    def test_grouped_cross_validation_keeps_bulletins_together(self) -> None:
        rows = []
        for group in range(10):
            for index in range(12):
                rows.append(
                    {
                        "cnn_correct": index != 0,
                        "score_cnn_top1": 0.2 + 0.05 * (index % 8),
                        "verdict_bsv_top1": "Coherent"
                        if index % 2
                        else "Non_corrobore",
                        "bulletin_id": f"b{group}",
                    }
                )
        folds, summary = prepare.grouped_cross_validation(pd.DataFrame(rows), folds=5)

        self.assertEqual(folds["fold"].nunique(), 5)
        self.assertEqual(set(folds["model"]), {"M0", "M1"})
        self.assertEqual(int(summary.loc[0, "n_bulletins"]), 10)

    def test_grouped_bootstrap_is_reproducible_and_returns_or_interval(self) -> None:
        rows = []
        for group in range(12):
            for index in range(16):
                rows.append(
                    {
                        "cnn_correct": index not in {0, 7, (group % 5) + 1},
                        "score_cnn_top1": 0.3 + 0.04 * (index % 10),
                        "verdict_bsv_top1": "Coherent"
                        if index % 3
                        else "Non_corrobore",
                        "bulletin_id": f"b{group}",
                    }
                )
        primary = pd.DataFrame(rows)

        bootstrap, summary = prepare.grouped_bulletin_bootstrap(
            primary, iterations=25, seed=42
        )
        bootstrap_again, _ = prepare.grouped_bulletin_bootstrap(
            primary, iterations=25, seed=42
        )

        self.assertEqual(len(bootstrap), 25)
        pd.testing.assert_frame_equal(bootstrap, bootstrap_again)
        self.assertLess(
            summary.loc[0, "bootstrap_ci95_or_low"],
            summary.loc[0, "bootstrap_ci95_or_high"],
        )

    def test_predicted_class_diagnostics_are_driven_by_errors(self) -> None:
        primary = pd.DataFrame(
            [
                {
                    "classe_cnn_top1": "a",
                    "cnn_correct": True,
                    "verdict_bsv_top1": "Coherent",
                },
                {
                    "classe_cnn_top1": "a",
                    "cnn_correct": False,
                    "verdict_bsv_top1": "Non_corrobore",
                },
                {
                    "classe_cnn_top1": "b",
                    "cnn_correct": True,
                    "verdict_bsv_top1": "Coherent",
                },
            ]
        )

        diagnostics = prepare.predicted_class_diagnostics(primary).set_index(
            "classe_cnn_top1"
        )

        self.assertEqual(int(diagnostics.loc["a", "cnn_incorrect"]), 1)
        self.assertEqual(int(diagnostics.loc["a", "error_verdict_non_corrobore"]), 1)
        self.assertTrue(bool(diagnostics.loc["b", "zero_outcome_cell"]))

    def test_class_adjusted_sensitivity_uses_firth_on_separation(self) -> None:
        rows = []
        for group in range(10):
            for index in range(12):
                predicted_class = "a" if index < 6 else "b"
                correct = True if predicted_class == "b" else index not in {0, 1}
                rows.append(
                    {
                        "cnn_correct": correct,
                        "score_cnn_top1": 0.35 + 0.03 * index,
                        "verdict_bsv_top1": "Coherent"
                        if index % 2
                        else "Non_corrobore",
                        "bulletin_id": f"b{group}",
                        "classe_cnn_top1": predicted_class,
                    }
                )
        primary = pd.DataFrame(rows)

        coefficients, comparison, diagnostics = prepare.class_adjusted_sensitivity(
            primary
        )

        self.assertEqual(set(comparison["comparison"]), {"MC_vs_M2", "M1_vs_M2"})
        self.assertTrue(bool(diagnostics.loc[0, "firth_required"]))
        self.assertTrue((coefficients["method"] == "firth").all())
        self.assertFalse(bool(diagnostics.loc[0, "verdict_class_interaction_fitted"]))


if __name__ == "__main__":
    unittest.main()
