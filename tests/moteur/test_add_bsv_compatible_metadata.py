from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import add_bsv_compatible_metadata as metadata
import moteur_compatibilite as moteur


class ScenarioDrawTests(unittest.TestCase):
    def test_exact_target_quotas_and_reproducibility(self) -> None:
        first = metadata.tirer_scenarios(3761, seed=2026)
        second = metadata.tirer_scenarios(3761, seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(first.count("favorable"), 1505)
        self.assertEqual(first.count("non_favorable"), 1504)
        self.assertEqual(first.count("insuffisant"), 752)

    def test_draw_does_not_receive_classes_or_cnn_predictions(self) -> None:
        scenarios = metadata.tirer_scenarios(4, seed=1)
        self.assertEqual(len(scenarios), 4)


class MetadataConstructionTests(unittest.TestCase):
    def make_data(self) -> moteur.EngineData:
        bulletins = [
            {
                "bulletin_id": "b_fav",
                "region": "Grand Est",
                "date_bulletin_parsed": date(2025, 7, 1),
            },
            {
                "bulletin_id": "b_nonfav",
                "region": "Grand Est",
                "date_bulletin_parsed": date(2025, 7, 8),
            },
        ]
        direct_agent = {
            "bulletin_id": "b_fav",
            "culture": "tomate",
            "classe_dataset": "tomato_late_blight",
            "agent_resolu": "Phytophthora infestans",
            "resolution_status": "resolved_direct",
        }
        other_agent = {
            "bulletin_id": "b_nonfav",
            "culture": "tomate",
            "classe_dataset": "tomato_early_blight",
            "agent_resolu": "Alternaria solani",
            "resolution_status": "resolved_direct",
        }
        return moteur.EngineData(
            bulletins=bulletins,
            observations=[direct_agent, other_agent],
            referential_by_class={
                "tomato_late_blight": {"agent": "Phytophthora infestans"},
                "tomato_early_blight": {"agent": "Alternaria solani"},
            },
            bulletins_by_region={"grand est": bulletins},
            observations_by_bulletin={
                "b_fav": [direct_agent],
                "b_nonfav": [other_agent],
            },
            covered_classes={"tomato_late_blight", "tomato_early_blight"},
            covered_agents={"phytophthora infestans", "alternaria solani"},
        )

    def source(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "observation_id": f"obs_{index:06d}",
                    "image_path": f"image_{index}.jpg",
                    "label_masked": True,
                    "true_label": "tomato_late_blight",
                    "cnn_prediction": "must_not_be_used",
                    "score_cnn": index / 10,
                }
                for index in range(1, 11)
            ]
        )

    def test_keeps_every_image_and_adds_auditable_fields(self) -> None:
        result = metadata.construire_lignes_metadata(
            self.source(), self.make_data(), seed=7
        )

        self.assertEqual(len(result), 10)
        self.assertEqual(result["observation_id"].nunique(), 10)
        for column in metadata.ASSIGNMENT_COLUMNS:
            self.assertIn(column, result.columns)

    def test_same_seed_is_reproducible_and_cnn_values_do_not_matter(self) -> None:
        source_a = self.source()
        source_b = self.source()
        source_b["cnn_prediction"] = "another_prediction"
        source_b["score_cnn"] = 0.999

        first = metadata.construire_lignes_metadata(source_a, self.make_data(), seed=9)
        second = metadata.construire_lignes_metadata(source_b, self.make_data(), seed=9)

        pd.testing.assert_frame_equal(
            first[metadata.ASSIGNMENT_COLUMNS],
            second[metadata.ASSIGNMENT_COLUMNS],
        )

    def test_favorable_and_non_favorable_follow_bulletin_content(self) -> None:
        catalogue = metadata.construire_catalogue_contextes(self.make_data())

        self.assertEqual(
            catalogue["tomato_late_blight"]["favorable"][0].bulletin_id, "b_fav"
        )
        self.assertEqual(
            catalogue["tomato_late_blight"]["non_favorable"][0].bulletin_id,
            "b_nonfav",
        )

    def test_insufficient_context_has_an_explicit_controlled_cause(self) -> None:
        result = metadata.construire_lignes_metadata(
            self.source().iloc[:1],
            self.make_data(),
            seed=4,
            proportions={"favorable": 0, "non_favorable": 0, "insuffisant": 1},
        )

        self.assertEqual(result.iloc[0]["scenario_realise"], "insuffisant")
        self.assertIn(
            result.iloc[0]["cause_insuffisance"],
            {"sans_bulletin", "hors_periode_litterature"},
        )
        self.assertEqual(result.iloc[0]["bulletin_id_assigne"], "")

    def test_noncovered_class_uses_structural_insufficiency(self) -> None:
        source = self.source().iloc[:1].copy()
        source["true_label"] = "tomato_target_spot"
        result = metadata.construire_lignes_metadata(
            source,
            self.make_data(),
            seed=4,
            proportions={"favorable": 0, "non_favorable": 0, "insuffisant": 1},
        )

        self.assertEqual(
            result.iloc[0]["cause_insuffisance"], "classe_non_couverte_bsv"
        )
        self.assertNotEqual(result.iloc[0]["bulletin_id_assigne"], "")


if __name__ == "__main__":
    unittest.main()
