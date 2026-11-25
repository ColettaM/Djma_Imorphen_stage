from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import pipeline_config
import run_pipeline as pipeline


class PipelineConfigTests(unittest.TestCase):
    def test_shared_configuration_uses_fourteen_day_window(self) -> None:
        config = pipeline_config.load_pipeline_config()
        self.assertEqual(config["max_bulletin_age_days"], 14)
        self.assertEqual(config["expected_images"], 3761)


class PipelineValidationTests(unittest.TestCase):
    def test_query_validator_accepts_only_closed_blind_schema(self) -> None:
        query = [
            {
                "image_id": "obs_1",
                "date_observation": "2025-06-01",
                "region": "Grand Est",
                "top_k": [{"classe": "tomato_healthy", "score": 0.8}],
            }
        ]
        with patch.object(pipeline, "read_json_list", return_value=query):
            self.assertIn("aveugles", pipeline.validate_query(1, 3))

        leaked_query = [{**query[0], "true_label": "tomato_healthy"}]
        with patch.object(pipeline, "read_json_list", return_value=leaked_query):
            with self.assertRaisesRegex(pipeline.PipelineError, "Schema requete"):
                pipeline.validate_query(1, 3)

    def test_analysis_validator_blocks_cause_proof_mismatch(self) -> None:
        responses = [
            [{"check": "base", "passed": "True", "detail": ""}],
            [{"image_id": "obs_1"}],
            [
                {
                    "cnn_correct": "True",
                    "cause_coherente_moteur": "False",
                    "nb_images": "1",
                }
            ],
        ]
        with patch.object(pipeline, "read_csv_rows", side_effect=responses):
            with self.assertRaisesRegex(pipeline.PipelineError, "Cause insuffisance"):
                pipeline.validate_analysis(1)


if __name__ == "__main__":
    unittest.main()
