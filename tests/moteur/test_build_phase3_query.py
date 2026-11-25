from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import build_phase3_query


class BuildPhase3QueryTests(unittest.TestCase):
    def test_can_keep_top1_and_skips_failed_inference(self) -> None:
        records = [
            {
                "observation_id": "obs_000001",
                "date_observation": "2025-07-18",
                "region": "PACA",
                "statut_bsv": "classe_saine",
                "classe_cnn_source": "tomato_healthy",
                "bioagresseur_bsv": "Sain",
                "type_bioagresseur_bsv": "sain",
                "inference_ok": True,
                "predictions": [
                    {"rank": 1, "label": "tomato_late_blight", "score": 0.8},
                    {"rank": 2, "label": "tomato_healthy", "score": 0.1},
                ],
            },
            {
                "observation_id": "obs_000002",
                "date_observation": "2025-07-18",
                "region": "PACA",
                "inference_ok": False,
                "predictions": [
                    {"rank": 1, "label": "tomato_healthy", "score": 0.7},
                ],
            },
        ]

        queries = build_phase3_query.build_queries(records, top_k=1)

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["image_id"], "obs_000001")
        self.assertNotIn("statut_bsv", queries[0])
        self.assertNotIn("classe_cnn_source", queries[0])
        self.assertNotIn("bioagresseur_bsv", queries[0])
        self.assertNotIn("type_bioagresseur_bsv", queries[0])
        self.assertEqual(
            set(queries[0]), {"image_id", "date_observation", "region", "top_k"}
        )
        self.assertEqual(len(queries[0]["top_k"]), 1)
        self.assertEqual(
            queries[0]["top_k"][0],
            {"classe": "tomato_late_blight", "score": 0.8},
        )

    def test_can_keep_more_predictions_when_requested(self) -> None:
        records = [
            {
                "observation_id": "obs_000001",
                "date_observation": "2025-07-18",
                "region": "PACA",
                "inference_ok": True,
                "predictions": [
                    {"rank": 1, "label": "a", "score": 0.8},
                    {"rank": 2, "label": "b", "score": 0.1},
                    {"rank": 3, "label": "c", "score": 0.1},
                ],
            }
        ]

        queries = build_phase3_query.build_queries(records, top_k=2)

        self.assertEqual([item["classe"] for item in queries[0]["top_k"]], ["a", "b"])

    def test_default_keeps_top_three_and_strips_all_extra_fields(self) -> None:
        records = [
            {
                "observation_id": "obs_000001",
                "date_observation": "2025-07-18",
                "region": "PACA",
                "true_label": "secret",
                "scenario_assigne": "favorable",
                "scenario_realisable": True,
                "inference_ok": True,
                "predictions": [
                    {"rank": rank, "label": label, "score": 1 / rank, "secret": "x"}
                    for rank, label in enumerate(("a", "b", "c"), start=1)
                ],
            }
        ]

        query = build_phase3_query.build_queries(records)[0]

        self.assertEqual(
            set(query), {"image_id", "date_observation", "region", "top_k"}
        )
        self.assertEqual(len(query["top_k"]), 3)
        self.assertTrue(
            all(set(prediction) == {"classe", "score"} for prediction in query["top_k"])
        )


if __name__ == "__main__":
    unittest.main()
