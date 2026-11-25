from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import inference_cnn


class InferenceReuseTests(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def metadata(self, image_path: Path) -> pd.Series:
        return pd.Series(
            {
                "observation_id": "obs_000001",
                "image_path": str(image_path),
                "date_observation": "2025-07-01",
                "region": "Grand Est",
                "departement": "Bas-Rhin",
            }
        )

    def test_reuses_only_identical_pixels_and_removes_experimental_fields(self) -> None:
        old = Path("fixture_previous.jpg")
        new = Path("fixture_current.jpg")
        previous = {
            "observation_id": "obs_000001",
            "image_path": str(old),
            "classe_cnn_source": "secret",
            "scenario_assigne": "favorable",
            "predictions": [{"rank": 1, "label": "class_a", "score": 0.9}],
            "inference_ok": True,
        }

        # Test unitaire autonome : la phase B ne doit dependre d'aucun artefact
        # produit plus tard par la phase C. Les deux empreintes simulent ici des
        # fichiers existants dont les pixels sont strictement identiques.
        with patch.object(
            inference_cnn.Path, "is_file", return_value=True
        ), patch.object(inference_cnn, "sha256_file", return_value="same-sha256"):
            reused = inference_cnn.record_reutilisable(self.metadata(new), previous, {})

        self.assertIsNotNone(reused)
        self.assertEqual(set(reused), inference_cnn.OUTPUT_KEYS)
        self.assertNotIn("classe_cnn_source", reused)
        self.assertNotIn("scenario_assigne", reused)

    def test_rejects_reuse_when_pixels_differ(self) -> None:
        old = Path("fixture_previous.jpg")
        new = Path("fixture_current.jpg")
        previous = {
            "image_path": str(old),
            "predictions": [{"rank": 1, "label": "class_a", "score": 0.9}],
            "inference_ok": True,
        }

        with patch.object(
            inference_cnn.Path, "is_file", return_value=True
        ), patch.object(
            inference_cnn, "sha256_file", side_effect=lambda path: str(path)
        ):
            self.assertIsNone(
                inference_cnn.record_reutilisable(self.metadata(new), previous, {})
            )


if __name__ == "__main__":
    unittest.main()
