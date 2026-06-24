from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import moteur_compatibilite as moteur


class ResolveAgentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rows = moteur.read_dicts_auto(
            moteur.DEFAULT_REFERENTIAL_FILE,
            delimiter=";",
        )
        cls.referential = moteur.load_referential_entries(rows)

    def resolve(self, culture: str, bioagresseur: str) -> tuple[str, str, str, str]:
        return moteur.resolve_observation(
            {
                "culture": culture,
                "bioagresseur": bioagresseur,
            },
            self.referential,
        )

    def test_mildiou_tomate_resolves_to_late_blight(self) -> None:
        agent, classe_dataset, status, message = self.resolve("tomate", "Mildiou")

        self.assertEqual(agent, "Phytophthora infestans")
        self.assertEqual(classe_dataset, "late_blight")
        self.assertEqual(status, "resolved_direct")
        self.assertEqual(message, "")

    def test_mildiou_pomme_de_terre_resolves_via_associated_host(self) -> None:
        agent, classe_dataset, status, message = self.resolve(
            "pomme de terre", "Mildiou"
        )

        self.assertEqual(agent, "Phytophthora infestans")
        self.assertEqual(classe_dataset, "late_blight")
        self.assertEqual(status, "resolved_hote_associe")
        self.assertEqual(message, "")

    def test_mildiou_salade_is_outside_cnn_scope(self) -> None:
        agent, classe_dataset, status, message = self.resolve("salade", "Mildiou")

        self.assertEqual(agent, "")
        self.assertEqual(classe_dataset, "")
        self.assertEqual(status, "hors_perimetre_cnn")
        self.assertIn("culture/hote non compatible", message)

    def test_unknown_pair_is_unresolved(self) -> None:
        agent, classe_dataset, status, message = self.resolve("carotte", "Inconnu")

        self.assertEqual(agent, "")
        self.assertEqual(classe_dataset, "")
        self.assertEqual(status, "non_resolu")
        self.assertIn("aucune", message)


if __name__ == "__main__":
    unittest.main()
