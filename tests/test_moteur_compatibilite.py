from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import moteur_compatibilite as moteur


class BulletinSelectionTests(unittest.TestCase):
    def test_selects_latest_bulletin_not_after_observation(self) -> None:
        bulletins = {
            "grand est": [
                {"bulletin_id": "b1", "date_bulletin_parsed": date(2025, 5, 1)},
                {"bulletin_id": "b2", "date_bulletin_parsed": date(2025, 5, 8)},
                {"bulletin_id": "b3", "date_bulletin_parsed": date(2025, 5, 15)},
            ]
        }

        selected = moteur.trouver_bulletin_retenu(
            bulletins,
            "Grand Est",
            date(2025, 5, 10),
        )

        self.assertEqual(selected["bulletin_id"], "b2")

    def test_returns_none_before_first_bulletin(self) -> None:
        bulletins = {
            "grand est": [
                {"bulletin_id": "b1", "date_bulletin_parsed": date(2025, 5, 1)},
            ]
        }

        selected = moteur.trouver_bulletin_retenu(
            bulletins,
            "Grand Est",
            date(2025, 4, 30),
        )

        self.assertIsNone(selected)


class WeatherTests(unittest.TestCase):
    def test_empty_weather_is_indeterminable(self) -> None:
        signal, status = moteur.extraire_meteo_favorable("", "late_blight")

        self.assertIsNone(signal)
        self.assertEqual(status, "non_utilisee")

    def test_image_only_weather_is_indeterminable(self) -> None:
        signal, status = moteur.extraire_meteo_favorable(
            "Prévisions à 7 jours, source Météo France.",
            "late_blight",
            "image_seule",
        )

        self.assertIsNone(signal)
        self.assertEqual(status, "non_utilisee")

    def test_too_long_weather_is_indeterminable(self) -> None:
        signal, status = moteur.extraire_meteo_favorable(
            "Pluies annoncées.",
            "late_blight",
            "trop_longue",
        )

        self.assertIsNone(signal)
        self.assertEqual(status, "non_utilisee")

    def test_readable_weather_without_keyword_is_neutral(self) -> None:
        signal, status = moteur.extraire_meteo_favorable(
            "Prévisions disponibles pour les prochains jours.",
            "late_blight",
        )

        self.assertIsNone(signal)
        self.assertEqual(status, "neutre")

    def test_wet_weather_favors_late_blight(self) -> None:
        signal, status = moteur.extraire_meteo_favorable(
            "Temps humide avec des pluies régulières.",
            "late_blight",
        )

        self.assertTrue(signal)
        self.assertEqual(status, "favorable")

    def test_dry_weather_is_unfavorable_to_late_blight(self) -> None:
        signal, status = moteur.extraire_meteo_favorable(
            "Temps sec et ensoleillé.",
            "late_blight",
        )

        self.assertFalse(signal)
        self.assertEqual(status, "defavorable")


class RiskTests(unittest.TestCase):
    def test_extracts_strongest_risk(self) -> None:
        rows = [
            {"observation": "Risque faible."},
            {"analyse_risque": "Le risque est eleve."},
        ]

        self.assertEqual(moteur.extraire_niveau_risque(rows), "fort")


class DecisionTests(unittest.TestCase):
    direct_row = {"resolution_status": "resolved_direct"}
    host_row = {"resolution_status": "resolved_hote_associe"}

    def decide(self, **overrides):
        values = {
            "bulletin_present": True,
            "dans_fenetre_saisonniere": True,
            "agent_couvert_corpus": True,
            "agent_rows_bulletin": [self.direct_row],
            "direct_tomato_rows_bulletin": [self.direct_row],
            "tomato_compatible_rows_bulletin": [self.direct_row],
            "niveau_risque": "moyen",
            "meteo_status": "neutre",
        }
        values.update(overrides)
        return moteur.appliquer_regle_decision(**values)

    def test_direct_medium_risk_neutral_weather_is_agreement(self) -> None:
        status, _ = self.decide()
        self.assertEqual(status, "Coherent")

    def test_direct_low_risk_is_partial(self) -> None:
        status, _ = self.decide(niveau_risque="faible")
        self.assertEqual(status, "Partiel")

    def test_direct_strong_risk_is_reinforced(self) -> None:
        status, _ = self.decide(niveau_risque="fort")
        self.assertEqual(status, "Coherent_renforce")

    def test_host_only_is_partial(self) -> None:
        status, _ = self.decide(
            agent_rows_bulletin=[self.host_row],
            direct_tomato_rows_bulletin=[],
            tomato_compatible_rows_bulletin=[self.host_row],
        )
        self.assertEqual(status, "Partiel")

    def test_no_direct_tomato_but_host_signal_is_partial(self) -> None:
        status, _ = self.decide(
            agent_rows_bulletin=[self.host_row],
            direct_tomato_rows_bulletin=[],
            tomato_compatible_rows_bulletin=[self.host_row],
            meteo_status="favorable",
        )
        self.assertEqual(status, "Partiel")

    def test_absent_agent_favorable_weather_is_partial(self) -> None:
        status, _ = self.decide(
            agent_rows_bulletin=[],
            niveau_risque=None,
            meteo_status="favorable",
        )
        self.assertEqual(status, "Non_corrobore")

    def test_absent_agent_neutral_weather_is_not_corrobated(self) -> None:
        status, _ = self.decide(
            agent_rows_bulletin=[],
            niveau_risque=None,
            meteo_status="neutre",
        )
        self.assertEqual(status, "Non_corrobore")

    def test_missing_weather_does_not_block_decision(self) -> None:
        status, _ = self.decide(meteo_status="non_utilisee")
        self.assertEqual(status, "Coherent")

    def test_outside_season_is_indeterminate(self) -> None:
        status, _ = self.decide(dans_fenetre_saisonniere=False)
        self.assertEqual(status, "Indetermine")


if __name__ == "__main__":
    unittest.main()
