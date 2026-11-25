from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

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
    def test_weather_is_kept_informative_only(self) -> None:
        signal, status = moteur.extraire_meteo_favorable(
            "Temps humide avec des pluies régulières.",
            "tomato_late_blight",
        )

        self.assertIsNone(signal)
        self.assertEqual(status, "informatif")


class LiteratureWindowTests(unittest.TestCase):
    def test_late_blight_is_plausible_during_summer(self) -> None:
        in_window, window, source = moteur.expected_window_for_prediction(
            date(2025, 7, 18),
            "tomato_late_blight",
        )

        self.assertTrue(in_window)
        self.assertEqual(window, (date(2025, 4, 1), date(2025, 10, 31)))
        self.assertEqual(source, "litterature")

    def test_spider_mites_are_plausible_until_december(self) -> None:
        in_window, window, source = moteur.expected_window_for_prediction(
            date(2025, 12, 15),
            "tomato_spider_mites_two_spotted_spider_mite",
        )

        self.assertTrue(in_window)
        self.assertEqual(window, (date(2025, 4, 1), date(2025, 12, 31)))
        self.assertEqual(source, "litterature")

    def test_late_blight_is_not_plausible_during_winter(self) -> None:
        in_window, _, source = moteur.expected_window_for_prediction(
            date(2025, 1, 15),
            "tomato_late_blight",
        )

        self.assertFalse(in_window)
        self.assertEqual(source, "litterature")

    def test_leaf_blight_has_no_operational_window(self) -> None:
        in_window, window, source = moteur.expected_window_for_prediction(
            date(2025, 7, 18),
            "tomato_leaf_blight",
        )

        self.assertFalse(in_window)
        self.assertIsNone(window)
        self.assertEqual(source, "non_definie_litterature")

    def test_healthy_has_no_literature_window(self) -> None:
        in_window, window, source = moteur.expected_window_for_prediction(
            date(2025, 7, 18),
            "tomato_healthy",
        )

        self.assertFalse(in_window)
        self.assertIsNone(window)
        self.assertEqual(source, "non_definie_litterature")


class RiskTests(unittest.TestCase):
    def test_extracts_strongest_risk(self) -> None:
        rows = [
            {"observation": "Risque faible."},
            {"analyse_risque": "Le risque est eleve."},
        ]

        self.assertEqual(moteur.extraire_niveau_risque(rows), "fort")


class QueryLoadingTests(unittest.TestCase):
    def test_load_queries_drops_experimental_truth_fields(self) -> None:
        payload = """[
          {
            "observation_id": "obs_000001",
            "date_observation": "2025-07-18",
            "region": "PACA",
            "statut_bsv": "compatible_bsv",
            "classe_cnn_source": "tomato_late_blight",
            "bioagresseur_bsv": "Mildiou",
            "type_bioagresseur_bsv": "maladie",
            "predictions": [
              {"label": "tomato_late_blight", "score": 0.8}
            ]
          }
        ]"""
        path = Path(__file__).resolve().parent / "_tmp_query_leakage_test.json"
        try:
            path.write_text(payload, encoding="utf-8")
            query = moteur.load_queries(path)[0]
        finally:
            path.unlink(missing_ok=True)

        self.assertFalse(hasattr(query, "statut_bsv"))
        self.assertFalse(hasattr(query, "classe_cnn_source"))
        self.assertFalse(hasattr(query, "bioagresseur_bsv"))
        self.assertFalse(hasattr(query, "type_bioagresseur_bsv"))
        self.assertEqual(query.top_k[0].classe, "tomato_late_blight")


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
        status, _, evidence = self.decide()
        self.assertEqual(status, "Coherent")
        self.assertEqual(evidence, "direct")

    def test_direct_low_risk_is_partial(self) -> None:
        status, _, evidence = self.decide(niveau_risque="faible")
        self.assertEqual(status, "Coherent")
        self.assertEqual(evidence, "direct")

    def test_direct_strong_risk_is_reinforced(self) -> None:
        status, _, evidence = self.decide(niveau_risque="fort")
        self.assertEqual(status, "Coherent")
        self.assertEqual(evidence, "direct")

    def test_host_only_is_partial(self) -> None:
        status, _, evidence = self.decide(
            agent_rows_bulletin=[self.host_row],
            direct_tomato_rows_bulletin=[],
            tomato_compatible_rows_bulletin=[self.host_row],
        )
        self.assertEqual(status, "Coherent")
        self.assertEqual(evidence, "hote_associe")

    def test_no_direct_tomato_but_host_signal_is_partial(self) -> None:
        status, _, evidence = self.decide(
            agent_rows_bulletin=[self.host_row],
            direct_tomato_rows_bulletin=[],
            tomato_compatible_rows_bulletin=[self.host_row],
            meteo_status="favorable",
        )
        self.assertEqual(status, "Coherent")
        self.assertEqual(evidence, "hote_associe")

    def test_absent_agent_favorable_weather_is_partial(self) -> None:
        status, _, evidence = self.decide(
            agent_rows_bulletin=[],
            niveau_risque=None,
            meteo_status="favorable",
        )
        self.assertEqual(status, "Non_corrobore")
        self.assertEqual(evidence, "absent")

    def test_absent_agent_neutral_weather_is_not_corrobated(self) -> None:
        status, _, evidence = self.decide(
            agent_rows_bulletin=[],
            niveau_risque=None,
            meteo_status="neutre",
        )
        self.assertEqual(status, "Non_corrobore")
        self.assertEqual(evidence, "absent")

    def test_missing_weather_does_not_block_decision(self) -> None:
        status, _, evidence = self.decide(meteo_status="non_utilisee")
        self.assertEqual(status, "Coherent")
        self.assertEqual(evidence, "direct")

    def test_outside_season_is_indeterminate(self) -> None:
        status, _, evidence = self.decide(dans_fenetre_saisonniere=False)
        self.assertEqual(status, "Indetermine")
        self.assertEqual(evidence, "hors_periode_litterature")

    def test_missing_bulletin_is_indeterminate(self) -> None:
        status, _, evidence = self.decide(bulletin_present=False)
        self.assertEqual(status, "Indetermine")
        self.assertEqual(evidence, "sans_bulletin")

    def test_noncovered_is_structural_even_outside_period(self) -> None:
        status, _, evidence = self.decide(
            agent_couvert_corpus=False,
            dans_fenetre_saisonniere=False,
            bulletin_present=False,
        )
        self.assertEqual(status, "Indetermine")
        self.assertEqual(evidence, "non_couvert")


class HealthyPredictionTests(unittest.TestCase):
    def make_data(self, bulletins=None, observations_by_bulletin=None):
        bulletins = bulletins or []
        observations_by_bulletin = observations_by_bulletin or {}
        bulletins_by_region = {}
        for bulletin in bulletins:
            bulletins_by_region.setdefault(
                moteur.region_key(bulletin["region"]), []
            ).append(bulletin)

        return moteur.EngineData(
            bulletins=bulletins,
            observations=[],
            referential_by_class={},
            bulletins_by_region=bulletins_by_region,
            observations_by_bulletin=observations_by_bulletin,
            covered_classes=set(),
            covered_agents=set(),
        )

    def test_healthy_without_bulletin_is_indeterminate(self) -> None:
        query = moteur.CompatibilityQuery(
            image_id="obs_healthy_no_bulletin",
            date_observation=date(2025, 7, 10),
            region="PACA",
            top_k=[moteur.Prediction("tomato_healthy", 0.9)],
        )

        rows = moteur.evaluer_requete(query, self.make_data())

        self.assertEqual(rows[0]["compatibilite"], "Indetermine")
        self.assertEqual(rows[0]["niveau_preuve"], "sans_bulletin")

    def test_healthy_with_bulletin_is_not_filtered_by_literature_window(self) -> None:
        bulletin = {
            "bulletin_id": "b1",
            "region": "PACA",
            "date_bulletin_parsed": date(2025, 12, 20),
            "prevision_meteo": "",
            "meteo_extraction_status": "lisible",
        }
        query = moteur.CompatibilityQuery(
            image_id="obs_healthy_winter",
            date_observation=date(2025, 12, 25),
            region="PACA",
            top_k=[moteur.Prediction("tomato_healthy", 0.9)],
        )

        rows = moteur.evaluer_requete(query, self.make_data(bulletins=[bulletin]))

        self.assertEqual(rows[0]["compatibilite"], "Coherent")
        self.assertEqual(rows[0]["niveau_preuve"], "direct")


if __name__ == "__main__":
    unittest.main()
