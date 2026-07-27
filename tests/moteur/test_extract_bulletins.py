from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import extract_bulletins


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_TEXT_DIR = PROJECT_ROOT / "outputs" / "moteur" / "extracted_text"


class MetadataParsingTests(unittest.TestCase):
    def test_grand_est_filename(self) -> None:
        filename = "BSV12_LEG_GE_S26_2025.txt"

        self.assertEqual(extract_bulletins.extract_bulletin_number(filename), 12)

    def test_pays_de_la_loire_filename(self) -> None:
        filename = "16_20250612_BSV-maraichage-n°16.txt"

        self.assertEqual(extract_bulletins.extract_bulletin_number(filename), 16)
        self.assertEqual(
            extract_bulletins.date_from_filename(filename),
            date(2025, 6, 12),
        )

    def test_current_pdl_filename(self) -> None:
        filename = "BSV_maraichage_no_16_20250612.txt"

        self.assertEqual(extract_bulletins.extract_bulletin_number(filename), 16)
        self.assertEqual(
            extract_bulletins.date_from_filename(filename),
            date(2025, 6, 12),
        )

    def test_paca_french_filename(self) -> None:
        filename = "bsv_maraichage_no14_du_18_juillet_2025.txt"

        self.assertEqual(extract_bulletins.extract_bulletin_number(filename), 14)
        self.assertEqual(
            extract_bulletins.date_from_filename(filename),
            date(2025, 7, 18),
        )

    def test_paca_compact_day_month_year_filename(self) -> None:
        filename = "bsv_maraichage_no15_exceptionnel_blettes_25072025.txt"

        self.assertEqual(extract_bulletins.extract_bulletin_number(filename), 15)
        self.assertEqual(
            extract_bulletins.date_from_filename(filename),
            date(2025, 7, 25),
        )

    def test_paca_bulletin_id_uses_date_to_avoid_same_number_collisions(self) -> None:
        first = extract_bulletins.build_bulletin_id(
            "paca",
            2023,
            10,
            "bsv_maraichage_no10_du_26_mai_2023",
        )
        second = extract_bulletins.build_bulletin_id(
            "paca",
            2023,
            10,
            "bsv_mraichage_no10_du_27_mai_2023",
        )

        self.assertEqual(first, "paca_2023_bsv10_20230526")
        self.assertEqual(second, "paca_2023_bsv10_20230527")
        self.assertNotEqual(first, second)

    def test_french_date_from_text(self) -> None:
        text = "BSV n°22 – 10 septembre 2025"

        self.assertEqual(
            extract_bulletins.date_from_text(text),
            date(2025, 9, 10),
        )

    def test_pays_de_la_loire_forecast_week_starts_on_bulletin_date(self) -> None:
        self.assertEqual(
            extract_bulletins.forecast_period(
                date(2025, 8, 7),
                "pays_de_la_loire",
            ),
            "07/08/2025-13/08/2025",
        )

    def test_grand_est_forecast_week_starts_next_day(self) -> None:
        self.assertEqual(
            extract_bulletins.forecast_period(
                date(2025, 8, 7),
                "grand_est",
            ),
            "08/08/2025-14/08/2025",
        )

    def test_explicit_forecast_period(self) -> None:
        self.assertEqual(
            extract_bulletins.extract_explicit_forecast_period(
                "Periode prevue du 07/08/2025 au 13/08/2025."
            ),
            "07/08/2025-13/08/2025",
        )


class WeatherExtractionTests(unittest.TestCase):
    def test_pays_de_la_loire_weather_section(self) -> None:
        text = """Prévision Météorologique P.2
PREVISION METEOROLOGIQUE
Des pluies sont annoncées.
BSV MARAICHAGE N° 16 – DU 12 JUIN 2025
EVALUATION DES RISQUES
Contenu à exclure.
"""

        self.assertEqual(
            extract_bulletins.extract_pays_de_la_loire_weather(text),
            "Des pluies sont annoncées.",
        )

    def test_grand_est_uses_detailed_weather_section(self) -> None:
        text = """DONNÉES MÉTÉO
ASPERGE
Résumé de première page.
--- PAGE 3 ---
DONNÉES MÉTÉO
Prévisions à 7 jours : temps sec.
BSV 12 – P.3
ASPERGE
Contenu à exclure.
"""

        self.assertEqual(
            extract_bulletins.extract_grand_est_weather(text),
            "Prévisions à 7 jours : temps sec.",
        )


class GeneralWeatherExtractionTests(unittest.TestCase):
    def test_prefers_informative_weather_block(self) -> None:
        text = """--- PAGE 1 ---
SOMMAIRE
PREVISION METEOROLOGIQUE
Tomate, aubergine, poivron, salade, chou.
--- PAGE 2 ---
PREVISIONS METEOROLOGIQUES
Des pluies et des températures fraîches sont annoncées.
Ces conditions humides sont favorables aux maladies.
ALLIACEES
Contenu sanitaire à exclure.
"""

        result = extract_bulletins.extract_weather_general(text)

        self.assertEqual(result.status, "lisible")
        self.assertTrue(result.method.startswith("general:"))
        self.assertIn("pluies", result.text)
        self.assertNotIn("Contenu sanitaire", result.text)

    def test_handles_unknown_template(self) -> None:
        text = """PREVISION METEOROLOGIQUE
Temps sec et ensoleillé avec des températures élevées.
RAVAGEURS
Contenu à exclure.
"""

        result = extract_bulletins.extract_weather(text, "inconnu")

        self.assertEqual(result.status, "lisible")
        self.assertEqual(result.method, "general:heading:PREVISION METEOROLOGIQUE")

    def test_marks_image_only_section(self) -> None:
        text = """DONNEES METEO
Prévisions à 7 jours.
Source Météo France, ville de Colmar.
ASPERGE
Contenu à exclure.
"""

        result = extract_bulletins.extract_weather_general(text)

        self.assertEqual(result.status, "image_seule")

    def test_nature_comes_from_retained_block_not_summary(self) -> None:
        text = """--- PAGE 1 ---
SOMMAIRE
BILAN METEOROLOGIQUE P.1
--- PAGE 2 ---
PREVISION METEOROLOGIQUE
Des pluies et des températures fraîches sont annoncées.
ALLIACEES
Contenu à exclure.
"""

        result = extract_bulletins.extract_weather_general(text)

        self.assertEqual(result.nature, "prevision")
        self.assertEqual(result.method, "general:heading:PREVISION METEOROLOGIQUE")


class TemplateProfileIntegrationTests(unittest.TestCase):
    def extract(self, relative_path: str) -> extract_bulletins.Bulletin:
        bulletin, _ = extract_bulletins.extract_bulletin(
            EXTRACTED_TEXT_DIR / relative_path,
            EXTRACTED_TEXT_DIR,
        )
        return bulletin

    def test_pdl_2023_bilan(self) -> None:
        bulletin = self.extract("pdl/maraichage/2023/BSV_maraichage_no_04_20230330.txt")

        self.assertEqual(bulletin.template_id, "pdl_2023_bilan")
        self.assertEqual(bulletin.nature_meteo, "bilan_tendance")
        self.assertEqual(bulletin.meteo_extraction_status, "lisible")
        self.assertTrue(bulletin.meteo_extraction_method.startswith("general:"))
        self.assertGreater(bulletin.meteo_extraction_score, 0)
        self.assertEqual(bulletin.periode_prevision_meteo, "")
        self.assertIn("Weather Measures", bulletin.prevision_meteo)
        self.assertNotIn("ABONNEMENT BSV", bulletin.prevision_meteo)

    def test_pdl_2024_prevision(self) -> None:
        bulletin = self.extract("pdl/maraichage/2024/BSV_maraichage_no_06_20240419.txt")

        self.assertEqual(bulletin.template_id, "pdl_2024_prevision")
        self.assertEqual(bulletin.meteo_extraction_status, "lisible")
        self.assertEqual(
            bulletin.periode_prevision_meteo,
            "19/04/2024-25/04/2024",
        )
        self.assertLess(len(bulletin.prevision_meteo.split()), 150)

    def test_pdl_2025_prevision_risque(self) -> None:
        bulletin = self.extract("pdl/maraichage/2025/BSV_maraichage_no_07_20250403.txt")

        self.assertEqual(bulletin.template_id, "pdl_2025_prevision_risque")
        self.assertEqual(bulletin.nature_meteo, "prevision")
        self.assertEqual(bulletin.meteo_extraction_status, "lisible")
        self.assertNotIn("EVALUATION DES RISQUES", bulletin.prevision_meteo)

    def test_grand_est_image_only_weather(self) -> None:
        bulletin = self.extract(
            "grandest/legumes_maraichage/2025/BSV02_LEG_GE_S16_2025.txt"
        )

        self.assertEqual(bulletin.template_id, "grand_est_standard")
        self.assertEqual(bulletin.nature_meteo, "prevision")
        self.assertEqual(bulletin.meteo_extraction_status, "image_seule")
        self.assertEqual(
            bulletin.periode_prevision_meteo,
            "17/04/2025-23/04/2025",
        )

    def test_grand_est_without_observation_or_weather(self) -> None:
        bulletin = self.extract(
            "grandest/legumes_maraichage/2025/BSV01_LEG_GE_S07_2025.txt"
        )

        self.assertEqual(bulletin.template_id, "grand_est_sans_observation")
        self.assertEqual(bulletin.nature_meteo, "absent")
        self.assertEqual(bulletin.meteo_extraction_status, "absente")
        self.assertEqual(
            bulletin.meteo_extraction_method,
            "fallback:grand_est_sans_observation",
        )
        self.assertEqual(bulletin.periode_prevision_meteo, "")

    def test_paca_maraichage_has_no_weather_false_positive(self) -> None:
        bulletin = self.extract(
            "paca/maraichage/2025/bsv_maraichage_no14_du_18_juillet_2025.txt"
        )

        self.assertEqual(bulletin.bulletin_id, "paca_2025_bsv14_20250718")
        self.assertEqual(bulletin.region, "Provence-Alpes-Côte d'Azur")
        self.assertEqual(bulletin.date_bulletin, "2025-07-18")
        self.assertEqual(bulletin.template_id, "paca_maraichage")
        self.assertEqual(bulletin.nature_meteo, "absent")
        self.assertEqual(bulletin.meteo_extraction_status, "absente")
        self.assertEqual(bulletin.prevision_meteo, "")


if __name__ == "__main__":
    unittest.main()
