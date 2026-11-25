from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import prepare_corpus


class DocumentClassificationTests(unittest.TestCase):
    def test_classifies_bilans(self) -> None:
        names = [
            "BSV_bilan_2023_partie_1_20231222.pdf",
            "BSVBilan_LEG_GE_2025.pdf",
            "BSV_maraichage_bilan_2025_cultures_ete_2025_20251024.pdf",
        ]
        self.assertEqual(
            [prepare_corpus.classify_document(name) for name in names],
            ["bilan", "bilan", "bilan"],
        )

    def test_classifies_catalogue(self) -> None:
        self.assertEqual(
            prepare_corpus.classify_document("2025_2026_Catalogue_formations.pdf"),
            "catalogue",
        )

    def test_keeps_flash_as_exploitable_type(self) -> None:
        self.assertEqual(
            prepare_corpus.classify_document("BSV_maraichage_flash_no_01_20230517.pdf"),
            "flash",
        )

    def test_classifies_paca_alert_as_flash(self) -> None:
        self.assertEqual(
            prepare_corpus.classify_document("bsv_maraichage_alerte_tolcndv_2025.pdf"),
            "flash",
        )

    def test_classifies_regular_bulletin(self) -> None:
        self.assertEqual(
            prepare_corpus.classify_document("BSV_maraichage_no_12_20230601.pdf"),
            "bulletin",
        )

    def test_extracts_number_and_date(self) -> None:
        name = "BSV_maraichage_no_12_20230601.pdf"
        self.assertEqual(prepare_corpus.extract_bulletin_number(name), "12")
        self.assertEqual(prepare_corpus.extract_date_from_filename(name), "2023-06-01")

    def test_extracts_paca_french_dates(self) -> None:
        cases = {
            "bsv_maraichage_no21_du_21_novembre_2025.pdf": "2025-11-21",
            "bsv_maraichage_no5_du_10-mars_2023.pdf": "2023-03-10",
            "bsv_maraichage_no7_du_7_avril_2023.pdf": "2023-04-07",
        }

        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    prepare_corpus.extract_date_from_filename(name), expected
                )

    def test_extracts_paca_date_with_folder_year_fallback(self) -> None:
        self.assertEqual(
            prepare_corpus.extract_date_from_filename(
                "bsv_maraichage_no23_du_19_decembre.pdf",
                fallback_year="2025",
            ),
            "2025-12-19",
        )

    def test_extracts_paca_compact_day_month_year_date(self) -> None:
        self.assertEqual(
            prepare_corpus.extract_date_from_filename(
                "bsv_maraichage_no15_exceptionnel_blettes_25072025.pdf",
            ),
            "2025-07-25",
        )


if __name__ == "__main__":
    unittest.main()
