from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "observations"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import extract_observations


class ObservationHelpersTests(unittest.TestCase):
    def test_bulletin_id_from_pays_de_la_loire_stem(self) -> None:
        self.assertEqual(
            extract_observations.bulletin_id_from_stem(
                "pays_de_la_loire_2025",
                "16_20250612_BSV-maraichage-n°16",
            ),
            "pdl_2025_bsv16",
        )

    def test_bulletin_id_from_current_grand_est_stem(self) -> None:
        self.assertEqual(
            extract_observations.bulletin_id_from_stem(
                "grandest",
                "BSV22_LEG_GE_S37_2025",
            ),
            "grandest_2025_bsv22",
        )

    def test_current_nested_table_metadata(self) -> None:
        root = PROJECT_ROOT / "outputs" / "tables"
        path = (
            root
            / "pdl"
            / "maraichage"
            / "2025"
            / "BSV_maraichage_no_16_20250612"
            / "page_07_table_1.csv"
        )

        self.assertEqual(
            extract_observations.table_metadata(path, root),
            ("pdl", "BSV_maraichage_no_16_20250612"),
        )

    def test_loads_current_bulletin_registry(self) -> None:
        registry = extract_observations.load_bulletin_registry(
            PROJECT_ROOT / "data" / "processed" / "bsv_bulletins.csv"
        )

        self.assertEqual(len(registry), 157)
        self.assertEqual(
            registry["grandest_2025_bsv01"]["template_id"],
            "grand_est_sans_observation",
        )

    def test_discovers_grand_est_texts_across_all_years(self) -> None:
        paths = extract_observations.discover_grand_est_texts(
            PROJECT_ROOT / "outputs" / "extracted_text"
        )
        years = {
            path.relative_to(PROJECT_ROOT / "outputs" / "extracted_text").parts[2]
            for path in paths
        }

        self.assertEqual(years, {"2023", "2024", "2025"})
        self.assertEqual(len(paths), 65)

    def test_extract_departments(self) -> None:
        self.assertEqual(
            extract_observations.extract_departments(
                "44 ; 49 ; 85",
                "Présence à La Roche sur Yon (85)",
            ),
            "44;49;85",
        )

    def test_quantitative_indicators(self) -> None:
        self.assertEqual(
            extract_observations.extract_quantitative_indicators(
                "50% des plantes et 2 captures observées.",
            ),
            "50%; 2 captures",
        )

    def test_risk_level_prioritizes_explicit_null_risk(self) -> None:
        self.assertEqual(
            extract_observations.extract_risk_level(
                "Le risque est nul depuis 4 jours. La reserve de spores est faible mais en hausse."
            ),
            "nul",
        )

    def test_risk_level_detects_no_risk(self) -> None:
        self.assertEqual(
            extract_observations.extract_risk_level("Pas de risque actuellement."),
            "nul",
        )

    def test_risk_level_ignores_non_risk_strength(self) -> None:
        self.assertEqual(
            extract_observations.extract_risk_level(
                "Le nombre d'individus observes est fort."
            ),
            "",
        )

    def test_split_observation_by_culture(self) -> None:
        observations = extract_observations.split_observation_by_culture(
            (
                "5% sur tomates à Loire-Authion ; "
                "60% sur aubergines à Loire-Authion ; "
                "10% sur aubergines à Saumur ; "
                "60% sur poivrons à Loire-Authion"
            ),
            ["tomate", "aubergine", "poivron"],
        )

        self.assertEqual(
            observations["tomate"],
            "5% sur tomates à Loire-Authion",
        )
        self.assertEqual(
            observations["aubergine"],
            "60% sur aubergines à Loire-Authion; 10% sur aubergines à Saumur",
        )
        self.assertEqual(
            observations["poivron"],
            "60% sur poivrons à Loire-Authion",
        )


class PaysDeLaLoireTableTests(unittest.TestCase):
    def test_extract_observation_table(self) -> None:
        root = FIXTURES_DIR / "tables"
        table = (
            root
            / "pays_de_la_loire_2025"
            / "16_20250612_BSV-maraichage-n°16"
            / "page_07_table_1.csv"
        )
        rows = extract_observations.extract_pays_de_la_loire_table(table, root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].bulletin_id, "pdl_2025_bsv16")
        self.assertEqual(rows[0].type, "maladie")
        self.assertEqual(rows[0].bioagresseur, "Anthracnose")
        self.assertEqual(rows[0].departements, "44")

    def test_extract_observation_table_splits_multi_culture_observation(self) -> None:
        root = FIXTURES_DIR / "tables"
        table = (
            root
            / "pays_de_la_loire_2025"
            / "16_20250612_BSV-maraichage-n°16"
            / "page_07_table_2.csv"
        )
        rows = extract_observations.extract_pays_de_la_loire_table(table, root)
        by_culture = {
            row.culture: row.observation
            for row in rows
        }

        self.assertEqual(set(by_culture), {"tomate", "aubergine", "poivron"})
        self.assertIn("tomates", by_culture["tomate"])
        self.assertNotIn("aubergines", by_culture["tomate"])
        self.assertIn("aubergines", by_culture["aubergine"])
        self.assertNotIn("poivrons", by_culture["aubergine"])
        self.assertIn("poivrons", by_culture["poivron"])


class PaysDeLaLoire2023NarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text_dir = PROJECT_ROOT / "outputs" / "extracted_text"
        cls.tables_dir = PROJECT_ROOT / "outputs" / "tables"
        cls.type_map = extract_observations.load_type_referential(
            PROJECT_ROOT / "data" / "raw" / "referentiel_types.csv"
        )

    def test_splits_two_layout_columns(self) -> None:
        left, right = extract_observations.split_pdl_2023_columns(
            "Mildiou                                                        Botrytis"
        )

        self.assertEqual(left, "Mildiou")
        self.assertEqual(right, "Botrytis")

    def test_extracts_tomato_mildew_from_regular_bulletin(self) -> None:
        path = (
            self.text_dir
            / "pdl"
            / "maraichage"
            / "2023"
            / "BSV_maraichage_no_14_20230615.txt"
        )
        rows = extract_observations.extract_pdl_2023_text(
            path,
            self.text_dir,
            self.tables_dir,
            self.type_map,
        )

        matches = [
            row
            for row in rows
            if row.culture == "tomate" and row.bioagresseur == "Mildiou"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].type, "maladie")
        self.assertIn("10%", matches[0].observation)
        self.assertFalse(
            any("INDIQUE" in extract_observations.normalize(row.bioagresseur) for row in rows)
        )

    def test_extracts_flash_tomato_mildew(self) -> None:
        path = (
            self.text_dir
            / "pdl"
            / "maraichage"
            / "2023"
            / "BSV_maraichage_flash_no_01_20230517.txt"
        )
        rows = extract_observations.extract_pdl_2023_flash_text(
            path,
            self.text_dir,
            self.type_map,
        )

        self.assertTrue(
            any(
                row.culture == "tomate"
                and extract_observations.normalize(row.bioagresseur) == "MILDIOU"
                for row in rows
            )
        )

    def test_same_page_table_support_requires_context_match(self) -> None:
        page_tables = [
            (
                Path("page_02_table_1.csv"),
                "Tomate Mildiou observé à Nantes (44), 25% des plants",
            ),
            (
                Path("page_02_table_2.csv"),
                "Pluviométrie et températures",
            ),
        ]

        matches = extract_observations.matching_pdl_2023_table_support(
            page_tables,
            "Mildiou",
            ["tomate"],
        )

        self.assertEqual(matches, [page_tables[0]])


class GrandEstTextTests(unittest.TestCase):
    def test_extract_observation_block(self) -> None:
        root = FIXTURES_DIR / "text"
        text = root / "grand_est_2025" / "BSV22_LEG_GE_S37_2025.txt"
        rows = extract_observations.extract_grand_est_text(text, root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].bulletin_id, "grandest_2025_bsv22")
        self.assertEqual(rows[0].culture, "salade")
        self.assertEqual(rows[0].bioagresseur, "Pucerons")
        self.assertEqual(rows[0].departements, "54")


if __name__ == "__main__":
    unittest.main()
