from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "moteur"))

import extract_text


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "extract_text"


class DiscoverPdfsTests(unittest.TestCase):
    def test_uses_only_paths_from_corpus_list(self) -> None:
        self.assertEqual(
            extract_text.discover_pdfs(
                FIXTURES_DIR,
                FIXTURES_DIR / "corpus.txt",
            ),
            [(FIXTURES_DIR / "pdl" / "bulletin.pdf").resolve()],
        )

    def test_rejects_path_outside_input_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "hors du corpus"):
            extract_text.discover_pdfs(
                FIXTURES_DIR,
                FIXTURES_DIR / "outside.txt",
            )

    def test_reports_missing_listed_pdf(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "1 PDF"):
            extract_text.discover_pdfs(
                FIXTURES_DIR,
                FIXTURES_DIR / "missing.txt",
            )

    def test_can_discover_all_pdfs_without_list(self) -> None:
        expected = sorted(
            [
                (FIXTURES_DIR / "pdl" / "bulletin.pdf").resolve(),
                (FIXTURES_DIR / "pdl" / "bilan.pdf").resolve(),
                (FIXTURES_DIR / "nested" / "other.pdf").resolve(),
            ]
        )
        self.assertEqual(extract_text.discover_pdfs(FIXTURES_DIR), expected)

    def test_paca_nested_output_paths_are_preserved(self) -> None:
        input_dir = Path("data/raw/moteur/bsv_pdf")
        output_dir = Path("outputs/moteur/extracted_text")
        tables_dir = Path("outputs/moteur/tables")
        pdf_path = (
            input_dir
            / "paca"
            / "maraichage"
            / "2025"
            / "bsv_maraichage_no21_du_21_novembre_2025.pdf"
        )

        self.assertEqual(
            extract_text.txt_path_for(pdf_path, input_dir, output_dir),
            output_dir
            / "paca"
            / "maraichage"
            / "2025"
            / "bsv_maraichage_no21_du_21_novembre_2025.txt",
        )
        self.assertEqual(
            extract_text.table_dir_for(pdf_path, input_dir, tables_dir),
            tables_dir
            / "paca"
            / "maraichage"
            / "2025"
            / "bsv_maraichage_no21_du_21_novembre_2025",
        )


if __name__ == "__main__":
    unittest.main()
