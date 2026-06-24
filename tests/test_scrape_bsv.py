from __future__ import annotations

import sys
import unittest
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scrape_bsv


class ExtractPdfLinksTests(unittest.TestCase):
    def test_detects_pdf_with_query_string_and_deduplicates(self) -> None:
        soup = BeautifulSoup(
            """
            <a href="/docs/bsv-01.pdf?download=1">BSV 1</a>
            <a href="/docs/bsv-01.pdf?download=1">Duplicate</a>
            <a href="/docs/page.html">Not PDF</a>
            """,
            "html.parser",
        )

        links = scrape_bsv.extract_pdf_links(soup, "https://example.test")

        self.assertEqual(
            links,
            [("bsv-01.pdf", "https://example.test/docs/bsv-01.pdf?download=1")],
        )

    def test_grand_est_archive_url_pattern(self) -> None:
        grand_est = next(
            source for source in scrape_bsv.SOURCES
            if source["source_id"] == "grandest"
        )
        filiere = grand_est["filieres"][0]

        self.assertEqual(
            filiere["url_pattern"].format(annee=2024),
            "https://grandest.chambres-agriculture.fr/"
            "agritheque/ressources-documentation/bulletins-techniques/"
            "bulletin-de-sante-du-vegetal/"
            "bsv-legumes-grand-est-archives-2024",
        )


class PdfValidationTests(unittest.TestCase):
    def test_existing_project_pdf_is_valid(self) -> None:
        pdf = next((PROJECT_ROOT / "data" / "raw").rglob("*.pdf"))
        self.assertTrue(scrape_bsv.is_valid_pdf_file(pdf))

    def test_dry_run_does_not_use_network_or_create_file(self) -> None:
        class ForbiddenSession:
            def get(self, *args, **kwargs):
                raise AssertionError("network must not be used in dry-run")

        destination = PROJECT_ROOT / "outputs" / "never_created_by_dry_run.pdf"
        destination.unlink(missing_ok=True)

        status = scrape_bsv.download_pdf(
            "https://example.test/document.pdf",
            destination,
            ForbiddenSession(),
            dry_run=True,
        )

        self.assertEqual(status, "dry_run")
        self.assertFalse(destination.exists())


class SessionTests(unittest.TestCase):
    def test_session_has_retry_adapter(self) -> None:
        with scrape_bsv.create_session() as session:
            retries = session.get_adapter("https://").max_retries

        self.assertEqual(retries.total, 4)
        self.assertIn(429, retries.status_forcelist)
        self.assertGreater(retries.backoff_factor, 0)


class LoggingTests(unittest.TestCase):
    def test_configure_logging_writes_a_timestamped_event(self) -> None:
        log_file = PROJECT_ROOT / "outputs" / "logs" / "scrape_bsv_test.log"
        log_file.unlink(missing_ok=True)

        scrape_bsv.configure_logging(log_file)
        scrape_bsv.LOGGER.info("TEST_EVENT | status=ok")
        for handler in scrape_bsv.LOGGER.handlers:
            handler.flush()

        content = log_file.read_text(encoding="utf-8")
        self.assertIn("INFO | TEST_EVENT | status=ok", content)

        for handler in list(scrape_bsv.LOGGER.handlers):
            handler.close()
            scrape_bsv.LOGGER.removeHandler(handler)
        log_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
