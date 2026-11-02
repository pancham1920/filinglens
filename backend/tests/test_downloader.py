"""Run from backend: uv run --offline python -m unittest discover -s tests -v."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from app.ingestion import downloader as d


REAL_CLIENT = httpx.Client
FILING = {
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "cik": "0000320193",
    "form_type": "10-K",
    "fiscal_year": 2024,
    "period_end": "2024-09-28",
    "filed_at": "2024-11-01",
    "accession_number": "0000320193-24-000123",
    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
}
# Deliberately small synthetic HTML; tests never download real filings.
HTML = b"<!doctype html><HTML><body>Test filing</body></HTML>"
BLOCK_PAGE = b"<html>Your Request Originates from an Undeclared Automated Tool</html>"


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.manifest_path = self.root / "manifest.json"
        self.write_manifest([FILING])

        for name, value in [
            ("MANIFEST_PATH", self.manifest_path),
            ("CACHE_DIR", self.cache),
        ]:
            patcher = patch.object(d, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        sleeper = patch.object(d.time, "sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)

    def write_manifest(self, filings):
        self.manifest_path.write_text(
            json.dumps({"version": 1, "filings": filings}), encoding="utf-8"
        )

    def cache_path(self, filing=FILING):
        return self.cache / f"{filing['accession_number']}.html"

    def client(self, handler):
        client = REAL_CLIENT(
            transport=httpx.MockTransport(handler), follow_redirects=False
        )
        self.addCleanup(client.close)
        return client

    def run_batch(self, handler, settings_error=None):
        """Run the real workflow against temporary files and a fake HTTP transport."""
        clients = []

        def create_client(**kwargs):
            client = REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)
            clients.append(client)
            return client

        with (
            patch.object(d.httpx, "Client", side_effect=create_client) as factory,
            patch.object(
                d,
                "Settings",
                return_value=SimpleNamespace(
                    sec_user_agent="FilingLens/test test@example.com"
                ),
                side_effect=settings_error,
            ) as settings,
            redirect_stdout(io.StringIO()) as output,
        ):
            code = d.main()

        self.assertTrue(all(client.is_closed for client in clients))
        return code, output.getvalue(), factory, settings

    def summary(self, output):
        lines = [line for line in output.splitlines() if line.startswith("SUMMARY: ")]
        self.assertEqual(len(lines), 1)
        return json.loads(lines[0].removeprefix("SUMMARY: "))

    def test_manifest_validates_and_dump_preserves_dictionary_dates(self):
        manifest = d.Manifest(version=1, filings=[FILING])
        dumped = manifest.filings[0].model_dump(mode="json")
        self.assertEqual(dumped["period_end"], "2024-09-28")
        self.assertEqual(dumped["cik"], "0000320193")

    def test_manifest_rejects_invalid_metadata_and_sources(self):
        changes = [
            {"cik": "123"},
            {"accession_number": "../outside"},
            {"accession_number": "0000320193-24-000213"},
            {"form_type": "10-Q"},
            {"period_end": "2024-13-40"},
            {"source_url": "https://example.com/filing.htm"},
            {"source_url": FILING["source_url"].replace("https:", "http:")},
            {"source_url": FILING["source_url"] + "?download=1"},
            {"source_url": FILING["source_url"] + "#section"},
            {"source_url": FILING["source_url"].replace("aapl-20240928.htm", "../a.htm")},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                d.Filing(**{**FILING, **change})
        incomplete = dict(FILING)
        incomplete.pop("source_url")
        with self.assertRaises(ValidationError):
            d.Filing(**incomplete)

    def test_manifest_rejects_empty_duplicate_and_unsupported_version(self):
        for data in [
            {"version": 1, "filings": []},
            {"version": 1, "filings": [FILING, FILING]},
            {"version": 2, "filings": [FILING]},
        ]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                d.Manifest.model_validate(data)

    def test_shared_html_validation_rejects_error_pages(self):
        d.validate_html(HTML)
        for content in [
            b"", b"not HTML", BLOCK_PAGE,
            b"<html>Request Rate Threshold Exceeded</html>",
        ]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                d.validate_html(content)

    def test_atomic_save_creates_parent_and_leaves_only_final_file(self):
        destination = self.root / "new" / "filing.html"
        d.save_atomically(destination, HTML)
        self.assertEqual(destination.read_bytes(), HTML)
        self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_failed_atomic_rename_preserves_previous_file_and_cleans_temp(self):
        destination = self.cache_path()
        destination.write_bytes(HTML)
        with patch.object(Path, "replace", side_effect=OSError("rename failure")):
            with self.assertRaises(OSError):
                d.save_atomically(destination, b"replacement")
        self.assertEqual(destination.read_bytes(), HTML)
        self.assertEqual(list(self.cache.iterdir()), [destination])

    def test_download_then_cache_makes_one_request_total(self):
        requests = []

        def handler(request):
            requests.append(request)
            self.assertEqual(request.headers["User-Agent"], "FilingLens/test test@example.com")
            return httpx.Response(200, content=HTML)

        code, output, _, _ = self.run_batch(handler)
        self.assertEqual(code, 0)
        self.assertEqual(self.summary(output), {"downloaded": 1, "cached": 0, "failed": 0})
        self.assertEqual(self.cache_path().read_bytes(), HTML)
        self.sleep.assert_called_once_with(1)

        code, output, factory, settings = self.run_batch(handler)
        self.assertEqual(code, 0)
        self.assertEqual(self.summary(output), {"downloaded": 0, "cached": 1, "failed": 0})
        factory.assert_not_called()
        settings.assert_not_called()
        self.assertEqual(len(requests), 1)

    def test_invalid_cache_is_replaced_with_validated_download(self):
        self.cache_path().write_bytes(BLOCK_PAGE)
        code, output, _, _ = self.run_batch(lambda request: httpx.Response(200, content=HTML))
        self.assertEqual(code, 0)
        self.assertEqual(self.summary(output)["downloaded"], 1)
        self.assertEqual(self.cache_path().read_bytes(), HTML)

    def test_success_status_with_block_page_is_not_saved(self):
        code, output, _, _ = self.run_batch(lambda request: httpx.Response(200, content=BLOCK_PAGE))
        self.assertEqual(code, 1)
        self.assertEqual(self.summary(output)["failed"], 1)
        self.assertFalse(self.cache_path().exists())

    def test_rate_limit_honors_retry_after_and_succeeds(self):
        responses = iter([
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, content=HTML),
        ])
        client = self.client(lambda request: next(responses))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(d.download_html(client, FILING["source_url"]), HTML)
        self.sleep.assert_any_call(7.0)
        self.assertEqual(sum(call.args == (1,) for call in self.sleep.call_args_list), 2)

    def test_transient_server_error_can_recover(self):
        responses = iter([httpx.Response(503), httpx.Response(200, content=HTML)])
        with redirect_stdout(io.StringIO()):
            content = d.download_html(self.client(lambda request: next(responses)), FILING["source_url"])
        self.assertEqual(content, HTML)
        self.sleep.assert_any_call(2.0)

    def test_permanent_error_and_redirect_do_not_retry_or_save(self):
        for status in [403, 404, 302]:
            with self.subTest(status=status):
                requests = []

                def handler(request):
                    requests.append(request)
                    return httpx.Response(status, headers={"Location": "https://example.com"})

                code, output, _, _ = self.run_batch(handler)
                self.assertEqual(code, 1)
                self.assertEqual(len(requests), 1)
                self.assertIn(f"HTTP {status}", output)
                self.assertFalse(self.cache_path().exists())

    def test_transport_failures_stop_after_three_attempts(self):
        for error_type in [httpx.ConnectTimeout, httpx.ConnectError]:
            with self.subTest(error_type=error_type):
                requests = []

                def handler(request):
                    requests.append(request)
                    raise error_type("simulated", request=request)

                code, output, _, _ = self.run_batch(handler)
                self.assertEqual(code, 1)
                self.assertEqual(len(requests), 3)
                self.assertEqual(self.summary(output)["failed"], 1)
                self.assertFalse(self.cache_path().exists())

    def test_server_failures_stop_after_three_attempts(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(503)

        code, output, _, _ = self.run_batch(handler)
        self.assertEqual(code, 1)
        self.assertEqual(len(requests), 3)
        self.assertIn("HTTP 503", output)

    def test_retry_after_date_and_malformed_header(self):
        class FixedClock:
            @staticmethod
            def now(tz):
                return datetime(2026, 1, 1, tzinfo=timezone.utc)

        retry_at = FixedClock.now(timezone.utc) + timedelta(seconds=20)
        with patch.object(d, "datetime", FixedClock):
            delay = d.retry_delay(httpx.Response(429, headers={"Retry-After": format_datetime(retry_at)}), 1)
        self.assertEqual(delay, 20)
        self.assertEqual(d.retry_delay(httpx.Response(429, headers={"Retry-After": "invalid"}), 1), 2)

    def test_long_retry_after_stops_instead_of_retrying_early(self):
        with self.assertRaisesRegex(ValueError, "Run again later"):
            d.retry_delay(httpx.Response(429, headers={"Retry-After": "120"}), 1)

    def test_batch_counts_cached_downloaded_and_failed_and_continues(self):
        filings = []
        for suffix in ["000123", "000124", "000125"]:
            filing = copy.deepcopy(FILING)
            filing["accession_number"] = "0000320193-24-" + suffix
            filing["source_url"] = f"https://www.sec.gov/Archives/edgar/data/320193/000032019324{suffix}/test.htm"
            filings.append(filing)
        self.write_manifest(filings)
        self.cache_path(filings[2]).write_bytes(HTML)
        requests = []

        def handler(request):
            requests.append(str(request.url))
            return httpx.Response(404 if str(request.url) == filings[0]["source_url"] else 200, content=HTML)

        code, output, factory, _ = self.run_batch(handler)
        self.assertEqual(code, 1)
        self.assertEqual(self.summary(output), {"downloaded": 1, "cached": 1, "failed": 1})
        self.assertEqual(requests, [filings[0]["source_url"], filings[1]["source_url"]])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(self.cache_path(filings[1]).read_bytes(), HTML)

    def test_write_failure_is_reported_without_counting_download_as_success(self):
        with patch.object(d, "save_atomically", side_effect=OSError("disk full")):
            code, output, _, _ = self.run_batch(lambda request: httpx.Response(200, content=HTML))
        self.assertEqual(code, 1)
        self.assertEqual(self.summary(output), {"downloaded": 0, "cached": 0, "failed": 1})
        self.assertIn("disk full", output)

    def test_bad_json_and_missing_manifest_fail_before_network_setup(self):
        self.manifest_path.write_text("{")
        code, output, factory, _ = self.run_batch(lambda request: self.fail("Unexpected HTTP request"))
        self.assertEqual(code, 1)
        self.assertIn("Invalid manifest JSON", output)
        factory.assert_not_called()
        self.manifest_path.unlink()
        code, output, factory, _ = self.run_batch(lambda request: self.fail("Unexpected HTTP request"))
        self.assertEqual(code, 1)
        self.assertIn("Cannot read or write", output)
        factory.assert_not_called()

    def test_invalid_manifest_fails_before_network_setup(self):
        self.write_manifest([FILING, FILING])
        code, output, factory, _ = self.run_batch(lambda request: self.fail("Unexpected HTTP request"))
        self.assertEqual(code, 1)
        self.assertIn("duplicate accession", output)
        factory.assert_not_called()

    def test_settings_error_does_not_expose_input_values(self):
        secret = "PRIVATE_SENTINEL_DO_NOT_PRINT"
        try:
            d.Settings(sec_user_agent={"sensitive": secret}, _env_file=None)
        except ValidationError as exc:
            settings_error = exc
        else:
            self.fail("Expected a settings validation error")
        code, output, factory, _ = self.run_batch(
            lambda request: self.fail("Unexpected HTTP request"),
            settings_error=settings_error,
        )
        self.assertEqual(code, 1)
        self.assertIn("Invalid settings or manifest", output)
        self.assertNotIn(secret, output)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
