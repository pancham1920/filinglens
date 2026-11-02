import json
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

import re
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError, model_validator


# Find the project root from this file's location.
# ingestion → app → backend → filinglens
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


class Settings(BaseSettings):
    # Required configuration, loaded from the environment or .env.
    sec_user_agent: str

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

class Filing(BaseModel):
    company: str = Field(min_length=1)
    ticker: str = Field(min_length=1)

    # Keep identifiers as strings to preserve leading zeros.
    cik: str = Field(pattern=r"^\d{10}$")
    accession_number: str = Field(
        pattern=r"^\d{10}-\d{2}-\d{6}$"
    )

    # This MVP supports annual 10-K filings.
    form_type: Literal["10-K"]
    fiscal_year: int = Field(ge=1990, le=2100)

    period_end: date
    filed_at: date
    source_url: str

    @model_validator(mode="after")
    def validate_source_url(self) -> "Filing":
        url = urlsplit(self.source_url)

        # SEC archive directories use an unpadded CIK
        # and an accession number without hyphens.
        expected_directory = (
            f"/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession_number.replace('-', '')}/"
        )

        filename = url.path.removeprefix(expected_directory)

        if (
            url.scheme != "https"
            or url.netloc != "www.sec.gov"
            or not url.path.startswith(expected_directory)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.html?", filename)
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "source_url must be a direct SEC HTML URL "
                "matching the filing's CIK and accession number"
            )

        return self


class Manifest(BaseModel):
    version: Literal[1]
    filings: list[Filing] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_filings(self) -> "Manifest":
        accessions = [
            filing.accession_number
            for filing in self.filings
        ]

        if len(accessions) != len(set(accessions)):
            raise ValueError(
                "Manifest contains duplicate accession numbers"
            )

        return self


def validate_html(content: bytes) -> None:
    """Reject obvious non-HTML responses and known SEC access-limit pages."""
    lowered = content.lower()
    if b"<html" not in lowered[:10000]:
        raise ValueError("Response does not look like HTML")

    # SEC can return an HTML explanation instead of the requested filing.
    # This is a sanity check, not proof that a document is the correct filing.
    blocked_markers = (
        b"your request originates from an undeclared automated tool",
        b"request rate threshold exceeded",
    )
    if any(marker in lowered for marker in blocked_markers):
        raise ValueError("SEC returned an access-limit page instead of a filing")


def save_atomically(destination: Path, content: bytes) -> None:
    """Write a complete temporary file before publishing the final cache path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None

    try:
        # Use the same directory so replacement stays on the same filesystem.
        # A unique temporary name avoids collisions between simultaneous writes.
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(content)

        # The file is closed before renaming. Readers see the old complete file
        # or the new complete file, rather than a partially written final file.
        temporary.replace(destination)
    finally:
        # Clean up on ordinary failures. After a successful rename it is gone.
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def retry_delay(response: httpx.Response, attempt: int) -> float:
    """Respect Retry-After without retrying sooner than normal backoff."""
    delay = float(2 ** attempt)
    value = response.headers.get("Retry-After")

    if value:
        try:
            # Retry-After may contain a number of seconds.
            delay = max(delay, float(value))
        except ValueError:
            try:
                # Alternatively it may contain an HTTP date.
                retry_at = parsedate_to_datetime(value)
                seconds = (
                    retry_at - datetime.now(timezone.utc)
                ).total_seconds()
                delay = max(delay, seconds)
            except (ValueError, TypeError, OverflowError):
                # An invalid header leaves the normal backoff unchanged.
                pass

    # Stop instead of retrying earlier than the server requests.
    if delay > 60:
        raise ValueError(
            "Server requested a long retry delay. Run again later."
        )

    return max(1.0, delay)


def download_html(client: httpx.Client, url: str) -> bytes:
    """Return validated HTML, retrying only transport or temporary HTTP failures."""
    max_attempts = 3
    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        # Three total attempts, including the first request. Spacing applies
        # to this process; other simultaneous downloaders have separate limits.
        time.sleep(1)
        print(f"Downloading: attempt {attempt}/{max_attempts}")

        try:
            response = client.get(url)
        except httpx.TransportError:
            if attempt == max_attempts:
                raise

            # Increasing backoff is additional to the one-second spacing.
            delay = 2 ** attempt
            print(f"Network failure; retrying in {delay} seconds")
            time.sleep(delay)
            continue

        if (
            response.status_code in retryable_statuses
            and attempt < max_attempts
        ):
            delay = retry_delay(response, attempt)
            print(
                f"HTTP {response.status_code}; "
                f"retrying in {delay:.1f} seconds"
            )
            time.sleep(delay)
            continue

        # Permanent HTTP errors and exhausted retries stop before saving.
        response.raise_for_status()
        content = response.content
        validate_html(content)

        print("HTTP status:", response.status_code)
        print("Content type:", response.headers.get("content-type"))
        print("Downloaded bytes:", len(content))
        return content

    raise RuntimeError("Download attempts exhausted")


def run() -> dict[str, int]:
    """Process every validated filing and report individual outcomes."""
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    manifest_data = json.loads(manifest_text)

    # Reject an invalid manifest before processing any entry.
    manifest = Manifest.model_validate(manifest_data)
    counts = {"downloaded": 0, "cached": 0, "failed": 0}

    # Open a shared client only when a filing needs downloading. ExitStack
    # closes it at the end, while cache-only runs need no settings or HTTP client.
    with ExitStack() as stack:
        client = None

        for entry in manifest.filings:
            filing = entry.model_dump(mode="json")
            accession = filing["accession_number"]
            print(f"\nProcessing: {filing['company']} FY{filing['fiscal_year']} ({accession})")
            print("URL:", filing["source_url"])

            try:
                destination = CACHE_DIR / f"{accession}.html"

                if destination.is_file():
                    cached_content = destination.read_bytes()
                    try:
                        validate_html(cached_content)
                    except ValueError as exc:
                        print(f"Invalid cached HTML; downloading again: {exc}")
                    else:
                        print("CACHED:", destination)
                        counts["cached"] += 1
                        # Move to the next filing instead of ending the entire run.
                        continue

                if client is None:
                    # Global configuration errors are handled by main().
                    settings = Settings()  # pyright: ignore[reportCallIssue]
                    client = stack.enter_context(httpx.Client(
                        headers={"User-Agent": settings.sec_user_agent},
                        timeout=httpx.Timeout(30.0, connect=10.0),
                        follow_redirects=False,
                    ))

                content = download_html(client, filing["source_url"])
                save_atomically(destination, content)
                print("Saved filing:", destination)
                counts["downloaded"] += 1

            except ValidationError:
                # Do not mistake invalid application settings for a filing failure
                # or print their raw input values as part of an error message.
                raise
            except (httpx.HTTPError, OSError, ValueError) as exc:
                counts["failed"] += 1
                if isinstance(exc, httpx.TimeoutException):
                    message = "Download timed out after the allowed attempts."
                elif isinstance(exc, httpx.TransportError):
                    message = (
                        "Network connection failed. Check connectivity "
                        "and approved proxy configuration."
                    )
                elif isinstance(exc, httpx.HTTPStatusError):
                    message = f"SEC returned HTTP {exc.response.status_code}."
                elif isinstance(exc, OSError):
                    message = f"Cannot read or write a required file: {exc}"
                else:
                    message = str(exc)
                print(f"FAILED {accession}: {message}")
                # Continue processing later entries even if this filing failed.

    print(f"\nSUMMARY: {json.dumps(counts)}")
    return counts


def main() -> int:
    """Translate expected failures into readable messages and an exit status."""
    try:
        counts = run()
        # A mixed run is unsuccessful if any filing failed, even when others saved.
        return 1 if counts["failed"] else 0

    except ValidationError as exc:
        print("ERROR: Invalid settings or manifest.")
        # Omit input values so settings validation does not expose .env contents.
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in error["loc"])
            print(f"  {location}: {error['msg']}")

    # Specific exceptions must precede their broader parent classes below.
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: Invalid manifest JSON "
            f"at line {exc.lineno}, column {exc.colno}."
        )

    except httpx.TimeoutException:
        print("ERROR: Download timed out after the allowed attempts.")

    except httpx.TransportError:
        print(
            "ERROR: Network connection failed. "
            "Check connectivity and approved proxy configuration."
        )

    except httpx.HTTPStatusError as exc:
        print(f"ERROR: SEC returned HTTP {exc.response.status_code}.")

    except OSError as exc:
        print(f"ERROR: Cannot read or write a required file: {exc}")

    except ValueError as exc:
        print(f"ERROR: {exc}")

    return 1


# Run only when invoked as a script/module, not when imported.
if __name__ == "__main__":
    raise SystemExit(main())
