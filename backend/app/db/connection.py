from pathlib import Path

import psycopg
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DatabaseSettings(BaseSettings):
    # SecretStr prevents accidental display of the connection string.
    database_url: SecretStr

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )


def get_connection() -> psycopg.Connection:
    settings = DatabaseSettings()  # pyright: ignore[reportCallIssue]

    return psycopg.connect(
        settings.database_url.get_secret_value(),
        connect_timeout=5,
    )


def main() -> None:
    # The context manager closes the connection afterward.
    with get_connection() as connection:
        row = connection.execute(
            "SELECT current_database(), current_user"
        ).fetchone()

        if row is None:
            raise RuntimeError("Database query returned no result")

        print("Database connected:", row[0])
        print("Database user:", row[1])

        vector = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()

        if vector is None:
            raise RuntimeError("pgvector is not enabled")

        print("pgvector version:", vector[0])


if __name__ == "__main__":
    main()