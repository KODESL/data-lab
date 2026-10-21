"""Runtime configuration. Everything is driven by environment variables so the same code
runs on a laptop (local Parquet), in docker compose (MinIO + Postgres) and in the cloud
(ADLS + ADW/Postgres)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MADRID_", env_file=".env", extra="ignore")

    # "mock" generates a synthetic panel; "real" runs the connectors against INE & co.
    data_mode: str = "mock"

    # Root of the lake. A local path, or an s3:// / abfss:// URL.
    lake_root: str = "data/lake"

    # Optional S3-compatible store (MinIO locally, any S3 in the cloud)
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # Optional serving database for analysts / BI. The API itself reads gold Parquet.
    database_url: str | None = None

    # Mock panel shape
    mock_municipios: int = 179
    year_start: int = 2000
    year_end: int = 2025
    horizon_end: int = 2036
    small_threshold: int = 20_000
    seed: int = 28

    @property
    def lake_path(self) -> Path:
        return Path(self.lake_root)

    def zone(self, name: str) -> str:
        """Return the path/URL of a lake zone (bronze, silver, gold)."""
        return f"{self.lake_root.rstrip('/')}/{name}"


settings = Settings()
