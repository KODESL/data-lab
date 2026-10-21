"""Thin read/write layer over the lake. Tables are Parquet files, one per table and zone.
DuckDB is the query engine; Polars is the in-memory frame. Local paths and S3 both work."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import polars as pl

from .config import settings


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if settings.s3_endpoint:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        host = settings.s3_endpoint.replace("http://", "").replace("https://", "")
        con.execute(f"SET s3_endpoint='{host}';")
        con.execute(f"SET s3_use_ssl={'true' if settings.s3_endpoint.startswith('https') else 'false'};")
        con.execute("SET s3_url_style='path';")
        con.execute(f"SET s3_access_key_id='{settings.s3_access_key}';")
        con.execute(f"SET s3_secret_access_key='{settings.s3_secret_key}';")
    return con


def table_path(zone: str, table: str) -> str:
    return f"{settings.zone(zone)}/{table}.parquet"


def write(df: pl.DataFrame, zone: str, table: str) -> str:
    path = table_path(zone, table)
    if "://" not in path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    else:
        con = connect()
        con.register("df", df.to_arrow())
        con.execute(f"COPY df TO '{path}' (FORMAT PARQUET)")
    return path


def read(zone: str, table: str) -> pl.DataFrame:
    path = table_path(zone, table)
    if "://" not in path:
        return pl.read_parquet(path)
    con = connect()
    return pl.from_arrow(con.execute(f"SELECT * FROM read_parquet('{path}')").arrow())


def exists(zone: str, table: str) -> bool:
    path = table_path(zone, table)
    if "://" not in path:
        return os.path.exists(path)
    try:
        connect().execute(f"SELECT 1 FROM read_parquet('{path}') LIMIT 1")
        return True
    except duckdb.Error:
        return False


def sql(query: str) -> pl.DataFrame:
    """Run SQL with every lake table available as <zone>_<table>, e.g. silver_observations."""
    con = connect()
    for zone in ("bronze", "silver", "gold"):
        root = settings.zone(zone)
        if "://" in root:
            continue
        for p in Path(root).glob("*.parquet"):
            con.execute(f"CREATE VIEW {zone}_{p.stem} AS SELECT * FROM read_parquet('{p}')")
    return pl.from_arrow(con.execute(query).arrow())
