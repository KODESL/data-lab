"""Pipeline jobs as a CLI. Each command is one container job in the cloud and one `make`
target locally:

  madrid seed            bronze/silver from mock data (or connectors when DATA_MODE=real)
  madrid build-gold      features, scores, LSI on history
  madrid forecast        indicator forecasts to the horizon + LSI on them + ranking
  madrid publish         copy gold into Postgres (optional, for analysts / BI)
  madrid run-all         the four above in order
"""

from __future__ import annotations

import time

import polars as pl
import typer

from . import lake, mock
from . import lsi as lsi_mod
from .config import settings
from .forecast import forecast_indicators, lsi_on_forecast
from .indicators import PROFILES, dim_indicator

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _log(msg: str) -> None:
    typer.echo(f"[madrid] {msg}")


@app.command()
def seed() -> None:
    """Create silver tables. Mock mode generates them; real mode runs the connectors."""
    if settings.data_mode == "mock":
        dim, obs = mock.generate()
        _log(f"mock panel: {dim.height} municipalities, {obs.height} observations")
    else:  # pragma: no cover
        from . import connectors  # noqa: F401  (real INE/CKAN/SEPE connectors live here)
        raise typer.Exit("real connectors not wired yet: set MADRID_DATA_MODE=mock")
    lake.write(dim, "silver", "dim_municipio")
    lake.write(dim_indicator(), "silver", "dim_indicator")
    lake.write(obs, "silver", "observations")
    _log(f"silver written to {settings.zone('silver')}")


@app.command("build-gold")
def build_gold() -> None:
    obs = lake.read("silver", "observations")
    out = lsi_mod.build(obs)
    for name, df in out.items():
        lake.write(df, "gold", name)
    dim = lake.read("silver", "dim_municipio")
    top = lsi_mod.rank(out["lsi"], dim, int(obs["year"].max()))
    _log("top 5 small municipalities today (default profile):")
    for r in top.head(5).iter_rows(named=True):
        _log(f"  {r['rank']:>2} {r['name']:<16} LSI {r['lsi']:.1f}")


@app.command()
def forecast() -> None:
    t0 = time.time()
    feat = lake.read("gold", "features")
    bounds = lake.read("gold", "bounds")
    fc = forecast_indicators(feat)
    lake.write(fc, "gold", "indicator_forecast")
    idx = lsi_on_forecast(fc, bounds)
    lake.write(idx, "gold", "lsi_forecast")
    # Ranking at the horizon with a p10/p90 rank interval
    dim = lake.read("silver", "dim_municipio")
    rows = []
    for profile in PROFILES:
        col = f"lsi_{profile}"
        h = idx.filter(pl.col("year") == settings.horizon_end).join(dim.select("municipality_id", "name", "is_small"), on="municipality_id")
        h = h.filter(pl.col("is_small"))
        ranks = {}
        for q in ("p10", "p50", "p90"):
            r = h.filter(pl.col("quantile") == q).sort(col, descending=True).with_row_index("rank", offset=1)
            ranks[q] = r.select("municipality_id", "name", pl.col(col).alias(f"lsi_{q}"), pl.col("rank").alias(f"rank_{q}"))
        out = ranks["p50"].join(ranks["p10"].drop("name"), on="municipality_id").join(ranks["p90"].drop("name"), on="municipality_id")
        # v0 rank band = spread of the rank under the p10/p50/p90 scores (all municipalities
        # move together, so this is a crude band; the Monte Carlo in the lab replaces it).
        out = out.with_columns(
            pl.lit(profile).alias("profile"),
            pl.min_horizontal("rank_p10", "rank_p50", "rank_p90").alias("rank_lo"),
            pl.max_horizontal("rank_p10", "rank_p50", "rank_p90").alias("rank_hi"),
        ).with_columns((pl.col("rank_hi") <= 20).alias("robust_top20")).sort("rank_p50")
        rows.append(out)
    ranking = pl.concat(rows)
    lake.write(ranking, "gold", "ranking_horizon")
    _log(f"forecast + ranking for {settings.horizon_end} in {time.time() - t0:.0f}s")
    for r in ranking.filter(pl.col("profile") == "default").head(5).iter_rows(named=True):
        _log(f"  {r['rank_p50']:>2} {r['name']:<16} LSI {r['lsi_p50']:.1f}  rank band [{r['rank_lo']}, {r['rank_hi']}]")


@app.command()
def publish() -> None:
    """Load gold tables into Postgres (DATABASE_URL). Skipped when not configured."""
    if not settings.database_url:
        _log("no MADRID_DATABASE_URL set, skipping publish")
        return
    import psycopg

    tables = ["features", "lsi", "indicator_forecast", "lsi_forecast", "ranking_horizon"]
    with psycopg.connect(settings.database_url) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS gold; CREATE SCHEMA IF NOT EXISTS silver;")
            for zone, name in [("silver", "dim_municipio"), ("silver", "dim_indicator"), ("silver", "observations")] + [("gold", t) for t in tables]:
                df = lake.read(zone, name)
                cols = ", ".join(f'"{c}" {_pg_type(df[c].dtype)}' for c in df.columns)
                cur.execute(f'DROP TABLE IF EXISTS {zone}."{name}"; CREATE TABLE {zone}."{name}" ({cols});')
                with cur.copy(f'COPY {zone}."{name}" FROM STDIN') as copy:
                    for row in df.iter_rows():
                        copy.write_row(row)
                _log(f"published {zone}.{name} ({df.height} rows)")
        con.commit()


def _pg_type(dt: pl.DataType) -> str:
    if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return "BIGINT"
    if dt in (pl.Float32, pl.Float64):
        return "DOUBLE PRECISION"
    if dt == pl.Boolean:
        return "BOOLEAN"
    return "TEXT"


@app.command("run-all")
def run_all() -> None:
    seed()
    build_gold()
    forecast()
    publish()


if __name__ == "__main__":
    app()
