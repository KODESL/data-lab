# data-lab: Madrid municipios pipeline and analysis lab

Data pipeline, Living-Standards Index (LSI) and forecasting lab for the municipalities of the
Comunidad de Madrid. The question it is built to answer: of the small municipalities, which will
offer the best living standards in 2036, for which kind of newcomer, and why.

It runs fully on your laptop with synthetic data (no cloud account, no network after install).
The same code runs on real public sources (INE, Comunidad de Madrid, SEPE, ...) and in the cloud
by changing environment variables. The architecture document describes the full target design;
this repository is the working lab for it.

Contents

1. [Prerequisites](#1-prerequisites)
2. [Run it locally without Docker](#2-run-it-locally-without-docker)
3. [Run the full lab with Docker](#3-run-the-full-lab-with-docker)
4. [Test it: first examples](#4-test-it-first-examples)
5. [Work in the lab: notebooks and your first algorithm](#5-work-in-the-lab-notebooks-and-your-first-algorithm)
6. [What is in the box](#6-what-is-in-the-box)
7. [Configuration](#7-configuration)
8. [Switching to real data and to the cloud](#8-switching-to-real-data-and-to-the-cloud)
9. [Troubleshooting](#9-troubleshooting)

## 1. Prerequisites

| Tool | Version | Needed for | Install |
| --- | --- | --- | --- |
| Python | 3.11 or newer | everything | python.org, or `uv python install 3.12` |
| uv | any recent | environments and running commands | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| make | any | the shortcuts below (optional, every target is one command) | Xcode CLT on macOS |
| Docker Desktop | 24 or newer | the full lab only (section 3) | docker.com |

Apple Silicon and x86 both work. First install downloads about 400 MB of wheels.

## 2. Run it locally without Docker

```bash
git clone https://github.com/KODESL/data-lab.git
cd data-lab
make setup        # creates .venv, installs the package + [models,lab,dev] extras, copies .env.example to .env
make run          # seed -> build-gold -> forecast -> publish (publish is skipped without Postgres)
```

`make run` takes about 45 to 60 seconds; the forecast step is the slow one (20 indicators x 179
municipalities, one damped-Holt fit each). Expected tail of the output:

```
[madrid] mock panel: 179 municipalities, 93080 observations
[madrid] silver written to data/lake/silver
[madrid] top 5 small municipalities today (default profile):
[madrid]    1 Municipio 172    LSI 64.6
[madrid]    2 Municipio 005    LSI 62.4
...
[madrid] forecast + ranking for 2036 in 44s
[madrid]    1 Municipio 172    LSI 63.7  rank band [1, 19]
...
[madrid] no MADRID_DATABASE_URL set, skipping publish
```

The mock generator is seeded, so you get exactly these numbers. After the run you have a local
lake under `data/lake/`:

```
data/lake/silver/  dim_municipio.parquet  dim_indicator.parquet  observations.parquet
data/lake/gold/    features  scores  lsi  bounds  indicator_forecast  lsi_forecast  ranking_horizon  (.parquet)
```

Without `make`, the same four commands are:

```bash
uv venv && uv pip install -e ".[models,lab,dev]" && cp -n .env.example .env
uv run madrid seed && uv run madrid build-gold && uv run madrid forecast
```

## 3. Run the full lab with Docker

```bash
cp .env.example .env
docker compose up --build
```

First build takes a few minutes (GDAL and the ML stack). When it settles you have:

| Service | URL | What it is |
| --- | --- | --- |
| lab | http://localhost:8888 | JupyterLab, no token, with `src/` and `notebooks/` mounted live |
| api | http://localhost:8000/docs | FastAPI over the gold layer, hot reload |
| mlflow | http://localhost:5000 | experiment tracking, artifacts stored in MinIO |
| minio | http://localhost:9001 | S3-compatible lake (user `madrid`, password `madridmadrid`), buckets bronze/silver/gold/mlflow |
| db | localhost:5432 | PostGIS (`madrid`/`madrid`, database `madrid`), schemas `silver` and `gold` |

The `seed` service runs `madrid run-all` once (this time with Postgres configured, so `publish`
loads the gold tables into `gold.*`) and exits; `api` and `lab` wait for it. To re-run the
pipeline after changing code: `docker compose run --rm seed`. To stop: `docker compose down`
(add `-v` to also drop the data volumes).

## 4. Test it: first examples

### 4.1 The API

Terminal 1: `make api` (or, with Docker, the API is already up on :8000). Open
http://localhost:8000/docs for the interactive Swagger page, or use the smoke script, which
walks every endpoint and prints a readable summary:

```bash
python scripts/smoke.py                # standard library only, no venv needed
python scripts/smoke.py http://localhost:8000
```

Expected output (abridged):

```
== top 5 small municipalities today, default profile
 1 28172 Municipio 172    LSI 64.6  hous   43  work   83  acce   55  serv   94  envi   61  vita   66
 ...
== top 5 in 2036 for a young family, with rank band
 1 Municipio 090    LSI 64.1  band [1, 12]  robust_top20=True
 ...
== trajectory of 28172: last 3 history years and first 3 forecast years
hist 2025  LSI 64.6
fcst 2026  p10 59.8  p50 64.5  p90 68.9
== why 28172 ranks where it does: top 3 drivers vs the small-town median
vitality     score  65.5  median  41.4  contribution +0.077
== scenario: housing weighs 3x, and 28172 loses 40 access points
28172: rank 6 -> 96
all smoke checks passed
```

The individual calls, if you prefer curl:

```bash
# catalogue of indicators, dimensions, weight profiles
curl -s localhost:8000/catalog | python3 -m json.tool | head -40

# ranking today and at the horizon (2036), per profile; small municipalities only by default
curl -s "localhost:8000/ranking?year=2025&profile=default&limit=10"
curl -s "localhost:8000/ranking?year=2036&profile=retiree&limit=10"
curl -s "localhost:8000/ranking?year=2025&small_only=false&limit=5"

# one municipality: score trajectory with p10/p50/p90, and which dimensions drive its rank
curl -s "localhost:8000/municipio/28172/trajectory?profile=young_family"
curl -s "localhost:8000/municipio/28172/explain"

# what-if: your own weights and/or point shifts on a municipality's dimension scores
curl -s -X POST localhost:8000/scenario -H 'content-type: application/json' -d '{
  "year": 2036,
  "weights": {"housing": 3, "work": 1, "access": 2, "services": 1, "environment": 1, "vitality": 1},
  "deltas": {"28090": {"access": 25}},
  "limit": 10
}'
```

Every response is a filtered read of a pre-computed gold table. The only computation at
request time is the scenario endpoint, which re-weights six already-scored dimensions.

### 4.2 The data, with SQL

Any Parquet file in `data/lake/<zone>/` is queryable as `<zone>_<name>`:

```bash
uv run python
```

```python
from madrid import lake
lake.sql("SELECT * FROM gold_ranking_horizon WHERE profile = 'default' ORDER BY rank_p50 LIMIT 10")
lake.sql("SELECT year, round(avg(lsi_default), 1) AS mean_lsi FROM gold_lsi GROUP BY 1 ORDER BY 1")
lake.sql("""
  SELECT o.year, o.value FROM silver_observations o
  WHERE o.municipality_id = '28172' AND o.indicator_id = 'price_m2' ORDER BY 1
""")
lake.read("silver", "dim_indicator")      # the indicator catalogue as a Polars frame
```

With Docker you can also point any SQL client at Postgres (`gold.lsi`, `gold.ranking_horizon`,
`silver.observations`, ...) once the `seed` service has finished.

### 4.3 The automated test

```bash
make test        # one end-to-end test on a 40-municipality panel: seed, gold, forecast, API calls
```

## 5. Work in the lab: notebooks and your first algorithm

```bash
make lab         # JupyterLab on http://localhost:8888 (or use the Docker `lab` service)
```

Notebooks are plain `.py` files with `# %%` cell markers (jupytext style), so they diff well in
git. In JupyterLab, right-click a file in `notebooks/` and choose Open With > Notebook, or run
them as scripts.

`notebooks/01_explore_lsi.py` (run top to bottom, about 5 seconds):

1. loads silver and gold and prints their shapes
2. runs a DuckDB SQL summary of every indicator (count, mean, sd, nulls)
3. shows today's top 15 small municipalities and how the podium changes per weight profile
4. plots the LSI trajectory of the top 5 against the small-town median
5. compares the six dimension scores of the top 10 against the median

`notebooks/02_forecast_lab.py` (needs the `[models]` extra, installed by `make setup`; about 30 seconds):

1. backtests the shipped baseline (damped Holt per series) with cutoff 2015, horizon 10, sMAPE per indicator
2. builds a supervised frame with lags, first difference and a spatial lag (mean of the 5 nearest neighbours)
3. trains a global LightGBM per indicator, direct horizon 5, and prints `lgb_wins` per indicator

That second notebook is the harness every new model goes through. The loop for writing your own
algorithm:

1. Put the model in a module under `src/madrid/` (for example `src/madrid/models/gravity.py`).
   The package is installed editable, so the notebook sees your edits after a kernel restart or
   `importlib.reload`.
2. Import it in a notebook, backtest it against the baseline with the same cutoff and metric.
3. If it wins, call it from `src/madrid/forecast.py` (`forecast_indicators` is the hook) and run
   `make forecast`; the API and the ranking pick it up with no other change.
4. Log the run with MLflow (`mlflow.start_run()`; the tracking URI is set in Docker, or
   `export MLFLOW_TRACKING_URI=http://localhost:5000`).

Things worth trying in the first hour:

- change the weights in `PROFILES` (`src/madrid/indicators.py`), run `make gold`, see the podium move
- change `MADRID_SMALL_THRESHOLD` in `.env` to 10000, run `make run`, see who enters the shortlist
- add an indicator to `INDICATORS` and a series to `mock.py`, run `make run`; it flows through the index automatically
- replace the crude rank band with a Monte Carlo over weights (Dirichlet around the profile) in notebook 02

## 6. What is in the box

```
src/madrid/
  config.py        env-driven settings (data mode, lake root, S3, Postgres, horizon, threshold)
  lake.py          read/write Parquet tables per zone; lake.sql() runs DuckDB over all of them
  indicators.py    indicator catalogue (unit, polarity, dimension, source), weight profiles
  mock.py          seeded synthetic panel: 179 municipalities x 26 years x 20 indicators, silver schema
  lsi.py           feature matrix, winsorize + min-max, polarity, dimension means, geometric-mean LSI, ranking
  forecast.py      damped-Holt forecasts to the horizon with p10/p50/p90, LSI on the forecasts
  jobs.py          CLI: madrid seed | build-gold | forecast | publish | run-all
api/main.py        FastAPI: /health /catalog /municipios /ranking /municipio/{id}/trajectory
                   /municipio/{id}/explain  POST /scenario
notebooks/         01 explore the index, 02 forecast backtest harness
scripts/smoke.py   walks the API and prints a readable summary
db/init/           Postgres/PostGIS bootstrap (schemas, a helper view)
tests/             end-to-end test on a small panel
Dockerfile         one multi-arch image: jobs, API and JupyterLab (target `lab`)
docker-compose.yml db, minio, mlflow, seed, api, lab
.github/workflows  ruff + pytest + docker build on every push
```

Silver schema, the contract every data source must produce:

| table | columns |
| --- | --- |
| `observations` | municipality_id (INE 5-digit), indicator_id, year, value, unit, source, quality_flag |
| `dim_municipio` | municipality_id, name, x_km, y_km, dist_km, altitude_m, population_2000, population_latest, is_small |
| `dim_indicator` | indicator_id, name, unit, dimension, polarity (+1 / -1 / 0), source, cadence |

Gold tables: `features` (wide), `scores` (0 to 100 per indicator), `lsi` (six dimension scores +
`lsi_<profile>`), `bounds` (normalization bounds), `indicator_forecast` (p10/p50/p90 per
indicator-year), `lsi_forecast` (quantile column), `ranking_horizon` (per profile: rank_p50,
rank_lo, rank_hi, robust_top20).

## 7. Configuration

All settings are environment variables with the `MADRID_` prefix, read from `.env`
(see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MADRID_DATA_MODE` | `mock` | `mock` generates the panel; `real` runs the connectors |
| `MADRID_LAKE_ROOT` | `data/lake` | local path, `s3://bucket/prefix` or `abfss://...` |
| `MADRID_S3_ENDPOINT` / `_ACCESS_KEY` / `_SECRET_KEY` | unset | S3-compatible lake (MinIO in compose) |
| `MADRID_DATABASE_URL` | unset | Postgres for `madrid publish`; the API does not need it |
| `MADRID_MOCK_MUNICIPIOS` | 179 | size of the synthetic panel |
| `MADRID_YEAR_START` / `_YEAR_END` | 2000 / 2025 | history window |
| `MADRID_HORIZON_END` | 2036 | forecast horizon |
| `MADRID_SMALL_THRESHOLD` | 20000 | population below which a municipality is "small" |
| `MADRID_SEED` | 28 | random seed of the mock generator |

## 8. Switching to real data and to the cloud

Real data: implement connectors under `src/madrid/connectors/` (dlt resources for INE, the
Comunidad de Madrid CKAN portal, SEPE, Hacienda, Catastro, CRTM GTFS, AEMET) that write the
`observations` long schema and a real `dim_municipio` (INE codes, names, CNIG geometry), then
`MADRID_DATA_MODE=real madrid seed`. Index, forecasts, API and notebooks need no change.

Cloud: the Docker image is the unit of deployment. Azure Container Apps Jobs run `madrid seed`
and `madrid build-gold` against `MADRID_LAKE_ROOT=abfss://...`; the OCI ARM VM runs
`madrid forecast` and `madrid publish` against Oracle ADW or Postgres. See the architecture
document for the full design, costs and the data-source catalogue.

## 9. Troubleshooting

- `make setup` fails on `lightgbm` or `geopandas` on macOS: install `libomp` and `gdal` with
  Homebrew (`brew install libomp gdal`) or drop the `[geo]` extra; the core pipeline does not need it.
- `madrid run-all` says `No module named madrid`: run through `uv run ...` or activate `.venv`.
- API returns 500 on `/ranking?year=2036`: `forecast` has not run yet; `make forecast`.
- Port already in use: `uvicorn api.main:app --port 8001`, or edit the `ports:` in compose.
- Docker build is slow or fails on GDAL: use the no-Docker path for the lab; Docker is only
  needed for Postgres, MinIO and MLflow.
- Reset everything: `rm -rf data/lake && make run` (or `docker compose down -v`).
