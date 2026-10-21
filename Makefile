.PHONY: setup seed gold forecast publish run api lab test up down logs

# --- no-Docker path (uv + local Parquet) ---
setup:            ## create .venv with core + models + lab + dev extras
	uv venv && uv pip install -e ".[models,lab,dev]"
	cp -n .env.example .env || true

seed:             ## silver tables from mock data
	uv run madrid seed

gold:             ## features, scores, LSI on history
	uv run madrid build-gold

forecast:         ## indicator forecasts to 2036 + ranking
	uv run madrid forecast

publish:          ## copy gold into Postgres (needs MADRID_DATABASE_URL)
	uv run madrid publish

run:              ## the whole pipeline
	uv run madrid run-all

api:              ## FastAPI with hot reload on :8000
	uv run uvicorn api.main:app --reload

lab:              ## JupyterLab on :8888
	uv run jupyter lab --notebook-dir=.

test:
	uv run pytest -q

# --- Docker path ---
up:               ## full lab: db, minio, mlflow, seed, api, jupyter
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f seed api lab

smoke:            ## walk the API endpoints (API must be running)
	python3 scripts/smoke.py
