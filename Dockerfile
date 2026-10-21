# One image for the lab (JupyterLab), the jobs and the API. Multi-arch: builds on
# Apple Silicon and on the OCI ARM VM as well as on x86 Azure jobs.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgeos-dev libgdal-dev gdal-bin libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /work
COPY pyproject.toml README.md ./
COPY src ./src
# Core + models + geo + lab + dev extras, installed into the system interpreter
RUN uv pip install -e ".[models,geo,lab,dev]"

# ---- jobs / api target (default) ----
FROM base AS runtime
COPY api ./api
COPY db ./db
ENV MADRID_LAKE_ROOT=/work/data/lake
CMD ["madrid", "--help"]

# ---- lab target: JupyterLab on 8888 ----
FROM runtime AS lab
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", "--notebook-dir=/work"]
