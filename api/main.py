"""Read-only API over the gold layer. It never runs a model: every endpoint is a filtered
read of pre-computed tables, except /scenario, which re-weights already-scored dimensions.

Run locally:  uvicorn api.main:app --reload   (docs at http://localhost:8000/docs)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from madrid import lake
from madrid.config import settings
from madrid.indicators import DIMENSIONS, INDICATORS, PROFILES
from madrid.lsi import lsi

app = FastAPI(title="Madrid municipios API", version="0.1.0")


@lru_cache
def gold(name: str) -> pl.DataFrame:
    return lake.read("gold", name)


@lru_cache
def silver(name: str) -> pl.DataFrame:
    return lake.read("silver", name)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "data_mode": settings.data_mode, "lake": settings.lake_root}


@app.get("/catalog")
def catalog() -> dict:
    return {"indicators": INDICATORS, "dimensions": DIMENSIONS, "profiles": PROFILES,
            "years": {"start": settings.year_start, "end": settings.year_end, "horizon": settings.horizon_end}}


@app.get("/municipios")
def municipios(small_only: bool = False) -> list[dict]:
    df = silver("dim_municipio")
    if small_only:
        df = df.filter(pl.col("is_small"))
    return df.select("municipality_id", "name", "population_latest", "is_small", "dist_km", "altitude_m").to_dicts()


@app.get("/ranking")
def ranking(year: int = Query(default=None), profile: str = "default", small_only: bool = True, limit: int = 30) -> list[dict]:
    if profile not in PROFILES:
        raise HTTPException(400, f"unknown profile, use one of {list(PROFILES)}")
    if year is None or year <= settings.year_end:
        year = year or settings.year_end
        idx = gold("lsi").filter(pl.col("year") == year)
        dim = silver("dim_municipio")
        df = idx.join(dim.select("municipality_id", "name", "is_small"), on="municipality_id")
        if small_only:
            df = df.filter(pl.col("is_small"))
        col = f"lsi_{profile}"
        df = df.sort(col, descending=True).with_row_index("rank", offset=1)
        return df.select("rank", "municipality_id", "name", pl.col(col).alias("lsi"), *DIMENSIONS).head(limit).to_dicts()
    if year != settings.horizon_end:
        raise HTTPException(400, f"forecast ranking is available for {settings.horizon_end} only")
    return gold("ranking_horizon").filter(pl.col("profile") == profile).head(limit).to_dicts()


@app.get("/municipio/{municipality_id}/trajectory")
def trajectory(municipality_id: str, profile: str = "default") -> dict:
    col = f"lsi_{profile}"
    hist = gold("lsi").filter(pl.col("municipality_id") == municipality_id).sort("year")
    if hist.is_empty():
        raise HTTPException(404, "unknown municipality")
    fc = gold("lsi_forecast").filter(pl.col("municipality_id") == municipality_id)
    fcw = fc.pivot(on="quantile", index="year", values=col).sort("year")
    return {
        "municipality_id": municipality_id,
        "history": hist.select("year", pl.col(col).alias("lsi"), *DIMENSIONS).to_dicts(),
        "forecast": fcw.to_dicts(),
    }


@app.get("/municipio/{municipality_id}/explain")
def explain(municipality_id: str, year: int | None = None, profile: str = "default") -> dict:
    """Decompose the score gap vs the median small municipality by dimension."""
    year = year or settings.year_end
    idx = gold("lsi").filter(pl.col("year") == year)
    dim = silver("dim_municipio")
    small = idx.join(dim.select("municipality_id", "is_small"), on="municipality_id").filter(pl.col("is_small"))
    me = idx.filter(pl.col("municipality_id") == municipality_id)
    if me.is_empty():
        raise HTTPException(404, "unknown municipality")
    w = PROFILES[profile]
    contrib = []
    for d in DIMENSIONS:
        mine, med = float(me[d][0]), float(small[d].median())
        # contribution to the log-score gap under the geometric mean
        contrib.append({"dimension": d, "score": mine, "median_small": med, "weight": w[d],
                        "contribution": w[d] * (np.log(max(mine, 1)) - np.log(max(med, 1)))})
    contrib.sort(key=lambda c: -abs(c["contribution"]))
    return {"municipality_id": municipality_id, "year": year, "profile": profile, "drivers": contrib}


class Scenario(BaseModel):
    year: int = Field(default=None, description="defaults to the horizon year")
    weights: dict[str, float] | None = Field(default=None, description="dimension weights, any scale")
    deltas: dict[str, dict[str, float]] = Field(default_factory=dict,
        description="municipality_id -> {dimension: points added to that dimension score}")
    small_only: bool = True
    limit: int = 30


@app.post("/scenario")
def scenario(s: Scenario) -> list[dict]:
    """Re-weight and/or shift dimension scores, return the re-ranked list with rank changes."""
    year = s.year or settings.horizon_end
    if year <= settings.year_end:
        base = gold("lsi").filter(pl.col("year") == year)
    else:
        base = gold("lsi_forecast").filter((pl.col("year") == year) & (pl.col("quantile") == "p50"))
    dim = silver("dim_municipio")
    df = base.join(dim.select("municipality_id", "name", "is_small"), on="municipality_id")
    if s.small_only:
        df = df.filter(pl.col("is_small"))
    w = s.weights or PROFILES["default"]
    if set(w) != set(DIMENSIONS):
        raise HTTPException(400, f"weights must cover exactly {DIMENSIONS}")
    before = df.with_columns(lsi(df, w).alias("lsi")).sort("lsi", descending=True).with_row_index("rank_before", offset=1)
    shifted = df
    for mid, dd in s.deltas.items():
        for d, pts in dd.items():
            if d not in DIMENSIONS:
                raise HTTPException(400, f"unknown dimension {d}")
            shifted = shifted.with_columns(
                pl.when(pl.col("municipality_id") == mid).then(pl.col(d) + pts).otherwise(pl.col(d)).clip(0, 100).alias(d)
            )
    after = shifted.with_columns(lsi(shifted, w).alias("lsi")).sort("lsi", descending=True).with_row_index("rank", offset=1)
    out = after.join(before.select("municipality_id", "rank_before"), on="municipality_id").sort("rank")
    return out.select("rank", "rank_before", "municipality_id", "name", "lsi", *DIMENSIONS).head(s.limit).to_dicts()
