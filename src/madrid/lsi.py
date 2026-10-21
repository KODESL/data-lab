"""Feature matrix and Living-Standards Index (LSI).

Steps (OECD/JRC composite-indicator handbook):
 1. pivot silver observations to a wide municipality x year matrix, interpolate gaps
 2. winsorize each indicator at p1/p99 over all municipality-years
 3. min-max normalize to 0..100 over the whole panel (comparable across years)
 4. flip polarity so 100 is always "better"
 5. dimension sub-index = weighted arithmetic mean of its indicators
 6. LSI = weighted geometric mean of the six dimensions (weak compensability)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .indicators import DIMENSIONS, INDICATORS, PROFILES

SCORED = [i for i in INDICATORS if i["polarity"] != 0]


def features(obs: pl.DataFrame) -> pl.DataFrame:
    """Wide matrix: one row per municipality-year, one column per indicator, gaps interpolated."""
    wide = obs.pivot(on="indicator_id", index=["municipality_id", "year"], values="value").sort(
        "municipality_id", "year"
    )
    cols = [i["indicator_id"] for i in INDICATORS]
    wide = wide.with_columns(
        [pl.col(c).interpolate().forward_fill().backward_fill().over("municipality_id") for c in cols]
    )
    return wide


def normalized(feat: pl.DataFrame, lo_hi: dict[str, tuple[float, float]] | None = None) -> tuple[pl.DataFrame, dict]:
    """Return (0..100 polarity-adjusted scores per indicator, bounds used)."""
    bounds = {}
    out = feat.select("municipality_id", "year")
    for ind in SCORED:
        c = ind["indicator_id"]
        x = feat[c].to_numpy().astype(float)
        if lo_hi and c in lo_hi:
            lo, hi = lo_hi[c]
        else:
            lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        bounds[c] = (float(lo), float(hi))
        s = (np.clip(x, lo, hi) - lo) / (hi - lo) * 100.0
        if ind["polarity"] < 0:
            s = 100.0 - s
        out = out.with_columns(pl.Series(c, s))
    return out, bounds


def dimension_scores(scores: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for d in DIMENSIONS:
        cols = [i["indicator_id"] for i in SCORED if i["dimension"] == d]
        exprs.append(pl.mean_horizontal([pl.col(c) for c in cols]).alias(d))
    return scores.select("municipality_id", "year", *exprs)


def lsi(dims: pl.DataFrame, weights: dict[str, float] | None = None) -> pl.Series:
    """Weighted geometric mean of dimension scores (floored at 1 to keep the log finite)."""
    w = weights or PROFILES["default"]
    tot = sum(w.values())
    logs = sum((w[d] / tot) * np.log(np.clip(dims[d].to_numpy(), 1.0, 100.0)) for d in DIMENSIONS)
    return pl.Series("lsi", np.exp(logs))


def build(obs: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Full gold build on history: features, dimension scores, LSI for every profile."""
    feat = features(obs)
    scores, bounds = normalized(feat)
    dims = dimension_scores(scores)
    index = dims.clone()
    for name, w in PROFILES.items():
        index = index.with_columns(lsi(dims, w).alias(f"lsi_{name}"))
    bounds_df = pl.DataFrame({"indicator_id": list(bounds), "lo": [b[0] for b in bounds.values()], "hi": [b[1] for b in bounds.values()]})
    return dict(features=feat, scores=scores, lsi=index, bounds=bounds_df)


def rank(index: pl.DataFrame, dim_mun: pl.DataFrame, year: int, profile: str = "default", small_only: bool = True) -> pl.DataFrame:
    col = f"lsi_{profile}"
    df = index.filter(pl.col("year") == year).join(dim_mun.select("municipality_id", "name", "is_small", "population_latest"), on="municipality_id")
    if small_only:
        df = df.filter(pl.col("is_small"))
    return df.sort(col, descending=True).with_row_index("rank", offset=1).select(
        "rank", "municipality_id", "name", "population_latest", pl.col(col).alias("lsi"), *DIMENSIONS
    )
