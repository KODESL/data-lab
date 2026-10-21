"""Baseline indicator forecasts to the horizon and the LSI on top of them.

v0 = damped Holt trend per municipality x indicator (statsmodels), with a residual-based
p10/p90 band. This is the baseline every fancier model (LightGBM global model, panel
regressions) must beat in backtests before replacing it; see notebooks/02_forecast_lab.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from statsmodels.tsa.holtwinters import Holt

from .config import settings
from .indicators import INDICATORS, PROFILES
from .lsi import dimension_scores, lsi, normalized


def _holt(y: np.ndarray, h: int) -> tuple[np.ndarray, float]:
    y = y.astype(float)
    if np.allclose(y, y[0]):
        return np.repeat(y[-1], h), 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True)
    resid_sd = float(np.std(fit.resid)) if len(fit.resid) else 0.0
    return fit.forecast(h), resid_sd


def forecast_indicators(feat: pl.DataFrame) -> pl.DataFrame:
    """Long table: municipality_id, indicator_id, year, p10, p50, p90 for horizon years."""
    last = int(feat["year"].max())
    h = settings.horizon_end - last
    years = np.arange(last + 1, settings.horizon_end + 1)
    z = 1.2816  # 80% band
    rows = []
    for mid, g in feat.group_by("municipality_id", maintain_order=True):
        g = g.sort("year")
        for ind in INDICATORS:
            c = ind["indicator_id"]
            y = g[c].to_numpy()
            fc, sd = _holt(y, h)
            # uncertainty grows with the square root of the horizon
            band = z * sd * np.sqrt(np.arange(1, h + 1))
            rows.append(pl.DataFrame({
                "municipality_id": mid[0], "indicator_id": c, "year": years,
                "p10": fc - band, "p50": fc, "p90": fc + band,
            }))
    return pl.concat(rows)


def lsi_on_forecast(fc: pl.DataFrame, bounds: pl.DataFrame) -> pl.DataFrame:
    """Score forecast years with the same bounds as history, for p10/p50/p90."""
    lo_hi = {r["indicator_id"]: (r["lo"], r["hi"]) for r in bounds.iter_rows(named=True)}
    polarity = {i["indicator_id"]: i["polarity"] for i in INDICATORS}
    # The index's p10 is the pessimistic case: the p10 of higher-is-better indicators and the
    # p90 of lower-is-better ones; p90 is the mirror image.
    def pick(q: str) -> pl.Expr:
        if q == "p50":
            return pl.col("p50")
        other = "p90" if q == "p10" else "p10"
        return pl.when(pl.col("indicator_id").replace_strict(polarity, return_dtype=pl.Int64) >= 0).then(pl.col(q)).otherwise(pl.col(other))

    outs = []
    for q in ("p10", "p50", "p90"):
        wide = fc.with_columns(pick(q).alias("v")).pivot(on="indicator_id", index=["municipality_id", "year"], values="v").sort("municipality_id", "year")
        scores, _ = normalized(wide, lo_hi)
        dims = dimension_scores(scores)
        idx = dims.clone()
        for name, w in PROFILES.items():
            idx = idx.with_columns(lsi(dims, w).alias(f"lsi_{name}"))
        outs.append(idx.with_columns(pl.lit(q).alias("quantile")))
    return pl.concat(outs)
