# %% [markdown]
# # 02 · Forecast lab: beat the baseline
# The shipped baseline is a damped Holt trend per series (`madrid.forecast`). This notebook
# shows the backtest harness: fit on years <= cutoff, score the next H years, compare any
# candidate (here a global LightGBM with lags and a spatial lag) against the baseline.
# Only a model that wins here gets wired into the `forecast` job.

# %%
import numpy as np
import polars as pl

from madrid import lake
from madrid.forecast import _holt
from madrid.indicators import INDICATORS

feat = lake.read("gold", "features")
dim = lake.read("silver", "dim_municipio")
CUTOFF, H = 2015, 10
cols = [i["indicator_id"] for i in INDICATORS]

# %% Baseline backtest: sMAPE per indicator at horizon H
def smape(a, f):
    return 100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-9))

rows = []
for mid, g in feat.group_by("municipality_id", maintain_order=True):
    g = g.sort("year")
    train, test = g.filter(pl.col("year") <= CUTOFF), g.filter(pl.col("year") > CUTOFF).head(H)
    for c in cols:
        fc, _ = _holt(train[c].to_numpy(), H)
        rows.append({"municipality_id": mid[0], "indicator_id": c, "smape": smape(test[c].to_numpy(), fc[: len(test)])})
baseline = pl.DataFrame(rows).group_by("indicator_id").agg(pl.col("smape").mean()).sort("smape")
baseline

# %% Candidate: global LightGBM, direct multi-horizon, with lags + a spatial lag feature
# (needs the [models] extra). Build the supervised frame once.
import lightgbm as lgb

xy = dim.select("municipality_id", "x_km", "y_km").to_pandas().set_index("municipality_id")
# k nearest neighbours by centroid distance (queen contiguity once real polygons exist)
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=6).fit(xy.values)
_, nbr = nn.kneighbors(xy.values)
neighbours = {m: [xy.index[j] for j in nbr[i][1:]] for i, m in enumerate(xy.index)}

def supervised(c: str, horizon: int) -> pl.DataFrame:
    base = feat.select("municipality_id", "year", c).sort("municipality_id", "year")
    lagged = base.with_columns(
        *[pl.col(c).shift(k).over("municipality_id").alias(f"lag{k}") for k in (1, 2, 3, 5)],
        (pl.col(c) - pl.col(c).shift(1).over("municipality_id")).alias("d1"),
        pl.col(c).shift(-horizon).over("municipality_id").alias("target"),
    )
    # spatial lag: mean of neighbours' current value
    nb = pl.DataFrame({"municipality_id": list(neighbours), "nb": [neighbours[m] for m in neighbours]}).explode("nb")
    sl = (nb.join(base.rename({"municipality_id": "nb", c: "nb_val"}), on="nb")
            .group_by("municipality_id", "year").agg(pl.col("nb_val").mean().alias("spatial_lag")))
    return lagged.join(sl, on=["municipality_id", "year"]).drop_nulls()

def backtest_lgb(c: str, horizon: int) -> float:
    df = supervised(c, horizon)
    X = ["lag1", "lag2", "lag3", "lag5", "d1", "spatial_lag", "year"]
    tr = df.filter(pl.col("year") + horizon <= CUTOFF)
    te = df.filter((pl.col("year") == CUTOFF))  # predict CUTOFF + horizon from CUTOFF
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, verbose=-1)
    m.fit(tr.select(X).to_numpy(), tr["target"].to_numpy())
    pred = m.predict(te.select(X).to_numpy())
    return smape(te["target"].to_numpy(), pred)

# %% Compare at horizon 5 for every indicator (a few seconds)
cand = pl.DataFrame({"indicator_id": cols, "smape_lgb_h5": [backtest_lgb(c, 5) for c in cols]})
baseline.join(cand, on="indicator_id").with_columns((pl.col("smape_lgb_h5") < pl.col("smape")).alias("lgb_wins"))

# %% [markdown]
# Next steps in this lab:
# * `linearmodels.PanelOLS` with entity effects for the gravity model on `silver_flows_evr` once real EVR data is in
# * `mapie` to calibrate the p10/p90 bands on these backtest residuals
# * `shap.TreeExplainer(m)` on the winning LightGBM for the explanation table
# * log every run with `mlflow.start_run()` (tracking URI is set in compose)
