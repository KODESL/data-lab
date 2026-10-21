"""Synthetic panel of Madrid-like municipalities, in the exact silver schema.

The generator is deliberately structured (random effects + trends + a distance-to-Madrid
gradient + spatial neighbours) so that models have something to find, and it is seeded so
every run produces the same data. Replace with real connectors by setting MADRID_DATA_MODE=real.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import settings
from .indicators import INDICATORS


def _municipios(rng: np.random.Generator, n: int) -> pl.DataFrame:
    # Pseudo INE codes 28001..28xxx, Madrid capital gets 28079 with a large population.
    # Madrid capital keeps its real code 28079 and sits in row 0; the rest are synthetic.
    others = [i for i in range(1, n + 10) if i != 79][: n - 1]
    codes = ["28079"] + [f"28{i:03d}" for i in others]
    names = ["Madrid"] + [f"Municipio {i:03d}" for i in others]
    # Positions: Madrid at the origin, others scattered 5..90 km away
    r = rng.uniform(5, 90, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(theta), r * np.sin(theta)
    altitude = np.clip(600 + 8 * r + rng.normal(0, 120, n), 450, 1500)
    # Populations: log-normal, more people closer to Madrid
    pop = np.exp(rng.normal(8.2, 1.4, n) - 0.012 * r).astype(int)
    pop = np.clip(pop, 120, 250_000)
    df = pl.DataFrame(
        dict(municipality_id=codes, name=names, x_km=x, y_km=y, dist_km=r, altitude_m=altitude,
             population_2000=pop)
    )
    # Madrid capital: at the origin, big population
    is_madrid = pl.col("municipality_id") == "28079"
    return df.with_columns(
        pl.when(is_madrid).then(pl.lit(2_900_000)).otherwise(pl.col("population_2000")).alias("population_2000"),
        pl.when(is_madrid).then(0.0).otherwise(pl.col("dist_km")).alias("dist_km"),
        pl.when(is_madrid).then(0.0).otherwise(pl.col("x_km")).alias("x_km"),
        pl.when(is_madrid).then(0.0).otherwise(pl.col("y_km")).alias("y_km"),
    )


def generate() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (dim_municipio, observations) in silver schema."""
    rng = np.random.default_rng(settings.seed)
    n = settings.mock_municipios
    years = np.arange(settings.year_start, settings.year_end + 1)
    T = len(years)
    t = (years - settings.year_start) / 10.0  # decades

    dim = _municipios(rng, n)
    dist = dim["dist_km"].to_numpy()
    pop0 = dim["population_2000"].to_numpy().astype(float)
    alt = dim["altitude_m"].to_numpy()

    # Latent "attractiveness" random effect per municipality, spatially smoothed
    xy = dim.select("x_km", "y_km").to_numpy()
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * 15.0**2))
    w /= w.sum(1, keepdims=True)
    u = w @ rng.normal(0, 1, n)  # smoothed random effect
    u = (u - u.mean()) / u.std()

    def panel(base, trend, noise, re_scale=0.0, ar=0.6):
        """base, trend: arrays (n,) ; returns (n, T) with AR(1) noise."""
        out = np.empty((n, T))
        e = rng.normal(0, noise, (n, T))
        for k in range(T):
            prev = e[:, k - 1] if k else 0
            e[:, k] = ar * prev + e[:, k]
        for k in range(T):
            out[:, k] = base + trend * t[k] + re_scale * u + e[:, k]
        return out

    growth = 0.08 + 0.05 * u - 0.0008 * dist  # per decade log growth
    population = pop0[:, None] * np.exp(growth[:, None] * t[None, :] + rng.normal(0, 0.01, (n, T)).cumsum(1))

    price_m2 = panel(2600 - 18 * dist, 350, 80, re_scale=250)
    price_m2 = np.clip(price_m2, 600, None)
    median_income = panel(26000 - 60 * dist, 2200, 600, re_scale=2500)
    affordability = median_income / (price_m2 * 90 / 25)
    new_housing_share = np.clip(panel(18 + 0.05 * dist, 6, 1.5, re_scale=3), 2, 70)
    unemployment_rate = np.clip(panel(12 + 0.03 * dist, -1.5, 1.2, re_scale=-1.5) + 4 * np.sin((years - 2008) / 3)[None, :], 2, 35)
    gini = np.clip(panel(0.33, -0.004, 0.006, re_scale=-0.01), 0.22, 0.5)
    pt_time_sol = np.clip(panel(12 + 1.4 * dist, -4, 2, re_scale=-3), 8, 200)
    car_time_sol = np.clip(panel(10 + 1.0 * dist, -1, 1.5), 8, 150)
    hospital_time = np.clip(panel(6 + 0.35 * dist, -1, 1.5), 3, 60)
    schools_per_1000 = np.clip(panel(4.5 - 0.004 * dist, 0.2, 0.3, re_scale=0.4), 0.5, 12)
    tertiary_share = np.clip(panel(28 - 0.12 * dist, 5, 1.0, re_scale=4), 5, 70)
    broadband_share = np.clip(panel(20 - 0.3 * dist, 32, 2, re_scale=4), 0, 100)
    no2_mean = np.clip(panel(34 - 0.3 * dist, -4, 1.5), 4, 60)
    heat_days = np.clip(panel(14 - 0.012 * (alt - 600), 6, 2.5), 0, 80)
    crime_rate = np.clip(panel(42 - 0.25 * dist, -2, 3, re_scale=-4), 5, 120)
    net_migration_rate = panel(2 + 1.5 * u - 0.03 * dist, 1.0, 3.0)
    share_25_44 = np.clip(panel(32 - 0.04 * dist, -3.5, 0.8, re_scale=1.5), 15, 45)
    ageing_index = np.clip(panel(0.9 + 0.012 * dist, 0.35, 0.06, re_scale=-0.15), 0.3, 6)
    investment_pc = np.clip(panel(180 + 0.4 * dist, 10, 40, re_scale=30), 20, 900)

    series = dict(
        population=population, price_m2=price_m2, affordability=affordability,
        new_housing_share=new_housing_share, median_income=median_income,
        unemployment_rate=unemployment_rate, gini=gini, pt_time_sol=pt_time_sol,
        car_time_sol=car_time_sol, hospital_time=hospital_time, schools_per_1000=schools_per_1000,
        tertiary_share=tertiary_share, broadband_share=broadband_share, no2_mean=no2_mean,
        heat_days=heat_days, crime_rate=crime_rate, net_migration_rate=net_migration_rate,
        share_25_44=share_25_44, ageing_index=ageing_index, investment_pc=investment_pc,
    )
    ids = dim["municipality_id"].to_list()
    frames = []
    for ind in INDICATORS:
        m = series[ind["indicator_id"]]
        frames.append(pl.DataFrame({
            "municipality_id": np.repeat(ids, T),
            "indicator_id": ind["indicator_id"],
            "year": np.tile(years, n),
            "value": m.ravel().astype(float),
            "unit": ind["unit"],
            "source": "mock:" + ind["source"],
            "quality_flag": "synthetic",
        }))
    obs = pl.concat(frames)
    # Small-town noise: drop ~3% of values at random to exercise imputation
    mask = rng.uniform(size=len(obs)) < 0.03
    obs = obs.with_columns(pl.when(pl.Series(mask)).then(None).otherwise(pl.col("value")).alias("value"))

    dim = dim.with_columns(
        pl.Series("population_latest", population[:, -1].astype(int)),
    ).with_columns((pl.col("population_latest") < settings.small_threshold).alias("is_small"))
    return dim, obs
