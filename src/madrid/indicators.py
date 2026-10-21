"""Indicator catalogue: the contract between silver and the index.

polarity: +1 means higher is better, -1 means lower is better.
dimension: the LSI dimension the indicator feeds.
"""

from __future__ import annotations

import polars as pl

INDICATORS: list[dict] = [
    # Housing and affordability
    dict(indicator_id="price_m2", name="Housing price per m2", unit="EUR/m2", dimension="housing", polarity=-1, source="MITMA", cadence="yearly"),
    dict(indicator_id="affordability", name="Median income / annual cost of 90 m2", unit="ratio", dimension="housing", polarity=1, source="derived", cadence="yearly"),
    dict(indicator_id="new_housing_share", name="Share of dwellings built after 2000", unit="%", dimension="housing", polarity=1, source="Catastro", cadence="yearly"),
    # Work and income
    dict(indicator_id="median_income", name="Median household income", unit="EUR", dimension="work", polarity=1, source="INE ADRH", cadence="yearly"),
    dict(indicator_id="unemployment_rate", name="Registered unemployment rate", unit="%", dimension="work", polarity=-1, source="SEPE", cadence="monthly"),
    dict(indicator_id="gini", name="Income Gini", unit="index", dimension="work", polarity=-1, source="INE ADRH", cadence="yearly"),
    # Access and mobility
    dict(indicator_id="pt_time_sol", name="Public transport time to Sol", unit="min", dimension="access", polarity=-1, source="CRTM GTFS", cadence="yearly"),
    dict(indicator_id="car_time_sol", name="Car time to Sol", unit="min", dimension="access", polarity=-1, source="OSRM", cadence="yearly"),
    dict(indicator_id="hospital_time", name="Time to nearest hospital", unit="min", dimension="access", polarity=-1, source="OSRM", cadence="yearly"),
    # Services and education
    dict(indicator_id="schools_per_1000", name="Primary schools per 1,000 under-15", unit="per 1000", dimension="services", polarity=1, source="Almudena", cadence="yearly"),
    dict(indicator_id="tertiary_share", name="Share with tertiary education", unit="%", dimension="services", polarity=1, source="Censo", cadence="yearly"),
    dict(indicator_id="broadband_share", name="Coverage at 100 Mbps or more", unit="%", dimension="services", polarity=1, source="MINECO", cadence="yearly"),
    # Environment and safety
    dict(indicator_id="no2_mean", name="NO2 annual mean", unit="ug/m3", dimension="environment", polarity=-1, source="CM air quality", cadence="yearly"),
    dict(indicator_id="heat_days", name="Days above 35 C", unit="days", dimension="environment", polarity=-1, source="AEMET", cadence="yearly"),
    dict(indicator_id="crime_rate", name="Crimes per 1,000 inhabitants", unit="per 1000", dimension="environment", polarity=-1, source="Interior", cadence="yearly"),
    # Demographic vitality
    dict(indicator_id="population", name="Registered population", unit="persons", dimension="vitality", polarity=0, source="INE Padron", cadence="yearly"),
    dict(indicator_id="net_migration_rate", name="Net migration per 1,000", unit="per 1000", dimension="vitality", polarity=1, source="INE EVR", cadence="yearly"),
    dict(indicator_id="share_25_44", name="Share aged 25 to 44", unit="%", dimension="vitality", polarity=1, source="INE Padron", cadence="yearly"),
    dict(indicator_id="ageing_index", name="Ageing index (65+ / under 15)", unit="ratio", dimension="vitality", polarity=-1, source="INE Padron", cadence="yearly"),
    dict(indicator_id="investment_pc", name="Municipal investment per capita", unit="EUR", dimension="vitality", polarity=1, source="Hacienda", cadence="yearly"),
]

DIMENSIONS = ["housing", "work", "access", "services", "environment", "vitality"]

# Preset weight profiles over the six dimensions (sum to 1)
PROFILES: dict[str, dict[str, float]] = {
    "default": {d: 1 / 6 for d in DIMENSIONS},
    "young_family": dict(housing=0.25, work=0.15, access=0.15, services=0.25, environment=0.12, vitality=0.08),
    "remote_worker": dict(housing=0.28, work=0.08, access=0.10, services=0.22, environment=0.24, vitality=0.08),
    "retiree": dict(housing=0.18, work=0.05, access=0.22, services=0.25, environment=0.22, vitality=0.08),
}


def dim_indicator() -> pl.DataFrame:
    return pl.DataFrame(INDICATORS)
