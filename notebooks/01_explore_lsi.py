# %% [markdown]
# # 01 · Explore the panel and the Living-Standards Index
# Runs on whatever is in the lake (mock by default). Open in JupyterLab (right-click →
# "Open With → Notebook", thanks to jupytext-style `# %%` cells) or run as a script.

# %%
import matplotlib.pyplot as plt
import polars as pl

from madrid import lake
from madrid.indicators import DIMENSIONS, PROFILES
from madrid.lsi import rank

obs = lake.read("silver", "observations")
dim = lake.read("silver", "dim_municipio")
idx = lake.read("gold", "lsi")
print(obs.shape, dim.shape, idx.shape)

# %% SQL over the lake: every table is available as <zone>_<table>
lake.sql("""
    SELECT indicator_id, count(*) n, round(avg(value), 2) mean, round(stddev(value), 2) sd,
           sum(value IS NULL) nulls
    FROM silver_observations GROUP BY 1 ORDER BY 1
""")

# %% Today's ranking of small municipalities, default profile
top = rank(idx, dim, year=int(idx["year"].max()))
top.head(15)

# %% How much does the profile change the podium?
pl.concat([
    rank(idx, dim, year=2025, profile=p).head(10).select(pl.lit(p).alias("profile"), "rank", "name", "lsi")
    for p in PROFILES
]).pivot(on="profile", index="rank", values="name")

# %% Trajectories of the top 5 vs the small-town median
top5 = top.head(5)["municipality_id"].to_list()
small = idx.join(dim.select("municipality_id", "is_small"), on="municipality_id").filter(pl.col("is_small"))
med = small.group_by("year").agg(pl.col("lsi_default").median().alias("median")).sort("year")
fig, ax = plt.subplots(figsize=(9, 4))
for mid in top5:
    g = idx.filter(pl.col("municipality_id") == mid).sort("year")
    ax.plot(g["year"], g["lsi_default"], label=dim.filter(pl.col("municipality_id") == mid)["name"][0])
ax.plot(med["year"], med["median"], "k--", label="small-town median")
ax.set_ylabel("LSI (default profile)"); ax.legend(); plt.show()

# %% Which dimensions separate the top from the rest? (2025, small only)
s25 = small.filter(pl.col("year") == 2025)
pl.DataFrame({
    "dimension": DIMENSIONS,
    "top10_mean": [s25.sort("lsi_default", descending=True).head(10)[d].mean() for d in DIMENSIONS],
    "all_small_median": [s25[d].median() for d in DIMENSIONS],
})
