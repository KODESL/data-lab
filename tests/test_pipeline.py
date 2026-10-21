import tempfile

import polars as pl
from fastapi.testclient import TestClient


def test_end_to_end(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("MADRID_LAKE_ROOT", tmp)
    monkeypatch.setenv("MADRID_MOCK_MUNICIPIOS", "40")
    monkeypatch.setenv("MADRID_HORIZON_END", "2030")
    from importlib import reload

    import madrid.config as cfg
    reload(cfg)
    import madrid.lake as lake
    reload(lake)
    import madrid.jobs as jobs
    reload(jobs)

    jobs.seed()
    jobs.build_gold()
    jobs.forecast()

    idx = lake.read("gold", "lsi")
    assert idx["lsi_default"].min() >= 1 and idx["lsi_default"].max() <= 100
    rk = lake.read("gold", "ranking_horizon")
    assert rk.filter(pl.col("profile") == "default")["rank_p50"].min() == 1

    import api.main as api
    reload(api)
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    r = client.get("/ranking", params={"year": 2030, "profile": "young_family"}).json()
    assert r and r[0]["rank_p50"] == 1
    mid = r[0]["municipality_id"]
    assert client.get(f"/municipio/{mid}/trajectory").json()["forecast"]
    sc = client.post("/scenario", json={"deltas": {mid: {"access": -40}}, "limit": 200}).json()
    me = next(x for x in sc if x["municipality_id"] == mid)
    assert me["rank"] >= me["rank_before"]  # worse access can never improve the rank
