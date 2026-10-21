#!/usr/bin/env python3
"""Smoke test against a running API. Usage: python scripts/smoke.py [base_url]

Walks the same path a user of the UI would: catalogue, today's ranking, the 2036 ranking
for a profile, one municipality's trajectory and drivers, and a what-if scenario.
Standard library only, so it runs outside the project's virtualenv too."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def post(path: str, body: dict):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def step(title: str) -> None:
    print(f"\n== {title}")


step("health")
print(get("/health"))

step("catalog")
cat = get("/catalog")
print("dimensions:", cat["dimensions"])
print("profiles:  ", list(cat["profiles"]))
print("indicators:", len(cat["indicators"]), "| years:", cat["years"])

step("top 5 small municipalities today, default profile")
for r in get("/ranking?year=2025&profile=default&limit=5"):
    print(f"{r['rank']:>2} {r['municipality_id']} {r['name']:<16} LSI {r['lsi']:.1f}  "
          + "  ".join(f"{d[:4]} {r[d]:4.0f}" for d in cat["dimensions"]))

step("top 5 in 2036 for a young family, with rank band")
rows = get("/ranking?year=2036&profile=young_family&limit=5")
for r in rows:
    print(f"{r['rank_p50']:>2} {r['name']:<16} LSI {r['lsi_p50']:.1f}  band [{r['rank_lo']}, {r['rank_hi']}]  robust_top20={r['robust_top20']}")
top = get("/ranking?year=2036&limit=1")[0]["municipality_id"]

step(f"trajectory of {top}: last 3 history years and first 3 forecast years")
t = get(f"/municipio/{top}/trajectory")
for r in t["history"][-3:]:
    print(f"hist {r['year']}  LSI {r['lsi']:.1f}")
for r in t["forecast"][:3]:
    print(f"fcst {r['year']}  p10 {r['p10']:.1f}  p50 {r['p50']:.1f}  p90 {r['p90']:.1f}")

step(f"why {top} ranks where it does: top 3 drivers vs the small-town median")
for c in get(f"/municipio/{top}/explain")["drivers"][:3]:
    print(f"{c['dimension']:<12} score {c['score']:5.1f}  median {c['median_small']:5.1f}  contribution {c['contribution']:+.3f}")

step(f"scenario: housing weighs 3x, and {top} loses 40 access points")
rows = post("/scenario", {
    "weights": {"housing": 3, "work": 1, "access": 1, "services": 1, "environment": 1, "vitality": 1},
    "deltas": {top: {"access": -40}},
    "limit": 200,
})
me = next(r for r in rows if r["municipality_id"] == top)
print(f"{top}: rank {me['rank_before']} -> {me['rank']}")
for r in rows[:5]:
    print(f"{r['rank']:>2} (was {r['rank_before']:>2}) {r['name']:<16} LSI {r['lsi']:.1f}")

assert me["rank"] >= me["rank_before"], "losing access must not improve the rank"
print("\nall smoke checks passed")
