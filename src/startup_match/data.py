"""Synthetic but structurally realistic startup / founder / funding / investor data.

Every startup has a hidden quality ``q``. It drives funding cadence, traction
and founder strength, but each observable is noisy, so no single feature gives
the answer away. A per-year funding climate adds the kind of time drift that
makes time-based validation necessary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATA_END, HORIZON_DAYS, STAGES, TEST_END

# Startups describe themselves in product language; investors write theses in
# market language. The shared terms are what an embedding model must bridge.
SECTORS: dict[str, dict[str, list[str]]] = {
    "healthtech": {
        "shared": ["healthcare", "clinical", "patients", "hospitals"],
        "product": ["medical imaging", "radiology triage", "remote patient monitoring", "clinical documentation", "diagnostic assays"],
        "thesis": ["healthcare AI", "digital health", "clinical automation", "care delivery"],
    },
    "fintech": {
        "shared": ["payments", "banking", "financial", "money"],
        "product": ["cross-border remittance", "card issuing API", "SME lending", "fraud scoring", "payroll payouts"],
        "thesis": ["financial infrastructure", "embedded finance", "payments rails", "credit"],
    },
    "climate": {
        "shared": ["carbon", "energy", "emissions", "grid"],
        "product": ["battery analytics", "carbon accounting", "solar asset monitoring", "heat pump installation", "EV charging"],
        "thesis": ["decarbonisation", "clean energy", "climate tech", "energy transition"],
    },
    "devtools": {
        "shared": ["developers", "software", "code", "engineering teams"],
        "product": ["CI pipeline acceleration", "observability tracing", "API testing", "code review automation", "feature flags"],
        "thesis": ["developer productivity", "infrastructure software", "open-source tooling", "DevOps"],
    },
    "edtech": {
        "shared": ["learning", "students", "teachers", "education"],
        "product": ["adaptive tutoring", "exam preparation", "classroom analytics", "language practice", "skills bootcamps"],
        "thesis": ["future of learning", "workforce upskilling", "K-12", "education access"],
    },
    "logistics": {
        "shared": ["supply chain", "shipping", "freight", "warehouses"],
        "product": ["route optimisation", "last-mile delivery", "freight brokerage", "inventory forecasting", "cold chain tracking"],
        "thesis": ["supply chain resilience", "logistics automation", "trade", "mobility"],
    },
    "cybersecurity": {
        "shared": ["security", "threats", "identity", "compliance"],
        "product": ["phishing detection", "cloud posture management", "passwordless login", "SOC automation", "vulnerability scanning"],
        "thesis": ["cyber defence", "zero trust", "identity security", "enterprise risk"],
    },
    "agritech": {
        "shared": ["farmers", "crops", "agriculture", "food"],
        "product": ["soil sensing", "crop yield prediction", "farm marketplace", "precision irrigation", "agri settlement ledger"],
        "thesis": ["food systems", "agricultural productivity", "rural markets", "farm-to-fork"],
    },
}
SECTOR_NAMES = list(SECTORS)
COUNTRIES = ["IN", "SG", "US", "UK", "DE", "AE", "AU", "CA"]
EMPLOYERS = ["a large bank", "a global consultancy", "a cloud provider", "a national hospital", "a logistics major", "a university lab", "a big-tech firm", "a government agency"]
ROLES = ["software engineer", "product manager", "data scientist", "operations lead", "researcher", "sales director", "designer"]

# Funding climate by year (log-odds shift): the 2021 boom, the 2022-24 downturn.
CLIMATE = {y: 0.0 for y in range(2010, 2030)}
CLIMATE.update({2020: 0.2, 2021: 0.7, 2022: -0.1, 2023: -0.6, 2024: -0.45, 2025: -0.2})

STAGE_AMOUNT_USD = {"pre-seed": 0.4e6, "seed": 2e6, "series-a": 9e6, "series-b": 25e6, "series-c": 55e6, "series-d": 110e6}


@dataclass
class Dataset:
    startups: pd.DataFrame   # one row per startup, static attributes + hidden truth
    founders: pd.DataFrame   # one row per founder
    rounds: pd.DataFrame     # one row per funding round (dated events)
    investors: pd.DataFrame  # one row per investor
    snapshots: pd.DataFrame  # one prediction point per startup + label


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _startup_name(rng: np.random.Generator) -> str:
    a = ["Nova", "Kite", "Lumen", "Orbit", "Pivot", "Vertex", "Delta", "Atlas", "Ember", "Flux", "Quill", "Rivet", "Sable", "Tidal", "Umbra", "Zephyr"]
    b = ["Labs", "AI", "Works", "Stack", "Pay", "Health", "Grid", "Path", "Logic", "Base", "Loop", "Forge"]
    return f"{rng.choice(a)}{rng.choice(b)}"


def generate(n_startups: int = 6000, n_investors: int = 400, seed: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- startups
    rows, founder_rows, round_rows = [], [], []
    for sid in range(n_startups):
        sector_name = SECTOR_NAMES[rng.integers(len(SECTOR_NAMES))]
        sector = SECTORS[sector_name]
        q = rng.normal()
        founded = pd.Timestamp("2012-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 12)))
        products = rng.choice(sector["product"], size=2, replace=False)
        shared = rng.choice(sector["shared"], size=2, replace=False)
        desc = (f"{products[0][0].upper() + products[0][1:]} and {products[1]} for {shared[0]}; "
                f"we help {shared[1]} teams move faster.")
        rows.append(dict(startup_id=sid, name=f"{_startup_name(rng)}-{sid}", sector=sector_name,
                         country=rng.choice(COUNTRIES), founded=founded, description=desc, q_true=q))

        # founders: experience and exits correlate with quality, noisily
        for _ in range(int(rng.integers(1, 4))):
            years = max(0.0, rng.normal(7 + 2.0 * q, 4))
            exits = int(rng.poisson(max(0.05, 0.25 + 0.35 * q)))
            technical = bool(rng.random() < 0.55)
            role = "software engineer" if technical and rng.random() < 0.6 else rng.choice(ROLES)
            bio = (f"Former {role} at {rng.choice(EMPLOYERS)} with {years:.0f} years of experience in "
                   f"{rng.choice(sector['shared'])}" + (f"; previously founded {exits} company(ies) that exited" if exits else "") + ".")
            founder_rows.append(dict(startup_id=sid, years_experience=years, prior_exits=exits,
                                     technical=technical, bio=bio))

        # funding rounds: cadence and continuation driven by q and climate
        date = founded + pd.Timedelta(days=int(rng.exponential(240)) + 30)
        for stage_idx, stage in enumerate(STAGES):
            if date > DATA_END:
                break
            amount = STAGE_AMOUNT_USD[stage] * float(np.exp(0.35 * q + rng.normal(0, 0.4)))
            round_rows.append(dict(startup_id=sid, date=date, stage=stage, stage_idx=stage_idx, amount_usd=amount))
            p_next = _sigmoid(0.9 + 1.2 * q + CLIMATE.get(date.year, 0.0) - 0.35 * stage_idx)
            if rng.random() > p_next:
                break
            gap_days = int(np.clip(rng.exponential(max(150.0, 560 - 140 * q)), 90, 1800))
            next_date = date + pd.Timedelta(days=gap_days)
            # a bad climate in the target year can delay or kill the raise
            if rng.random() > _sigmoid(1.5 + 2.0 * CLIMATE.get(next_date.year, 0.0) + 0.5 * q):
                next_date += pd.Timedelta(days=int(rng.integers(180, 540)))
            date = next_date

    startups = pd.DataFrame(rows)
    founders = pd.DataFrame(founder_rows)
    rounds = pd.DataFrame(round_rows).sort_values(["startup_id", "date"]).reset_index(drop=True)

    # --------------------------------------------------------------- snapshots
    first_round = rounds.groupby("startup_id")["date"].min()
    snap_rows = []
    for sid, first in first_round.items():
        lo = first + pd.Timedelta(days=30)
        if lo >= TEST_END:
            continue
        span = (TEST_END - lo).days
        snap = lo + pd.Timedelta(days=int(rng.integers(0, span)))
        q = startups.at[sid, "q_true"]
        # traction observed at the snapshot (noisy proxies of q)
        snap_rows.append(dict(
            startup_id=sid, snapshot_date=snap,
            revenue_growth_yoy=float(np.clip(rng.normal(0.4 + 0.45 * q, 0.6), -0.9, 6.0)),
            customers=int(np.exp(rng.normal(3.5 + 0.6 * q, 1.0))),
            headcount=int(max(2, np.exp(rng.normal(2.4 + 0.3 * q, 0.6)))),
            hiring_growth_6m=float(rng.normal(0.08 + 0.10 * q, 0.15)),
            web_traffic_growth_6m=float(rng.normal(0.10 + 0.15 * q, 0.35)),
        ))
    snapshots = pd.DataFrame(snap_rows)
    snapshots["label"] = _labels(snapshots, rounds)

    # --------------------------------------------------------------- investors
    inv_rows = []
    for iid in range(n_investors):
        k = int(rng.integers(1, 4))
        pref = list(rng.choice(SECTOR_NAMES, size=k, replace=False))
        lo_stage = int(rng.integers(0, 4))
        stages = STAGES[lo_stage: lo_stage + int(rng.integers(2, 4))]
        tmin = STAGE_AMOUNT_USD[stages[0]] * 0.05
        tmax = STAGE_AMOUNT_USD[stages[-1]] * 0.25
        risk = float(rng.uniform(0, 1))
        thesis_terms = []
        for s in pref:
            thesis_terms += list(rng.choice(SECTORS[s]["thesis"], size=2, replace=False))
            thesis_terms.append(str(rng.choice(SECTORS[s]["shared"])))
        style = "high-conviction bets on ambitious founders" if risk > 0.6 else (
            "capital-efficient companies with proven traction" if risk < 0.35 else "balanced early-stage portfolios")
        thesis = f"We back {', '.join(thesis_terms)} companies at {' to '.join([stages[0], stages[-1]])}; we favour {style}."
        inv_rows.append(dict(investor_id=iid, name=f"Fund {iid:03d}", preferred_sectors=pref, stages=stages,
                             ticket_min=tmin, ticket_max=tmax, risk_appetite=risk, thesis=thesis))
    investors = pd.DataFrame(inv_rows)

    return Dataset(startups, founders, rounds, investors, snapshots)


def _labels(snapshots: pd.DataFrame, rounds: pd.DataFrame) -> np.ndarray:
    """1 if the startup raises any round in (snapshot, snapshot + 24 months]."""
    by_sid = {sid: g["date"].to_numpy() for sid, g in rounds.groupby("startup_id")}
    horizon = np.timedelta64(HORIZON_DAYS, "D")
    out = np.zeros(len(snapshots), dtype=int)
    for i, (sid, snap) in enumerate(zip(snapshots["startup_id"], snapshots["snapshot_date"])):
        d = by_sid.get(sid)
        if d is None:
            continue
        s = np.datetime64(snap)
        out[i] = int(((d > s) & (d <= s + horizon)).any())
    return out
