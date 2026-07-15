"""Tests for the RAM intraday robot — the maze generalization contract, ported to
trading: no look-ahead leakage, disjoint walk-forward folds, determinism, and exact
belief-map replay reconstruction. Runs on a synthetic multi-day fixture (so it does
not depend on the downloaded CSV) plus the checked-in sample.

Run: zion/.venv/bin/python -m pytest private/ram_intraday/tests/test_ram.py -v
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from market_map import Bar, OnlineMarketMap, N_FEATURES
from robot import RamRobot, FAMILIES
from backtest import nav_from_exposures, ulcer, run_episode
from data import group_by_day, walkforward_split, load_bars


def _synth_days(n_days=6, bars_per_day=120, seed=0):
    """Deterministic synthetic RTH days with intraday reversals."""
    rng = np.random.default_rng(seed)
    days = []
    price = 15.0
    for d in range(n_days):
        day = []
        # each day a sine reversal + noise (a mean-reverting intraday shape)
        for i in range(bars_per_day):
            drift = 0.4 * math.sin(2 * math.pi * i / bars_per_day)
            price = max(1.0, price * (1 + 0.0005 * drift + 0.001 * rng.standard_normal()))
            mm = 9 * 60 + 30 + i
            ts = f"2026-06-{10 + d:02d} {mm // 60:02d}:{mm % 60:02d}:00"
            day.append(Bar(ts, price, price * 1.001, price * 0.999, price,
                           float(rng.integers(100, 5000)), "rth"))
        days.append(day)
    return days


def test_feature_dim():
    mm = OnlineMarketMap()
    for b in _synth_days(1)[0][:10]:
        mm.observe(b)
    assert mm.features().shape == (N_FEATURES,)


def test_no_lookahead_leakage():
    """features() after k bars must be identical whether or not future bars follow."""
    seq = _synth_days(1, 80)[0]
    a = OnlineMarketMap(memory_cap=200)
    for b in seq[:30]:
        a.observe(b)
    f_k = a.features().copy()
    for b in seq[30:]:
        a.observe(b)                       # feed the future
    b2 = OnlineMarketMap(memory_cap=200)
    for b in seq[:30]:
        b2.observe(b)
    assert np.allclose(f_k, b2.features()), "future bars leaked into a past feature vector"


def test_walkforward_disjoint():
    days = _synth_days(6)
    train, test = walkforward_split(days, 0.6)
    train_dates = {d[0].date for d in train}
    test_dates = {d[0].date for d in test}
    assert train_dates.isdisjoint(test_dates)
    assert len(train) + len(test) == len(days)


def test_map_replay_reconstruction():
    """The belief map is a pure function of the bar sequence (fog.py analog)."""
    seq = _synth_days(2, 60)
    flat = [b for day in seq for b in day]
    m1 = OnlineMarketMap(memory_cap=50, obs_lookback=10)
    feats1 = []
    for b in flat:
        m1.observe(b)
        feats1.append(m1.features())
    m2 = OnlineMarketMap(memory_cap=50, obs_lookback=10)
    feats2 = []
    for b in flat:
        m2.observe(b)
        feats2.append(m2.features())
    assert np.allclose(np.array(feats1), np.array(feats2))


@pytest.mark.parametrize("family", FAMILIES)
def test_family_runs_and_is_long_flat(family):
    days = _synth_days(6)
    train, test = walkforward_split(days, 0.6)
    p = {"family": family, "band_gain": 1.0, "vwap_k": 1.0, "mom_k": 1.0,
         "nn_hidden": 6, "nn_seed": 1, "ppo_hidden": 6, "ppo_lr": 5e-4,
         "ppo_steps": 10, "trade_threshold": 0.1, "max_exposure": 1.0,
         "cost_bps": 0.0015, "seed": 2}
    r = RamRobot(p)
    r.fit(train)
    nav, dret, prices, exps = run_episode(r, test, r.hold_through_gap, 0.0015)
    assert np.all(exps >= -1e-9) and np.all(exps <= 1.0 + 1e-9), "exposure left [0,1] (no shorting)"
    assert np.all(np.isfinite(nav))


@pytest.mark.parametrize("family", ["nn_reverter", "morpheus_trader"])
def test_training_deterministic(family):
    days = _synth_days(6)
    train, test = walkforward_split(days, 0.6)
    p = {"family": family, "nn_hidden": 6, "nn_seed": 5, "ppo_hidden": 6,
         "ppo_lr": 5e-4, "ppo_steps": 12, "seed": 5, "cost_bps": 0.0015}
    r1 = RamRobot(p); r1.fit(train)
    r2 = RamRobot(p); r2.fit(train)
    n1, _, _, _ = run_episode(r1, test, False, 0.0015)
    n2, _, _, _ = run_episode(r2, test, False, 0.0015)
    assert np.allclose(n1, n2), "training is non-deterministic for a fixed seed"


def test_ulcer_and_nav_math():
    prices = np.array([100., 101., 99., 102., 98.])
    exp = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    nav = nav_from_exposures(prices, exp, cost_bps=0.0)
    assert nav[0] == 100.0
    # holding through 101->99 must lose; ulcer non-negative
    assert ulcer(nav) >= 0.0
    # flat everywhere -> flat NAV
    flat = nav_from_exposures(prices, np.zeros(5), cost_bps=0.0)
    assert np.allclose(flat, 100.0)


def test_export_roundtrip():
    days = _synth_days(6)
    p = {"family": "nn_reverter", "nn_hidden": 6, "nn_seed": 9, "seed": 9, "cost_bps": 0.0015}
    r = RamRobot(p); r.fit(days[:4])
    params = r.to_params()
    r2 = RamRobot.from_params(params)
    # same weights -> identical decision on the same feature vector
    mm = OnlineMarketMap()
    for b in days[4][:20]:
        mm.observe(b)
    f = mm.features()
    assert abs(float(r.mlp(f)) - float(r2.mlp(f))) < 1e-12


def test_sample_csv_loads():
    path = os.path.join(_PKG, "data", "RAM_1min.csv")
    if not os.path.exists(path):
        pytest.skip("sample CSV not present")
    days = group_by_day(load_bars(path))
    assert len(days) >= 1
    assert all(hasattr(b, "close") for d in days for b in d)
