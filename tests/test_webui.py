"""Tests for the webui cockpit backend — the engine is the source of truth.

The headline guarantee (test_engine_parity_*) is that the tracer that feeds the UI
produces NAV/exposures *identical* to backtest.run_episode, so the cockpit can never
drift from the engine. The rest cover the genuinely new numeric code: resample
bucketing, trade derivation, return baselining, replay determinism, and the additive
belief_snapshot(). Runs on a synthetic multi-day fixture + the checked-in sample.

Run: .venv/bin/python -m pytest tests/test_webui.py -v
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

from market_map import Bar, OnlineMarketMap, FEATURE_NAMES
from robot import RamRobot
from backtest import run_episode
from data import group_by_day, load_bars
from webui import replay as RP
from webui import resample as RS


def _synth_days(n_days=6, bars_per_day=120, seed=0):
    """Deterministic RTH days with intraday reversals (mirrors test_ram)."""
    rng = np.random.default_rng(seed)
    days = []
    price = 15.0
    for d in range(n_days):
        day = []
        for i in range(bars_per_day):
            drift = 0.4 * math.sin(2 * math.pi * i / bars_per_day)
            price = max(1.0, price * (1 + 0.0005 * drift + 0.001 * rng.standard_normal()))
            mm = 9 * 60 + 30 + i
            ts = f"2026-06-{10 + d:02d} {mm // 60:02d}:{mm % 60:02d}:00"
            day.append(Bar(ts, price, price * 1.001, price * 0.999, price,
                           float(rng.integers(100, 5000)), "rth"))
        days.append(day)
    return days


def _flat_bars(days):
    return [b for day in days for b in day]


# ------------------------- engine parity (source of truth) ---------------- #

@pytest.mark.parametrize("hold_gap", [False, True])
def test_engine_parity(hold_gap):
    """traced_episode nav/exposures/prices must EQUAL run_episode exactly."""
    days = _synth_days(5)
    r1 = RamRobot({"family": "band_reverter", "hold_through_gap": hold_gap})
    nav0, dret0, prices0, exps0 = run_episode(r1, days, hold_gap, r1.cost_bps)

    r2 = RamRobot({"family": "band_reverter", "hold_through_gap": hold_gap})
    tr = RP.traced_episode(r2, days, hold_gap, r2.cost_bps)

    assert np.array_equal(np.array(tr["exposures"]), exps0)
    assert np.array_equal(np.array(tr["prices"]), prices0)
    assert np.allclose(np.array(tr["nav"]), nav0)


def test_engine_parity_on_sample_csv():
    csv = os.path.join(_PKG, "data", "RAM_1min.csv")
    if not os.path.exists(csv):
        pytest.skip("sample CSV not present")
    days = group_by_day(load_bars(csv))
    r1 = RamRobot(); nav0, _, prices0, exps0 = run_episode(r1, days, r1.hold_through_gap, r1.cost_bps)
    r2 = RamRobot(); tr = RP.traced_episode(r2, days, r2.hold_through_gap, r2.cost_bps)
    assert np.allclose(np.array(tr["nav"]), nav0)
    assert np.array_equal(np.array(tr["exposures"]), exps0)


def test_traced_belief_aligned():
    """Every bar has a features vector and a belief snapshot, aligned to prices."""
    days = _synth_days(3)
    tr = RP.traced_episode(RamRobot(), days, False, 0.001)
    n = len(tr["prices"])
    assert len(tr["features"]) == n and len(tr["belief"]) == n
    assert all(len(f) == len(FEATURE_NAMES) for f in tr["features"])
    # belief price tracks the bar close
    assert tr["belief"][-1]["price"] == pytest.approx(tr["prices"][-1])


# ------------------------- trade derivation ------------------------------- #

def test_derive_trades_side_and_threshold():
    ts = [f"t{i}" for i in range(6)]
    prices = [10, 11, 12, 13, 14, 15]
    exps = [0.0, 0.8, 0.8, 0.3, 0.3, 0.0]   # buy, hold, sell, hold, sell
    tr = RP.derive_trades(ts, prices, exps)
    assert [t["side"] for t in tr] == ["buy", "sell", "sell"]
    assert [t["i"] for t in tr] == [1, 3, 5]
    assert tr[0]["dExposure"] == pytest.approx(0.8)
    assert tr[0]["exposure_after"] == pytest.approx(0.8)
    # a threshold suppresses the small moves
    tr2 = RP.derive_trades(ts, prices, exps, threshold=0.6)
    assert [t["i"] for t in tr2] == [1]     # only the 0.8 jump survives


def test_native_trades_match_executed():
    """At 1-min (mean rollup over 1-bar buckets) the derived trades equal the
    deadzone-executed exposure changes in the raw episode."""
    days = _synth_days(4)
    tr = RP.traced_episode(RamRobot(), days, False, 0.001)
    exps = np.array(tr["exposures"])
    executed = int(np.count_nonzero(np.abs(np.diff(np.concatenate([[0.0], exps]))) > 1e-9))
    assert len(RP.derive_trades(tr["ts"], tr["prices"], tr["exposures"])) == executed


# ------------------------- return baselining ------------------------------ #

def test_rebase_pct_matches_manual():
    days = _synth_days(4)
    tr = RP.traced_episode(RamRobot(), days, False, 0.001)
    prices, nav = tr["prices"], tr["nav"]
    start, end = 200, 400
    base = start - 1
    ram = RP.rebase_pct(prices[start:end], prices[base])
    port = RP.rebase_pct(nav[start:end], nav[base])
    assert ram[0] == pytest.approx((prices[start] / prices[base] - 1) * 100)
    assert ram[-1] == pytest.approx((prices[end - 1] / prices[base] - 1) * 100)
    assert port[-1] == pytest.approx((nav[end - 1] / nav[base] - 1) * 100)


def test_rebase_handles_zero_base():
    assert RP.rebase_pct([1, 2, 3], 0) == [0.0, 0.0, 0.0]


# ------------------------- resample --------------------------------------- #

def test_resample_1m_is_identity():
    bars = _flat_bars(_synth_days(2))
    bk = RS.resample_bars(bars, "1m")
    assert len(bk) == len(bars)
    assert bk[5]["close"] == pytest.approx(bars[5].close)
    assert bk[5]["i_start"] == 5 and bk[5]["i_end"] == 6


def test_resample_ohlcv_correct():
    bars = _flat_bars(_synth_days(2, bars_per_day=60))
    bk = RS.resample_bars(bars, "10m")
    for b in bk:
        seg = bars[b["i_start"]:b["i_end"]]
        assert b["open"] == pytest.approx(seg[0].open)
        assert b["close"] == pytest.approx(seg[-1].close)
        assert b["high"] == pytest.approx(max(x.high for x in seg))
        assert b["low"] == pytest.approx(min(x.low for x in seg))
        assert b["volume"] == pytest.approx(sum(x.volume for x in seg))


def test_resample_no_cross_day_buckets():
    """Intraday buckets never span two dates."""
    days = _synth_days(3, bars_per_day=95)
    bars = _flat_bars(days)
    for tf in ("10m", "30m", "1h", "2h"):
        for b in RS.resample_bars(bars, tf):
            seg = bars[b["i_start"]:b["i_end"]]
            assert len({x.date for x in seg}) == 1, f"{tf} bucket crossed a day"


def test_resample_day_matches_group_by_day():
    days = _synth_days(4)
    bars = _flat_bars(days)
    bk = RS.resample_bars(bars, "day")
    assert len(bk) == len(days)
    for b, day in zip(bk, days):
        assert b["close"] == pytest.approx(day[-1].close)
        assert b["high"] == pytest.approx(max(x.high for x in day))


def test_rollup_mean_vs_last():
    bars = _flat_bars(_synth_days(3))
    tr = RP.traced_episode(RamRobot(), bars_days := group_by_day(bars), False, 0.001)
    bk = RS.resample_bars(bars, "day")
    last = RS.rollup_last(tr["exposures"], bk)
    mean = RS.rollup_mean(tr["exposures"], bk)
    # end-of-day flatten => last exposure is ~0 every day; the mean is not
    assert all(abs(x) < 1e-9 for x in last)
    assert any(x > 0.05 for x in mean)
    # at 1m each bucket is one bar => mean reproduces the exact series
    bk1 = RS.resample_bars(bars, "1m")
    assert RS.rollup_mean(tr["exposures"], bk1) == pytest.approx(tr["exposures"])


# ------------------------- determinism ------------------------------------ #

def test_replay_determinism():
    days = _synth_days(4)
    a = RP.traced_episode(RamRobot(), days, False, 0.001)
    b = RP.traced_episode(RamRobot(), days, False, 0.001)
    assert np.array_equal(np.array(a["nav"]), np.array(b["nav"]))
    assert a["belief"][-1] == b["belief"][-1]


# ------------------------- belief_snapshot -------------------------------- #

def test_belief_snapshot_matches_features():
    """The raw snapshot must agree with features() where they overlap."""
    m = OnlineMarketMap()
    for b in _synth_days(1, 40)[0]:
        m.observe(b)
    m.set_position(0.6, b.close)
    snap = m.belief_snapshot()
    f = m.features()
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert snap["pos_in_range"] == pytest.approx(f[fi["pos_in_session_range"]])
    assert snap["exposure"] == pytest.approx(f[fi["exposure"]])
    # VWAP consistent with the dist_from_vwap feature sign
    dist = (snap["price"] - snap["vwap"]) / snap["vwap"]
    assert np.sign(dist) == np.sign(f[fi["dist_from_vwap"]]) or abs(dist) < 1e-9


def test_belief_snapshot_empty():
    snap = OnlineMarketMap().belief_snapshot()
    assert snap["price"] is None and snap["entry_price"] is None
    assert snap["exposure"] == 0.0
