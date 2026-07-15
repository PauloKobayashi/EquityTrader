"""Backtest engine for the long/flat continuous-exposure RAM robot.

NAV / ulcer math is transplanted from the qqq ancestor (private/qqq_gt04/trader.py):
ulcer uses the SUM-of-squared-drawdowns convention. Exposure is continuous in [0,1]
(long/flat). A held position is marked across the overnight gap (the return between a
day's last bar and the next day's first bar) — that gap P&L is real and is where the
Korea-driven signal pays off; if the robot's `hold_through_gap` is off it is flattened
to 0 at each day's final bar so it carries nothing overnight.

Pure numpy + stdlib (no zion_ge).
"""
from __future__ import annotations

import numpy as np

from market_map import Bar


def ulcer(nav: np.ndarray) -> float:
    """sqrt(sum(drawdown^2)) — qqq trader.py convention (SUM, not mean)."""
    nav = np.asarray(nav, dtype=np.float64)
    if nav.size == 0:
        return 999.9
    cummax = np.maximum.accumulate(nav)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(cummax > 0, (nav - cummax) / cummax, 0.0)
    return float(np.sqrt(np.sum(dd ** 2)))


def nav_from_exposures(prices: np.ndarray, exposures: np.ndarray,
                       cost_bps: float, start: float = 100.0) -> np.ndarray:
    """Continuous-exposure NAV.

    exposures[t] is decided from info up to bar t and earns the return from t->t+1.
    Turnover cost = |Δexposure| * cost_bps charged when exposure changes.
    """
    prices = np.asarray(prices, dtype=np.float64)
    e = np.asarray(exposures, dtype=np.float64)
    n = len(prices)
    nav = np.empty(n, dtype=np.float64)
    nav[0] = start
    prev_e = 0.0
    for t in range(n - 1):
        # turnover cost applied when the position at t differs from t-1
        turn = abs(e[t] - prev_e)
        val = nav[t] * (1.0 - turn * cost_bps)
        r = prices[t + 1] / prices[t] - 1.0 if prices[t] > 0 else 0.0
        nav[t + 1] = val * (1.0 + e[t] * r)
        prev_e = e[t]
    return nav


def run_episode(robot, days: list[list[Bar]], hold_through_gap: bool,
                cost_bps: float):
    """Run the robot across a sequence of days, chaining NAV through overnight gaps.

    Returns (nav_array, per_day_return_pct, full_prices, full_exposures).
    per_day_return is the NAV %-change within each day's own window (for the
    walk-forward objective); the chained NAV includes gap returns when held.
    """
    robot.reset()
    all_prices: list[float] = []
    all_exp: list[float] = []
    day_bounds: list[tuple[int, int]] = []
    idx = 0
    for day in days:
        start = idx
        for i, bar in enumerate(day):
            exp = robot.on_bar(bar)
            if not hold_through_gap and i == len(day) - 1:
                exp = 0.0                      # flat into the overnight gap
                robot.exposure = 0.0
                robot.map.set_position(0.0, float(bar.close))
            all_prices.append(float(bar.close))
            all_exp.append(exp)
            idx += 1
        day_bounds.append((start, idx))

    prices = np.array(all_prices, dtype=np.float64)
    exps = np.array(all_exp, dtype=np.float64)
    if len(prices) < 2:
        return np.array([100.0]), np.array([0.0]), prices, exps

    nav = nav_from_exposures(prices, exps, cost_bps)
    per_day_ret = []
    for (s, e) in day_bounds:
        seg = nav[s:e]
        if len(seg) >= 2 and seg[0] > 0:
            per_day_ret.append((seg[-1] / seg[0] - 1.0) * 100.0)
    return nav, np.array(per_day_ret, dtype=np.float64), prices, exps


def summarize(nav: np.ndarray, per_day_ret: np.ndarray) -> dict:
    return {
        "final_return_pct": float(nav[-1] - 100.0) if len(nav) else 0.0,
        "median_day_return_pct": float(np.median(per_day_ret)) if len(per_day_ret) else 0.0,
        "mean_day_return_pct": float(np.mean(per_day_ret)) if len(per_day_ret) else 0.0,
        "ulcer": ulcer(nav),
        "n_days": int(len(per_day_ret)),
    }
