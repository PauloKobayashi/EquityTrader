"""Engine-parity replay with per-bar belief capture.

``run_episode`` (backtest.py) is the source of truth for NAV / exposures, but it
throws away the belief state after each bar. ``traced_episode`` re-runs the SAME
loop bar-for-bar (identical ``on_bar`` order, identical end-of-day forced-flat when
``hold_through_gap`` is off) and additionally records, per bar, the observation
vector (``features``) and the raw belief (``belief_snapshot``) the robot used.

The parity test (tests/test_webui.py) asserts ``traced_episode``'s nav/exposures are
*identical* to ``run_episode``, so this tracer can never silently drift from the
engine. NAV is recomputed here via the engine's own ``nav_from_exposures``.
"""
from __future__ import annotations

import numpy as np

from backtest import nav_from_exposures
from market_map import FEATURE_NAMES


def traced_episode(robot, days, hold_through_gap: bool, cost_bps: float) -> dict:
    """Drive ``robot`` across ``days``, capturing the belief behind every decision.

    Mirrors ``backtest.run_episode`` exactly, then returns native per-1-min-bar
    arrays plus the authoritative ``nav`` (from ``nav_from_exposures``).
    """
    robot.reset()
    ts: list[str] = []
    prices: list[float] = []
    exposures: list[float] = []
    feats: list[list[float]] = []
    belief: list[dict] = []
    day_bounds: list[tuple[int, int]] = []
    idx = 0
    for day in days:
        start = idx
        for i, bar in enumerate(day):
            exp = robot.on_bar(bar)
            if not hold_through_gap and i == len(day) - 1:
                exp = 0.0                       # flat into the overnight gap
                robot.exposure = 0.0
                robot.map.set_position(0.0, float(bar.close))
            # capture AFTER the gap-flatten so belief matches the recorded exposure
            ts.append(bar.ts)
            prices.append(float(bar.close))
            exposures.append(exp)
            feats.append([float(x) for x in robot.map.features()])
            belief.append(robot.map.belief_snapshot())
            idx += 1
        day_bounds.append((start, idx))

    p = np.array(prices, dtype=np.float64)
    e = np.array(exposures, dtype=np.float64)
    if len(p) < 2:
        nav = np.array([100.0] * max(len(p), 1), dtype=np.float64)
    else:
        nav = nav_from_exposures(p, e, cost_bps)

    per_day_ret = []
    for (s, en) in day_bounds:
        seg = nav[s:en]
        if len(seg) >= 2 and seg[0] > 0:
            per_day_ret.append((seg[-1] / seg[0] - 1.0) * 100.0)

    return {
        "ts": ts,
        "prices": prices,
        "exposures": exposures,
        "nav": nav.tolist(),
        "per_day_ret": per_day_ret,
        "features": feats,
        "feature_names": list(FEATURE_NAMES),
        "belief": belief,
        "day_bounds": day_bounds,
    }


def derive_trades(ts, prices, exposures, threshold: float = 0.0) -> list[dict]:
    """A trade is any bar where exposure changed (``|Δexposure| > threshold``).

    Matches how ``run_episode`` / live ``_rebalance`` treat a trade: the deadzone in
    ``on_bar`` already collapses sub-threshold moves, so at the native tick
    ``threshold=0`` recovers exactly the executed trades. On rolled-up (coarser)
    exposure series this collapses each bucket's net move into one marker.
    """
    trades = []
    prev = 0.0
    for i in range(len(exposures)):
        d = exposures[i] - prev
        if abs(d) > threshold and abs(d) > 1e-9:
            trades.append({
                "i": i,
                "ts": ts[i],
                "price": float(prices[i]),
                "side": "buy" if d > 0 else "sell",
                "dExposure": float(d),
                "exposure_after": float(exposures[i]),
            })
        prev = exposures[i]
    return trades


def rebase_pct(values, base) -> list[float]:
    """Rebase a series to a baseline value, in percent: ``(v/base - 1) * 100``.

    Presentation only (division on engine-computed values) — this does not
    re-implement NAV. ``base`` is typically the last value before the visible window.
    """
    if base is None or base == 0:
        return [0.0 for _ in values]
    return [(float(v) / float(base) - 1.0) * 100.0 for v in values]
