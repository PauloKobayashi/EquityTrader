"""Human-readable 'why' behind a decision, per robot family.

Given the robot and the exact observation vector at a bar, reconstruct the raw
policy target (via the robot's own ``_policy_target`` — no re-implementation) and
explain, in one line, what the rule wanted and whether the deadzone let it trade.
Example (band_reverter): "price high in day range (pos=0.90) -> rule wants flat;
|Δexp| 0.42 > threshold 0.22 -> SELL".
"""
from __future__ import annotations

import numpy as np

from market_map import FEATURE_NAMES

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _action(exp_before: float, raw: float, threshold: float) -> str:
    d = raw - exp_before
    if abs(d) <= threshold:
        return "HOLD"
    return "BUY" if d > 0 else "SELL"


def explain(robot, f, exp_before: float) -> dict:
    """Return {text, raw_target, action} for a decision at observation ``f``."""
    fv = np.asarray(f, dtype=np.float64)
    raw = float(robot._policy_target(fv))          # engine's own policy math
    thr = float(robot.trade_threshold)
    action = _action(exp_before, raw, thr)
    d = raw - exp_before
    cmp = ">" if abs(d) > thr else "<="
    tail = f"|Δexp| {abs(d):.2f} {cmp} threshold {thr:.2f} -> {action}"
    fam = robot.family

    if fam == "band_reverter":
        pos = float(fv[_IDX["pos_in_session_range"]])
        where = "high" if pos > 0.6 else ("low" if pos < 0.4 else "mid")
        wants = "flat" if raw < exp_before else "long"
        head = f"price {where} in day range (pos={pos:.2f}) -> rule wants {wants}"
    elif fam == "vwap_reverter":
        dv = float(fv[_IDX["dist_from_vwap"]])
        side = "above" if dv > 0 else "below"
        head = f"price {side} VWAP (dist={dv:+.2f}) -> mean-revert toward VWAP"
    elif fam == "momentum":
        m = float(fv[_IDX["momentum"]])
        trend = "up" if m > 0 else "down"
        head = f"{trend} momentum (mom={m:+.2f}) -> trend-follow"
    elif fam in ("nn_reverter", "morpheus_trader"):
        pos = float(fv[_IDX["pos_in_session_range"]])
        dv = float(fv[_IDX["dist_from_vwap"]])
        head = (f"MLP policy on belief (pos={pos:.2f}, dvwap={dv:+.2f}) "
                f"-> target {raw:.2f}")
    else:
        head = f"policy target {raw:.2f}"

    gap_bias = float(robot.p.get("gap_fade_bias", 0.0))
    if abs(gap_bias) > 1e-9:
        g = float(fv[_IDX["overnight_gap"]])
        head += f"; gap-fade {gap_bias:.2f}*gap({g:+.2f})"

    return {"text": f"{head}; {tail}", "raw_target": raw, "action": action}
