"""Data loading for RAM 1-min bars. Pure stdlib (csv) — no pandas, no zion_ge — so
the exported standalone robot can reuse it to read a CSV and trade.

CSV schema: datetime,open,high,low,close,volume,session
(session in {premarket, rth, afterhours}; ET wall-clock timestamps).
"""
from __future__ import annotations

import csv
from collections import OrderedDict

from market_map import Bar


def load_bars(path: str) -> list[Bar]:
    bars: list[Bar] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                bars.append(Bar(
                    ts=row["datetime"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]) if row.get("volume") not in (None, "") else 0.0,
                    session=row.get("session", "rth") or "rth",
                ))
            except (ValueError, KeyError):
                continue
    bars.sort(key=lambda b: b.ts)
    return bars


def group_by_day(bars: list[Bar]) -> list[list[Bar]]:
    """Return bars grouped into per-day lists, in chronological order."""
    days: "OrderedDict[str, list[Bar]]" = OrderedDict()
    for b in bars:
        days.setdefault(b.date, []).append(b)
    return list(days.values())


def walkforward_split(days: list[list[Bar]], train_frac: float = 0.6):
    """Split whole days into disjoint train / test blocks (walk-forward)."""
    n = len(days)
    if n < 2:
        return days, days  # degenerate: reuse (smoke only)
    n_train = max(1, int(round(n * train_frac)))
    n_train = min(n_train, n - 1)
    return days[:n_train], days[n_train:]
