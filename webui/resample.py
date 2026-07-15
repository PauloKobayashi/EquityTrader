"""Resample native 1-min bars into coarser display buckets (ET-session aware).

The robot's *signal* stays at its native 1-min tick — we never re-run it at a
coarser cadence. For a coarser display we bucket the bars for the price chart and
then roll the per-1-min series (exposure, nav) up to those same buckets (last value
in the bucket). Trades at a coarse timeframe fall out of diffing the rolled-up
exposure (see replay.derive_trades) — each bucket's net move becomes one marker.

Each bucket carries ``i_start`` / ``i_end`` (its native-bar index span) so callers
can align any per-1-min array to the buckets without recomputing anything.

Pure stdlib — no pandas, no zion_ge.
"""
from __future__ import annotations

from datetime import date

# display timeframes, coarsest-first ordering handled by the UI
TIMEFRAMES = ["1m", "10m", "30m", "1h", "2h", "day", "week", "month"]

# sub-day frames: bucket width in wall-clock minutes (aligned to the day, never crossing it)
_SUBDAY = {"10m": 10, "30m": 30, "1h": 60, "2h": 120}


def _isoweek(d: str) -> tuple[int, int]:
    y, w, _ = date.fromisoformat(d).isocalendar()
    return (y, w)


def _keyfn(tf: str):
    if tf in _SUBDAY:
        step = _SUBDAY[tf]
        return lambda b: (b.date, b.minute_of_day // step)
    if tf == "day":
        return lambda b: b.date
    if tf == "week":
        return lambda b: _isoweek(b.date)
    if tf == "month":
        return lambda b: b.ts[:7]
    raise ValueError(f"unknown timeframe: {tf}")


def _bucket(group, s: int, e: int) -> dict:
    return {
        "ts": group[0].ts,
        "open": float(group[0].open),
        "high": float(max(x.high for x in group)),
        "low": float(min(x.low for x in group)),
        "close": float(group[-1].close),
        "volume": float(sum(x.volume for x in group)),
        "i_start": s,
        "i_end": e,
    }


def resample_bars(bars, tf: str) -> list[dict]:
    """Aggregate 1-min ``bars`` into ``tf`` buckets with native index spans.

    Sub-day frames bucket on wall-clock minute boundaries within a day (no
    cross-day buckets); day/week/month group whole days.
    """
    if tf == "1m":
        return [_bucket([b], i, i + 1) for i, b in enumerate(bars)]
    keyfn = _keyfn(tf)
    out: list[dict] = []
    cur, key, s = None, None, 0
    for i, b in enumerate(bars):
        k = keyfn(b)
        if k != key:
            if cur is not None:
                out.append(_bucket(cur, s, i))
            cur, key, s = [b], k, i
        else:
            cur.append(b)
    if cur:
        out.append(_bucket(cur, s, len(bars)))
    return out


def rollup_last(values, buckets: list[dict]) -> list:
    """Roll a per-1-min series up to buckets by taking the last value in each bucket.

    Used for nav — the state as of the bucket's close.
    """
    return [values[b["i_end"] - 1] for b in buckets]


def rollup_mean(values, buckets: list[dict]) -> list:
    """Roll a per-1-min series up to buckets by averaging over each bucket.

    Used for the exposure signal band: 'last in bucket' would read 0 at day/week/
    month frames (the robot flattens into every overnight gap), hiding all intraday
    commitment. The mean shows the average exposure held over the period; at 1m each
    bucket is one bar so it reproduces the exact per-tick signal (and exact trades).
    """
    out = []
    for b in buckets:
        seg = values[b["i_start"]:b["i_end"]]
        out.append(sum(seg) / len(seg) if seg else 0.0)
    return out


def rollup_at(rows: list, buckets: list[dict]) -> list:
    """Pick the per-1-min row (e.g. a features vector or belief dict) at each
    bucket's last native bar — the belief as of the bucket's close."""
    return [rows[b["i_end"] - 1] for b in buckets]
