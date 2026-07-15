"""OnlineMarketMap — the robot's self-built internal "map" of the intraday market.

Transplanted from the maze-runner ``OnlineMapper`` (benchmarks/maze_runner/mapper.py):
the robot maintains a *belief state* it builds incrementally from the bar stream and
reads back as its observation. The strict rule it inherits — the **information
contract** — is that ``features()`` may look only at bars already observed (past +
present), never the future. Training rewards/labels elsewhere may peek ahead; the
observation the policy sees may not. This is the classic look-ahead-leakage guard.

Pure ``numpy`` + stdlib (NO zion_ge imports) so the exported standalone robot can
carry this file verbatim with no engine dependency.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Observation vector length produced by features(). Kept small on purpose:
# the maze sweep found small-perception policies generalize best.
FEATURE_NAMES = [
    "dist_from_vwap",
    "pos_in_session_range",
    "momentum",
    "vol_regime",
    "visit_density",
    "minutes_to_rth_close",
    "sess_premarket",
    "sess_rth",
    "sess_afterhours",
    "low_liquidity",
    "overnight_gap",
    "exposure",
    "unrealized_pnl",
]
N_FEATURES = len(FEATURE_NAMES)

_RTH_OPEN_MIN = 9 * 60 + 30   # 09:30 ET
_RTH_CLOSE_MIN = 16 * 60      # 16:00 ET


@dataclass
class Bar:
    """One minute bar. ``ts`` is a 'YYYY-MM-DD HH:MM:SS' ET string."""

    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    session: str  # 'premarket' | 'rth' | 'afterhours'

    @property
    def date(self) -> str:
        return self.ts[:10]

    @property
    def minute_of_day(self) -> int:
        hh = int(self.ts[11:13])
        mm = int(self.ts[14:16])
        return hh * 60 + mm


def _tanh(x: float) -> float:
    return float(np.tanh(x))


class OnlineMarketMap:
    """Incrementally-built belief state, updated one bar at a time.

    Parameters mirror the maze mapper's perception genes:
      obs_lookback : bars back used for the momentum feature
      memory_cap   : max bars retained (older forgotten) — a regularizer / anti-clone lever
      ladder_bins  : resolution of the price ladder used for visit_density
      vol_window   : window for realized-vol regime estimate
    """

    def __init__(
        self,
        obs_lookback: int = 15,
        memory_cap: int = 120,
        ladder_bins: int = 20,
        vol_window: int = 30,
        low_liquidity_volume: float = 100.0,
    ) -> None:
        self.obs_lookback = max(2, int(obs_lookback))
        self.memory_cap = max(self.obs_lookback + 2, int(memory_cap))
        self.ladder_bins = max(4, int(ladder_bins))
        self.vol_window = max(3, int(vol_window))
        self.low_liquidity_volume = float(low_liquidity_volume)
        self.reset()

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self._prices: deque[float] = deque(maxlen=self.memory_cap)
        self._rets: deque[float] = deque(maxlen=self.memory_cap)
        self._cur_date: str | None = None
        self._prev_day_last_price: float | None = None
        self._day_vwap_pv = 0.0
        self._day_vwap_v = 0.0
        self._day_high = -np.inf
        self._day_low = np.inf
        self._overnight_gap = 0.0
        self._last_price: float | None = None
        self._last_bar: Bar | None = None
        self._exposure = 0.0
        self._entry_price: float | None = None

    # -- position bookkeeping (robot tells the map what it holds) -----------

    def set_position(self, exposure: float, price: float) -> None:
        """Record the robot's current exposure so it becomes part of the belief."""
        prev = self._exposure
        if exposure > 1e-9 and prev <= 1e-9:
            self._entry_price = price            # opening from flat
        elif exposure <= 1e-9:
            self._entry_price = None             # flat
        elif abs(exposure - prev) > 1e-9 and self._entry_price is not None:
            # scaling an existing position: blend the entry basis
            self._entry_price = (
                self._entry_price * min(prev, exposure)
                + price * max(0.0, exposure - prev)
            ) / max(exposure, 1e-9)
        self._exposure = float(exposure)

    # -- perception ---------------------------------------------------------

    def observe(self, bar: Bar) -> None:
        """Fold one bar into the belief. Past/present only — never the future."""
        price = float(bar.close)

        if bar.date != self._cur_date:
            # New session-day: the overnight gap (Korea-digested move) surfaces here.
            if self._prev_day_last_price:
                self._overnight_gap = (price - self._prev_day_last_price) / self._prev_day_last_price
            else:
                self._overnight_gap = 0.0
            self._cur_date = bar.date
            self._day_vwap_pv = 0.0
            self._day_vwap_v = 0.0
            self._day_high = -np.inf
            self._day_low = np.inf

        if self._last_price is not None and self._last_price > 0:
            self._rets.append(price / self._last_price - 1.0)

        self._prices.append(price)
        vol = max(float(bar.volume), 0.0)
        # Use a floor weight so zero-volume extended bars still nudge VWAP.
        w = vol if vol > 0 else 1.0
        self._day_vwap_pv += price * w
        self._day_vwap_v += w
        self._day_high = max(self._day_high, float(bar.high))
        self._day_low = min(self._day_low, float(bar.low))

        # Carry the day's last price so the *next* day can compute its gap.
        self._prev_day_last_price = price
        self._last_price = price
        self._last_bar = bar

    # -- observation vector -------------------------------------------------

    def features(self) -> np.ndarray:
        """Return the fixed-length observation the policy consumes."""
        bar = self._last_bar
        if bar is None:
            return np.zeros(N_FEATURES, dtype=np.float64)
        price = float(bar.close)

        vwap = self._day_vwap_pv / self._day_vwap_v if self._day_vwap_v > 0 else price
        dist_from_vwap = (price - vwap) / vwap if vwap > 0 else 0.0

        rng = self._day_high - self._day_low
        pos_in_range = (price - self._day_low) / rng if rng > 1e-9 else 0.5
        pos_in_range = float(np.clip(pos_in_range, 0.0, 1.0))

        if len(self._prices) > self.obs_lookback:
            past = self._prices[-self.obs_lookback - 1]
            momentum = (price - past) / past if past > 0 else 0.0
        else:
            momentum = 0.0

        if len(self._rets) >= 3:
            n = min(self.vol_window, len(self._rets))
            vol_regime = float(np.std(list(self._rets)[-n:]))
        else:
            vol_regime = 0.0

        visit_density = self._visit_density(price)

        mod = bar.minute_of_day
        mins_to_close = (_RTH_CLOSE_MIN - mod) / 390.0
        minutes_to_rth_close = float(np.clip(mins_to_close, 0.0, 1.0))

        sess = bar.session
        low_liq = 1.0 if (bar.volume < self.low_liquidity_volume or sess != "rth") else 0.0

        if self._exposure > 1e-9 and self._entry_price:
            unrealized = price / self._entry_price - 1.0
        else:
            unrealized = 0.0

        return np.array([
            _tanh(dist_from_vwap * 50.0),
            pos_in_range,
            _tanh(momentum * 50.0),
            _tanh(vol_regime * 200.0),
            visit_density,
            minutes_to_rth_close,
            1.0 if sess == "premarket" else 0.0,
            1.0 if sess == "rth" else 0.0,
            1.0 if sess == "afterhours" else 0.0,
            low_liq,
            _tanh(self._overnight_gap * 20.0),
            float(np.clip(self._exposure, 0.0, 1.0)),
            _tanh(unrealized * 20.0),
        ], dtype=np.float64)

    def _visit_density(self, price: float) -> float:
        """Fraction of *retained* prices sitting in the current price's ladder bin.

        Because it reads only the memory-capped deque, it forgets naturally — the
        maze mapper's bounded-memory behaviour, which is the anti-clone lever."""
        if len(self._prices) < 2:
            return 0.0
        lo, hi = self._day_low, self._day_high
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            return 0.0
        edges = np.linspace(lo, hi, self.ladder_bins + 1)
        cur_bin = int(np.clip(np.digitize(price, edges) - 1, 0, self.ladder_bins - 1))
        arr = np.asarray(self._prices, dtype=np.float64)
        bins = np.clip(np.digitize(arr, edges) - 1, 0, self.ladder_bins - 1)
        return float(np.count_nonzero(bins == cur_bin) / len(arr))

    # -- convenience --------------------------------------------------------

    @property
    def dist_from_vwap(self) -> float:
        f = self.features()
        return float(f[0])

    @property
    def vol_regime_raw(self) -> float:
        if len(self._rets) >= 3:
            n = min(self.vol_window, len(self._rets))
            return float(np.std(list(self._rets)[-n:]))
        return 0.0
