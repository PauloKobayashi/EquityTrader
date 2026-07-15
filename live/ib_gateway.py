"""Live RAM trading against the Mac IB Gateway using the official ``ibapi``
(EClient/EWrapper) — same pattern as private/qqq_gt0*/interactiveBrokers.ipynb.

PAPER-FIRST: defaults to the IB Gateway paper port 4002. Live (4001) requires an
explicit opt-in flag in run_live.py. Long/flat only (never shorts). Extended hours
supported: LMT + outsideRth outside RTH, MKT during RTH.

Flow:
  1. reqHistoricalData(useRTH=0) warms the OnlineMarketMap with today's + recent bars.
  2. reqRealTimeBars (5-sec) are aggregated to 1-min and fed to robot.on_bar().
  3. On exposure change beyond the robot's threshold, placeOrder sizes to
     target_shares = round(exposure * capital / price); BUY to add, SELL to trim.
  4. reqPositions reconciles the broker position with the intended exposure.

ibapi is NOT bundled: `pip install ibapi` (or install from the TWS API download).
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
import time
from zoneinfo import ZoneInfo

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from market_map import Bar
from robot import RamRobot

ET = ZoneInfo("America/New_York")

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    _IBAPI_OK = True
except Exception:  # noqa: BLE001
    # Distinct stub bases so `class IBApp(EWrapper, EClient)` still defines cleanly
    # (two `object` bases would be a duplicate-base TypeError). Construction raises.
    class _StubWrapper:  # noqa: D401
        pass

    class _StubClient:
        def __init__(self, *a, **k):
            pass

    EWrapper = _StubWrapper   # type: ignore
    EClient = _StubClient     # type: ignore
    Contract = None           # type: ignore
    Order = None              # type: ignore
    _IBAPI_OK = False

PAPER_PORT = 4002
LIVE_PORT = 4001


def ram_contract() -> "Contract":
    c = Contract()
    c.symbol = "RAM"
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = "SMART"
    c.primaryExchange = "BATS"   # RAM primary listing (IBKR contract_id 895507567)
    return c


def _session_of(dt_et: _dt.datetime) -> str | None:
    t = dt_et.hour * 60 + dt_et.minute
    if 4 * 60 <= t < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= t < 16 * 60:
        return "rth"
    if 16 * 60 <= t < 20 * 60:
        return "afterhours"
    return None


class IBApp(EWrapper, EClient):
    def __init__(self, robot: RamRobot, capital: float = 100_000.0,
                 port: int = PAPER_PORT, client_id: int = 17, dry_run: bool = False):
        if not _IBAPI_OK:
            raise RuntimeError("ibapi not installed — run `pip install ibapi`")
        EClient.__init__(self, self)
        self.robot = robot
        self.capital = float(capital)
        self.port = int(port)
        self.client_id = int(client_id)
        self.dry_run = bool(dry_run)          # feed bars + print orders, do not place
        self.contract = ram_contract()
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._cur_min: str | None = None      # 'YYYY-MM-DD HH:MM'
        self._acc = None                       # accumulating 1-min bar
        self.broker_position = 0.0             # shares held, from reqPositions
        self._warm_done = False

    # -- id management ------------------------------------------------------

    def nextValidId(self, orderId: int):      # EWrapper callback
        with self._id_lock:
            self._next_id = max(self._next_id, orderId)
        # kick off warm-up + streaming once we have an id
        if not self._warm_done:
            self._start()

    def _new_id(self) -> int:
        with self._id_lock:
            oid = self._next_id
            self._next_id += 1
            return oid

    # -- startup ------------------------------------------------------------

    def _start(self):
        self._warm_done = True
        # Warm-up: recent extended-hours 1-min bars to seed the belief map.
        self.reqHistoricalData(
            self._new_id(), self.contract, "", "2 D", "1 min",
            "TRADES", 0, 1, False, [])
        self.reqPositions()

    # -- historical warm-up -------------------------------------------------

    def historicalData(self, reqId, bar):     # EWrapper callback
        # bar.date like '20260715  09:31:00' (ET). Feed straight into the map.
        ts = _parse_ib_date(bar.date)
        if ts is None:
            return
        sess = _session_of(ts) or "rth"
        self.robot.on_bar(Bar(ts.strftime("%Y-%m-%d %H:%M:%S"),
                              bar.open, bar.high, bar.low, bar.close,
                              float(bar.volume), sess))

    def historicalDataEnd(self, reqId, start, end):
        print(f"[warmup] seeded belief map; exposure={self.robot.exposure:.2f}")
        # Begin live streaming (5-sec realtime bars, extended hours).
        self.reqRealTimeBars(self._new_id(), self.contract, 5, "TRADES", False, [])

    # -- realtime streaming (5-sec -> 1-min aggregation) --------------------

    def realtimeBar(self, reqId, btime, open_, high, low, close, volume,
                    wap, count):              # EWrapper callback
        dt_et = _dt.datetime.fromtimestamp(btime, tz=ET)
        minute = dt_et.strftime("%Y-%m-%d %H:%M")
        if self._cur_min is None:
            self._cur_min, self._acc = minute, _MinAcc(dt_et, open_, high, low, close, volume)
        elif minute != self._cur_min:
            self._emit_minute()               # previous minute finished
            self._cur_min, self._acc = minute, _MinAcc(dt_et, open_, high, low, close, volume)
        else:
            self._acc.update(high, low, close, volume)

    def _emit_minute(self):
        if self._acc is None:
            return
        dt_et = self._acc.start
        sess = _session_of(dt_et)
        if sess is None:
            self._acc = None
            return
        bar = Bar(dt_et.strftime("%Y-%m-%d %H:%M:00"), self._acc.open, self._acc.high,
                  self._acc.low, self._acc.close, self._acc.volume, sess)
        prev_exp = self.robot.exposure
        target = self.robot.on_bar(bar)
        if abs(target - prev_exp) > 1e-9:
            self._rebalance(target, bar.close, sess)
        self._acc = None

    # -- order routing (long/flat only) -------------------------------------

    def _rebalance(self, target_exposure: float, price: float, session: str):
        target_shares = max(0, round(target_exposure * self.capital / max(price, 1e-6)))
        delta = target_shares - int(round(self.broker_position))
        if delta == 0:
            return
        action = "BUY" if delta > 0 else "SELL"
        qty = abs(delta)
        in_rth = session == "rth"
        order = _mk_order(action, qty, price, in_rth)
        oid = self._new_id()
        tag = "MKT" if in_rth else f"LMT@{order.lmtPrice}"
        print(f"[order] {action} {qty} RAM {tag} ({session}) "
              f"target_exp={target_exposure:.2f} target_sh={target_shares}")
        if not self.dry_run:
            self.placeOrder(oid, self.contract, order)
            self.broker_position = target_shares   # optimistic; corrected by position()

    # -- position reconciliation --------------------------------------------

    def position(self, account, contract, pos, avgCost):  # EWrapper callback
        if getattr(contract, "symbol", None) == "RAM":
            self.broker_position = float(pos)

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # 2104/2106/2158 are benign "market data farm connected" notices.
        if errorCode not in (2104, 2106, 2158, 2100):
            print(f"[ib-error] id={reqId} code={errorCode} {errorString}")


class _MinAcc:
    def __init__(self, start, o, h, l, c, v):
        self.start, self.open, self.high, self.low, self.close, self.volume = start, o, h, l, c, v

    def update(self, h, l, c, v):
        self.high = max(self.high, h)
        self.low = min(self.low, l)
        self.close = c
        self.volume += v


def _mk_order(action: str, qty: int, price: float, in_rth: bool) -> "Order":
    o = Order()
    o.action = action
    o.totalQuantity = qty
    o.tif = "DAY"
    o.eTradeOnly = False
    o.firmQuoteOnly = False
    if in_rth:
        o.orderType = "MKT"
        o.outsideRth = False
    else:
        # Extended hours: marketable limit, allow outside RTH.
        o.orderType = "LMT"
        pad = 0.002 * price               # 20 bps marketable pad
        o.lmtPrice = round(price + pad if action == "BUY" else price - pad, 2)
        o.outsideRth = True
    return o


def _parse_ib_date(s: str):
    s = s.strip()
    try:
        if s.isdigit():                    # epoch seconds
            return _dt.datetime.fromtimestamp(int(s), tz=ET)
        fmt = "%Y%m%d %H:%M:%S" if ":" in s else "%Y%m%d"
        return _dt.datetime.strptime(" ".join(s.split()), fmt).replace(tzinfo=ET)
    except (ValueError, TypeError):
        return None


def run(robot: RamRobot, capital: float, port: int, client_id: int,
        dry_run: bool, host: str = "127.0.0.1"):
    app = IBApp(robot, capital=capital, port=port, client_id=client_id, dry_run=dry_run)
    print(f"[connect] {host}:{port} clientId={client_id} "
          f"{'PAPER' if port == PAPER_PORT else 'LIVE'} dry_run={dry_run}")
    app.connect(host, port, client_id)
    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    try:
        while t.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[shutdown] disconnecting")
        app.disconnect()
