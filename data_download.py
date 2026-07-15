"""Download RAM full 1-min history (pre-market + RTH + after-hours) from the Mac IB
Gateway via ``ibapi`` reqHistoricalData, paging backward with endDateTime.

WHY this and not the IBKR MCP: the MCP get_price_history only ever returns the most
recent <=1000 trailing bars (no date paging), so it cannot assemble RAM's full multi-
week minute history. reqHistoricalData supports endDateTime + duration, so it pages
back day-by-day (useRTH=0 for extended hours). Writes data/RAM_1min.csv with the
schema the pipeline expects: datetime,open,high,low,close,volume,session.

Usage (Gateway must be running; paper 4002 or live 4001 both serve historical data):
  pip install ibapi
  python data_download.py --days 25 --port 4002

The IBKR MCP remains handy for a quick recent-slice sanity check (see README).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import threading
import time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    _IBAPI_OK = True
except Exception:  # noqa: BLE001
    class _StubWrapper:
        pass

    class _StubClient:
        def __init__(self, *a, **k):
            pass

    EWrapper = _StubWrapper   # type: ignore
    EClient = _StubClient     # type: ignore
    Contract = None           # type: ignore
    _IBAPI_OK = False


def _session_of(dt_et: _dt.datetime) -> str | None:
    t = dt_et.hour * 60 + dt_et.minute
    if 4 * 60 <= t < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= t < 16 * 60:
        return "rth"
    if 16 * 60 <= t < 20 * 60:
        return "afterhours"
    return None


def ram_contract() -> "Contract":
    c = Contract()
    c.symbol = "RAM"
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = "SMART"
    c.primaryExchange = "BATS"
    return c


class _Downloader(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.rows: dict[str, tuple] = {}      # datetime -> row (de-dupe)
        self._done = threading.Event()
        self._chunk_done = threading.Event()
        self._ready = threading.Event()
        self.paced = False                    # last chunk hit a pacing violation (162)
        self.no_data = False                  # last chunk returned "no data"

    def nextValidId(self, orderId):
        self._ready.set()

    def historicalData(self, reqId, bar):
        ts = _parse_ib_date(bar.date)
        if ts is None:
            return
        sess = _session_of(ts)
        if sess is None:
            return
        key = ts.strftime("%Y-%m-%d %H:%M:%S")
        self.rows[key] = (key, bar.open, bar.high, bar.low, bar.close,
                          int(bar.volume), sess)

    def historicalDataEnd(self, reqId, start, end):
        self._chunk_done.set()

    def error(self, reqId, code, msg, advancedOrderRejectJson=""):
        if code not in (2104, 2106, 2158, 2100, 2107, 2119):
            print(f"[ib-error] id={reqId} code={code} {msg}")
        if code == 162 and "pacing" in msg.lower():
            self.paced = True                 # rate limited — caller backs off + retries
            self._chunk_done.set()
        elif code in (162, 165, 200, 321, 366):  # no-data / bad-contract style
            self.no_data = True               # nothing older here (likely reached inception)
            self._chunk_done.set()


def _parse_ib_date(s: str):
    s = " ".join(s.strip().split())
    try:
        if s.isdigit():
            return _dt.datetime.fromtimestamp(int(s), tz=ET)
        fmt = "%Y%m%d %H:%M:%S" if ":" in s else "%Y%m%d"
        return _dt.datetime.strptime(s, fmt).replace(tzinfo=ET)
    except (ValueError, TypeError):
        return None


def download(days: int, port: int, host: str, client_id: int, out_path: str):
    if not _IBAPI_OK:
        raise SystemExit("ibapi not installed — run `pip install ibapi`")
    app = _Downloader()
    app.connect(host, port, client_id)
    threading.Thread(target=app.run, daemon=True).start()
    if not app._ready.wait(10):
        raise SystemExit(f"could not connect to IB Gateway at {host}:{port}")

    contract = ram_contract()
    # Page backward one day per request (an extended ~16h day < the ~1000-bar/req limit).
    #
    # IBKR historical-data pacing (respected below):
    #   * <=60 requests per rolling 10-minute window (else error 162 "pacing violation")
    #   * no identical request within 15s; <=6 identical within 2s
    # We keep a rolling window of request timestamps and sleep if we'd exceed the cap,
    # plus a small fixed gap between requests. Because endDateTime moves back each time,
    # successive requests are NOT identical (so the 15s rule doesn't bite) — UNTIL we hit
    # inception and `earliest` stops moving, at which point we STOP (avoids firing the
    # same request repeatedly, which WOULD trip the identical-request rule).
    MAX_PER_WINDOW = 55                        # safety margin under the 60/10min ceiling
    WINDOW_S = 600
    MIN_GAP_S = 2.0                            # small gap between (non-identical) requests
    req_times: list[float] = []

    def _pace():
        now = time.time()
        req_times[:] = [t for t in req_times if now - t < WINDOW_S]
        if len(req_times) >= MAX_PER_WINDOW:
            wait = WINDOW_S - (now - req_times[0]) + 1
            print(f"[pace] {len(req_times)} reqs in window — sleeping {wait:.0f}s")
            time.sleep(max(0.0, wait))
        req_times.append(time.time())

    end = ""                                    # "" == now
    reqid = 1
    prev_earliest: str | None = None
    for _ in range(days):
        app._chunk_done.clear()
        app.paced = app.no_data = False
        _pace()
        n_before = len(app.rows)
        app.reqHistoricalData(reqid, contract, end, "1 D", "1 min",
                              "TRADES", 0, 1, False, [])
        if not app._chunk_done.wait(30):
            print(f"[warn] chunk {reqid} timed out")
        gained = len(app.rows) - n_before
        earliest = min(app.rows) if app.rows else None
        print(f"[chunk {reqid}] end={end or 'now':<20} +{gained} bars  earliest={earliest}")
        reqid += 1

        if app.paced:                           # rate limited: back off and retry same end
            print("[pace] pacing violation — backing off 30s and retrying this chunk")
            time.sleep(30)
            continue

        if earliest is None or earliest == prev_earliest:
            print("[done] no older data returned (reached inception) — stopping")
            break
        prev_earliest = earliest
        # Anchor the next window one second BEFORE the earliest bar so we page strictly
        # backward. US-equity exchange tz is ET, so the plain (no-suffix) format is
        # unambiguous — older ibapi rejects/misparses a trailing 'US/Eastern' token.
        dt = _dt.datetime.strptime(earliest, "%Y-%m-%d %H:%M:%S") - _dt.timedelta(seconds=1)
        end = dt.strftime("%Y%m%d %H:%M:%S")
        time.sleep(MIN_GAP_S)
    app.disconnect()

    rows = [app.rows[k] for k in sorted(app.rows)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["datetime", "open", "high", "low", "close", "volume", "session"])
        w.writerows(rows)

    from collections import Counter
    dcount = Counter(r[0][:10] for r in rows)
    scount = Counter(r[6] for r in rows)
    print(f"wrote {len(rows)} bars -> {out_path}")
    print(f"days ({len(dcount)}): {sorted(dcount)}")
    print(f"sessions: {dict(scount)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=25)
    ap.add_argument("--port", type=int, default=4002)   # paper serves historical too
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-id", type=int, default=19)
    ap.add_argument("--out", default=os.path.join(_HERE, "data", "RAM_1min.csv"))
    args = ap.parse_args()
    download(args.days, args.port, args.host, args.client_id, args.out)


if __name__ == "__main__":
    main()
