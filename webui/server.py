"""Flask server for the RAM intraday cockpit (forked from the zion Explorer skeleton).

Serves a single-file Plotly.js UI and a small JSON/SSE API. Every number is computed
server-side by driving the existing engine (see webui.replay / robots / rationale) —
the frontend only renders. PR #1 ships the Replay path; Live (IB Gateway) is PR #2.

Run:  python -m webui.server [--csv data/RAM_1min.csv] [--port 5055]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

from flask import Flask, Response, jsonify, request, stream_with_context

# import engine + webui modules with the package dir on sys.path
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)
import sys
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from data import load_bars, group_by_day
from market_map import FEATURE_NAMES
from webui import replay as _replay
from webui import resample as _resample
from webui import robots as _robots
from webui import rationale as _rationale

_DEFAULT_CSV = os.path.join(_REPO_DIR, "data", "RAM_1min.csv")


# --------------------------------------------------------------------------- #
# Server state — bars loaded once; traced episodes / bucketings memoized.
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.bars = load_bars(csv_path)
        self.days = group_by_day(self.bars)
        self._robots: dict[str, object] = {}          # robot_id -> RamRobot
        self._traces: dict[str, dict] = {}            # robot_id -> traced_episode
        self._buckets: dict[str, list] = {}           # tf -> resampled buckets
        self._lock = threading.Lock()

    def robot(self, robot_id: str):
        with self._lock:
            if robot_id not in self._robots:
                self._robots[robot_id] = _robots.load_robot(robot_id, _REPO_DIR, self.days)
            return self._robots[robot_id]

    def trace(self, robot_id: str) -> dict:
        with self._lock:
            if robot_id not in self._traces:
                r = self._robots.get(robot_id) or _robots.load_robot(robot_id, _REPO_DIR, self.days)
                self._robots[robot_id] = r
                self._traces[robot_id] = _replay.traced_episode(
                    r, self.days, r.hold_through_gap, r.cost_bps)
            return self._traces[robot_id]

    def buckets(self, tf: str) -> list:
        with self._lock:
            if tf not in self._buckets:
                self._buckets[tf] = _resample.resample_bars(self.bars, tf)
            return self._buckets[tf]


# --------------------------------------------------------------------------- #
# SSE broadcaster — one bounded queue per subscriber, drop-on-full back-pressure
# (forked from zion api/sse.py, sync/threaded variant for the Flask dev server).
# --------------------------------------------------------------------------- #
class SSEBroadcaster:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass    # slow consumer: drop rather than block the feed


# --------------------------------------------------------------------------- #
# ReplayFeed — stream native 1-min bars through a fresh robot at `speed` bars/sec,
# publishing the same event shape a live feed will (PR #2). NAV is chained
# incrementally with the exact nav_from_exposures recurrence (O(1)/bar).
# --------------------------------------------------------------------------- #
class ReplayFeed(threading.Thread):
    def __init__(self, state: State, broadcaster: SSEBroadcaster,
                 robot_id: str, speed: float, start_index: int = 0):
        super().__init__(daemon=True)
        self.state = state
        self.bc = broadcaster
        self.robot_id = robot_id
        self.speed = max(0.1, float(speed))
        self.start_index = max(0, int(start_index))
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        robot = _robots.load_robot(self.robot_id, _REPO_DIR, self.state.days)
        robot.reset()
        cost = robot.cost_bps
        htg = robot.hold_through_gap

        nav = 100.0
        e_prev = 0.0          # exposure at t-1
        e_cur = None          # exposure at t
        p_cur = None          # price at t
        idx = 0
        for day in self.state.days:
            for i, bar in enumerate(day):
                if self._stop.is_set():
                    return
                exp = robot.on_bar(bar)
                if not htg and i == len(day) - 1:
                    exp = 0.0
                    robot.exposure = 0.0
                    robot.map.set_position(0.0, float(bar.close))
                price = float(bar.close)
                # finalize nav for this bar using the previous bar's state
                if p_cur is not None:
                    turn = abs(e_cur - e_prev) * cost
                    val = nav * (1.0 - turn)
                    r = price / p_cur - 1.0 if p_cur > 0 else 0.0
                    nav = val * (1.0 + e_cur * r)
                    e_prev = e_cur
                trade = e_cur is not None and abs(exp - e_cur) > 1e-9
                # emit only from start_index onward (belief is warmed silently before)
                if idx >= self.start_index:
                    d = exp - (e_cur if e_cur is not None else 0.0)
                    feats = [float(x) for x in robot.map.features()]
                    ev = {
                        "ts": bar.ts,
                        "price": price,
                        "exposure": float(exp),
                        "nav": float(nav),
                        "features": feats,
                        "belief": robot.map.belief_snapshot(),
                        "trade": bool(trade),
                        "side": ("buy" if d > 0 else "sell") if trade else None,
                        "dExposure": float(d),
                        "index": idx,
                    }
                    if trade:
                        ev["rationale"] = _rationale.explain(
                            robot, feats, e_cur if e_cur is not None else 0.0)["text"]
                    self.bc.publish(ev)
                e_cur = exp
                p_cur = price
                idx += 1
                self._stop.wait(1.0 / self.speed)
        self.bc.publish({"done": True})


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(csv_path: str) -> Flask:
    app = Flask(__name__)
    state = State(csv_path)
    broadcaster = SSEBroadcaster()
    feed_holder: dict[str, ReplayFeed] = {}

    @app.route("/")
    def index():
        with open(os.path.join(_PKG_DIR, "index.html")) as f:
            return f.read()

    @app.route("/api/config")
    def config():
        return jsonify({
            "csv": os.path.basename(state.csv_path),
            "timeframes": _resample.TIMEFRAMES,
            "feature_names": list(FEATURE_NAMES),
            "n_bars": len(state.bars),
            "n_days": len(state.days),
            "first_ts": state.bars[0].ts if state.bars else None,
            "last_ts": state.bars[-1].ts if state.bars else None,
        })

    @app.route("/api/robots")
    def robots():
        return jsonify(_robots.list_robots(_REPO_DIR))

    @app.route("/api/bars")
    def bars():
        tf = request.args.get("tf", "1m")
        bk = state.buckets(tf)
        return jsonify({
            "tf": tf,
            "ts": [b["ts"] for b in bk],
            "open": [b["open"] for b in bk],
            "high": [b["high"] for b in bk],
            "low": [b["low"] for b in bk],
            "close": [b["close"] for b in bk],
            "volume": [b["volume"] for b in bk],
        })

    @app.route("/api/replay")
    def api_replay():
        robot_id = request.args.get("robot", "default")
        tf = request.args.get("tf", "1m")
        window = request.args.get("window", type=int)     # bucket count; None => all
        end = request.args.get("end", type=int)           # exclusive bucket index

        robot = state.robot(robot_id)
        tr = state.trace(robot_id)
        bk = state.buckets(tf)
        n = len(bk)
        exp_b = _resample.rollup_mean(tr["exposures"], bk)   # avg commitment (see rollup_mean)
        nav_b = _resample.rollup_last(tr["nav"], bk)          # end-of-bucket NAV
        price_b = [b["close"] for b in bk]
        feat_b = _resample.rollup_at(tr["features"], bk)
        belief_b = _resample.rollup_at(tr["belief"], bk)
        ts_b = [b["ts"] for b in bk]

        end_idx = n if end is None else max(1, min(end, n))
        start_idx = 0 if not window else max(0, end_idx - window)
        base_idx = start_idx - 1 if start_idx > 0 else start_idx
        base_price = price_b[base_idx]
        base_nav = nav_b[base_idx]

        sl = slice(start_idx, end_idx)
        price_s = price_b[sl]
        nav_s = nav_b[sl]

        # bucket-level trades across the full series, then keep those in view
        all_trades = _replay.derive_trades(ts_b, price_b, exp_b)
        trades = []
        for t in all_trades:
            if start_idx <= t["i"] < end_idx:
                exp_before = exp_b[t["i"] - 1] if t["i"] > 0 else 0.0
                t = dict(t)
                t["rationale"] = _rationale.explain(robot, feat_b[t["i"]], exp_before)["text"]
                trades.append(t)

        return jsonify({
            "robot": robot_id,
            "family": robot.family,
            "tf": tf,
            "feature_names": list(FEATURE_NAMES),
            "n_total": n,
            "start_index": start_idx,
            "end_index": end_idx,
            "baseline_price": base_price,
            "baseline_nav": base_nav,
            "bars": {
                "ts": [b["ts"] for b in bk[sl]],
                "open": [b["open"] for b in bk[sl]],
                "high": [b["high"] for b in bk[sl]],
                "low": [b["low"] for b in bk[sl]],
                "close": price_s,
                "volume": [b["volume"] for b in bk[sl]],
            },
            "exposure": exp_b[sl],
            "nav": nav_s,
            "ram_pct": _replay.rebase_pct(price_s, base_price),
            "port_pct": _replay.rebase_pct(nav_s, base_nav),
            "features": feat_b[sl],
            "belief": belief_b[sl],
            "trades": trades,
        })

    @app.route("/api/mode", methods=["POST"])
    def api_mode():
        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "replay")
        # stop any running feed
        old = feed_holder.pop("feed", None)
        if old is not None:
            old.stop()
        if source == "replay" and body.get("play"):
            feed = ReplayFeed(
                state, broadcaster,
                robot_id=body.get("robot_id", "default"),
                speed=float(body.get("speed", 10.0)),
                start_index=int(body.get("start_index", 0)),
            )
            feed_holder["feed"] = feed
            feed.start()
            return jsonify({"ok": True, "streaming": True, "source": source})
        return jsonify({"ok": True, "streaming": False, "source": source})

    @app.route("/api/stream")
    def api_stream():
        q = broadcaster.subscribe()

        @stream_with_context
        def gen():
            try:
                # prelude comment so the connection opens immediately
                yield ": connected\n\n"
                while True:
                    try:
                        ev = q.get(timeout=15.0)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(ev)}\n\n"
                    if ev.get("done"):
                        break
            finally:
                broadcaster.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_DEFAULT_CSV)
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = create_app(args.csv)
    print(f"RAM cockpit -> http://{args.host}:{args.port}  (csv={os.path.basename(args.csv)})")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
