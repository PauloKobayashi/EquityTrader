# Realtime WebUI for the RAM Intraday Trading Robot

## Context

The RAM intraday trading robot (belief-map GE robot for the 2x-DRAM ETF **RAM**) now
lives in its own standalone, zion-free repo: **`/Users/paulokobayashi/Documents/EquityTrader`**
(`github.com/PauloKobayashi/EquityTrader`, currently one commit, robot core =
`robot.py`, `market_map.py`, `backtest.py`, `data.py`, `_mlp.py`, live feed in
`live/ib_gateway.py`). There is **no frontend anywhere** for it — you can only run
`run.py`/`run_live.py` and read JSON.

We want a **realtime, browser-based cockpit** that shows what the robot is doing:
historical price, the model's exposure signal, buy/sell trades as arrows, the RAM
vs. portfolio return, a replay of any past window, a **day-focus** live view, and
visual "thinking" hints (the belief state) around each decision. It must run against
the **live IB Gateway (paper)** feed during market hours and fall back to a **replay
clock** off-hours. This is greenfield UI added to the EquityTrader repo.

### Ground rule (from project feedback)
The **engine is the source of truth**. All numbers the UI shows — signals, trades,
NAV, returns — are computed server-side by driving the *existing* engine functions
(`RamRobot.on_bar`, `backtest.run_episode`, `OnlineMarketMap.features`). The frontend
only renders server-computed arrays; **NAV/signals are never re-implemented in JS.**

## The robot, in one line
`RamRobot.on_bar(Bar) -> exposure ∈ [0,1]` (long/flat). A **trade** = an exposure
change past `trade_threshold`. `backtest.run_episode(robot, days, hold_through_gap,
cost_bps)` returns `(nav, per_day_ret, prices, exposures)`. The belief is 13 named
features from `OnlineMarketMap.features()` (`FEATURE_NAMES` in `market_map.py`), plus
internals VWAP / day-high / day-low / entry_price / overnight_gap.

## Decisions (confirmed with the user)
- **Location:** new `webui/` package **in the EquityTrader repo**, zion-free core.
- **Realtime source:** **live IB Gateway (paper, 4002)** primary; **replay clock**
  fallback off-hours/down. UI toggle: `Live | Replay` + speed.
- **Returns (both accumulated over the *visible* window, baselined to the last price
  before the window):** `RAM % = price[now]/price[baseline] − 1` (buy&hold benchmark);
  `Portfolio % = nav[now]/nav[baseline] − 1` (robot NAV). A 1-day view shows the 1-day
  return even early in the day.
- **Robot source:** load `robot_export/robot_params.json` if present, else the default
  phenotype; **also allow picking a run/individual from zion's Mongo** (optional, only
  when zion is importable — see Robots below).
- **Timeframes:** 1m, 10m, 30m, 1h, 2h, day, week, month.

---

## Architecture — new `webui/` package (in EquityTrader)

```
webui/
  __init__.py
  server.py      Flask app (fork the zion Explorer skeleton pattern):
                 GET /                       -> inline index.html
                 GET /api/robots             -> selectable robots
                 GET /api/bars               -> resampled OHLCV for a timeframe/window
                 GET /api/replay             -> per-bar prices/exposures/features/nav,
                                                derived trades, RAM% & Portfolio% series
                 GET /api/stream  (SSE)      -> live per-minute events
                 POST /api/mode              -> {source: live|replay, speed, robot_id, csv}
  feed.py        SSEBroadcaster + LiveFeed (IB Gateway hook) + ReplayFeed (clock)
  resample.py    1-min bars -> {10m,30m,1h,2h,1d,1w,1M} OHLCV + exposure/return rollups
  robots.py      resolve a robot: export json | default phenotype | Mongo pick (optional)
  rationale.py   per-family human-readable "why" string for a decision
  index.html     single-page Plotly.js UI (inline CSS/JS), served by server.py
  README.md
tests/test_webui.py
requirements-webui.txt   # flask (+ optional pymongo); Plotly.js from CDN, no server dep
```

Small **backward-compatible** change to `live/ib_gateway.py`: add an optional
`on_minute_cb` hook fired inside `IBApp._emit_minute` after the target is computed,
passing the finished `Bar`, target exposure, and `robot.map.features()`. The webui's
`LiveFeed` runs with `dry_run=True` (monitor only — **no order routing from the UI**)
and subscribes to that hook to publish SSE events. Existing `run_live.py` behavior is
unchanged (hook defaults to `None`).

### Data flow
- **Historical base:** `data/RAM_1min.csv` via `data.load_bars` / `group_by_day`
  (git-ignored; must exist locally — `data_download.py`). Optionally freshen the tail
  from the IBKR MCP `get_price_history` (≤1000 recent bars).
- **Live:** IB Gateway paper via the `on_minute_cb` hook above.
- **Replay clock:** `ReplayFeed` streams stored bars forward through a fresh robot at
  `speed` bars/sec, publishing the **same event shape** as `LiveFeed`.

### `resample.py` (new; the one genuinely new numeric code, unit-tested)
Aggregate native 1-min bars into buckets respecting ET sessions:
- sub-day (10m/30m/1h/2h): OHLC = first-open / max-high / min-low / last-close, V=sum,
  bucketed on wall-clock minute boundaries within a day (no cross-day buckets);
- day/week/month: built from `group_by_day` groups.
The robot's signal stays at its **native 1-min tick**; for a coarser display, roll
exposure up to the bucket (last exposure in bucket) and collapse the trades within a
bucket into net buy/sell markers. Returns roll up by compounding.

### Returns baselining (server-side, reuses `run_episode`)
To honor "accumulated over the visible window with a warm belief state": run the robot
over `[warmup_start .. now]` via `run_episode`, then **slice** `nav`/`prices` to the
visible window and rebase each to its first value. RAM% from prices, Portfolio% from
nav. Recomputed whenever timeframe/window changes.

### Trades from exposures (server-side)
Derive the trade list by diffing the `exposures` array: an entry where
`|Δexposure| > 0` at bar t → `{ts, price, side: 'buy' if Δ>0 else 'sell', dExposure,
exposure_after}`. (Matches how `run_episode` and live `_rebalance` treat a trade.)

### `robots.py`
`list_robots()` returns: the exported robot (if `robot_export/robot_params.json`
exists), a `default` phenotype robot, and — **only if zion_ge is importable** (e.g.
`PYTHONPATH=zion/src` set) — the runs from Mongo. `load_robot(id)`:
- export/default → `RamRobot.from_params` / `RamRobot(default)`;
- mongo → reuse `strategy_exporter`'s Mongo→phenotype path
  (`best_phenotype_from_mongo` / decode via `grammars.ram_grammar.decode_phenotype`)
  → `RamRobot(phenotype)`. If zion isn't available the run-picker is hidden.

---

## Frontend — `webui/index.html` (Plotly.js, forked from the Explorer skeleton)

Reuse from `zion/src/zion_ge/explorer/app.py`: the inline single-file dark-theme
skeleton, shared `plotlyLayout`/`plotlyConfig`, the `Plotly.react` update-in-place
loop, `switchTab` purge-and-rerender, and `preserveChartRanges`. Plotly.js 2.26 from
CDN. Panels:

1. **Price chart** — candlestick (line-close toggle) for the selected timeframe/window.
   Overlays:
   - **Trade arrows:** `scatter` markers `triangle-up` (buy, green) / `triangle-down`
     (sell, red), sized by |Δexposure| (elite-marker pattern, Explorer `app.py:4157`).
   - **Signal band:** stepped (`shape:'hv'`) exposure line / shaded 0..1 area = the
     model's target exposure over time.
   - **Belief overlays (thinking):** VWAP line, day high/low band, entry-price marker.
2. **Returns panel** — two `scattergl` lines with `fill:'tozeroy'` (Explorer conv-chart
   style): **RAM %** (buy&hold) and **Portfolio %** (robot NAV) over the visible window;
   a **table** beside it showing the current RAM % and Portfolio % (accumulated for the
   period), updated live/last-value on offline.
3. **Timeframe selector** — dropdown `1m/10m/30m/1h/2h/day/week/month` → re-fetch
   `/api/bars` + `/api/replay`, `Plotly.react`.
4. **Scrolling behavior** — keep a fixed visible width of `N` points. While
   `points < N`, append to the right and **grow** the x-range (fills space to the
   right). Once full, on each new point **shift the x-range left by one step** so the
   newest point sits at the right edge and older points scroll off-left. Implemented by
   setting `xaxis.range` explicitly per update (autorange off). Auto-slide **pauses**
   when the user has manually panned back into history (resume button).
5. **Mode toggle** — `Live | Replay` + speed slider → `POST /api/mode`. Live subscribes
   to `EventSource('/api/stream')`; each event appends one point and drives the scroll.
6. **Replay of previous windows** — window/date picker + playback controls
   (play/pause/step-fwd/step-back/speed/time-slider) modeled on the maze `ReplayState`
   (`step_idx`, `speed`, pause; `benchmarks/maze_runner/viz_v2.py`). Scrubs `step_idx`
   across the `/api/replay` arrays.
7. **Day-focus mode** — layout zoomed to one trading day, realtime price + exposure +
   trades for the focused day, with the thinking panel prominent.
8. **Thinking panel ("what the robot is thinking")** — the 13 belief features at the
   current/hovered bar as labeled bars/gauges (`FEATURE_NAMES`), plus a **before→after**
   comparison at each decision (feature vector at t-1 vs t and the resulting Δexposure)
   and a short **rationale string** from `rationale.py` (e.g. band_reverter: *"price
   high in day range (pos=0.90) → rule wants flat; Δexp 0.42 > threshold 0.22 →
   SELL"*). Price chart annotates each decision point with this tooltip so you see the
   trigger before and after.

---

## Files to create / touch (all in `/Users/paulokobayashi/Documents/EquityTrader`)
- **Create:** `webui/{__init__.py, server.py, feed.py, resample.py, robots.py,
  rationale.py, index.html, README.md}`, `requirements-webui.txt`, `tests/test_webui.py`.
- **Edit (backward compatible):** `live/ib_gateway.py` — add optional `on_minute_cb`
  hook to `IBApp` / `run`. `README.md` — document the webui + how to launch.
- **Reuse unchanged:** `robot.py`, `market_map.py`, `backtest.py`, `data.py`,
  `strategy_exporter.py`, `grammars/ram_grammar.py`.

## Reuse map (do not re-implement)
- Signals/NAV/returns: `backtest.run_episode`, `nav_from_exposures`, `summarize`,
  `RamRobot.on_bar` (`robot.py`), `OnlineMarketMap.features` (`market_map.py`).
- Live feed plumbing: `live/ib_gateway.py` (`IBApp`, `_MinAcc`, `_session_of`,
  `ram_contract`, `_parse_ib_date`) via the new hook.
- Frontend skeleton: `zion/src/zion_ge/explorer/app.py` (inline Flask+Plotly template,
  `plotlyLayout`, `Plotly.react`, `switchTab`, `preserveChartRanges`).
- Playback model: `benchmarks/maze_runner/viz_v2.py` `ReplayState` (concept only).
- Mongo robot pick: `strategy_exporter.best_phenotype_from_mongo` + `decode_phenotype`.
- SSE pattern: zion `api/sse.py` broadcaster / `dashboard/components/sse_client.py`.

## Workflow
Per project convention, do this on a **feature branch in the EquityTrader repo**
(e.g. `feat/webui-realtime`), then open a **PR** — unprompted.

## Verification (end-to-end)
1. **Unit tests** (`tests/test_webui.py`, run with the repo's python):
   - `resample.py`: bucket boundaries + OHLC/volume correctness at each timeframe; no
     cross-day buckets for intraday frames.
   - Trade derivation from an exposures array (side/size/threshold).
   - Return baselining: RAM% and Portfolio% over a window equal manual
     `price[now]/price[base]-1` and `nav[now]/nav[base]-1`.
   - Replay determinism (same window → identical arrays).
2. **Engine-parity check (source-of-truth):** for a fixed window, assert the webui's
   `/api/replay` nav/exposures **exactly equal** `run_episode` (and `run.py`) output —
   the UI must not drift from the engine.
3. **Run the app:** `python -m webui.server` (default CSV `data/RAM_1min.csv`), open in
   Chrome (Claude-in-Chrome). Verify: candlesticks render with buy/sell arrows + signal
   band; timeframe switching; the scroll-fill-then-push-left behavior; returns table
   matches computed values; replay playback + step controls; day-focus thinking panel
   updates before/after a decision.
4. **Live path:** with a paper IB Gateway on 4002, run in **Live** mode (dry-run/monitor,
   no orders) and confirm per-minute SSE events append points and drive the scroll;
   without a Gateway, confirm graceful fallback to **Replay** mode.
