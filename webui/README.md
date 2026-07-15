# webui — RAM intraday cockpit

Realtime browser cockpit for the RAM robot. Zion-free. **The engine is the source of
truth**: every number is computed server-side by driving the *existing* engine — the
frontend only renders server-computed arrays (NAV/signals are never re-implemented in
JS). See `../docs/webui-realtime-plan.md` for the full design.

## Run
```bash
pip install -r ../requirements-webui.txt
python -m webui.server            # http://127.0.0.1:5055
```

## Modules
| file | role |
|------|------|
| `server.py`   | Flask app + JSON/SSE API; `SSEBroadcaster` + `ReplayFeed` (replay clock). |
| `replay.py`   | `traced_episode` — engine-parity replay that also captures the belief per bar; `derive_trades`, `rebase_pct`. |
| `resample.py` | 1-min bars → `{10m,30m,1h,2h,day,week,month}` OHLCV buckets (ET-session aware) + exposure/nav rollups. |
| `robots.py`   | resolve a robot: `robot_export/` json \| default phenotype \| Mongo pick (optional, zion-only). |
| `rationale.py`| per-family human-readable "why" behind a decision. |
| `index.html`  | single-file Plotly.js UI (price + trades + signal band + returns + replay + thinking panel). |

## API
- `GET /api/config` — timeframes, feature names, bar range.
- `GET /api/robots` — selectable robots.
- `GET /api/bars?tf=` — resampled OHLCV.
- `GET /api/replay?robot=&tf=&window=&end=` — per-bucket prices/exposure/nav/features/
  belief, derived trades, and RAM% / Portfolio% rebased to the window baseline.
- `GET /api/stream` (SSE) — replay-clock per-minute events.
- `POST /api/mode` — `{source:'replay', play, speed, robot_id, start_index}`.

## Engine touch
One additive, backward-compatible method was added to the engine:
`OnlineMarketMap.belief_snapshot()` (`../market_map.py`) — the raw, un-squashed
internals (VWAP, day high/low, entry price, overnight gap) the thinking panel needs.

## Guarantee
`tests/test_webui.py` asserts `traced_episode`'s NAV/exposures equal `run_episode`
exactly. If they ever diverge, the tests fail — the UI cannot silently drift.
