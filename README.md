# RAM Intraday Trading Robot

A maze-runner-inspired grammatical-evolution setup that evolves a self-contained
intraday trading robot for **RAM** — the *Roundhill T-REX 2X Long DRAM Daily Target
ETF* (IBKR contract_id `895507567`, BATS/US). Optimized with the Zion-GE engine;
runs live against the **Mac IB Gateway**.

## Requirements & setup

The **robot core** (`market_map.py`, `_mlp.py`, `robot.py`, `data.py`, `backtest.py`)
is pure `numpy` + stdlib — no external engine — so an exported robot and the tests run
standalone. The **evolution** layer (`fitness.py`, `grammars/ram_grammar.py`) imports the
**Zion-GE** engine, which lives in a separate checkout; make it importable before running
`ge`:

```bash
pip install numpy                      # robot core + tests
pip install ibapi                      # live trading + data download (IB Gateway)
# Zion-GE engine (for `ge run`): install it, or point PYTHONPATH at its src/
export PYTHONPATH=/path/to/zion/src
```

Run commands below assume the repo root as the working directory (the configs reference
repo-relative modules `fitness` and `grammars.ram_grammar`). Market data is **not**
committed (broker-sourced) — populate `data/RAM_1min.csv` first with `data_download.py`
(see **Data**).

## Why this design

RAM launched ~2026-06-24, so it has **very little history** (~weeks of minute bars),
is wildly volatile (annualized vol ~237%, 20–30% daily swings), and shows frequent
sharp **reversals**. Overfitting is the enemy. We therefore transplant the ideas that
made the maze `morpheus`/`nn` robots *generalize* rather than the SPX/TSLib forecasting
stack:

- **`OnlineMarketMap`** (`market_map.py`) — the robot's self-built internal "map": an
  incrementally-updated belief of the intraday landscape (VWAP, session support/
  resistance, a visited-price ladder, vol regime, session context, the overnight gap,
  its own position). The maze `OnlineMapper`, ported to markets.
- **The information contract** — `features()` sees only past/present bars; only training
  rewards/labels may peek at the future. This is the classic look-ahead-leakage guard
  and the single most important idea carried over (tested in `tests/test_ram.py`).
- **Small memory / small models** as anti-overfit levers; **walk-forward disjoint days**
  with out-of-sample scoring; an explicit **`robustness_gap`** objective that penalizes
  overfitting directly.

Positioning is **long/flat only** (exposure ∈ [0,1], never short).

## Robot families (grammar-selectable)

| family | kind | idea |
|---|---|---|
| `band_reverter` | rule | buy low / sell high within the day's range (primary) |
| `vwap_reverter` | rule | mean-revert toward VWAP |
| `momentum` | rule | trend-follow |
| `nn_reverter` | tiny MLP (imitation) | imitates a *realizable* belief-state expert |
| `morpheus_trader` | tiny MLP (REINFORCE) | inner RL policy over the belief state |

"Morpheus" exists on **two orthogonal axes**: the inner `morpheus_trader` robot family
(a policy that *is* the robot) and the outer `morpheus` **selector** (deep-RL that
*evolves* robots, `configs/ram_intraday_morpheus.yaml`).

## Objectives (NSGA-II, 3)

`oos_return` (max, held-out) · `ulcer` (min) · `robustness_gap` = train − oos (min).

## Layout

```
market_map.py         OnlineMarketMap belief state           (zion-free)
_mlp.py               tiny numpy MLP (nn / morpheus)         (zion-free)
robot.py              RamRobot + 5 families                  (zion-free)
data.py               CSV loader + day/walk-forward split    (zion-free)
backtest.py           continuous-exposure NAV + ulcer        (zion-free)
fitness.py            evaluate() -> EvalResult  (the GE seam) (imports zion_ge)
grammars/ram_grammar.py   fresh maze-inspired grammar
configs/              ram_intraday{,_hybrid,_morpheus}.yaml
data/RAM_1min.csv     minute bars (datetime,open,high,low,close,volume,session)
strategy_exporter.py  freeze winner -> robot_export/
robot_export/         THE standalone robot (no zion, no retraining)
data_download.py      ibapi full-history downloader
live/ib_gateway.py    ibapi live feed + orders (paper-first)
live/run_live.py      drive the exported robot live
tests/test_ram.py     leakage / split / determinism / replay
```

## Data

> **Important:** the IBKR MCP `get_price_history` only returns the most recent ≤1000
> trailing bars (no date paging), so it can fetch a *recent sample* but **not** RAM's
> full multi-week minute history. The checked-in `data/RAM_1min.csv` is such a sample.

For the full history, run the ibapi downloader against a running Gateway (paper 4002
serves historical data too):

```bash
pip install ibapi
python data_download.py --days 25 --port 4002
```

The downloader respects IBKR historical-data **pacing limits**: it pages backward one
day per request (each extended day ≈ 960 one-min bars, under the per-request cap), keeps
a rolling window capped at 55 requests / 10 min (under IBKR's 60/10-min ceiling), leaves
a small gap between requests, backs off on a pacing violation (error 162), and **stops
automatically once it pages past RAM's inception** (no older data → no repeated identical
requests). `endDateTime` is sent with an explicit `US/Eastern` timezone. RAM's whole
history is only ~15–25 requests, comfortably inside the limits.

RAM has **pre-market + RTH + after-hours** bars (~16h/day) but **no true-overnight**
bars (not on IBKR's overnight venue). The Korea-driven DRAM signal (Samsung/SK Hynix
trade ~20:00–02:30 ET, inside the dataless gap) surfaces as the **pre-market open** and
the `overnight_gap` feature.

## Evolve

```bash
# baseline NSGA-II
PYTHONPATH=zion/src ge run -c configs/ram_intraday.yaml
# best generalizer: hybrid micro-GA + NSGA-II
PYTHONPATH=zion/src ge run -c configs/ram_intraday_hybrid.yaml
# deep-RL outer optimizer (needs the external Morpheus package; falls back to NSGA-II)
PYTHONPATH=zion/src ge run -c configs/ram_intraday_morpheus.yaml
```

Runs land in Mongo (`runs`, `evaluations`) and the Evolution Explorer (`:5051`).

## Export the winner + run it standalone

```bash
PYTHONPATH=zion/src python strategy_exporter.py --run ram_intraday_v1
cd robot_export
python run.py ../data/RAM_1min.csv          # no zion_ge, no retraining
```

## Live (paper-first)

Start IB Gateway (paper on 4002), then:

```bash
pip install ibapi
cd live
python run_live.py --robot ../robot_export              # PAPER, real paper orders
python run_live.py --robot ../robot_export --dry-run    # feed + printed decisions only
python run_live.py --robot ../robot_export --live --i-understand-live   # LIVE (guarded)
```

Default is the **paper port 4002**; live (4001) requires **both** `--live` and
`--i-understand-live`. Long/flat only; MKT during RTH, LMT+`outsideRth` in pre/after
market; positions reconciled via `reqPositions`.

## Cockpit (webui)

A realtime, browser-based cockpit that shows what the robot is doing: price with the
model's exposure signal, buy/sell trade arrows, RAM (buy&hold) vs. portfolio (robot
NAV) return, replay of any past window, and a "thinking" panel exposing the belief
state behind each decision. **The engine is the source of truth** — every number is
computed server-side by driving the *existing* engine (`RamRobot.on_bar`,
`backtest.run_episode` / `nav_from_exposures`, `OnlineMarketMap.features` /
`belief_snapshot`); the frontend only renders server-computed arrays.

```bash
pip install -r requirements-webui.txt      # flask + numpy (Plotly.js from CDN)
python -m webui.server                      # -> http://127.0.0.1:5055  (csv=data/RAM_1min.csv)
python -m webui.server --csv path/to.csv --port 5055
```

Timeframes `1m/10m/30m/1h/2h/day/week/month`; the robot runs at its native 1-min tick
and coarser frames roll it up (exposure = average commitment over the bucket, NAV =
end-of-bucket). Modes: **Replay** (a clock that streams stored bars through a fresh
robot over SSE, driving the live scroll) is shipped; **Live** (IB Gateway paper feed)
is the next increment. Pick a robot from `robot_export/` (if exported), the default
phenotype, or — when `zion_ge`+`pymongo` are importable — a GE run from Mongo.

## Tests

```bash
python -m pytest tests/test_ram.py tests/test_webui.py -v
```

`tests/test_webui.py` pins the cockpit to the engine: its `traced_episode` NAV /
exposures must equal `run_episode` exactly, so the UI can never drift.
