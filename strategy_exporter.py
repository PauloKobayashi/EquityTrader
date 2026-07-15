"""Freeze an evolved RAM robot into a self-contained, runnable artifact.

Unlike the maze benchmark (which re-trains from a seed), we serialize the *frozen*
robot — phenotype + trained MLP weights — so ``robot_export/`` runs by itself with
NO retraining and NO zion_ge dependency. The SAME ``RamRobot`` class backs the
backtest, this export, and live trading (single source of truth).

Usage:
  # export the best individual of a finished run (reads Mongo):
  python strategy_exporter.py --run ram_intraday_v1
  # or export an explicit phenotype JSON:
  python strategy_exporter.py --phenotype '{"family":"band_reverter","band_gain":1.1,...}'
  # or export a sane default:
  python strategy_exporter.py --default
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from data import load_bars, group_by_day
from robot import RamRobot

# core files copied verbatim into the export (all zion-free)
_CORE_FILES = ["market_map.py", "_mlp.py", "robot.py", "data.py", "backtest.py"]

_RUN_PY = '''"""Standalone RAM robot runner — no zion_ge, no retraining.

Usage: python run.py path/to/RAM_1min.csv [--capital 100000]
Streams the CSV through the frozen robot and prints target exposure per bar plus
the final NAV. This is the exact policy the live trader (live/run_live.py) drives.
"""
import json, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import RamRobot
from data import load_bars, group_by_day
from backtest import run_episode, summarize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_params.json")) as f:
        params = json.load(f)
    robot = RamRobot.from_params(params)
    days = group_by_day(load_bars(args.csv))
    cost = params["phenotype"].get("cost_bps", 0.0015)
    nav, dret, prices, exps = run_episode(robot, days, robot.hold_through_gap, cost)
    if not args.quiet:
        for i in range(0, len(prices), max(1, len(prices)//20 or 1)):
            print(f"bar {i:5d} price={prices[i]:.2f} exposure={exps[i]:.2f}")
    s = summarize(nav, dret)
    print(json.dumps({"final_nav_pct": s["final_return_pct"],
                      "median_day_pct": s["median_day_return_pct"],
                      "ulcer": s["ulcer"], "n_days": s["n_days"],
                      "family": params["phenotype"]["family"]}, indent=2))

if __name__ == "__main__":
    main()
'''


def best_phenotype_from_mongo(run_name: str) -> dict:
    """Decode the best (objective-0) individual of a run into a phenotype."""
    import pymongo
    from zion_ge.core.gene import PipelineGenotype
    from grammars.ram_grammar import decode_phenotype

    class _Ind:
        def __init__(self, geno, iid):
            self.genotype = geno
            self.id = iid

    c = pymongo.MongoClient(os.environ.get("DATABASE_URL", "mongodb://localhost:27017/zion"))
    try:
        db = c.get_default_database()
    except pymongo.errors.ConfigurationError:
        db = c["zion"]
    # Pick the most recent run doc with this name (checkpoints re-write the run row).
    run = db.runs.find_one({"name": run_name}, sort=[("updated_at", -1)])
    if not run:
        raise SystemExit(f"run '{run_name}' not found in Mongo")
    # Objectives + metadata live in `evaluations`, stored in RAW natural orientation
    # (verified: objective_vector[0] == metadata.grammar_objs.oos_return). oos_return is
    # a MAX objective, so the winner is the LARGEST objective_vector[0]. (The engine's
    # internal NSGA-II uses sign-flipped copies on the individuals; the persisted eval
    # docs are raw — do NOT invert here.)
    best = None
    for e in db.evaluations.find({"run_id": run["_id"], "is_valid": True}):
        ov = e.get("objective_vector")
        if not ov:
            continue
        if best is None or ov[0] > best["objective_vector"][0]:
            best = e
    if best is None:
        raise SystemExit("no valid evaluations found for run")
    ind = db.individuals.find_one({"_id": best["individual_id"]})
    if not ind or "genotype_blob" not in ind:
        raise SystemExit("winning individual / genotype not found")
    geno = PipelineGenotype.from_bytes(ind["genotype_blob"])
    pheno = decode_phenotype(_Ind(geno, str(ind["_id"])))
    ov = best["objective_vector"]
    print(f"best eval oos_return={ov[0]:+.3f}%/day  ulcer={ov[1]:.3f}  "
          f"robustness_gap={ov[2]:+.3f}  family={pheno['family']}")
    return pheno


def export_robot(phenotype: dict, data_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    days = group_by_day(load_bars(data_path))
    robot = RamRobot(phenotype)
    robot.fit(days)                                  # fit on ALL available history
    params = robot.to_params()
    with open(os.path.join(out_dir, "robot_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    for fn in _CORE_FILES:
        shutil.copy(os.path.join(_PKG_DIR, fn), os.path.join(out_dir, fn))
    with open(os.path.join(out_dir, "run.py"), "w") as f:
        f.write(_RUN_PY)
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(
            "Self-contained RAM intraday robot.\n"
            f"Family: {phenotype['family']}\n\n"
            "Run a backtest on a CSV (datetime,open,high,low,close,volume,session):\n"
            "  python run.py RAM_1min.csv\n\n"
            "No zion_ge, no retraining. The same RamRobot drives live/run_live.py.\n"
        )
    print(f"exported {phenotype['family']} robot -> {out_dir}")
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run name in Mongo (export its best individual)")
    ap.add_argument("--phenotype", help="phenotype JSON string")
    ap.add_argument("--default", action="store_true", help="export a default band_reverter")
    ap.add_argument("--data", default=os.path.join(_PKG_DIR, "data", "RAM_1min.csv"))
    ap.add_argument("--out", default=os.path.join(_PKG_DIR, "robot_export"))
    args = ap.parse_args()

    if args.run:
        pheno = best_phenotype_from_mongo(args.run)
    elif args.phenotype:
        pheno = json.loads(args.phenotype)
    else:
        pheno = {"family": "band_reverter", "band_gain": 1.1, "obs_lookback": 15,
                 "memory_cap": 120, "ladder_bins": 20, "vol_window": 30,
                 "trade_threshold": 0.12, "max_exposure": 1.0, "gap_fade_bias": 0.0,
                 "trade_premarket": False, "trade_afterhours": False,
                 "hold_through_gap": False, "seed": 42}
    export_robot(pheno, args.data, args.out)


if __name__ == "__main__":
    main()
