"""Zion-GE evaluation adapter for the RAM intraday robot.

Bridges: genotype -> phenotype -> RamRobot -> walk-forward backtest -> 3 objectives.

Objectives (natural orientation; the engine applies max/min signs from the config):
  oos_return     (max) : median per-day return on the HELD-OUT test days
  ulcer          (min) : SUM-drawdown ulcer over the test window
  robustness_gap (min) : train_return - oos_return  (penalize overfitting directly)

Only rule families skip training; nn/morpheus are fit on train days, then scored
purely out-of-sample — the maze disjoint-train/test generalization contract.
"""
from __future__ import annotations

import os
import sys
import time
import uuid

# This module is imported by the engine via its dotted path
# (private.ram_intraday.fitness), but the robot core uses bare sibling imports
# (so the same files export as a standalone robot). Put our own dir on sys.path
# first so those bare imports resolve regardless of how we were imported.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from zion_ge.executors.base import EvalConfig, EvalResult

from grammars.ram_grammar import decode_phenotype
from data import load_bars, group_by_day, walkforward_split
from robot import RamRobot
from backtest import run_episode, ulcer as _ulcer, summarize

OBJECTIVE_NAMES = ("oos_return", "ulcer", "robustness_gap")

# Fixed, conservative round-trip cost for a volatile 2x ETF (slippage + commission).
# NOT evolvable — kept out of the grammar so evolution cannot cheat costs down.
FIXED_COST_BPS = 0.0015

_PENALTY = [-1.0e9, 1.0e9, 1.0e9]

_DATA_CACHE: dict[str, list] = {}


def _default_data_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("RAM_DATA", os.path.join(here, "data", "RAM_1min.csv"))


def _load_days(path: str):
    if path not in _DATA_CACHE:
        _DATA_CACHE[path] = group_by_day(load_bars(path))
    return _DATA_CACHE[path]


def _to_uuid(ind) -> uuid.UUID:
    iid = getattr(ind, "id", None)
    if isinstance(iid, uuid.UUID):
        return iid
    if isinstance(iid, str):
        try:
            return uuid.UUID(iid)
        except ValueError:
            return uuid.uuid4()
    return uuid.uuid4()


def evaluate(individual, config: EvalConfig | None = None) -> EvalResult:
    t0 = time.time()
    ind_id = _to_uuid(individual)
    try:
        pheno = decode_phenotype(individual)
        pheno["cost_bps"] = FIXED_COST_BPS
        robot = RamRobot(pheno)

        days = _load_days(_default_data_path())
        train_days, test_days = walkforward_split(days, train_frac=0.6)

        robot.fit(train_days)

        _, train_ret, _, _ = run_episode(
            robot, train_days, robot.hold_through_gap, FIXED_COST_BPS)
        nav_test, test_ret, _, _ = run_episode(
            robot, test_days, robot.hold_through_gap, FIXED_COST_BPS)

        import numpy as np
        oos_return = float(np.median(test_ret)) if len(test_ret) else 0.0
        train_return = float(np.median(train_ret)) if len(train_ret) else 0.0
        ulcer_val = _ulcer(nav_test)
        robustness_gap = train_return - oos_return

        meta = {
            "family": pheno["family"],
            "grammar_objs": {
                "oos_return": oos_return,
                "train_return": train_return,
                "ulcer": ulcer_val,
                "robustness_gap": robustness_gap,
                **summarize(nav_test, test_ret),
                "n_train_days": len(train_days),
                "n_test_days": len(test_days),
            },
            "exception": "OK",
        }
        return EvalResult(
            individual_id=ind_id,
            objective_vector=[oos_return, ulcer_val, robustness_gap],
            objective_names=list(OBJECTIVE_NAMES),
            is_valid=True,
            runtime_ms=(time.time() - t0) * 1000.0,
            metadata=meta,
        )
    except Exception as exc:  # noqa: BLE001 — penalty vector on any failure (qqq convention)
        return EvalResult(
            individual_id=ind_id,
            objective_vector=list(_PENALTY),
            objective_names=list(OBJECTIVE_NAMES),
            is_valid=False,
            error_message=str(exc),
            runtime_ms=(time.time() - t0) * 1000.0,
            metadata={"exception": repr(exc)},
        )
