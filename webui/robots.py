"""Resolve which robot the cockpit drives.

Three sources, in priority order:
  export  — ``robot_export/robot_params.json`` if a frozen robot has been exported
            (loaded via ``RamRobot.from_params`` — weights included, no retraining);
  default — a sane band_reverter phenotype (always available);
  mongo   — the best individual of a zion GE run, **only when zion_ge + pymongo are
            importable** (reuses ``strategy_exporter.best_phenotype_from_mongo`` +
            ``grammars.ram_grammar.decode_phenotype``). Hidden otherwise.

NN families (nn_reverter / morpheus_trader) loaded from a bare phenotype (default /
mongo) are fit on the provided history so their MLP has weights; export-loaded robots
already carry trained weights.
"""
from __future__ import annotations

import json
import os

from robot import RamRobot

# same default the exporter freezes when no run/phenotype is given
_DEFAULT_PHENO = {
    "family": "band_reverter", "band_gain": 1.1, "obs_lookback": 15,
    "memory_cap": 120, "ladder_bins": 20, "vol_window": 30,
    "trade_threshold": 0.12, "max_exposure": 1.0, "gap_fade_bias": 0.0,
    "trade_premarket": False, "trade_afterhours": False,
    "hold_through_gap": False, "seed": 42,
}


def _export_path(pkg_dir: str) -> str:
    return os.path.join(pkg_dir, "robot_export", "robot_params.json")


def _zion_ok() -> bool:
    try:
        import pymongo  # noqa: F401
        import zion_ge  # noqa: F401
        return True
    except Exception:
        return False


def _mongo_run_names() -> list[str]:
    """Best-effort list of zion run names that decode to a RAM robot."""
    try:
        import pymongo
        c = pymongo.MongoClient(
            os.environ.get("DATABASE_URL", "mongodb://localhost:27017/zion"),
            serverSelectionTimeoutMS=500,
        )
        try:
            db = c.get_default_database()
        except pymongo.errors.ConfigurationError:
            db = c["zion"]
        names = db.runs.distinct("name")
        return sorted(n for n in names if n)
    except Exception:
        return []


def list_robots(pkg_dir: str) -> list[dict]:
    """Selectable robots for the UI dropdown."""
    out: list[dict] = []
    exp = _export_path(pkg_dir)
    if os.path.exists(exp):
        try:
            with open(exp) as f:
                fam = json.load(f).get("phenotype", {}).get("family", "?")
        except Exception:
            fam = "?"
        out.append({"id": "export", "label": f"Exported robot ({fam})",
                    "source": "export", "family": fam})
    out.append({"id": "default", "label": "Default (band_reverter)",
                "source": "default", "family": "band_reverter"})
    if _zion_ok():
        for name in _mongo_run_names():
            out.append({"id": f"mongo:{name}", "label": f"run: {name}",
                        "source": "mongo", "family": "?"})
    return out


def load_robot(robot_id: str | None, pkg_dir: str, days=None) -> RamRobot:
    """Instantiate a robot for ``robot_id``. ``days`` (grouped bars) is used to fit
    NN families loaded from a bare phenotype; ignored for export-loaded robots."""
    rid = robot_id or "default"
    if rid == "export":
        with open(_export_path(pkg_dir)) as f:
            return RamRobot.from_params(json.load(f))
    if rid.startswith("mongo:"):
        from strategy_exporter import best_phenotype_from_mongo
        pheno = best_phenotype_from_mongo(rid.split(":", 1)[1])
        robot = RamRobot(pheno)
        if days is not None:
            robot.fit(days)             # rule families no-op; nn/morpheus train
        return robot
    # default
    robot = RamRobot(dict(_DEFAULT_PHENO))
    if days is not None:
        robot.fit(days)
    return robot
