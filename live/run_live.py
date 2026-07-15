"""Drive the exported RAM robot live against the Mac IB Gateway.

PAPER by default (port 4002). LIVE (4001) requires BOTH --live and
--i-understand-live so real money is never traded by accident.

Examples:
  # paper, real orders on the paper account:
  python run_live.py --robot ../robot_export
  # paper, no orders (feed + printed decisions only):
  python run_live.py --robot ../robot_export --dry-run
  # LIVE money (guarded):
  python run_live.py --robot ../robot_export --live --i-understand-live
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (_HERE, _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot import RamRobot
import ib_gateway  # noqa: E402  (sibling in live/)


def load_robot(robot_dir: str) -> RamRobot:
    with open(os.path.join(robot_dir, "robot_params.json")) as f:
        return RamRobot.from_params(json.load(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=os.path.join(_PKG, "robot_export"),
                    help="exported robot dir (contains robot_params.json)")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--client-id", type=int, default=17)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true",
                    help="feed bars + print intended orders, but do not place them")
    ap.add_argument("--live", action="store_true", help="use the LIVE port 4001")
    ap.add_argument("--i-understand-live", action="store_true",
                    help="second confirmation required to trade real money")
    args = ap.parse_args()

    if args.live:
        if not args.i_understand_live:
            raise SystemExit("Refusing LIVE trading: pass --i-understand-live to confirm.")
        port = ib_gateway.LIVE_PORT
    else:
        port = ib_gateway.PAPER_PORT

    robot = load_robot(args.robot)
    print(f"[robot] family={robot.family} max_exp={robot.max_exposure} "
          f"threshold={robot.trade_threshold} sessions={sorted(robot._enabled)}")
    ib_gateway.run(robot, capital=args.capital, port=port,
                   client_id=args.client_id, dry_run=args.dry_run, host=args.host)


if __name__ == "__main__":
    main()
