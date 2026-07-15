"""webui — a realtime browser cockpit for the RAM intraday robot.

Zion-free. Every number the UI shows (signals, trades, NAV, returns) is computed
server-side by driving the *existing* engine (``RamRobot.on_bar``,
``backtest.run_episode`` / ``nav_from_exposures``, ``OnlineMarketMap.features`` /
``belief_snapshot``). The frontend only renders server-computed arrays — NAV and
signals are never re-implemented in JS. See ``docs/webui-realtime-plan.md``.
"""
