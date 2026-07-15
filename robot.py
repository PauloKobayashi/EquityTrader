"""RamRobot — the evolved intraday trading robot for RAM.

Maze-runner-inspired: the robot owns an ``OnlineMarketMap`` belief state (its
self-built "map") and a policy that turns the belief into a target **exposure in
[0, 1]** (long/flat only — never short). Five families, mirroring the maze
robot families:

  band_reverter   : rule — buy low / sell high within the day's range (primary)
  vwap_reverter   : rule — mean-revert toward VWAP
  momentum        : rule — trend-follow
  nn_reverter     : tiny MLP imitating a *realizable* expert (the band rule)
  morpheus_trader : tiny MLP trained by REINFORCE (reward may use the future;
                    the observation may not — the information contract)

Pure numpy + stdlib (NO zion_ge import) so the exporter can ship this file as-is.
"""
from __future__ import annotations

import numpy as np

from market_map import Bar, N_FEATURES, OnlineMarketMap
from _mlp import MLP

FAMILIES = ["band_reverter", "vwap_reverter", "momentum", "nn_reverter", "morpheus_trader"]
_NN_FAMILIES = {"nn_reverter", "morpheus_trader"}

# Feature indices (see market_map.FEATURE_NAMES)
_F_DVWAP = 0
_F_POSRANGE = 1
_F_MOM = 2
_F_GAP = 10


def _default_phenotype() -> dict:
    return {
        # perception
        "obs_lookback": 15, "memory_cap": 120, "ladder_bins": 20, "vol_window": 30,
        # strategy
        "family": "band_reverter",
        "band_gain": 1.0, "vwap_k": 1.0, "mom_k": 1.0,
        "nn_hidden": 8, "nn_seed": 0,
        "ppo_lr": 3e-3, "ppo_steps": 30, "ppo_hidden": 8,
        # trading
        "trade_threshold": 0.1, "max_exposure": 1.0, "cost_bps": 0.001,
        "trade_premarket": False, "trade_afterhours": False,
        "hold_through_gap": False, "gap_fade_bias": 0.0, "seed": 42,
    }


class RamRobot:
    def __init__(self, phenotype: dict | None = None):
        p = _default_phenotype()
        if phenotype:
            p.update(phenotype)
        self.p = p
        self.family = p["family"]
        self.max_exposure = float(np.clip(p["max_exposure"], 0.0, 1.0))
        self.trade_threshold = float(p["trade_threshold"])
        self.cost_bps = float(p["cost_bps"])
        self.gap_fade_bias = float(p["gap_fade_bias"])
        self.hold_through_gap = bool(p["hold_through_gap"])
        self._enabled = {"rth"}
        if p.get("trade_premarket"):
            self._enabled.add("premarket")
        if p.get("trade_afterhours"):
            self._enabled.add("afterhours")
        self.map = OnlineMarketMap(
            obs_lookback=p["obs_lookback"], memory_cap=p["memory_cap"],
            ladder_bins=p["ladder_bins"], vol_window=p["vol_window"],
        )
        self.exposure = 0.0
        self.mlp: MLP | None = None
        if self.family == "nn_reverter":
            self.mlp = MLP(N_FEATURES, int(p["nn_hidden"]), seed=int(p["nn_seed"]))
        elif self.family == "morpheus_trader":
            self.mlp = MLP(N_FEATURES, int(p["ppo_hidden"]), seed=int(p["nn_seed"]))

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self.map.reset()
        self.exposure = 0.0

    # -- policies (feature vector -> raw target exposure) -------------------

    def _band_target(self, f: np.ndarray) -> float:
        # buy low within the day's range: pos=0 (low) -> long, pos=1 (high) -> flat
        return float(np.clip((1.0 - f[_F_POSRANGE]) * self.p["band_gain"], 0.0, 1.0))

    def _vwap_target(self, f: np.ndarray) -> float:
        return float(np.clip(0.5 - self.p["vwap_k"] * f[_F_DVWAP], 0.0, 1.0))

    def _mom_target(self, f: np.ndarray) -> float:
        return float(np.clip(0.5 + self.p["mom_k"] * f[_F_MOM], 0.0, 1.0))

    def _rule_target(self, f: np.ndarray) -> float:
        if self.family == "band_reverter":
            return self._band_target(f)
        if self.family == "vwap_reverter":
            return self._vwap_target(f)
        return self._mom_target(f)

    def _expert_target(self, f: np.ndarray) -> float:
        """Realizable imitation target for nn_reverter: the band rule on the SAME
        features (a function of the observation, so imitation is realizable — the
        maze lesson that the true-solution expert is NOT realizable)."""
        return self._band_target(f)

    def _policy_target(self, f: np.ndarray) -> float:
        if self.family in _NN_FAMILIES and self.mlp is not None:
            raw = float(self.mlp(f))
        else:
            raw = self._rule_target(f)
        raw = raw - self.gap_fade_bias * float(f[_F_GAP])   # fade the overnight gap
        return float(np.clip(raw, 0.0, self.max_exposure))

    # -- streaming decision -------------------------------------------------

    def on_bar(self, bar: Bar) -> float:
        self.map.observe(bar)
        f = self.map.features()
        price = float(bar.close)
        if bar.session not in self._enabled:
            target = self.exposure                       # not allowed to trade now: hold
        else:
            raw = self._policy_target(f)
            if abs(raw - self.exposure) > self.trade_threshold:
                target = raw
            else:
                target = self.exposure                   # deadzone: no trade
        target = float(np.clip(target, 0.0, self.max_exposure))
        self.map.set_position(target, price)
        self.exposure = target
        return target

    # -- training (nn / morpheus families) ----------------------------------

    def fit(self, train_days: list[list[Bar]]) -> None:
        if self.family == "nn_reverter":
            self._fit_imitation(train_days)
        elif self.family == "morpheus_trader":
            self._fit_reinforce(train_days)
        # rule families: nothing to fit

    def _rollout_features(self, days, target_fn):
        """Replay days under target_fn, collecting (features, target) per bar."""
        X, Y = [], []
        self.reset()
        for day in days:
            for bar in day:
                self.map.observe(bar)
                f = self.map.features()
                t = target_fn(f)
                X.append(f.copy())
                Y.append(t)
                self.map.set_position(t, float(bar.close))
                self.exposure = t
        return np.array(X), np.array(Y)

    def _fit_imitation(self, train_days) -> None:
        # generate the expert trajectory under the band rule, then regress toward it
        band = RamRobot({**self.p, "family": "band_reverter"})
        band.reset()
        X, Y = [], []
        for day in train_days:
            for bar in day:
                band.map.observe(bar)
                f = band.map.features()
                t = band._expert_target(f)
                X.append(f.copy()); Y.append(t)
                band.map.set_position(t, float(bar.close)); band.exposure = t
        if not X:
            return
        self.mlp.fit_mse(np.array(X), np.array(Y), epochs=200,
                         lr=3e-3, seed=int(self.p["nn_seed"]))

    def _fit_reinforce(self, train_days) -> None:
        # precompute per-day price arrays for reward = a_t * r_{t+1} - turnover cost
        day_prices = [np.array([b.close for b in d], dtype=np.float64) for d in train_days]
        cost = self.cost_bps
        max_e = self.max_exposure

        def episodes(policy: MLP, rng):
            X, acts, rews = [], [], []
            self.reset()
            for di, day in enumerate(train_days):
                prices = day_prices[di]
                prev_a = 0.0
                for i, bar in enumerate(day):
                    self.map.observe(bar)
                    f = self.map.features()
                    mean = float(policy(f))
                    a = float(np.clip(mean + rng.normal(0, 0.15), 0.0, max_e))
                    if i + 1 < len(prices) and prices[i] > 0:
                        r = prices[i + 1] / prices[i] - 1.0
                    else:
                        r = 0.0
                    reward = a * r - abs(a - prev_a) * cost
                    X.append(f.copy()); acts.append(a); rews.append(reward)
                    self.map.set_position(a, float(bar.close)); self.exposure = a
                    prev_a = a
            return np.array(X), np.array(acts), np.array(rews)

        self.mlp.reinforce_fit(episodes, epochs=int(self.p["ppo_steps"]),
                               lr=float(self.p["ppo_lr"]), seed=int(self.p["seed"]))

    # -- serialization (for the standalone export) --------------------------

    def to_params(self) -> dict:
        out = {"phenotype": dict(self.p)}
        if self.mlp is not None:
            out["mlp"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in self.mlp.state_dict().items()}
        return out

    @classmethod
    def from_params(cls, params: dict) -> "RamRobot":
        robot = cls(params["phenotype"])
        if "mlp" in params and robot.mlp is not None:
            sd = {k: np.array(v) for k, v in params["mlp"].items()}
            robot.mlp.load_state_dict(sd)
        return robot
