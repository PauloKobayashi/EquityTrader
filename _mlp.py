"""Tiny numpy MLP used by the nn_reverter (imitation) and morpheus_trader (RL) robot
families. Pure numpy so the exported robot needs no torch — the nets are deliberately
small (tiny data → generalization-first, per the maze findings).

One hidden layer, tanh activation, sigmoid output in [0, 1] (long/flat exposure).
Supports:
  - supervised MSE fit (imitation of a realizable expert)         -> nn_reverter
  - REINFORCE policy-gradient step (Gaussian around the output)   -> morpheus_trader
Weights serialize to / from plain dicts of numpy arrays (npz-friendly).
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class MLP:
    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        self.n_in = int(n_in)
        self.n_hidden = max(1, int(n_hidden))
        rng = np.random.default_rng(int(seed))
        s1 = np.sqrt(1.0 / self.n_in)
        s2 = np.sqrt(1.0 / self.n_hidden)
        self.W1 = rng.normal(0, s1, size=(self.n_in, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        self.W2 = rng.normal(0, s2, size=(self.n_hidden, 1))
        self.b2 = np.zeros(1)

    # -- inference ----------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., n_in) -> exposure in [0,1], shape (...,)."""
        x = np.atleast_2d(x)
        h = np.tanh(x @ self.W1 + self.b1)
        out = _sigmoid(h @ self.W2 + self.b2)
        return out.reshape(-1)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.forward(x)[0])

    # -- supervised (imitation) training ------------------------------------

    def fit_mse(self, X: np.ndarray, y: np.ndarray, epochs: int = 200,
                lr: float = 1e-2, l2: float = 1e-4, seed: int = 0) -> None:
        """Fit exposure targets y in [0,1] from features X via Adam + MSE."""
        X = np.atleast_2d(X).astype(np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = X.shape[0]
        if n == 0:
            return
        opt = _Adam([self.W1, self.b1, self.W2, self.b2], lr=lr)
        rng = np.random.default_rng(int(seed))
        batch = min(64, n)
        for _ in range(int(epochs)):
            idx = rng.permutation(n)
            for start in range(0, n, batch):
                bi = idx[start:start + batch]
                xb, yb = X[bi], y[bi]
                # forward
                z1 = xb @ self.W1 + self.b1
                h = np.tanh(z1)
                z2 = h @ self.W2 + self.b2
                out = _sigmoid(z2)
                # backward (MSE)
                m = xb.shape[0]
                dout = (out - yb) * out * (1 - out) / m
                gW2 = h.T @ dout + l2 * self.W2
                gb2 = dout.sum(axis=0)
                dh = (dout @ self.W2.T) * (1 - h ** 2)
                gW1 = xb.T @ dh + l2 * self.W1
                gb1 = dh.sum(axis=0)
                opt.step([gW1, gb1, gW2, gb2])

    # -- REINFORCE (policy-gradient) step -----------------------------------

    def reinforce_fit(self, episodes, epochs: int = 30, lr: float = 3e-3,
                      sigma: float = 0.15, l2: float = 1e-4, seed: int = 0):
        """episodes: callable(policy_mean_fn) -> list of (X, actions, returns).

        Gaussian policy: action ~ clip(mean + sigma*eps, 0, 1). Advantage = reward
        minus a running baseline. Reward is supplied by the caller (it MAY use realized
        future returns — that's the training signal, not an observation)."""
        opt = _Adam([self.W1, self.b1, self.W2, self.b2], lr=lr)
        rng = np.random.default_rng(int(seed))
        baseline = 0.0
        for _ in range(int(epochs)):
            X, acts, rews = episodes(self, rng)
            if X is None or len(X) == 0:
                continue
            X = np.atleast_2d(X).astype(np.float64)
            acts = np.asarray(acts, dtype=np.float64).reshape(-1, 1)
            rews = np.asarray(rews, dtype=np.float64).reshape(-1, 1)
            baseline = 0.9 * baseline + 0.1 * float(rews.mean())
            adv = rews - baseline
            adv = adv / (np.std(adv) + 1e-8)
            # forward
            z1 = X @ self.W1 + self.b1
            h = np.tanh(z1)
            z2 = h @ self.W2 + self.b2
            mean = _sigmoid(z2)
            # d logpi/d mean for Gaussian = (a - mean)/sigma^2 ; through sigmoid:
            dmean = (acts - mean) / (sigma ** 2)
            g = -(dmean * adv) * mean * (1 - mean) / X.shape[0]  # ascent -> negate
            gW2 = h.T @ g + l2 * self.W2
            gb2 = g.sum(axis=0)
            dh = (g @ self.W2.T) * (1 - h ** 2)
            gW1 = X.T @ dh + l2 * self.W1
            gb1 = dh.sum(axis=0)
            opt.step([gW1, gb1, gW2, gb2])

    # -- serialization ------------------------------------------------------

    def state_dict(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "n_in": np.array(self.n_in), "n_hidden": np.array(self.n_hidden)}

    def load_state_dict(self, sd: dict) -> None:
        self.W1 = np.asarray(sd["W1"], dtype=np.float64)
        self.b1 = np.asarray(sd["b1"], dtype=np.float64)
        self.W2 = np.asarray(sd["W2"], dtype=np.float64)
        self.b2 = np.asarray(sd["b2"], dtype=np.float64)
        self.n_in = int(np.asarray(sd["n_in"]))
        self.n_hidden = int(np.asarray(sd["n_hidden"]))

    @classmethod
    def from_state_dict(cls, sd: dict) -> "MLP":
        m = cls(int(np.asarray(sd["n_in"])), int(np.asarray(sd["n_hidden"])))
        m.load_state_dict(sd)
        return m


class _Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
