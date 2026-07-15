"""Fresh, maze-runner-inspired grammar for the RAM intraday robot.

Three phases (no SPX/TSLib reuse):
  perception : belief-map genes (small windows generalize best — the maze finding)
  strategy   : robot family (band/vwap/momentum/nn/morpheus) + its hyperparams
  trading    : exposure/threshold/session/gap genes

Transaction cost is intentionally NOT a gene (it is a fixed market constant applied
in fitness) so evolution cannot cheat by shrinking costs. Long/flat only.
"""
from __future__ import annotations

import numpy as np

from zion_ge.core.gene import ModuleGene, PipelineGenotype
from zion_ge.core.types import PortKind, PortSpec, PortType
from zion_ge.grammar.schema import (
    GrammarSpec,
    ModuleSpec,
    ParamSpec,
    PhaseSpec,
    RepairRule,
)

_SCALAR = PortType(kind=PortKind.SCALAR)
_BOOL = ["0", "1"]


# ---------------------------------------------------------------------------
# Module specs
# ---------------------------------------------------------------------------

def _perception_module() -> ModuleSpec:
    return ModuleSpec(
        name="perception", type_id=0,
        input_ports=[], output_ports=[PortSpec(name="belief", port_type=_SCALAR)],
        params=[
            ParamSpec(name="obs_lookback", dtype="int", lower=5, upper=40, default=15),
            ParamSpec(name="memory_cap", dtype="int", lower=30, upper=300, default=120),
            ParamSpec(name="ladder_bins", dtype="int", lower=8, upper=40, default=20),
            ParamSpec(name="vol_window", dtype="int", lower=10, upper=60, default=30),
        ],
        tags=frozenset({"perception"}),
    )


def _strategy_modules() -> list[ModuleSpec]:
    inp = [PortSpec(name="belief", port_type=_SCALAR)]
    out = [PortSpec(name="exposure", port_type=_SCALAR)]
    band = ModuleSpec(
        name="band_reverter", type_id=1, input_ports=inp, output_ports=out,
        params=[ParamSpec(name="band_gain", dtype="float", lower=0.5, upper=1.5, default=1.0)],
        tags=frozenset({"strategy", "rule"}),
    )
    vwap = ModuleSpec(
        name="vwap_reverter", type_id=2, input_ports=inp, output_ports=out,
        params=[ParamSpec(name="vwap_k", dtype="float", lower=0.2, upper=3.0, default=1.0)],
        tags=frozenset({"strategy", "rule"}),
    )
    mom = ModuleSpec(
        name="momentum", type_id=3, input_ports=inp, output_ports=out,
        params=[ParamSpec(name="mom_k", dtype="float", lower=0.2, upper=3.0, default=1.0)],
        tags=frozenset({"strategy", "rule"}),
    )
    nn = ModuleSpec(
        name="nn_reverter", type_id=4, input_ports=inp, output_ports=out,
        params=[
            ParamSpec(name="nn_hidden", dtype="int", lower=4, upper=16, default=8),
            ParamSpec(name="nn_seed", dtype="int", lower=0, upper=9999, default=0),
        ],
        tags=frozenset({"strategy", "nn"}),
    )
    morpheus = ModuleSpec(
        name="morpheus_trader", type_id=5, input_ports=inp, output_ports=out,
        params=[
            ParamSpec(name="ppo_hidden", dtype="int", lower=4, upper=16, default=8),
            # RL-safe LR band (the maze lesson: PPO LR must be 1e-4..1e-3)
            ParamSpec(name="ppo_lr", dtype="float", lower=1e-4, upper=1e-3, default=3e-4),
            ParamSpec(name="ppo_steps", dtype="int", lower=10, upper=50, default=30),
            ParamSpec(name="nn_seed", dtype="int", lower=0, upper=9999, default=0),
        ],
        tags=frozenset({"strategy", "rl"}),
    )
    return [band, vwap, mom, nn, morpheus]


def _trading_module() -> ModuleSpec:
    return ModuleSpec(
        name="trading", type_id=10,
        input_ports=[PortSpec(name="exposure", port_type=_SCALAR)],
        output_ports=[PortSpec(name="orders", port_type=_SCALAR)],
        params=[
            ParamSpec(name="trade_threshold", dtype="float", lower=0.02, upper=0.4, default=0.1),
            ParamSpec(name="max_exposure", dtype="float", lower=0.3, upper=1.0, default=1.0),
            ParamSpec(name="trade_premarket", dtype="categorical", choices=_BOOL, default=0),
            ParamSpec(name="trade_afterhours", dtype="categorical", choices=_BOOL, default=0),
            ParamSpec(name="hold_through_gap", dtype="categorical", choices=_BOOL, default=0),
            ParamSpec(name="gap_fade_bias", dtype="float", lower=0.0, upper=1.0, default=0.0),
            ParamSpec(name="seed", dtype="int", lower=1, upper=9999, default=42),
        ],
        tags=frozenset({"trading"}),
    )


# ---------------------------------------------------------------------------
# Grammar builder
# ---------------------------------------------------------------------------

def build_grammar() -> GrammarSpec:
    perception = _perception_module()
    strat = _strategy_modules()
    trading = _trading_module()
    phases = [
        PhaseSpec(name="perception", allowed_modules=["perception"], min_length=1, max_length=1),
        PhaseSpec(name="strategy", allowed_modules=[m.name for m in strat], min_length=1, max_length=1),
        PhaseSpec(name="trading", allowed_modules=["trading"], min_length=1, max_length=1),
    ]
    return GrammarSpec(
        name="ram_grammar", version="1.0", phases=phases,
        modules=[perception] + strat + [trading],
        repair_rules=[RepairRule(condition="bounds_violation", action="clamp",
                                 params={"apply_to": "all"})],
        metadata={
            "description": "Maze-inspired RAM intraday belief-map robot: family + belief + trading genes",
            "target": "RAM",
            "objectives": ["oos_return", "ulcer", "robustness_gap"],
        },
    )


# ---------------------------------------------------------------------------
# Phenotype decoder (same pattern as spx_simple._decode_param)
# ---------------------------------------------------------------------------

def _decode_param(gene: ModuleGene, i: int, spec: ParamSpec):
    if i >= len(gene.params):
        if spec.dtype == "categorical" and spec.choices:
            return spec.choices[int(spec.default or 0)]
        return spec.default
    raw = float(gene.params[i])
    if spec.dtype == "categorical" and spec.choices:
        idx = int(np.clip(np.floor(raw * len(spec.choices)), 0, len(spec.choices) - 1))
        return spec.choices[idx]
    if spec.dtype == "int":
        lo, hi = spec.lower or 0, spec.upper or 100
        return int(np.clip(round(lo + raw * (hi - lo)), lo, hi))
    if spec.dtype == "float":
        lo, hi = spec.lower or 0.0, spec.upper or 1.0
        return float(np.clip(lo + raw * (hi - lo), lo, hi))
    return raw


def _decode_gene(gene: ModuleGene, spec: ModuleSpec) -> dict:
    return {p.name: _decode_param(gene, i, p) for i, p in enumerate(spec.params)}


def decode_phenotype(individual) -> dict:
    """EvoIndividual -> flat phenotype dict consumed by RamRobot."""
    grammar = build_grammar()
    genotype: PipelineGenotype = individual.genotype
    ph: dict = {}

    pg = genotype.phases.get("perception", [])
    if pg:
        ph.update(_decode_gene(pg[0], grammar.get_module_by_name("perception")))
    else:
        ph.update({"obs_lookback": 15, "memory_cap": 120, "ladder_bins": 20, "vol_window": 30})

    sg = genotype.phases.get("strategy", [])
    strat_mods = grammar.get_modules_for_phase("strategy")
    if sg:
        gene = sg[0]
        tid = gene.decode_type_id(len(strat_mods))
        mod = strat_mods[tid]
        ph["family"] = mod.name
        ph.update(_decode_gene(gene, mod))
    else:
        ph["family"] = "band_reverter"
        ph["band_gain"] = 1.0

    tg = genotype.phases.get("trading", [])
    if tg:
        tp = _decode_gene(tg[0], grammar.get_module_by_name("trading"))
        ph.update(tp)
    else:
        ph.update({"trade_threshold": 0.1, "max_exposure": 1.0, "trade_premarket": "0",
                   "trade_afterhours": "0", "hold_through_gap": "0", "gap_fade_bias": 0.0, "seed": 42})

    # normalize categoricals to bools
    for k in ("trade_premarket", "trade_afterhours", "hold_through_gap"):
        ph[k] = str(ph.get(k, "0")) == "1"
    # nn_seed alias for morpheus (grammar names it nn_seed there too)
    ph.setdefault("nn_seed", ph.get("seed", 0))
    return ph
