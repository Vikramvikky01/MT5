#!/usr/bin/env python
"""Leak check: does your strategy's edge survive the removal of the edge?

Run a strategy on synthetic data WITH injected regime drift, then on a
driftless random walk where no edge exists by construction. A correct
backtester shows a large drop. If expectancy stays high on the driftless
control, you have a look-ahead bug or a cost model that is too kind — the
strategy is reading the future, not predicting it.

    python scripts/leak_check.py
    python scripts/leak_check.py --strategy donchian_breakout --seeds 6

Interpretation guide:
    drift >> control, control ~= 0     -> healthy, no leak
    control also strongly positive     -> LEAK. Stop and find it.
    both negative                      -> costs dominate; fine, just no edge.

Noise floor: with N trades, the standard error on mean R is roughly
1.2/sqrt(N). With 70 trades that is +/-0.14R, so treat anything inside
+/-0.15R on the control as zero.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantbot.broker.paper_broker import CostModel
from quantbot.core.config import demo_spec, load_config, synthetic_bars
from quantbot.core.risk import RiskLimits, Sizer
from quantbot.engine.backtester import Backtester
from quantbot.strategy.base import build_strategy


def run_set(strat_cfg: dict, trend_strength: float, seeds: list[int],
            bars_n: int) -> tuple[float, list[float], list[int]]:
    exps, counts = [], []
    for sd in seeds:
        bars = synthetic_bars(bars_n, seed=sd, trend_strength=trend_strength)
        strat = build_strategy(strat_cfg)
        strat.symbols = ["EURUSD"]
        res = Backtester(
            {"EURUSD": bars}, {"EURUSD": demo_spec("EURUSD", "fx5")}, [strat],
            sizer=Sizer(risk_per_trade=0.005),
            limits=RiskLimits(trading_hours_utc=None),
            costs=CostModel(slippage_points=2.0,
                            spread_points={"default": 12},
                            commission_per_lot={"default": 7.0}),
            starting_balance=10_000,
        ).run()
        n = res.stats.get("trades", 0)
        exps.append(res.stats.get("expectancy_r", 0.0) if n else 0.0)
        counts.append(n)
    return float(np.mean(exps)), exps, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--strategy", help="instance name from config; default = first")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bars", type=int, default=5000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    cfg = load_config(args.config)
    enabled = [s for s in cfg["strategies"] if s.get("enabled", True)]
    if args.strategy:
        enabled = [s for s in enabled
                   if args.strategy in (s.get("instance"), s.get("name"))]
    if not enabled:
        print("no matching strategy", file=sys.stderr)
        return 1
    sc = enabled[0]

    seeds = list(range(201, 201 + args.seeds))
    print(f"strategy: {sc.get('instance', sc['name'])}  seeds={len(seeds)}  "
          f"bars={args.bars}\n")

    drift_mean, drift_all, drift_n = run_set(sc, 0.35, seeds, args.bars)
    ctrl_mean, ctrl_all, ctrl_n = run_set(sc, 0.00, seeds, args.bars)

    print(f"{'WITH regime drift':<34} {drift_mean:+.4f}R  "
          f"{[round(x, 3) for x in drift_all]}  trades={drift_n}")
    print(f"{'DRIFTLESS control':<34} {ctrl_mean:+.4f}R  "
          f"{[round(x, 3) for x in ctrl_all]}  trades={ctrl_n}")

    total_ctrl = sum(ctrl_n)
    noise = 1.2 / np.sqrt(max(total_ctrl, 1))
    print(f"\ncontrol noise floor ~ +/-{noise:.3f}R ({total_ctrl} trades total)")

    if ctrl_mean > 3 * noise and ctrl_mean > 0.15:
        print("\n*** SUSPECT LEAK: the control has no edge to find, yet the "
              "strategy profits on it. Check for negative shifts, exits priced "
              "before the decision bar, or unrealistic costs. ***")
        return 2
    print("\nHealthy: edge scales with the injected drift and collapses on the "
          "control. No evidence of look-ahead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
