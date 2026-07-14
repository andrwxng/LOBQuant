"""
latency_sweep.py — Strategy PnL as a function of order-entry latency.

The simulator's `latency` parameter delays every strategy submission and
cancellation by a fixed interval.  This sweep measures how the validated
strategies' edge decays as latency grows, on the held-out validation seeds.

Flow events arrive at ~5.9/s (mean gap ~170 ms), so latencies well below
that barely bind; the interesting range is comparable to and above the
inter-event time.

Usage:
    python scripts/latency_sweep.py
"""

from __future__ import annotations

import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_sim import PoissonFlow, Simulator
from lob_sim.strategy import (
    FairValueAvellanedaStoikov, QueueAwareFairValueAvellanedaStoikov,
)

FLOW_KWARGS = dict(lambda_lim=5.0, lambda_mkt=0.3, lambda_cancel=0.3, max_qty=3)
STRAT_KWARGS = dict(gamma=0.01, alpha=0.005, sigma=0.86, k=0.213, T=3600.0,
                    max_inventory=30, max_half_spread=8)
SIM_KWARGS = dict(duration=3600.0, snapshot_interval=0.0,
                  initial_depth=20, initial_qty_per_level=5)
VALIDATE_SEEDS = list(range(13, 25))

# Capped at 1 s: superseded quotes linger ~2x latency before their delayed
# cancels land, and the flow's cancellation intensity scales with the number
# of resting orders (CST), so multi-second latencies inflate the event rate
# and run times dramatically.  The decay + crossover are visible by 1 s.
LATENCIES = [0.0, 0.05, 0.2, 1.0]
# patience selected on train seeds by scripts/gamma_sweep.py --strategy queue
QUEUE_PATIENCE = 0


def run(strat_cls, latency: float, seed: int, **extra) -> float:
    flow = PoissonFlow(seed=seed, **FLOW_KWARGS)
    strat = strat_cls(**STRAT_KWARGS, **extra)
    sim = Simulator(flow=flow, strategy=strat, latency=latency, **SIM_KWARGS)
    return sim.run().final_pnl


def main() -> None:
    variants = [
        ("Fair-value AS", FairValueAvellanedaStoikov, {}),
        (f"Queue-aware (patience={QUEUE_PATIENCE})",
         QueueAwareFairValueAvellanedaStoikov,
         dict(queue_patience=QUEUE_PATIENCE)),
    ]

    print(f"PnL vs order-entry latency (validation seeds "
          f"{VALIDATE_SEEDS[0]}-{VALIDATE_SEEDS[-1]}, 1h sessions):\n")
    header = "| Latency | " + " | ".join(name for name, _, _ in variants) + " |"
    print(header)
    print("|" + "---------|" * (len(variants) + 1))
    for latency in LATENCIES:
        cells = []
        for _, cls, extra in variants:
            pnls = [run(cls, latency, seed, **extra) for seed in VALIDATE_SEEDS]
            cells.append(f"{statistics.mean(pnls):+,.0f} "
                         f"± {statistics.stdev(pnls):,.0f}")
        label = f"{latency*1000:.0f} ms" if latency < 1 else f"{latency:.0f} s"
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
