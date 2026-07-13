"""
benchmark.py — Engine throughput measurement.

Times (a) the raw matching engine + Poisson flow with no strategy and
(b) a full simulation with the AS strategy attached, and reports events
processed per second (median of 3 repeats).

"Events" counts every heap entry processed by the simulator loop: flow
arrivals plus (in the full run) strategy cancel/submit actions.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import statistics
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_sim import PoissonFlow, Simulator
from lob_sim.strategy import AvellanedaStoikov

DURATION = 3600.0          # 1-hour session per repeat
REPEATS = 3
# High event rate to get a meaningful sample per run
FLOW_KWARGS = dict(lambda_lim=20.0, lambda_mkt=4.0, lambda_cancel=1.0)


def _count_events(result, with_strategy: bool) -> int:
    # mid_ts gets one point per processed event with a valid mid — the
    # closest cheap proxy for heap entries processed
    return len(result.mid_ts)


def bench(with_strategy: bool) -> tuple[float, int]:
    rates = []
    n_events = 0
    for rep in range(REPEATS):
        flow = PoissonFlow(seed=100 + rep, **FLOW_KWARGS)
        strat = AvellanedaStoikov(T=DURATION) if with_strategy else None
        sim = Simulator(flow=flow, strategy=strat, duration=DURATION,
                        snapshot_interval=0.0)
        t0 = time.perf_counter()
        result = sim.run()
        elapsed = time.perf_counter() - t0
        n_events = _count_events(result, with_strategy)
        rates.append(n_events / elapsed)
    return statistics.median(rates), n_events


def main() -> None:
    rate_raw, n_raw = bench(with_strategy=False)
    print(f"engine + flow (no strategy): {n_raw:,} events/run, "
          f"{rate_raw:,.0f} events/sec (median of {REPEATS})")

    rate_full, n_full = bench(with_strategy=True)
    print(f"full sim (AS strategy):      {n_full:,} events/run, "
          f"{rate_full:,.0f} events/sec (median of {REPEATS})")


if __name__ == "__main__":
    main()
