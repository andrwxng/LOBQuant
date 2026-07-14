"""
queue_analysis.py — Empirical fill probability as a function of queue position.

Cont & de Larrard (2013) model fill likelihood as driven by queue dynamics:
an order's chance of executing depends on how much quantity is ahead of it
in the FIFO queue.  This script measures that curve for the fair-value
strategy's quotes: each quote's queue position (qty ahead) is recorded at
its first sighting after landing, and the outcome is whether it was ever
passively filled.

The resulting monotone-decreasing curve is the empirical justification for
queue-aware requoting (QueueAwareFairValueAvellanedaStoikov): cancelling a
resting order forfeits its place in this curve.

Usage:
    python scripts/queue_analysis.py
"""

from __future__ import annotations

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_sim import PoissonFlow, Simulator
from lob_sim.strategy import FairValueAvellanedaStoikov

FLOW_KWARGS = dict(lambda_lim=5.0, lambda_mkt=0.3, lambda_cancel=0.3, max_qty=3)
STRAT_KWARGS = dict(gamma=0.01, alpha=0.005, sigma=0.86, k=0.213, T=3600.0,
                    max_inventory=30, max_half_spread=8)
SIM_KWARGS = dict(duration=3600.0, snapshot_interval=0.0,
                  initial_depth=20, initial_qty_per_level=5)
SEEDS = list(range(1, 13))

BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 10**9)]


def _bucket(qty_ahead: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= qty_ahead <= hi:
            return i
    return len(BUCKETS) - 1


class QueueInstrumentedFV(FairValueAvellanedaStoikov):
    """Records each live quote's (orders_ahead, qty_ahead) at first sighting.

    Quotes filled before the next book update are never sighted and are
    excluded — a mild censoring of the fastest fills, noted in the README.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.qpos_log: dict = {}    # oid -> qty_ahead at first sighting

    def on_book_update(self, book, ts):
        for oid in (self._bid_id, self._ask_id):
            if oid is not None and oid not in self.qpos_log:
                qp = book.queue_position(oid)
                if qp is not None:
                    self.qpos_log[oid] = qp[1]
        return super().on_book_update(book, ts)


def main() -> None:
    sighted = defaultdict(int)
    filled = defaultdict(int)

    for seed in SEEDS:
        flow = PoissonFlow(seed=seed, **FLOW_KWARGS)
        strat = QueueInstrumentedFV(**STRAT_KWARGS)
        result = Simulator(flow=flow, strategy=strat, **SIM_KWARGS).run()

        filled_ids = {t.passive_order_id for t in result.trades
                      if t.passive_order_id in strat.all_submitted_ids}
        for oid, qty_ahead in strat.qpos_log.items():
            b = _bucket(qty_ahead)
            sighted[b] += 1
            if oid in filled_ids:
                filled[b] += 1

    print(f"Fill probability vs queue position at first sighting "
          f"({len(SEEDS)} seeds × 1h, fair-value strategy):\n")
    print("| qty ahead in queue | quotes | filled | fill probability |")
    print("|--------------------|--------|--------|------------------|")
    for i, (lo, hi) in enumerate(BUCKETS):
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
        n = sighted[i]
        p = filled[i] / n if n else float('nan')
        print(f"| {label} | {n} | {filled[i]} | {p:.1%} |")


if __name__ == "__main__":
    main()
