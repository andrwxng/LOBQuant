"""
calibrate_k.py — Empirically calibrate the AS fill-intensity decay parameter.

The Avellaneda-Stoikov model assumes a quote at depth δ from the mid is
filled at Poisson rate λ(δ) = A·exp(−k·δ).  The strategy currently assumes
k=1.5.  This script measures realized fill intensity per depth bucket from
instrumented simulations and fits (A, k) by least squares on ln λ(δ).

Method
------
* Run the baseline AS strategy across several γ (which quote at different
  depths) and pool every quote submission.
* For each quote: depth = distance from mid at submission; exposure ends at
  its fill or at the next same-side requote, whichever comes first.
* λ(δ) = total fills at depth δ / total exposure time at depth δ.

Usage:
    python scripts/calibrate_k.py
"""

from __future__ import annotations

import math
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_sim import PoissonFlow, Simulator
from lob_sim.events import Side, SubmitLimit
from lob_sim.strategy import AvellanedaStoikov

FLOW_KWARGS = dict(lambda_lim=5.0, lambda_mkt=0.3, lambda_cancel=0.3, max_qty=3)
STRAT_KWARGS = dict(sigma=0.86, k=1.5, T=3600.0, max_inventory=30,
                    max_half_spread=8)
SIM_KWARGS = dict(duration=3600.0, snapshot_interval=0.0,
                  initial_depth=20, initial_qty_per_level=5)

CALIB_GAMMAS = [0.0005, 0.002, 0.01, 0.05]   # span shallow to deep quoting
CALIB_SEEDS = [1, 2, 3]
MIN_DEPTH, MAX_DEPTH = 1, 10                  # buckets (integer ticks)
MIN_BUCKET_FILLS = 10                         # exclude sparse buckets from fit


class InstrumentedAS(AvellanedaStoikov):
    """Baseline AS that logs (order_id, side, price, submit_ts, mid) per quote."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.quote_log: list = []

    def on_book_update(self, book, ts):
        actions = super().on_book_update(book, ts)
        mid = book.mid()
        if mid is not None:
            for a in actions:
                if isinstance(a, SubmitLimit):
                    self.quote_log.append(
                        (a.order_id, a.side, a.price_ticks, ts, mid))
        return actions


def collect(gamma: float, seed: int, time_at: dict, fills_at: dict) -> None:
    flow = PoissonFlow(seed=seed, **FLOW_KWARGS)
    strat = InstrumentedAS(gamma=gamma, **STRAT_KWARGS)
    result = Simulator(flow=flow, strategy=strat, **SIM_KWARGS).run()

    first_fill_ts = {}
    for t in result.trades:
        if t.passive_order_id not in first_fill_ts:
            first_fill_ts[t.passive_order_id] = t.timestamp

    # Exposure of each quote ends at the next same-side submission
    by_side = {Side.BID: [], Side.ASK: []}
    for entry in strat.quote_log:
        by_side[entry[1]].append(entry)

    duration = SIM_KWARGS['duration']
    for side, entries in by_side.items():
        for i, (oid, _side, price, submit_ts, mid) in enumerate(entries):
            end_ts = entries[i + 1][3] if i + 1 < len(entries) else duration
            fill_ts = first_fill_ts.get(oid)

            # Signed depth: positive = passive distance behind the mid.
            # Excludes marketable submissions (depth <= 0), which the
            # passive fill-intensity model does not describe.
            depth = mid - price if side == Side.BID else price - mid
            bucket = round(depth)
            if not (MIN_DEPTH <= bucket <= MAX_DEPTH):
                continue

            if fill_ts is not None and fill_ts <= end_ts:
                exposure = fill_ts - submit_ts
                fills_at[bucket] += 1
            else:
                exposure = end_ts - submit_ts
            time_at[bucket] += max(0.0, exposure)


def main() -> None:
    time_at: dict = defaultdict(float)
    fills_at: dict = defaultdict(int)
    for gamma in CALIB_GAMMAS:
        for seed in CALIB_SEEDS:
            collect(gamma, seed, time_at, fills_at)

    print("Fill intensity by quote depth "
          f"({len(CALIB_GAMMAS)} γ × {len(CALIB_SEEDS)} seeds × 1h):\n")
    print("| depth (ticks) | fills | exposure (s) | λ (fills/s) |")
    print("|---------------|-------|--------------|-------------|")
    points = []
    for d in sorted(time_at):
        lam = fills_at[d] / time_at[d] if time_at[d] > 0 else float('nan')
        print(f"| {d} | {fills_at[d]} | {time_at[d]:,.0f} | {lam:.4f} |")
        if fills_at[d] >= MIN_BUCKET_FILLS:
            points.append((d, math.log(lam)))

    if len(points) < 2:
        print("\nNot enough populated buckets to fit.")
        return

    # Least-squares fit: ln λ = ln A − k·δ
    n = len(points)
    sx = sum(d for d, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(d * d for d, _ in points)
    sxy = sum(d * y for d, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n

    k_hat = -slope
    a_hat = math.exp(intercept)
    print(f"\nFit over {n} buckets with ≥{MIN_BUCKET_FILLS} fills:")
    print(f"  k̂ = {k_hat:.3f}   (strategy currently assumes k = "
          f"{STRAT_KWARGS['k']})")
    print(f"  Â = {a_hat:.4f} fills/s at the mid")


if __name__ == "__main__":
    main()
