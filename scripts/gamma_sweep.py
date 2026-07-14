"""
gamma_sweep.py — Reproducible parameter sweeps for the README results tables.

Two modes, both fully deterministic (identical output on every run):

  python scripts/gamma_sweep.py
      Baseline Avellaneda-Stoikov γ-sweep: 6 γ × 12 seeds × 1-hour sessions.

  python scripts/gamma_sweep.py --strategy fv
      Fair-value anchored variant (FairValueAvellanedaStoikov):
      1. Grid-search (α, γ) on TRAIN seeds 1-12.
      2. Select the cell with the best mean/std PnL across seeds.
      3. Report the chosen cell on held-out VALIDATION seeds 13-24, next to
         the plain-AS baseline at the same γ.  Only validation numbers
         belong in the README.
      4. Robustness: re-run the chosen cell on validation seeds under
         perturbed flow regimes.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_sim import PoissonFlow, Simulator, metrics
from lob_sim.strategy import (
    AvellanedaStoikov, FairValueAvellanedaStoikov,
    ImbalanceFairValueAvellanedaStoikov,
    QueueAwareFairValueAvellanedaStoikov,
)

# Flow / strategy config mirrors the original README sweep so the tables
# stay comparable
FLOW_KWARGS = dict(lambda_lim=5.0, lambda_mkt=0.3, lambda_cancel=0.3, max_qty=3)
STRAT_KWARGS = dict(sigma=0.86, k=1.5, T=3600.0, max_inventory=30,
                    max_half_spread=8)
SIM_KWARGS = dict(duration=3600.0, snapshot_interval=0.0,
                  initial_depth=20, initial_qty_per_level=5)

GAMMAS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
SEEDS = list(range(1, 13))
BUCKET_SECONDS = 60.0

# Fair-value mode: train/validate split and search grid
TRAIN_SEEDS = list(range(1, 13))
VALIDATE_SEEDS = list(range(13, 25))
FV_ALPHAS = [0.002, 0.005, 0.01, 0.02]
FV_GAMMAS = [0.002, 0.005, 0.01]

# Fill-intensity decay measured by scripts/calibrate_k.py (independent of
# any PnL tuning; see README).  The strategy default assumes k=1.5.
K_CALIBRATED = 0.213

# Realistic maker-taker venue economics, in ticks per contract
MAKER_REBATE = -0.3
TAKER_FEE = 1.0

# Imbalance mode: β grid searched on train seeds on top of the fv-selected
# configuration (γ=0.01, α=0.005, calibrated k)
IMB_BETAS = [0.5, 1.0, 2.0, 4.0]
IMB_BASE = dict(alpha=0.005, k=K_CALIBRATED)
IMB_GAMMA = 0.01

# Queue mode: requote-patience grid on the same base configuration
QUEUE_PATIENCES = [0, 1, 2, 3]


def run_one(
    gamma: float,
    seed: int,
    strat_cls=AvellanedaStoikov,
    flow_overrides: dict | None = None,
    **strat_extra,
) -> dict:
    flow_kwargs = dict(FLOW_KWARGS)
    if flow_overrides:
        flow_kwargs.update(flow_overrides)
    flow = PoissonFlow(seed=seed, **flow_kwargs)
    # strat_extra may override STRAT_KWARGS entries (e.g. calibrated k)
    strat = strat_cls(gamma=gamma, **{**STRAT_KWARGS, **strat_extra})
    sim = Simulator(flow=flow, strategy=strat, **SIM_KWARGS)
    result = sim.run()
    return metrics.summary_report(
        result, strat.all_submitted_ids,
        total_quotes=max(1, strat.n_quotes_submitted),
        bucket_seconds=BUCKET_SECONDS,
    )


def _mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float('nan')


def _std(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.stdev(xs) if len(xs) > 1 else float('nan')


def _row(reports: list) -> dict:
    return {
        'pnl_mean': _mean([r['final_pnl_ticks'] for r in reports]),
        'pnl_std': _std([r['final_pnl_ticks'] for r in reports]),
        'sharpe': _mean([r['sharpe'] for r in reports]),
        'max_dd': _mean([r['max_drawdown'] for r in reports]),
        'inv_std': _mean([r['inv_std'] for r in reports]),
        'fill_rate': _mean([r['fill_rate'] for r in reports]),
        'adv_sel': _mean([r['overall'] for r in reports]),
    }


def sweep_as() -> None:
    t_start = time.time()
    print(f"γ-sweep: {len(GAMMAS)} γ × {len(SEEDS)} seeds × "
          f"{SIM_KWARGS['duration']:.0f}s sessions\n")

    rows = []
    for gamma in GAMMAS:
        r = _row([run_one(gamma, seed) for seed in SEEDS])
        r['gamma'] = gamma
        rows.append(r)
        print(f"  γ={gamma:<7g} pnl={r['pnl_mean']:>8.0f}±{r['pnl_std']:<8.0f}"
              f" sharpe={r['sharpe']:>6.2f} dd={r['max_dd']:>6.0f}"
              f" inv_std={r['inv_std']:>5.2f} fill={r['fill_rate']:.1%}"
              f" adv={r['adv_sel']:>6.2f}")

    print(f"\ndone in {time.time() - t_start:.1f}s\n")
    print("Markdown table for README:\n")
    print("| γ | Mean PnL | PnL std | Sharpe | Max DD | Inv std | Fill rate | Adv sel |")
    print("|---|----------|---------|--------|--------|---------|-----------|---------|")
    for r in rows:
        print(f"| {r['gamma']:g} | {r['pnl_mean']:+,.0f} | {r['pnl_std']:,.0f} "
              f"| {r['sharpe']:.2f} | {r['max_dd']:,.0f} | {r['inv_std']:.2f} "
              f"| {r['fill_rate']:.1%} | {r['adv_sel']:+.2f} |")


def sweep_fv() -> None:
    t_start = time.time()
    print(f"Fair-value grid search: {len(FV_ALPHAS)}α × {len(FV_GAMMAS)}γ "
          f"on TRAIN seeds {TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}\n")

    best = None
    for alpha in FV_ALPHAS:
        for gamma in FV_GAMMAS:
            pnls = [run_one(gamma, seed, FairValueAvellanedaStoikov,
                            alpha=alpha)['final_pnl_ticks']
                    for seed in TRAIN_SEEDS]
            mean, std = _mean(pnls), _std(pnls)
            score = mean / std if std and std > 0 else float('-inf')
            print(f"  α={alpha:<6g} γ={gamma:<6g} pnl={mean:>7.0f}±{std:<7.0f}"
                  f" score={score:>6.2f}")
            if best is None or score > best['score']:
                best = dict(alpha=alpha, gamma=gamma, score=score)

    alpha, gamma = best['alpha'], best['gamma']
    print(f"\nSelected on train seeds: α={alpha:g}, γ={gamma:g} "
          f"(score={best['score']:.2f})")
    print(f"\nValidation (held-out seeds {VALIDATE_SEEDS[0]}-{VALIDATE_SEEDS[-1]}):\n")

    val_fv = _row([run_one(gamma, seed, FairValueAvellanedaStoikov, alpha=alpha)
                   for seed in VALIDATE_SEEDS])
    val_as = _row([run_one(gamma, seed) for seed in VALIDATE_SEEDS])
    val_fv_k = _row([run_one(gamma, seed, FairValueAvellanedaStoikov,
                             alpha=alpha, k=K_CALIBRATED)
                     for seed in VALIDATE_SEEDS])
    val_fv_fees = _row([run_one(gamma, seed, FairValueAvellanedaStoikov,
                                alpha=alpha, k=K_CALIBRATED,
                                maker_fee=MAKER_REBATE, taker_fee=TAKER_FEE)
                        for seed in VALIDATE_SEEDS])

    print("| Strategy | Mean PnL | PnL std | Sharpe | Max DD | Inv std | Fill rate | Adv sel |")
    print("|----------|----------|---------|--------|--------|---------|-----------|---------|")
    for label, r in ((f"AS baseline (γ={gamma:g})", val_as),
                     (f"Fair-value AS (γ={gamma:g}, α={alpha:g})", val_fv),
                     (f"Fair-value AS + calibrated k={K_CALIBRATED}", val_fv_k),
                     (f"… + maker rebate {-MAKER_REBATE:g}, taker fee {TAKER_FEE:g}",
                      val_fv_fees)):
        print(f"| {label} | {r['pnl_mean']:+,.0f} | {r['pnl_std']:,.0f} "
              f"| {r['sharpe']:.2f} | {r['max_dd']:,.0f} | {r['inv_std']:.2f} "
              f"| {r['fill_rate']:.1%} | {r['adv_sel']:+.2f} |")

    print("\nRobustness of the selected cell (validation seeds, perturbed flow):\n")
    regimes = [
        ("baseline flow", {}),
        ("2x market-order rate", dict(lambda_mkt=0.6)),
        ("2x reference volatility", dict(sigma_ref=1.0)),
        ("larger orders (max_qty=5)", dict(max_qty=5)),
    ]
    print("| Flow regime | Mean PnL | PnL std |")
    print("|-------------|----------|---------|")
    for label, overrides in regimes:
        pnls = [run_one(gamma, seed, FairValueAvellanedaStoikov, alpha=alpha,
                        flow_overrides=overrides)['final_pnl_ticks']
                for seed in VALIDATE_SEEDS]
        print(f"| {label} | {_mean(pnls):+,.0f} | {_std(pnls):,.0f} |")

    print(f"\ndone in {time.time() - t_start:.1f}s")


def sweep_imb() -> None:
    """β grid for the imbalance skew, layered on the fv-selected config."""
    t_start = time.time()
    print(f"Imbalance β search on TRAIN seeds (base: γ={IMB_GAMMA:g}, "
          f"α={IMB_BASE['alpha']:g}, k={IMB_BASE['k']:g})\n")

    # β=0 benchmark is the plain fair-value strategy with the same base
    bench = [run_one(IMB_GAMMA, seed, FairValueAvellanedaStoikov, **IMB_BASE)
             ['final_pnl_ticks'] for seed in TRAIN_SEEDS]
    print(f"  β=0 (benchmark)  pnl={_mean(bench):>7.0f}±{_std(bench):<7.0f}")

    best = dict(beta=0.0, score=_mean(bench) / _std(bench))
    for beta in IMB_BETAS:
        pnls = [run_one(IMB_GAMMA, seed, ImbalanceFairValueAvellanedaStoikov,
                        beta=beta, **IMB_BASE)['final_pnl_ticks']
                for seed in TRAIN_SEEDS]
        mean, std = _mean(pnls), _std(pnls)
        score = mean / std if std and std > 0 else float('-inf')
        print(f"  β={beta:<6g}         pnl={mean:>7.0f}±{std:<7.0f}"
              f" score={score:>6.2f}")
        if score > best['score']:
            best = dict(beta=beta, score=score)

    print(f"\nSelected on train seeds: β={best['beta']:g}")
    print(f"\nValidation (held-out seeds "
          f"{VALIDATE_SEEDS[0]}-{VALIDATE_SEEDS[-1]}):\n")
    val_fv = _row([run_one(IMB_GAMMA, seed, FairValueAvellanedaStoikov,
                           **IMB_BASE) for seed in VALIDATE_SEEDS])
    rows = [("Fair-value AS (β=0)", val_fv)]
    if best['beta'] > 0:
        val_imb = _row([run_one(IMB_GAMMA, seed,
                                ImbalanceFairValueAvellanedaStoikov,
                                beta=best['beta'], **IMB_BASE)
                        for seed in VALIDATE_SEEDS])
        rows.append((f"+ imbalance skew (β={best['beta']:g})", val_imb))

    print("| Strategy | Mean PnL | PnL std | Max DD | Fill rate | Adv sel |")
    print("|----------|----------|---------|--------|-----------|---------|")
    for label, r in rows:
        print(f"| {label} | {r['pnl_mean']:+,.0f} | {r['pnl_std']:,.0f} "
              f"| {r['max_dd']:,.0f} | {r['fill_rate']:.1%} "
              f"| {r['adv_sel']:+.2f} |")
    print(f"\ndone in {time.time() - t_start:.1f}s")


def sweep_queue() -> None:
    """Requote-patience grid for queue-aware quoting on the fv config."""
    t_start = time.time()
    print(f"Queue-patience search on TRAIN seeds (base: γ={IMB_GAMMA:g}, "
          f"α={IMB_BASE['alpha']:g}, k={IMB_BASE['k']:g})\n")

    bench = [run_one(IMB_GAMMA, seed, FairValueAvellanedaStoikov, **IMB_BASE)
             ['final_pnl_ticks'] for seed in TRAIN_SEEDS]
    best = dict(patience=None, score=_mean(bench) / _std(bench))
    print(f"  base FV (both-side requote)  pnl={_mean(bench):>7.0f}"
          f"±{_std(bench):<7.0f} score={best['score']:>6.2f}")

    for patience in QUEUE_PATIENCES:
        pnls = [run_one(IMB_GAMMA, seed, QueueAwareFairValueAvellanedaStoikov,
                        queue_patience=patience, **IMB_BASE)['final_pnl_ticks']
                for seed in TRAIN_SEEDS]
        mean, std = _mean(pnls), _std(pnls)
        score = mean / std if std and std > 0 else float('-inf')
        print(f"  patience={patience}                   pnl={mean:>7.0f}"
              f"±{std:<7.0f} score={score:>6.2f}")
        if score > best['score']:
            best = dict(patience=patience, score=score)

    if best['patience'] is None:
        print("\nNo patience level beat the base FV strategy on train seeds.")
        return

    patience = best['patience']
    print(f"\nSelected on train seeds: patience={patience}")
    print(f"\nValidation (held-out seeds "
          f"{VALIDATE_SEEDS[0]}-{VALIDATE_SEEDS[-1]}):\n")
    val_fv = _row([run_one(IMB_GAMMA, seed, FairValueAvellanedaStoikov,
                           **IMB_BASE) for seed in VALIDATE_SEEDS])
    val_q = _row([run_one(IMB_GAMMA, seed, QueueAwareFairValueAvellanedaStoikov,
                          queue_patience=patience, **IMB_BASE)
                  for seed in VALIDATE_SEEDS])
    print("| Strategy | Mean PnL | PnL std | Max DD | Fill rate | Adv sel |")
    print("|----------|----------|---------|--------|-----------|---------|")
    for label, r in (("Fair-value AS (both-side requote)", val_fv),
                     (f"Queue-aware (patience={patience})", val_q)):
        print(f"| {label} | {r['pnl_mean']:+,.0f} | {r['pnl_std']:,.0f} "
              f"| {r['max_dd']:,.0f} | {r['fill_rate']:.1%} "
              f"| {r['adv_sel']:+.2f} |")
    print(f"\ndone in {time.time() - t_start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=['as', 'fv', 'imb', 'queue'],
                        default='as',
                        help="'as' = baseline γ-sweep, 'fv' = fair-value "
                             "grid search + held-out validation, 'imb' = "
                             "imbalance-skew β search on the fv config, "
                             "'queue' = requote-patience search")
    args = parser.parse_args()
    if args.strategy == 'as':
        sweep_as()
    elif args.strategy == 'fv':
        sweep_fv()
    elif args.strategy == 'imb':
        sweep_imb()
    else:
        sweep_queue()


if __name__ == "__main__":
    main()
