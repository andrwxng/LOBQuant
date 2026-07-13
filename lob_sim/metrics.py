"""
metrics.py — Performance analytics for market-making simulations.

All functions accept raw time-series data produced by Simulator.run()
(lists of (timestamp, value) tuples) plus the trades list.

Units
-----
Prices / PnL values are in ticks unless noted otherwise.
Annualisation factor: 252 trading days × 23,400 seconds/day = 5,896,800 s/yr.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .events import Side, Trade


SECONDS_PER_YEAR = 252 * 23_400   # ~5.9 million


# ── PnL helpers ──────────────────────────────────────────────────────────────

def terminal_pnl(
    cash: float,
    inventory: int,
    final_mid: Optional[float],
) -> float:
    """Cash + liquidation value at mid."""
    if final_mid is None:
        return cash
    return cash + inventory * final_mid


def pnl_series(
    pnl_ts: List[Tuple[float, float]],
    bucket_seconds: float = 60.0,
) -> Tuple[List[float], List[float]]:
    """
    Resample PnL time series into fixed-size buckets.
    Returns (times, pnl_values) aligned to bucket boundaries.
    """
    if not pnl_ts:
        return [], []

    t_start = pnl_ts[0][0]
    t_end = pnl_ts[-1][0]
    n_buckets = max(1, int((t_end - t_start) / bucket_seconds) + 1)

    bucket_pnl: Dict[int, float] = {}
    for t, pnl in pnl_ts:
        b = int((t - t_start) / bucket_seconds)
        bucket_pnl[b] = pnl   # last value in bucket wins

    times = []
    values = []
    last_pnl = 0.0
    for b in range(n_buckets):
        times.append(t_start + b * bucket_seconds)
        last_pnl = bucket_pnl.get(b, last_pnl)
        values.append(last_pnl)

    return times, values


def pnl_increments(pnl_values: List[float]) -> List[float]:
    """First-difference of PnL series."""
    return [pnl_values[i] - pnl_values[i - 1] for i in range(1, len(pnl_values))]


# ── Risk metrics ─────────────────────────────────────────────────────────────

def sharpe_ratio(
    pnl_ts: List[Tuple[float, float]],
    bucket_seconds: float = 60.0,
    annualise: bool = True,
) -> float:
    """
    Annualised Sharpe ratio of bucketed PnL increments.
    Returns NaN if insufficient data or zero variance.
    """
    _, values = pnl_series(pnl_ts, bucket_seconds)
    incs = pnl_increments(values)
    if len(incs) < 2:
        return float('nan')
    mu = statistics.mean(incs)
    sigma = statistics.stdev(incs)
    if sigma == 0:
        return float('nan')
    sr = mu / sigma
    if annualise:
        buckets_per_year = SECONDS_PER_YEAR / bucket_seconds
        sr *= math.sqrt(buckets_per_year)
    return sr


def max_drawdown(pnl_ts: List[Tuple[float, float]]) -> float:
    """
    Maximum peak-to-trough drawdown (in ticks).
    Returns a positive number representing the magnitude of the drawdown.
    """
    if not pnl_ts:
        return 0.0
    values = [v for _, v in pnl_ts]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def volatility(
    pnl_ts: List[Tuple[float, float]],
    bucket_seconds: float = 60.0,
    annualise: bool = True,
) -> float:
    """Standard deviation of bucketed PnL increments, annualised."""
    _, values = pnl_series(pnl_ts, bucket_seconds)
    incs = pnl_increments(values)
    if len(incs) < 2:
        return float('nan')
    sigma = statistics.stdev(incs)
    if annualise:
        buckets_per_year = SECONDS_PER_YEAR / bucket_seconds
        sigma *= math.sqrt(buckets_per_year)
    return sigma


# ── Fill rate ─────────────────────────────────────────────────────────────────

def fill_rate(
    strategy_order_ids: set,
    trades: List[Trade],
    total_quotes_submitted: int,
) -> float:
    """
    Fraction of submitted quotes that received at least one fill.
    fill_rate = unique_filled_order_ids / total_quotes_submitted
    """
    if total_quotes_submitted == 0:
        return float('nan')
    filled_ids = {t.passive_order_id for t in trades
                  if t.passive_order_id in strategy_order_ids}
    return len(filled_ids) / total_quotes_submitted


# ── Adverse selection ─────────────────────────────────────────────────────────

def adverse_selection(
    trades: List[Trade],
    mid_ts: List[Tuple[float, float]],
    strategy_order_ids: set,
    horizon_seconds: float = 1.0,
) -> Dict[str, float]:
    """
    Average mid-price change in the *horizon_seconds* after each fill,
    signed so that positive = adverse (mid moved against us after fill).

    For a passive BID fill: we bought, then mid goes DOWN → adverse.
    Sign convention: adverse_selection > 0 means we got picked off.

    Returns dict with keys: 'overall', 'bid_fills', 'ask_fills'.
    """
    if not mid_ts:
        return {'overall': float('nan'), 'bid_fills': float('nan'),
                'ask_fills': float('nan')}

    # Build a function to look up mid at or after a given timestamp
    mid_times = [t for t, _ in mid_ts]
    mid_vals = [v for _, v in mid_ts]

    def mid_at(ts: float) -> Optional[float]:
        # Binary search for first mid_time >= ts
        lo, hi = 0, len(mid_times) - 1
        while lo <= hi:
            m = (lo + hi) // 2
            if mid_times[m] < ts:
                lo = m + 1
            else:
                hi = m - 1
        if lo >= len(mid_times):
            return None
        return mid_vals[lo]

    strategy_fills = [t for t in trades if t.passive_order_id in strategy_order_ids]

    bid_adverse = []
    ask_adverse = []

    for trade in strategy_fills:
        fill_ts = trade.timestamp
        mid_before = mid_at(fill_ts)
        mid_after = mid_at(fill_ts + horizon_seconds)
        if mid_before is None or mid_after is None:
            continue

        delta_mid = mid_after - mid_before

        # passive side is the opposite of aggressor
        if trade.aggressor_side == Side.ASK:
            # Aggressor sold to us → we bought (passive BID)
            # Adverse if mid goes down after we buy
            adverse = -delta_mid
            bid_adverse.append(adverse)
        else:
            # Aggressor bought from us → we sold (passive ASK)
            # Adverse if mid goes up after we sell
            adverse = delta_mid
            ask_adverse.append(adverse)

    all_adverse = bid_adverse + ask_adverse
    return {
        'overall': statistics.mean(all_adverse) if all_adverse else float('nan'),
        'bid_fills': statistics.mean(bid_adverse) if bid_adverse else float('nan'),
        'ask_fills': statistics.mean(ask_adverse) if ask_adverse else float('nan'),
        'n_fills': len(all_adverse),
    }


# ── Inventory profile ─────────────────────────────────────────────────────────

def inventory_profile(
    inventory_ts: List[Tuple[float, int]],
) -> Dict[str, object]:
    """
    Histogram and time-in-state statistics for inventory.

    Returns dict with:
        'histogram': list of (inventory_level, fraction_of_time)
        'mean': float
        'std': float
        'max_long': int
        'max_short': int
        'time_flat': float  (fraction of time with zero inventory)
    """
    if not inventory_ts:
        return {}

    times = [t for t, _ in inventory_ts]
    invs = [v for _, v in inventory_ts]

    total_time = times[-1] - times[0] if len(times) > 1 else 1.0
    if total_time <= 0:
        total_time = 1.0

    # Count time in each inventory state
    state_time: Dict[int, float] = defaultdict(float)
    for i, (t, inv) in enumerate(inventory_ts[:-1]):
        dt = times[i + 1] - t
        state_time[inv] += dt

    histogram = sorted(
        [(inv, t / total_time) for inv, t in state_time.items()],
        key=lambda x: x[0],
    )

    # Time-weighted mean and variance; clamp variance at 0 against float error
    mean = sum(inv * w for inv, w in state_time.items()) / total_time
    var = sum((inv ** 2) * w for inv, w in state_time.items()) / total_time - mean ** 2

    return {
        'histogram': histogram,
        'mean': mean,
        'std': math.sqrt(max(0.0, var)),
        'max_long': max(invs),
        'max_short': min(invs),
        'time_flat': state_time.get(0, 0.0) / total_time,
    }


# ── Summary report ───────────────────────────────────────────────────────────

def summary_report(
    result,
    strategy_order_ids: set,
    total_quotes: int,
    bucket_seconds: float = 60.0,
) -> Dict[str, object]:
    """
    Compute all metrics and return as a flat dict.

    Parameters
    ----------
    result          : SimulationResult
    strategy_order_ids : set of order IDs submitted by strategy
    total_quotes    : total number of quotes submitted by strategy
    bucket_seconds  : PnL bucketing interval
    """
    report = {
        'final_pnl_ticks': result.final_pnl,
        'final_inventory': result.final_inventory,
        'n_trades_total': len(result.trades),
        'n_strategy_fills': sum(
            1 for t in result.trades if t.passive_order_id in strategy_order_ids
        ),
        'sharpe': sharpe_ratio(result.pnl_ts, bucket_seconds),
        'max_drawdown': max_drawdown(result.pnl_ts),
        'annualised_vol': volatility(result.pnl_ts, bucket_seconds),
        'fill_rate': fill_rate(strategy_order_ids, result.trades, total_quotes),
        **adverse_selection(result.trades, result.mid_ts, strategy_order_ids),
        **{
            'inv_' + k: v
            for k, v in inventory_profile(result.inventory_ts).items()
            if k != 'histogram'
        },
    }
    return report
