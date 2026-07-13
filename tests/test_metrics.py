"""
test_metrics.py — Unit tests for the analytics module.

All tests use small hand-constructed inputs with exactly computable
expected values; no simulation randomness is involved except in the
summary_report integration test.
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from lob_sim import metrics
from lob_sim.events import Side, Trade
from lob_sim.simulator import run_simulation


def _trade(trade_id: int, aggressor_side: Side, passive_id: int,
           price: int, qty: int, ts: float) -> Trade:
    return Trade(
        trade_id=trade_id,
        aggressor_side=aggressor_side,
        aggressor_order_id=999,
        passive_order_id=passive_id,
        price_ticks=price,
        qty=qty,
        timestamp=ts,
    )


class TestTerminalPnl:
    def test_liquidates_at_mid(self):
        assert metrics.terminal_pnl(cash=-300.0, inventory=3, final_mid=105.0) == 15.0

    def test_none_mid_returns_cash(self):
        assert metrics.terminal_pnl(cash=42.0, inventory=5, final_mid=None) == 42.0


class TestPnlSeries:
    def test_bucketing_last_value_wins_and_forward_fill(self):
        pnl_ts = [(0.0, 0.0), (30.0, 5.0), (65.0, 10.0), (200.0, 20.0)]
        times, values = metrics.pnl_series(pnl_ts, bucket_seconds=60.0)
        # bucket 0: last value 5; bucket 1: 10; bucket 2: empty → forward-fill
        # 10; bucket 3: 20
        assert times == [0.0, 60.0, 120.0, 180.0]
        assert values == [5.0, 10.0, 10.0, 20.0]

    def test_empty_input(self):
        assert metrics.pnl_series([]) == ([], [])

    def test_increments(self):
        assert metrics.pnl_increments([5.0, 10.0, 10.0, 20.0]) == [5.0, 0.0, 10.0]


class TestMaxDrawdown:
    def test_known_series(self):
        pnl_ts = [(0.0, 0.0), (1.0, 10.0), (2.0, 3.0), (3.0, 8.0), (4.0, -2.0)]
        assert metrics.max_drawdown(pnl_ts) == 12.0   # peak 10 → trough -2

    def test_monotonic_series_has_zero_drawdown(self):
        pnl_ts = [(float(i), float(i)) for i in range(10)]
        assert metrics.max_drawdown(pnl_ts) == 0.0

    def test_empty(self):
        assert metrics.max_drawdown([]) == 0.0


class TestSharpe:
    def test_insufficient_data_is_nan(self):
        assert math.isnan(metrics.sharpe_ratio([(0.0, 1.0)]))

    def test_flat_pnl_is_nan(self):
        pnl_ts = [(60.0 * i, 5.0) for i in range(10)]
        assert math.isnan(metrics.sharpe_ratio(pnl_ts))

    def test_noisy_uptrend_is_positive(self):
        # Alternate +3/+1 increments: positive mean, positive variance
        pnl, series = 0.0, []
        for i in range(20):
            pnl += 3.0 if i % 2 == 0 else 1.0
            series.append((60.0 * i, pnl))
        assert metrics.sharpe_ratio(series) > 0

    def test_volatility_non_negative(self):
        pnl, series = 0.0, []
        for i in range(20):
            pnl += 3.0 if i % 2 == 0 else 1.0
            series.append((60.0 * i, pnl))
        assert metrics.volatility(series) > 0


class TestFillRate:
    def test_half_filled(self):
        strategy_ids = {1, 2, 3, 4}
        trades = [
            _trade(1, Side.BID, passive_id=1, price=100, qty=1, ts=0.0),
            _trade(2, Side.BID, passive_id=2, price=100, qty=1, ts=1.0),
            _trade(3, Side.BID, passive_id=2, price=100, qty=1, ts=2.0),  # same order again
            _trade(4, Side.BID, passive_id=99, price=100, qty=1, ts=3.0),  # not ours
        ]
        assert metrics.fill_rate(strategy_ids, trades, total_quotes_submitted=4) == 0.5

    def test_zero_quotes_is_nan(self):
        assert math.isnan(metrics.fill_rate(set(), [], 0))


class TestAdverseSelection:
    def test_passive_bid_fill_adverse_when_mid_falls(self):
        """We bought (aggressor ASK) and mid fell 10 → adverse = +10."""
        mid_ts = [(0.0, 100.0), (1.0, 90.0)]
        trades = [_trade(1, Side.ASK, passive_id=1, price=100, qty=1, ts=0.0)]
        out = metrics.adverse_selection(trades, mid_ts, {1}, horizon_seconds=1.0)
        assert out['bid_fills'] == pytest.approx(10.0)
        assert out['overall'] == pytest.approx(10.0)
        assert out['n_fills'] == 1

    def test_passive_ask_fill_adverse_when_mid_rises(self):
        """We sold (aggressor BID) and mid rose 10 → adverse = +10."""
        mid_ts = [(0.0, 100.0), (1.0, 110.0)]
        trades = [_trade(1, Side.BID, passive_id=1, price=100, qty=1, ts=0.0)]
        out = metrics.adverse_selection(trades, mid_ts, {1}, horizon_seconds=1.0)
        assert out['ask_fills'] == pytest.approx(10.0)

    def test_favorable_move_is_negative(self):
        """We bought and mid rose → adverse is negative (we gained)."""
        mid_ts = [(0.0, 100.0), (1.0, 105.0)]
        trades = [_trade(1, Side.ASK, passive_id=1, price=100, qty=1, ts=0.0)]
        out = metrics.adverse_selection(trades, mid_ts, {1}, horizon_seconds=1.0)
        assert out['bid_fills'] == pytest.approx(-5.0)

    def test_no_mid_data_is_nan(self):
        out = metrics.adverse_selection([], [], set())
        assert math.isnan(out['overall'])


class TestInventoryProfile:
    def test_time_weighted_statistics(self):
        # inv=0 for 10 s, inv=5 for 30 s, total 40 s
        inventory_ts = [(0.0, 0), (10.0, 5), (30.0, 5), (40.0, 0)]
        out = metrics.inventory_profile(inventory_ts)
        assert out['mean'] == pytest.approx(3.75)          # (0*10 + 5*30) / 40
        assert out['std'] == pytest.approx(math.sqrt(25 * 30 / 40 - 3.75 ** 2))
        assert out['time_flat'] == pytest.approx(0.25)     # 10 / 40
        assert out['max_long'] == 5
        assert out['max_short'] == 0
        assert dict(out['histogram']) == pytest.approx({0: 0.25, 5: 0.75})

    def test_empty_input(self):
        assert metrics.inventory_profile([]) == {}


class TestSummaryReport:
    def test_report_on_real_simulation(self):
        result = run_simulation(duration=60.0, T=60.0, seed=5)
        # run_simulation doesn't expose the strategy, so approximate the ID
        # set from the trade log (strategy IDs start at 2_000_000)
        strategy_ids = {t.passive_order_id for t in result.trades
                        if t.passive_order_id >= 2_000_000}
        report = metrics.summary_report(result, strategy_ids,
                                        total_quotes=max(1, len(strategy_ids)))
        for key in ('final_pnl_ticks', 'final_inventory', 'n_trades_total',
                    'n_strategy_fills', 'sharpe', 'max_drawdown',
                    'annualised_vol', 'fill_rate', 'overall', 'n_fills',
                    'inv_mean', 'inv_std', 'inv_time_flat'):
            assert key in report, f"missing key: {key}"
        assert report['n_trades_total'] == len(result.trades)
        assert report['max_drawdown'] >= 0.0
