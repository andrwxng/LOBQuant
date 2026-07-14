"""
test_simulator.py — End-to-end tests for the event-driven simulation loop.

Key checks:
  * A full run produces trades, dense time series, and snapshots
  * Same seed → bit-identical results (determinism)
  * Strategy inventory/cash exactly match a reconstruction from the trade log
    (covers both passive fills and strategy-as-aggressor fills)
  * PnL series is marked to a real mid even when one book side empties
    (no cash-only spikes)
  * Inventory respects the hard limit
  * Running without a strategy works
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob_sim.flow import PoissonFlow
from lob_sim.simulator import Simulator, run_simulation
from lob_sim.strategy import AvellanedaStoikov, FairValueAvellanedaStoikov

STRATEGY_ID_MIN = 2_000_000


def _run(seed: int = 7, duration: float = 120.0):
    """Run a short simulation, returning (result, strategy)."""
    flow = PoissonFlow(seed=seed)
    strat = AvellanedaStoikov(T=duration)
    sim = Simulator(flow=flow, strategy=strat, duration=duration,
                    snapshot_interval=30.0)
    return sim.run(), strat


class TestRunBasics:
    def test_run_produces_output(self):
        result, strat = _run()
        assert len(result.trades) > 0
        assert len(result.mid_ts) > 0
        assert len(result.pnl_ts) > 0
        assert len(result.inventory_ts) > 0
        assert len(result.snapshots) > 0
        assert strat.n_quotes_submitted > 0

    def test_timestamps_nondecreasing(self):
        result, _ = _run()
        for series in (result.mid_ts, result.pnl_ts, result.inventory_ts):
            times = [t for t, _ in series]
            assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_final_snapshot_appended(self):
        result, _ = _run()
        assert result.snapshots[-1].timestamp == 120.0

    def test_run_without_strategy(self):
        flow = PoissonFlow(seed=3)
        sim = Simulator(flow=flow, strategy=None, duration=60.0,
                        snapshot_interval=30.0)
        result = sim.run()
        assert len(result.mid_ts) > 0
        assert result.pnl_ts == []
        assert result.inventory_ts == []


class TestDeterminism:
    def test_same_seed_same_result(self):
        r1 = run_simulation(duration=60.0, T=60.0, seed=11)
        r2 = run_simulation(duration=60.0, T=60.0, seed=11)
        assert len(r1.trades) == len(r2.trades)
        assert r1.final_pnl == r2.final_pnl
        assert r1.final_inventory == r2.final_inventory
        assert ([(t.price_ticks, t.qty) for t in r1.trades] ==
                [(t.price_ticks, t.qty) for t in r2.trades])

    def test_different_seed_different_result(self):
        r1 = run_simulation(duration=60.0, T=60.0, seed=11)
        r2 = run_simulation(duration=60.0, T=60.0, seed=12)
        # Astronomically unlikely to coincide
        assert ([(t.price_ticks, t.qty) for t in r1.trades] !=
                [(t.price_ticks, t.qty) for t in r2.trades])


class TestFillAccountingConsistency:
    def test_strategy_state_matches_trade_log(self):
        """
        Reconstruct inventory and cash from the raw trade log (both passive
        and aggressor strategy fills) and compare with the strategy's own
        accounting.  Regression test for the bug where strategy quotes that
        crossed the spread executed as aggressor and were never accounted.
        """
        result, strat = _run(seed=7)

        inv, cash = 0, 0.0
        for t in result.trades:
            if t.passive_order_id in strat.all_submitted_ids:
                if t.aggressor_side.value == 'ask':
                    inv += t.qty            # aggressor sold to us: we bought
                    cash -= t.price_ticks * t.qty
                else:
                    inv -= t.qty            # aggressor bought from us: we sold
                    cash += t.price_ticks * t.qty
            if t.aggressor_order_id >= STRATEGY_ID_MIN:
                if t.aggressor_side.value == 'bid':
                    inv += t.qty
                    cash -= t.price_ticks * t.qty
                else:
                    inv -= t.qty
                    cash += t.price_ticks * t.qty

        assert strat.inventory == inv
        assert strat.cash == cash

    def test_inventory_respects_hard_limit(self):
        result, strat = _run(seed=7)
        limit = strat.max_inventory + strat.order_size
        for _, inv in result.inventory_ts:
            assert abs(inv) <= limit


class TestNoRequoteFeedbackLoop:
    def test_pathological_config_terminates(self):
        """
        Regression test: with tight quotes (small gamma) the strategy's own
        quotes become the BBO.  If the strategy were re-notified after its
        own submits/cancels, each requote would move the mid and trigger
        another requote at +EPSILON — freezing simulated time and processing
        millions of events (observed: 2M+ calls stuck at t=552 of 3600).
        With notification restricted to external events, the event count is
        bounded by ~5x the flow event count.
        """
        flow = PoissonFlow(seed=1, lambda_lim=5.0, lambda_mkt=0.3,
                           lambda_cancel=0.3, max_qty=3)
        strat = AvellanedaStoikov(gamma=0.0005, sigma=0.86, k=1.5, T=3600.0,
                                  max_inventory=30, max_half_spread=8)
        sim = Simulator(flow=flow, strategy=strat, duration=60.0,
                        snapshot_interval=0.0, initial_depth=20,
                        initial_qty_per_level=5)
        result = sim.run()
        # ~6 flow events/sec * 60 s = ~360 external events; each spawns at
        # most 4 actions.  20_000 is orders of magnitude below the runaway.
        assert len(result.pnl_ts) < 20_000


class TestLatency:
    def _config(self, seed, latency):
        flow = PoissonFlow(seed=seed, lambda_lim=5.0, lambda_mkt=0.3,
                           lambda_cancel=0.3, max_qty=3)
        strat = FairValueAvellanedaStoikov(
            alpha=0.005, gamma=0.01, sigma=0.86, k=1.5, T=600.0,
            max_inventory=30, max_half_spread=8)
        sim = Simulator(flow=flow, strategy=strat, duration=600.0,
                        snapshot_interval=0.0, initial_depth=20,
                        initial_qty_per_level=5, latency=latency)
        return sim, strat

    def test_zero_latency_matches_default(self):
        sim_a, _ = self._config(seed=5, latency=0.0)
        r_a = sim_a.run()
        flow = PoissonFlow(seed=5, lambda_lim=5.0, lambda_mkt=0.3,
                           lambda_cancel=0.3, max_qty=3)
        strat = FairValueAvellanedaStoikov(
            alpha=0.005, gamma=0.01, sigma=0.86, k=1.5, T=600.0,
            max_inventory=30, max_half_spread=8)
        sim_b = Simulator(flow=flow, strategy=strat, duration=600.0,
                          snapshot_interval=0.0, initial_depth=20,
                          initial_qty_per_level=5)   # latency omitted
        r_b = sim_b.run()
        assert r_a.final_pnl == r_b.final_pnl
        assert len(r_a.trades) == len(r_b.trades)

    def test_accounting_consistent_under_latency(self):
        """Fills landing while cancels are in flight must still be accounted:
        strategy state must exactly match a trade-log reconstruction."""
        sim, strat = self._config(seed=7, latency=0.5)
        result = sim.run()
        inv, cash = 0, 0.0
        for t in result.trades:
            if t.passive_order_id in strat.all_submitted_ids:
                if t.aggressor_side.value == 'ask':
                    inv += t.qty
                    cash -= t.price_ticks * t.qty
                else:
                    inv -= t.qty
                    cash += t.price_ticks * t.qty
            if t.aggressor_order_id >= STRATEGY_ID_MIN:
                if t.aggressor_side.value == 'bid':
                    inv += t.qty
                    cash -= t.price_ticks * t.qty
                else:
                    inv -= t.qty
                    cash += t.price_ticks * t.qty
        assert strat.inventory == inv
        assert strat.cash == cash


class TestFairValueProfitability:
    def test_validated_config_is_profitable(self):
        """
        Deterministic regression test for the headline result: the
        fair-value anchored strategy (γ=0.01, α=0.005, selected on training
        seeds 1-12) is profitable on held-out seeds.  Simulations are
        seed-deterministic, so this cannot flake; it fails only if a code
        change degrades the strategy or the simulator.
        """
        pnls = []
        for seed in range(13, 19):   # first 6 validation seeds
            flow = PoissonFlow(seed=seed, lambda_lim=5.0, lambda_mkt=0.3,
                               lambda_cancel=0.3, max_qty=3)
            strat = FairValueAvellanedaStoikov(
                alpha=0.005, gamma=0.01, sigma=0.86, k=1.5, T=3600.0,
                max_inventory=30, max_half_spread=8)
            sim = Simulator(flow=flow, strategy=strat, duration=3600.0,
                            snapshot_interval=0.0, initial_depth=20,
                            initial_qty_per_level=5)
            pnls.append(sim.run().final_pnl)
        assert sum(pnls) / len(pnls) > 0, f"per-seed PnL: {pnls}"


class TestPnLMarking:
    def test_no_cash_only_spikes_in_pnl_series(self):
        """
        When one book side momentarily empties, PnL must be marked at the
        last known mid, not collapse to cash-only.  The old bug produced
        single-event PnL jumps of ~inventory * mid (tens of thousands of
        ticks); real per-event changes are bounded by a few hundred.
        """
        result, _ = _run(seed=7)
        pnls = [p for _, p in result.pnl_ts]
        jumps = [abs(pnls[i] - pnls[i - 1]) for i in range(1, len(pnls))]
        assert max(jumps) < 2000

    def test_final_pnl_consistent_with_state(self):
        result, strat = _run(seed=7)
        final_mid = result.mid_ts[-1][1]
        expected = strat.cash + strat.inventory * final_mid
        # final_pnl is marked at the final book mid (or last known mid),
        # which is the last recorded mid in the series
        assert abs(result.final_pnl - expected) < 1e-6
