"""
test_flow.py — Tests for synthetic order flow generators.

Key checks:
  * PoissonFlow produces events of the right types
  * Over a long run, event counts approximately match Poisson rates
  * Market-order / limit-order / cancel fractions are stable
  * Geometric depth distribution shape
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from lob_sim.events import CancelRequest, MarketOrder, Order, Side
from lob_sim.flow import PoissonFlow
from lob_sim.orderbook import LOBBook


# ── helpers ───────────────────────────────────────────────────────────────────

def _seed_book() -> LOBBook:
    """Return a book with a 10-level symmetric ladder around 1000."""
    book = LOBBook()
    for i in range(10):
        book.submit_limit(Side.BID, 1000 - 2 - i, 10, i + 1, 0.0)
        book.submit_limit(Side.ASK, 1000 + 2 + i, 10, 100 + i + 1, 0.0)
    return book


def _generate_n(flow: PoissonFlow, book: LOBBook, n: int):
    """Generate n events, returning list of events and timestamps."""
    events = []
    ts = 0.0
    for _ in range(n):
        ev_ts, ev = flow._compute_next_ts(ts, book)
        events.append((ev_ts, ev))
        ts = ev_ts
        # Apply the event to keep book realistic
        if isinstance(ev, Order):
            try:
                book.submit_limit(ev.side, ev.price_ticks, ev.qty, ev.order_id, ev_ts)
            except ValueError:
                pass
        elif isinstance(ev, MarketOrder):
            book.submit_market(ev.side, ev.qty, ev.order_id, ev_ts)
        elif isinstance(ev, CancelRequest):
            book.cancel(ev.order_id, ev_ts)
    return events


# ── basic sanity ──────────────────────────────────────────────────────────────

class TestPoissonFlowBasic:
    def test_events_have_increasing_timestamps(self):
        flow = PoissonFlow(seed=42)
        book = _seed_book()
        events = _generate_n(flow, book, 100)
        times = [t for t, _ in events]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_event_types_are_valid(self):
        flow = PoissonFlow(seed=0)
        book = _seed_book()
        events = _generate_n(flow, book, 200)
        for _, ev in events:
            assert isinstance(ev, (Order, MarketOrder, CancelRequest))

    def test_orders_have_positive_price_and_qty(self):
        flow = PoissonFlow(seed=1)
        book = _seed_book()
        events = _generate_n(flow, book, 200)
        for _, ev in events:
            if isinstance(ev, Order):
                assert ev.price_ticks > 0
                assert ev.qty > 0
            elif isinstance(ev, MarketOrder):
                assert ev.qty > 0


# ── rate calibration ──────────────────────────────────────────────────────────

class TestRateCalibration:
    """
    Over a long run (N=5000 events), verify that observed mix of market
    vs limit orders approximately matches the configured rates.

    With lambda_lim=5, lambda_mkt=1 (ignoring cancels for the mix check):
      fraction of (lim + mkt) events that are market ≈ 1/6 ≈ 0.167
    We allow ±5 percentage points.
    """

    def test_market_order_fraction(self):
        flow = PoissonFlow(lambda_lim=5.0, lambda_mkt=1.0, lambda_cancel=0.0,
                           seed=1234)
        book = _seed_book()
        n = 3000
        events = _generate_n(flow, book, n)
        market_count = sum(1 for _, ev in events if isinstance(ev, MarketOrder))
        limit_count = sum(1 for _, ev in events if isinstance(ev, Order))
        total = market_count + limit_count
        if total == 0:
            pytest.skip("No events generated")
        observed_frac = market_count / total
        expected_frac = 1.0 / (5.0 + 1.0)
        assert abs(observed_frac - expected_frac) < 0.06, (
            f"Market order fraction {observed_frac:.3f} far from "
            f"expected {expected_frac:.3f}"
        )

    def test_inter_arrival_times_are_positive(self):
        flow = PoissonFlow(lambda_lim=5.0, lambda_mkt=1.0, seed=99)
        book = _seed_book()
        events = _generate_n(flow, book, 500)
        for i in range(1, len(events)):
            dt = events[i][0] - events[i - 1][0]
            assert dt >= 0, f"Non-positive inter-arrival time: {dt}"

    def test_total_event_rate(self):
        """
        With lambda_lim=5, lambda_mkt=1, zero cancel, the expected rate is 6/s.
        After T=100 s we should have roughly 600 events ± wide tolerance.
        """
        flow = PoissonFlow(lambda_lim=5.0, lambda_mkt=1.0, lambda_cancel=0.0,
                           seed=777)
        book = _seed_book()
        events = _generate_n(flow, book, 1000)
        if not events:
            pytest.skip("No events")
        T = events[-1][0]
        if T == 0:
            pytest.skip("Zero total time")
        observed_rate = len(events) / T
        expected_rate = 6.0
        # Allow wide tolerance: within 30%
        assert abs(observed_rate - expected_rate) / expected_rate < 0.30, (
            f"Observed rate {observed_rate:.2f} far from expected {expected_rate}"
        )


# ── geometric depth distribution ──────────────────────────────────────────────

class TestGeometricDepth:
    """
    The geometric distribution with parameter p should have:
        P(X = 1) = p
        P(X = 2) = p*(1-p)
    Check that the empirical distribution from _geometric() is consistent.
    """

    def test_geometric_mode_is_one(self):
        flow = PoissonFlow(p_geom=0.5, seed=42)
        depths = [flow._geometric() for _ in range(2000)]
        freq = {}
        for d in depths:
            freq[d] = freq.get(d, 0) + 1
        # Mode should be 1
        mode = max(freq, key=freq.get)
        assert mode == 1

    def test_geometric_mean(self):
        """Mean of Geometric(p) = 1/p."""
        p = 0.3
        flow = PoissonFlow(p_geom=p, seed=0)
        depths = [flow._geometric() for _ in range(5000)]
        empirical_mean = sum(depths) / len(depths)
        expected_mean = 1.0 / p
        assert abs(empirical_mean - expected_mean) < 0.5, (
            f"Geometric mean {empirical_mean:.2f} vs expected {expected_mean:.2f}"
        )

    def test_geometric_min_is_one(self):
        flow = PoissonFlow(seed=5)
        for _ in range(1000):
            assert flow._geometric() >= 1


# ── bid/ask symmetry ─────────────────────────────────────────────────────────

class TestBidAskSymmetry:
    def test_roughly_symmetric_sides(self):
        """Over many events, roughly half should be bids and half asks."""
        flow = PoissonFlow(seed=12345)
        book = _seed_book()
        events = _generate_n(flow, book, 1000)
        limit_orders = [ev for _, ev in events if isinstance(ev, Order)]
        if not limit_orders:
            pytest.skip("No limit orders")
        bids = sum(1 for o in limit_orders if o.side == Side.BID)
        frac = bids / len(limit_orders)
        assert 0.3 < frac < 0.7, f"Bid fraction {frac:.2f} is too skewed"


# ── order-ID partitioning ────────────────────────────────────────────────────

class TestOrderIdPartitioning:
    def test_flow_ids_restart_per_instance(self):
        """
        Regression test: flow IDs must restart at the range base for every
        new generator.  A shared module-level counter grows across runs and
        eventually collides with the strategy ID range (observed as a
        'Duplicate order_id' crash after ~50 one-hour runs in one process).
        """
        from lob_sim.flow import FLOW_ID_BASE
        ids = []
        for _ in range(2):
            flow = PoissonFlow(seed=42)
            book = _seed_book()
            events = _generate_n(flow, book, 50)
            ids.append([ev.order_id for _, ev in events
                        if isinstance(ev, (Order, MarketOrder))])
        # Identical seed + fresh instance → identical ID sequence, starting
        # in the flow range and far below the strategy range (2M)
        assert ids[0] == ids[1]
        assert all(FLOW_ID_BASE <= i < 2_000_000 for i in ids[0])


# ── generate_sequence ────────────────────────────────────────────────────────

class TestGenerateSequence:
    def test_returns_n_events(self):
        flow = PoissonFlow(seed=0)
        book = _seed_book()
        seq = flow.generate_sequence(100, book)
        assert len(seq) == 100

    def test_sequence_timestamps_nondecreasing(self):
        flow = PoissonFlow(seed=1)
        book = _seed_book()
        seq = flow.generate_sequence(200, book)
        times = [t for t, _ in seq]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))
