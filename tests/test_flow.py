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
from lob_sim.events import CancelRequest, MarketOrder, ModifyRequest, Order, Side
from lob_sim.flow import LOBSTER_ID_BASE, LOBSTERReplay, PoissonFlow
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


# ── LOBSTERReplay ─────────────────────────────────────────────────────────────

def _write_csv(path, rows) -> str:
    path.write_text("\n".join(",".join(str(c) for c in row) for row in rows))
    return str(path)


def _apply(book: LOBBook, event):
    """Mirror simulator.py's event dispatch for a standalone unit test."""
    if isinstance(event, Order):
        return book.submit_limit(event.side, event.price_ticks, event.qty,
                                 event.order_id, event.timestamp)
    if isinstance(event, MarketOrder):
        return book.submit_market(event.side, event.qty, event.order_id,
                                  event.timestamp)
    if isinstance(event, CancelRequest):
        book.cancel(event.order_id, event.timestamp)
        return []
    if isinstance(event, ModifyRequest):
        book.reduce_qty(event.order_id, event.delta_qty, event.timestamp)
        return []
    raise TypeError(f"unhandled event type: {type(event)}")


class TestLOBSTERReplay:
    """
    Fixture (tick_divisor=1, so raw price columns equal tick prices):

    Orderbook file (seeds the book, 2 levels):
        ask1=102/5, bid1=98/5, ask2=103/3, bid2=97/3

    Message file (5 rows, raw order_id 5001 is a genuine new order; raw
    order_id 98 is never introduced by a type-1 message and must resolve
    against the seeded level):
        1) type=1 new     BID id=5001 price=99 qty=10
        2) type=2 reduce  id=5001 by 4                 -> qty 10 -> 6
        3) type=3 cancel  id=98 price=98 (pre-existing) -> seeded BID@98 gone
        4) type=4 exec    id=5001 price=99 qty=6, direction=1 (resting BID)
                           -> aggressor is ASK, sweeps id=5001 entirely
        5) type=5 hidden  price=100 qty=2 direction=-1  -> not replayed
    """

    def _orderbook_rows(self):
        return [[102, 5, 98, 5, 103, 3, 97, 3]]

    def _message_rows(self):
        return [
            [34200.1, 1, 5001, 10, 99, 1],
            [34200.2, 2, 5001, 4, 99, 1],
            [34200.3, 3, 98, 5, 98, 1],
            [34200.4, 4, 5001, 6, 99, 1],
            [34200.5, 5, 9999, 2, 100, -1],
        ]

    def _make_replay(self, tmp_path, with_seed=True):
        msg_path = _write_csv(tmp_path / "messages.csv", self._message_rows())
        ob_path = None
        if with_seed:
            ob_path = _write_csv(tmp_path / "orderbook.csv",
                                 self._orderbook_rows())
        return LOBSTERReplay(message_file=msg_path, orderbook_file=ob_path,
                             n_levels=2, tick_divisor=1)

    def _drain(self, flow):
        events = []
        while True:
            ev = flow.next_event(0.0, None)
            if ev is None:
                break
            events.append(ev)
        return events

    # ── ID remapping ─────────────────────────────────────────────────────

    def test_all_ids_in_reserved_range(self, tmp_path):
        flow = self._make_replay(tmp_path)
        events = self._drain(flow)
        ids = [e.order_id for e in events if hasattr(e, 'order_id')]
        assert ids, "expected at least one ID-bearing event"
        assert all(i >= LOBSTER_ID_BASE for i in ids)
        # None of the internal ids collide with the raw LOBSTER ids used
        assert 5001 not in ids and 98 not in ids and 9999 not in ids

    def test_id_reused_across_messages_maps_consistently(self, tmp_path):
        flow = self._make_replay(tmp_path)
        events = self._drain(flow)
        new_order = next(e for e in events if isinstance(e, Order)
                         and e.price_ticks == 99)
        reduce_ev = next(e for e in events if isinstance(e, ModifyRequest))
        assert reduce_ev.order_id == new_order.order_id

    # ── Seeding and unresolved fallback ─────────────────────────────────

    def test_seed_creates_one_order_per_level(self, tmp_path):
        flow = self._make_replay(tmp_path)
        assert len(flow._level_seed_ids) == 4
        assert (Side.ASK, 102) in flow._level_seed_ids
        assert (Side.BID, 98) in flow._level_seed_ids

    def test_unresolvable_cancel_falls_back_to_seed(self, tmp_path):
        flow = self._make_replay(tmp_path, with_seed=True)
        events = self._drain(flow)
        cancels = [e for e in events if isinstance(e, CancelRequest)]
        assert len(cancels) == 1
        assert cancels[0].order_id == flow._level_seed_ids[(Side.BID, 98)]
        assert flow.n_unresolved == 0

    def test_unresolvable_cancel_dropped_without_seed_file(self, tmp_path):
        flow = self._make_replay(tmp_path, with_seed=False)
        events = self._drain(flow)
        assert not any(isinstance(e, CancelRequest) for e in events)
        assert flow.n_unresolved == 1

    # ── Execution direction (the aggressor-inversion bug) ────────────────

    def test_visible_execution_aggressor_is_opposite_resting_side(self, tmp_path):
        """direction=1 means the RESTING order was a buy limit; the
        aggressor that hit it must be a sell (ASK)."""
        flow = self._make_replay(tmp_path)
        events = self._drain(flow)
        execs = [e for e in events if isinstance(e, MarketOrder)]
        assert len(execs) == 1
        assert execs[0].side == Side.ASK
        assert execs[0].qty == 6

    def test_hidden_execution_not_replayed_as_event(self, tmp_path):
        flow = self._make_replay(tmp_path)
        events = self._drain(flow)
        assert len(events) == len([e for e in events
                                    if isinstance(e, MarketOrder)]) + \
               len([e for e in events if not isinstance(e, MarketOrder)])
        # No event carries qty=2 at price 100 (that's the hidden execution)
        assert not any(getattr(e, 'qty', None) == 2 and
                       getattr(e, 'price_ticks', None) == 100
                       for e in events)

    def test_hidden_execution_recorded_separately(self, tmp_path):
        flow = self._make_replay(tmp_path)
        assert len(flow.hidden_trades) == 1
        ts, side, price, qty = flow.hidden_trades[0]
        assert ts == pytest.approx(0.4)
        assert (side, price, qty) == (Side.ASK, 100, 2)

    # ── End-to-end book reconstruction ────────────────────────────────────

    def test_full_replay_matches_hand_computed_book_state(self, tmp_path):
        flow = self._make_replay(tmp_path)
        LOBBook.debug = True
        book = LOBBook()
        trades_seen = []
        while True:
            ev = flow.next_event(0.0, book)
            if ev is None:
                break
            trades_seen.extend(_apply(book, ev))

        # Seed asks untouched: 102(5), 103(3)
        assert book.best_ask() == 102
        # Seed bid@98 was cancelled; new bid@99 (qty6) was fully executed;
        # only the untouched seed bid@97(3) remains
        assert book.best_bid() == 97
        assert book.order_qty(flow._level_seed_ids[(Side.BID, 97)]) == 3

        # The type-4 execution produced exactly one trade: ASK aggressor
        # sweeping the id=5001 order (now internal) at price 99, qty 6
        assert len(trades_seen) == 1
        t = trades_seen[0]
        assert t.aggressor_side == Side.ASK
        assert t.price_ticks == 99
        assert t.qty == 6

    # ── Interface basics (peek / reset) ───────────────────────────────────

    def test_peek_next_ts_reflects_head_of_stream(self, tmp_path):
        flow = self._make_replay(tmp_path)
        first_ts = flow.peek_next_ts(0.0)
        assert first_ts == 0.0   # first seed order, at ts=0.0

    def test_peek_next_ts_is_inf_when_exhausted(self, tmp_path):
        flow = self._make_replay(tmp_path)
        self._drain(flow)
        assert flow.peek_next_ts(0.0) == float('inf')

    def test_reset_replays_from_start(self, tmp_path):
        flow = self._make_replay(tmp_path)
        first_pass = self._drain(flow)
        flow.reset()
        second_pass = self._drain(flow)
        assert [type(e) for e in first_pass] == [type(e) for e in second_pass]
        assert [e.order_id for e in first_pass if hasattr(e, 'order_id')] == \
               [e.order_id for e in second_pass if hasattr(e, 'order_id')]

    def test_timestamps_nondecreasing(self, tmp_path):
        flow = self._make_replay(tmp_path)
        events = self._drain(flow)
        times = [e.timestamp for e in events]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


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
