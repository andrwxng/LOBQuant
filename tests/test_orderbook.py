"""
test_orderbook.py — Unit tests for the LOB matching engine.

Coverage:
  * Basic limit-order resting
  * Crossing limits (immediate match)
  * Partial fills
  * Cancellations (full & partial-filled orders)
  * Market orders walking multiple price levels
  * Market order exhausting the entire book
  * Debug-mode invariant assertions
  * spread / mid / snapshot
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from lob_sim.events import Side
from lob_sim.orderbook import LOBBook


# ── helpers ───────────────────────────────────────────────────────────────────

def make_book(debug: bool = False) -> LOBBook:
    b = LOBBook()
    LOBBook.debug = debug
    return b


def bid(book: LOBBook, price: int, qty: int, oid: int, ts: float = 0.0):
    return book.submit_limit(Side.BID, price, qty, oid, ts)


def ask(book: LOBBook, price: int, qty: int, oid: int, ts: float = 0.0):
    return book.submit_limit(Side.ASK, price, qty, oid, ts)


def market_buy(book: LOBBook, qty: int, oid: int, ts: float = 0.0):
    return book.submit_market(Side.BID, qty, oid, ts)


def market_sell(book: LOBBook, qty: int, oid: int, ts: float = 0.0):
    return book.submit_market(Side.ASK, qty, oid, ts)


# ── resting orders ────────────────────────────────────────────────────────────

class TestResting:
    def test_bid_rests_in_book(self):
        book = make_book()
        trades = bid(book, 100, 5, 1)
        assert trades == []
        assert book.best_bid() == 100
        assert book.best_ask() is None
        assert book.order_qty(1) == 5

    def test_ask_rests_in_book(self):
        book = make_book()
        trades = ask(book, 100, 5, 1)
        assert trades == []
        assert book.best_ask() == 100
        assert book.best_bid() is None
        assert book.order_qty(1) == 5

    def test_multiple_bids_sorted_descending(self):
        book = make_book()
        bid(book, 99, 5, 1)
        bid(book, 101, 5, 2)
        bid(book, 100, 5, 3)
        assert book.best_bid() == 101
        snap = book.snapshot(depth=3)
        prices = [p for p, _ in snap.bids]
        assert prices == [101, 100, 99]

    def test_multiple_asks_sorted_ascending(self):
        book = make_book()
        ask(book, 102, 5, 1)
        ask(book, 100, 5, 2)
        ask(book, 101, 5, 3)
        assert book.best_ask() == 100
        snap = book.snapshot(depth=3)
        prices = [p for p, _ in snap.asks]
        assert prices == [100, 101, 102]

    def test_spread_and_mid(self):
        book = make_book()
        bid(book, 98, 5, 1)
        ask(book, 102, 5, 2)
        assert book.spread() == 4
        assert book.mid() == 100.0


# ── crossing limits ───────────────────────────────────────────────────────────

class TestCrossing:
    def test_crossing_bid_full_fill(self):
        """Buy limit at 102 crosses resting ask at 100."""
        book = make_book(debug=True)
        ask(book, 100, 10, oid=1)
        trades = bid(book, 102, 10, oid=2)
        assert len(trades) == 1
        t = trades[0]
        assert t.price_ticks == 100   # passive order's price
        assert t.qty == 10
        assert t.passive_order_id == 1
        assert t.aggressor_side == Side.BID
        # Both orders consumed
        assert not book.has_order(1)
        assert not book.has_order(2)
        assert book.best_ask() is None
        assert book.best_bid() is None

    def test_crossing_ask_full_fill(self):
        """Sell limit at 98 crosses resting bid at 100."""
        book = make_book(debug=True)
        bid(book, 100, 10, oid=1)
        trades = ask(book, 98, 10, oid=2)
        assert len(trades) == 1
        t = trades[0]
        assert t.price_ticks == 100
        assert t.qty == 10
        assert t.passive_order_id == 1
        assert t.aggressor_side == Side.ASK

    def test_crossing_bid_no_match_when_at_same_level(self):
        """Bid at 100 when best ask is 101 → no match."""
        book = make_book(debug=True)
        ask(book, 101, 5, oid=1)
        trades = bid(book, 100, 5, oid=2)
        assert trades == []
        assert book.has_order(1)
        assert book.has_order(2)


# ── partial fills ─────────────────────────────────────────────────────────────

class TestPartialFills:
    def test_bid_partially_fills_ask(self):
        """Buy 3 from a resting ask of 10 — ask should have 7 remaining."""
        book = make_book(debug=True)
        ask(book, 100, 10, oid=1)
        trades = bid(book, 100, 3, oid=2)
        assert len(trades) == 1
        assert trades[0].qty == 3
        assert book.order_qty(1) == 7   # 10 - 3
        assert not book.has_order(2)    # fully consumed

    def test_ask_partially_fills_bid(self):
        """Sell 4 into a resting bid of 10."""
        book = make_book(debug=True)
        bid(book, 100, 10, oid=1)
        trades = ask(book, 100, 4, oid=2)
        assert len(trades) == 1
        assert trades[0].qty == 4
        assert book.order_qty(1) == 6

    def test_large_bid_consumes_multiple_asks(self):
        """Bid of 25 consumes three ask levels of 10 each."""
        book = make_book(debug=True)
        ask(book, 100, 10, oid=1)
        ask(book, 101, 10, oid=2)
        ask(book, 102, 10, oid=3)
        trades = bid(book, 102, 25, oid=4)
        assert sum(t.qty for t in trades) == 25
        # First two levels fully consumed, third partially
        assert not book.has_order(1)
        assert not book.has_order(2)
        assert book.order_qty(3) == 5   # 30 - 25

    def test_residual_rests_after_partial_match(self):
        """A crossing bid that only partially fills should rest the remainder."""
        book = make_book(debug=True)
        ask(book, 100, 3, oid=1)
        bid(book, 100, 10, oid=2)   # 3 filled, 7 should rest
        assert not book.has_order(1)
        assert book.order_qty(2) == 7
        assert book.best_bid() == 100


# ── cancellations ─────────────────────────────────────────────────────────────

class TestCancellations:
    def test_cancel_resting_order(self):
        book = make_book(debug=True)
        bid(book, 100, 5, oid=1)
        result = book.cancel(1, ts=1.0)
        assert result is True
        assert not book.has_order(1)
        assert book.best_bid() is None

    def test_cancel_nonexistent_order(self):
        book = make_book()
        result = book.cancel(999, ts=1.0)
        assert result is False

    def test_cancel_middle_order_at_level(self):
        """Cancel the middle of three orders at the same price."""
        book = make_book(debug=True)
        bid(book, 100, 5, oid=1)
        bid(book, 100, 5, oid=2)
        bid(book, 100, 5, oid=3)
        book.cancel(2, ts=1.0)
        assert not book.has_order(2)
        assert book.has_order(1)
        assert book.has_order(3)
        assert book.best_bid() == 100

    def test_cancel_partially_filled_order(self):
        """Partial fill then cancel the remainder."""
        book = make_book(debug=True)
        ask(book, 100, 10, oid=1)   # resting ask of 10
        bid(book, 100, 4, oid=2)    # fills 4 from order 1
        assert book.order_qty(1) == 6
        result = book.cancel(1, ts=2.0)
        assert result is True
        assert not book.has_order(1)
        assert book.best_ask() is None

    def test_cancel_cleans_up_empty_level(self):
        """After cancelling the only order at a level, the level should disappear."""
        book = make_book(debug=True)
        ask(book, 100, 5, oid=1)
        ask(book, 101, 5, oid=2)
        book.cancel(1, ts=1.0)
        # Level 100 should be gone
        assert 100 not in book._asks
        assert book.best_ask() == 101


# ── market orders ─────────────────────────────────────────────────────────────

class TestMarketOrders:
    def test_market_buy_single_level(self):
        book = make_book(debug=True)
        ask(book, 100, 10, oid=1)
        trades = market_buy(book, 5, oid=99)
        assert len(trades) == 1
        assert trades[0].qty == 5
        assert book.order_qty(1) == 5

    def test_market_sell_single_level(self):
        book = make_book(debug=True)
        bid(book, 100, 10, oid=1)
        trades = market_sell(book, 3, oid=99)
        assert len(trades) == 1
        assert trades[0].qty == 3
        assert book.order_qty(1) == 7

    def test_market_buy_walks_multiple_levels(self):
        book = make_book(debug=True)
        ask(book, 100, 5, oid=1)
        ask(book, 101, 5, oid=2)
        ask(book, 102, 5, oid=3)
        trades = market_buy(book, 12, oid=99)
        assert sum(t.qty for t in trades) == 12
        assert not book.has_order(1)
        assert not book.has_order(2)
        assert book.order_qty(3) == 3

    def test_market_buy_exhausts_book(self):
        """Market order larger than entire book leaves no resting asks."""
        book = make_book(debug=True)
        ask(book, 100, 5, oid=1)
        ask(book, 101, 5, oid=2)
        trades = market_buy(book, 100, oid=99)
        assert sum(t.qty for t in trades) == 10
        assert book.best_ask() is None
        assert book.best_bid() is None

    def test_market_order_aggressor_side_recorded(self):
        book = make_book()
        ask(book, 100, 5, oid=1)
        trades = market_buy(book, 5, oid=99)
        assert trades[0].aggressor_side == Side.BID

        book2 = make_book()
        bid(book2, 100, 5, oid=1)
        trades2 = market_sell(book2, 5, oid=99)
        assert trades2[0].aggressor_side == Side.ASK


# ── price-time priority ───────────────────────────────────────────────────────

class TestPriceTimePriority:
    def test_fifo_at_same_level(self):
        """First resting order at a price level should be filled first."""
        book = make_book(debug=True)
        ask(book, 100, 3, oid=1, ts=0.0)   # first
        ask(book, 100, 3, oid=2, ts=1.0)   # second
        trades = bid(book, 100, 3, oid=99)
        assert len(trades) == 1
        assert trades[0].passive_order_id == 1   # FIFO: order 1 first

    def test_price_priority_over_time(self):
        """Better price wins over earlier arrival."""
        book = make_book()
        ask(book, 101, 5, oid=1, ts=0.0)   # worse price but earlier
        ask(book, 100, 5, oid=2, ts=1.0)   # better price
        trades = market_buy(book, 5, oid=99)
        assert trades[0].passive_order_id == 2   # price priority


# ── book invariants ───────────────────────────────────────────────────────────

class TestInvariants:
    def test_duplicate_order_id_raises(self):
        book = make_book()
        bid(book, 100, 5, oid=1)
        with pytest.raises(ValueError):
            bid(book, 100, 5, oid=1)

    def test_zero_qty_raises(self):
        book = make_book()
        with pytest.raises(ValueError):
            book.submit_limit(Side.BID, 100, 0, 1, 0.0)

    def test_negative_price_raises(self):
        book = make_book()
        with pytest.raises(ValueError):
            book.submit_limit(Side.BID, -1, 5, 1, 0.0)

    def test_debug_invariants_pass_after_complex_sequence(self):
        """Run a complex mixed sequence; debug mode should never fire."""
        book = make_book(debug=True)
        for i in range(1, 6):
            bid(book, 100 - i, 5, oid=i)
            ask(book, 100 + i, 5, oid=100 + i)
        # crossing limit
        trades = bid(book, 103, 8, oid=200)
        assert sum(t.qty for t in trades) > 0
        # cancel one
        book.cancel(2, ts=5.0)
        # market order
        market_buy(book, 3, oid=300)
        market_sell(book, 3, oid=301)
        # assertions in debug mode check all through
        book._assert_invariants()


# ── snapshot ──────────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_snapshot_depth_limit(self):
        book = make_book()
        for i in range(20):
            bid(book, 100 - i, 5, oid=i + 1)
            ask(book, 101 + i, 5, oid=100 + i + 1)
        snap = book.snapshot(depth=5)
        assert len(snap.bids) == 5
        assert len(snap.asks) == 5

    def test_snapshot_aggregates_qty_per_level(self):
        book = make_book()
        bid(book, 100, 3, oid=1)
        bid(book, 100, 4, oid=2)   # same level
        ask(book, 101, 7, oid=3)
        snap = book.snapshot()
        assert snap.bids[0] == (100, 7)   # 3 + 4
        assert snap.asks[0] == (101, 7)
