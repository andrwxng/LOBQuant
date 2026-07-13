"""
orderbook.py — Price-time priority matching engine.

Design decisions
----------------
* Prices are integer ticks throughout.  Float conversion happens only at
  the presentation layer (metrics, notebooks).
* Bid book: SortedDict keyed by *negative* tick so iteration yields
  descending price (best bid first).
* Ask book: SortedDict keyed by positive tick, ascending (best ask first).
* Each price level is a collections.deque of (order_id, qty) tuples — FIFO.
* Cancellation: we keep a flat dict  order_id -> OrderRecord  that stores
  the side, price_ticks, and remaining qty.  Cancellation walks the deque
  at that price level and removes the first matching order_id.  This is
  O(n) per level in the worst case, but levels are typically short.  For
  research purposes this is acceptable; a production system would use a
  doubly-linked list per level for true O(1) removal.
* DEBUG_MODE (set LOBBook.debug = True) enables invariant assertions after
  every mutation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from sortedcontainers import SortedDict

from .events import BookSnapshot, Side, Trade


@dataclass
class _OrderRecord:
    """Internal metadata kept per resting order."""
    side: Side
    price_ticks: int
    qty: int            # remaining (unfilled) qty


class LOBBook:
    """
    Central limit order book with price-time priority matching.

    All external prices are integer ticks.
    """

    debug: bool = False   # class-level flag; flip to True for invariant checks

    def __init__(self) -> None:
        # Bid book: key = -price_ticks so SortedDict iterates best-bid first
        self._bids: SortedDict = SortedDict()   # {-price_ticks: deque[(oid, qty)]}
        # Ask book: key = +price_ticks, ascending
        self._asks: SortedDict = SortedDict()   # {+price_ticks: deque[(oid, qty)]}

        # O(1) lookup for cancel / partial-fill tracking
        self._orders: Dict[int, _OrderRecord] = {}

        self._trade_count = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def submit_limit(
        self,
        side: Side,
        price_ticks: int,
        qty: int,
        order_id: int,
        ts: float,
    ) -> List[Trade]:
        """
        Submit a limit order.  May immediately match against resting orders.
        Returns list of resulting Trade objects (empty if no match).
        Any residual qty rests in the book.
        """
        if order_id in self._orders:
            raise ValueError(f"Duplicate order_id {order_id}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if price_ticks <= 0:
            raise ValueError(f"price_ticks must be positive, got {price_ticks}")

        trades: List[Trade] = []
        remaining = qty

        if side == Side.BID:
            # Match against resting asks at or below our limit price
            while remaining > 0 and self._asks:
                best_ask_price = self._asks.keys()[0]   # smallest key = best ask
                if best_ask_price > price_ticks:
                    break                               # no more matching asks
                level = self._asks[best_ask_price]
                trades.extend(
                    self._fill_level(level, best_ask_price, Side.ASK,
                                     Side.BID, order_id, remaining, ts)
                )
                remaining = qty - sum(t.qty for t in trades)
                if not level:
                    del self._asks[best_ask_price]
        else:  # ASK
            # Match against resting bids at or above our limit price
            while remaining > 0 and self._bids:
                best_bid_key = self._bids.keys()[0]    # most negative = highest price
                if -best_bid_key < price_ticks:
                    break
                level = self._bids[best_bid_key]
                trades.extend(
                    self._fill_level(level, best_bid_key, Side.BID,
                                     Side.ASK, order_id, remaining, ts)
                )
                remaining = qty - sum(t.qty for t in trades)
                if not level:
                    del self._bids[best_bid_key]

        if remaining > 0:
            # Rest the unfilled portion
            self._orders[order_id] = _OrderRecord(side, price_ticks, remaining)
            if side == Side.BID:
                key = -price_ticks
                if key not in self._bids:
                    self._bids[key] = deque()
                self._bids[key].append((order_id, remaining))
            else:
                if price_ticks not in self._asks:
                    self._asks[price_ticks] = deque()
                self._asks[price_ticks].append((order_id, remaining))

        if self.debug:
            self._assert_invariants()
        return trades

    def submit_market(
        self,
        side: Side,
        qty: int,
        order_id: int,
        ts: float,
    ) -> List[Trade]:
        """
        Submit a market order.  Matches against the opposite side.
        Returns Trades.  Any unfilled qty is silently discarded (no resting).
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        trades: List[Trade] = []
        remaining = qty

        if side == Side.BID:
            # Consume asks
            while remaining > 0 and self._asks:
                key = self._asks.keys()[0]
                level = self._asks[key]
                prev_len = len(trades)
                trades.extend(
                    self._fill_level(level, key, Side.ASK,
                                     Side.BID, order_id, remaining, ts)
                )
                new_filled = sum(t.qty for t in trades[prev_len:])
                remaining -= new_filled
                if not level:
                    del self._asks[key]
        else:
            # Consume bids
            while remaining > 0 and self._bids:
                key = self._bids.keys()[0]
                level = self._bids[key]
                prev_len = len(trades)
                trades.extend(
                    self._fill_level(level, key, Side.BID,
                                     Side.ASK, order_id, remaining, ts)
                )
                new_filled = sum(t.qty for t in trades[prev_len:])
                remaining -= new_filled
                if not level:
                    del self._bids[key]

        if self.debug:
            self._assert_invariants()
        return trades

    def cancel(self, order_id: int, ts: float) -> bool:
        """
        Cancel a resting order.
        Returns True if the order was found and cancelled, False otherwise.
        """
        rec = self._orders.pop(order_id, None)
        if rec is None:
            return False

        if rec.side == Side.BID:
            key = -rec.price_ticks
            book = self._bids
        else:
            key = rec.price_ticks
            book = self._asks

        level = book.get(key)
        if level is None:
            # Should not happen if internal state is consistent
            if self.debug:
                raise AssertionError(
                    f"cancel: order {order_id} in _orders but price level missing"
                )
            return False

        # Walk deque and remove the entry for this order_id
        removed = False
        for i, (oid, _qty) in enumerate(level):
            if oid == order_id:
                del level[i]           # O(n) — acceptable for research use
                removed = True
                break

        if not level:
            del book[key]

        if self.debug:
            self._assert_invariants()
        return removed

    # ── Book state queries ──────────────────────────────────────────────────

    def best_bid(self) -> Optional[int]:
        """Return best bid price in ticks, or None if book is empty."""
        if not self._bids:
            return None
        return -self._bids.keys()[0]

    def best_ask(self) -> Optional[int]:
        """Return best ask price in ticks, or None if book is empty."""
        if not self._asks:
            return None
        return self._asks.keys()[0]

    def mid(self) -> Optional[float]:
        """Return mid price in ticks (float because it may be a half-tick)."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> Optional[int]:
        """Return bid-ask spread in ticks."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def snapshot(self, depth: int = 10) -> BookSnapshot:
        """Return top-of-book snapshot up to *depth* price levels per side."""
        bids: list[tuple[int, int]] = []
        for key in list(self._bids.keys())[:depth]:
            price = -key
            qty = sum(q for _, q in self._bids[key])
            bids.append((price, qty))

        asks: list[tuple[int, int]] = []
        for key in list(self._asks.keys())[:depth]:
            price = key
            qty = sum(q for _, q in self._asks[key])
            asks.append((price, qty))

        return BookSnapshot(
            timestamp=0.0,  # caller should set
            bids=bids,
            asks=asks,
            mid_ticks=self.mid(),
            spread_ticks=self.spread(),
        )

    def order_qty(self, order_id: int) -> Optional[int]:
        """Return remaining qty for a resting order, or None if not found."""
        rec = self._orders.get(order_id)
        return rec.qty if rec is not None else None

    def has_order(self, order_id: int) -> bool:
        return order_id in self._orders

    def total_bid_qty(self) -> int:
        return sum(q for level in self._bids.values() for _, q in level)

    def total_ask_qty(self) -> int:
        return sum(q for level in self._asks.values() for _, q in level)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _fill_level(
        self,
        level: deque,
        level_key: int,
        passive_side: Side,
        aggressor_side: Side,
        aggressor_id: int,
        remaining: int,
        ts: float,
    ) -> List[Trade]:
        """
        Consume orders from *level* deque until *remaining* is exhausted.
        Updates _orders for passive fills.  Returns list of Trade objects.
        The level deque is mutated in place.
        """
        trades: List[Trade] = []
        while level and remaining > 0:
            passive_id, passive_qty = level[0]
            fill_qty = min(remaining, passive_qty)
            # Execution price is the passive order's price (price-time priority)
            if passive_side == Side.ASK:
                exec_price = level_key          # positive ticks
            else:
                exec_price = -level_key         # negate back to positive

            self._trade_count += 1
            trades.append(Trade(
                trade_id=self._trade_count,
                aggressor_side=aggressor_side,
                aggressor_order_id=aggressor_id,
                passive_order_id=passive_id,
                price_ticks=exec_price,
                qty=fill_qty,
                timestamp=ts,
            ))
            remaining -= fill_qty

            # Update or remove passive resting order
            passive_rec = self._orders.get(passive_id)
            if passive_rec is not None:
                passive_rec.qty -= fill_qty
                if passive_rec.qty == 0:
                    del self._orders[passive_id]
                    level.popleft()
                else:
                    # Partial fill — update qty in deque in place
                    level[0] = (passive_id, passive_rec.qty)
            else:
                # Passive order already removed (shouldn't normally happen)
                level.popleft()

        return trades

    # ── Invariant checker (debug mode) ──────────────────────────────────────

    def _assert_invariants(self) -> None:
        """Verify internal consistency.  Called after every mutation in debug mode."""
        # No crossed book
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            assert bb < ba, f"Crossed book: bid {bb} >= ask {ba}"

        # Every resting order in _orders has a matching entry in the book
        for oid, rec in self._orders.items():
            if rec.side == Side.BID:
                key = -rec.price_ticks
                level = self._bids.get(key)
            else:
                key = rec.price_ticks
                level = self._asks.get(key)
            assert level is not None, (
                f"order {oid} in _orders but level missing (side={rec.side}, "
                f"price={rec.price_ticks})"
            )
            level_ids = {o for o, _ in level}
            assert oid in level_ids, (
                f"order {oid} in _orders but not in level deque"
            )

        # Every entry in book levels has a matching _orders entry
        for key, level in self._bids.items():
            for oid, qty in level:
                assert oid in self._orders, (
                    f"bid level entry {oid} has no _orders record"
                )
                assert qty > 0, f"Zero qty in bid level for order {oid}"

        for key, level in self._asks.items():
            for oid, qty in level:
                assert oid in self._orders, (
                    f"ask level entry {oid} has no _orders record"
                )
                assert qty > 0, f"Zero qty in ask level for order {oid}"

        # No empty levels
        for key in list(self._bids.keys()):
            assert len(self._bids[key]) > 0, f"Empty bid level at {-key}"
        for key in list(self._asks.keys()):
            assert len(self._asks[key]) > 0, f"Empty ask level at {key}"
