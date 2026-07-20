"""
simulator.py — Event-driven simulation loop.

The simulator ties together:
  * A LOBBook (matching engine)
  * An OrderFlowGenerator (synthetic or replay)
  * A Strategy (market-maker)

Event loop
----------
1.  A priority queue (min-heap) of (timestamp, sequence_no, event) drives
    the loop.  The sequence_no breaks ties deterministically.
2.  The flow generator produces the next arrival time via Poisson draws;
    each new flow event is inserted into the heap.
3.  After every external market event the strategy is called; its returned
    actions are inserted with timestamp = current_ts + EPSILON so they execute
    "immediately" but after the triggering event.  The strategy is NOT
    re-notified after its own actions are processed — that would create a
    requote feedback loop whenever its quotes define the BBO.
4.  Fills belonging to the strategy are detected by comparing passive
    order_ids against strategy._our_orders.

Outputs
-------
SimulationResult contains:
    * trades           : List[Trade]
    * inventory_ts     : List[(time, inventory)]
    * pnl_ts           : List[(time, pnl_ticks)]
    * mid_ts           : List[(time, mid_ticks)]
    * spread_ts        : List[(time, spread_ticks)]
    * snapshots        : List[BookSnapshot]   (at snapshot_interval seconds)
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .events import (
    BookSnapshot, Cancel, CancelRequest, MarketOrder,
    ModifyRequest, Order, Side, SubmitLimit, Trade,
)
from .flow import OrderFlowGenerator, PoissonFlow
from .orderbook import LOBBook
from .strategy import AvellanedaStoikov, Strategy


EPSILON = 1e-9   # tiny time delta so strategy actions execute after triggers

_seq_counter = itertools.count()


@dataclass
class SimulationResult:
    trades: List[Trade] = field(default_factory=list)
    inventory_ts: List[Tuple[float, int]] = field(default_factory=list)
    pnl_ts: List[Tuple[float, float]] = field(default_factory=list)
    mid_ts: List[Tuple[float, float]] = field(default_factory=list)
    spread_ts: List[Tuple[float, Optional[int]]] = field(default_factory=list)
    snapshots: List[BookSnapshot] = field(default_factory=list)
    final_inventory: int = 0
    final_pnl: float = 0.0


def _entry(ts: float, event) -> tuple:
    """Build a heap entry: (ts, seq, event)."""
    return (ts, next(_seq_counter), event)


class Simulator:
    """
    Discrete-event simulator for a single-asset limit order book.

    Parameters
    ----------
    flow            : OrderFlowGenerator instance
    strategy        : Strategy instance (or None for no strategy)
    duration        : simulation duration in seconds
    snapshot_interval : seconds between full book snapshots (0 = no snapshots)
    latency         : seconds between a strategy decision and its order
                      reaching the book (applied to submissions and cancels;
                      0 = instantaneous, the pre-latency behaviour)
    """

    def __init__(
        self,
        flow: OrderFlowGenerator,
        strategy: Optional[Strategy] = None,
        duration: float = 28800.0,
        snapshot_interval: float = 60.0,
        initial_mid: int = 1000,
        initial_spread: int = 4,
        initial_depth: int = 10,
        initial_qty_per_level: int = 5,
        latency: float = 0.0,
    ) -> None:
        self.flow = flow
        self.strategy = strategy
        self.duration = duration
        self.snapshot_interval = snapshot_interval
        self.latency = latency
        self.initial_mid = initial_mid
        self.initial_spread = initial_spread
        self.initial_depth = initial_depth
        self.initial_qty_per_level = initial_qty_per_level

    def _seed_book(self, book: LOBBook) -> int:
        """Populate the book with an initial symmetric ladder of limit orders."""
        oid = 1
        half = self.initial_spread // 2
        for i in range(self.initial_depth):
            bid_price = self.initial_mid - half - i
            ask_price = self.initial_mid + half + i
            book.submit_limit(Side.BID, bid_price,
                              self.initial_qty_per_level, oid, 0.0)
            oid += 1
            book.submit_limit(Side.ASK, ask_price,
                              self.initial_qty_per_level, oid, 0.0)
            oid += 1
        return oid

    def run(self) -> SimulationResult:
        """
        Execute the simulation and return results.
        """
        book = LOBBook()
        self._seed_book(book)
        result = SimulationResult()

        if self.strategy is not None:
            self.strategy.reset(ts=0.0)

        heap: list = []

        # Schedule the first flow event
        ts = 0.0
        first_event = self.flow.next_event(ts, book)
        if first_event is not None:
            heapq.heappush(heap, _entry(first_event.timestamp, first_event))

        # Schedule first snapshot
        next_snapshot_ts = self.snapshot_interval if self.snapshot_interval > 0 else float('inf')

        # Last known mid, used to mark PnL when one book side is momentarily
        # empty (mid() is None).  Marking at cash-only in those moments would
        # inject spurious ±inventory*mid jumps into the PnL series.
        last_mid: Optional[float] = book.mid()

        while heap:
            entry = heapq.heappop(heap)
            current_ts, _, event = entry

            if current_ts > self.duration:
                break

            # ── Snapshot ─────────────────────────────────────────────────
            while current_ts >= next_snapshot_ts and next_snapshot_ts <= self.duration:
                snap = book.snapshot(depth=10)
                snap.timestamp = next_snapshot_ts
                result.snapshots.append(snap)
                next_snapshot_ts += self.snapshot_interval

            # ── Process event ─────────────────────────────────────────────
            trades: List[Trade] = []

            if isinstance(event, Order):
                trades = book.submit_limit(
                    event.side, event.price_ticks, event.qty,
                    event.order_id, current_ts,
                )
            elif isinstance(event, MarketOrder):
                trades = book.submit_market(
                    event.side, event.qty, event.order_id, current_ts,
                )
            elif isinstance(event, CancelRequest):
                book.cancel(event.order_id, current_ts)
            elif isinstance(event, ModifyRequest):
                book.reduce_qty(event.order_id, event.delta_qty, current_ts)
            elif isinstance(event, SubmitLimit):
                # Strategy action
                if not book.has_order(event.order_id):
                    trades = book.submit_limit(
                        event.side, event.price_ticks, event.qty,
                        event.order_id, current_ts,
                    )
            elif isinstance(event, Cancel):
                # Strategy cancel action
                book.cancel(event.order_id, current_ts)

            # ── Handle fills ──────────────────────────────────────────────
            result.trades.extend(trades)

            if self.strategy is not None and trades:
                for trade in trades:
                    if self.strategy.is_our_order(trade.passive_order_id):
                        self.strategy.on_fill(
                            order_id=trade.passive_order_id,
                            side=Side.BID if trade.aggressor_side == Side.ASK else Side.ASK,
                            price_ticks=trade.price_ticks,
                            qty=trade.qty,
                            ts=current_ts,
                            maker=True,
                        )
                    # A strategy quote priced through the BBO executes as the
                    # aggressor; those fills must be accounted too.
                    if self.strategy.is_our_order(trade.aggressor_order_id):
                        self.strategy.on_fill(
                            order_id=trade.aggressor_order_id,
                            side=trade.aggressor_side,
                            price_ticks=trade.price_ticks,
                            qty=trade.qty,
                            ts=current_ts,
                            maker=False,
                        )

            # ── Notify strategy ───────────────────────────────────────────
            # Only on external market events.  Notifying after the strategy's
            # own submits/cancels creates a feedback loop: when its quotes are
            # the BBO, each requote moves the mid, triggering another requote
            # at +EPSILON — simulated time freezes while the quote ladder
            # walks away from fair value.
            if self.strategy is not None and not isinstance(event, (SubmitLimit, Cancel)):
                actions = self.strategy.on_book_update(book, current_ts)
                for action in actions:
                    action_ts = current_ts + self.latency + EPSILON
                    if isinstance(action, SubmitLimit):
                        action.timestamp = action_ts
                    elif isinstance(action, Cancel):
                        action.timestamp = action_ts
                    heapq.heappush(heap, _entry(action_ts, action))

            # ── Record time series ────────────────────────────────────────
            mid = book.mid()
            spread = book.spread()

            if mid is not None:
                result.mid_ts.append((current_ts, mid))
                last_mid = mid
            if spread is not None:
                result.spread_ts.append((current_ts, spread))

            if self.strategy is not None:
                result.inventory_ts.append((current_ts, self.strategy.inventory))
                result.pnl_ts.append(
                    (current_ts, self.strategy.pnl(mid if mid is not None else last_mid))
                )

            # ── Schedule next flow event ──────────────────────────────────
            if not isinstance(event, (SubmitLimit, Cancel)):
                next_event = self.flow.next_event(current_ts, book)
                if next_event is not None:
                    heapq.heappush(heap, _entry(next_event.timestamp, next_event))

        # Final snapshot
        final_snap = book.snapshot(depth=10)
        final_snap.timestamp = self.duration
        result.snapshots.append(final_snap)

        if self.strategy is not None:
            final_mid = book.mid()
            result.final_inventory = self.strategy.inventory
            result.final_pnl = self.strategy.pnl(
                final_mid if final_mid is not None else last_mid
            )

        return result


# ── Convenience factory ───────────────────────────────────────────────────────

def run_simulation(
    gamma: float = 0.1,
    sigma: float = 0.05,
    k: float = 1.5,
    T: float = 28800.0,
    lambda_lim: float = 5.0,
    lambda_mkt: float = 1.0,
    lambda_cancel: float = 0.3,
    duration: float = 28800.0,
    snapshot_interval: float = 60.0,
    seed: Optional[int] = 42,
    order_size: int = 1,
    max_inventory: int = 50,
    max_half_spread: int = 50,
) -> SimulationResult:
    """
    Convenience wrapper: build a PoissonFlow + AvellanedaStoikov simulator
    and run it for *duration* seconds.
    """
    flow = PoissonFlow(
        lambda_lim=lambda_lim,
        lambda_mkt=lambda_mkt,
        lambda_cancel=lambda_cancel,
        seed=seed,
    )
    strategy = AvellanedaStoikov(
        gamma=gamma,
        sigma=sigma,
        k=k,
        T=T,
        order_size=order_size,
        max_inventory=max_inventory,
        max_half_spread=max_half_spread,
    )
    sim = Simulator(
        flow=flow,
        strategy=strategy,
        duration=duration,
        snapshot_interval=snapshot_interval,
    )
    return sim.run()
