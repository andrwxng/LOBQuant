"""
flow.py — Synthetic order-flow generators.

Both generators expose:
    next_event(ts: float, book: LOBBook) -> Event

where Event is one of Order | MarketOrder | CancelRequest.

PoissonFlow
-----------
Models the Cont-Stoikov-Talreja (2010) framework.  Three independent
Poisson processes generate:

  1. Limit orders: arrive at distance d ticks from the BBO.
     d is drawn from a geometric distribution with parameter p_geom.
     Rate at distance d: λ_lim(d) = λ_lim_base * p_geom * (1-p_geom)^(d-1)
     (equivalently, the first event of a Poisson process with rate λ_lim_base
     occurs at tick distance d).

  2. Market orders: arrive at rate λ_mkt (symmetric for buy/sell).

  3. Cancellations: each resting order is cancelled at rate λ_cancel.
     We approximate this by selecting a random resting order at each
     cancellation event.

LOBSTERReplay
-------------
Streams events from a LOBSTER-format message file.  LOBSTER provides
per-stock order book data with microsecond timestamps.

Message file columns (CSV, no header):
    timestamp, event_type, order_id, size, price, direction
where price is in 10000ths of a dollar and direction is the side of the
order the message concerns.  For new-order, cancel, and modify messages
that is the order's own side.  For execution messages (type 4/5) LOBSTER's
convention is that `direction` gives the side of the *resting* order being
hit, so the aggressor is the opposite side — see LOBSTERReplay for detail.

We re-scale timestamps relative to the file start so t=0 corresponds to
the first event.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from .events import CancelRequest, MarketOrder, ModifyRequest, Order, Side
from .orderbook import LOBBook


Event = Union[Order, MarketOrder, CancelRequest, ModifyRequest]

# Flow order IDs start here; strategy IDs start at 2_000_000 (strategy.py)
# and the seeded book uses 1..N.  Counters are per-instance so that fresh
# generators restart the range — a shared module counter grows across runs
# and eventually collides with the strategy range.
FLOW_ID_BASE = 1_000_000


# ── Abstract interface ────────────────────────────────────────────────────────

class OrderFlowGenerator(ABC):
    @abstractmethod
    def next_event(self, ts: float, book: LOBBook) -> Optional[Event]:
        """
        Generate the next event given current time *ts* and book state.
        Returns None when the generator is exhausted (replay finished, etc.).
        """
        ...

    @abstractmethod
    def peek_next_ts(self, current_ts: float) -> float:
        """
        Return the timestamp of the next event without consuming it.
        Used by the simulator to schedule arrivals.
        """
        ...


# ── PoissonFlow ───────────────────────────────────────────────────────────────

class PoissonFlow(OrderFlowGenerator):
    """
    Poisson arrival model following Cont-Stoikov-Talreja (2010).

    Parameters
    ----------
    lambda_lim   : base arrival rate of limit orders (events/second)
    lambda_mkt   : arrival rate of market orders (events/second)
    lambda_cancel: per-order cancellation rate (events/second/order)
    p_geom       : parameter of geometric distribution for depth placement
    min_qty, max_qty : uniform range for order quantities
    mid_init_ticks   : initial reference mid price (ticks)
    half_spread_init : initial half-spread for reference price when book empty
    sigma_ref    : volatility of the reference mid-price GBM (ticks/√s).
                   Drives the "fundamental" price process that order placement
                   is anchored to, keeping the book stable regardless of
                   liquidity depletion.
    seed         : RNG seed for reproducibility

    Design note
    -----------
    Limit orders are placed relative to a *reference mid price* (ref_mid)
    that follows a GBM with diffusion sigma_ref, rather than relative to the
    current BBO.  This prevents the pathological case where the book is
    depleted by market orders and the strategy's extreme quotes become the BBO,
    anchoring all subsequent arrivals far from economic fair value.

    The current BBO is still used to set prices when it is close to ref_mid
    (within max_bbo_deviation ticks).  When the BBO deviates far from ref_mid,
    ref_mid overrides it, providing mean reversion.
    """

    def __init__(
        self,
        lambda_lim: float = 5.0,
        lambda_mkt: float = 1.0,
        lambda_cancel: float = 0.5,
        p_geom: float = 0.3,
        min_qty: int = 1,
        max_qty: int = 10,
        mid_init_ticks: int = 1000,
        half_spread_init: int = 2,
        sigma_ref: float = 0.5,
        max_bbo_deviation: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self.lambda_lim = lambda_lim
        self.lambda_mkt = lambda_mkt
        self.lambda_cancel = lambda_cancel
        self.p_geom = p_geom
        self.min_qty = min_qty
        self.max_qty = max_qty
        self.mid_init_ticks = mid_init_ticks
        self.half_spread_init = half_spread_init
        self.sigma_ref = sigma_ref
        self.max_bbo_deviation = max_bbo_deviation

        self._rng = random.Random(seed)
        self._id_counter = itertools.count(FLOW_ID_BASE)
        self._next_ts: Optional[float] = None

        # Reference mid-price state: follows GBM, stepped forward each event
        self._ref_mid: float = float(mid_init_ticks)
        self._ref_ts: float = 0.0

    # ── Internal helpers ────────────────────────────────────────────────────

    def _exp(self, rate: float) -> float:
        """Draw inter-arrival time from Exp(rate)."""
        return self._rng.expovariate(rate)

    def _geometric(self) -> int:
        """Geometric distribution: number of trials until first success."""
        p = self.p_geom
        # geometric PMF: P(X=k) = (1-p)^(k-1)*p
        # CDF inversion: k = ceil(log(U)/log(1-p))
        u = self._rng.random()
        return max(1, math.ceil(math.log(u) / math.log(1 - p)))

    def _sample_qty(self) -> int:
        return self._rng.randint(self.min_qty, self.max_qty)

    def _total_rate(self, n_resting: int) -> float:
        """Total event rate (limit + market + cancel)."""
        return self.lambda_lim + self.lambda_mkt + self.lambda_cancel * n_resting

    def _advance_ref_mid(self, next_ts: float) -> None:
        """Step the reference mid-price forward via GBM (arithmetic approximation)."""
        dt = next_ts - self._ref_ts
        if dt <= 0:
            return
        # Arithmetic Brownian motion: dS = sigma * dW
        shock = self._rng.gauss(0.0, 1.0) * self.sigma_ref * math.sqrt(dt)
        self._ref_mid = max(1.0, self._ref_mid + shock)
        self._ref_ts = next_ts

    def _compute_next_ts(self, current_ts: float, book: LOBBook) -> Tuple[float, Event]:
        """Sample the next event and its timestamp."""
        n_resting = len(book._orders)
        rate_lim = self.lambda_lim
        rate_mkt = self.lambda_mkt
        rate_cancel = self.lambda_cancel * max(n_resting, 1)
        total_rate = rate_lim + rate_mkt + rate_cancel

        dt = self._exp(total_rate)
        next_ts = current_ts + dt

        # Advance the reference mid-price GBM to next_ts
        self._advance_ref_mid(next_ts)

        u = self._rng.random() * total_rate
        if u < rate_lim:
            event = self._sample_limit(next_ts, book)
        elif u < rate_lim + rate_mkt:
            event = self._sample_market(next_ts, book)
        else:
            event = self._sample_cancel(next_ts, book)

        return next_ts, event

    def _reference_bbo(self, book: LOBBook) -> Tuple[float, float]:
        """
        Return (bb, ba) to use for order placement.
        If the current BBO is within max_bbo_deviation of ref_mid, use it.
        Otherwise fall back to ref_mid ± half_spread_init, preventing
        extreme-price orders when the book is depleted.
        """
        ref = self._ref_mid
        half = self.half_spread_init
        bb_ref = ref - half
        ba_ref = ref + half

        bbo_mid = book.mid()
        if bbo_mid is not None and abs(bbo_mid - ref) <= self.max_bbo_deviation:
            bb_raw = book.best_bid()
            ba_raw = book.best_ask()
            bb = float(bb_raw) if bb_raw is not None else bb_ref
            ba = float(ba_raw) if ba_raw is not None else ba_ref
        else:
            bb = bb_ref
            ba = ba_ref

        return bb, ba

    def _sample_limit(self, ts: float, book: LOBBook) -> Order:
        """Generate a limit order at a random depth from the reference BBO."""
        side = self._rng.choice([Side.BID, Side.ASK])
        depth = self._geometric()
        qty = self._sample_qty()

        bb, ba = self._reference_bbo(book)

        if side == Side.BID:
            price_ticks = max(1, int(bb) - (depth - 1))
        else:
            price_ticks = max(1, int(ba) + (depth - 1))

        return Order(
            order_id=next(self._id_counter),
            side=side,
            price_ticks=price_ticks,
            qty=qty,
            timestamp=ts,
        )

    def _sample_market(self, ts: float, book: LOBBook) -> MarketOrder:
        """Generate a market order."""
        side = self._rng.choice([Side.BID, Side.ASK])
        qty = self._sample_qty()
        return MarketOrder(
            order_id=next(self._id_counter),
            side=side,
            qty=qty,
            timestamp=ts,
        )

    def _sample_cancel(self, ts: float, book: LOBBook) -> Optional[CancelRequest]:
        """Cancel a randomly chosen resting order."""
        if not book._orders:
            # No resting orders; re-sample as a limit order instead
            return self._sample_limit(ts, book)
        oid = self._rng.choice(list(book._orders.keys()))
        return CancelRequest(order_id=oid, timestamp=ts)

    # ── Public interface ────────────────────────────────────────────────────

    def peek_next_ts(self, current_ts: float) -> float:
        # We can't peek without a book reference; return current_ts + tiny dt
        # The simulator should call next_event() directly.
        if self._next_ts is None:
            return current_ts + self._exp(self.lambda_lim + self.lambda_mkt)
        return self._next_ts

    def next_event(self, ts: float, book: LOBBook) -> Optional[Event]:
        # The sample methods stamp the drawn timestamp onto the event at
        # construction, so no post-hoc attribute injection is needed.
        _, event = self._compute_next_ts(ts, book)
        return event

    def generate_sequence(
        self,
        n_events: int,
        book: LOBBook,
        start_ts: float = 0.0,
    ) -> List[Tuple[float, Event]]:
        """
        Generate *n_events* events.  Returns list of (timestamp, event) tuples
        in ascending timestamp order.
        """
        events: List[Tuple[float, Event]] = []
        ts = start_ts
        for _ in range(n_events):
            next_ts, event = self._compute_next_ts(ts, book)
            events.append((next_ts, event))
            ts = next_ts
        return events


# ── LOBSTERReplay ─────────────────────────────────────────────────────────────

# Reserved ID range for replayed/synthetic-seed orders, disjoint from the
# seeded book (1..N), PoissonFlow (FLOW_ID_BASE=1_000_000+), and strategies
# (2_000_000+).  Raw LOBSTER order IDs are never used directly as internal
# order IDs — see LOBSTERReplay._new_internal_id.
LOBSTER_ID_BASE = 3_000_000

# LOBSTER marks an absent price level with ±9999999999; anything at or
# beyond this magnitude is treated as "no level" rather than a real price.
_LOBSTER_EMPTY_LEVEL_SENTINEL = 9_999_999_999


class LOBSTERReplay(OrderFlowGenerator):
    """
    Replay events from a LOBSTER message file.

    LOBSTER message format (CSV, no header):
        time, type, order_id, size, price, direction
    where:
        type: 1=new limit, 2=partial cancel (reduce qty), 3=full cancel
              (delete), 4=execution against a visible order, 5=execution
              against a hidden order, 7=trading halt
        price: in 10000ths of a dollar (i.e., multiply by 1e-4 to get dollars)
        direction: for types 1/2/3, the side of the order the message
                   concerns.  For types 4/5, LOBSTER documents this as the
                   side of the *resting* (passive) order being executed —
                   so the incoming aggressor is on the opposite side.  A
                   direction=1 (buy limit order) execution means a sell
                   aggressor hit it, and vice versa.

    Order identity
    --------------
    LOBSTER order IDs are remapped to a reserved internal range
    (LOBSTER_ID_BASE+) so they cannot collide with the flow, strategy, or
    seeded-book ID ranges.  A message can reference an order that was
    already resting before the replay window began (e.g. placed pre-market)
    and therefore never appeared as a type-1 "new order" message in this
    file.  Without `orderbook_file`, such messages have no way to resolve
    to an internal order and are dropped (see `n_unresolved`).  With
    `orderbook_file`, the first row seeds the book with one synthetic
    resting order per non-empty price level (aggregate quantity — LOBSTER's
    orderbook file does not expose individual pre-existing order identity,
    only per-level totals), and unresolvable messages fall back to the
    seeded order at their (side, price) instead of being dropped.

    Hidden executions (type 5) match against liquidity with no
    representation in the visible book.  Replaying them as ordinary market
    orders would incorrectly consume visible depth, so they are not turned
    into events; they are recorded in `hidden_trades` instead.
    """

    def __init__(
        self,
        message_file: str,
        orderbook_file: Optional[str] = None,
        n_levels: int = 10,
        tick_divisor: int = 100,
        time_scale: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        message_file  : path to LOBSTER messages CSV
        orderbook_file : path to the matching LOBSTER orderbook CSV.  When
                         given, its first row seeds the book (see class
                         docstring).  Columns per level: ask_price,
                         ask_size, bid_price, bid_size, repeated n_levels
                         times — LOBSTER's standard orderbook-file layout.
        n_levels      : number of price levels present in orderbook_file
        tick_divisor  : divide raw LOBSTER price by this to get tick integer
                        (LOBSTER prices are in 10000ths of $, so divisor=100
                         gives 1-cent ticks)
        time_scale    : multiply all timestamps by this (e.g. speed up replay)

        Attributes set after loading
        -----------------------------
        hidden_trades : list of (ts, resting_side, price_ticks, qty) for
                        type-5 hidden executions — not replayed as book
                        events (see class docstring), exposed here for
                        anyone computing total traded volume.
        n_unresolved  : count of type-2/3 messages whose order_id could not
                        be resolved to an internal order (no orderbook_file,
                        and the id was never introduced by an earlier
                        type-1 message in this file) and were dropped.
        """
        self.tick_divisor = tick_divisor
        self.time_scale = time_scale
        self._id_counter = itertools.count(LOBSTER_ID_BASE)
        self._id_map: Dict[int, int] = {}
        self._level_seed_ids: Dict[Tuple[Side, int], int] = {}
        self.hidden_trades: List[Tuple[float, Side, int, int]] = []
        self.n_unresolved = 0
        self._events: List[Tuple[float, Event]] = []
        self._idx = 0
        self._t0: Optional[float] = None

        if orderbook_file is not None:
            self._load_seed(orderbook_file, n_levels)
        self._load_messages(message_file)

    # ── Loading ──────────────────────────────────────────────────────────

    def _new_internal_id(self, raw_id: int) -> int:
        """Map (or remap, on ID reuse) a raw LOBSTER order_id to an internal
        id in the reserved range."""
        iid = next(self._id_counter)
        self._id_map[raw_id] = iid
        return iid

    def _resolve_id(self, raw_id: int, side: Side,
                    price_ticks: int) -> Optional[int]:
        """Look up the internal id for a raw LOBSTER order_id, falling back
        to the seeded synthetic order at (side, price) if the raw id was
        never introduced by a type-1 message in this file."""
        if raw_id in self._id_map:
            return self._id_map[raw_id]
        return self._level_seed_ids.get((side, price_ticks))

    def _load_seed(self, path: str, n_levels: int) -> None:
        """Seed the book from the first row of a LOBSTER orderbook file: one
        synthetic resting order per non-empty level, at ts=0.0 (replayed
        before any message)."""
        with open(path, newline='') as f:
            row = next(csv.reader(f), None)
        if row is None:
            return

        for i in range(n_levels):
            base = 4 * i
            if base + 3 >= len(row):
                break
            try:
                ask_price_raw = int(row[base])
                ask_size = int(row[base + 1])
                bid_price_raw = int(row[base + 2])
                bid_size = int(row[base + 3])
            except (ValueError, IndexError):
                continue

            if ask_size > 0 and abs(ask_price_raw) < _LOBSTER_EMPTY_LEVEL_SENTINEL:
                self._add_seed_level(Side.ASK, ask_price_raw, ask_size)
            if bid_size > 0 and abs(bid_price_raw) < _LOBSTER_EMPTY_LEVEL_SENTINEL:
                self._add_seed_level(Side.BID, bid_price_raw, bid_size)

    def _add_seed_level(self, side: Side, price_raw: int, size: int) -> None:
        price_ticks = max(1, price_raw // self.tick_divisor)
        iid = next(self._id_counter)
        self._level_seed_ids[(side, price_ticks)] = iid
        self._events.append((0.0, Order(
            order_id=iid, side=side, price_ticks=price_ticks,
            qty=size, timestamp=0.0,
        )))

    def _load_messages(self, path: str) -> None:
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    ts_raw = float(row[0])
                    etype = int(row[1])
                    raw_oid = int(row[2])
                    size = int(row[3])
                    price_raw = int(row[4])
                    direction = int(row[5])
                except (ValueError, IndexError):
                    continue

                if self._t0 is None:
                    self._t0 = ts_raw
                ts = (ts_raw - self._t0) * self.time_scale

                order_side = Side.BID if direction == 1 else Side.ASK
                price_ticks = max(1, price_raw // self.tick_divisor)

                if etype == 1:
                    iid = self._new_internal_id(raw_oid)
                    event: Event = Order(
                        order_id=iid, side=order_side,
                        price_ticks=price_ticks, qty=size, timestamp=ts,
                    )
                elif etype == 2:
                    iid = self._resolve_id(raw_oid, order_side, price_ticks)
                    if iid is None:
                        self.n_unresolved += 1
                        continue
                    event = ModifyRequest(order_id=iid, delta_qty=size,
                                          timestamp=ts)
                elif etype == 3:
                    iid = self._resolve_id(raw_oid, order_side, price_ticks)
                    if iid is None:
                        self.n_unresolved += 1
                        continue
                    event = CancelRequest(order_id=iid, timestamp=ts)
                elif etype == 4:
                    # Visible execution: aggressor is opposite the resting
                    # order's side.  The aggressor has no raw LOBSTER id
                    # (the message's order_id names the resting order), so
                    # it gets a fresh internal id like any synthetic
                    # aggressor.
                    event = MarketOrder(
                        order_id=next(self._id_counter),
                        side=order_side.opposite(), qty=size, timestamp=ts,
                    )
                elif etype == 5:
                    # Hidden execution — see class docstring.
                    self.hidden_trades.append(
                        (ts, order_side, price_ticks, size))
                    continue
                else:
                    continue   # skip halt (7) / other types

                self._events.append((ts, event))

    # ── Public interface ────────────────────────────────────────────────────

    def peek_next_ts(self, current_ts: float) -> float:
        if self._idx >= len(self._events):
            return float('inf')
        return self._events[self._idx][0]

    def next_event(self, ts: float, book: LOBBook) -> Optional[Event]:
        if self._idx >= len(self._events):
            return None
        _, event = self._events[self._idx]
        self._idx += 1
        return event

    def reset(self) -> None:
        self._idx = 0
