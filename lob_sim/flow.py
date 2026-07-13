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

Message file columns (space-separated):
    timestamp, event_type, order_id, size, price, direction
where direction 1=buy, -1=sell; price is in dollars×10000 (integer cents).

We re-scale timestamps relative to the file start so t=0 corresponds to
the first event.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from .events import CancelRequest, MarketOrder, Order, Side
from .orderbook import LOBBook


Event = Union[Order, MarketOrder, CancelRequest]

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

class LOBSTERReplay(OrderFlowGenerator):
    """
    Replay events from a LOBSTER message file.

    LOBSTER message format (CSV, no header):
        time, type, order_id, size, price, direction
    where:
        type: 1=new limit, 2=partial cancel, 3=full cancel, 4=exec visible,
              5=exec hidden, 7=trading halt
        price: in 10000ths of a dollar (i.e., multiply by 1e-4 to get dollars)
        direction: 1=buy, -1=sell

    We map LOBSTER prices to internal ticks by dividing by a configurable
    tick_size_lobster (default 100, i.e. 1 cent ticks internally).
    """

    def __init__(
        self,
        message_file: str,
        tick_divisor: int = 100,
        time_scale: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        message_file : path to LOBSTER messages CSV
        tick_divisor : divide raw LOBSTER price by this to get tick integer
                       (LOBSTER prices are in 10000ths of $, so divisor=100
                        gives 1-cent ticks)
        time_scale   : multiply all timestamps by this (e.g. speed up replay)
        """
        self.tick_divisor = tick_divisor
        self.time_scale = time_scale
        self._id_counter = itertools.count(FLOW_ID_BASE)
        self._events: List[Tuple[float, Event]] = []
        self._idx = 0
        self._t0: Optional[float] = None
        self._load(message_file)

    def _load(self, path: str) -> None:
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    ts_raw = float(row[0])
                    etype = int(row[1])
                    oid = int(row[2])
                    size = int(row[3])
                    price_raw = int(row[4])
                    direction = int(row[5])
                except (ValueError, IndexError):
                    continue

                if self._t0 is None:
                    self._t0 = ts_raw
                ts = (ts_raw - self._t0) * self.time_scale

                side = Side.BID if direction == 1 else Side.ASK
                price_ticks = max(1, price_raw // self.tick_divisor)

                if etype == 1:
                    event: Event = Order(
                        order_id=oid,
                        side=side,
                        price_ticks=price_ticks,
                        qty=size,
                        timestamp=ts,
                    )
                elif etype in (2, 3):
                    event = CancelRequest(order_id=oid, timestamp=ts)
                elif etype in (4, 5):
                    # Execution — treated as market aggressor hitting passive
                    event = MarketOrder(
                        order_id=next(self._id_counter),
                        side=side,
                        qty=size,
                        timestamp=ts,
                    )
                else:
                    continue   # skip halt / other types

                self._events.append((ts, event))

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
