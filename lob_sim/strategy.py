"""
strategy.py — Market-making strategies.

Abstract base class
-------------------
Strategy.on_book_update(book, ts) -> List[Action]
    Called after every external market event (not after the strategy's own
    actions).  Returns a list of SubmitLimit or Cancel actions to be
    inserted into the event queue.

AvellanedaStoikov
-----------------
Classic mean-variance optimal market-making (Avellaneda & Stoikov 2008).

Reservation price:
    r(t) = mid(t) - q * γ * σ² * (T - t)

Optimal half-spread:
    δ(t) = ½ * γ * σ² * (T - t) + (1/γ) * ln(1 + γ/k)

Quotes:
    bid  = r(t) - δ(t)   (rounded down to nearest tick)
    ask  = r(t) + δ(t)   (rounded up to nearest tick)

where:
    q  = current inventory (positive = long)
    γ  = risk-aversion coefficient
    σ  = (annualised) volatility per tick
    k  = fill-intensity decay parameter
    T  = terminal time (seconds from start of session)
    t  = elapsed time

Inventory is updated on fill notifications passed via on_fill().
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .events import Action, Cancel, Side, SubmitLimit
from .orderbook import LOBBook


# Strategy order IDs start here — above the flow range (1_000_000+) and the
# seeded book (1..N).  Per-instance counter, restarted by reset(), so the
# range never drifts into collision across many runs in one process.
STRATEGY_ID_BASE = 2_000_000


# ── Abstract base ─────────────────────────────────────────────────────────────

class Strategy(ABC):
    """
    Abstract market-making strategy.

    Subclasses implement on_book_update() which returns zero or more Actions.
    """

    @abstractmethod
    def on_book_update(self, book: LOBBook, ts: float) -> List[Action]:
        """Called after every external market event.  Return actions to execute."""
        ...

    @abstractmethod
    def on_fill(
        self,
        order_id: int,
        side: Side,
        price_ticks: int,
        qty: int,
        ts: float,
        maker: bool = True,
    ) -> None:
        """
        Called when one of our orders is (partially) filled.
        maker=True: our resting quote was hit (passive fill).
        maker=False: our order crossed the spread and executed as aggressor.
        """
        ...

    @abstractmethod
    def pnl(self, mid_ticks: Optional[float]) -> float:
        """Return current mark-to-market PnL in ticks."""
        ...


# ── Avellaneda-Stoikov ────────────────────────────────────────────────────────

class AvellanedaStoikov(Strategy):
    """
    Optimal market-making strategy from Avellaneda & Stoikov (2008).

    Parameters
    ----------
    gamma       : risk-aversion coefficient (>0; larger = tighter inventory)
    sigma       : volatility in ticks/sqrt(second)
    k           : fill-intensity decay parameter (from Poisson fill model)
    T           : total session length in seconds
    order_size  : size of each resting quote in contracts
    min_spread  : minimum half-spread in ticks (floor; prevents quotes crossing)
    max_inventory : hard inventory limit; quotes skewed off when breached
    tick_size   : tick size in price units (for rounding)
    maker_fee   : fee in ticks per contract charged on passive fills
                  (negative = rebate, as on maker-taker venues)
    taker_fee   : fee in ticks per contract charged when our order crosses
                  the spread and executes as aggressor
    """

    def __init__(
        self,
        gamma: float = 0.1,
        sigma: float = 0.05,
        k: float = 1.5,
        T: float = 28800.0,   # 8 hours in seconds
        order_size: int = 1,
        min_spread: int = 1,
        max_half_spread: int = 50,
        max_inventory: int = 50,
        tick_size: int = 1,
        maker_fee: float = 0.0,
        taker_fee: float = 0.0,
    ) -> None:
        # max_half_spread caps δ to prevent quotes from wandering far outside
        # the seeded book when T-t is large.  Without this, the AS formula
        # produces half-spreads of hundreds of ticks at session start for
        # typical (σ, γ) values, making the strategy's quotes inaccessible to
        # the background Poisson flow.
        self.max_half_spread = max_half_spread
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.T = T
        self.order_size = order_size
        self.min_spread = min_spread
        self.max_inventory = max_inventory
        self.tick_size = tick_size
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # State
        self._id_counter = itertools.count(STRATEGY_ID_BASE)
        self.inventory: int = 0         # contracts (+ = long)
        self.cash: float = 0.0          # realised cash in ticks

        self._bid_id: Optional[int] = None
        self._ask_id: Optional[int] = None
        self._last_bid_ticks: Optional[int] = None
        self._last_ask_ticks: Optional[int] = None
        self._last_mid: Optional[float] = None

        # Every order we have submitted: oid -> (side, price).  Entries are
        # kept after fills and cancels — under latency a fill can land while
        # our cancel is still in flight, and it must still be recognised as
        # ours.  IDs are never reused within a run, so stale entries are
        # harmless.
        self._our_orders: Dict[int, Tuple[Side, int]] = {}
        # Orders we have issued a cancel for (may not have landed yet)
        self._cancel_sent: set = set()
        # Orders superseded while still in flight (submission not yet landed
        # at requote time): once they land they must be cancelled.  Kept
        # small so the GC pass is O(|stale|), not O(all orders ever).
        self._maybe_stale: set = set()
        # Every order ever submitted this session (for fill-rate / adverse-sel analysis)
        self.all_submitted_ids: set = set()
        self.n_quotes_submitted: int = 0

        # Session start time (set on first call)
        self._t0: Optional[float] = None

    # ── AS formulas ─────────────────────────────────────────────────────────

    def reservation_price(self, mid: float, t: float) -> float:
        """
        r(t) = mid - q * γ * σ² * (T - t)
        """
        tau = max(0.0, self.T - (t - (self._t0 or 0.0)))
        return mid - self.inventory * self.gamma * (self.sigma ** 2) * tau

    def half_spread(self, t: float) -> float:
        """
        δ(t) = ½ * γ * σ² * (T - t) + (1/γ) * ln(1 + γ/k)
        """
        tau = max(0.0, self.T - (t - (self._t0 or 0.0)))
        term1 = 0.5 * self.gamma * (self.sigma ** 2) * tau
        term2 = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        return term1 + term2

    def fair_value(self, book: LOBBook) -> Optional[float]:
        """
        Price the quotes are centred on.  The base class uses the
        instantaneous mid; subclasses may override with a smoothed or
        signal-adjusted estimate (see FairValueAvellanedaStoikov).
        Called exactly once per compute_quotes call.
        """
        return book.mid()

    def compute_quotes(
        self, book: LOBBook, ts: float
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Return (bid_ticks, ask_ticks) or (None, None) if no fair value exists.
        Prices are rounded to nearest tick.
        """
        fair = self.fair_value(book)
        if fair is None:
            return None, None

        r = self.reservation_price(fair, ts)
        # Clamp reservation price to within max_half_spread of fair value.
        # Without this, large inventory combined with large σ²·τ pushes r
        # hundreds of ticks away, making quotes inaccessible to the
        # background flow and allowing the strategy's extreme orders to
        # accidentally become the BBO when the book thins.
        r = max(fair - self.max_half_spread, min(fair + self.max_half_spread, r))

        # Fee-floored: the half-spread must at least cover the per-side
        # maker fee, or every round trip locks in a loss.  (A rebate —
        # negative maker_fee — never tightens the floor below min_spread.)
        delta = max(self.min_spread, self.maker_fee,
                    min(self.half_spread(ts), self.max_half_spread))

        raw_bid = r - delta
        raw_ask = r + delta

        # Round to ticks
        bid_ticks = max(1, int(math.floor(raw_bid / self.tick_size) * self.tick_size))
        ask_ticks = max(bid_ticks + self.tick_size,
                        int(math.ceil(raw_ask / self.tick_size) * self.tick_size))

        # Withdraw the relevant side when inventory limit is reached
        if self.inventory >= self.max_inventory:
            bid_ticks = 1
        if self.inventory <= -self.max_inventory:
            ask_ticks = int(fair) + self.max_half_spread + 1

        return bid_ticks, ask_ticks

    # ── Strategy interface ───────────────────────────────────────────────────

    def _collect_stale(self, book: LOBBook, ts: float) -> List[Action]:
        """
        Cancel superseded quotes that have landed in the book since being
        replaced.  Exact for order_size=1; with larger sizes a partially
        filled superseded remnant may rest until filled or swept.
        """
        actions: List[Action] = []
        for oid in list(self._maybe_stale):
            if oid in self._cancel_sent:
                self._maybe_stale.discard(oid)
            elif book.has_order(oid):
                actions.append(Cancel(order_id=oid, timestamp=ts))
                self._cancel_sent.add(oid)
                self._maybe_stale.discard(oid)
        return actions

    def on_book_update(self, book: LOBBook, ts: float) -> List[Action]:
        if self._t0 is None:
            self._t0 = ts

        actions: List[Action] = []
        mid = book.mid()
        if mid is None:
            return actions

        bid_ticks, ask_ticks = self.compute_quotes(book, ts)
        if bid_ticks is None or ask_ticks is None:
            return actions

        # Garbage-collect superseded quotes that have since landed.  Only
        # populated when submissions were in flight during a requote
        # (latency > 0); a no-op at zero latency.
        actions.extend(self._collect_stale(book, ts))

        # Determine if we need to requote
        mid_moved = (self._last_mid is None or
                     abs(mid - self._last_mid) >= self.tick_size)
        bid_changed = bid_ticks != self._last_bid_ticks
        ask_changed = ask_ticks != self._last_ask_ticks

        if not (mid_moved or bid_changed or ask_changed):
            return actions   # no requote needed (may still carry GC cancels)

        # Cancel existing quotes; a quote still in flight (not yet landed)
        # cannot be cancelled yet, so mark it for GC once it lands
        if self._bid_id is not None:
            if book.has_order(self._bid_id):
                actions.append(Cancel(order_id=self._bid_id, timestamp=ts))
                self._cancel_sent.add(self._bid_id)
            else:
                self._maybe_stale.add(self._bid_id)
            self._bid_id = None

        if self._ask_id is not None:
            if book.has_order(self._ask_id):
                actions.append(Cancel(order_id=self._ask_id, timestamp=ts))
                self._cancel_sent.add(self._ask_id)
            else:
                self._maybe_stale.add(self._ask_id)
            self._ask_id = None

        # Submit new quotes
        if self.inventory < self.max_inventory:
            new_bid_id = next(self._id_counter)
            actions.append(SubmitLimit(
                side=Side.BID,
                price_ticks=bid_ticks,
                qty=self.order_size,
                order_id=new_bid_id,
                timestamp=ts,
            ))
            self._bid_id = new_bid_id
            self._our_orders[new_bid_id] = (Side.BID, bid_ticks)
            self.all_submitted_ids.add(new_bid_id)
            self.n_quotes_submitted += 1

        if self.inventory > -self.max_inventory:
            new_ask_id = next(self._id_counter)
            actions.append(SubmitLimit(
                side=Side.ASK,
                price_ticks=ask_ticks,
                qty=self.order_size,
                order_id=new_ask_id,
                timestamp=ts,
            ))
            self._ask_id = new_ask_id
            self._our_orders[new_ask_id] = (Side.ASK, ask_ticks)
            self.all_submitted_ids.add(new_ask_id)
            self.n_quotes_submitted += 1

        self._last_mid = mid
        self._last_bid_ticks = bid_ticks
        self._last_ask_ticks = ask_ticks

        return actions

    def on_fill(
        self,
        order_id: int,
        side: Side,
        price_ticks: int,
        qty: int,
        ts: float,
        maker: bool = True,
    ) -> None:
        """Update inventory and cash on fill."""
        if side == Side.BID:
            # We bought qty contracts at price_ticks
            self.inventory += qty
            self.cash -= price_ticks * qty
        else:
            # We sold qty contracts at price_ticks
            self.inventory -= qty
            self.cash += price_ticks * qty

        self.cash -= (self.maker_fee if maker else self.taker_fee) * qty

        # A filled superseded order no longer needs garbage collection
        # (exact for order_size=1, where every fill is a full fill)
        self._maybe_stale.discard(order_id)

        # The _our_orders entry is kept even after a fill: a partially filled
        # quote still rests in the book and later fills on it must still be
        # recognised as ours by the simulator.

        # Reset cached quote IDs so we re-submit on next book update
        if order_id == self._bid_id:
            self._bid_id = None
            self._last_bid_ticks = None
        elif order_id == self._ask_id:
            self._ask_id = None
            self._last_ask_ticks = None

    def pnl(self, mid_ticks: Optional[float]) -> float:
        """Cash + mark-to-market value of inventory."""
        if mid_ticks is None:
            return self.cash
        return self.cash + self.inventory * mid_ticks

    def unrealized_pnl(self, mid_ticks: float) -> float:
        return self.inventory * mid_ticks

    def realized_pnl(self) -> float:
        return self.cash

    # ── Convenience ─────────────────────────────────────────────────────────

    def is_our_order(self, order_id: int) -> bool:
        return order_id in self._our_orders

    def reset(self, ts: float = 0.0) -> None:
        """Reset strategy state for a new simulation run."""
        self._id_counter = itertools.count(STRATEGY_ID_BASE)
        self.inventory = 0
        self.cash = 0.0
        self._bid_id = None
        self._ask_id = None
        self._last_bid_ticks = None
        self._last_ask_ticks = None
        self._last_mid = None
        self._our_orders.clear()
        self._cancel_sent.clear()
        self._maybe_stale.clear()
        self.all_submitted_ids.clear()
        self.n_quotes_submitted = 0
        self._t0 = ts


# ── Fair-value anchored variant ──────────────────────────────────────────────

class FairValueAvellanedaStoikov(AvellanedaStoikov):
    """
    AS variant that centres quotes on an exponentially weighted moving
    average of the mid instead of the instantaneous mid.

    Motivation: quoting around the instantaneous mid re-centres onto every
    dislocated price immediately after a sweep, so the strategy
    systematically buys tops and sells bottoms (quote-chasing).  Anchoring
    to a slow EWMA lets quotes lag transient dislocations.  In this
    simulator's flow model that flips the naive AS maker from a consistent
    loser to profitable (see README, "Improving on naive AS").

    Parameters
    ----------
    alpha : EWMA weight applied once per book update.  Smaller = slower
            fair value.  At ~6 external events/second, alpha=0.005 gives an
            effective averaging horizon on the order of half a minute.
    """

    def __init__(self, alpha: float = 0.005, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self._fair: Optional[float] = None

    def fair_value(self, book: LOBBook) -> Optional[float]:
        mid = book.mid()
        if mid is None:
            return self._fair          # quote around last known fair value
        if self._fair is None:
            self._fair = mid
        else:
            self._fair = (1.0 - self.alpha) * self._fair + self.alpha * mid
        return self._fair

    def reset(self, ts: float = 0.0) -> None:
        super().reset(ts)
        self._fair = None


class QueueAwareFairValueAvellanedaStoikov(FairValueAvellanedaStoikov):
    """
    Fair-value variant with queue-position-aware requoting.

    Two changes versus the base requote logic:
    1. Per-side: only the side whose desired price actually moved is
       cancelled and re-submitted.  (The base cancels both sides whenever
       anything changes, forfeiting FIFO queue position on the unchanged
       side for nothing.)
    2. Patience: a side is left alone when the desired price is within
       queue_patience ticks of the resting quote.  Cancel-replace sends the
       order to the back of the FIFO queue, so a small price improvement is
       not worth the lost priority (cf. Cont & de Larrard 2013 on the value
       of queue position).

    queue_patience=0 still keeps unchanged sides in place; larger values
    trade price accuracy for queue seniority.
    """

    def __init__(self, queue_patience: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.queue_patience = queue_patience

    def on_book_update(self, book: LOBBook, ts: float) -> List[Action]:
        if self._t0 is None:
            self._t0 = ts

        actions: List[Action] = []
        mid = book.mid()
        if mid is None:
            return actions

        bid_ticks, ask_ticks = self.compute_quotes(book, ts)
        if bid_ticks is None or ask_ticks is None:
            return actions

        # Garbage-collect superseded quotes (same as base class)
        actions.extend(self._collect_stale(book, ts))

        def needs_requote(current_id: Optional[int],
                          last_px: Optional[int], new_px: int) -> bool:
            if current_id is None or not book.has_order(current_id):
                return True                    # no live quote on this side
            if last_px is None:
                return True
            return abs(new_px - last_px) > self.queue_patience

        if needs_requote(self._bid_id, self._last_bid_ticks, bid_ticks):
            if self._bid_id is not None:
                if book.has_order(self._bid_id):
                    actions.append(Cancel(order_id=self._bid_id, timestamp=ts))
                    self._cancel_sent.add(self._bid_id)
                else:
                    self._maybe_stale.add(self._bid_id)   # still in flight
            self._bid_id = None
            if self.inventory < self.max_inventory:
                new_bid_id = next(self._id_counter)
                actions.append(SubmitLimit(
                    side=Side.BID, price_ticks=bid_ticks,
                    qty=self.order_size, order_id=new_bid_id, timestamp=ts,
                ))
                self._bid_id = new_bid_id
                self._our_orders[new_bid_id] = (Side.BID, bid_ticks)
                self.all_submitted_ids.add(new_bid_id)
                self.n_quotes_submitted += 1
                self._last_bid_ticks = bid_ticks

        if needs_requote(self._ask_id, self._last_ask_ticks, ask_ticks):
            if self._ask_id is not None:
                if book.has_order(self._ask_id):
                    actions.append(Cancel(order_id=self._ask_id, timestamp=ts))
                    self._cancel_sent.add(self._ask_id)
                else:
                    self._maybe_stale.add(self._ask_id)   # still in flight
            self._ask_id = None
            if self.inventory > -self.max_inventory:
                new_ask_id = next(self._id_counter)
                actions.append(SubmitLimit(
                    side=Side.ASK, price_ticks=ask_ticks,
                    qty=self.order_size, order_id=new_ask_id, timestamp=ts,
                ))
                self._ask_id = new_ask_id
                self._our_orders[new_ask_id] = (Side.ASK, ask_ticks)
                self.all_submitted_ids.add(new_ask_id)
                self.n_quotes_submitted += 1
                self._last_ask_ticks = ask_ticks

        self._last_mid = mid
        return actions


class ImbalanceFairValueAvellanedaStoikov(FairValueAvellanedaStoikov):
    """
    Fair-value variant that additionally skews the anchor by top-of-book
    imbalance:

        fair' = EWMA(mid) + beta * (Qb - Qa) / (Qb + Qa)

    where Qb/Qa are total quantities on the top *imbalance_depth* levels of
    each side.  Rationale: a one-sided book (e.g. asks depleted by a sweep)
    means the next market order moves the price further in that direction,
    so expected short-horizon drift points toward the thin side.

    The adjustment is applied to the returned value only — it does not feed
    back into the EWMA state.
    """

    def __init__(self, beta: float = 1.0, imbalance_depth: int = 3,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.imbalance_depth = imbalance_depth

    def fair_value(self, book: LOBBook) -> Optional[float]:
        fair = super().fair_value(book)
        if fair is None:
            return None
        snap = book.snapshot(depth=self.imbalance_depth)
        qb = sum(q for _, q in snap.bids)
        qa = sum(q for _, q in snap.asks)
        if qb + qa > 0:
            fair += self.beta * (qb - qa) / (qb + qa)
        return fair
