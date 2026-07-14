"""
test_strategy.py — Tests for the Avellaneda-Stoikov strategy.

Key checks:
  * Reservation price formula with known inputs
  * Half-spread formula with known inputs
  * Quote prices match AS formulas (rounded to ticks)
  * Inventory update on fill
  * PnL accounting
  * Actions returned (cancel + re-submit) when mid changes
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from lob_sim.events import Cancel, Side, SubmitLimit
from lob_sim.orderbook import LOBBook
from lob_sim.strategy import (
    AvellanedaStoikov, FairValueAvellanedaStoikov,
    ImbalanceFairValueAvellanedaStoikov,
    QueueAwareFairValueAvellanedaStoikov,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def flat_book(mid: int = 1000, spread: int = 4) -> LOBBook:
    """Create a book with one bid and one ask to establish a mid."""
    book = LOBBook()
    book.submit_limit(Side.BID, mid - spread // 2, 100, 1, 0.0)
    book.submit_limit(Side.ASK, mid + spread // 2, 100, 2, 0.0)
    return book


# ── formula verification ──────────────────────────────────────────────────────

class TestASFormulas:
    """Verify exact AS formula values with known parameters."""

    GAMMA = 0.1
    SIGMA = 2.0
    K = 1.5
    T = 1000.0

    def _strat(self, inventory: int = 0) -> AvellanedaStoikov:
        s = AvellanedaStoikov(
            gamma=self.GAMMA,
            sigma=self.SIGMA,
            k=self.K,
            T=self.T,
        )
        s._t0 = 0.0
        s.inventory = inventory
        return s

    def test_reservation_price_zero_inventory(self):
        """With q=0, reservation price = mid."""
        s = self._strat(inventory=0)
        mid = 1000.0
        r = s.reservation_price(mid, t=0.0)
        assert r == pytest.approx(mid)

    def test_reservation_price_positive_inventory(self):
        """With q>0, reservation price < mid (positive skew downward)."""
        s = self._strat(inventory=5)
        mid = 1000.0
        t = 0.0
        tau = self.T - t
        expected_r = mid - 5 * self.GAMMA * (self.SIGMA ** 2) * tau
        assert s.reservation_price(mid, t) == pytest.approx(expected_r)

    def test_reservation_price_negative_inventory(self):
        """With q<0, reservation price > mid."""
        s = self._strat(inventory=-3)
        mid = 1000.0
        r = s.reservation_price(mid, t=0.0)
        assert r > mid

    def test_half_spread_formula(self):
        """δ = 0.5*γ*σ²*(T-t) + (1/γ)*ln(1+γ/k)"""
        s = self._strat()
        t = 0.0
        tau = self.T - t
        expected_term1 = 0.5 * self.GAMMA * (self.SIGMA ** 2) * tau
        expected_term2 = (1 / self.GAMMA) * math.log(1 + self.GAMMA / self.K)
        expected_delta = expected_term1 + expected_term2
        assert s.half_spread(t) == pytest.approx(expected_delta)

    def test_half_spread_decreases_toward_terminal(self):
        """δ should decrease as t → T (tau → 0)."""
        s = self._strat()
        delta_early = s.half_spread(t=0.0)
        delta_late = s.half_spread(t=self.T * 0.9)
        assert delta_late < delta_early

    def test_half_spread_floor_at_terminal(self):
        """At t=T, τ=0, δ = (1/γ)*ln(1+γ/k) only."""
        s = self._strat()
        delta_terminal = s.half_spread(t=self.T)
        expected = (1 / self.GAMMA) * math.log(1 + self.GAMMA / self.K)
        assert delta_terminal == pytest.approx(expected)


# ── quote computation ─────────────────────────────────────────────────────────

class TestQuoteComputation:
    GAMMA = 0.1
    SIGMA = 2.0
    K = 1.5
    T = 1000.0

    def _strat(self, **kwargs) -> AvellanedaStoikov:
        defaults = dict(gamma=self.GAMMA, sigma=self.SIGMA, k=self.K, T=self.T)
        defaults.update(kwargs)
        s = AvellanedaStoikov(**defaults)
        s._t0 = 0.0
        return s

    def test_bid_below_mid_ask_above_mid(self):
        """With zero inventory, bid < mid < ask."""
        book = flat_book(mid=1000, spread=4)
        s = self._strat()
        bid_t, ask_t = s.compute_quotes(book, ts=0.0)
        mid = book.mid()
        assert bid_t < mid
        assert ask_t > mid

    def test_quotes_symmetric_at_zero_inventory(self):
        """With q=0, bid and ask should be equidistant from mid."""
        book = flat_book(mid=1000, spread=4)
        s = self._strat()
        bid_t, ask_t = s.compute_quotes(book, ts=0.0)
        mid = book.mid()
        # Due to rounding, allow 1 tick difference
        assert abs((mid - bid_t) - (ask_t - mid)) <= 1

    def test_positive_inventory_skews_quotes_down(self):
        """With long inventory, both quotes shift down vs zero-inventory."""
        book = flat_book(mid=1000)
        s_flat = self._strat()
        s_long = self._strat()
        s_long.inventory = 10

        bid_flat, ask_flat = s_flat.compute_quotes(book, ts=0.0)
        bid_long, ask_long = s_long.compute_quotes(book, ts=0.0)

        assert bid_long <= bid_flat
        assert ask_long <= ask_flat

    def test_negative_inventory_skews_quotes_up(self):
        """With short inventory, both quotes shift up."""
        book = flat_book(mid=1000)
        s_flat = self._strat()
        s_short = self._strat()
        s_short.inventory = -10

        bid_flat, ask_flat = s_flat.compute_quotes(book, ts=0.0)
        bid_short, ask_short = s_short.compute_quotes(book, ts=0.0)

        assert bid_short >= bid_flat
        assert ask_short >= ask_flat

    def test_quotes_are_positive_ticks(self):
        book = flat_book(mid=1000)
        s = self._strat()
        bid_t, ask_t = s.compute_quotes(book, ts=0.0)
        assert bid_t >= 1
        assert ask_t >= 1

    def test_ask_strictly_above_bid(self):
        book = flat_book(mid=1000)
        s = self._strat(min_spread=1)
        bid_t, ask_t = s.compute_quotes(book, ts=0.0)
        assert ask_t > bid_t

    def test_returns_none_on_empty_book(self):
        book = LOBBook()   # no mid
        s = self._strat()
        bid_t, ask_t = s.compute_quotes(book, ts=0.0)
        assert bid_t is None
        assert ask_t is None


# ── on_book_update actions ────────────────────────────────────────────────────

class TestOnBookUpdate:
    def test_first_update_submits_two_quotes(self):
        book = flat_book(mid=1000)
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        actions = s.on_book_update(book, ts=0.0)
        submits = [a for a in actions if isinstance(a, SubmitLimit)]
        assert len(submits) == 2
        sides = {a.side for a in submits}
        assert sides == {Side.BID, Side.ASK}

    def test_unchanged_mid_produces_no_actions(self):
        book = flat_book(mid=1000)
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        s.on_book_update(book, ts=0.0)
        # Simulate the strategy orders being actually in the book
        # Force last_mid to match current mid so no requote is needed
        actions2 = s.on_book_update(book, ts=0.001)   # tiny dt, no mid change
        # Should not re-quote if nothing changed
        # (bid/ask ticks unchanged, mid unchanged)
        submits = [a for a in actions2 if isinstance(a, SubmitLimit)]
        assert len(submits) == 0

    def test_mid_change_triggers_requote(self):
        """Moving the BBO should cause cancel + re-submit."""
        book = flat_book(mid=1000)
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        actions1 = s.on_book_update(book, ts=0.0)
        # Register our bid and ask as if they actually went into the book
        for a in actions1:
            if isinstance(a, SubmitLimit):
                book.submit_limit(a.side, a.price_ticks, a.qty, a.order_id, 0.0)

        # Move the mid by adding better quotes to the same book (+10 tick shift).
        # cancel the old BBO first, then add new ones at mid=1010.
        book.cancel(1, ts=0.5)  # remove original bid at 998
        book.cancel(2, ts=0.5)  # remove original ask at 1002
        book.submit_limit(Side.BID, 1008, 100, 999_001, 0.5)
        book.submit_limit(Side.ASK, 1012, 100, 999_002, 0.5)

        actions2 = s.on_book_update(book, ts=1.0)
        cancels = [a for a in actions2 if isinstance(a, Cancel)]
        submits = [a for a in actions2 if isinstance(a, SubmitLimit)]
        assert len(cancels) == 2    # cancel old bid and ask
        assert len(submits) == 2    # re-submit new bid and ask


# ── fill accounting ───────────────────────────────────────────────────────────

class TestFillAccounting:
    def test_bid_fill_increases_inventory(self):
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        s.on_fill(order_id=1, side=Side.BID, price_ticks=100,
                  qty=5, ts=1.0)
        assert s.inventory == 5
        assert s.cash == -500   # paid 100 * 5

    def test_ask_fill_decreases_inventory(self):
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        s.inventory = 5
        s.on_fill(order_id=1, side=Side.ASK, price_ticks=102,
                  qty=5, ts=1.0)
        assert s.inventory == 0
        assert s.cash == 510    # received 102 * 5

    def test_round_trip_pnl(self):
        """Buy at 100, sell at 102 → PnL = 2 ticks per contract."""
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        s.on_fill(order_id=1, side=Side.BID, price_ticks=100, qty=1, ts=0.0)
        s.on_fill(order_id=2, side=Side.ASK, price_ticks=102, qty=1, ts=1.0)
        # inventory = 0, cash = 102 - 100 = 2
        assert s.inventory == 0
        assert s.cash == 2.0
        assert s.pnl(mid_ticks=100.0) == 2.0

    def test_pnl_includes_unrealized(self):
        """PnL = cash + q * mid."""
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        s.on_fill(order_id=1, side=Side.BID, price_ticks=100, qty=3, ts=0.0)
        # inventory = 3, cash = -300
        # mid moves to 105 → unrealized = 3 * 105 = 315, pnl = 315 - 300 = 15
        assert s.pnl(mid_ticks=105.0) == pytest.approx(15.0)


# ── fee accounting ───────────────────────────────────────────────────────────

class TestFees:
    def _strat(self, **kw):
        return AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0, **kw)

    def test_maker_rebate_credits_cash(self):
        s = self._strat(maker_fee=-0.3)   # rebate
        s.on_fill(order_id=1, side=Side.BID, price_ticks=100, qty=5,
                  ts=0.0, maker=True)
        assert s.cash == pytest.approx(-500 + 0.3 * 5)

    def test_taker_fee_debits_cash(self):
        s = self._strat(taker_fee=1.0)
        s.on_fill(order_id=1, side=Side.ASK, price_ticks=100, qty=2,
                  ts=0.0, maker=False)
        assert s.cash == pytest.approx(200 - 1.0 * 2)

    def test_maker_fee_not_applied_to_taker_fill(self):
        s = self._strat(maker_fee=-0.3, taker_fee=1.0)
        s.on_fill(order_id=1, side=Side.BID, price_ticks=100, qty=1,
                  ts=0.0, maker=False)
        assert s.cash == pytest.approx(-100 - 1.0)

    def test_zero_fees_preserve_old_accounting(self):
        s = self._strat()
        s.on_fill(order_id=1, side=Side.BID, price_ticks=100, qty=5, ts=0.0)
        assert s.cash == -500

    def test_fee_floor_widens_spread(self):
        """With maker_fee=3, the half-spread is floored at 3 ticks so a
        round trip can never lock in a sub-fee spread."""
        s = AvellanedaStoikov(gamma=0.001, sigma=0.05, k=1.5, T=100.0,
                              maker_fee=3.0)
        s._t0 = 0.0
        bid_t, ask_t = s.compute_quotes(flat_book(mid=1000, spread=4), ts=0.0)
        assert bid_t <= 997
        assert ask_t >= 1003

    def test_rebate_does_not_tighten_floor(self):
        s_fee = AvellanedaStoikov(gamma=0.001, sigma=0.05, k=1.5, T=100.0,
                                  maker_fee=-0.5)
        s_ref = AvellanedaStoikov(gamma=0.001, sigma=0.05, k=1.5, T=100.0)
        s_fee._t0 = s_ref._t0 = 0.0
        book = flat_book(mid=1000, spread=4)
        assert s_fee.compute_quotes(book, 0.0) == s_ref.compute_quotes(book, 0.0)


# ── imbalance-adjusted fair value ────────────────────────────────────────────

class TestImbalanceFairValue:
    PARAMS = dict(gamma=0.001, sigma=0.05, k=1.5, T=100.0)

    def test_balanced_book_no_adjustment(self):
        book = flat_book(mid=1000, spread=4)   # 100 qty on each side
        s = ImbalanceFairValueAvellanedaStoikov(beta=2.0, **self.PARAMS)
        assert s.fair_value(book) == pytest.approx(1000.0)

    def test_heavy_bid_side_shifts_fair_up(self):
        book = LOBBook()
        book.submit_limit(Side.BID, 998, 300, 1, 0.0)   # heavy bid
        book.submit_limit(Side.ASK, 1002, 100, 2, 0.0)
        s = ImbalanceFairValueAvellanedaStoikov(beta=2.0, **self.PARAMS)
        # imbalance = (300-100)/400 = 0.5 → fair = 1000 + 2.0*0.5
        assert s.fair_value(book) == pytest.approx(1001.0)

    def test_heavy_ask_side_shifts_fair_down(self):
        book = LOBBook()
        book.submit_limit(Side.BID, 998, 100, 1, 0.0)
        book.submit_limit(Side.ASK, 1002, 300, 2, 0.0)   # heavy ask
        s = ImbalanceFairValueAvellanedaStoikov(beta=2.0, **self.PARAMS)
        assert s.fair_value(book) == pytest.approx(999.0)

    def test_adjustment_does_not_pollute_ewma_state(self):
        book = LOBBook()
        book.submit_limit(Side.BID, 998, 300, 1, 0.0)
        book.submit_limit(Side.ASK, 1002, 100, 2, 0.0)
        s = ImbalanceFairValueAvellanedaStoikov(beta=2.0, **self.PARAMS)
        s.fair_value(book)
        assert s._fair == pytest.approx(1000.0)   # EWMA holds the raw mid


# ── queue-aware requoting ────────────────────────────────────────────────────

class TestQueueAwareStrategy:
    # alpha=1.0 makes fair value equal the instantaneous mid, isolating the
    # requote logic from EWMA lag
    PARAMS = dict(gamma=0.001, sigma=0.05, k=1.5, T=100.0, alpha=1.0)

    def _apply(self, book, actions, ts=0.0):
        for a in actions:
            if isinstance(a, SubmitLimit):
                book.submit_limit(a.side, a.price_ticks, a.qty, a.order_id, ts)
            elif isinstance(a, Cancel):
                book.cancel(a.order_id, ts)

    def test_no_actions_when_nothing_moves(self):
        book = flat_book(mid=1000, spread=4)
        s = QueueAwareFairValueAvellanedaStoikov(queue_patience=0, **self.PARAMS)
        self._apply(book, s.on_book_update(book, ts=0.0))
        assert s.on_book_update(book, ts=0.001) == []

    # For book-movement tests the strategy must quote far from the BBO so
    # that its own resting orders don't set the mid: δ clamps at
    # max_half_spread=50 → quotes at mid∓50, flow orders control the BBO.
    WIDE = dict(gamma=0.1, sigma=2.0, k=1.5, T=1000.0, alpha=1.0)

    def test_move_within_patience_keeps_quotes(self):
        """A 1-tick shift in both desired prices is tolerated at patience=2."""
        book = flat_book(mid=1000, spread=4)   # bid 998 / ask 1002
        s = QueueAwareFairValueAvellanedaStoikov(queue_patience=2, **self.WIDE)
        self._apply(book, s.on_book_update(book, ts=0.0))   # quotes 950/1050
        # Shift the whole book up 1 tick: mid 1000 → 1001
        book.cancel(1, ts=0.5)
        book.cancel(2, ts=0.5)
        book.submit_limit(Side.BID, 999, 100, 901, 0.5)
        book.submit_limit(Side.ASK, 1003, 100, 902, 0.5)
        assert s.on_book_update(book, ts=0.0) == []

    def test_same_move_requotes_at_zero_patience(self):
        book = flat_book(mid=1000, spread=4)
        s = QueueAwareFairValueAvellanedaStoikov(queue_patience=0, **self.WIDE)
        self._apply(book, s.on_book_update(book, ts=0.0))
        book.cancel(1, ts=0.5)
        book.cancel(2, ts=0.5)
        book.submit_limit(Side.BID, 999, 100, 901, 0.5)
        book.submit_limit(Side.ASK, 1003, 100, 902, 0.5)
        actions = s.on_book_update(book, ts=0.0)
        assert len([a for a in actions if isinstance(a, Cancel)]) == 2
        assert len([a for a in actions if isinstance(a, SubmitLimit)]) == 2

    def test_only_moved_side_requotes(self):
        """When rounding shifts only the desired ask, the resting bid keeps
        its queue position."""
        book = flat_book(mid=1000, spread=4)   # bid 998 / ask 1002
        s = QueueAwareFairValueAvellanedaStoikov(queue_patience=0, **self.WIDE)
        self._apply(book, s.on_book_update(book, ts=0.0))   # quotes 950/1050
        bid_id_before = s._bid_id
        # Move only the best bid 998 → 999: mid 1000.5.  Desired bid stays
        # floor(1000.5-50)=950; desired ask moves to ceil(1000.5+50)=1051.
        book.cancel(1, ts=0.5)
        book.submit_limit(Side.BID, 999, 100, 901, 0.5)
        actions = s.on_book_update(book, ts=0.0)
        assert s._bid_id == bid_id_before          # bid untouched
        cancelled = {a.order_id for a in actions if isinstance(a, Cancel)}
        assert bid_id_before not in cancelled
        submits = [a for a in actions if isinstance(a, SubmitLimit)]
        assert len(submits) == 1 and submits[0].side == Side.ASK

    def test_dead_side_resubmitted(self):
        """A filled quote (cleared id) is replaced on the next update."""
        book = flat_book(mid=1000, spread=4)
        s = QueueAwareFairValueAvellanedaStoikov(queue_patience=2, **self.PARAMS)
        self._apply(book, s.on_book_update(book, ts=0.0))
        bid_id = s._bid_id
        book.cancel(bid_id, ts=0.5)                # simulate the fill removing it
        s.on_fill(order_id=bid_id, side=Side.BID, price_ticks=999, qty=1, ts=0.5)
        actions = s.on_book_update(book, ts=1.0)
        submits = [a for a in actions if isinstance(a, SubmitLimit)]
        assert len(submits) == 1 and submits[0].side == Side.BID


# ── fair-value anchored variant ──────────────────────────────────────────────

class TestFairValueStrategy:
    # Params chosen so delta is small (~1 tick): quote centring is then
    # visible directly in the bid/ask prices.
    PARAMS = dict(gamma=0.001, sigma=0.05, k=1.5, T=100.0)

    def test_base_class_fair_value_is_mid(self):
        book = flat_book(mid=1000, spread=4)
        s = AvellanedaStoikov(**self.PARAMS)
        assert s.fair_value(book) == book.mid()

    def test_ewma_seeds_with_first_mid(self):
        book = flat_book(mid=1000, spread=4)
        s = FairValueAvellanedaStoikov(alpha=0.005, **self.PARAMS)
        assert s.fair_value(book) == 1000.0

    def test_ewma_lags_step_jump(self):
        """After one update at mid=1000 and one at mid=1050:
        fair = (1-a)*1000 + a*1050."""
        alpha = 0.02
        s = FairValueAvellanedaStoikov(alpha=alpha, **self.PARAMS)
        s.fair_value(flat_book(mid=1000, spread=4))
        fair = s.fair_value(flat_book(mid=1050, spread=4))
        assert fair == pytest.approx((1 - alpha) * 1000.0 + alpha * 1050.0)

    def test_ewma_converges_to_constant_mid(self):
        book = flat_book(mid=1000, spread=4)
        s = FairValueAvellanedaStoikov(alpha=0.1, **self.PARAMS)
        for _ in range(50):
            fair = s.fair_value(book)
        assert fair == pytest.approx(1000.0)

    def test_fair_value_carried_when_book_empties(self):
        s = FairValueAvellanedaStoikov(alpha=0.005, **self.PARAMS)
        s.fair_value(flat_book(mid=1000, spread=4))
        assert s.fair_value(LOBBook()) == pytest.approx(1000.0)

    def test_quotes_anchor_to_fair_not_mid(self):
        """After a 50-tick mid jump, quotes must stay near the (lagging)
        EWMA rather than re-centring on the new mid — the whole point of
        the variant.  The base class re-centres immediately."""
        s = FairValueAvellanedaStoikov(alpha=0.005, **self.PARAMS)
        s._t0 = 0.0
        s.fair_value(flat_book(mid=1000, spread=4))   # seed EWMA at 1000
        jumped = flat_book(mid=1050, spread=4)

        bid_fv, ask_fv = s.compute_quotes(jumped, ts=0.0)
        # fair ≈ 1000.25; delta ≈ 1 tick → quotes within a few ticks of 1000
        assert 995 <= bid_fv <= 1000
        assert 1001 <= ask_fv <= 1005

        base = AvellanedaStoikov(**self.PARAMS)
        base._t0 = 0.0
        bid_as, _ = base.compute_quotes(jumped, ts=0.0)
        assert bid_as >= 1045   # base class chases the new mid

    def test_reset_clears_fair_value(self):
        s = FairValueAvellanedaStoikov(alpha=0.005, **self.PARAMS)
        s.fair_value(flat_book(mid=1000, spread=4))
        s.reset(ts=0.0)
        assert s._fair is None


# ── reset ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_restarts_id_counter(self):
        """IDs restart at the range base after reset() so long-running
        processes never drift the strategy range."""
        from lob_sim.strategy import STRATEGY_ID_BASE
        book = flat_book(mid=1000)
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        first_ids = sorted(a.order_id for a in s.on_book_update(book, ts=0.0)
                           if isinstance(a, SubmitLimit))
        s.reset(ts=0.0)
        second_ids = sorted(a.order_id for a in s.on_book_update(book, ts=0.0)
                            if isinstance(a, SubmitLimit))
        assert first_ids == second_ids
        assert first_ids[0] == STRATEGY_ID_BASE

    def test_reset_clears_state(self):
        s = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, T=1000.0)
        s.inventory = 10
        s.cash = -1000.0
        s._bid_id = 99
        s.reset(ts=0.0)
        assert s.inventory == 0
        assert s.cash == 0.0
        assert s._bid_id is None
