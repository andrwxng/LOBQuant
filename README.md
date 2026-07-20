# LOBQuant — Limit Order Book Simulator

A research-quality Python implementation of a price-time priority matching engine with an Avellaneda-Stoikov optimal market-making strategy, discrete-event simulation, and analytics.

---

## Package layout

```
lob_sim/
  __init__.py      re-exports public API
  events.py        dataclasses: Order, MarketOrder, CancelRequest, ModifyRequest, Trade, BookSnapshot, Action types
  orderbook.py     matching engine (LOBBook)
  flow.py          synthetic flow generators: PoissonFlow, LOBSTERReplay
  strategy.py      Strategy ABC + AvellanedaStoikov + fair-value/imbalance variants
  simulator.py     event-driven loop + run_simulation() factory
  metrics.py       PnL, Sharpe, drawdown, fill rate, adverse selection, inventory profile
tests/
  test_orderbook.py
  test_flow.py
  test_strategy.py
  test_simulator.py
  test_metrics.py
scripts/
  gamma_sweep.py     reproducible sweeps (source of every results table below)
  calibrate_k.py     empirical fill-intensity decay calibration
  queue_analysis.py  fill probability vs FIFO queue position
  latency_sweep.py   PnL vs order-entry latency
  benchmark.py       engine throughput measurement
notebooks/
  01_engine_sanity.ipynb
  02_flow_calibration.ipynb
  03_market_making.ipynb
```

---

## Installation

```bash
pip install sortedcontainers numpy pytest matplotlib
```

Run tests:
```bash
pytest tests/ -v
```

---

## Architecture

### Matching engine (`orderbook.py`)

`LOBBook` implements a standard limit order book with price-time priority (FIFO within a price level).

**Data structures**
- `_bids`: `SortedDict` keyed by `-price_ticks` so iteration gives descending (best-bid-first) order.
- `_asks`: `SortedDict` keyed by `+price_ticks`, ascending.
- Each price level is a `collections.deque` of `(order_id, remaining_qty)` tuples.
- `_orders: Dict[int, _OrderRecord]` provides O(1) lookup of any resting order's side, price, and remaining qty.

**Cancellation complexity**: O(n) per price level in the worst case, where n is the number of resting orders at that level. In practice levels are short. A production system would use a doubly-linked list per level for O(1) mid-deque removal; this is noted as future work.

**Price representation**: All prices are integer ticks throughout. Float conversion only occurs at presentation boundaries (notebooks, metrics). This avoids floating-point accumulation errors in matching.

**Debug mode**: `LOBBook.debug = True` enables invariant assertions after every mutation (no crossed book, consistent `_orders` ↔ level deque cross-reference, no empty levels).

### Order flow (`flow.py`)

Two generators share the interface `next_event(ts, book) -> Event`:

**PoissonFlow** follows Cont-Stoikov-Talreja (2010):
- Three independent Poisson processes: limit orders, market orders, cancellations.
- Limit order depth from BBO is drawn from a Geometric(p) distribution.
- Each call to `_compute_next_ts()` draws the next inter-arrival time from Exp(total_rate) and selects event type proportional to per-type rates.
- Configurable: λ_lim, λ_mkt, λ_cancel, p_geom, order size range, seed.

**LOBSTERReplay** streams events from a LOBSTER-format CSV message file, rescaling timestamps and mapping LOBSTER integer prices to internal ticks. Details, including how to validate the engine against real exchange data and calibrate `PoissonFlow` from it, are in [Testing against real data](#testing-against-real-data) below.

### Strategy (`strategy.py`)

**AvellanedaStoikov** implements the Avellaneda & Stoikov (2008) stochastic control solution:

```
Reservation price:   r(t) = mid(t) - q · γ · σ² · (T - t)
Optimal half-spread: δ(t) = ½ · γ · σ² · (T - t) + (1/γ) · ln(1 + γ/k)
Quotes:              bid = r(t) - δ(t)   (rounded down to tick)
                     ask = r(t) + δ(t)   (rounded up to tick)
```

The strategy cancels and re-submits quotes whenever the mid price changes by at least one tick. Fills update inventory and cash; PnL = cash + q × mid.

### Simulator (`simulator.py`)

An event-driven loop using a min-heap (`heapq`). Events are `(timestamp, sequence_no, event)` tuples. After every book mutation the strategy is notified and its returned actions are inserted at `current_ts + ε`, ensuring they execute immediately after the triggering event but before the next external arrival.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.1 | Risk-aversion coefficient. Higher = tighter inventory control, lower PnL. |
| `sigma` | 0.05 | Volatility in ticks/√s (assumed by the strategy; see calibration). |
| `k` | 1.5 | Fill-intensity decay (from the Poisson fill rate model). |
| `T` | 28800 | Session length in seconds (8 hours). |
| `lambda_lim` | 5.0 | Limit order arrival rate (events/s). |
| `lambda_mkt` | 1.0 | Market order arrival rate (events/s). |
| `lambda_cancel` | 0.3 | Per-order cancellation rate. |
| `p_geom` | 0.3 | Geometric distribution parameter for order depth. |

## Calibration procedure

1. Estimate `sigma` from historical mid-price returns: `σ = std(Δmid) / √Δt`.
2. Fit `lambda_mkt` and `lambda_lim` by counting market vs limit order arrivals per second in exchange data.
3. Fit `k` by regressing fill probability against queue position or half-spread.
4. Tune `gamma` by running a γ-sweep (see notebook 03) and selecting the risk/return point on the frontier that matches your drawdown tolerance.

---

## Results (1-hour session, 12-seed γ-sweep)

Reproduce with `python scripts/gamma_sweep.py` (fully deterministic — identical
table on every run) and `python scripts/benchmark.py`.

**Flow config**: λ_lim=5, λ_mkt=0.3, λ_cancel=0.3, max_qty=3, book depth 20×20.  
**Strategy config**: σ=0.86, k=1.5, T=3600, max_inventory=30, max_half_spread=8.

> An earlier version of this table showed large positive PnL and *negative*
> adverse selection for every γ. Those numbers were artifacts of three
> simulator bugs (unaccounted aggressor fills, PnL marked at cash-only when
> one book side emptied, and a requote feedback loop), all since fixed and
> covered by regression tests. The corrected results below are less
> flattering and more instructive.

### Throughput

Pure-Python engine, single core (Apple Silicon):

| Configuration | Events/sec |
|---------------|-----------|
| Matching engine + Poisson flow (no strategy) | ~258,000 |
| Full simulation (AS strategy attached) | ~175,000 |

### γ-sweep summary (12 seeds × 1 hour)

| γ | Mean PnL | PnL std | Sharpe | Max DD | Inv std | Fill rate | Adv sel |
|---|----------|---------|--------|--------|---------|-----------|---------|
| 0.0005 | −97 | 76 | −54.6 | 154 | 1.44 | 12.1% | +0.38 |
| 0.001 | −204 | 71 | −134.2 | 244 | 0.95 | 10.0% | +0.64 |
| 0.002 | −322 | 84 | −192.6 | 342 | 0.70 | 9.5% | +0.66 |
| 0.005 | −275 | 52 | −220.7 | 295 | 0.41 | 9.0% | +0.89 |
| 0.01 | −178 | 47 | −148.6 | 195 | 0.39 | 7.1% | +1.73 |
| 0.05 | −115 | 38 | −94.3 | 128 | 0.39 | 5.6% | +2.93 |

Key observations:
- **The naive AS maker loses money at every γ**, and adverse selection is
  *positive* everywhere (+0.4 to +2.9 ticks per fill): market orders that
  sweep multiple levels move the mid in the aggressor's direction, so fills
  systematically precede adverse mid moves. This is the economically expected
  sign — the earlier "negative adverse selection" (mid moving in the maker's
  favour after every fill) should have been a red flag.
- **Wider spreads reduce but do not eliminate the loss**: raising γ from
  0.002 to 0.05 cuts the mean loss from −322 to −115 ticks and halves the
  fill rate, but per-fill adverse selection *rises* (only strongly informed
  sweeps reach the wider quotes).
- **Inventory control works as designed**: inventory std falls monotonically
  in γ (1.44 → 0.39), confirming the reservation-price skew does its job even
  though the spread term cannot cover pick-off costs.

### σ calibration

The strategy assumes σ=0.86 ticks/√s, but realized mid-price volatility from
1-second returns is **σ̂ ≈ 3.83** — a 4.5× gap, because market orders sweeping
multiple levels produce jump-like moves that a pure-diffusion σ underestimates.
Re-running the sweep with the corrected σ=3.83 narrows the loss (e.g. −105 vs
−275 at γ=0.005) by widening quotes, but does not flip profitability: under
this flow model the maker's edge is bounded by mechanical adverse selection,
which spread width alone cannot overcome.

---

## Improving on naive AS: fair-value anchored quoting

Reproduce with `python scripts/gamma_sweep.py --strategy fv` (deterministic).

### Diagnosis: quote-chasing

Adverse selection on the baseline strategy *grows* with measurement horizon
(+0.38 ticks/fill at 1 s → +1.17 at 120 s): price moves in this flow model
are permanent, so there is no mean reversion to harvest. The leak is
different — because the baseline re-centres its quotes on the instantaneous
mid after every move, it immediately posts a fresh bid near each newly
elevated price and a fresh ask near each newly depressed one, systematically
buying tops and selling bottoms.

### Fix

`FairValueAvellanedaStoikov` centres quotes on an EWMA of the mid instead of
the instantaneous mid (`alpha` = EWMA weight per book update). Quotes lag
transient dislocations rather than chasing them. A related intuitive fix —
post-only clamping so quotes never cross the spread — was also tested and
**hurt** (−142 vs −97 baseline at γ=0.0005): clamping to the visible BBO
re-anchors quotes to the dislocated price, which is the disease itself.

### Methodology and results

(α, γ) selected by grid search on **training seeds 1–12** (best mean/std
PnL → α=0.005, γ=0.01); everything below is reported on **held-out
validation seeds 13–24**:

| Strategy | Mean PnL | PnL std | Max DD | Fill rate | Adv sel |
|----------|----------|---------|--------|-----------|---------|
| AS baseline (γ=0.01) | −188 | 46 | 209 | 7.5% | +1.57 |
| Fair-value AS (γ=0.01, α=0.005) | **+419** | 52 | 31 | 8.3% | +1.20 |
| Fair-value AS + calibrated k=0.213 | **+768** | 94 | 22 | 8.7% | +1.18 |
| … + maker rebate 0.3 / taker fee 1.0 | **+825** | 102 | 23 | 8.7% | +1.18 |

The third row uses the fill-intensity decay measured by
`python scripts/calibrate_k.py`: pooled over instrumented runs, realized
λ(δ) = A·e^(−kδ) fits **k̂ ≈ 0.21** versus the assumed k=1.5 — fill
probability decays ~7× more slowly with depth than the model assumed, so
wider quotes cost far less fill rate than the formula believes. Since k̂ is
measured from fill/exposure data (not tuned on PnL), feeding it back is
calibration, not curve-fitting.

The fourth row applies maker-taker venue economics (`maker_fee=-0.3`,
`taker_fee=1.0` ticks/contract on the strategy): because ~99% of fills are
passive, the rebate dominates and adds ~+57/hr. Fee-aware accounting also
reframes the *baseline* result — naive AS at γ=0.0005 loses only ~0.31
ticks/fill, so on a rebate venue it would sit near breakeven even without
the fair-value anchor.

### Order-book imbalance skew

`ImbalanceFairValueAvellanedaStoikov` additionally shifts the anchor by
top-3-level queue imbalance, `fair += β·(Qb−Qa)/(Qb+Qa)` — a one-sided book
predicts short-horizon drift toward the thin side. β selected on training
seeds (β=2), reported on validation seeds
(`python scripts/gamma_sweep.py --strategy imb`):

| Strategy | Mean PnL | PnL std | Max DD |
|----------|----------|---------|--------|
| Fair-value AS (β=0) | +768 | 94 | 22 |
| + imbalance skew (β=2) | **+892** | 94 | 23 |

One caveat: the imbalance signal jitters, so the strategy requotes far more
often (fill rate per submitted quote drops from 8.7% to 1.4%). Message
traffic is free in this simulator but is a real cost on actual venues
(order-to-trade ratios, rate limits) — a production version would debounce
the signal.

### Queue position

`LOBBook.queue_position(order_id)` exposes (orders ahead, qty ahead) in the
FIFO queue.  Measuring fill probability against queue position at first
sighting (`python scripts/queue_analysis.py`) gives the monotone curve that
Cont & de Larrard (2013) model:

| qty ahead in queue | quotes | fill probability |
|--------------------|--------|------------------|
| 0 | 19,227 | 9.1% |
| 1–2 | 3,040 | 5.8% |
| 3–5 | 3,380 | 4.1% |
| 6–10 | 1,352 | 3.5% |
| 11+ | 144 | 2.8% |

Cancel-replace sends an order to the back of this curve, which motivates
`QueueAwareFairValueAvellanedaStoikov`: requote each side independently and
only when its desired price moved by more than `queue_patience` ticks.
Selected on training seeds (patience=0 — i.e. pure per-side requoting),
reported on validation seeds (`python scripts/gamma_sweep.py --strategy queue`):

| Strategy | Mean PnL | PnL std | Max DD | Fill rate | Adv sel |
|----------|----------|---------|--------|-----------|---------|
| Fair-value AS (both-side requote) | +768 | 94 | 22 | 8.7% | +1.18 |
| Queue-aware (patience=0) | **+1,141** | 79 | 34 | 8.6% | +1.15 |

The entire +49% gain comes from *not* cancelling the side whose price didn't
move: the base requote logic forfeits FIFO seniority on both sides whenever
either changes.  Larger patience raises mean PnL further on training seeds
but with proportionally more variance; the selection rule (mean/std) prefers
patience=0.

### Latency

`Simulator(latency=…)` delays every strategy submission and cancellation by
a fixed interval (order-entry latency; market-data latency is not modeled).
Strategy state stays consistent under in-flight fills and cancels — orders
are tracked until confirmed gone, and superseded in-flight quotes are
garbage-collected once they land.  PnL on validation seeds
(`python scripts/latency_sweep.py`):

| Latency | Fair-value AS | Queue-aware (patience=0) |
|---------|---------------|--------------------------|
| 0 ms | +768 ± 94 | +1,141 ± 79 |
| 50 ms | +710 ± 48 | +1,126 ± 132 |
| 200 ms | +682 ± 119 | +864 ± 122 |
| 1 s | +622 ± 74 | +455 ± 168 |

Two observations. First, the edge decays with latency for both variants, as
it must.  Second, the ranking *inverts* around the mean inter-event time
(~170 ms): queue-aware quoting acts on finer-grained signals, which is an
advantage when orders land promptly and a liability when every decision is
a second stale — at 1 s the simpler both-side strategy wins.  Speed and
signal granularity are complements, not independent choices.

### Robustness (selected cell, validation seeds, perturbed flow)

| Flow regime | Mean PnL | PnL std |
|-------------|----------|---------|
| baseline flow | +419 | 52 |
| 2× market-order rate | +1,245 | 91 |
| 2× reference volatility | +400 | 89 |
| larger orders (max_qty=5) | +380 | 95 |

The edge grows when market-order flow doubles (more fills against a lagging
anchor) and survives volatility and order-size perturbations.

**Caveat**: this edge exploits the structure of *this* flow model — limit
orders anchor to the current BBO, making quote-chasing systematically toxic
and a lagged anchor systematically valuable. That is the real lesson: a
market maker's edge is a model of the order flow, not a quoting formula.
Against real exchange data (LOBSTERReplay) none of these numbers transfer
without recalibration.

---

## Testing against real data

Three separate things can be tested against real data, and they are not
equally hard.

**Calibrate the flow model.** Fit `PoissonFlow`'s λ_lim, λ_mkt, λ_cancel,
p_geom, and σ from real event counts and price returns (see Calibration
procedure above) and re-run the strategy ladder under realistic parameters.
No replay needed — this is the cheapest and most statistically clean check,
and the honest expectation is that it *closes* the edge shown above, since
that edge was discovered inside this specific synthetic flow model.

**Validate the matching engine.** LOBSTER ships each message file with a
companion orderbook file recording the true book state after every message.
Feed the messages through `LOBBook` via `LOBSTERReplay` and diff the
reconstructed book against that ground truth after every event — percentage
agreement is a hard correctness metric for the engine against a real
exchange feed.

**Replay a strategy — with an explicit fill-model assumption.** Historical
messages describe what happened *without* your strategy in the book. The
moment a simulated quote would have filled, reality would have diverged,
since the resting order behind you wouldn't have been hit. There is no
fully correct fix; the standard approach is a no-market-impact fill model —
treat a quote as filled when the tape trades through its price, or trades
at its price after enough volume has printed to exhaust the queue ahead of
it (`LOBBook.queue_position()` is built for exactly this). State the
assumption and report results conditional on it, as any serious backtest
does.

### `LOBSTERReplay` usage

```python
from lob_sim import LOBSTERReplay, Simulator

flow = LOBSTERReplay(
    message_file="AMZN_2012-06-21_message.csv",
    orderbook_file="AMZN_2012-06-21_orderbook.csv",  # seeds pre-existing liquidity
    n_levels=10,           # levels present in the orderbook file
    tick_divisor=100,      # LOBSTER prices are in 10000ths of $; 100 -> 1-cent ticks
)
# initial_depth=0: LOBSTERReplay seeds the book itself from real prices;
# Simulator's default synthetic ladder (centred on initial_mid=1000) would
# otherwise coexist with and immediately cross the real price levels.
sim = Simulator(flow=flow, strategy=None, duration=23400.0, initial_depth=0)
result = sim.run()
```

Notes on what the replay does and does not recover from LOBSTER data:

- **Order identity.** Raw LOBSTER order IDs are remapped to a reserved
  internal range. A message can reference an order that was already
  resting before the file starts (e.g. placed pre-market); such orders
  never appear as a type-1 "new order" message, so they cannot be resolved
  to an individual internal ID — LOBSTER's orderbook file gives only
  per-level aggregate quantity, not individual pre-existing order
  identity. With `orderbook_file` supplied, the first row seeds one
  synthetic resting order per non-empty level, and unresolvable
  cancel/modify messages fall back to that level's synthetic order.
  Without it, such messages are dropped; `flow.n_unresolved` counts them.
- **Execution direction.** LOBSTER's `direction` field on execution
  messages (type 4/5) names the side of the *resting* order being hit, not
  the aggressor — a buy-limit-order execution means a sell aggressor. The
  aggressor's `MarketOrder.side` is the opposite of `direction`.
- **Partial cancellations** (type 2) reduce a resting order's quantity in
  place via `LOBBook.reduce_qty()`, preserving its FIFO queue position —
  unlike a cancel-and-resubmit, which would send it to the back of the
  queue.
- **Hidden executions** (type 5) match against liquidity with no
  representation in the visible book. Replaying them as ordinary market
  orders would incorrectly consume visible depth, so they are not turned
  into book events; they are collected in `flow.hidden_trades` instead.

---

## Limitations and live-trading considerations

### Lookahead bias
This simulation uses no future information: the strategy observes only the current book state and elapsed time. However, the flow parameters (λ_lim, λ_mkt, k) are assumed constant across the session. In live markets these parameters are non-stationary (intraday seasonality, regime changes). Calibrating on in-sample data and backtesting on out-of-sample data is essential before drawing conclusions about edge.

### Transaction cost assumptions
Fees default to zero but are modeled: `AvellanedaStoikov(maker_fee=…, taker_fee=…)` charges (or rebates, if negative) ticks per contract on each fill, with the simulator distinguishing passive from aggressor executions. The results section reports a maker-rebate scenario. Quotes are fee-floored: the half-spread δ never drops below the per-side maker fee, so a round trip cannot lock in a sub-fee spread.

### Adverse selection
The simulator models adverse selection implicitly through the Poisson fill intensity: informed traders arriving as market orders move the mid against the market maker after the fill. The `metrics.adverse_selection()` function measures the average mid-price change in the 1-second horizon after each fill (signed so that positive = adverse). In a real venue, adverse selection is substantially larger than in this model because (a) informed traders use limit orders as well, (b) latency allows other participants to update quotes faster than our model assumes, and (c) correlated order flow creates burst-fill episodes. The AS model's `k` parameter partially captures fill-intensity effects but does not model queue dynamics or latency explicitly.

---

## Future work

- **Market-data latency**: The simulator models order-entry latency; a stale *view* of the book (data latency) is a separate and larger effect.
- **Multi-asset / correlation**: Extend to pairs trading or ETF arbitrage strategies.
- **Hidden / iceberg orders**: Add a `hidden_qty` field to `Order` for hidden reserve.
- **Auction phases**: Opening and closing auction mechanics (batch matching).
- **Multi-venue**: Fragmented liquidity across venues; smart order routing.
- **Non-stationary parameters**: Intraday λ seasonality; regime-switching volatility model.
- **Reinforcement learning baseline**: Compare AS against a learned policy (e.g. DQN on the book state).

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
- Cont, R., Stoikov, S. & Talreja, R. (2010). *A stochastic model for order book dynamics*. Operations Research, 58(3), 549–563.
- Cont, R. & de Larrard, A. (2013). *Price dynamics in a Markovian limit order market*. SIAM Journal on Financial Mathematics, 4(1), 1–25.
