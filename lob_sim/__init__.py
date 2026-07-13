"""
lob_sim — Limit order book simulator with market-making strategies.
"""

from .events import (
    Action, BookSnapshot, Cancel, CancelRequest, EventType,
    MarketOrder, Order, Side, SubmitLimit, Trade,
)
from .orderbook import LOBBook
from .flow import LOBSTERReplay, OrderFlowGenerator, PoissonFlow
from .strategy import AvellanedaStoikov, FairValueAvellanedaStoikov, Strategy
from .simulator import SimulationResult, Simulator, run_simulation
from . import metrics

__all__ = [
    # events
    "Action", "BookSnapshot", "Cancel", "CancelRequest", "EventType",
    "MarketOrder", "Order", "Side", "SubmitLimit", "Trade",
    # engine
    "LOBBook",
    # flow
    "LOBSTERReplay", "OrderFlowGenerator", "PoissonFlow",
    # strategy
    "AvellanedaStoikov", "FairValueAvellanedaStoikov", "Strategy",
    # simulator
    "SimulationResult", "Simulator", "run_simulation",
    # analytics
    "metrics",
]
