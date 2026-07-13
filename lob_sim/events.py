"""
events.py — Core data structures for the LOB simulator.

All prices are stored as integer ticks to avoid floating-point issues.
Use LOBConfig.to_ticks() / from_ticks() for conversion at the boundary.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class Side(enum.Enum):
    BID = "bid"
    ASK = "ask"

    def opposite(self) -> "Side":
        return Side.ASK if self == Side.BID else Side.BID


class EventType(enum.Enum):
    LIMIT_ORDER = "limit_order"
    MARKET_ORDER = "market_order"
    CANCEL = "cancel"
    BOOK_UPDATE = "book_update"


@dataclass
class Order:
    order_id: int
    side: Side
    price_ticks: int          # integer tick price; never a float
    qty: int
    timestamp: float
    event_type: EventType = EventType.LIMIT_ORDER

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"Order qty must be positive, got {self.qty}")
        if self.price_ticks <= 0:
            raise ValueError(f"Price must be positive, got {self.price_ticks}")


@dataclass
class MarketOrder:
    order_id: int
    side: Side
    qty: int
    timestamp: float
    event_type: EventType = EventType.MARKET_ORDER

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"Market order qty must be positive, got {self.qty}")


@dataclass
class CancelRequest:
    order_id: int
    timestamp: float
    event_type: EventType = EventType.CANCEL


@dataclass
class Trade:
    trade_id: int
    aggressor_side: Side
    aggressor_order_id: int
    passive_order_id: int
    price_ticks: int
    qty: int
    timestamp: float

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"Trade qty must be positive, got {self.qty}")


@dataclass
class BookSnapshot:
    timestamp: float
    bids: list[tuple[int, int]]   # [(price_ticks, qty), ...] best first
    asks: list[tuple[int, int]]   # [(price_ticks, qty), ...] best first
    mid_ticks: Optional[float] = None
    spread_ticks: Optional[int] = None


# ── Strategy action types ────────────────────────────────────────────────────

@dataclass
class SubmitLimit:
    side: Side
    price_ticks: int
    qty: int
    order_id: int
    timestamp: float


@dataclass
class Cancel:
    order_id: int
    timestamp: float


# Union for strategy return values
Action = SubmitLimit | Cancel
