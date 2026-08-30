"""Order types and their trigger rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from .instruments import Pair
from .rates import Quote


class Side(str, Enum):
    """Direction of a trade."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    """How an order reaches the market."""

    MARKET = "market"
    LIMIT = "limit"  # fills at a better price than the current market
    STOP = "stop"  # fills once the market moves through the trigger


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """A resting instruction that becomes a position once triggered."""

    id: int
    pair: Pair
    side: Side
    units: Decimal
    type: OrderType
    price: Decimal  # trigger price
    created_on: date
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_on: Optional[date] = None
    fill_price: Optional[Decimal] = None

    def is_triggered(self, quote: Quote) -> bool:
        """Whether the market has reached this order's trigger price.

        A buy executes at the ask and a sell at the bid, so each case tests the
        side of the book the order would actually cross.
        """
        if self.status is not OrderStatus.PENDING:
            return False
        if self.type is OrderType.MARKET:
            return True
        if self.type is OrderType.LIMIT:
            # A limit buy waits for the price to fall to the trigger or below.
            return quote.ask <= self.price if self.side is Side.BUY else quote.bid >= self.price
        # A stop buy waits for the price to rise through the trigger.
        return quote.ask >= self.price if self.side is Side.BUY else quote.bid <= self.price

    def fill_side_price(self, quote: Quote) -> Decimal:
        """The price this order fills at once triggered."""
        return quote.ask if self.side is Side.BUY else quote.bid
