"""The virtual account: positions, closed trades, and the money maths.

Sizes are expressed in **units of the base currency** — 10,000 units of
EUR/USD is 0.1 of a standard lot. Profit on a position is naturally
denominated in the **quote** currency and is converted into the account
currency when it is realised or marked to market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import List, Optional

from .instruments import Pair, quantize_money
from .orders import Order, Side
from .rates import Quote


class InsufficientMargin(Exception):
    """Raised when an order would need more margin than the account has free."""


@dataclass
class Position:
    """An open exposure to a currency pair."""

    id: int
    pair: Pair
    side: Side
    units: Decimal
    entry_price: Decimal
    opened_on: date
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    def exit_price(self, quote: Quote) -> Decimal:
        """The price this position would close at right now.

        Closing a long means selling, which happens at the bid; closing a
        short means buying back at the ask.
        """
        return quote.bid if self.side is Side.BUY else quote.ask

    def pnl_in_quote(self, exit_price: Decimal) -> Decimal:
        """Profit in the quote currency if closed at ``exit_price``."""
        move = Decimal(exit_price) - self.entry_price
        if self.side is Side.SELL:
            move = -move
        return self.units * move

    def unrealized_in_quote(self, quote: Quote) -> Decimal:
        return self.pnl_in_quote(self.exit_price(quote))

    def pips(self, exit_price: Decimal) -> Decimal:
        """Signed profit in pips, positive when the trade is winning."""
        move = Decimal(exit_price) - self.entry_price
        if self.side is Side.SELL:
            move = -move
        return move / self.pair.pip_size

    def hit_stop_loss(self, quote: Quote) -> bool:
        if self.stop_loss is None:
            return False
        price = self.exit_price(quote)
        return price <= self.stop_loss if self.side is Side.BUY else price >= self.stop_loss

    def hit_take_profit(self, quote: Quote) -> bool:
        if self.take_profit is None:
            return False
        price = self.exit_price(quote)
        return price >= self.take_profit if self.side is Side.BUY else price <= self.take_profit


@dataclass
class Trade:
    """A completed round trip, recorded in the account's history."""

    id: int
    pair: Pair
    side: Side
    units: Decimal
    entry_price: Decimal
    exit_price: Decimal
    opened_on: date
    closed_on: date
    pnl: Decimal  # in the account currency
    pips: Decimal
    reason: str  # manual | stop_loss | take_profit | margin_call

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class Account:
    """A funded paper trading account."""

    currency: str = "USD"
    balance: Decimal = Decimal("10000.00")
    leverage: int = 30
    current_date: date = field(default_factory=date.today)
    positions: List[Position] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    _next_id: int = 1

    def __post_init__(self) -> None:
        self.currency = self.currency.upper()
        self.balance = quantize_money(self.balance)
        if self.leverage < 1:
            raise ValueError(f"leverage must be at least 1, got {self.leverage}")

    def take_id(self) -> int:
        """Issue the next identifier, shared across orders, positions and trades."""
        issued = self._next_id
        self._next_id += 1
        return issued

    @property
    def open_positions(self) -> List[Position]:
        return list(self.positions)

    @property
    def pending_orders(self) -> List[Order]:
        from .orders import OrderStatus

        return [order for order in self.orders if order.status is OrderStatus.PENDING]

    def position(self, position_id: int) -> Position:
        for position in self.positions:
            if position.id == position_id:
                return position
        raise KeyError(f"no open position with id {position_id}")

    def order(self, order_id: int) -> Order:
        for order in self.orders:
            if order.id == order_id:
                return order
        raise KeyError(f"no order with id {order_id}")

    @property
    def realized_pnl(self) -> Decimal:
        return quantize_money(sum((trade.pnl for trade in self.trades), Decimal(0)))

    def record_trade(self, trade: Trade) -> None:
        """Book a closed trade and move its profit into the cash balance."""
        self.trades.append(trade)
        self.balance = quantize_money(self.balance + trade.pnl)
