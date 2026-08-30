"""The simulation engine: fills orders, marks positions, and moves the clock.

The engine is the only place that mutates an :class:`~fxpaper.account.Account`.
It samples one price per simulated day from the rate provider, which is enough
resolution for daily ECB fixings and keeps the simulation reproducible.

Known simplification: an order fills exactly at its trigger price. Gap risk —
where a real fill would be worse than the stop you asked for — is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from .account import Account, InsufficientMargin, Position, Trade
from .instruments import Pair, quantize_money
from .orders import Order, OrderStatus, OrderType, Side
from .rates import DEFAULT_SPREAD_PIPS, Quote, RateProvider, RateUnavailable, convert, spread_quote

# Closing all positions is forced below this margin level, in percent
DEFAULT_MARGIN_CALL_LEVEL = Decimal("50")


@dataclass
class Event:
    """Something the simulation did on its own while time moved forward."""

    on: date
    kind: str  # order_filled | stop_loss | take_profit | margin_call
    message: str
    ref_id: Optional[int] = None

    def __str__(self) -> str:
        return f"[{self.on}] {self.kind}: {self.message}"


class Engine:
    """Drives an account against a price source."""

    def __init__(
        self,
        account: Account,
        provider: RateProvider,
        spread_pips: Decimal = DEFAULT_SPREAD_PIPS,
        margin_call_level: Decimal = DEFAULT_MARGIN_CALL_LEVEL,
    ):
        self.account = account
        self.provider = provider
        self.spread_pips = Decimal(spread_pips)
        self.margin_call_level = Decimal(margin_call_level)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    def quote(self, pair: Pair, on: Optional[date] = None) -> Quote:
        """Current tradeable bid/ask for ``pair``."""
        on = on or self.account.current_date
        mid = self.provider.rate(pair, on)
        return spread_quote(pair, mid, on, self.spread_pips)

    def to_account_currency(self, amount: Decimal, currency: str, on: Optional[date] = None) -> Decimal:
        """Convert an amount into the account's currency."""
        return convert(
            amount, currency, self.account.currency, self.provider, on or self.account.current_date
        )

    # ------------------------------------------------------------------
    # Account valuation
    # ------------------------------------------------------------------
    def unrealized_pnl(self, on: Optional[date] = None) -> Decimal:
        """Mark every open position to market, in the account currency."""
        total = Decimal(0)
        for position in self.account.positions:
            try:
                quote = self.quote(position.pair, on)
            except RateUnavailable:
                continue  # no price today; carry the position at cost
            in_quote = position.unrealized_in_quote(quote)
            total += self.to_account_currency(in_quote, position.pair.quote, on)
        return quantize_money(total)

    def equity(self, on: Optional[date] = None) -> Decimal:
        return quantize_money(self.account.balance + self.unrealized_pnl(on))

    def margin_for(self, pair: Pair, units: Decimal, on: Optional[date] = None) -> Decimal:
        """Margin required to hold ``units`` of ``pair``, in the account currency."""
        notional = self.to_account_currency(Decimal(units), pair.base, on)
        return quantize_money(notional / self.account.leverage)

    def used_margin(self, on: Optional[date] = None) -> Decimal:
        """Margin tied up by open positions, in the account currency.

        Positions the provider cannot price today are skipped, matching how
        :meth:`unrealized_pnl` carries them at cost. A data gap should not halt
        the simulation or fire a spurious margin call.
        """
        total = Decimal(0)
        for position in self.account.positions:
            try:
                total += self.margin_for(position.pair, position.units, on)
            except RateUnavailable:
                continue
        return quantize_money(total)

    def free_margin(self, on: Optional[date] = None) -> Decimal:
        return quantize_money(self.equity(on) - self.used_margin(on))

    def margin_level(self, on: Optional[date] = None) -> Optional[Decimal]:
        """Equity as a percentage of used margin, or ``None`` with no positions."""
        used = self.used_margin(on)
        if used == 0:
            return None
        return (self.equity(on) / used * 100).quantize(Decimal("0.01"))

    # ------------------------------------------------------------------
    # Opening and closing
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_exits(
        side: Side,
        fill_price: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
    ) -> None:
        """Reject stop-loss / take-profit levels that sit on the wrong side."""
        if side is Side.BUY:
            if stop_loss is not None and stop_loss >= fill_price:
                raise ValueError(f"stop-loss {stop_loss} must be below the buy price {fill_price}")
            if take_profit is not None and take_profit <= fill_price:
                raise ValueError(
                    f"take-profit {take_profit} must be above the buy price {fill_price}"
                )
        else:
            if stop_loss is not None and stop_loss <= fill_price:
                raise ValueError(f"stop-loss {stop_loss} must be above the sell price {fill_price}")
            if take_profit is not None and take_profit >= fill_price:
                raise ValueError(
                    f"take-profit {take_profit} must be below the sell price {fill_price}"
                )

    def open_position(
        self,
        pair: Pair,
        side: Side,
        units: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        on: Optional[date] = None,
        fill_price: Optional[Decimal] = None,
    ) -> Position:
        """Open a position at the current market price.

        Raises:
            ValueError: if ``units`` is not positive or the exits are inverted.
            InsufficientMargin: if the account cannot fund the position.
        """
        units = Decimal(units)
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        on = on or self.account.current_date
        if fill_price is None:
            quote = self.quote(pair, on)
            fill_price = quote.ask if side is Side.BUY else quote.bid
        self._validate_exits(side, fill_price, stop_loss, take_profit)

        required = self.margin_for(pair, units, on)
        available = self.free_margin(on)
        if required > available:
            raise InsufficientMargin(
                f"{pair} {units:,} needs {required} {self.account.currency} margin "
                f"but only {available} is free"
            )

        position = Position(
            id=self.account.take_id(),
            pair=pair,
            side=side,
            units=units,
            entry_price=fill_price,
            opened_on=on,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.account.positions.append(position)
        return position

    def place_order(
        self,
        pair: Pair,
        side: Side,
        units: Decimal,
        order_type: OrderType,
        price: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        on: Optional[date] = None,
    ) -> Order:
        """Rest a limit or stop order to be filled on a later bar."""
        units = Decimal(units)
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if order_type is OrderType.MARKET:
            raise ValueError("use open_position() for market orders")
        self._validate_exits(side, Decimal(price), stop_loss, take_profit)
        order = Order(
            id=self.account.take_id(),
            pair=pair,
            side=side,
            units=units,
            type=order_type,
            price=Decimal(price),
            created_on=on or self.account.current_date,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.account.orders.append(order)
        return order

    def cancel_order(self, order_id: int) -> Order:
        order = self.account.order(order_id)
        if order.status is not OrderStatus.PENDING:
            raise ValueError(f"order {order_id} is already {order.status.value}")
        order.status = OrderStatus.CANCELLED
        return order

    def close_position(
        self,
        position_id: int,
        reason: str = "manual",
        on: Optional[date] = None,
        exit_price: Optional[Decimal] = None,
    ) -> Trade:
        """Close a position and book the realised profit into the balance."""
        position = self.account.position(position_id)
        on = on or self.account.current_date
        if exit_price is None:
            exit_price = position.exit_price(self.quote(position.pair, on))

        pnl_quote = position.pnl_in_quote(exit_price)
        pnl = quantize_money(self.to_account_currency(pnl_quote, position.pair.quote, on))
        trade = Trade(
            id=position.id,
            pair=position.pair,
            side=position.side,
            units=position.units,
            entry_price=position.entry_price,
            exit_price=Decimal(exit_price),
            opened_on=position.opened_on,
            closed_on=on,
            pnl=pnl,
            pips=position.pips(exit_price).quantize(Decimal("0.1")),
            reason=reason,
        )
        self.account.positions.remove(position)
        self.account.record_trade(trade)
        return trade

    def close_all(self, reason: str = "manual", on: Optional[date] = None) -> List[Trade]:
        return [
            self.close_position(position.id, reason=reason, on=on)
            for position in list(self.account.positions)
        ]

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    def advance(self, days: int = 1) -> List[Event]:
        """Move the clock forward, processing each day's fills and exits."""
        if days < 1:
            raise ValueError(f"days must be at least 1, got {days}")
        events: List[Event] = []
        for _ in range(days):
            self.account.current_date += timedelta(days=1)
            events.extend(self.process_day(self.account.current_date))
        return events

    def process_day(self, on: date) -> List[Event]:
        """Fill triggered orders, honour stop-loss / take-profit, check margin."""
        quotes = self._quotes_for_day(on)
        events: List[Event] = []
        events.extend(self._fill_pending_orders(on, quotes))
        events.extend(self._apply_exits(on, quotes))
        events.extend(self._check_margin_call(on))
        return events

    def _quotes_for_day(self, on: date) -> Dict[Pair, Quote]:
        """Price every pair the account currently cares about, skipping gaps."""
        pairs = {order.pair for order in self.account.pending_orders}
        pairs |= {position.pair for position in self.account.positions}
        quotes: Dict[Pair, Quote] = {}
        for pair in pairs:
            try:
                quotes[pair] = self.quote(pair, on)
            except RateUnavailable:
                continue
        return quotes

    def _fill_pending_orders(self, on: date, quotes: Dict[Pair, Quote]) -> List[Event]:
        events: List[Event] = []
        for order in self.account.pending_orders:
            quote = quotes.get(order.pair)
            if quote is None or not order.is_triggered(quote):
                continue
            # Fill at the trigger price: a limit never fills worse than asked,
            # and gap risk on stops is out of scope (see module docstring).
            fill_price = order.price
            try:
                position = self.open_position(
                    pair=order.pair,
                    side=order.side,
                    units=order.units,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    on=on,
                    fill_price=fill_price,
                )
            except InsufficientMargin as exc:
                order.status = OrderStatus.CANCELLED
                events.append(
                    Event(on, "order_cancelled", f"order {order.id} rejected: {exc}", order.id)
                )
                continue
            order.status = OrderStatus.FILLED
            order.filled_on = on
            order.fill_price = fill_price
            events.append(
                Event(
                    on,
                    "order_filled",
                    f"{order.type.value} {order.side.value} {order.units:,} {order.pair} "
                    f"@ {order.pair.quantize(fill_price)} (position {position.id})",
                    position.id,
                )
            )
        return events

    def _apply_exits(self, on: date, quotes: Dict[Pair, Quote]) -> List[Event]:
        events: List[Event] = []
        for position in list(self.account.positions):
            quote = quotes.get(position.pair)
            if quote is None:
                continue
            if position.hit_stop_loss(quote):
                reason, level = "stop_loss", position.stop_loss
            elif position.hit_take_profit(quote):
                reason, level = "take_profit", position.take_profit
            else:
                continue
            trade = self.close_position(position.id, reason=reason, on=on, exit_price=level)
            events.append(
                Event(
                    on,
                    reason,
                    f"position {trade.id} {trade.pair} closed @ "
                    f"{trade.pair.quantize(trade.exit_price)} for "
                    f"{trade.pnl} {self.account.currency}",
                    trade.id,
                )
            )
        return events

    def _check_margin_call(self, on: date) -> List[Event]:
        level = self.margin_level(on)
        if level is None or level >= self.margin_call_level:
            return []
        trades = self.close_all(reason="margin_call", on=on)
        return [
            Event(
                on,
                "margin_call",
                f"margin level {level}% below {self.margin_call_level}%; "
                f"closed {len(trades)} position(s)",
            )
        ]

    # ------------------------------------------------------------------
    def snapshot(self, on: Optional[date] = None) -> Dict[str, object]:
        """A summary of the account suitable for display or JSON output."""
        on = on or self.account.current_date
        return {
            "date": on.isoformat(),
            "currency": self.account.currency,
            "balance": str(self.account.balance),
            "equity": str(self.equity(on)),
            "unrealized_pnl": str(self.unrealized_pnl(on)),
            "realized_pnl": str(self.account.realized_pnl),
            "used_margin": str(self.used_margin(on)),
            "free_margin": str(self.free_margin(on)),
            "margin_level": None if self.margin_level(on) is None else str(self.margin_level(on)),
            "leverage": self.account.leverage,
            "open_positions": len(self.account.positions),
            "pending_orders": len(self.account.pending_orders),
            "closed_trades": len(self.account.trades),
        }
