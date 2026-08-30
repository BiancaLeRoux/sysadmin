"""Tests for order filling, margin, exits and the simulation clock."""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from fxpaper.account import InsufficientMargin
from fxpaper.engine import Engine
from fxpaper.instruments import Pair
from fxpaper.orders import OrderStatus, OrderType, Side

from .conftest import START, FixedProvider

EURUSD = Pair("EUR", "USD")
USDJPY = Pair("USD", "JPY")


# ----------------------------------------------------------------------
# Opening positions
# ----------------------------------------------------------------------
def test_a_buy_fills_at_the_ask_and_a_sell_at_the_bid(engine, eurusd):
    quote = engine.quote(eurusd)
    long_position = engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    short_position = engine.open_position(eurusd, Side.SELL, Decimal("10000"))
    assert long_position.entry_price == quote.ask
    assert short_position.entry_price == quote.bid


def test_opening_rejects_non_positive_size(engine, eurusd):
    with pytest.raises(ValueError, match="units must be positive"):
        engine.open_position(eurusd, Side.BUY, Decimal("0"))
    with pytest.raises(ValueError, match="units must be positive"):
        engine.open_position(eurusd, Side.BUY, Decimal("-100"))


def test_a_long_stop_loss_must_sit_below_the_fill(engine, eurusd):
    with pytest.raises(ValueError, match="stop-loss"):
        engine.open_position(eurusd, Side.BUY, Decimal("10000"), stop_loss=Decimal("1.20000"))


def test_a_long_take_profit_must_sit_above_the_fill(engine, eurusd):
    with pytest.raises(ValueError, match="take-profit"):
        engine.open_position(eurusd, Side.BUY, Decimal("10000"), take_profit=Decimal("1.00000"))


def test_a_short_stop_loss_must_sit_above_the_fill(engine, eurusd):
    with pytest.raises(ValueError, match="stop-loss"):
        engine.open_position(eurusd, Side.SELL, Decimal("10000"), stop_loss=Decimal("1.00000"))


def test_a_short_take_profit_must_sit_below_the_fill(engine, eurusd):
    with pytest.raises(ValueError, match="take-profit"):
        engine.open_position(eurusd, Side.SELL, Decimal("10000"), take_profit=Decimal("1.20000"))


# ----------------------------------------------------------------------
# Margin
# ----------------------------------------------------------------------
def test_margin_is_notional_over_leverage(engine, eurusd):
    # 10,000 EUR at 1.10 is 11,000 USD; at 30:1 that needs 366.67 USD
    assert engine.margin_for(eurusd, Decimal("10000")) == Decimal("366.67")


def test_margin_converts_the_base_currency_into_the_account_currency(engine, usdjpy):
    # The base of USD/JPY is already USD, so 10,000 units is 10,000 USD notional
    assert engine.margin_for(usdjpy, Decimal("10000")) == Decimal("333.33")


def test_used_margin_adds_up_across_positions(engine, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    assert engine.used_margin() == Decimal("733.34")


def test_free_margin_is_equity_less_used_margin(engine, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    assert engine.free_margin() == engine.equity() - engine.used_margin()


def test_an_oversized_position_is_refused(engine, eurusd):
    # 10,000 USD at 30:1 supports ~300,000 USD notional, so 5m units is too big
    with pytest.raises(InsufficientMargin):
        engine.open_position(eurusd, Side.BUY, Decimal("5000000"))


def test_a_refused_position_is_not_recorded(engine, eurusd):
    with pytest.raises(InsufficientMargin):
        engine.open_position(eurusd, Side.BUY, Decimal("5000000"))
    assert engine.account.positions == []


def test_margin_level_is_none_without_positions(engine):
    assert engine.margin_level() is None


def test_margin_level_is_equity_over_used_margin(engine, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    expected = (engine.equity() / engine.used_margin() * 100).quantize(Decimal("0.01"))
    assert engine.margin_level() == expected


# ----------------------------------------------------------------------
# Valuation
# ----------------------------------------------------------------------
def test_a_fresh_long_is_down_by_the_spread(engine, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    # 1 pip of spread on 10,000 units is 1 USD
    assert engine.unrealized_pnl() == Decimal("-1.00")


def test_equity_tracks_an_unrealized_gain(engine, provider, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    provider.set(eurusd, "1.11000")
    # Entry 1.10005, now exiting at the bid of 1.10995: 99 pips on 10k = 99 USD
    assert engine.unrealized_pnl() == Decimal("99.00")
    assert engine.equity() == Decimal("10099.00")


def test_profit_on_a_jpy_pair_is_converted_into_the_account_currency(engine, provider, usdjpy):
    engine.open_position(usdjpy, Side.BUY, Decimal("10000"))
    provider.set(usdjpy, "151.000")
    # 10,000 units * (150.995 - 150.005) = 9,900 JPY, converted at USD/JPY 151
    assert engine.unrealized_pnl() == Decimal("65.56")


# ----------------------------------------------------------------------
# Closing
# ----------------------------------------------------------------------
def test_closing_realizes_profit_into_the_balance(engine, provider, eurusd):
    position = engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    provider.set(eurusd, "1.11000")
    opening_balance = engine.account.balance
    trade = engine.close_position(position.id)
    assert trade.pnl == Decimal("99.00")
    assert engine.account.balance == opening_balance + Decimal("99.00")
    assert engine.account.positions == []
    assert engine.account.trades == [trade]


def test_closing_a_losing_trade_reduces_the_balance(engine, provider, eurusd):
    position = engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    provider.set(eurusd, "1.09000")
    trade = engine.close_position(position.id)
    assert trade.pnl < 0
    assert engine.account.balance < Decimal("10000")


def test_close_all_closes_every_position(engine, eurusd, usdjpy):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    engine.open_position(usdjpy, Side.SELL, Decimal("10000"))
    trades = engine.close_all()
    assert len(trades) == 2
    assert engine.account.positions == []


def test_close_all_on_an_empty_account_is_harmless(engine):
    assert engine.close_all() == []


def test_closing_an_unknown_position_raises(engine):
    with pytest.raises(KeyError):
        engine.close_position(42)


def test_a_closed_trade_keeps_the_reason(engine, eurusd):
    position = engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    assert engine.close_position(position.id, reason="manual").reason == "manual"


# ----------------------------------------------------------------------
# Pending orders
# ----------------------------------------------------------------------
def test_a_market_order_cannot_be_rested(engine, eurusd):
    with pytest.raises(ValueError, match="market orders"):
        engine.place_order(
            eurusd, Side.BUY, Decimal("10000"), OrderType.MARKET, Decimal("1.10000")
        )


def test_a_resting_order_does_not_open_a_position(engine, eurusd):
    engine.place_order(eurusd, Side.BUY, Decimal("10000"), OrderType.LIMIT, Decimal("1.05000"))
    assert engine.account.positions == []
    assert len(engine.account.pending_orders) == 1


def test_a_buy_limit_fills_once_the_ask_drops_to_it(engine, provider, eurusd):
    engine.place_order(eurusd, Side.BUY, Decimal("10000"), OrderType.LIMIT, Decimal("1.05000"))
    provider.set(eurusd, "1.08000")
    assert engine.advance(1) == []  # still above the limit

    provider.set(eurusd, "1.04000")
    events = engine.advance(1)
    assert [event.kind for event in events] == ["order_filled"]
    assert engine.account.positions[0].entry_price == Decimal("1.05000")


def test_a_sell_limit_fills_once_the_bid_rises_to_it(engine, provider, eurusd):
    engine.place_order(eurusd, Side.SELL, Decimal("10000"), OrderType.LIMIT, Decimal("1.15000"))
    provider.set(eurusd, "1.16000")
    events = engine.advance(1)
    assert [event.kind for event in events] == ["order_filled"]
    assert engine.account.positions[0].side is Side.SELL


def test_a_buy_stop_fills_once_the_ask_rises_through_it(engine, provider, eurusd):
    engine.place_order(eurusd, Side.BUY, Decimal("10000"), OrderType.STOP, Decimal("1.15000"))
    provider.set(eurusd, "1.12000")
    assert engine.advance(1) == []  # not yet through the trigger

    provider.set(eurusd, "1.16000")
    assert [event.kind for event in engine.advance(1)] == ["order_filled"]


def test_a_sell_stop_fills_once_the_bid_falls_through_it(engine, provider, eurusd):
    engine.place_order(eurusd, Side.SELL, Decimal("10000"), OrderType.STOP, Decimal("1.05000"))
    provider.set(eurusd, "1.04000")
    assert [event.kind for event in engine.advance(1)] == ["order_filled"]


def test_a_filled_order_is_marked_and_not_filled_twice(engine, provider, eurusd):
    order = engine.place_order(
        eurusd, Side.BUY, Decimal("10000"), OrderType.LIMIT, Decimal("1.09000")
    )
    provider.set(eurusd, "1.08000")
    engine.advance(1)
    assert order.status is OrderStatus.FILLED
    assert order.fill_price == Decimal("1.09000")
    assert engine.advance(1) == []
    assert len(engine.account.positions) == 1


def test_a_cancelled_order_never_fills(engine, provider, eurusd):
    order = engine.place_order(
        eurusd, Side.BUY, Decimal("10000"), OrderType.LIMIT, Decimal("1.09000")
    )
    engine.cancel_order(order.id)
    provider.set(eurusd, "1.08000")
    assert engine.advance(1) == []
    assert engine.account.positions == []


def test_cancelling_twice_is_an_error(engine, eurusd):
    order = engine.place_order(
        eurusd, Side.BUY, Decimal("10000"), OrderType.LIMIT, Decimal("1.09000")
    )
    engine.cancel_order(order.id)
    with pytest.raises(ValueError, match="already cancelled"):
        engine.cancel_order(order.id)


def test_an_order_carries_its_exits_onto_the_position(engine, provider, eurusd):
    engine.place_order(
        eurusd,
        Side.BUY,
        Decimal("10000"),
        OrderType.LIMIT,
        Decimal("1.09000"),
        stop_loss=Decimal("1.08000"),
        take_profit=Decimal("1.12000"),
    )
    provider.set(eurusd, "1.08950")
    engine.advance(1)
    position = engine.account.positions[0]
    assert position.stop_loss == Decimal("1.08000")
    assert position.take_profit == Decimal("1.12000")


def test_an_unaffordable_order_is_cancelled_rather_than_filled(engine, provider, eurusd):
    order = engine.place_order(
        eurusd, Side.BUY, Decimal("5000000"), OrderType.LIMIT, Decimal("1.09000")
    )
    provider.set(eurusd, "1.08000")
    events = engine.advance(1)
    assert [event.kind for event in events] == ["order_cancelled"]
    assert order.status is OrderStatus.CANCELLED
    assert engine.account.positions == []


# ----------------------------------------------------------------------
# Exits during simulation
# ----------------------------------------------------------------------
def test_a_stop_loss_closes_the_position_at_its_level(engine, provider, eurusd):
    position = engine.open_position(
        eurusd, Side.BUY, Decimal("10000"), stop_loss=Decimal("1.09000")
    )
    provider.set(eurusd, "1.08000")
    events = engine.advance(1)
    assert [event.kind for event in events] == ["stop_loss"]
    assert engine.account.positions == []
    trade = engine.account.trades[0]
    assert trade.id == position.id
    assert trade.exit_price == Decimal("1.09000")
    assert trade.reason == "stop_loss"
    assert trade.pnl < 0


def test_a_take_profit_closes_the_position_at_its_level(engine, provider, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"), take_profit=Decimal("1.12000"))
    provider.set(eurusd, "1.13000")
    events = engine.advance(1)
    assert [event.kind for event in events] == ["take_profit"]
    trade = engine.account.trades[0]
    assert trade.exit_price == Decimal("1.12000")
    assert trade.pnl > 0


def test_a_short_is_stopped_out_when_the_market_rises(engine, provider, eurusd):
    engine.open_position(eurusd, Side.SELL, Decimal("10000"), stop_loss=Decimal("1.12000"))
    provider.set(eurusd, "1.13000")
    assert [event.kind for event in engine.advance(1)] == ["stop_loss"]
    assert engine.account.trades[0].pnl < 0


def test_a_position_without_exits_survives_a_long_simulation(engine, provider, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    provider.set(eurusd, "1.05000")
    engine.advance(30)
    assert len(engine.account.positions) == 1


def test_only_the_touched_position_is_closed(engine, provider, eurusd, usdjpy):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"), stop_loss=Decimal("1.09000"))
    engine.open_position(usdjpy, Side.BUY, Decimal("10000"))
    provider.set(eurusd, "1.08000")
    engine.advance(1)
    assert [p.pair for p in engine.account.positions] == [usdjpy]


# ----------------------------------------------------------------------
# Margin call
# ----------------------------------------------------------------------
def test_a_collapse_in_equity_forces_a_margin_call(engine, provider, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("250000"))
    provider.set(eurusd, "1.06500")  # a ~350 pip loss wipes out most of the equity
    events = engine.advance(1)
    assert [event.kind for event in events] == ["margin_call"]
    assert engine.account.positions == []
    assert engine.account.trades[0].reason == "margin_call"


def test_no_margin_call_while_the_account_is_healthy(engine, provider, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    provider.set(eurusd, "1.09900")
    assert engine.advance(1) == []


# ----------------------------------------------------------------------
# The clock
# ----------------------------------------------------------------------
def test_advance_moves_the_date_forward(engine):
    engine.advance(5)
    assert engine.account.current_date == START + timedelta(days=5)


def test_advance_requires_at_least_one_day(engine):
    with pytest.raises(ValueError, match="at least 1"):
        engine.advance(0)


def test_events_are_stamped_with_the_day_they_happened(engine, provider, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"), stop_loss=Decimal("1.09000"))
    provider.set(eurusd, "1.08000")
    events = engine.advance(3)
    assert len(events) == 1
    assert events[0].on == START + timedelta(days=1)


def test_a_missing_price_leaves_a_position_untouched(engine, account, eurusd):
    """A provider gap must not close positions or crash the simulation."""
    engine.open_position(eurusd, Side.BUY, Decimal("10000"), stop_loss=Decimal("1.00000"))
    engine.provider = FixedProvider()  # every lookup now raises RateUnavailable
    assert engine.advance(1) == []
    assert len(account.positions) == 1
    # Unpriceable positions are carried at cost rather than marked
    assert engine.unrealized_pnl() == Decimal("0.00")


# ----------------------------------------------------------------------
# Snapshot
# ----------------------------------------------------------------------
def test_snapshot_reports_the_account_state(engine, eurusd):
    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    snapshot = engine.snapshot()
    assert snapshot["currency"] == "USD"
    assert snapshot["open_positions"] == 1
    assert snapshot["closed_trades"] == 0
    assert snapshot["leverage"] == 30
    assert snapshot["date"] == START.isoformat()


def test_snapshot_is_json_friendly(engine, eurusd):
    import json

    engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    assert json.loads(json.dumps(engine.snapshot()))["open_positions"] == 1


# ----------------------------------------------------------------------
# Books balance end to end
# ----------------------------------------------------------------------
def test_balance_change_equals_the_sum_of_realized_profit(engine, provider, eurusd, usdjpy):
    opening = engine.account.balance
    engine.open_position(eurusd, Side.BUY, Decimal("10000"), take_profit=Decimal("1.12000"))
    engine.open_position(usdjpy, Side.SELL, Decimal("10000"), stop_loss=Decimal("152.000"))
    provider.set(eurusd, "1.13000")
    provider.set(usdjpy, "153.000")
    engine.advance(1)

    assert len(engine.account.trades) == 2
    assert engine.account.balance == opening + engine.account.realized_pnl
    assert engine.equity() == engine.account.balance  # nothing left open


def test_a_flat_account_has_no_used_margin(engine, eurusd):
    position = engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    engine.close_position(position.id)
    assert engine.used_margin() == Decimal("0.00")
    assert engine.free_margin() == engine.equity()
