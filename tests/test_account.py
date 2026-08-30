"""Tests for position P&L arithmetic and the account ledger."""

from datetime import date
from decimal import Decimal

import pytest

from fxpaper.account import Account, Position, Trade
from fxpaper.instruments import Pair
from fxpaper.orders import Side
from fxpaper.rates import spread_quote

EURUSD = Pair("EUR", "USD")
USDJPY = Pair("USD", "JPY")
DAY = date(2024, 1, 2)


def make_position(side=Side.BUY, entry="1.10000", pair=EURUSD, units="10000", **kwargs):
    return Position(
        id=1,
        pair=pair,
        side=side,
        units=Decimal(units),
        entry_price=Decimal(entry),
        opened_on=DAY,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Direction of profit
# ----------------------------------------------------------------------
def test_long_profits_when_the_price_rises():
    position = make_position(Side.BUY, "1.10000")
    # 100 pips on 10,000 units is 100 * 0.0001 * 10,000 = 100 quote units
    assert position.pnl_in_quote(Decimal("1.11000")) == Decimal("100.00000")


def test_long_loses_when_the_price_falls():
    position = make_position(Side.BUY, "1.10000")
    assert position.pnl_in_quote(Decimal("1.09000")) == Decimal("-100.00000")


def test_short_profits_when_the_price_falls():
    position = make_position(Side.SELL, "1.10000")
    assert position.pnl_in_quote(Decimal("1.09000")) == Decimal("100.00000")


def test_short_loses_when_the_price_rises():
    position = make_position(Side.SELL, "1.10000")
    assert position.pnl_in_quote(Decimal("1.11000")) == Decimal("-100.00000")


def test_pnl_scales_linearly_with_size():
    small = make_position(units="10000").pnl_in_quote(Decimal("1.11000"))
    large = make_position(units="50000").pnl_in_quote(Decimal("1.11000"))
    assert large == small * 5


def test_pips_are_signed_by_profitability_not_direction():
    long_win = make_position(Side.BUY, "1.10000").pips(Decimal("1.10500"))
    short_win = make_position(Side.SELL, "1.10000").pips(Decimal("1.09500"))
    assert long_win == Decimal("50")
    assert short_win == Decimal("50")


def test_pips_use_the_jpy_convention_for_jpy_pairs():
    position = make_position(pair=USDJPY, entry="150.000")
    assert position.pips(Decimal("150.500")) == Decimal("50")


# ----------------------------------------------------------------------
# Which side of the book a position exits on
# ----------------------------------------------------------------------
def test_a_long_exits_at_the_bid_and_a_short_at_the_ask():
    quote = spread_quote(EURUSD, Decimal("1.10000"), DAY, Decimal("2.0"))
    assert make_position(Side.BUY).exit_price(quote) == quote.bid
    assert make_position(Side.SELL).exit_price(quote) == quote.ask


def test_a_round_trip_at_an_unchanged_mid_loses_the_spread():
    """Buying at the ask and selling at the bid must cost money."""
    quote = spread_quote(EURUSD, Decimal("1.10000"), DAY, Decimal("2.0"))
    position = make_position(Side.BUY, entry=quote.ask)
    assert position.unrealized_in_quote(quote) < 0
    # 2 pips of spread on 10,000 units is 2 * 0.0001 * 10,000 = 2 quote units
    assert position.unrealized_in_quote(quote) == Decimal("-2.00000")


# ----------------------------------------------------------------------
# Stop-loss and take-profit detection
# ----------------------------------------------------------------------
def test_long_stop_loss_triggers_when_the_bid_falls_to_it():
    position = make_position(Side.BUY, "1.10000", stop_loss=Decimal("1.09000"))
    below = spread_quote(EURUSD, Decimal("1.08950"), DAY, Decimal("1.0"))
    above = spread_quote(EURUSD, Decimal("1.09500"), DAY, Decimal("1.0"))
    assert position.hit_stop_loss(below) is True
    assert position.hit_stop_loss(above) is False


def test_short_stop_loss_triggers_when_the_ask_rises_to_it():
    position = make_position(Side.SELL, "1.10000", stop_loss=Decimal("1.11000"))
    up = spread_quote(EURUSD, Decimal("1.11500"), DAY, Decimal("1.0"))
    down = spread_quote(EURUSD, Decimal("1.10500"), DAY, Decimal("1.0"))
    assert position.hit_stop_loss(up) is True
    assert position.hit_stop_loss(down) is False


def test_long_take_profit_triggers_when_the_bid_reaches_it():
    position = make_position(Side.BUY, "1.10000", take_profit=Decimal("1.11000"))
    hit = spread_quote(EURUSD, Decimal("1.11500"), DAY, Decimal("1.0"))
    assert position.hit_take_profit(hit) is True


def test_short_take_profit_triggers_when_the_ask_falls_to_it():
    position = make_position(Side.SELL, "1.10000", take_profit=Decimal("1.09000"))
    hit = spread_quote(EURUSD, Decimal("1.08500"), DAY, Decimal("1.0"))
    assert position.hit_take_profit(hit) is True


def test_no_exit_levels_never_trigger():
    position = make_position(Side.BUY, "1.10000")
    anywhere = spread_quote(EURUSD, Decimal("9.00000"), DAY, Decimal("1.0"))
    assert position.hit_stop_loss(anywhere) is False
    assert position.hit_take_profit(anywhere) is False


# ----------------------------------------------------------------------
# The account ledger
# ----------------------------------------------------------------------
def test_ids_are_unique_and_increasing():
    account = Account()
    assert [account.take_id() for _ in range(3)] == [1, 2, 3]


def test_balance_is_rounded_to_cents_on_creation():
    assert Account(balance=Decimal("10000.005")).balance == Decimal("10000.01")


def test_leverage_must_be_at_least_one():
    with pytest.raises(ValueError):
        Account(leverage=0)


def test_currency_is_upper_cased():
    assert Account(currency="usd").currency == "USD"


def make_trade(pnl):
    return Trade(
        id=1,
        pair=EURUSD,
        side=Side.BUY,
        units=Decimal("10000"),
        entry_price=Decimal("1.10000"),
        exit_price=Decimal("1.11000"),
        opened_on=DAY,
        closed_on=DAY,
        pnl=Decimal(pnl),
        pips=Decimal("100"),
        reason="manual",
    )


def test_recording_a_trade_moves_profit_into_the_balance():
    account = Account(balance=Decimal("10000"))
    account.record_trade(make_trade("125.50"))
    assert account.balance == Decimal("10125.50")
    assert account.realized_pnl == Decimal("125.50")


def test_recording_a_loss_reduces_the_balance():
    account = Account(balance=Decimal("10000"))
    account.record_trade(make_trade("-75.25"))
    assert account.balance == Decimal("9924.75")


def test_realized_pnl_sums_the_whole_history():
    account = Account(balance=Decimal("10000"))
    account.record_trade(make_trade("100"))
    account.record_trade(make_trade("-40"))
    assert account.realized_pnl == Decimal("60.00")
    assert account.balance == Decimal("10060.00")


def test_is_win_reflects_the_sign_of_profit():
    assert make_trade("1").is_win is True
    assert make_trade("-1").is_win is False
    assert make_trade("0").is_win is False


def test_looking_up_a_missing_position_raises():
    with pytest.raises(KeyError):
        Account().position(99)


def test_looking_up_a_missing_order_raises():
    with pytest.raises(KeyError):
        Account().order(99)
