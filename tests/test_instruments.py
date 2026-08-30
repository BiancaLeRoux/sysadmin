"""Tests for currency pair parsing and FX quoting conventions."""

from decimal import Decimal

import pytest

from fxpaper.instruments import InvalidPair, Pair, quantize_money


@pytest.mark.parametrize(
    "text",
    ["EUR/USD", "eur/usd", "EUR_USD", "EUR-USD", "eurusd", "  EUR/USD  "],
)
def test_parse_accepts_common_notations(text):
    assert Pair.parse(text) == Pair("EUR", "USD")


@pytest.mark.parametrize("text", ["EURUS", "EUR/US", "", "EUR//USD", "12/USD", "EUR/USD/GBP"])
def test_parse_rejects_malformed_input(text):
    with pytest.raises(InvalidPair):
        Pair.parse(text)


def test_pair_normalises_case():
    pair = Pair("eur", "usd")
    assert (pair.base, pair.quote) == ("EUR", "USD")
    assert str(pair) == "EUR/USD"


def test_pair_rejects_identical_currencies_regardless_of_case():
    with pytest.raises(InvalidPair):
        Pair("eur", "EUR")


def test_inverse_swaps_base_and_quote():
    assert Pair("EUR", "USD").inverse == Pair("USD", "EUR")


def test_pip_size_is_smaller_for_four_decimal_quotes():
    assert Pair("EUR", "USD").pip_size == Decimal("0.0001")
    assert Pair("GBP", "USD").pip_size == Decimal("0.0001")


def test_pip_size_is_one_hundredth_for_jpy_quotes():
    # A JPY-quoted pair prices to 3 decimals, so its pip is 0.01, not 0.0001.
    assert Pair("USD", "JPY").pip_size == Decimal("0.01")
    assert Pair("EUR", "JPY").pip_size == Decimal("0.01")


def test_pip_size_follows_the_quote_currency_not_the_base():
    # JPY as the *base* does not change the convention; the quote decides.
    assert Pair("JPY", "USD").pip_size == Decimal("0.0001")


def test_price_precision_matches_the_pip_convention():
    assert Pair("EUR", "USD").price_precision == 5
    assert Pair("USD", "JPY").price_precision == 3


def test_quantize_rounds_to_display_precision():
    assert Pair("EUR", "USD").quantize(Decimal("1.234567")) == Decimal("1.23457")
    assert Pair("USD", "JPY").quantize(Decimal("150.98765")) == Decimal("150.988")


def test_pips_between_counts_in_the_right_units():
    eurusd = Pair("EUR", "USD")
    assert eurusd.pips_between(Decimal("1.1000"), Decimal("1.1050")) == Decimal("50")
    assert eurusd.pips_between(Decimal("1.1050"), Decimal("1.1000")) == Decimal("-50")

    usdjpy = Pair("USD", "JPY")
    assert usdjpy.pips_between(Decimal("150.00"), Decimal("150.50")) == Decimal("50")


def test_quantize_money_rounds_half_up_to_cents():
    assert quantize_money(Decimal("10.005")) == Decimal("10.01")
    assert quantize_money(Decimal("10.004")) == Decimal("10.00")
    assert quantize_money(Decimal("-10.005")) == Decimal("-10.01")


def test_pairs_are_hashable_and_usable_as_dict_keys():
    assert {Pair("EUR", "USD"): 1}[Pair.parse("eurusd")] == 1
