"""Shared fixtures.

Tests use a fixed-price provider wherever the exact price matters, so
assertions are about the trading logic rather than about the shape of a
random walk.
"""

from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import pytest

from fxpaper.account import Account
from fxpaper.engine import Engine
from fxpaper.instruments import Pair
from fxpaper.rates import RateUnavailable

START = date(2024, 1, 2)


class FixedProvider:
    """A provider whose prices the test sets directly.

    ``rates`` maps a pair to its current mid. ``set`` changes a mid so a test
    can walk the market to a level and check what the engine does.
    """

    def __init__(self, rates: Optional[Dict[Pair, Decimal]] = None):
        self.rates: Dict[Pair, Decimal] = dict(rates or {})

    def set(self, pair: Pair, mid: str | Decimal) -> None:
        self.rates[pair] = Decimal(str(mid))

    def rate(self, pair: Pair, on: Optional[date] = None) -> Decimal:
        if pair in self.rates:
            return self.rates[pair]
        if pair.inverse in self.rates:
            return Decimal(1) / self.rates[pair.inverse]
        raise RateUnavailable(f"no fixed rate for {pair}")

    def series(self, pair: Pair, start: date, end: date) -> List[Tuple[date, Decimal]]:
        return [(start, self.rate(pair, start))]


@pytest.fixture
def eurusd() -> Pair:
    return Pair("EUR", "USD")


@pytest.fixture
def usdjpy() -> Pair:
    return Pair("USD", "JPY")


@pytest.fixture
def provider(eurusd, usdjpy) -> FixedProvider:
    return FixedProvider({eurusd: Decimal("1.10000"), usdjpy: Decimal("150.000")})


@pytest.fixture
def account() -> Account:
    return Account(currency="USD", balance=Decimal("10000"), leverage=30, current_date=START)


@pytest.fixture
def engine(account, provider) -> Engine:
    # A 1-pip spread keeps the arithmetic in tests easy to follow.
    return Engine(account, provider, spread_pips=Decimal("1.0"))
