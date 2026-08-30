"""fxpaper — an FX paper trading simulator.

Fund a virtual account, take long and short positions on currency pairs at
realistic bid/ask prices, attach stop-loss and take-profit levels, then step
the clock forward and watch them trigger. No real money, no broker.

Typical library use::

    from datetime import date
    from decimal import Decimal
    from fxpaper import Account, Engine, Pair, Side, SyntheticProvider

    account = Account(currency="USD", balance=Decimal("10000"), current_date=date(2024, 1, 2))
    engine = Engine(account, SyntheticProvider(seed=7))
    engine.open_position(Pair.parse("EUR/USD"), Side.BUY, Decimal("10000"))
    events = engine.advance(days=10)

The command line entry point lives in :mod:`fxpaper.cli`.
"""

from .account import Account, InsufficientMargin, Position, Trade
from .engine import Engine, Event
from .instruments import InvalidPair, Pair, quantize_money
from .orders import Order, OrderStatus, OrderType, Side
from .rates import (
    CsvProvider,
    FrankfurterProvider,
    Quote,
    RateProvider,
    RateUnavailable,
    SyntheticProvider,
    convert,
    provider_from_env,
    spread_quote,
)
from .storage import StateError, load, save

__version__ = "0.1.0"

__all__ = [
    "Account",
    "CsvProvider",
    "Engine",
    "Event",
    "FrankfurterProvider",
    "InsufficientMargin",
    "InvalidPair",
    "Order",
    "OrderStatus",
    "OrderType",
    "Pair",
    "Position",
    "Quote",
    "RateProvider",
    "RateUnavailable",
    "Side",
    "StateError",
    "SyntheticProvider",
    "Trade",
    "__version__",
    "convert",
    "load",
    "provider_from_env",
    "quantize_money",
    "save",
    "spread_quote",
]
