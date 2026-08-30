"""Currency pair definitions and FX price conventions.

A pair is written ``BASE/QUOTE`` (e.g. ``EUR/USD``) and its price is the number
of units of the quote currency required to buy one unit of the base currency.

Only the Python standard library is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

# Currencies quoted with two decimal places rather than four. A "pip" is the
# last decimal place of the conventional quote, so these pairs use 0.01.
TWO_DECIMAL_QUOTES = frozenset({"JPY", "HUF", "KRW"})

_PAIR_RE = re.compile(r"^([A-Za-z]{3})[/_\-]?([A-Za-z]{3})$")


class InvalidPair(ValueError):
    """Raised when a string cannot be read as a currency pair."""


@dataclass(frozen=True, order=True)
class Pair:
    """An FX currency pair, e.g. ``EUR/USD``."""

    base: str
    quote: str

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass __setattr__ to normalise to upper case first,
        # so validation below sees the canonical form.
        base = str(self.base).strip().upper()
        quote = str(self.quote).strip().upper()
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "quote", quote)
        if not (base.isalpha() and len(base) == 3 and quote.isalpha() and len(quote) == 3):
            raise InvalidPair(f"currency codes must be 3 letters: {base}/{quote}")
        if base == quote:
            raise InvalidPair(f"base and quote must differ: {base}/{quote}")

    @classmethod
    def parse(cls, text: str) -> "Pair":
        """Build a pair from ``EUR/USD``, ``EUR_USD``, ``EUR-USD`` or ``eurusd``."""
        match = _PAIR_RE.match(text.strip())
        if not match:
            raise InvalidPair(f"cannot parse currency pair: {text!r}")
        return cls(match.group(1).upper(), match.group(2).upper())

    def __str__(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def inverse(self) -> "Pair":
        """The same two currencies quoted the other way round."""
        return Pair(self.quote, self.base)

    @property
    def pip_size(self) -> Decimal:
        """Value of one pip, expressed in the quote currency."""
        return Decimal("0.01") if self.quote in TWO_DECIMAL_QUOTES else Decimal("0.0001")

    @property
    def price_precision(self) -> int:
        """Decimal places used when displaying a price for this pair."""
        return 3 if self.quote in TWO_DECIMAL_QUOTES else 5

    def quantize(self, price: Decimal) -> Decimal:
        """Round a raw price to this pair's display precision."""
        exponent = Decimal(1).scaleb(-self.price_precision)
        return Decimal(price).quantize(exponent)

    def pips_between(self, start: Decimal, end: Decimal) -> Decimal:
        """Signed distance from ``start`` to ``end`` measured in pips."""
        return (Decimal(end) - Decimal(start)) / self.pip_size


def quantize_money(amount: Decimal) -> Decimal:
    """Round a cash amount to two decimal places (half-up, not banker's rounding)."""
    return Decimal(amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
