"""Price sources for the paper trading simulator.

Three providers implement the same :class:`RateProvider` interface:

``SyntheticProvider``
    A deterministic seeded random walk. Needs no network, so it is the default
    and is what the test suite runs against.
``CsvProvider``
    Reads ``date,rate`` CSV files you supply, one per pair.
``FrankfurterProvider``
    Live ECB daily reference rates from api.frankfurter.dev (free, no API key).
    Requires outbound internet, which some sandboxes block.

Providers return *mid* rates. Tradeable bid/ask are derived from the mid by
:func:`spread_quote`, so the simulator charges a spread on every round trip.

Only the Python standard library and the ``requests`` package are used.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import requests

from .instruments import Pair

# Timeout for HTTP requests to a live rate API (seconds)
REQUEST_TIMEOUT = int(os.getenv("FXPAPER_TIMEOUT", "10"))

# Half-spread is applied either side of the mid; this is the full spread in pips
DEFAULT_SPREAD_PIPS = Decimal(os.getenv("FXPAPER_SPREAD_PIPS", "1.0"))

# Anchor date for the synthetic walk; day 0 uses the pair's seed price exactly
SYNTHETIC_EPOCH = date(2020, 1, 1)

# Daily log-return standard deviation for the synthetic walk (~0.5%, typical
# of major FX pairs)
SYNTHETIC_VOLATILITY = 0.005

# Plausible starting prices, all quoted in their conventional market direction.
SEED_PRICES: Dict[Pair, Decimal] = {
    Pair("EUR", "USD"): Decimal("1.08500"),
    Pair("GBP", "USD"): Decimal("1.27000"),
    Pair("AUD", "USD"): Decimal("0.66000"),
    Pair("NZD", "USD"): Decimal("0.61000"),
    Pair("USD", "JPY"): Decimal("149.500"),
    Pair("USD", "CHF"): Decimal("0.88000"),
    Pair("USD", "CAD"): Decimal("1.36000"),
    Pair("EUR", "GBP"): Decimal("0.85400"),
    Pair("EUR", "JPY"): Decimal("162.200"),
    Pair("EUR", "CHF"): Decimal("0.95500"),
    Pair("GBP", "JPY"): Decimal("189.900"),
}


class RateUnavailable(Exception):
    """Raised when a provider cannot supply a rate for a pair and date."""


@dataclass(frozen=True)
class Quote:
    """A tradeable two-sided price."""

    pair: Pair
    on: date
    bid: Decimal
    ask: Decimal

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread_pips(self) -> Decimal:
        return (self.ask - self.bid) / self.pair.pip_size

    def __str__(self) -> str:
        return (
            f"{self.pair} {self.pair.quantize(self.bid)} / "
            f"{self.pair.quantize(self.ask)} ({self.spread_pips:.1f} pips)"
        )


class RateProvider(Protocol):
    """The interface every price source implements."""

    def rate(self, pair: Pair, on: Optional[date] = None) -> Decimal:
        """Mid rate for ``pair`` on ``on`` (defaults to the latest available)."""

    def series(self, pair: Pair, start: date, end: date) -> List[Tuple[date, Decimal]]:
        """Mid rates for every available day in ``[start, end]``, ascending."""


def spread_quote(
    pair: Pair,
    mid: Decimal,
    on: date,
    spread_pips: Decimal = DEFAULT_SPREAD_PIPS,
) -> Quote:
    """Derive a bid/ask quote by applying half the spread either side of ``mid``.

    Both sides are rounded to the pair's quoting precision, so fills and stored
    prices look like real broker quotes rather than 28-digit Decimals.
    """
    half = (Decimal(spread_pips) * pair.pip_size) / 2
    return Quote(
        pair=pair,
        on=on,
        bid=pair.quantize(Decimal(mid) - half),
        ask=pair.quantize(Decimal(mid) + half),
    )


class _SeriesByDayMixin:
    """Default ``series`` implementation that walks day by day."""

    def series(self, pair: Pair, start: date, end: date) -> List[Tuple[date, Decimal]]:
        out: List[Tuple[date, Decimal]] = []
        current = start
        while current <= end:
            try:
                out.append((current, self.rate(pair, current)))
            except RateUnavailable:
                pass  # gaps (weekends, holidays) are simply absent from the series
            current += timedelta(days=1)
        return out


class SyntheticProvider(_SeriesByDayMixin):
    """Deterministic pseudo-random price history. No network required.

    The walk is a pure function of ``(seed, pair, day)``: asking for the same
    date twice always gives the same rate, in any order, across processes. That
    makes simulations and tests reproducible.

    Rates are internally computed in a canonical direction (the one listed in
    :data:`SEED_PRICES`) and inverted on request, so ``USD/EUR`` is always
    exactly ``1 / EUR/USD``.
    """

    def __init__(self, seed: int = 7, volatility: float = SYNTHETIC_VOLATILITY):
        self.seed = seed
        self.volatility = volatility
        # pair -> list of cumulative multipliers indexed by day offset
        self._walks: Dict[Pair, List[Decimal]] = {}

    def _canonical(self, pair: Pair) -> Tuple[Pair, bool]:
        """Return the pair as priced internally, plus whether to invert."""
        if pair in SEED_PRICES:
            return pair, False
        if pair.inverse in SEED_PRICES:
            return pair.inverse, True
        # Unknown pair: price it in the requested direction from a derived seed.
        return pair, False

    def _seed_price(self, pair: Pair) -> Decimal:
        if pair in SEED_PRICES:
            return SEED_PRICES[pair]
        # Derive a stable, plausible starting price for pairs we don't list.
        digest = hashlib.sha256(f"{self.seed}:{pair}".encode()).hexdigest()
        # Map the digest into roughly 0.50 .. 1.75
        fraction = int(digest[:8], 16) / 0xFFFFFFFF
        return Decimal(str(round(0.5 + fraction * 1.25, 5)))

    def _daily_return(self, pair: Pair, day: int) -> float:
        rng = random.Random(f"{self.seed}:{pair}:{day}")
        return rng.gauss(0.0, self.volatility)

    def _walk_to(self, pair: Pair, day: int) -> Decimal:
        """Cumulative price multiplier at ``day``, extending the cached walk."""
        walk = self._walks.setdefault(pair, [Decimal(1)])
        while len(walk) <= day:
            step = Decimal(str(round(1.0 + self._daily_return(pair, len(walk)), 8)))
            walk.append(walk[-1] * step)
        return walk[day]

    def rate(self, pair: Pair, on: Optional[date] = None) -> Decimal:
        on = on or date.today()
        canonical, invert = self._canonical(pair)
        day = (on - SYNTHETIC_EPOCH).days
        if day < 0:
            raise RateUnavailable(f"synthetic history starts at {SYNTHETIC_EPOCH}")
        price = self._seed_price(canonical) * self._walk_to(canonical, day)
        if invert:
            price = Decimal(1) / price
        return price


class CsvProvider(_SeriesByDayMixin):
    """Reads historical mid rates from ``date,rate`` CSV files.

    One file per pair, named after the pair with no separator, e.g.
    ``EURUSD.csv`` in the configured directory::

        date,rate
        2024-01-02,1.09400
        2024-01-03,1.09210
    """

    def __init__(self, directory: os.PathLike | str):
        self.directory = Path(directory)
        self._cache: Dict[Pair, Dict[date, Decimal]] = {}

    def _load(self, pair: Pair) -> Dict[date, Decimal]:
        if pair in self._cache:
            return self._cache[pair]
        path = self.directory / f"{pair.base}{pair.quote}.csv"
        if not path.exists():
            # Fall back to the inverted file and invert its prices.
            inverse_path = self.directory / f"{pair.quote}{pair.base}.csv"
            if inverse_path.exists():
                inverted = {
                    day: Decimal(1) / rate
                    for day, rate in self._load(pair.inverse).items()
                    if rate != 0
                }
                self._cache[pair] = inverted
                return inverted
            raise RateUnavailable(f"no CSV for {pair} in {self.directory}")
        rows: Dict[date, Decimal] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows[date.fromisoformat(row["date"].strip())] = Decimal(row["rate"].strip())
                except (KeyError, ValueError, AttributeError) as exc:
                    raise RateUnavailable(f"malformed row in {path}: {row!r}") from exc
        if not rows:
            raise RateUnavailable(f"{path} contains no rows")
        self._cache[pair] = rows
        return rows

    def rate(self, pair: Pair, on: Optional[date] = None) -> Decimal:
        rows = self._load(pair)
        if on is None:
            return rows[max(rows)]
        if on in rows:
            return rows[on]
        # Use the most recent earlier observation, as markets close at weekends.
        earlier = [day for day in rows if day <= on]
        if not earlier:
            raise RateUnavailable(f"no {pair} rate on or before {on}")
        return rows[max(earlier)]


class FrankfurterProvider(_SeriesByDayMixin):
    """Live ECB daily reference rates via https://api.frankfurter.dev.

    The ECB publishes everything against EUR, so a pair with no EUR leg is
    derived as a cross rate::

        AUD/JPY = (EUR/JPY) / (EUR/AUD)

    Rates are published once per business day; requests for a weekend or
    holiday return the previous business day's fixing.

    Note: this provider needs outbound internet access. In sandboxes that block
    it, use :class:`SyntheticProvider` or :class:`CsvProvider` instead.
    """

    BASE_URL = "https://api.frankfurter.dev/v1"

    def __init__(self, base_url: Optional[str] = None, timeout: int = REQUEST_TIMEOUT):
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._cache: Dict[date | None, Dict[str, Decimal]] = {}

    def _eur_rates(self, on: Optional[date]) -> Dict[str, Decimal]:
        """All EUR-based rates for a date, cached per date."""
        if on in self._cache:
            return self._cache[on]
        endpoint = "latest" if on is None else on.isoformat()
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, params={"base": "EUR"}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RateUnavailable(f"could not fetch rates from {url}: {exc}") from exc
        except ValueError as exc:
            raise RateUnavailable(f"invalid JSON from {url}: {exc}") from exc

        raw = payload.get("rates")
        if not isinstance(raw, dict):
            raise RateUnavailable(f"unexpected response shape from {url}: {payload!r}")
        rates = {code: Decimal(str(value)) for code, value in raw.items()}
        rates["EUR"] = Decimal(1)
        self._cache[on] = rates
        return rates

    def rate(self, pair: Pair, on: Optional[date] = None) -> Decimal:
        rates = self._eur_rates(on)
        try:
            eur_base = rates[pair.base]
            eur_quote = rates[pair.quote]
        except KeyError as exc:
            raise RateUnavailable(f"ECB publishes no rate for {exc.args[0]}") from exc
        if eur_base == 0:
            raise RateUnavailable(f"zero EUR rate for {pair.base}")
        return eur_quote / eur_base


def provider_from_env() -> RateProvider:
    """Build the provider named by ``FXPAPER_PROVIDER`` (default ``synthetic``)."""
    name = os.getenv("FXPAPER_PROVIDER", "synthetic").strip().lower()
    if name == "synthetic":
        return SyntheticProvider(seed=int(os.getenv("FXPAPER_SEED", "7")))
    if name == "csv":
        directory = os.getenv("FXPAPER_CSV_DIR")
        if not directory:
            raise RateUnavailable("FXPAPER_PROVIDER=csv requires FXPAPER_CSV_DIR")
        return CsvProvider(directory)
    if name == "frankfurter":
        return FrankfurterProvider()
    raise RateUnavailable(
        f"unknown FXPAPER_PROVIDER {name!r}; expected synthetic, csv or frankfurter"
    )


def convert(
    amount: Decimal,
    from_currency: str,
    to_currency: str,
    provider: RateProvider,
    on: Optional[date] = None,
) -> Decimal:
    """Convert ``amount`` between currencies using the provider's mid rate."""
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency == to_currency:
        return Decimal(amount)
    try:
        return Decimal(amount) * provider.rate(Pair(from_currency, to_currency), on)
    except RateUnavailable:
        # Try the other direction before giving up.
        rate = provider.rate(Pair(to_currency, from_currency), on)
        if rate == 0:
            raise
        return Decimal(amount) / rate
