"""Tests for the three rate providers and the spread model."""

import json
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
import requests

from fxpaper.instruments import Pair
from fxpaper.rates import (
    SYNTHETIC_EPOCH,
    CsvProvider,
    FrankfurterProvider,
    RateUnavailable,
    SyntheticProvider,
    convert,
    provider_from_env,
    spread_quote,
)

EURUSD = Pair("EUR", "USD")
USDJPY = Pair("USD", "JPY")


# ----------------------------------------------------------------------
# spread_quote
# ----------------------------------------------------------------------
def test_spread_is_applied_evenly_either_side_of_the_mid():
    quote = spread_quote(EURUSD, Decimal("1.10000"), date(2024, 1, 2), Decimal("2.0"))
    assert quote.bid == Decimal("1.09990")
    assert quote.ask == Decimal("1.10010")
    assert quote.mid == Decimal("1.10000")
    assert quote.spread_pips == Decimal("2")


def test_spread_uses_the_jpy_pip_size_for_jpy_pairs():
    quote = spread_quote(USDJPY, Decimal("150.000"), date(2024, 1, 2), Decimal("2.0"))
    # 2 pips on a JPY pair is 0.02, not 0.0002
    assert quote.bid == Decimal("149.990")
    assert quote.ask == Decimal("150.010")


def test_quote_prices_are_rounded_to_pair_precision():
    quote = spread_quote(EURUSD, Decimal("1.123456789"), date(2024, 1, 2), Decimal("1.0"))
    assert quote.bid == Decimal("1.12341")
    assert quote.ask == Decimal("1.12351")


def test_ask_always_exceeds_bid():
    quote = spread_quote(EURUSD, Decimal("1.10000"), date(2024, 1, 2), Decimal("0.5"))
    assert quote.ask > quote.bid


# ----------------------------------------------------------------------
# SyntheticProvider
# ----------------------------------------------------------------------
def test_synthetic_is_deterministic_for_a_seed():
    a = SyntheticProvider(seed=42)
    b = SyntheticProvider(seed=42)
    day = date(2024, 6, 1)
    assert a.rate(EURUSD, day) == b.rate(EURUSD, day)


def test_synthetic_differs_between_seeds():
    day = date(2024, 6, 1)
    assert SyntheticProvider(seed=1).rate(EURUSD, day) != SyntheticProvider(seed=2).rate(
        EURUSD, day
    )


def test_synthetic_is_order_independent():
    """Asking for a late date first must not change an earlier date's rate."""
    forwards = SyntheticProvider(seed=5)
    early = forwards.rate(EURUSD, date(2024, 1, 2))

    backwards = SyntheticProvider(seed=5)
    backwards.rate(EURUSD, date(2025, 1, 2))  # jump ahead first
    assert backwards.rate(EURUSD, date(2024, 1, 2)) == early


def test_synthetic_starts_at_the_seed_price_on_the_epoch():
    provider = SyntheticProvider(seed=3)
    assert provider.rate(EURUSD, SYNTHETIC_EPOCH) == Decimal("1.08500")


def test_synthetic_inverse_is_the_exact_reciprocal():
    provider = SyntheticProvider(seed=9)
    day = date(2024, 3, 1)
    assert provider.rate(Pair("USD", "EUR"), day) == Decimal(1) / provider.rate(EURUSD, day)


def test_synthetic_rejects_dates_before_its_history():
    with pytest.raises(RateUnavailable):
        SyntheticProvider().rate(EURUSD, date(2019, 12, 31))


def test_synthetic_prices_unknown_pairs_stably():
    provider = SyntheticProvider(seed=11)
    exotic = Pair("SEK", "NOK")
    day = date(2024, 5, 5)
    assert provider.rate(exotic, day) == provider.rate(exotic, day)
    assert provider.rate(exotic, day) > 0


def test_synthetic_series_covers_every_day_in_range():
    provider = SyntheticProvider(seed=4)
    series = provider.series(EURUSD, date(2024, 1, 1), date(2024, 1, 10))
    assert len(series) == 10
    assert [day for day, _ in series] == sorted(day for day, _ in series)


def test_synthetic_stays_in_a_plausible_range():
    """A 0.5% daily vol should not send a major pair to absurd levels."""
    provider = SyntheticProvider(seed=7)
    rate = provider.rate(EURUSD, date(2024, 1, 2))
    assert Decimal("0.5") < rate < Decimal("2.5")


# ----------------------------------------------------------------------
# CsvProvider
# ----------------------------------------------------------------------
@pytest.fixture
def csv_dir(tmp_path):
    (tmp_path / "EURUSD.csv").write_text(
        "date,rate\n2024-01-02,1.09400\n2024-01-03,1.09210\n2024-01-05,1.09800\n",
        encoding="utf-8",
    )
    return tmp_path


def test_csv_reads_an_exact_date(csv_dir):
    assert CsvProvider(csv_dir).rate(EURUSD, date(2024, 1, 3)) == Decimal("1.09210")


def test_csv_falls_back_to_the_previous_observation(csv_dir):
    # 4 Jan is missing, so the 3 Jan fixing stands (as it would over a weekend).
    assert CsvProvider(csv_dir).rate(EURUSD, date(2024, 1, 4)) == Decimal("1.09210")


def test_csv_without_a_date_returns_the_latest_row(csv_dir):
    assert CsvProvider(csv_dir).rate(EURUSD) == Decimal("1.09800")


def test_csv_raises_before_the_first_observation(csv_dir):
    with pytest.raises(RateUnavailable):
        CsvProvider(csv_dir).rate(EURUSD, date(2023, 12, 31))


def test_csv_inverts_when_only_the_opposite_file_exists(csv_dir):
    rate = CsvProvider(csv_dir).rate(Pair("USD", "EUR"), date(2024, 1, 2))
    assert rate == Decimal(1) / Decimal("1.09400")


def test_csv_reports_a_missing_pair(csv_dir):
    with pytest.raises(RateUnavailable, match="no CSV"):
        CsvProvider(csv_dir).rate(Pair("AUD", "CAD"), date(2024, 1, 2))


def test_csv_reports_a_malformed_row(tmp_path):
    (tmp_path / "EURUSD.csv").write_text("date,rate\nnot-a-date,1.1\n", encoding="utf-8")
    with pytest.raises(RateUnavailable, match="malformed row"):
        CsvProvider(tmp_path).rate(EURUSD, date(2024, 1, 2))


# ----------------------------------------------------------------------
# FrankfurterProvider — HTTP is mocked; the live API is not called.
# ----------------------------------------------------------------------
def _ecb_response(rates):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"base": "EUR", "date": "2024-01-02", "rates": rates}
    return response


def test_frankfurter_reads_a_eur_based_pair_directly():
    with patch("requests.get", return_value=_ecb_response({"USD": 1.0954})) as mock_get:
        rate = FrankfurterProvider().rate(EURUSD, date(2024, 1, 2))
    assert rate == Decimal("1.0954")
    url = mock_get.call_args.args[0]
    assert url.endswith("/2024-01-02")


def test_frankfurter_uses_the_latest_endpoint_without_a_date():
    with patch("requests.get", return_value=_ecb_response({"USD": 1.0954})) as mock_get:
        FrankfurterProvider().rate(EURUSD)
    assert mock_get.call_args.args[0].endswith("/latest")


def test_frankfurter_derives_a_cross_rate():
    """The ECB only publishes against EUR, so AUD/JPY is a derived cross."""
    with patch("requests.get", return_value=_ecb_response({"AUD": 1.6, "JPY": 160.0})):
        rate = FrankfurterProvider().rate(Pair("AUD", "JPY"), date(2024, 1, 2))
    # EUR/JPY 160 divided by EUR/AUD 1.6 gives AUD/JPY 100
    assert rate == Decimal("100")


def test_frankfurter_handles_eur_as_the_quote_currency():
    with patch("requests.get", return_value=_ecb_response({"USD": 1.25})):
        rate = FrankfurterProvider().rate(Pair("USD", "EUR"), date(2024, 1, 2))
    assert rate == Decimal("0.8")


def test_frankfurter_caches_one_request_per_date():
    provider = FrankfurterProvider()
    with patch("requests.get", return_value=_ecb_response({"USD": 1.1, "JPY": 160.0})) as mock_get:
        provider.rate(EURUSD, date(2024, 1, 2))
        provider.rate(Pair("EUR", "JPY"), date(2024, 1, 2))
    assert mock_get.call_count == 1


def test_frankfurter_wraps_network_failures():
    with patch("requests.get", side_effect=requests.Timeout("timed out")):
        with pytest.raises(RateUnavailable, match="could not fetch"):
            FrankfurterProvider().rate(EURUSD, date(2024, 1, 2))


def test_frankfurter_wraps_http_errors():
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("404")
    with patch("requests.get", return_value=response):
        with pytest.raises(RateUnavailable):
            FrankfurterProvider().rate(EURUSD, date(2024, 1, 2))


def test_frankfurter_wraps_invalid_json():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.side_effect = json.JSONDecodeError("bad", doc="{}", pos=0)
    with patch("requests.get", return_value=response):
        with pytest.raises(RateUnavailable, match="invalid JSON"):
            FrankfurterProvider().rate(EURUSD, date(2024, 1, 2))


def test_frankfurter_reports_an_unpublished_currency():
    with patch("requests.get", return_value=_ecb_response({"USD": 1.0954})):
        with pytest.raises(RateUnavailable, match="XYZ"):
            FrankfurterProvider().rate(Pair("EUR", "XYZ"), date(2024, 1, 2))


def test_frankfurter_sends_a_timeout():
    with patch("requests.get", return_value=_ecb_response({"USD": 1.1})) as mock_get:
        FrankfurterProvider(timeout=3).rate(EURUSD, date(2024, 1, 2))
    assert mock_get.call_args.kwargs["timeout"] == 3


# ----------------------------------------------------------------------
# convert and provider_from_env
# ----------------------------------------------------------------------
def test_convert_is_a_no_op_for_the_same_currency():
    provider = SyntheticProvider()
    assert convert(Decimal("100"), "USD", "USD", provider) == Decimal("100")


def test_convert_uses_the_direct_rate():
    class Stub:
        def rate(self, pair, on=None):
            assert pair == Pair("EUR", "USD")
            return Decimal("1.1")

    assert convert(Decimal("100"), "EUR", "USD", Stub()) == Decimal("110.0")


def test_convert_inverts_when_only_the_opposite_rate_exists():
    class Stub:
        def rate(self, pair, on=None):
            if pair == Pair("EUR", "USD"):
                return Decimal("1.25")
            raise RateUnavailable("only EUR/USD is known")

    # 125 USD at EUR/USD 1.25 is 100 EUR
    assert convert(Decimal("125"), "USD", "EUR", Stub()) == Decimal("100")


def test_provider_from_env_defaults_to_synthetic(monkeypatch):
    monkeypatch.delenv("FXPAPER_PROVIDER", raising=False)
    assert isinstance(provider_from_env(), SyntheticProvider)


def test_provider_from_env_builds_frankfurter(monkeypatch):
    monkeypatch.setenv("FXPAPER_PROVIDER", "frankfurter")
    assert isinstance(provider_from_env(), FrankfurterProvider)


def test_provider_from_env_requires_a_csv_directory(monkeypatch):
    monkeypatch.setenv("FXPAPER_PROVIDER", "csv")
    monkeypatch.delenv("FXPAPER_CSV_DIR", raising=False)
    with pytest.raises(RateUnavailable, match="FXPAPER_CSV_DIR"):
        provider_from_env()


def test_provider_from_env_rejects_an_unknown_name(monkeypatch):
    monkeypatch.setenv("FXPAPER_PROVIDER", "bloomberg")
    with pytest.raises(RateUnavailable, match="unknown"):
        provider_from_env()
