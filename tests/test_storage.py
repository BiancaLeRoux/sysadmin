"""Tests for JSON persistence of account state."""

import json
from datetime import date
from decimal import Decimal

import pytest

from fxpaper.account import Account
from fxpaper.engine import Engine
from fxpaper.instruments import Pair
from fxpaper.orders import OrderStatus, OrderType, Side
from fxpaper.storage import SCHEMA_VERSION, StateError, account_to_dict, load, save

EURUSD = Pair("EUR", "USD")
USDJPY = Pair("USD", "JPY")


@pytest.fixture
def populated(engine, provider, eurusd, usdjpy):
    """An account with an open position, a resting order and a closed trade."""
    closed = engine.open_position(eurusd, Side.BUY, Decimal("10000"))
    provider.set(eurusd, "1.11000")
    engine.close_position(closed.id)
    provider.set(eurusd, "1.10000")
    engine.open_position(
        usdjpy,
        Side.SELL,
        Decimal("25000"),
        stop_loss=Decimal("152.000"),
        take_profit=Decimal("148.000"),
    )
    engine.place_order(eurusd, Side.BUY, Decimal("5000"), OrderType.LIMIT, Decimal("1.05000"))
    return engine.account


def test_round_trip_preserves_the_whole_account(populated, tmp_path):
    path = save(populated, tmp_path / "state.json")
    restored = load(path)

    assert restored.currency == populated.currency
    assert restored.balance == populated.balance
    assert restored.leverage == populated.leverage
    assert restored.current_date == populated.current_date
    assert len(restored.positions) == len(populated.positions)
    assert len(restored.orders) == len(populated.orders)
    assert len(restored.trades) == len(populated.trades)


def test_round_trip_preserves_position_detail(populated, tmp_path):
    restored = load(save(populated, tmp_path / "state.json"))
    original, copy = populated.positions[0], restored.positions[0]
    assert copy.pair == original.pair
    assert copy.side is original.side
    assert copy.units == original.units
    assert copy.entry_price == original.entry_price
    assert copy.stop_loss == original.stop_loss
    assert copy.take_profit == original.take_profit
    assert copy.opened_on == original.opened_on


def test_round_trip_preserves_order_detail(populated, tmp_path):
    restored = load(save(populated, tmp_path / "state.json"))
    order = restored.orders[0]
    assert order.type is OrderType.LIMIT
    assert order.status is OrderStatus.PENDING
    assert order.price == Decimal("1.05000")


def test_round_trip_preserves_trade_detail(populated, tmp_path):
    restored = load(save(populated, tmp_path / "state.json"))
    original, copy = populated.trades[0], restored.trades[0]
    assert copy.pnl == original.pnl
    assert copy.pips == original.pips
    assert copy.reason == original.reason
    assert copy.exit_price == original.exit_price


def test_decimals_survive_without_precision_loss(tmp_path):
    account = Account(balance=Decimal("10000.01"), current_date=date(2024, 1, 2))
    restored = load(save(account, tmp_path / "state.json"))
    assert restored.balance == Decimal("10000.01")
    assert isinstance(restored.balance, Decimal)


def test_ids_continue_after_a_reload(populated, tmp_path):
    expected = populated.take_id() + 1
    save(populated, tmp_path / "state.json")
    restored = load(tmp_path / "state.json")
    # take_id above consumed one; the saved file already had the incremented value
    assert restored.take_id() == expected


def test_saved_file_is_readable_json(populated, tmp_path):
    path = save(populated, tmp_path / "state.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA_VERSION
    assert data["currency"] == "USD"


def test_save_creates_missing_directories(populated, tmp_path):
    path = save(populated, tmp_path / "nested" / "deeper" / "state.json")
    assert path.exists()


def test_loading_a_missing_file_explains_how_to_start(tmp_path):
    with pytest.raises(StateError, match="fxpaper init"):
        load(tmp_path / "absent.json")


def test_loading_malformed_json_raises(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StateError):
        load(path)


def test_loading_incomplete_state_raises(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema": SCHEMA_VERSION, "currency": "USD"}), encoding="utf-8")
    with pytest.raises(StateError):
        load(path)


def test_an_unknown_schema_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    payload = account_to_dict(Account(current_date=date(2024, 1, 2)))
    payload["schema"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StateError, match="schema"):
        load(path)


def test_a_reloaded_account_keeps_trading_correctly(populated, provider, tmp_path):
    """Persistence must not change what the engine does next."""
    restored = load(save(populated, tmp_path / "state.json"))
    engine = Engine(restored, provider, spread_pips=Decimal("1.0"))
    provider.set(USDJPY, "147.000")  # through the short's take-profit
    events = engine.advance(1)
    assert [event.kind for event in events] == ["take_profit"]
    assert restored.trades[-1].pnl > 0
