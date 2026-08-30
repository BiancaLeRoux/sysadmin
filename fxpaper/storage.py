"""JSON persistence for account state.

State is written to the path in ``FXPAPER_STATE`` (default ``fxpaper_state.json``
in the working directory). Decimals are stored as strings so no precision is
lost on the round trip.
"""

from __future__ import annotations

import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from .account import Account, Position, Trade
from .instruments import Pair
from .orders import Order, OrderStatus, OrderType, Side

STATE_PATH = os.getenv("FXPAPER_STATE", "fxpaper_state.json")

SCHEMA_VERSION = 1


class StateError(Exception):
    """Raised when a state file is missing or cannot be read."""


def _dec(value: Optional[Any]) -> Optional[str]:
    return None if value is None else str(value)


def _opt_decimal(value: Optional[Any]) -> Optional[Decimal]:
    return None if value is None else Decimal(str(value))


def _opt_date(value: Optional[str]) -> Optional[date]:
    return None if value is None else date.fromisoformat(value)


def position_to_dict(position: Position) -> Dict[str, Any]:
    return {
        "id": position.id,
        "pair": str(position.pair),
        "side": position.side.value,
        "units": _dec(position.units),
        "entry_price": _dec(position.entry_price),
        "opened_on": position.opened_on.isoformat(),
        "stop_loss": _dec(position.stop_loss),
        "take_profit": _dec(position.take_profit),
    }


def position_from_dict(data: Dict[str, Any]) -> Position:
    return Position(
        id=int(data["id"]),
        pair=Pair.parse(data["pair"]),
        side=Side(data["side"]),
        units=Decimal(data["units"]),
        entry_price=Decimal(data["entry_price"]),
        opened_on=date.fromisoformat(data["opened_on"]),
        stop_loss=_opt_decimal(data.get("stop_loss")),
        take_profit=_opt_decimal(data.get("take_profit")),
    )


def order_to_dict(order: Order) -> Dict[str, Any]:
    return {
        "id": order.id,
        "pair": str(order.pair),
        "side": order.side.value,
        "units": _dec(order.units),
        "type": order.type.value,
        "price": _dec(order.price),
        "created_on": order.created_on.isoformat(),
        "stop_loss": _dec(order.stop_loss),
        "take_profit": _dec(order.take_profit),
        "status": order.status.value,
        "filled_on": order.filled_on.isoformat() if order.filled_on else None,
        "fill_price": _dec(order.fill_price),
    }


def order_from_dict(data: Dict[str, Any]) -> Order:
    return Order(
        id=int(data["id"]),
        pair=Pair.parse(data["pair"]),
        side=Side(data["side"]),
        units=Decimal(data["units"]),
        type=OrderType(data["type"]),
        price=Decimal(data["price"]),
        created_on=date.fromisoformat(data["created_on"]),
        stop_loss=_opt_decimal(data.get("stop_loss")),
        take_profit=_opt_decimal(data.get("take_profit")),
        status=OrderStatus(data.get("status", "pending")),
        filled_on=_opt_date(data.get("filled_on")),
        fill_price=_opt_decimal(data.get("fill_price")),
    )


def trade_to_dict(trade: Trade) -> Dict[str, Any]:
    return {
        "id": trade.id,
        "pair": str(trade.pair),
        "side": trade.side.value,
        "units": _dec(trade.units),
        "entry_price": _dec(trade.entry_price),
        "exit_price": _dec(trade.exit_price),
        "opened_on": trade.opened_on.isoformat(),
        "closed_on": trade.closed_on.isoformat(),
        "pnl": _dec(trade.pnl),
        "pips": _dec(trade.pips),
        "reason": trade.reason,
    }


def trade_from_dict(data: Dict[str, Any]) -> Trade:
    return Trade(
        id=int(data["id"]),
        pair=Pair.parse(data["pair"]),
        side=Side(data["side"]),
        units=Decimal(data["units"]),
        entry_price=Decimal(data["entry_price"]),
        exit_price=Decimal(data["exit_price"]),
        opened_on=date.fromisoformat(data["opened_on"]),
        closed_on=date.fromisoformat(data["closed_on"]),
        pnl=Decimal(data["pnl"]),
        pips=Decimal(data["pips"]),
        reason=data["reason"],
    )


def account_to_dict(account: Account) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "currency": account.currency,
        "balance": _dec(account.balance),
        "leverage": account.leverage,
        "current_date": account.current_date.isoformat(),
        "next_id": account._next_id,
        "positions": [position_to_dict(p) for p in account.positions],
        "orders": [order_to_dict(o) for o in account.orders],
        "trades": [trade_to_dict(t) for t in account.trades],
    }


def account_from_dict(data: Dict[str, Any]) -> Account:
    version = data.get("schema", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise StateError(f"unsupported state schema {version}; expected {SCHEMA_VERSION}")
    account = Account(
        currency=data["currency"],
        balance=Decimal(data["balance"]),
        leverage=int(data["leverage"]),
        current_date=date.fromisoformat(data["current_date"]),
        positions=[position_from_dict(p) for p in data.get("positions", [])],
        orders=[order_from_dict(o) for o in data.get("orders", [])],
        trades=[trade_from_dict(t) for t in data.get("trades", [])],
    )
    account._next_id = int(data.get("next_id", 1))
    return account


def save(account: Account, path: os.PathLike | str = STATE_PATH) -> Path:
    """Write the account to ``path`` and return the path written."""
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(account_to_dict(account), handle, indent=2)
        handle.write("\n")
    return target


def load(path: os.PathLike | str = STATE_PATH) -> Account:
    """Read an account from ``path``.

    Raises:
        StateError: if the file does not exist or is not valid state.
    """
    source = Path(path)
    if not source.exists():
        raise StateError(f"no account at {source}; run 'fxpaper init' first")
    try:
        with source.open(encoding="utf-8") as handle:
            return account_from_dict(json.load(handle))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise StateError(f"could not read account state from {source}: {exc}") from exc
