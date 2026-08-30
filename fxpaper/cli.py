"""Command line interface for the fxpaper simulator.

Run ``fxpaper --help`` (or ``python -m fxpaper --help``) for the full command
list. Every command reads and writes the JSON state file, so a session is just
a sequence of shell invocations::

    fxpaper init --balance 10000 --currency USD --date 2024-01-02
    fxpaper buy EUR/USD --units 10000 --sl 1.0750 --tp 1.1000
    fxpaper advance --days 10
    fxpaper history
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Sequence

from .account import Account, InsufficientMargin
from .engine import DEFAULT_MARGIN_CALL_LEVEL, Engine
from .instruments import InvalidPair, Pair
from .orders import OrderType, Side
from .rates import DEFAULT_SPREAD_PIPS, RateUnavailable, provider_from_env
from .storage import STATE_PATH, StateError, load, save


# ----------------------------------------------------------------------
# Small output helpers
# ----------------------------------------------------------------------
def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a plain fixed-width table."""
    if not rows:
        return "(none)"
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = [
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    ]
    return "\n".join([line, rule, *body])


def _money(amount: Decimal, currency: str) -> str:
    return f"{amount:,.2f} {currency}"


def _signed(amount: Decimal) -> str:
    return f"+{amount:,.2f}" if amount > 0 else f"{amount:,.2f}"


def _decimal(text: str, what: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise SystemExit(f"error: {what} must be a number, got {text!r}") from exc


def _pair(text: str) -> Pair:
    try:
        return Pair.parse(text)
    except InvalidPair as exc:
        raise SystemExit(f"error: {exc}") from exc


# ----------------------------------------------------------------------
# Engine wiring
# ----------------------------------------------------------------------
def _build_engine(args: argparse.Namespace, account: Account) -> Engine:
    return Engine(
        account=account,
        provider=provider_from_env(),
        spread_pips=Decimal(str(args.spread)),
        margin_call_level=DEFAULT_MARGIN_CALL_LEVEL,
    )


def _load_engine(args: argparse.Namespace) -> Engine:
    account = load(args.state)
    return _build_engine(args, account)


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    start = date.fromisoformat(args.date) if args.date else date.today()
    if os.path.exists(args.state) and not args.force:
        print(f"error: {args.state} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    account = Account(
        currency=args.currency.upper(),
        balance=Decimal(str(args.balance)),
        leverage=args.leverage,
        current_date=start,
    )
    path = save(account, args.state)
    print(
        f"Opened a {_money(account.balance, account.currency)} account at {start} "
        f"with {account.leverage}:1 leverage."
    )
    print(f"State: {path}")
    return 0


def cmd_quote(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    for text in args.pairs:
        pair = _pair(text)
        quote = engine.quote(pair)
        print(
            f"{pair}  bid {pair.quantize(quote.bid)}  ask {pair.quantize(quote.ask)}  "
            f"spread {quote.spread_pips:.1f} pips  ({quote.on})"
        )
    return 0


def _open(args: argparse.Namespace, side: Side) -> int:
    engine = _load_engine(args)
    pair = _pair(args.pair)
    try:
        position = engine.open_position(
            pair=pair,
            side=side,
            units=_decimal(str(args.units), "units"),
            stop_loss=_decimal(str(args.sl), "stop-loss") if args.sl else None,
            take_profit=_decimal(str(args.tp), "take-profit") if args.tp else None,
        )
    except (InsufficientMargin, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save(engine.account, args.state)
    print(
        f"#{position.id} {side.value.upper()} {position.units:,} {pair} "
        f"@ {pair.quantize(position.entry_price)}"
    )
    if position.stop_loss or position.take_profit:
        print(
            f"     stop-loss {position.stop_loss or '-'}   "
            f"take-profit {position.take_profit or '-'}"
        )
    print(f"     margin {_money(engine.margin_for(pair, position.units), engine.account.currency)}")
    return 0


def cmd_buy(args: argparse.Namespace) -> int:
    return _open(args, Side.BUY)


def cmd_sell(args: argparse.Namespace) -> int:
    return _open(args, Side.SELL)


def cmd_order(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    pair = _pair(args.pair)
    try:
        order = engine.place_order(
            pair=pair,
            side=Side(args.side),
            units=_decimal(str(args.units), "units"),
            order_type=OrderType(args.type),
            price=_decimal(str(args.price), "price"),
            stop_loss=_decimal(str(args.sl), "stop-loss") if args.sl else None,
            take_profit=_decimal(str(args.tp), "take-profit") if args.tp else None,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save(engine.account, args.state)
    print(
        f"#{order.id} {order.type.value.upper()} {order.side.value.upper()} "
        f"{order.units:,} {pair} @ {pair.quantize(order.price)} (pending)"
    )
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    try:
        order = engine.cancel_order(args.order_id)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save(engine.account, args.state)
    print(f"Cancelled order #{order.id}.")
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    rows: List[List[str]] = []
    for position in engine.account.positions:
        try:
            quote = engine.quote(position.pair)
            current = position.exit_price(quote)
            pnl = engine.to_account_currency(
                position.unrealized_in_quote(quote), position.pair.quote
            )
            current_text = str(position.pair.quantize(current))
            pnl_text = _signed(pnl)
            pips_text = _signed(position.pips(current))
        except RateUnavailable:
            current_text = pnl_text = pips_text = "n/a"
        rows.append(
            [
                str(position.id),
                str(position.pair),
                position.side.value.upper(),
                f"{position.units:,}",
                str(position.pair.quantize(position.entry_price)),
                current_text,
                pips_text,
                pnl_text,
                str(position.stop_loss or "-"),
                str(position.take_profit or "-"),
            ]
        )
    print(
        _table(
            ["ID", "PAIR", "SIDE", "UNITS", "ENTRY", "PRICE", "PIPS", "P&L", "SL", "TP"],
            rows,
        )
    )
    pending = engine.account.pending_orders
    if pending:
        print("\nPending orders")
        print(
            _table(
                ["ID", "PAIR", "SIDE", "UNITS", "TYPE", "TRIGGER"],
                [
                    [
                        str(o.id),
                        str(o.pair),
                        o.side.value.upper(),
                        f"{o.units:,}",
                        o.type.value,
                        str(o.pair.quantize(o.price)),
                    ]
                    for o in pending
                ],
            )
        )
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    try:
        trades = (
            engine.close_all()
            if args.all
            else [engine.close_position(args.position_id)]
        )
    except (KeyError, RateUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save(engine.account, args.state)
    if not trades:
        print("No open positions.")
        return 0
    for trade in trades:
        print(
            f"Closed #{trade.id} {trade.pair} {trade.side.value.upper()} "
            f"@ {trade.pair.quantize(trade.exit_price)}  "
            f"{_signed(trade.pips)} pips  {_signed(trade.pnl)} {engine.account.currency}"
        )
    print(f"Balance: {_money(engine.account.balance, engine.account.currency)}")
    return 0


def cmd_advance(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    try:
        events = engine.advance(args.days)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save(engine.account, args.state)
    print(f"Clock is now {engine.account.current_date}.")
    if events:
        print()
        for event in events:
            print(f"  {event}")
    else:
        print("Nothing triggered.")
    return 0


def cmd_account(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    snapshot = engine.snapshot()
    if args.json:
        print(json.dumps(snapshot, indent=2))
        return 0
    currency = engine.account.currency
    print(f"Date            {snapshot['date']}")
    print(f"Balance         {_money(Decimal(str(snapshot['balance'])), currency)}")
    print(f"Equity          {_money(Decimal(str(snapshot['equity'])), currency)}")
    print(f"Unrealized P&L  {_signed(Decimal(str(snapshot['unrealized_pnl'])))} {currency}")
    print(f"Realized P&L    {_signed(Decimal(str(snapshot['realized_pnl'])))} {currency}")
    print(f"Used margin     {_money(Decimal(str(snapshot['used_margin'])), currency)}")
    print(f"Free margin     {_money(Decimal(str(snapshot['free_margin'])), currency)}")
    print(f"Margin level    {snapshot['margin_level'] or '-'}")
    print(f"Leverage        {snapshot['leverage']}:1")
    print(
        f"Open {snapshot['open_positions']}   "
        f"Pending {snapshot['pending_orders']}   "
        f"Closed {snapshot['closed_trades']}"
    )
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    engine = _load_engine(args)
    trades = engine.account.trades
    print(
        _table(
            ["ID", "PAIR", "SIDE", "UNITS", "ENTRY", "EXIT", "PIPS", "P&L", "OPENED", "CLOSED", "REASON"],
            [
                [
                    str(t.id),
                    str(t.pair),
                    t.side.value.upper(),
                    f"{t.units:,}",
                    str(t.pair.quantize(t.entry_price)),
                    str(t.pair.quantize(t.exit_price)),
                    _signed(t.pips),
                    _signed(t.pnl),
                    t.opened_on.isoformat(),
                    t.closed_on.isoformat(),
                    t.reason,
                ]
                for t in trades
            ],
        )
    )
    if trades:
        wins = sum(1 for t in trades if t.is_win)
        noun = "trade" if len(trades) == 1 else "trades"
        print(
            f"\n{len(trades)} {noun}   {wins} won / {len(trades) - wins} lost   "
            f"net {_signed(engine.account.realized_pnl)} {engine.account.currency}"
        )
    return 0


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fxpaper",
        description="An FX paper trading simulator. No real money is involved.",
    )
    parser.add_argument(
        "--state",
        default=STATE_PATH,
        help=f"path to the account state file (default: {STATE_PATH})",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=float(DEFAULT_SPREAD_PIPS),
        help="simulated spread in pips (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a new account")
    init.add_argument("--balance", type=float, default=10000.0, help="starting balance")
    init.add_argument("--currency", default="USD", help="account currency (default: USD)")
    init.add_argument("--leverage", type=int, default=30, help="leverage, e.g. 30 for 30:1")
    init.add_argument("--date", help="simulation start date, YYYY-MM-DD (default: today)")
    init.add_argument("--force", action="store_true", help="overwrite an existing state file")
    init.set_defaults(func=cmd_init)

    quote = subparsers.add_parser("quote", help="show the current bid/ask")
    quote.add_argument("pairs", nargs="+", metavar="PAIR", help="e.g. EUR/USD")
    quote.set_defaults(func=cmd_quote)

    for name, handler, verb in (("buy", cmd_buy, "long"), ("sell", cmd_sell, "short")):
        sub = subparsers.add_parser(name, help=f"open a {verb} position at market")
        sub.add_argument("pair", metavar="PAIR", help="e.g. EUR/USD")
        sub.add_argument("--units", required=True, help="size in base-currency units")
        sub.add_argument("--sl", help="stop-loss price")
        sub.add_argument("--tp", help="take-profit price")
        sub.set_defaults(func=handler)

    order = subparsers.add_parser("order", help="rest a limit or stop order")
    order.add_argument("pair", metavar="PAIR")
    order.add_argument("--side", required=True, choices=[s.value for s in Side])
    order.add_argument("--type", required=True, choices=["limit", "stop"])
    order.add_argument("--units", required=True)
    order.add_argument("--price", required=True, help="trigger price")
    order.add_argument("--sl", help="stop-loss price")
    order.add_argument("--tp", help="take-profit price")
    order.set_defaults(func=cmd_order)

    cancel = subparsers.add_parser("cancel", help="cancel a pending order")
    cancel.add_argument("order_id", type=int, metavar="ID")
    cancel.set_defaults(func=cmd_cancel)

    positions = subparsers.add_parser("positions", help="list open positions")
    positions.set_defaults(func=cmd_positions)

    close = subparsers.add_parser("close", help="close a position")
    close.add_argument("position_id", type=int, nargs="?", metavar="ID")
    close.add_argument("--all", action="store_true", help="close every open position")
    close.set_defaults(func=cmd_close)

    advance = subparsers.add_parser("advance", help="move the clock forward")
    advance.add_argument("--days", type=int, default=1, help="days to advance (default: 1)")
    advance.set_defaults(func=cmd_advance)

    account = subparsers.add_parser("account", help="show balance, equity and margin")
    account.add_argument("--json", action="store_true", help="emit JSON instead of text")
    account.set_defaults(func=cmd_account)

    history = subparsers.add_parser("history", help="list closed trades")
    history.set_defaults(func=cmd_history)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "close" and not args.all and args.position_id is None:
        parser.error("close needs a position ID or --all")
    try:
        return args.func(args)
    except (StateError, RateUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
