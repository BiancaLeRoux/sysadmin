"""End-to-end tests driving the command line interface.

These run against the synthetic provider so they need no network, and each
test gets its own state file in a temporary directory.
"""

import json

import pytest

from fxpaper.cli import main


@pytest.fixture
def state(tmp_path, monkeypatch):
    """A state file path, with the environment pinned to a fixed synthetic seed."""
    monkeypatch.setenv("FXPAPER_PROVIDER", "synthetic")
    monkeypatch.setenv("FXPAPER_SEED", "7")
    return str(tmp_path / "state.json")


def run(state, *args):
    """Invoke the CLI with the shared --state flag; returns the exit code."""
    return main(["--state", state, *args])


def init(state, *extra):
    return run(state, "init", "--balance", "10000", "--date", "2024-01-02", *extra)


# ----------------------------------------------------------------------
# init
# ----------------------------------------------------------------------
def test_init_creates_a_state_file(state, capsys):
    assert init(state) == 0
    assert "10,000.00 USD" in capsys.readouterr().out
    assert json.loads(open(state).read())["currency"] == "USD"


def test_init_refuses_to_clobber_an_existing_account(state, capsys):
    init(state)
    assert init(state) == 1
    assert "already exists" in capsys.readouterr().err


def test_init_force_overwrites(state):
    init(state)
    assert init(state, "--force") == 0


def test_commands_need_an_account_first(state, capsys):
    assert run(state, "account") == 1
    assert "fxpaper init" in capsys.readouterr().err


# ----------------------------------------------------------------------
# quote
# ----------------------------------------------------------------------
def test_quote_prints_both_sides(state, capsys):
    init(state)
    assert run(state, "quote", "EUR/USD") == 0
    out = capsys.readouterr().out
    assert "EUR/USD" in out and "bid" in out and "ask" in out


def test_quote_accepts_several_pairs(state, capsys):
    init(state)
    run(state, "quote", "EUR/USD", "USD/JPY")
    out = capsys.readouterr().out
    assert "EUR/USD" in out and "USD/JPY" in out


def test_quote_rejects_a_bad_pair(state):
    init(state)
    with pytest.raises(SystemExit):
        run(state, "quote", "NOTAPAIR")


# ----------------------------------------------------------------------
# buy / sell
# ----------------------------------------------------------------------
def test_buy_opens_a_position(state, capsys):
    init(state)
    capsys.readouterr()
    assert run(state, "buy", "EUR/USD", "--units", "10000") == 0
    assert "BUY 10,000 EUR/USD" in capsys.readouterr().out
    assert len(json.loads(open(state).read())["positions"]) == 1


def test_sell_opens_a_short(state):
    init(state)
    assert run(state, "sell", "EUR/USD", "--units", "10000") == 0
    assert json.loads(open(state).read())["positions"][0]["side"] == "sell"


def test_buy_rejects_an_inverted_stop_loss(state, capsys):
    init(state)
    assert run(state, "buy", "EUR/USD", "--units", "10000", "--sl", "99") == 1
    assert "stop-loss" in capsys.readouterr().err


def test_buy_rejects_a_position_it_cannot_fund(state, capsys):
    init(state)
    assert run(state, "buy", "EUR/USD", "--units", "50000000") == 1
    assert "margin" in capsys.readouterr().err


def test_a_rejected_order_leaves_the_state_untouched(state):
    init(state)
    before = open(state).read()
    run(state, "buy", "EUR/USD", "--units", "50000000")
    assert open(state).read() == before


# ----------------------------------------------------------------------
# order / cancel
# ----------------------------------------------------------------------
def test_order_rests_a_limit(state, capsys):
    init(state)
    capsys.readouterr()
    code = run(
        state, "order", "EUR/USD", "--side", "buy", "--type", "limit",
        "--units", "10000", "--price", "1.05000",
    )
    assert code == 0
    assert "pending" in capsys.readouterr().out
    assert json.loads(open(state).read())["orders"][0]["status"] == "pending"


def test_cancel_removes_a_pending_order(state):
    init(state)
    run(
        state, "order", "EUR/USD", "--side", "buy", "--type", "limit",
        "--units", "10000", "--price", "1.05000",
    )
    order_id = json.loads(open(state).read())["orders"][0]["id"]
    assert run(state, "cancel", str(order_id)) == 0
    assert json.loads(open(state).read())["orders"][0]["status"] == "cancelled"


def test_cancelling_an_unknown_order_fails(state, capsys):
    init(state)
    assert run(state, "cancel", "999") == 1
    assert "999" in capsys.readouterr().err


# ----------------------------------------------------------------------
# positions / account / history
# ----------------------------------------------------------------------
def test_positions_is_empty_on_a_new_account(state, capsys):
    init(state)
    capsys.readouterr()
    run(state, "positions")
    assert "(none)" in capsys.readouterr().out


def test_positions_lists_an_open_trade(state, capsys):
    init(state)
    run(state, "buy", "EUR/USD", "--units", "10000")
    capsys.readouterr()
    run(state, "positions")
    out = capsys.readouterr().out
    assert "EUR/USD" in out and "BUY" in out


def test_account_json_output_is_machine_readable(state, capsys):
    init(state)
    capsys.readouterr()
    assert run(state, "account", "--json") == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["currency"] == "USD"
    assert snapshot["balance"] == "10000.00"


def test_account_text_output_shows_the_key_figures(state, capsys):
    init(state)
    capsys.readouterr()
    run(state, "account")
    out = capsys.readouterr().out
    for label in ("Balance", "Equity", "Free margin", "Leverage"):
        assert label in out


def test_history_is_empty_before_any_trade(state, capsys):
    init(state)
    capsys.readouterr()
    run(state, "history")
    assert "(none)" in capsys.readouterr().out


# ----------------------------------------------------------------------
# close
# ----------------------------------------------------------------------
def test_close_needs_an_id_or_all(state):
    init(state)
    with pytest.raises(SystemExit):
        run(state, "close")


def test_close_books_the_trade_into_history(state, capsys):
    init(state)
    run(state, "buy", "EUR/USD", "--units", "10000")
    position_id = json.loads(open(state).read())["positions"][0]["id"]
    capsys.readouterr()
    assert run(state, "close", str(position_id)) == 0
    assert "Closed" in capsys.readouterr().out

    data = json.loads(open(state).read())
    assert data["positions"] == []
    assert len(data["trades"]) == 1


def test_close_all_clears_every_position(state):
    init(state)
    run(state, "buy", "EUR/USD", "--units", "10000")
    run(state, "sell", "USD/JPY", "--units", "10000")
    assert run(state, "close", "--all") == 0
    assert json.loads(open(state).read())["positions"] == []


def test_close_all_on_a_flat_account_says_so(state, capsys):
    init(state)
    capsys.readouterr()
    assert run(state, "close", "--all") == 0
    assert "No open positions" in capsys.readouterr().out


def test_closing_an_unknown_position_fails(state, capsys):
    init(state)
    assert run(state, "close", "999") == 1
    assert "999" in capsys.readouterr().err


# ----------------------------------------------------------------------
# advance
# ----------------------------------------------------------------------
def test_advance_moves_the_clock(state, capsys):
    init(state)
    capsys.readouterr()
    assert run(state, "advance", "--days", "5") == 0
    assert "2024-01-07" in capsys.readouterr().out
    assert json.loads(open(state).read())["current_date"] == "2024-01-07"


def test_advance_rejects_zero_days(state, capsys):
    init(state)
    assert run(state, "advance", "--days", "0") == 1
    assert "at least 1" in capsys.readouterr().err


def test_advance_reports_a_triggered_take_profit(state, capsys):
    init(state)
    # A take-profit one pip above the ask fills on the next bar that ticks up.
    run(state, "buy", "EUR/USD", "--units", "10000", "--tp", "1.21200")
    capsys.readouterr()
    run(state, "advance", "--days", "30")
    assert "take_profit" in capsys.readouterr().out
    assert json.loads(open(state).read())["trades"][0]["reason"] == "take_profit"


# ----------------------------------------------------------------------
# A full session
# ----------------------------------------------------------------------
def test_a_full_session_keeps_the_books_balanced(state, capsys):
    """Balance moves by exactly the realised profit, and nothing is lost."""
    init(state)
    run(state, "buy", "EUR/USD", "--units", "10000", "--tp", "1.21500")
    run(state, "sell", "USD/JPY", "--units", "10000")
    run(state, "advance", "--days", "40")
    run(state, "close", "--all")
    capsys.readouterr()

    run(state, "account", "--json")
    snapshot = json.loads(capsys.readouterr().out)
    data = json.loads(open(state).read())

    from decimal import Decimal

    realized = sum(Decimal(t["pnl"]) for t in data["trades"])
    assert Decimal(snapshot["balance"]) == Decimal("10000.00") + realized
    assert Decimal(snapshot["realized_pnl"]) == realized
    assert snapshot["open_positions"] == 0
    assert snapshot["used_margin"] == "0.00"
    assert len(data["trades"]) == 2


def test_the_state_file_survives_every_command(state):
    init(state)
    for args in (
        ("buy", "EUR/USD", "--units", "10000"),
        ("positions",),
        ("advance", "--days", "3"),
        ("account",),
        ("close", "--all"),
        ("history",),
    ):
        assert run(state, *args) == 0
        json.loads(open(state).read())  # still valid JSON after each step
