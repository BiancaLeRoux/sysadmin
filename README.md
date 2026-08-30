# sysadmin

A small collection of operational tooling.

- **`automation_scripts.py`** — an SSH key rotation audit intended to run as a GitHub Action.
  It polls a lightweight agent on each host, flags keys older than a configurable limit, and
  writes a JSON report.
- **`fxpaper/`** — an FX paper trading simulator, documented below.

Run the whole test suite with `python -m pytest`.

---

# fxpaper

An FX paper trading simulator: a Python library with a command line interface.
Fund a virtual account, take long and short positions on currency pairs at realistic
bid/ask prices, attach stop-loss and take-profit levels, then step the clock forward and
watch them trigger.

**No real money and no broker connection are involved.** Nothing here places a real order.

## Install

```bash
pip install -r requirements.txt      # requests, pytest
pip install -e .                     # optional: puts `fxpaper` on your PATH
```

Without the editable install, use `python -m fxpaper` in place of `fxpaper` below.

## A worked session

```bash
fxpaper init --balance 10000 --currency USD --date 2024-01-02
fxpaper quote EUR/USD
fxpaper buy EUR/USD --units 10000 --sl 1.2000 --tp 1.2300
fxpaper positions
fxpaper advance --days 30            # steps the clock; stops and limits fire here
fxpaper history
fxpaper account
```

```
$ fxpaper advance --days 30
Clock is now 2024-02-01.

  [2024-01-09] take_profit: position 1 EUR/USD closed @ 1.23000 for 181.80 USD

$ fxpaper history
ID  PAIR     SIDE  UNITS   ENTRY    EXIT     PIPS     P&L      OPENED      CLOSED      REASON
--  -------  ----  ------  -------  -------  -------  -------  ----------  ----------  -----------
1   EUR/USD  BUY   10,000  1.21182  1.23000  +181.80  +181.80  2024-01-02  2024-01-09  take_profit

1 trade   1 won / 0 lost   net +181.80 USD
```

## Commands

| Command | What it does |
|---|---|
| `init` | Create an account (`--balance`, `--currency`, `--leverage`, `--date`, `--force`) |
| `quote PAIR...` | Show the current bid, ask and spread |
| `buy` / `sell` | Open a position at market (`--units`, `--sl`, `--tp`) |
| `order` | Rest a limit or stop order (`--side`, `--type`, `--price`, `--units`) |
| `cancel ID` | Cancel a pending order |
| `positions` | Open positions marked to market, plus resting orders |
| `advance --days N` | Move the clock forward, filling orders and honouring exits |
| `close ID` / `close --all` | Close positions and realise the profit |
| `account [--json]` | Balance, equity, margin and margin level |
| `history` | Every closed trade with its P&L and exit reason |

## How the simulation works

**Sizing.** Positions are measured in units of the base currency, so 10,000 units of
EUR/USD is 0.1 of a standard lot.

**Pips.** One pip is 0.0001 for most pairs and 0.01 for JPY-quoted pairs; the quote
currency decides, not the base.

**Spread.** The provider supplies a mid rate and the simulator puts half the spread either
side of it (1 pip by default). Buys fill at the ask and sells at the bid, so a round trip
that goes nowhere still loses the spread — as it would in reality.

**Profit.** A long earns `units × (exit − entry)` in the quote currency, which is then
converted into the account currency. Selling USD/JPY from a USD account correctly books
its yen profit back into dollars.

**Margin.** Required margin is the position's notional in the account currency divided by
the leverage (30:1 by default). Orders that would exceed free margin are refused, and if
equity falls below 50% of used margin every position is closed as a margin call.

**Exits.** `advance` samples one price per simulated day. On each new day it fills any
triggered limit or stop orders, then checks every position's stop-loss and take-profit.

**Known simplification:** an order fills exactly at its trigger price. Gap risk — where a
real fill would be worse than the stop you asked for — is not modelled, so results are
slightly optimistic compared with live trading.

## Price sources

Selected with `FXPAPER_PROVIDER`:

| Value | Source | Network |
|---|---|---|
| `synthetic` (default) | Seeded random walk from realistic starting prices | No |
| `csv` | Your own `date,rate` files | No |
| `frankfurter` | ECB daily reference rates from api.frankfurter.dev, free and keyless | Yes |

The synthetic walk is a pure function of `(seed, pair, day)`, so a simulation replays
identically every time and in any order.

```bash
# Your own history: one file per pair, named EURUSD.csv, with a date,rate header
FXPAPER_PROVIDER=csv FXPAPER_CSV_DIR=./data fxpaper quote EUR/USD

# Live ECB rates (needs outbound internet)
FXPAPER_PROVIDER=frankfurter fxpaper quote EUR/USD
```

The ECB publishes every rate against the euro, so a pair with no EUR leg is derived as a
cross: `AUD/JPY = (EUR/JPY) ÷ (EUR/AUD)`.

> **Note on `frankfurter`:** it is unit-tested against mocked HTTP but has not been
> exercised against the live API from this repository's development sandbox, whose network
> policy blocks outbound HTTPS. On a machine with internet, verify it with
> `FXPAPER_PROVIDER=frankfurter fxpaper quote EUR/USD`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FXPAPER_STATE` | `fxpaper_state.json` | Where account state is stored |
| `FXPAPER_PROVIDER` | `synthetic` | `synthetic`, `csv` or `frankfurter` |
| `FXPAPER_SEED` | `7` | Seed for the synthetic walk |
| `FXPAPER_CSV_DIR` | — | Directory of CSV files, required for `csv` |
| `FXPAPER_SPREAD_PIPS` | `1.0` | Simulated spread, also settable per command with `--spread` |
| `FXPAPER_TIMEOUT` | `10` | HTTP timeout in seconds for live providers |

## Library use

```python
from datetime import date
from decimal import Decimal
from fxpaper import Account, Engine, Pair, Side, SyntheticProvider

account = Account(currency="USD", balance=Decimal("10000"), current_date=date(2024, 1, 2))
engine = Engine(account, SyntheticProvider(seed=7))

engine.open_position(
    Pair.parse("EUR/USD"), Side.BUY, Decimal("10000"),
    stop_loss=Decimal("1.2000"), take_profit=Decimal("1.2300"),
)

for event in engine.advance(days=30):
    print(event)

print(engine.snapshot())
```

All prices and money use `decimal.Decimal`, never floats.

## Tests

```bash
python -m pytest          # 184 tests
```

The suite runs entirely offline: the synthetic provider needs no network, and the
Frankfurter provider is tested with `unittest.mock.patch("requests.get")`.
