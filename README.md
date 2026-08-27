# QuantBot — pluggable strategy framework for MT5 / Exness

A modular algorithmic trading system in Python. You add strategies as
self-contained plug-ins; each one declares its **entry conditions**, its
**stop-loss policy** (static or dynamic) and its **target ladder**. The same
strategy code runs in backtest, dry-run and live MT5 with no changes.

---

## 1. Why it is built this way

The central design decision: **strategies never talk to the broker and never
size positions.** They emit a `Signal` (side + stop + targets + confidence).
Everything else is infrastructure they cannot bypass.

```
                       ┌──────────────────────────────────────┐
   your strategies ──► │  Signal(side, stop, targets, conf)   │
   (plug-ins)          └───────────────┬──────────────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │  RiskManager (the veto)   │  daily loss, exposure,
                         │  Sizer (lots from risk)   │  spread, session, caps
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │   Broker (interface)      │
                         └──┬─────────────────────┬──┘
                            │                     │
                     PaperBroker            Mt5Broker
                     (backtest)             (live Exness)
```

Three properties fall out of this that are hard to retrofit later:

1. **One code path.** `compute()` runs vectorised over full history for
   backtests; the engine then reads only the last row live. There is no
   separate "live version" of the logic to drift out of sync.
2. **Risk cannot be bypassed.** A strategy cannot size its own position or
   place an order. Every trade passes the sizer and the guard layer.
3. **Asset-agnostic sizing.** All money maths goes through `SymbolSpec`
   (tick value, lot step, stops level from the broker), so 0.5% risk means the
   same thing on EURUSD, XAUUSD, BTCUSD and US30. No hardcoded pip values.

### Anti-look-ahead guarantees

The subtle failure mode in this domain is a backtest that reads the future.
Three mechanisms prevent it:

- `Mt5Broker.bars()` **drops the newest candle**, which is still forming.
  Strategies only ever see closed bars.
- The backtester's per-bar order is fixed: exits on bar *t* → stop updates →
  entries decided on *t* → filled at *t*'s close ± costs.
- When a bar's range contains both the stop and a target, **the stop fills
  first**. Same rule in the triple-barrier labeller, so the ML target matches
  what the engine actually does.

`tests/test_core.py` asserts causality directly: truncating history must not
change past feature or signal values.

### The leak check you should actually run

Causality tests catch static bugs. This catches the subtler ones:

```bash
python scripts/leak_check.py
```

It runs your strategy on synthetic data **with** injected regime drift, then on
a **driftless** random walk where no edge exists by construction. A correct
system shows the edge collapse. Measured on the bundled trend strategy:

```
WITH regime drift    +0.3285R   trades=[75, 117, 66, 34, 5]
DRIFTLESS control    +0.0480R   trades=[72, 99, 76, 81, 19]
control noise floor  ~ +/-0.064R
```

The control sits inside its own noise floor — healthy. If the control were
also strongly positive, the strategy would be reading the future, and the
script says so and exits non-zero.

**Corollary, and it matters:** a flattering number on synthetic data means
nothing. The demo backtest prints Sharpe ~4.7, purely because
`synthetic_bars` injects persistent 400-bar drift that trend-following is
built to capture. That is an artifact of the fixture, not evidence of edge.

### Two bugs this testing actually caught

Kept here because both are easy to reintroduce:

1. **Step-trail float truncation.** An exact 3R landed on
   `2.999999999999978`; `int()` truncated to 2 steps, so the trailing stop
   lagged a full step for the life of every trade. Fixed with an epsilon.
2. **A feature cache that never invalidated.** The cache keyed on
   `len(bars)` — but a rolling warmup window is *always* exactly `warmup`
   rows, so after the first computation it returned stale features forever.
   Live, the strategy would have frozen on its first signal permanently. Now
   keyed on the last bar's timestamp, with a regression test. This one
   surfaced only because a strategy reported zero trades *and* zero
   rejections; the aggregate stats looked plausible throughout.

---

## 2. Setup

### Windows (required for live trading)

MetaTrader 5's Python API is Windows-only and talks to a *running terminal*
over local IPC. **You do not need to write an MQL5 Expert Advisor.**

```powershell
git clone <your-repo> quantbot && cd quantbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Install **MetaTrader 5** and log into your Exness account.
2. In the terminal: **Tools → Options → Expert Advisors → Allow Algo Trading.**
3. Set credentials as environment variables (the YAML reads `${VAR}`):
   ```powershell
   setx MT5_LOGIN     "12345678"
   setx MT5_PASSWORD  "your-password"
   setx MT5_SERVER    "Exness-MT5Trial14"
   ```
   Use a **demo account** first. The exact server string is in the terminal
   under File → Login to Trade Account.
4. Verify the connection and dump your real contract specs:
   ```powershell
   python scripts\inspect_symbols.py --groups forex metals --csv specs.csv
   ```

### macOS / Linux

The `MetaTrader5` package will not install. Options, best first:

| Option | Notes |
|---|---|
| Windows VM / cloud VPS | Run terminal + bot together. What most people do. |
| Wine + Windows Python | Works, fiddly to maintain. |
| MQL5 socket bridge EA | An EA in the terminal exposes ticks/orders over ZeroMQ; implement `Broker` against it. The interface makes this a contained job — no other module changes. |

Research and backtesting work natively everywhere: `PaperBroker` needs no
broker connection at all.

---

## 3. Run it

```bash
# 1. verify the machinery on synthetic data — no broker needed
python scripts/run_backtest.py --synthetic --bars 6000

# 2. unit tests, including the look-ahead assertions
pytest tests -q

# 3. leak check — edge must collapse on the driftless control
python scripts/leak_check.py

# 4. backtest on real history pulled from your terminal
python scripts/run_backtest.py --bars 10000 --symbols EURUSD XAUUSD

# 5. train the ML model (walk-forward validated)
python scripts/train.py --symbols EURUSD --timeframe H1 --bars 30000 \
       --out models/ml_h1_eurusd.joblib

# 6. DRY RUN — logs every order it would send, sends nothing
python scripts/run_live.py

# 7. only when dry-run logs look right, for days not minutes
python scripts/run_live.py --live
```

VS Code launch configs for all of these are in `.vscode/launch.json` (F5).

---

## 4. Adding a strategy

One file, one class. This is the whole contract:

```python
# quantbot/strategy/library/my_edge.py
from ...core import indicators as ind
from ..base import Strategy, register

@register("my_edge")
class MyEdge(Strategy):
    warmup = 300
    params = {"lookback": 20, "rsi_max": 35}

    def compute(self, bars):
        bars["atr"] = ind.atr(bars, 14)
        bars["rsi"] = ind.rsi(bars["close"], 14)
        hi = bars["high"].rolling(self.p["lookback"]).max().shift(1)

        bars["long_entry"]  = (bars["close"] > hi) & (bars["rsi"] < self.p["rsi_max"])
        bars["short_entry"] = False
        bars["long_exit"]   = bars["rsi"] > 70
        bars["confidence"]  = 1.0
        return bars
```

Add it to `library/__init__.py`, then wire it up in `configs/config.yaml`:

```yaml
strategies:
  - name: my_edge
    instance: my_edge_h1        # lets you run the same class twice
    timeframe: H1
    symbols: [EURUSD, GBPUSD]
    params: {lookback: 30, rsi_max: 40}
    stop:                        # composite: base + modifiers
      - type: atr
        mult: 2.0
      - type: breakeven
        trigger_r: 1.0
      - type: trail_after
        trigger_r: 1.5
        inner: {type: chandelier, lookback: 20, mult: 2.5}
    target:
      type: r
      multiples: [1.0, 2.5]
      fractions: [0.5, 0.3]      # remaining 20% rides the trail
```

That is the extension point you asked for — strategies compose freely, each
with its own instruments, timeframe, stop behaviour and target ladder.

### Rules the base class enforces for you

- Only `compute()` is required; `entry()` reads the flag columns by default.
- A signal **must** carry a stop. `Signal.__post_init__` raises otherwise.
- A stop on the wrong side of entry is rejected before it reaches the broker.
- **Stops only ratchet.** `Broker.modify_stop` refuses any move that widens
  risk, so a buggy policy cannot enlarge a loss after entry.
- `compute()` must be causal. Never use a negative `shift()`.

### Stop policies available

| Base (pick one) | Modifiers (stack freely) |
|---|---|
| `fixed` (points/pips) | `breakeven` — to entry at *N*R |
| `percent` | `trail_after` — activate an inner policy at *N*R |
| `atr` — volatility-scaled | `step_trail` — lock *k*R for every *N*R gained |
| `structure` — beyond last swing | `time` — flatten after *N* bars |
| `chandelier` — trails extremes | |
| `ema` — rides a moving average | |

Targets: `r` (R multiples), `atr`, `channel`, `none` (let the trail decide).
Partial ladders are managed by the engine via partial closes, since MT5
supports only one TP per position.

---

## 5. The ML layer

The model answers one narrow question: **will the profit barrier be hit before
the stop barrier?** Narrow beats clever — a classifier whose target matches
the engine's real exit rules can be trusted; a "predict tomorrow's price"
regressor cannot.

- **Labels** — triple-barrier (`ml/dataset.py`): walk forward from each bar to
  ±*k*·ATR or a time limit. Ties resolve pessimistically to the stop.
- **Features** — ~30 scale-free columns (vol-normalised returns, ATR-relative
  distances, channel position, ADX, candle micro-structure, cyclical time) so
  one model generalises across symbols.
- **Validation** — purged/embargoed expanding walk-forward. Plain K-fold leaks
  badly here: a training row's label is built from bars that sit inside the
  test window.
- **Calibration** — isotonic, because the probability both gates entry *and*
  scales size. An uncalibrated 0.8 that is really 0.55 sizes up exactly when
  it shouldn't.

Read **`expectancy_r` before AUC.** AUC 0.55 with positive expectancy beats
AUC 0.65 with negative expectancy — the second is right about trades that
don't pay. `train.py` prints a loud warning and tells you not to trade a model
with negative out-of-sample expectancy.

**Keep the barriers and the config in sync.** If the model trained on
`profit_atr: 2.0, loss_atr: 1.0`, the strategy must use `atr` stop with
`mult: 1.0` and a `2R` target. `MlBarrier.on_start()` warns on mismatch,
because otherwise the probability describes a trade you are not placing.

---

## 6. Trading all Exness instruments

```yaml
universe:
  auto:
    enabled: true
    groups: [forex, metals, indices]
    max_symbols: 30
    max_spread_points: 30
```

`Mt5Broker.symbols()` enumerates everything with `SYMBOL_TRADE_MODE_FULL` and
tags each into forex/metals/indices/crypto/energy/stocks so `RiskLimits` can
cap exposure per group. Practical cautions:

- **Symbol suffixes.** Exness serves the same instrument as `EURUSD`,
  `EURUSDm` or `EURUSDz` by account type. `resolve_symbol()` handles this;
  never hardcode names.
- **Market Watch.** The terminal only streams symbols that are selected;
  `ensure_selected()` subscribes them.
- **Correlation is the real risk.** 20 FX pairs is roughly three bets, not
  twenty. `max_positions_per_group` exists for this reason — widening the
  universe without tightening group caps increases risk, it doesn't diversify.
- Scan cost grows linearly. 100 symbols × 3 strategies is 300 evaluations per
  bar; start with 10–30.

---

## 7. What this framework does *not* give you

Worth being blunt, because the gap between "working system" and "profitable
system" is where most of the effort actually goes:

- **No edge.** The bundled strategies are structurally sound examples, not
  money-makers. On synthetic random-walk data they produce roughly zero
  expectancy after costs — correctly, since there is nothing there to find.
  Finding a real edge is the hard part and this repo doesn't do it for you.
- **Costs are modelled, not measured.** Get your real spread/commission/swap
  from your own account and put them in `backtest.costs`. Optimistic costs are
  the single most common way a backtest lies.
- **No intrabar truth.** Bar data cannot tell you whether the stop or the
  target hit first inside one candle. We assume the stop. Tick-level testing
  is the only way to do better.
- **Overfitting is on you.** Every parameter you tune on the same history buys
  in-sample performance you will not keep. Hold out data you never look at.
- **Synthetic gold is unrealistically volatile** in `synthetic_bars`, which is
  why some sizing rejections appear in the demo run. Real XAUUSD H1 ATR is
  ~$4–6.

### Go-live checklist

1. `pytest tests -q` passes, look-ahead tests included.
2. `leak_check.py` reports healthy — edge collapses on the driftless control.
3. `inspect_symbols.py` — confirm tick values and lot steps on **your** account.
4. Backtest with **your** real costs, on data you did not tune against.
5. `run_live.py` in **dry run** for days; read the logs.
6. Demo account, `--live`, smallest possible risk.
7. Real money, `risk_per_trade` ≤ 0.25%, one strategy, one symbol.
8. Scale only after weeks of live results that match the backtest.

Nothing here is financial advice, and leveraged trading can lose more than
you deposit. Treat every number this system prints as a hypothesis.

---

## 8. Layout

```
quantbot/
├── core/
│   ├── contracts.py     SymbolSpec, Signal, Position, MarketSnapshot
│   ├── indicators.py    vectorised TA, no native deps
│   ├── risk.py          Sizer + RiskManager (audit this file first)
│   ├── exits.py         stop & target policies + YAML factory
│   └── config.py        config loading, synthetic data generator
├── strategy/
│   ├── base.py          Strategy ABC, @register, build_strategy
│   └── library/         ema_atr_trend, donchian_breakout, ml_barrier
├── ml/
│   ├── dataset.py       features + triple-barrier + purged walk-forward
│   └── model.py         training, metrics, ModelBundle persistence
├── broker/
│   ├── base.py          the Broker port
│   ├── mt5_broker.py    live Exness/MT5 adapter
│   └── paper_broker.py  simulated fills with costs
└── engine/
    ├── backtester.py    portfolio backtest + metrics
    └── live.py          bar-driven live loop, state recovery

scripts/
├── inspect_symbols.py   dump YOUR account's real contract specs — run first
├── run_backtest.py      synthetic / MT5 history / CSV
├── leak_check.py        drift vs driftless control
├── train.py             walk-forward ML training
└── run_live.py          dry-run and live
```
