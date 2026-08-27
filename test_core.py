"""Unit tests for the parts where a bug costs real money."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from quantbot.core.config import demo_spec, synthetic_bars
from quantbot.core.contracts import AccountState, Position, Side, StopSpec
from quantbot.core.exits import (AtrStop, BreakEven, ChandelierStop, Composite,
                                 RTargets, StepTrail, TrailAfter)
from quantbot.core.risk import RiskLimits, RiskManager, Sizer


# ---------------------------------------------------------------- sizing
def test_lot_rounding_never_increases_risk():
    spec = demo_spec("EURUSD", "fx5")
    acc = AccountState(balance=10000, equity=10000)
    s = Sizer(risk_per_trade=0.01)
    r = s.size(spec, acc, 1.10000, 1.09800)   # 20 pip stop = 200 points
    # 200 points * $1/point = $200 per lot; $100 risk -> 0.5 lots
    assert r.ok
    assert r.volume == pytest.approx(0.5, abs=1e-9)
    assert r.risk_money <= 100 * 1.001

def test_sizing_scales_with_stop_distance():
    spec = demo_spec("EURUSD", "fx5")
    acc = AccountState(balance=10000, equity=10000)
    s = Sizer(risk_per_trade=0.01)
    tight = s.size(spec, acc, 1.10, 1.099)
    wide = s.size(spec, acc, 1.10, 1.096)
    assert tight.volume > wide.volume
    assert tight.risk_money == pytest.approx(wide.risk_money, rel=0.05)

def test_gold_and_fx_risk_the_same_money():
    """The whole point of SymbolSpec: identical risk across asset classes."""
    acc = AccountState(balance=10000, equity=10000)
    s = Sizer(risk_per_trade=0.01)
    fx = s.size(demo_spec("EURUSD", "fx5"), acc, 1.10, 1.098)
    gold = s.size(demo_spec("XAUUSD", "gold"), acc, 1900.0, 1890.0)
    assert fx.ok and gold.ok
    assert fx.risk_money == pytest.approx(gold.risk_money, rel=0.15)

def test_stop_inside_broker_floor_is_rejected():
    spec = demo_spec("EURUSD", "fx5")
    acc = AccountState(balance=10000, equity=10000)
    r = Sizer().size(spec, acc, 1.10, 1.09999)   # 1 point stop
    assert not r.ok and "floor" in r.rejected

def test_tiny_account_rejects_rather_than_oversizing():
    spec = demo_spec("XAUUSD", "gold")
    acc = AccountState(balance=50, equity=50)
    r = Sizer(risk_per_trade=0.005).size(spec, acc, 1900, 1880)
    assert not r.ok   # must refuse, never round up to volume_min


# ----------------------------------------------------------------- stops
def _pos(side=Side.LONG, entry=1.10, stop=1.09):
    return Position(id="1", symbol="EURUSD", side=side, volume=0.5,
                    entry_price=entry, entry_time=pd.Timestamp("2024-01-01"),
                    strategy="t", stop=StopSpec(price=stop))

def test_r_multiple_maths():
    p = _pos()
    assert p.risk_distance == pytest.approx(0.01)
    assert p.r_multiple(1.12) == pytest.approx(2.0)
    assert p.r_multiple(1.09) == pytest.approx(-1.0)
    ps = _pos(Side.SHORT, 1.10, 1.11)
    assert ps.r_multiple(1.08) == pytest.approx(2.0)

def test_atr_stop_sits_below_entry_for_long():
    bars = synthetic_bars(400, seed=3)
    spec = demo_spec("EURUSD")
    sl = AtrStop(2.0).initial(Side.LONG, float(bars.close.iloc[-1]), bars, spec)
    assert sl < bars.close.iloc[-1]
    sl_s = AtrStop(2.0).initial(Side.SHORT, float(bars.close.iloc[-1]), bars, spec)
    assert sl_s > bars.close.iloc[-1]

def test_breakeven_only_fires_after_trigger():
    bars = synthetic_bars(400, seed=4)
    spec = demo_spec("EURUSD")
    p = _pos(entry=1.10, stop=1.09)
    bars = bars.copy(); bars.loc[bars.index[-1], "close"] = 1.105   # 0.5R
    assert BreakEven(trigger_r=1.0).update(p, bars, spec) is None
    bars.loc[bars.index[-1], "close"] = 1.115                      # 1.5R
    assert BreakEven(trigger_r=1.0).update(p, bars, spec) > p.entry_price

def test_composite_takes_tightest_stop():
    bars = synthetic_bars(400, seed=5)
    spec = demo_spec("EURUSD")
    p = _pos(entry=1.10, stop=1.09)
    bars = bars.copy(); bars.loc[bars.index[-1], "close"] = 1.14   # 4R
    comp = Composite([AtrStop(2.0), BreakEven(0.5), StepTrail(1.0, 0.5)])
    got = comp.update(p, bars, spec)
    # StepTrail at 4R locks 3.5R = 1.135; must beat plain breakeven 1.1005
    assert got == pytest.approx(1.135, abs=1e-4)

def test_step_trail_ratchets_upward():
    bars = synthetic_bars(400, seed=6).copy()
    spec = demo_spec("EURUSD")
    p = _pos(entry=1.10, stop=1.09)
    st = StepTrail(step_r=1.0, lock_r=0.5)
    prev = -np.inf
    for px in (1.11, 1.12, 1.13, 1.14):
        bars.loc[bars.index[-1], "close"] = px
        cur = st.update(p, bars, spec)
        assert cur > prev
        prev = cur


# --------------------------------------------------------------- targets
def test_r_targets_are_ordered_and_normalised():
    spec = demo_spec("EURUSD")
    bars = synthetic_bars(300, seed=7)
    ts = RTargets([1, 2, 3], [0.5, 0.5, 0.5]).build(Side.LONG, 1.10, 1.09, bars, spec)
    assert len(ts) == 3
    assert sum(t.volume_fraction for t in ts) == pytest.approx(1.0)
    assert ts[0].price < ts[1].price < ts[2].price
    assert ts[0].price == pytest.approx(1.11)

def test_short_targets_are_below_entry():
    spec = demo_spec("EURUSD")
    bars = synthetic_bars(300, seed=8)
    ts = RTargets([1, 2]).build(Side.SHORT, 1.10, 1.11, bars, spec)
    assert all(t.price < 1.10 for t in ts)


# ------------------------------------------------------------------ risk
def test_daily_loss_limit_halts_trading():
    rm = RiskManager(RiskLimits(daily_loss_limit=0.03), Sizer())
    t = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    rm.on_equity(AccountState(10000, 10000), t)
    assert rm.halted_reason is None
    rm.on_equity(AccountState(9600, 9600), t + pd.Timedelta(hours=2))
    assert rm.halted_reason and "daily loss" in rm.halted_reason

def test_drawdown_kill_switch():
    rm = RiskManager(RiskLimits(max_drawdown_limit=0.10, daily_loss_limit=0.99), Sizer())
    t = pd.Timestamp("2024-01-02", tz="UTC")
    rm.on_equity(AccountState(12000, 12000), t)
    rm.on_equity(AccountState(10500, 10500), t + pd.Timedelta(days=5))
    assert rm.halted_reason and "drawdown" in rm.halted_reason


# ------------------------------------------------------- no look-ahead
def test_features_are_causal():
    """Truncating history must not change past feature values.

    This is the single most important test in the suite: if it fails, your
    backtest is reading the future and your live results will not match.
    """
    from quantbot.ml.dataset import make_features
    bars = synthetic_bars(1200, seed=11)
    full = make_features(bars)
    trunc = make_features(bars.iloc[:-100])
    common = trunc.index[-50:]
    pd.testing.assert_frame_equal(
        full.loc[common], trunc.loc[common], check_exact=False, rtol=1e-9
    )

def test_strategy_signals_are_causal():
    from quantbot.strategy.library.ema_atr_trend import EmaAtrTrend
    s = EmaAtrTrend()
    bars = synthetic_bars(1500, seed=12)
    full = s.compute(bars.copy())
    trunc = s.compute(bars.iloc[:-80].copy())
    common = trunc.index[-40:]
    for col in ("long_entry", "short_entry"):
        pd.testing.assert_series_equal(
            full.loc[common, col], trunc.loc[common, col], check_names=False
        )


def test_feature_cache_invalidates_on_new_bar():
    """Regression: a rolling window is always `warmup` rows, so a cache keyed
    on length never invalidates and the strategy freezes on its first signal.
    """
    from quantbot.core.config import demo_spec
    from quantbot.core.contracts import AccountState, MarketSnapshot
    from quantbot.strategy.library.ema_atr_trend import EmaAtrTrend

    s = EmaAtrTrend()
    bars = synthetic_bars(1000, seed=13)
    spec = demo_spec("EURUSD")
    acc = AccountState(10000, 10000)

    def snap_at(i, n=400):
        w = bars.iloc[i - n + 1 : i + 1]
        return MarketSnapshot("EURUSD", "H1", w, spec, w.index[-1],
                              float(w.close.iloc[-1]), float(w.close.iloc[-1]), acc)

    a = s.features(snap_at(500))
    b = s.features(snap_at(600))
    assert len(a) == len(b) == 400          # window length is invariant...
    assert a.index[-1] != b.index[-1]       # ...but content must differ
    assert float(a["ema_fast"].iloc[-1]) != float(b["ema_fast"].iloc[-1])
