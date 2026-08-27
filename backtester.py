"""Bar-by-bar portfolio backtester.

Order of operations within each bar — this sequence IS the anti-look-ahead
guarantee, so read it before changing anything:

  1. advance the cursor to bar t
  2. process exits for positions opened at t-1 or earlier, using bar t's
     high/low (stop checked before target)
  3. update dynamic stops using bars up to and including t
  4. ask strategies for entries using bars up to and including t
  5. fill new entries at bar t's close +/- spread/slippage

A strategy therefore never sees bar t before deciding on bar t, and never
trades at a price that occurred before its decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..broker.paper_broker import CostModel, PaperBroker
from ..core.contracts import (
    ExitReason,
    MarketSnapshot,
    Position,
    Side,
    SymbolSpec,
)
from ..core.risk import RiskLimits, RiskManager, Sizer
from ..strategy.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    fills: pd.DataFrame
    stats: dict[str, float]
    per_strategy: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejections: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["=" * 58, "BACKTEST RESULT", "=" * 58]
        for k, v in self.stats.items():
            lines.append(f"{k:<28} {v:>14,.4f}" if isinstance(v, float)
                         else f"{k:<28} {v:>14}")
        return "\n".join(lines)


class Backtester:
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        specs: dict[str, SymbolSpec],
        strategies: list[Strategy],
        sizer: Sizer | None = None,
        limits: RiskLimits | None = None,
        costs: CostModel | None = None,
        starting_balance: float = 10_000.0,
    ):
        self.data = data
        self.specs = specs
        self.strategies = strategies
        self.sizer = sizer or Sizer()
        self.risk = RiskManager(limits or RiskLimits(), self.sizer)
        self.broker = PaperBroker(data, specs, starting_balance, costs)
        self.rejections: dict[str, int] = {}

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        for s in self.strategies:
            s.on_start()

        # Unified chronological index across all symbols.
        all_times = sorted(set().union(*[set(df.index) for df in self.data.values()]))
        positions_by_index: dict[str, int] = {}
        equity_points: list[tuple[datetime, float]] = []

        warmup = max(s.warmup for s in self.strategies)

        for t in all_times:
            self.broker.now = t
            # 1. advance cursors
            active: list[str] = []
            for sym, df in self.data.items():
                idx = df.index.get_indexer([t], method="ffill")[0]
                if idx < 0:
                    continue
                if df.index[idx] != t:
                    continue          # this symbol has no bar at t
                self.broker.cursor[sym] = idx
                active.append(sym)
            if not active:
                continue

            account = self.broker.account()
            self.risk.on_equity(account, t)

            # 2 & 3. manage open positions
            for pos in list(self.broker.positions()):
                if pos.symbol not in active:
                    continue
                self._manage(pos, t)

            # 4 & 5. look for entries
            if all(self.broker.cursor[s] >= warmup for s in active):
                self._scan_entries(active, t)

            equity_points.append((t, self.broker.account().equity))

        # flatten anything still open at the end
        for pos in list(self.broker.positions()):
            self.broker.close(pos, None, "end_of_data")

        equity = pd.Series(
            [e for _, e in equity_points],
            index=pd.DatetimeIndex([d for d, _ in equity_points]),
            name="equity",
        )
        fills = pd.DataFrame([f.__dict__ for f in self.broker.fills])
        return BacktestResult(
            equity=equity,
            fills=fills,
            stats=compute_stats(equity, fills),
            per_strategy=group_stats(fills, "strategy"),
            per_symbol=group_stats(fills, "symbol"),
            rejections=self.rejections,
        )

    # ------------------------------------------------------------------
    def _manage(self, pos: Position, t: datetime) -> None:
        bar = self.broker.current_bar(pos.symbol)
        spec = self.broker.spec(pos.symbol)
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        pos.bars_held += 1

        # track excursions (for MFE/MAE analytics and step trailing)
        if pos.side is Side.LONG:
            pos.max_favourable = max(pos.max_favourable, high)
            pos.max_adverse = min(pos.max_adverse, low)
        else:
            pos.max_favourable = min(pos.max_favourable, low)
            pos.max_adverse = max(pos.max_adverse, high)

        # --- STOP FIRST (pessimistic when both barriers sit in one bar) ---
        if pos.touches_stop(high, low):
            self.broker.close(pos, None, ExitReason.STOP.value, price=pos.stop.price)
            return

        # --- targets, nearest first ---------------------------------------
        for tgt in sorted(
            pos.targets, key=lambda x: x.price * (1 if pos.side is Side.LONG else -1)
        ):
            if tgt.filled:
                continue
            hit = high >= tgt.price if pos.side is Side.LONG else low <= tgt.price
            if not hit:
                continue
            vol = spec.round_volume(pos.initial_volume * tgt.volume_fraction)
            remaining_after = pos.volume - vol
            if remaining_after < spec.volume_min:
                vol = pos.volume        # avoid leaving an untradeable stub
            tgt.filled = True
            self.broker.close(pos, vol, f"{ExitReason.TARGET.value}:{tgt.label}",
                              price=tgt.price)
            if pos.volume <= 1e-9:
                return

        # --- time stop ----------------------------------------------------
        max_bars = pos.meta.get("max_bars")
        if max_bars and pos.bars_held >= max_bars:
            self.broker.close(pos, None, ExitReason.TIME.value, price=close)
            return

        # --- dynamic stop update -----------------------------------------
        strat = next((s for s in self.strategies if s.instance_name == pos.strategy), None)
        if strat is None:
            return
        bars = self.broker.bars(pos.symbol, strat.timeframe, strat.warmup)
        new_stop = strat.stop_policy.update(pos, bars, spec)
        if new_stop is not None:
            self.broker.modify_stop(pos, new_stop)

        # --- discretionary exit ------------------------------------------
        snap = self._snapshot(pos.symbol, strat, t)
        if snap is not None and strat.exit_signal(pos, snap):
            self.broker.close(pos, None, ExitReason.SIGNAL.value, price=close)

    # ------------------------------------------------------------------
    def _scan_entries(self, active: list[str], t: datetime) -> None:
        for strat in self.strategies:
            targets = strat.symbols or active
            for sym in targets:
                if sym not in active:
                    continue
                snap = self._snapshot(sym, strat, t)
                if snap is None:
                    continue
                try:
                    signal = strat.entry(snap)
                except ValueError:
                    continue          # invalid stop, already logged
                if signal is None:
                    continue

                spec = snap.spec
                atr_val = None
                feats = strat.features(snap)
                if "atr" in feats:
                    atr_val = float(feats["atr"].iloc[-1])

                verdict = self.risk.check(
                    signal, spec, snap.account, t, snap.open_positions,
                    {s: self.broker.spec(s) for s in self.data}, atr_val,
                )
                if not verdict:
                    self.rejections[verdict.reason] = self.rejections.get(verdict.reason, 0) + 1
                    continue

                sizing = self.sizer.size(
                    spec, snap.account, signal.entry_price, signal.stop.price,
                    signal.risk_multiplier,
                )
                if not sizing.ok:
                    key = f"sizing:{sizing.code or 'unknown'}"
                    self.rejections[key] = self.rejections.get(key, 0) + 1
                    continue

                pos = self.broker.open(signal, sizing.volume)
                if pos is not None:
                    pos.meta["max_bars"] = signal.max_bars_in_trade
                    pos.meta["risk_money"] = sizing.risk_money
                    strat.on_position_opened(pos)

    # ------------------------------------------------------------------
    def _snapshot(self, sym: str, strat: Strategy, t: datetime) -> MarketSnapshot | None:
        bars = self.broker.bars(sym, strat.timeframe, strat.warmup)
        if len(bars) < strat.warmup:
            return None
        spec = self.broker.spec(sym)
        bid, ask = self.broker.tick(sym)
        return MarketSnapshot(
            symbol=sym, timeframe=strat.timeframe, bars=bars, spec=spec,
            server_time=t, bid=bid, ask=ask, account=self.broker.account(),
            open_positions=self.broker.positions(),
        )


# ==========================================================================
# Metrics
# ==========================================================================
def compute_stats(equity: pd.Series, fills: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    rets = equity.pct_change().dropna()

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = (end / start) ** (1 / years) - 1 if start > 0 and years > 0 else 0.0

    # Annualisation factor inferred from the bar spacing.
    if len(equity) > 2:
        med_sec = np.median(np.diff(equity.index.values).astype("timedelta64[s]").astype(float))
        bars_per_year = (365.25 * 24 * 3600) / max(med_sec, 1)
    else:
        bars_per_year = 252.0
    sharpe = (rets.mean() / rets.std(ddof=0) * np.sqrt(bars_per_year)) if rets.std(ddof=0) else 0.0
    downside = rets[rets < 0].std(ddof=0)
    sortino = (rets.mean() / downside * np.sqrt(bars_per_year)) if downside else 0.0

    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = float(dd.min())

    stats: dict[str, float] = {
        "start_equity": start,
        "end_equity": end,
        "total_return": end / start - 1 if start else 0.0,
        "cagr": cagr,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd else 0.0,
    }

    if fills.empty:
        stats["trades"] = 0
        return stats

    closes = fills[fills["reason"] != "partial"]
    pnl = closes["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    stats.update(
        {
            "trades": int(len(closes)),
            "win_rate": float(len(wins) / len(closes)) if len(closes) else 0.0,
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() else float("inf"),
            "expectancy_money": float(pnl.mean()),
            "expectancy_r": float(closes["r_multiple"].mean()),
            "best_trade": float(pnl.max()),
            "worst_trade": float(pnl.min()),
            "total_commission": float(fills["commission"].sum()),
        }
    )
    return stats


def group_stats(fills: pd.DataFrame, by: str) -> pd.DataFrame:
    if fills.empty or by not in fills:
        return pd.DataFrame()
    g = fills.groupby(by)
    out = pd.DataFrame(
        {
            "trades": g.size(),
            "net_pnl": g["pnl"].sum(),
            "expectancy_r": g["r_multiple"].mean(),
            "win_rate": g["pnl"].apply(lambda s: (s > 0).mean()),
            "profit_factor": g["pnl"].apply(
                lambda s: s[s > 0].sum() / abs(s[s <= 0].sum()) if (s <= 0).any() else np.inf
            ),
        }
    )
    return out.sort_values("net_pnl", ascending=False)
