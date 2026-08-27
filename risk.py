"""Position sizing and portfolio guards.

Two responsibilities, kept apart from strategy logic on purpose:

  1. Sizer  — converts "risk 0.5% of equity" + a stop distance into lots that
              the broker will actually accept.
  2. RiskManager — the veto layer. Every signal passes through it; it can
              reject on daily loss, exposure, spread, session, correlation.

If you only audit one module in this repo, audit this one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Iterable

from .contracts import AccountState, Position, Side, Signal, SymbolSpec

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------
@dataclass
class SizingResult:
    volume: float
    risk_money: float          # what we actually risk after lot rounding
    stop_distance: float
    rejected: str | None = None
    code: str = ""             # stable category, for aggregating diagnostics

    @property
    def ok(self) -> bool:
        return self.volume > 0 and self.rejected is None


@dataclass
class Sizer:
    risk_per_trade: float = 0.005      # fraction of equity
    max_volume: float | None = None    # hard cap regardless of maths
    min_stop_buffer_spreads: float = 1.5   # SL must clear spread by this much
    round_up_tolerance: float = 0.0    # keep 0.0: never round lots up

    def size(
        self,
        spec: SymbolSpec,
        account: AccountState,
        entry: float,
        stop: float,
        risk_multiplier: float = 1.0,
    ) -> SizingResult:
        distance = abs(entry - stop)
        if distance <= 0:
            return SizingResult(0, 0, 0, "zero stop distance", "zero_stop")

        # Broker minimum distance + spread buffer.
        floor = max(spec.min_stop_distance, spec.spread * self.min_stop_buffer_spreads)
        if distance < floor:
            return SizingResult(
                0, 0, distance,
                f"stop {distance:.5f} inside broker floor {floor:.5f}",
                "stop_below_floor",
            )

        risk_money = account.equity * self.risk_per_trade * max(0.0, risk_multiplier)
        money_per_lot = spec.money_per_lot(distance)
        if money_per_lot <= 0:
            return SizingResult(0, 0, distance, "tick_value unavailable", "no_tick_value")

        raw = risk_money / money_per_lot
        volume = spec.round_volume(raw)

        if volume <= 0:
            return SizingResult(
                0, 0, distance,
                f"required {raw:.4f} lots < min {spec.volume_min} "
                f"(risk {risk_money:.2f} too small for this stop)",
                "below_min_lot",
            )
        if self.max_volume is not None:
            volume = min(volume, spec.round_volume(self.max_volume))

        actual_risk = money_per_lot * volume
        # Guard against the rounding pushing risk materially over budget.
        if actual_risk > risk_money * 1.35 and volume > spec.volume_min:
            volume = spec.round_volume(volume - spec.volume_step)
            actual_risk = money_per_lot * volume

        return SizingResult(volume, actual_risk, distance)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
@dataclass
class RiskLimits:
    max_open_positions: int = 8
    max_positions_per_symbol: int = 1
    max_positions_per_group: int = 4         # forex / metals / crypto ...
    max_positions_per_strategy: int = 4
    max_total_risk: float = 0.03             # sum of open 1R exposure / equity
    daily_loss_limit: float = 0.04           # stop trading for the day at -4%
    max_drawdown_limit: float = 0.20         # kill switch on peak-to-trough
    max_spread_points: dict[str, float] = field(default_factory=dict)
    max_spread_atr_ratio: float = 0.25       # skip if spread > 25% of ATR
    trading_hours_utc: tuple[int, int] | None = None   # e.g. (6, 20)
    blocked_weekdays: tuple[int, ...] = ()             # 0=Mon .. 6=Sun
    allow_hedging: bool = False              # both directions on one symbol


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOW = Verdict(True)


class RiskManager:
    """Stateful across the session: tracks daily P/L and equity peak."""

    def __init__(self, limits: RiskLimits, sizer: Sizer):
        self.limits = limits
        self.sizer = sizer
        self._day: date | None = None
        self._day_start_equity: float = 0.0
        self._equity_peak: float = 0.0
        self.halted_reason: str | None = None

    # ---- bookkeeping -----------------------------------------------------
    def on_equity(self, account: AccountState, now: datetime) -> None:
        today = now.date()
        if self._day != today:
            self._day = today
            self._day_start_equity = account.equity
            self.halted_reason = None
        self._equity_peak = max(self._equity_peak, account.equity)

        if self._day_start_equity > 0:
            day_ret = account.equity / self._day_start_equity - 1
            if day_ret <= -abs(self.limits.daily_loss_limit):
                self.halted_reason = f"daily loss limit hit ({day_ret:.2%})"
        if self._equity_peak > 0:
            dd = account.equity / self._equity_peak - 1
            if dd <= -abs(self.limits.max_drawdown_limit):
                self.halted_reason = f"max drawdown kill switch ({dd:.2%})"

    def open_risk_fraction(
        self, positions: Iterable[Position], specs: dict[str, SymbolSpec], equity: float
    ) -> float:
        if equity <= 0:
            return 1.0
        total = 0.0
        for p in positions:
            spec = specs.get(p.symbol)
            if spec is None:
                continue
            # Remaining risk: distance from price-agnostic current stop.
            dist = abs(p.entry_price - p.stop.price)
            total += spec.money_per_lot(dist) * p.volume
        return total / equity

    # ---- the veto --------------------------------------------------------
    def check(
        self,
        signal: Signal,
        spec: SymbolSpec,
        account: AccountState,
        now: datetime,
        positions: list[Position],
        specs: dict[str, SymbolSpec],
        atr_value: float | None = None,
    ) -> Verdict:
        L = self.limits

        if self.halted_reason:
            return Verdict(False, f"halted: {self.halted_reason}")
        if not spec.tradable:
            return Verdict(False, "symbol not tradable")
        if now.weekday() in L.blocked_weekdays:
            return Verdict(False, "blocked weekday")
        if L.trading_hours_utc:
            lo, hi = L.trading_hours_utc
            if not (lo <= now.hour < hi):
                return Verdict(False, f"outside trading hours {lo}-{hi} UTC")

        cap = L.max_spread_points.get(spec.group, L.max_spread_points.get("default"))
        if cap is not None and spec.spread_points > cap:
            return Verdict(False, f"spread {spec.spread_points:.1f}pts > {cap}")
        if atr_value and atr_value > 0 and spec.spread / atr_value > L.max_spread_atr_ratio:
            return Verdict(
                False, f"spread {spec.spread:.5f} > {L.max_spread_atr_ratio:.0%} of ATR"
            )

        if len(positions) >= L.max_open_positions:
            return Verdict(False, f"max open positions ({L.max_open_positions})")

        same_symbol = [p for p in positions if p.symbol == signal.symbol]
        if len(same_symbol) >= L.max_positions_per_symbol:
            return Verdict(False, "symbol already at position cap")
        if not L.allow_hedging and any(p.side is signal.side.opposite for p in same_symbol):
            return Verdict(False, "opposite position open and hedging disabled")

        group_count = sum(1 for p in positions if specs.get(p.symbol, spec).group == spec.group)
        if group_count >= L.max_positions_per_group:
            return Verdict(False, f"group '{spec.group}' at cap")

        strat_count = sum(1 for p in positions if p.strategy == signal.strategy)
        if strat_count >= L.max_positions_per_strategy:
            return Verdict(False, f"strategy '{signal.strategy}' at cap")

        if signal.stop is None:
            return Verdict(False, "signal has no stop")
        entry = signal.entry_price or 0.0
        prospective = self.sizer.size(spec, account, entry, signal.stop.price,
                                      signal.risk_multiplier)
        if not prospective.ok:
            # Use the stable code, not the formatted message: the message
            # embeds live numbers and would explode diagnostic cardinality.
            return Verdict(False, f"sizing:{prospective.code or 'failed'}")

        open_frac = self.open_risk_fraction(positions, specs, account.equity)
        new_frac = open_frac + prospective.risk_money / max(account.equity, 1e-9)
        if new_frac > L.max_total_risk:
            return Verdict(
                False,
                f"total open risk {new_frac:.2%} > cap {L.max_total_risk:.2%}",
            )

        return ALLOW
