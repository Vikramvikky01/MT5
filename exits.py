"""Stop-loss and target policies — the "base or dynamic SL" layer.

A strategy declares *what* it wants (`AtrStop(2.5)`, `RTargets([1, 2, 3])`)
and the engine calls:

    stop_policy.initial(...)   -> price, once, at entry
    stop_policy.update(...)    -> price | None, on every closed bar
    target_policy.build(...)   -> list[TargetSpec], once, at entry

`update` returning None means "leave it". The engine also enforces a ratchet:
a stop may only ever move in the favourable direction. A policy cannot widen
risk after the fact even if it tries — that is a deliberate safety property.

Compose freely:

    stop = Composite([
        AtrStop(mult=2.0),
        BreakEven(trigger_r=1.0, offset_r=0.1),
        TrailAfter(trigger_r=1.5, inner=ChandelierStop(20, 2.5)),
    ])
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from . import indicators as ind
from .contracts import Position, Side, StopSpec, SymbolSpec, TargetSpec


# ==========================================================================
# Stops
# ==========================================================================
class StopPolicy(ABC):
    kind: str = "stop"

    @abstractmethod
    def initial(
        self, side: Side, entry: float, bars: pd.DataFrame, spec: SymbolSpec
    ) -> float:
        ...

    def update(
        self, pos: Position, bars: pd.DataFrame, spec: SymbolSpec
    ) -> float | None:
        """Default: static stop, never moves."""
        return None

    # -- helper: push a stop outside the broker's minimum distance ---------
    @staticmethod
    def _respect_floor(price: float, ref: float, side: Side, spec: SymbolSpec) -> float:
        floor = max(spec.min_stop_distance, spec.spread)
        if side is Side.LONG:
            return spec.round_price(min(price, ref - floor))
        return spec.round_price(max(price, ref + floor))


@dataclass
class FixedPointsStop(StopPolicy):
    """Base SL: a flat distance. `points` is in broker points, `pips` in pips."""

    points: float | None = None
    pips: float | None = None
    kind: str = "fixed"

    def initial(self, side, entry, bars, spec):
        if self.pips is not None:
            dist = self.pips * spec.pip
        elif self.points is not None:
            dist = self.points * spec.point
        else:
            raise ValueError("FixedPointsStop needs points or pips")
        raw = entry - side.sign * dist
        return self._respect_floor(raw, entry, side, spec)


@dataclass
class PercentStop(StopPolicy):
    percent: float = 1.0
    kind: str = "percent"

    def initial(self, side, entry, bars, spec):
        raw = entry * (1 - side.sign * self.percent / 100)
        return self._respect_floor(raw, entry, side, spec)


@dataclass
class AtrStop(StopPolicy):
    """Volatility-scaled base SL — the sane default across mixed assets."""

    mult: float = 2.0
    period: int = 14
    kind: str = "atr"

    def initial(self, side, entry, bars, spec):
        a = float(ind.atr(bars, self.period).iloc[-1])
        raw = entry - side.sign * self.mult * a
        return self._respect_floor(raw, entry, side, spec)


@dataclass
class StructureStop(StopPolicy):
    """Beyond the last confirmed swing, padded by a fraction of ATR."""

    lookback: int = 20
    atr_pad: float = 0.25
    period: int = 14
    kind: str = "structure"

    def initial(self, side, entry, bars, spec):
        window = bars.iloc[-self.lookback:]
        a = float(ind.atr(bars, self.period).iloc[-1])
        level = float(window["low"].min()) if side is Side.LONG else float(window["high"].max())
        raw = level - side.sign * self.atr_pad * a
        return self._respect_floor(raw, entry, side, spec)


@dataclass
class ChandelierStop(StopPolicy):
    """Dynamic SL trailing the highest high (or lowest low) since entry."""

    lookback: int = 22
    mult: float = 3.0
    period: int = 14
    kind: str = "chandelier"

    def initial(self, side, entry, bars, spec):
        return self._level(side, bars, spec, entry)

    def update(self, pos, bars, spec):
        return self._level(pos.side, bars, spec, float(bars["close"].iloc[-1]))

    def _level(self, side, bars, spec, ref):
        window = bars.iloc[-self.lookback:]
        a = float(ind.atr(bars, self.period).iloc[-1])
        if side is Side.LONG:
            raw = float(window["high"].max()) - self.mult * a
        else:
            raw = float(window["low"].min()) + self.mult * a
        return self._respect_floor(raw, ref, side, spec)


@dataclass
class EmaStop(StopPolicy):
    """Ride a moving average — good for trend-following exits."""

    period: int = 21
    atr_pad: float = 0.5
    atr_period: int = 14
    kind: str = "ema"

    def initial(self, side, entry, bars, spec):
        return self._level(side, bars, spec, entry)

    def update(self, pos, bars, spec):
        return self._level(pos.side, bars, spec, float(bars["close"].iloc[-1]))

    def _level(self, side, bars, spec, ref):
        e = float(ind.ema(bars["close"], self.period).iloc[-1])
        a = float(ind.atr(bars, self.atr_period).iloc[-1])
        return self._respect_floor(e - side.sign * self.atr_pad * a, ref, side, spec)


# ---- modifiers (wrap or augment another policy) --------------------------
@dataclass
class BreakEven(StopPolicy):
    """Once price reaches `trigger_r`, move SL to entry (+ small offset)."""

    trigger_r: float = 1.0
    offset_r: float = 0.05
    kind: str = "breakeven"

    def initial(self, side, entry, bars, spec):
        raise NotImplementedError("BreakEven is a modifier — use inside Composite")

    def update(self, pos, bars, spec):
        price = float(bars["close"].iloc[-1])
        if pos.r_multiple(price) < self.trigger_r:
            return None
        target = pos.entry_price + pos.side.sign * self.offset_r * pos.risk_distance
        return spec.round_price(target)


@dataclass
class TrailAfter(StopPolicy):
    """Activate an inner dynamic policy only after `trigger_r` is reached."""

    trigger_r: float = 1.0
    inner: StopPolicy = field(default_factory=lambda: ChandelierStop())
    kind: str = "trail_after"

    def initial(self, side, entry, bars, spec):
        raise NotImplementedError("TrailAfter is a modifier — use inside Composite")

    def update(self, pos, bars, spec):
        price = float(bars["close"].iloc[-1])
        if pos.r_multiple(price) < self.trigger_r:
            return None
        return self.inner.update(pos, bars, spec)


@dataclass
class StepTrail(StopPolicy):
    """Classic ratchet: for every `step_r` of profit, lock in `lock_r`."""

    step_r: float = 1.0
    lock_r: float = 0.5
    kind: str = "step_trail"

    def initial(self, side, entry, bars, spec):
        raise NotImplementedError("StepTrail is a modifier — use inside Composite")

    def update(self, pos, bars, spec):
        r = pos.r_multiple(float(bars["close"].iloc[-1]))
        # Epsilon matters: price arithmetic makes an exact 3R land on
        # 2.999999999999978, and int() would truncate to 2, leaving the trail
        # a full step behind for the life of the trade.
        steps = math.floor(r / self.step_r + 1e-6)
        if steps < 1:
            return None
        locked = (steps - 1) * self.step_r + self.lock_r
        return spec.round_price(
            pos.entry_price + pos.side.sign * locked * pos.risk_distance
        )


@dataclass
class TimeStop(StopPolicy):
    """Not a price stop — signals the engine to flatten after N bars.
    Handled by the engine via `max_bars`; kept here for discoverability."""

    max_bars: int = 48
    kind: str = "time"

    def initial(self, side, entry, bars, spec):
        raise NotImplementedError("TimeStop is a modifier — use inside Composite")


@dataclass
class Composite(StopPolicy):
    """First policy supplies the initial stop; all policies may tighten it.

    Tightest wins: max() of candidates for longs, min() for shorts.
    """

    policies: Sequence[StopPolicy] = ()
    kind: str = "composite"

    def initial(self, side, entry, bars, spec):
        for p in self.policies:
            try:
                return p.initial(side, entry, bars, spec)
            except NotImplementedError:
                continue
        raise ValueError("Composite needs at least one base stop policy")

    def update(self, pos, bars, spec):
        candidates = [
            c for c in (p.update(pos, bars, spec) for p in self.policies) if c is not None
        ]
        if not candidates:
            return None
        return max(candidates) if pos.side is Side.LONG else min(candidates)

    @property
    def max_bars(self) -> int | None:
        for p in self.policies:
            if isinstance(p, TimeStop):
                return p.max_bars
        return None


# ==========================================================================
# Targets
# ==========================================================================
class TargetPolicy(ABC):
    @abstractmethod
    def build(
        self, side: Side, entry: float, stop: float, bars: pd.DataFrame, spec: SymbolSpec
    ) -> list[TargetSpec]:
        ...

    @staticmethod
    def _normalise(targets: list[TargetSpec]) -> list[TargetSpec]:
        """Fractions must sum to <= 1.0; the remainder rides the trailing stop."""
        total = sum(t.volume_fraction for t in targets)
        if total > 1.0:
            for t in targets:
                t.volume_fraction /= total
        return targets


@dataclass
class NoTarget(TargetPolicy):
    """Let the dynamic stop do all the work."""

    def build(self, side, entry, stop, bars, spec):
        return []


@dataclass
class RTargets(TargetPolicy):
    """Scale out at R multiples: RTargets([1, 2, 3], [0.5, 0.25, 0.25])."""

    multiples: Sequence[float] = (1.0, 2.0, 3.0)
    fractions: Sequence[float] | None = None

    def build(self, side, entry, stop, bars, spec):
        risk = abs(entry - stop)
        fracs = self.fractions or [1.0 / len(self.multiples)] * len(self.multiples)
        out = [
            TargetSpec(
                price=spec.round_price(entry + side.sign * m * risk),
                volume_fraction=f,
                label=f"{m:g}R",
            )
            for m, f in zip(self.multiples, fracs)
        ]
        return self._normalise(out)


@dataclass
class AtrTargets(TargetPolicy):
    multiples: Sequence[float] = (1.5, 3.0)
    fractions: Sequence[float] | None = None
    period: int = 14

    def build(self, side, entry, stop, bars, spec):
        a = float(ind.atr(bars, self.period).iloc[-1])
        fracs = self.fractions or [1.0 / len(self.multiples)] * len(self.multiples)
        out = [
            TargetSpec(spec.round_price(entry + side.sign * m * a), f, f"{m:g}ATR")
            for m, f in zip(self.multiples, fracs)
        ]
        return self._normalise(out)


@dataclass
class ChannelTargets(TargetPolicy):
    """Target the opposite side of a Donchian channel — mean-reversion style."""

    lookback: int = 20
    fraction: float = 1.0

    def build(self, side, entry, stop, bars, spec):
        ch = ind.donchian(bars, self.lookback, shift=0).iloc[-1]
        price = float(ch["upper"] if side is Side.LONG else ch["lower"])
        # Refuse targets that sit the wrong side of entry.
        if (price - entry) * side.sign <= 0:
            return []
        return [TargetSpec(spec.round_price(price), self.fraction, "channel")]


# ==========================================================================
# Config -> object factory (used by YAML strategy definitions)
# ==========================================================================
STOP_REGISTRY: dict[str, type[StopPolicy]] = {
    "fixed": FixedPointsStop,
    "percent": PercentStop,
    "atr": AtrStop,
    "structure": StructureStop,
    "chandelier": ChandelierStop,
    "ema": EmaStop,
    "breakeven": BreakEven,
    "trail_after": TrailAfter,
    "step_trail": StepTrail,
    "time": TimeStop,
}

TARGET_REGISTRY: dict[str, type[TargetPolicy]] = {
    "none": NoTarget,
    "r": RTargets,
    "atr": AtrTargets,
    "channel": ChannelTargets,
}


def build_stop(cfg: dict | list | None) -> StopPolicy:
    """`{type: atr, mult: 2}` or a list -> Composite. Nested `inner` supported."""
    if cfg is None:
        return AtrStop()
    if isinstance(cfg, list):
        return Composite([build_stop(c) for c in cfg])
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if "inner" in cfg:
        cfg["inner"] = build_stop(cfg["inner"])
    cls = STOP_REGISTRY[kind]
    return cls(**cfg)


def build_target(cfg: dict | None) -> TargetPolicy:
    if cfg is None:
        return NoTarget()
    cfg = dict(cfg)
    cls = TARGET_REGISTRY[cfg.pop("type")]
    return cls(**cfg)
