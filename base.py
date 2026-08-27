"""Strategy plug-in contract.

To add a strategy you write ONE class and decorate it:

    @register("my_edge")
    class MyEdge(Strategy):
        params = {"fast": 20, "slow": 50}

        def compute(self, bars):                 # vectorised, whole history
            bars["fast"] = ind.ema(bars.close, self.p["fast"])
            bars["long_entry"]  = bars.fast > bars.slow
            bars["short_entry"] = bars.fast < bars.slow
            return bars

        # `entry()` has a default implementation that reads the flag columns,
        # so most strategies only need compute().

Why the split?
  * `compute()` runs once over the full DataFrame -> backtests stay fast.
  * The engine then evaluates only the LAST row -> identical logic live.
  There is no second code path to keep in sync, which is where most retail
  bots quietly diverge between backtest and production.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import pandas as pd

from ..core.contracts import (
    MarketSnapshot,
    OrderType,
    Position,
    Side,
    Signal,
    StopSpec,
)
from ..core.exits import (
    Composite,
    NoTarget,
    StopPolicy,
    TargetPolicy,
    TimeStop,
    build_stop,
    build_target,
)

log = logging.getLogger(__name__)

REGISTRY: dict[str, type["Strategy"]] = {}


def register(name: str) -> Callable[[type["Strategy"]], type["Strategy"]]:
    def deco(cls: type["Strategy"]) -> type["Strategy"]:
        if name in REGISTRY:
            raise ValueError(f"strategy '{name}' already registered")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


class Strategy(ABC):
    """Base class. Subclasses override `compute` and optionally `entry`."""

    name: str = "unnamed"
    params: dict[str, Any] = {}          # defaults, overridden by config
    warmup: int = 200                    # bars needed before signals are valid
    timeframe: str = "H1"
    long_only: bool = False
    short_only: bool = False

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        stop: StopPolicy | None = None,
        target: TargetPolicy | None = None,
        symbols: list[str] | None = None,
        timeframe: str | None = None,
        risk_multiplier: float = 1.0,
        instance_name: str | None = None,
    ):
        self.p = {**self.params, **(params or {})}
        self.stop_policy = stop or build_stop(None)
        self.target_policy = target or NoTarget()
        self.symbols = symbols or []
        self.timeframe = timeframe or self.timeframe
        self.risk_multiplier = risk_multiplier
        # instance_name lets you run the same class twice with different params
        self.instance_name = instance_name or self.name
        self._cache: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}

    def __repr__(self) -> str:
        return f"<{self.instance_name} {self.timeframe} {self.p}>"

    # ---------------------------------------------------------------- hooks
    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Add indicator + signal columns. MUST be causal (no negative shifts).

        Conventional output columns:
          long_entry / short_entry : bool
          long_exit  / short_exit  : bool  (optional discretionary exit)
          confidence               : float 0..1 (optional)
        """

    def entry(self, snap: MarketSnapshot) -> Optional[Signal]:
        """Default: translate the flag columns on the last closed bar."""
        feats = self.features(snap)
        row = feats.iloc[-1]

        side: Side | None = None
        if bool(row.get("long_entry", False)) and not self.short_only:
            side = Side.LONG
        elif bool(row.get("short_entry", False)) and not self.long_only:
            side = Side.SHORT
        if side is None:
            return None

        return self.make_signal(snap, side, confidence=float(row.get("confidence", 1.0)))

    def exit_signal(self, pos: Position, snap: MarketSnapshot) -> bool:
        """Optional discretionary exit, e.g. trend flipped."""
        row = self.features(snap).iloc[-1]
        col = "long_exit" if pos.side is Side.LONG else "short_exit"
        return bool(row.get(col, False))

    # ------------------------------------------------------------- plumbing
    def features(self, snap: MarketSnapshot) -> pd.DataFrame:
        """Cached `compute` per symbol; recomputed when a new bar arrives.

        The cache key is the LAST BAR'S TIMESTAMP, not the row count. A rolling
        warmup window is always exactly `warmup` rows long, so keying on length
        would make the cache never invalidate — the strategy would freeze on
        its first computed signal for the rest of its life.
        """
        key = snap.symbol
        stamp = snap.bars.index[-1]
        cached = self._cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]
        feats = self.compute(snap.bars.copy())
        self._cache[key] = (stamp, feats)
        return feats

    def make_signal(
        self,
        snap: MarketSnapshot,
        side: Side,
        *,
        confidence: float = 1.0,
        order_type: OrderType = OrderType.MARKET,
        entry_price: float | None = None,
        meta: dict | None = None,
    ) -> Signal:
        """Attach stop + targets from the configured policies. Use this rather
        than constructing Signal by hand so exits stay consistent."""
        entry = entry_price if entry_price is not None else snap.entry_price_for(side)
        stop_price = self.stop_policy.initial(side, entry, snap.bars, snap.spec)

        # Sanity: the stop must be on the losing side of entry.
        if (entry - stop_price) * side.sign <= 0:
            log.warning(
                "%s produced an invalid stop (entry=%s stop=%s side=%s) — skipping",
                self.instance_name, entry, stop_price, side,
            )
            raise ValueError("stop on wrong side of entry")

        targets = self.target_policy.build(side, entry, stop_price, snap.bars, snap.spec)
        max_bars = getattr(self.stop_policy, "max_bars", None)

        return Signal(
            symbol=snap.symbol,
            side=side,
            strategy=self.instance_name,
            order_type=order_type,
            entry_price=entry,
            stop=StopSpec(price=stop_price, kind=self.stop_policy.kind),
            targets=targets,
            confidence=confidence,
            risk_multiplier=self.risk_multiplier * max(0.0, min(1.5, confidence)),
            max_bars_in_trade=max_bars,
            meta=meta or {},
        )

    # ---- optional lifecycle hooks ---------------------------------------
    def on_start(self) -> None:
        """Load models, warm caches."""

    def on_position_opened(self, pos: Position) -> None:
        pass

    def on_position_closed(self, pos: Position, reason: str) -> None:
        pass


# --------------------------------------------------------------------------
def build_strategy(cfg: dict[str, Any]) -> Strategy:
    """Instantiate from a config dict (see configs/config.yaml).

    {name: ema_atr_trend, instance: trend_fx, timeframe: H1,
     symbols: [EURUSDm], params: {...},
     stop: [{type: atr, mult: 2}, {type: breakeven, trigger_r: 1}],
     target: {type: r, multiples: [1, 2], fractions: [0.5, 0.5]}}
    """
    import quantbot.strategy.library  # noqa: F401  (populates REGISTRY)

    name = cfg["name"]
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. Known: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    return cls(
        params=cfg.get("params"),
        stop=build_stop(cfg.get("stop")),
        target=build_target(cfg.get("target")),
        symbols=cfg.get("symbols"),
        timeframe=cfg.get("timeframe"),
        risk_multiplier=cfg.get("risk_multiplier", 1.0),
        instance_name=cfg.get("instance"),
    )
