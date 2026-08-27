"""Live/paper trading loop against a Broker implementation.

Design notes that matter in production:

  * The loop is BAR-DRIVEN, not tick-driven. It wakes up, checks whether a new
    candle has closed for each (symbol, timeframe), and only then evaluates
    strategies. This makes live behaviour identical to the backtest.
  * Position management (trailing, partial targets) runs on every new bar too,
    plus an optional faster cadence for stop trailing only.
  * State is persisted to JSON so a restart re-adopts open trades instead of
    orphaning them or double-entering.
  * Every unhandled exception is caught per-symbol: one bad instrument must
    never take the whole bot down while positions are open.
"""

from __future__ import annotations

import json
import logging
import signal as os_signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..broker.base import Broker
from ..broker.mt5_broker import TF_SECONDS
from ..core.contracts import ExitReason, MarketSnapshot, Position, Side
from ..core.risk import RiskLimits, RiskManager, Sizer
from ..strategy.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    poll_seconds: float = 5.0
    manage_every_poll: bool = True     # trail stops between bar closes
    state_path: str = "state/live_state.json"
    heartbeat_minutes: int = 30
    close_on_shutdown: bool = False    # usually False: let stops do their job
    max_consecutive_errors: int = 20


class LiveEngine:
    def __init__(
        self,
        broker: Broker,
        strategies: list[Strategy],
        sizer: Sizer,
        limits: RiskLimits,
        universe: list[str],
        config: LiveConfig | None = None,
    ):
        self.broker = broker
        self.strategies = strategies
        self.sizer = sizer
        self.risk = RiskManager(limits, sizer)
        self.universe = universe
        self.cfg = config or LiveConfig()
        self._last_bar: dict[tuple[str, str], pd.Timestamp] = {}
        self._running = False
        self._errors = 0
        self._last_heartbeat = 0.0
        self._meta: dict[str, dict] = {}   # position id -> our extra fields

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        self._install_signal_handlers()
        for s in self.strategies:
            s.on_start()
        self._load_state()
        self._running = True
        log.info(
            "live engine started | %d strategies | %d symbols | risk %.2f%%/trade",
            len(self.strategies), len(self.universe), self.sizer.risk_per_trade * 100,
        )

        while self._running:
            try:
                self._iteration()
                self._errors = 0
            except KeyboardInterrupt:
                break
            except Exception:
                self._errors += 1
                log.exception("iteration failed (%d consecutive)", self._errors)
                if self._errors >= self.cfg.max_consecutive_errors:
                    log.critical("too many consecutive errors — shutting down")
                    break
            time.sleep(self.cfg.poll_seconds)

        self._shutdown()

    def _iteration(self) -> None:
        now = self.broker.server_time()
        account = self.broker.account()
        self.risk.on_equity(account, now)
        positions = self.broker.positions()
        self._restore_meta(positions)
        self._heartbeat(account, positions)

        # ---- 1. manage what is already open ---------------------------
        for pos in positions:
            try:
                self._manage_position(pos, now)
            except Exception:
                log.exception("manage failed for %s %s", pos.symbol, pos.id)

        if self.risk.halted_reason:
            return   # no new risk today

        # ---- 2. new bars -> new signals -------------------------------
        for strat in self.strategies:
            symbols = strat.symbols or self.universe
            for sym in symbols:
                try:
                    if not self._new_bar(sym, strat.timeframe):
                        continue
                    self._evaluate(strat, sym, now)
                except Exception:
                    log.exception("evaluate failed: %s on %s", strat.instance_name, sym)

    # ------------------------------------------------------- bar detection
    def _new_bar(self, symbol: str, timeframe: str) -> bool:
        """True exactly once per closed candle per (symbol, timeframe)."""
        bars = self.broker.bars(symbol, timeframe, 3)
        if bars.empty:
            return False
        latest = bars.index[-1]
        key = (symbol, timeframe)
        if self._last_bar.get(key) == latest:
            return False
        first_time = key not in self._last_bar
        self._last_bar[key] = latest
        if first_time:
            # Do not fire on the very first observation: that candle closed
            # before we started and acting on it is stale-signal trading.
            return False
        return True

    # ------------------------------------------------------------ decisions
    def _evaluate(self, strat: Strategy, symbol: str, now: datetime) -> None:
        snap = self._snapshot(symbol, strat, now)
        if snap is None:
            return
        try:
            signal = strat.entry(snap)
        except ValueError as e:
            log.debug("signal rejected by strategy validation: %s", e)
            return
        if signal is None:
            return

        specs = {p.symbol: self.broker.spec(p.symbol) for p in snap.open_positions}
        specs[symbol] = snap.spec
        feats = strat.features(snap)
        atr_val = float(feats["atr"].iloc[-1]) if "atr" in feats else None

        verdict = self.risk.check(
            signal, snap.spec, snap.account, now, snap.open_positions, specs, atr_val
        )
        if not verdict:
            log.info("SKIP %s %s: %s", symbol, signal.side.value, verdict.reason)
            return

        sizing = self.sizer.size(
            snap.spec, snap.account, signal.entry_price, signal.stop.price,
            signal.risk_multiplier,
        )
        if not sizing.ok:
            log.info("SKIP %s: %s", symbol, sizing.rejected)
            return

        log.info(
            "SIGNAL %s %s %s conf=%.2f | %.2f lots | risk %.2f %s (%.2f%%)",
            signal.strategy, symbol, signal.side.value, signal.confidence,
            sizing.volume, sizing.risk_money, snap.account.currency,
            100 * sizing.risk_money / max(snap.account.equity, 1e-9),
        )
        pos = self.broker.open(signal, sizing.volume)
        if pos is not None:
            self._meta[pos.id] = {
                "targets": [t.__dict__ for t in pos.targets],
                "initial_stop": pos.initial_stop,
                "initial_volume": pos.initial_volume,
                "strategy": pos.strategy,
                "max_bars": signal.max_bars_in_trade,
            }
            strat.on_position_opened(pos)
            self._save_state()

    def _manage_position(self, pos: Position, now: datetime) -> None:
        strat = next(
            (s for s in self.strategies if s.instance_name == pos.strategy), None
        )
        if strat is None:
            log.warning("position %s has no matching strategy '%s' — leaving alone",
                        pos.id, pos.strategy)
            return

        bars = self.broker.bars(pos.symbol, strat.timeframe, strat.warmup)
        if len(bars) < 20:
            return
        spec = self.broker.spec(pos.symbol)
        bid, ask = self.broker.tick(pos.symbol)
        price = bid if pos.side is Side.LONG else ask

        # -- partial targets (MT5 has one TP per position, so we manage the
        #    ladder ourselves with partial closes) -------------------------
        for tgt in pos.targets:
            if tgt.filled:
                continue
            hit = price >= tgt.price if pos.side is Side.LONG else price <= tgt.price
            if not hit:
                continue
            vol = spec.round_volume(pos.initial_volume * tgt.volume_fraction)
            if pos.volume - vol < spec.volume_min:
                vol = pos.volume
            fill = self.broker.close(pos, vol, f"{ExitReason.TARGET.value}:{tgt.label}")
            if fill:
                tgt.filled = True
                self._meta.setdefault(pos.id, {})["targets"] = [
                    t.__dict__ for t in pos.targets
                ]
                self._save_state()
            if pos.volume <= 1e-9:
                return

        # -- dynamic stop ---------------------------------------------------
        new_stop = strat.stop_policy.update(pos, bars, spec)
        if new_stop is not None:
            self.broker.modify_stop(pos, new_stop)

        # -- time stop ------------------------------------------------------
        max_bars = (self._meta.get(pos.id) or {}).get("max_bars")
        if max_bars:
            tf_sec = TF_SECONDS.get(strat.timeframe, 3600)
            held = (now - pos.entry_time).total_seconds() / tf_sec
            if held >= max_bars:
                self.broker.close(pos, None, ExitReason.TIME.value)
                return

        # -- discretionary exit ---------------------------------------------
        snap = self._snapshot(pos.symbol, strat, now)
        if snap is not None and strat.exit_signal(pos, snap):
            log.info("strategy exit for %s %s", pos.symbol, pos.id)
            self.broker.close(pos, None, ExitReason.SIGNAL.value)

    # -------------------------------------------------------------- helpers
    def _snapshot(self, symbol: str, strat: Strategy, now: datetime) -> MarketSnapshot | None:
        bars = self.broker.bars(symbol, strat.timeframe, strat.warmup)
        if len(bars) < strat.warmup:
            log.debug("%s: only %d/%d warmup bars", symbol, len(bars), strat.warmup)
            return None
        bid, ask = self.broker.tick(symbol)
        return MarketSnapshot(
            symbol=symbol, timeframe=strat.timeframe, bars=bars,
            spec=self.broker.spec(symbol), server_time=now, bid=bid, ask=ask,
            account=self.broker.account(), open_positions=self.broker.positions(),
        )

    def _restore_meta(self, positions: list[Position]) -> None:
        """Re-attach target ladders / initial stops after a restart."""
        from ..core.contracts import TargetSpec

        for p in positions:
            m = self._meta.get(p.id)
            if not m:
                continue
            if not p.targets and m.get("targets"):
                p.targets = [TargetSpec(**t) for t in m["targets"]]
            p.initial_stop = m.get("initial_stop", p.initial_stop)
            p.initial_volume = m.get("initial_volume", p.initial_volume)
            p.strategy = m.get("strategy", p.strategy)

    def _heartbeat(self, account, positions) -> None:
        if time.time() - self._last_heartbeat < self.cfg.heartbeat_minutes * 60:
            return
        self._last_heartbeat = time.time()
        open_risk = self.risk.open_risk_fraction(
            positions, {p.symbol: self.broker.spec(p.symbol) for p in positions},
            account.equity,
        )
        log.info(
            "heartbeat | equity %.2f %s | %d open | open risk %.2f%% | %s",
            account.equity, account.currency, len(positions), open_risk * 100,
            self.risk.halted_reason or "trading",
        )

    # ---------------------------------------------------------------- state
    def _save_state(self) -> None:
        path = Path(self.cfg.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "meta": self._meta,
            "last_bar": {f"{k[0]}|{k[1]}": str(v) for k, v in self._last_bar.items()},
        }, indent=2, default=str))

    def _load_state(self) -> None:
        path = Path(self.cfg.state_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._meta = data.get("meta", {})
            for k, v in data.get("last_bar", {}).items():
                sym, tf = k.split("|")
                self._last_bar[(sym, tf)] = pd.Timestamp(v)
            log.info("restored state: %d tracked positions", len(self._meta))
        except Exception:
            log.exception("could not load state — starting fresh")

    # ------------------------------------------------------------- shutdown
    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            log.warning("signal %s received — finishing current loop", signum)
            self._running = False

        for s in (os_signal.SIGINT, os_signal.SIGTERM):
            try:
                os_signal.signal(s, handler)
            except (ValueError, AttributeError):
                pass    # not in main thread / unsupported platform

    def _shutdown(self) -> None:
        self._save_state()
        if self.cfg.close_on_shutdown:
            for pos in self.broker.positions():
                self.broker.close(pos, None, "shutdown")
        log.info("engine stopped. open positions left with broker-side stops in place.")
        self.broker.disconnect()
