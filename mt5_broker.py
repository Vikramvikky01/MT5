"""Live broker adapter for the MetaTrader 5 terminal (Exness and others).

Requires the official package, which is Windows-only:
    pip install MetaTrader5

The terminal must be running, logged in, and have "Algo Trading" enabled.
The Python API talks to it over local IPC — you do NOT need to write an MQL5
Expert Advisor for this. (If you are on macOS/Linux, see README: run the
terminal + this bot in a Windows VM, or use the ZeroMQ bridge EA variant.)

Exness specifics handled here:
  * symbol suffixes — EURUSD may really be "EURUSDm" or "EURUSDz" depending
    on account type, so names are resolved fuzzily and cached.
  * filling modes — Exness rejects FOK on some instruments; the correct mode
    is read from the symbol's bitmask instead of guessed.
  * stops level — often 0, meaning "no fixed minimum", so we fall back to a
    spread-based floor rather than allowing a 1-point stop.
  * tick value — derived via order_calc_profit so gold, indices and crypto
    size correctly instead of being treated like a 5-digit FX pair.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..core.contracts import (
    AccountState,
    Fill,
    OrderType,
    Position,
    Side,
    Signal,
    StopSpec,
    SymbolSpec,
)
from .base import Broker

log = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:      # keeps the rest of the package importable everywhere
    mt5 = None


TIMEFRAMES: dict[str, str] = {
    "M1": "TIMEFRAME_M1", "M2": "TIMEFRAME_M2", "M3": "TIMEFRAME_M3",
    "M4": "TIMEFRAME_M4", "M5": "TIMEFRAME_M5", "M6": "TIMEFRAME_M6",
    "M10": "TIMEFRAME_M10", "M12": "TIMEFRAME_M12", "M15": "TIMEFRAME_M15",
    "M20": "TIMEFRAME_M20", "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1", "H2": "TIMEFRAME_H2", "H3": "TIMEFRAME_H3",
    "H4": "TIMEFRAME_H4", "H6": "TIMEFRAME_H6", "H8": "TIMEFRAME_H8",
    "H12": "TIMEFRAME_H12",
    "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1", "MN1": "TIMEFRAME_MN1",
}

TF_SECONDS: dict[str, int] = {
    "M1": 60, "M2": 120, "M3": 180, "M4": 240, "M5": 300, "M6": 360,
    "M10": 600, "M12": 720, "M15": 900, "M20": 1200, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600,
    "H8": 28800, "H12": 43200, "D1": 86400, "W1": 604800, "MN1": 2592000,
}

RETRYABLE = {10004, 10020, 10021, 10024}   # requote, price changed/off, too frequent


def _group_of(path: str, name: str) -> str:
    p = (path or "").lower()
    n = name.upper()
    if "crypto" in p or any(k in n for k in ("BTC", "ETH", "XRP", "SOL", "DOGE")):
        return "crypto"
    if "metal" in p or n.startswith(("XAU", "XAG", "XPT", "XPD")):
        return "metals"
    if "energ" in p or any(k in n for k in ("USOIL", "UKOIL", "XBR", "XTI", "NGAS")):
        return "energy"
    if "index" in p or "indic" in p or "cash" in p:
        return "indices"
    if "stock" in p or "share" in p or "equit" in p:
        return "stocks"
    if "forex" in p or "currenc" in p:
        return "forex"
    return "other"


class Mt5Broker(Broker):
    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        terminal_path: str | None = None,
        magic_base: int = 770000,
        deviation_points: int = 20,
        max_retries: int = 3,
        dry_run: bool = False,
    ):
        if mt5 is None:
            raise ImportError(
                "MetaTrader5 package not installed. It is Windows-only:\n"
                "  pip install MetaTrader5\n"
                "On macOS/Linux run the bot inside a Windows VM, or use the "
                "PaperBroker for research."
            )
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        self.magic_base = magic_base
        self.deviation = deviation_points
        self.max_retries = max_retries
        self.dry_run = dry_run
        self._symbol_map: dict[str, str] = {}
        self._spec_cache: dict[str, SymbolSpec] = {}
        self._tracked: dict[str, Position] = {}

    # ================================================================ setup
    def connect(self) -> None:
        kwargs: dict[str, Any] = {}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        if self.login:
            kwargs.update(login=int(self.login), password=self.password,
                          server=self.server)
        if not mt5.initialize(**kwargs):
            raise ConnectionError(f"mt5.initialize failed: {mt5.last_error()}")

        info = mt5.terminal_info()
        acc = mt5.account_info()
        if acc is None:
            raise ConnectionError(f"not logged in: {mt5.last_error()}")
        if info is not None and not info.trade_allowed and not self.dry_run:
            log.error(
                "Algo Trading is DISABLED in the terminal — orders will be "
                "rejected. Enable it: Tools > Options > Expert Advisors."
            )
        log.info(
            "connected: %s #%s (%s) balance=%.2f %s leverage=1:%s",
            acc.server, acc.login, acc.company, acc.balance, acc.currency,
            acc.leverage,
        )

    def disconnect(self) -> None:
        mt5.shutdown()

    # ========================================================= market data
    def resolve_symbol(self, wanted: str) -> str:
        """Map a canonical name to this account's actual symbol name.

        Exness serves the same instrument as EURUSD / EURUSDm / EURUSDz across
        Standard/Pro/Cent accounts, so hardcoding names breaks silently.
        """
        if wanted in self._symbol_map:
            return self._symbol_map[wanted]
        if mt5.symbol_info(wanted) is not None:
            self._symbol_map[wanted] = wanted
            return wanted

        base = wanted.upper()
        candidates = [
            s.name for s in mt5.symbols_get()
            if s.name.upper().startswith(base)
            and len(s.name) - len(base) <= 2
        ]
        if not candidates:
            raise KeyError(f"symbol '{wanted}' not found on this account")
        # Shortest match wins: 'EURUSDm' beats 'EURUSDmicro'.
        best = min(candidates, key=len)
        if best != wanted:
            log.info("resolved %s -> %s", wanted, best)
        self._symbol_map[wanted] = best
        return best

    def symbols(self, groups: tuple[str, ...] | None = None) -> list[str]:
        """Every tradable symbol, optionally filtered to instrument groups.

        This is what lets you scan "all Exness instruments" — but note the
        terminal only streams data for symbols in Market Watch, so we select
        them explicitly before use.
        """
        out = []
        for s in mt5.symbols_get():
            if s.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                continue
            g = _group_of(s.path, s.name)
            if groups and g not in groups:
                continue
            out.append(s.name)
        return sorted(out)

    def ensure_selected(self, symbol: str) -> str:
        sym = self.resolve_symbol(symbol)
        info = mt5.symbol_info(sym)
        if info is None:
            raise KeyError(sym)
        if not info.visible:
            if not mt5.symbol_select(sym, True):
                raise RuntimeError(f"symbol_select({sym}) failed: {mt5.last_error()}")
            time.sleep(0.05)   # give the terminal a moment to subscribe
        return sym

    def spec(self, symbol: str) -> SymbolSpec:
        sym = self.ensure_selected(symbol)
        i = mt5.symbol_info(sym)
        tick_size = i.trade_tick_size or i.point
        tick_value = i.trade_tick_value

        # trade_tick_value is unreliable for non-FX; derive it from the
        # terminal's own profit calculator, which knows the conversion chain.
        t = mt5.symbol_info_tick(sym)
        if t and t.ask:
            calc = mt5.order_calc_profit(
                mt5.ORDER_TYPE_BUY, sym, 1.0, t.ask, t.ask + tick_size
            )
            if calc is not None and calc > 0:
                tick_value = calc

        spec = SymbolSpec(
            name=sym,
            digits=i.digits,
            point=i.point,
            tick_size=tick_size,
            tick_value=tick_value,
            contract_size=i.trade_contract_size,
            volume_min=i.volume_min,
            volume_max=i.volume_max,
            volume_step=i.volume_step,
            stops_level_points=i.trade_stops_level,
            freeze_level_points=i.trade_freeze_level,
            spread_points=i.spread,
            filling_modes=i.filling_mode,
            group=_group_of(i.path, sym),
            profit_currency=i.currency_profit,
            swap_long=i.swap_long,
            swap_short=i.swap_short,
            tradable=i.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL,
        )
        self._spec_cache[sym] = spec
        return spec

    def bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Closed bars only. The newest candle from MT5 is still forming, so
        it is dropped — acting on it is the #1 source of backtest/live drift."""
        sym = self.ensure_selected(symbol)
        tf = getattr(mt5, TIMEFRAMES[timeframe])
        rates = mt5.copy_rates_from_pos(sym, tf, 0, count + 1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no rates for {sym} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={"real_volume": "volume"})
        return df.iloc[:-1]      # <- drop the forming bar

    def tick(self, symbol: str) -> tuple[float, float]:
        t = mt5.symbol_info_tick(self.ensure_selected(symbol))
        if t is None:
            raise RuntimeError(f"no tick for {symbol}")
        return float(t.bid), float(t.ask)

    def server_time(self) -> datetime:
        """Broker server time, taken from a tick (MT5 exposes no clock API).

        Note this is the *server's* timezone, not UTC. Exness servers are
        typically UTC+2/+3 (EET with DST), which matters for session filters.
        """
        for sym in list(self._symbol_map.values()) or ["EURUSD"]:
            t = mt5.symbol_info_tick(sym)
            if t and t.time:
                return datetime.fromtimestamp(t.time, tz=timezone.utc)
        return datetime.now(timezone.utc)

    # ============================================================== account
    def account(self) -> AccountState:
        a = mt5.account_info()
        return AccountState(
            balance=a.balance, equity=a.equity, margin=a.margin,
            free_margin=a.margin_free, currency=a.currency, leverage=a.leverage,
        )

    def magic_for(self, strategy: str) -> int:
        """Stable per-strategy magic number so we only manage our own trades
        and can attribute positions after a restart."""
        return self.magic_base + (abs(hash(strategy)) % 9000)

    def positions(self) -> list[Position]:
        raw = mt5.positions_get()
        if raw is None:
            return []
        out: list[Position] = []
        for p in raw:
            if p.magic < self.magic_base or p.magic > self.magic_base + 9000:
                continue     # not ours — leave manual trades alone
            key = str(p.ticket)
            known = self._tracked.get(key)
            pos = Position(
                id=key,
                symbol=p.symbol,
                side=Side.LONG if p.type == mt5.POSITION_TYPE_BUY else Side.SHORT,
                volume=p.volume,
                entry_price=p.price_open,
                entry_time=datetime.fromtimestamp(p.time, tz=timezone.utc),
                strategy=known.strategy if known else (p.comment or "unknown"),
                stop=StopSpec(price=p.sl or (known.stop.price if known else 0.0)),
                targets=known.targets if known else [],
                initial_volume=known.initial_volume if known else p.volume,
                initial_stop=known.initial_stop if known else (p.sl or p.price_open),
                swap=p.swap,
                bars_held=known.bars_held if known else 0,
                max_favourable=known.max_favourable if known else p.price_open,
                meta=known.meta if known else {},
            )
            self._tracked[key] = pos
            out.append(pos)
        # forget closed tickets
        live = {p.id for p in out}
        for k in list(self._tracked):
            if k not in live:
                self._tracked.pop(k, None)
        return out

    # ============================================================ execution
    def _filling_mode(self, spec: SymbolSpec) -> int:
        """Pick a filling mode the symbol actually supports.

        The symbol exposes a BITMASK (SYMBOL_FILLING_FOK=1, IOC=2) while the
        order takes an ENUM (ORDER_FILLING_FOK=0, IOC=1, RETURN=2). Confusing
        the two produces retcode 10030 'Unsupported filling mode'.
        """
        mask = spec.filling_modes
        if mask & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if mask & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _send(self, request: dict) -> Any:
        """order_send with bounded retries on transient price errors."""
        last = None
        for attempt in range(self.max_retries):
            result = mt5.order_send(request)
            if result is None:
                log.error("order_send returned None: %s", mt5.last_error())
                return None
            last = result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            if result.retcode in RETRYABLE:
                log.warning(
                    "retcode %s (%s) — retry %d/%d",
                    result.retcode, result.comment, attempt + 1, self.max_retries,
                )
                time.sleep(0.25 * (attempt + 1))
                # refresh price for market orders before retrying
                if request.get("action") == mt5.TRADE_ACTION_DEAL:
                    bid, ask = self.tick(request["symbol"])
                    request["price"] = ask if request["type"] == mt5.ORDER_TYPE_BUY else bid
                continue
            log.error("order rejected: retcode=%s %s | req=%s",
                      result.retcode, result.comment, request)
            return result
        return last

    def open(self, signal: Signal, volume: float) -> Position | None:
        spec = self.spec(signal.symbol)
        bid, ask = self.tick(spec.name)
        is_buy = signal.side is Side.LONG
        price = ask if is_buy else bid

        sl = spec.round_price(signal.stop.price)
        # Final broker-side validation: SL must clear stops_level from PRICE.
        floor = max(spec.min_stop_distance, spec.spread)
        if is_buy and sl > price - floor:
            sl = spec.round_price(price - floor)
        if not is_buy and sl < price + floor:
            sl = spec.round_price(price + floor)

        # Single TP only if there is exactly one full-size target; ladders are
        # managed by the engine via partial closes instead.
        tp = 0.0
        if len(signal.targets) == 1 and signal.targets[0].volume_fraction >= 0.999:
            tp = spec.round_price(signal.targets[0].price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": spec.name,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "magic": self.magic_for(signal.strategy),
            "comment": signal.strategy[:28],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(spec),
        }

        if self.dry_run:
            log.info("[DRY RUN] would send %s", request)
            return None

        result = self._send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return None

        ticket = str(result.order)
        # Resolve the actual position ticket (deal -> position).
        deal = mt5.history_deals_get(ticket=result.deal)
        if deal:
            ticket = str(deal[0].position_id)

        pos = Position(
            id=ticket,
            symbol=spec.name,
            side=signal.side,
            volume=result.volume,
            entry_price=result.price,
            entry_time=self.server_time(),
            strategy=signal.strategy,
            stop=StopSpec(price=sl, kind=signal.stop.kind),
            targets=list(signal.targets),
            initial_volume=result.volume,
            initial_stop=sl,
            meta=dict(signal.meta),
        )
        self._tracked[ticket] = pos
        log.info(
            "OPEN %s %s %.2f lots @ %.5f sl=%.5f (%s) risk=%.5f price-units",
            signal.side.value.upper(), spec.name, result.volume, result.price,
            sl, signal.strategy, abs(result.price - sl),
        )
        return pos

    def modify_stop(self, position: Position, stop_price: float) -> bool:
        spec = self.spec(position.symbol)
        bid, ask = self.tick(spec.name)
        ref = bid if position.side is Side.LONG else ask
        floor = max(spec.min_stop_distance, spec.spread)

        new_sl = spec.round_price(stop_price)
        if position.side is Side.LONG:
            new_sl = min(new_sl, spec.round_price(ref - floor))
            if new_sl <= position.stop.price:
                return False          # ratchet: never widen
        else:
            new_sl = max(new_sl, spec.round_price(ref + floor))
            if new_sl >= position.stop.price:
                return False

        # Freeze level: modifications are rejected too close to the market.
        if spec.freeze_level_points and abs(ref - new_sl) < spec.freeze_level_points * spec.point:
            return False

        if self.dry_run:
            log.info("[DRY RUN] would trail %s -> %.5f", position.id, new_sl)
            return False

        result = self._send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": spec.name,
            "position": int(position.id),
            "sl": new_sl,
            "tp": 0.0,
        })
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("TRAIL %s %s -> %.5f", spec.name, position.id, new_sl)
            position.stop.price = new_sl
            return True
        return False

    def close(self, position: Position, volume: float | None, reason: str) -> Fill | None:
        spec = self.spec(position.symbol)
        bid, ask = self.tick(spec.name)
        vol = spec.round_volume(volume if volume is not None else position.volume)
        vol = min(vol, position.volume)
        if vol <= 0:
            return None

        is_buy_close = position.side is Side.SHORT
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": spec.name,
            "volume": float(vol),
            "type": mt5.ORDER_TYPE_BUY if is_buy_close else mt5.ORDER_TYPE_SELL,
            "position": int(position.id),
            "price": ask if is_buy_close else bid,
            "deviation": self.deviation,
            "magic": self.magic_for(position.strategy),
            "comment": f"exit:{reason}"[:28],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(spec),
        }
        if self.dry_run:
            log.info("[DRY RUN] would close %s", request)
            return None

        result = self._send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return None

        pnl = position.side.sign * spec.money_per_lot(result.price - position.entry_price) * vol
        if (result.price - position.entry_price) * position.side.sign < 0:
            pnl = -abs(pnl)
        position.volume = spec.round_volume(position.volume - vol)
        position.realized_pnl += pnl
        log.info("CLOSE %s %.2f lots @ %.5f (%s) pnl=%.2f",
                 spec.name, vol, result.price, reason, pnl)
        return Fill(
            position_id=position.id, symbol=spec.name, side=position.side,
            volume=vol, price=result.price, time=self.server_time(),
            reason=reason, pnl=pnl,
            r_multiple=position.r_multiple(result.price),
            strategy=position.strategy,
        )
