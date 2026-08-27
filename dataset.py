"""Feature engineering + labelling for the ML strategies.

Two rules govern everything here:

  1. FEATURES look only backwards. Every column is built from data available
     at the close of that bar. No `shift(-n)` anywhere in `make_features`.
  2. LABELS look forwards, by design — and therefore the last `max_hold` rows
     have no valid label and must be dropped before training. `make_dataset`
     does that for you.

Labelling uses the triple-barrier method (López de Prado): from each bar, walk
forward until price hits the profit barrier (+k*ATR), the loss barrier
(-k*ATR), or a time limit. Label = 1 / -1 / 0. This matches what the trading
engine actually does, which fixed-horizon return labels do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import indicators as ind

FEATURE_COLUMNS: list[str] = []   # filled by make_features on first call


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking feature matrix. Scale-free wherever possible so one
    model can generalise across symbols with different price levels."""
    df = pd.DataFrame(index=bars.index)
    c, h, l = bars["close"], bars["high"], bars["low"]
    atr14 = ind.atr(bars, 14)
    atr_n = (atr14 / c).replace([np.inf, -np.inf], np.nan)

    # --- returns at several horizons, volatility-normalised ---------------
    for n in (1, 2, 3, 5, 10, 20, 50):
        df[f"ret_{n}"] = np.log(c / c.shift(n)) / (atr_n * np.sqrt(n)).replace(0, np.nan)

    # --- trend / location -------------------------------------------------
    for n in (21, 55, 200):
        e = ind.ema(c, n)
        df[f"dist_ema_{n}"] = (c - e) / (atr14.replace(0, np.nan))
        df[f"ema_slope_{n}"] = e.pct_change(5) / atr_n.replace(0, np.nan)

    ch = ind.donchian(bars, 20, shift=1)
    span = (ch["upper"] - ch["lower"]).replace(0, np.nan)
    df["channel_pos"] = (c - ch["lower"]) / span
    df["channel_width_atr"] = span / atr14.replace(0, np.nan)

    # --- oscillators / regime --------------------------------------------
    df["rsi_14"] = ind.rsi(c, 14) / 100
    adx = ind.adx(bars, 14)
    df["adx"] = adx["adx"] / 100
    df["di_diff"] = (adx["plus_di"] - adx["minus_di"]) / 100
    df["atr_pct"] = atr_n
    df["atr_ratio"] = atr14 / atr14.rolling(100).mean()
    df["vol_zscore"] = ind.zscore(ind.realized_vol(c, 20), 100)
    bb = ind.bollinger(bars, 20, 2.0)
    df["bb_width_rank"] = bb["width"].rolling(100).rank(pct=True)

    # --- candle micro-structure ------------------------------------------
    rng = (h - l).replace(0, np.nan)
    df["body_frac"] = (c - bars["open"]) / rng
    df["upper_wick"] = (h - np.maximum(c, bars["open"])) / rng
    df["lower_wick"] = (np.minimum(c, bars["open"]) - l) / rng
    df["gap"] = (bars["open"] - c.shift(1)) / atr14.replace(0, np.nan)

    # --- calendar / session (cyclical encoding, no fake ordinality) -------
    idx = pd.DatetimeIndex(df.index)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow"] = idx.dayofweek.astype(float)

    if "tick_volume" in bars:
        v = bars["tick_volume"].astype(float)
        df["vol_ratio"] = v / v.rolling(50).mean()

    df = df.replace([np.inf, -np.inf], np.nan)
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(df.columns)
    return df


# --------------------------------------------------------------------------
@dataclass
class BarrierConfig:
    profit_atr: float = 2.0
    loss_atr: float = 1.0
    max_hold: int = 24
    atr_period: int = 14
    side: int = 0    # 0 = label both directions, +1 long-only, -1 short-only


def triple_barrier_labels(bars: pd.DataFrame, cfg: BarrierConfig) -> pd.DataFrame:
    """Vectorised-ish triple barrier. Returns label, hit bar, and realised R.

    label: +1 profit barrier first, -1 loss barrier first, 0 timed out.
    When both barriers fall inside the same bar we assume the LOSS hit first —
    the pessimistic assumption, which stops the model learning from optimism
    the live engine will never reproduce.
    """
    atr = ind.atr(bars, cfg.atr_period).to_numpy()
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    n = len(bars)

    labels = np.zeros(n)
    hit_bars = np.full(n, np.nan)
    realised_r = np.zeros(n)

    direction = 1 if cfg.side >= 0 else -1

    for i in range(n - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            labels[i] = np.nan
            continue
        entry = close[i]
        tp = entry + direction * cfg.profit_atr * a
        sl = entry - direction * cfg.loss_atr * a
        end = min(i + cfg.max_hold, n - 1)

        outcome, bars_held = 0, end - i
        for j in range(i + 1, end + 1):
            hit_sl = low[j] <= sl if direction > 0 else high[j] >= sl
            hit_tp = high[j] >= tp if direction > 0 else low[j] <= tp
            if hit_sl:                      # pessimistic: loss checked first
                outcome, bars_held = -1, j - i
                break
            if hit_tp:
                outcome, bars_held = 1, j - i
                break
        labels[i] = outcome
        hit_bars[i] = bars_held
        if outcome == 1:
            realised_r[i] = cfg.profit_atr / cfg.loss_atr
        elif outcome == -1:
            realised_r[i] = -1.0
        else:
            exit_px = close[end]
            realised_r[i] = direction * (exit_px - entry) / (cfg.loss_atr * a)

    labels[n - 1] = np.nan
    return pd.DataFrame(
        {"label": labels, "bars_held": hit_bars, "realised_r": realised_r},
        index=bars.index,
    )


def make_dataset(
    bars: pd.DataFrame,
    cfg: BarrierConfig,
    binary: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Aligned (X, y, aux). Drops warmup NaNs and the unlabelable tail."""
    X = make_features(bars)
    lab = triple_barrier_labels(bars, cfg)

    y = lab["label"]
    if binary:
        # 1 = profit barrier hit first; 0 = loss or timeout. Directly answers
        # "should I take this trade with this exact stop and target?"
        y = (lab["label"] > 0).astype(int).where(lab["label"].notna())

    valid = X.notna().all(axis=1) & y.notna()
    # Drop the tail whose barriers extend past the data.
    valid.iloc[-cfg.max_hold:] = False
    return X[valid], y[valid].astype(int), lab[valid]


def purged_walk_forward(
    index: pd.DatetimeIndex,
    n_splits: int = 5,
    embargo: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits with a purge/embargo gap.

    Standard KFold leaks badly on labelled financial data: a training row near
    the boundary has a label built from bars that sit inside the test set. We
    therefore drop `embargo` rows between train and test on both sides.
    """
    n = len(index)
    fold = n // (n_splits + 1)
    splits = []
    for k in range(1, n_splits + 1):
        train_end = fold * k
        test_start = train_end + embargo
        test_end = min(test_start + fold, n)
        if test_start >= n or test_end - test_start < 20:
            continue
        splits.append(
            (np.arange(0, train_end), np.arange(test_start, test_end))
        )
    return splits
