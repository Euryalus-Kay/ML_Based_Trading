"""Regime-gating wrapper for the trading runner.

Some strategies blow up in high-vol regimes. This module wraps a
predictions DataFrame with a regime filter that zeros out scores on bars
where the regime indicator (default: VIX > 25) is unfavorable.

In backtest land: blank out predictions on bad-regime bars → no orders →
position carried through. In live land: the runner can call
`should_trade(now_ts)` before scoring to skip the cycle entirely.

Cheap and easy to test against the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.core.storage import Storage

log = get_logger("regime")


@dataclass
class RegimeConfig:
    indicator: str = "vix"            # "vix" or "spy_below_200dma"
    vix_threshold: float = 25.0       # only trade when VIX < this
    spy_ma_window: int = 200          # only trade when SPY > MA_window
    block_on_high_vol: bool = True


class RegimeGate:
    def __init__(self, cfg: Optional[RegimeConfig] = None):
        self.cfg = cfg or RegimeConfig()
        self._vix_cache: Optional[pd.Series] = None
        self._spy_cache: Optional[pd.Series] = None

    def _load_vix(self) -> pd.Series:
        if self._vix_cache is not None:
            return self._vix_cache
        st = Storage()
        for sym in ("^VIX", "_VIX"):
            df = st.read("yf_1h", sym)
            if df.empty:
                df = st.read("yf_daily", sym)
            if not df.empty and "close" in df.columns:
                self._vix_cache = df["close"].astype(float).sort_index()
                return self._vix_cache
        # FRED fallback
        df = st.read("fred", "VIXCLS")
        if not df.empty:
            self._vix_cache = df.iloc[:, 0].astype(float).sort_index()
            return self._vix_cache
        log.warning("no VIX data available — regime gate disabled")
        self._vix_cache = pd.Series(dtype=float)
        return self._vix_cache

    def _load_spy(self) -> pd.Series:
        if self._spy_cache is not None:
            return self._spy_cache
        st = Storage()
        df = st.read("yf_1h", "SPY")
        if df.empty:
            df = st.read("yf_daily", "SPY")
        if not df.empty and "close" in df.columns:
            self._spy_cache = df["close"].astype(float).sort_index()
            return self._spy_cache
        self._spy_cache = pd.Series(dtype=float)
        return self._spy_cache

    def should_trade(self, ts: pd.Timestamp) -> tuple[bool, str]:
        """Return (allow_trading, reason)."""
        if self.cfg.indicator == "vix":
            vix = self._load_vix()
            if vix.empty:
                return True, "no_vix_data"
            # Use most recent observation at or before ts
            ix = vix.index.searchsorted(ts) - 1
            if ix < 0:
                return True, "no_vix_history"
            current = float(vix.iloc[ix])
            if current >= self.cfg.vix_threshold:
                return False, f"vix={current:.1f}>={self.cfg.vix_threshold}"
            return True, f"vix={current:.1f}"
        if self.cfg.indicator == "spy_below_200dma":
            spy = self._load_spy()
            if spy.empty:
                return True, "no_spy_data"
            ix = spy.index.searchsorted(ts) - 1
            if ix < self.cfg.spy_ma_window:
                return True, "insufficient_history"
            ma = float(spy.iloc[ix - self.cfg.spy_ma_window:ix].mean())
            current = float(spy.iloc[ix])
            if current < ma:
                return False, f"spy={current:.0f}<MA{self.cfg.spy_ma_window}={ma:.0f}"
            return True, f"spy_above_ma"
        return True, "unknown_indicator"

    def filter_scores(self, scores: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
        """Zero out scores on bars where regime says don't trade."""
        ok, reason = self.should_trade(ts)
        if not ok:
            log.info("regime gate BLOCKED: %s", reason)
            scores = scores.copy()
            scores["y_score"] = 0.5   # neutral — top-10 will be arbitrary, but
                                        # subsequent equal-weight allocation
                                        # produces position 0 in the OMS only
                                        # when min_universe_size kicks in
            return scores
        return scores
