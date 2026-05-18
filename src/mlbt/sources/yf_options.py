"""Option-chain summary metrics via yfinance.

For each ticker we snapshot the near-the-money implied vol surface and a few
aggregate metrics:
  - atm_iv: ATM IV interpolated from the nearest call & put strike
  - put_call_volume_ratio
  - put_call_oi_ratio
  - skew_25d: 25-delta put IV minus 25-delta call IV (kept simple via strike
    proxy +/- 1 stdev from spot)
  - term_30d_iv: IV of expiry closest to 30 days

These are SNAPSHOT metrics — yfinance doesn't return historical chains, so
this source produces a single row at fetch time. Use it in a live collection
loop; for backfill you need a paid options vendor and a separate adapter.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd

from mlbt.core.base import DataSource, now_utc
from mlbt.core.registry import register
from mlbt.core.log import get_logger

log = get_logger("yf_options")


def _snapshot_one(symbol: str) -> dict | None:
    import yfinance as yf
    tk = yf.Ticker(symbol)
    try:
        expiries = tk.options
    except Exception:
        return None
    if not expiries:
        return None
    try:
        spot = float(tk.fast_info.get("last_price"))
    except Exception:
        hist = tk.history(period="1d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])
    today = pd.Timestamp.utcnow().normalize()

    # Pick the expiry whose DTE is closest to 30
    expiry_dates = [pd.Timestamp(e) for e in expiries]
    dtes = [(e - today.tz_localize(None)).days for e in expiry_dates]
    if not dtes:
        return None
    near30_idx = int(np.argmin([abs(d - 30) for d in dtes]))
    expiry = expiries[near30_idx]

    try:
        chain = tk.option_chain(expiry)
    except Exception:
        return None
    calls, puts = chain.calls, chain.puts
    if calls.empty or puts.empty:
        return None

    # ATM strike — nearest to spot
    calls = calls.assign(dist=(calls["strike"] - spot).abs())
    puts = puts.assign(dist=(puts["strike"] - spot).abs())
    atm_call = calls.sort_values("dist").iloc[0]
    atm_put = puts.sort_values("dist").iloc[0]
    atm_iv = float(np.nanmean([atm_call.get("impliedVolatility"),
                                atm_put.get("impliedVolatility")]))

    # 25-delta proxy: use strikes ~1 stdev from spot using atm_iv*sqrt(T)
    T = max(dtes[near30_idx], 1) / 365.0
    sigma = atm_iv if np.isfinite(atm_iv) else 0.25
    k_up = spot * np.exp(sigma * np.sqrt(T))      # OTM call strike proxy
    k_dn = spot * np.exp(-sigma * np.sqrt(T))     # OTM put strike proxy
    iv_otm_call = float(calls.iloc[(calls["strike"] - k_up).abs().argsort().iloc[0]]["impliedVolatility"])
    iv_otm_put = float(puts.iloc[(puts["strike"] - k_dn).abs().argsort().iloc[0]]["impliedVolatility"])
    skew_25d = iv_otm_put - iv_otm_call

    pc_vol = float(puts["volume"].fillna(0).sum() / max(calls["volume"].fillna(0).sum(), 1))
    pc_oi = float(puts["openInterest"].fillna(0).sum() /
                  max(calls["openInterest"].fillna(0).sum(), 1))

    return {
        "symbol": symbol,
        "spot": spot,
        "atm_iv": atm_iv,
        "term_30d_iv": atm_iv,
        "skew_25d": skew_25d,
        "put_call_volume_ratio": pc_vol,
        "put_call_oi_ratio": pc_oi,
        "dte_used": dtes[near30_idx],
    }


@register("yf_options")
class YfOptionsSnapshot(DataSource):
    name: str = "yf_options"
    frequency: str = "snapshot"
    schema = {
        "symbol": "object", "spot": "float64", "atm_iv": "float64",
        "term_30d_iv": "float64", "skew_25d": "float64",
        "put_call_volume_ratio": "float64", "put_call_oi_ratio": "float64",
        "dte_used": "float64",
    }
    publish_lag = pd.Timedelta(seconds=30)

    def fetch(self, start, end, *, symbols: List[str] | None = None, **kw: Any) -> pd.DataFrame:
        symbols = symbols or []
        rows = []
        ts = now_utc()
        for s in symbols:
            try:
                r = _snapshot_one(s)
                if r:
                    rows.append(r)
            except Exception as e:  # noqa: BLE001
                log.warning("options snapshot %s failed: %s", s, e)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex([ts] * len(df), tz="UTC")
        return df
