"""Orchestrator: pull every enabled source for the configured universe.

Reads `config/universe.yaml`, runs each `DataSource.fetch_safe` with the
appropriate args, and writes results to the on-disk raw store.

Run via:
    python -m mlbt.cli collect --start 2024-01-01 --end 2024-06-01
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from mlbt.core.log import get_logger
from mlbt.core.registry import all_sources
from mlbt.core.storage import Storage

log = get_logger("collect")


def load_universe(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else Path("config/universe.yaml")
    return yaml.safe_load(cfg_path.read_text())


def _equity_symbols(universe: dict) -> list[str]:
    eq = universe.get("equities", {})
    seen, out = set(), []
    for group in eq.values():
        for s in group:
            if s not in seen:
                seen.add(s); out.append(s)
    return out


def collect_all(start, end, universe_path: Optional[str] = None, *,
                only: Optional[list[str]] = None, storage: Optional[Storage] = None) -> dict:
    """Run all enabled sources. Returns dict {source_name: n_rows_written}."""
    universe = load_universe(universe_path)
    storage = storage or Storage()
    sources = [s for s in all_sources() if s.is_available()]
    if only:
        sources = [s for s in sources if s.name in only]

    eq = _equity_symbols(universe)
    indices = universe.get("indices", [])
    sector_etfs = universe.get("sector_etfs", [])
    futures = universe.get("futures", [])
    fx = universe.get("fx", [])
    crypto = universe.get("crypto_binance", [])
    fred_series = universe.get("fred_series", [])
    wiki_pages = universe.get("wikipedia_pages", [])

    yf_symbols = sorted(set(eq + indices + sector_etfs + futures + fx))
    bar = universe.get("bar", "5min")
    interval_map = {"1min": "1m", "5min": "5m", "15min": "15m", "1h": "1h", "1d": "1d"}
    yf_interval = interval_map.get(bar, "5m")

    rows_summary: dict[str, int] = {}

    for src in sources:
        log.info("=== %s ===", src.name)
        kwargs: dict[str, Any] = {}
        if src.name == "yf_intraday":
            kwargs["symbols"] = yf_symbols
            kwargs["interval"] = yf_interval
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for sym, sub in df.groupby("symbol"):
                    storage.write(src.name, sym, sub.drop(columns=["symbol"]))
                rows_summary[src.name] = len(df)

        elif src.name == "yf_daily":
            kwargs["symbols"] = yf_symbols
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for sym, sub in df.groupby("symbol"):
                    storage.write(src.name, sym, sub.drop(columns=["symbol"]))
                rows_summary[src.name] = len(df)

        elif src.name == "yf_options":
            kwargs["symbols"] = eq
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for sym, sub in df.groupby("symbol"):
                    storage.write(src.name, sym, sub.drop(columns=["symbol"]))
                rows_summary[src.name] = len(df)

        elif src.name == "binance_klines":
            kwargs["symbols"] = crypto
            kwargs["interval"] = bar
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for sym, sub in df.groupby("symbol"):
                    storage.write(src.name, sym, sub.drop(columns=["symbol"]))
                rows_summary[src.name] = len(df)

        elif src.name == "fred":
            kwargs["series"] = fred_series
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for col in df.columns:
                    storage.write(src.name, col, df[[col]].dropna())
                rows_summary[src.name] = len(df)

        elif src.name == "wiki_pageviews":
            kwargs["pages"] = wiki_pages
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for col in df.columns:
                    storage.write(src.name, col, df[[col]].dropna())
                rows_summary[src.name] = len(df)

        elif src.name == "sec_edgar":
            kwargs["symbols"] = eq
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for sym, sub in df.groupby("symbol"):
                    storage.write(src.name, sym, sub.drop(columns=["symbol"]))
                rows_summary[src.name] = len(df)

        elif src.name == "finra_short":
            kwargs["symbols"] = eq
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                for sym, sub in df.groupby("symbol"):
                    storage.write(src.name, sym, sub.drop(columns=["symbol"]))
                rows_summary[src.name] = len(df)

        else:
            # Default: source emits a single wide df with no symbol axis
            df = src.fetch_safe(start, end, **kwargs)
            if not df.empty:
                storage.write(src.name, "_default", df)
                rows_summary[src.name] = len(df)

    log.info("collect summary: %s", rows_summary)
    return rows_summary
