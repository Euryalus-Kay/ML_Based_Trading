"""Broker adapters.

Two implementations:
  - PaperBrokerAdapter:  in-process paper broker that fills at given prices
                         (used by the runner when --paper is set and Alpaca
                         keys are missing).
  - AlpacaBrokerAdapter: hits Alpaca paper or live trading API. Requires
                         ALPACA_KEY and ALPACA_SECRET env vars (loaded from
                         .env via python-dotenv).

Both expose the same `submit_order(symbol, qty, side)`, `get_positions()`,
`get_equity()`, `get_last_price(symbol)` interface so the runner is portable.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from mlbt.core.log import get_logger
from mlbt.trading.positions import Fill, PositionTracker

log = get_logger("broker")


@dataclass
class OrderResult:
    accepted: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    reason: Optional[str] = None


class BrokerAdapter(ABC):
    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str,
                      ref_price: Optional[float] = None) -> OrderResult: ...
    @abstractmethod
    def get_positions(self) -> Dict[str, float]: ...
    @abstractmethod
    def get_equity(self) -> float: ...
    @abstractmethod
    def get_last_price(self, symbol: str) -> Optional[float]: ...
    @abstractmethod
    def is_market_open(self) -> bool: ...


class PaperBrokerAdapter(BrokerAdapter):
    """In-process paper broker. Fills immediately at the supplied ref_price
    (or at last known close if ref_price is None).
    """
    def __init__(self, tracker: PositionTracker, slippage_bps: float = 5.0):
        self.tracker = tracker
        self.slippage_bps = slippage_bps
        self._last_prices: Dict[str, float] = {}

    def update_prices(self, prices: Dict[str, float]) -> None:
        self._last_prices.update(prices)

    def submit_order(self, symbol: str, qty: float, side: str,
                      ref_price: Optional[float] = None) -> OrderResult:
        if qty <= 0:
            return OrderResult(accepted=False, reason="zero or negative qty")
        px = ref_price if ref_price is not None else self._last_prices.get(symbol)
        if px is None:
            return OrderResult(accepted=False, reason="no price")
        slip = (self.slippage_bps / 1e4) * (1 if side == "buy" else -1)
        fill_px = px * (1 + slip)
        fee = abs(qty * fill_px) * (1.0 / 1e4)   # token 1 bp fee
        self.tracker.apply_fill(Fill(ts=pd.Timestamp.utcnow().tz_convert("UTC"),
                                       symbol=symbol, side=side, qty=qty,
                                       price=fill_px, fees=fee))
        return OrderResult(accepted=True, order_id=f"paper-{len(self.tracker.fills)}",
                              fill_price=fill_px, fill_qty=qty)

    def get_positions(self) -> Dict[str, float]:
        return {s: p.qty for s, p in self.tracker.positions.items()}

    def get_equity(self) -> float:
        return self.tracker.total_equity(self._last_prices)

    def get_last_price(self, symbol: str) -> Optional[float]:
        return self._last_prices.get(symbol)

    def is_market_open(self) -> bool:
        return True   # paper broker is always "open"

    def cancel_all_open_orders(self) -> int:
        return 0   # in-process paper fills immediately; no concept of pending


class AlpacaBrokerAdapter(BrokerAdapter):
    """Alpaca paper or live trading. Honors ALPACA_KEY / ALPACA_SECRET /
    ALPACA_PAPER env vars. Uses alpaca-py if installed.
    """
    def __init__(self, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError as e:  # pragma: no cover
            raise ImportError("alpaca-py is required: pip install alpaca-py") from e
        key = os.environ.get("ALPACA_KEY")
        secret = os.environ.get("ALPACA_SECRET")
        if not key or not secret:
            raise RuntimeError("ALPACA_KEY/ALPACA_SECRET missing — set in .env")
        self.paper = paper
        self.client = TradingClient(key, secret, paper=paper)
        self.data_client = StockHistoricalDataClient(key, secret)
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._MarketOrderRequest = MarketOrderRequest

    def submit_order(self, symbol: str, qty: float, side: str,
                      ref_price: Optional[float] = None) -> OrderResult:
        try:
            req = self._MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL,
                time_in_force=self._TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            return OrderResult(accepted=True, order_id=str(order.id))
        except Exception as e:  # noqa: BLE001
            log.warning("alpaca submit %s %s %s failed: %s", side, qty, symbol, e)
            return OrderResult(accepted=False, reason=str(e))

    def get_positions(self) -> Dict[str, float]:
        try:
            pos = self.client.get_all_positions()
            return {p.symbol: float(p.qty) for p in pos}
        except Exception as e:  # noqa: BLE001
            log.warning("alpaca positions failed: %s", e)
            return {}

    def get_equity(self) -> float:
        try:
            return float(self.client.get_account().equity)
        except Exception:
            return 0.0

    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            q = self.data_client.get_stock_latest_quote(req)
            return float(q[symbol].ask_price)
        except Exception as e:  # noqa: BLE001
            log.warning("alpaca quote %s failed: %s", symbol, e)
            return None

    def is_market_open(self) -> bool:
        try:
            return bool(self.client.get_clock().is_open)
        except Exception:
            return False

    def cancel_all_open_orders(self) -> int:
        """Cancel anything still pending so the next step has a clean slate."""
        try:
            results = self.client.cancel_orders()
            return len(results) if results else 0
        except Exception as e:  # noqa: BLE001
            log.warning("alpaca cancel_orders failed: %s", e)
            return 0


def make_broker(cfg, tracker: PositionTracker) -> BrokerAdapter:
    """Pick the right broker based on env + config."""
    if cfg.broker == "paper":
        return PaperBrokerAdapter(tracker, slippage_bps=cfg.bps_slippage_estimate)
    if cfg.broker == "alpaca":
        if not os.environ.get("ALPACA_KEY"):
            log.warning("ALPACA_KEY missing — falling back to paper broker")
            return PaperBrokerAdapter(tracker, slippage_bps=cfg.bps_slippage_estimate)
        return AlpacaBrokerAdapter(paper=cfg.paper)
    raise ValueError(f"unknown broker: {cfg.broker}")
