"""Position tracking.

A simple, fully-deterministic accounting layer: starts with equity = $100k
(or whatever broker reports), maintains per-symbol position {qty, avg_price},
applies fills with VWAP, and computes mark-to-market P&L from the latest
prices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realised_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealised_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_price)


@dataclass
class Fill:
    ts: pd.Timestamp
    symbol: str
    side: str             # "buy" or "sell"
    qty: float
    price: float
    fees: float = 0.0


class PositionTracker:
    def __init__(self, equity: float = 100_000.0,
                 state_path: Optional[str] = None):
        self.start_equity = equity
        self.cash = equity
        self.positions: Dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.state_path = state_path

    # -------- mutation -----------------------------------------------------
    def apply_fill(self, fill: Fill) -> None:
        pos = self.positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        signed_qty = fill.qty if fill.side == "buy" else -fill.qty
        # VWAP update for adding to a position; realise PnL when reducing.
        new_qty = pos.qty + signed_qty
        if pos.qty * signed_qty >= 0:
            # adding (same sign) — update VWAP
            total_cost = pos.avg_price * abs(pos.qty) + fill.price * abs(signed_qty)
            denom = abs(new_qty) if new_qty != 0 else 1.0
            pos.avg_price = total_cost / denom
        else:
            # reducing — realise PnL on closed quantity
            closing_qty = min(abs(pos.qty), abs(signed_qty))
            sign = 1 if pos.qty > 0 else -1
            realised = sign * closing_qty * (fill.price - pos.avg_price)
            pos.realised_pnl += realised
            if new_qty * pos.qty < 0:
                # crossed zero — remaining new_qty becomes a new position
                pos.avg_price = fill.price
        pos.qty = new_qty
        self.cash -= signed_qty * fill.price + fill.fees
        self.fills.append(fill)
        # Drop zero positions
        if abs(pos.qty) < 1e-9:
            del self.positions[fill.symbol]
        self._persist()

    # -------- queries ------------------------------------------------------
    def total_equity(self, prices: Dict[str, float]) -> float:
        mv = sum(pos.market_value(prices.get(sym, pos.avg_price))
                  for sym, pos in self.positions.items())
        return self.cash + mv

    def gross_exposure(self, prices: Dict[str, float]) -> float:
        equity = max(self.total_equity(prices), 1e-9)
        gross = sum(abs(pos.market_value(prices.get(sym, pos.avg_price)))
                    for sym, pos in self.positions.items())
        return gross / equity

    def unrealised_pnl(self, prices: Dict[str, float]) -> float:
        return sum(pos.unrealised_pnl(prices.get(sym, pos.avg_price))
                   for sym, pos in self.positions.items())

    def realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self.positions.values()) + \
               sum(f.fees for f in self.fills) * -1

    def drawdown(self, prices: Dict[str, float]) -> float:
        equity = self.total_equity(prices)
        return (equity - self.start_equity) / self.start_equity

    # -------- persistence --------------------------------------------------
    def _persist(self) -> None:
        if not self.state_path:
            return
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "start_equity": self.start_equity, "cash": self.cash,
            "positions": {s: asdict(pos) for s, pos in self.positions.items()},
            "n_fills": len(self.fills),
        }
        p.write_text(json.dumps(snapshot, indent=2, default=str))

    @classmethod
    def load(cls, state_path: str, default_equity: float = 100_000.0) -> "PositionTracker":
        p = Path(state_path)
        if not p.exists():
            return cls(equity=default_equity, state_path=state_path)
        data = json.loads(p.read_text())
        tracker = cls(equity=data.get("start_equity", default_equity), state_path=state_path)
        tracker.cash = data.get("cash", default_equity)
        for sym, pdata in data.get("positions", {}).items():
            tracker.positions[sym] = Position(**{k: v for k, v in pdata.items()
                                                  if k in ("symbol", "qty", "avg_price", "realised_pnl")})
        return tracker
