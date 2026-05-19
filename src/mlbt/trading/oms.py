"""Order Management System.

Converts a score table (per-symbol y_score from the live signal generator)
into target weights, then into orders against the current position book.

Logic:
  1. Rank symbols by score, long top_k, short bottom_k (if not long_only)
  2. Compute target weight per symbol (= gross_leverage / k for each leg)
  3. Diff target vs current → required orders
  4. Skip orders smaller than min_order_dollars to avoid dust
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("oms")


@dataclass
class TargetWeights:
    weights: Dict[str, float]  # symbol -> target portfolio weight in [-1, 1]
    ts: pd.Timestamp

    def gross(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    def net(self) -> float:
        return sum(self.weights.values())


@dataclass
class Order:
    symbol: str
    qty: float
    side: str            # "buy" | "sell"
    target_weight: float
    ref_price: float


class OrderManagementSystem:
    def __init__(self, cfg):
        self.cfg = cfg

    def scores_to_targets(self, scores: pd.DataFrame, ts: pd.Timestamp) -> TargetWeights:
        """scores: DataFrame with 'symbol' and 'y_score' columns.

        Returns TargetWeights with portfolio-weighted positions on the
        top-k (long) and bottom-k (short, if not long_only) symbols.
        """
        scores = scores.dropna(subset=["y_score"]).sort_values("y_score", ascending=False)
        if len(scores) < self.cfg.min_universe_size:
            log.warning("universe too small: %d (min %d) — flat",
                          len(scores), self.cfg.min_universe_size)
            return TargetWeights(weights={s: 0.0 for s in scores["symbol"]}, ts=ts)
        longs = scores.head(self.cfg.top_k)
        shorts = scores.tail(self.cfg.bottom_k) if self.cfg.bottom_k > 0 else pd.DataFrame(columns=scores.columns)

        if self.cfg.market_neutral and self.cfg.bottom_k > 0:
            long_w = self.cfg.gross_leverage / 2 / max(1, len(longs))
            short_w = -self.cfg.gross_leverage / 2 / max(1, len(shorts))
        else:
            long_w = self.cfg.gross_leverage / max(1, len(longs))
            short_w = 0.0 if self.cfg.bottom_k == 0 else \
                      -self.cfg.gross_leverage / max(1, len(shorts))

        weights: Dict[str, float] = {s: 0.0 for s in scores["symbol"]}
        for s in longs["symbol"]:
            weights[s] = min(long_w, self.cfg.max_position_pct)
        for s in shorts["symbol"]:
            weights[s] = max(short_w, -self.cfg.max_position_pct)
        return TargetWeights(weights=weights, ts=ts)

    def targets_to_orders(self,
                           targets: TargetWeights,
                           current_positions: Dict[str, float],   # symbol -> qty
                           prices: Dict[str, float],
                           equity: float,
                           min_order_dollars: float = 50.0) -> List[Order]:
        """Compute the orders needed to move from current_positions to targets."""
        orders: List[Order] = []
        all_symbols = set(targets.weights) | set(current_positions)
        for symbol in all_symbols:
            target_w = targets.weights.get(symbol, 0.0)
            cur_qty = current_positions.get(symbol, 0.0)
            px = prices.get(symbol)
            if not px or px <= 0:
                continue
            target_qty = target_w * equity / px
            delta_qty = target_qty - cur_qty
            if abs(delta_qty * px) < min_order_dollars:
                continue
            side = "buy" if delta_qty > 0 else "sell"
            orders.append(Order(symbol=symbol, qty=abs(delta_qty),
                                  side=side, target_weight=target_w,
                                  ref_price=px))
        # Sort by absolute dollar size descending — biggest moves first
        orders.sort(key=lambda o: -abs(o.qty * o.ref_price))
        return orders
