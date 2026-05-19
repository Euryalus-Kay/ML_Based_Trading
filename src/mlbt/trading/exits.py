"""Per-position exit tracker.

For each open position we maintain:
  - entry_ts, entry_price (set when the position opens or grows)
  - high_mark (max mark-to-market price seen since entry, for trailing stop)
  - last_score (most recent ML score, for score-decay exit)
  - bars_held (incremented each step the position is held)

ExitEvaluator.check(symbol, qty, current_price, current_score) returns
ExitReason | None.  The runner uses this to override the OMS targets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass
class TrackedPosition:
    symbol: str
    entry_ts: str
    entry_price: float
    high_mark: float
    bars_held: int = 0
    last_score: float = 0.5


@dataclass
class ExitDecision:
    reason: str             # "stop_loss" | "profit_take" | "trailing_stop" | "score_decay" | "time_exit"
    detail: str
    pl_pct: float


class ExitTracker:
    """Stateful per-symbol bookkeeping for exit rules."""

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path
        self.positions: Dict[str, TrackedPosition] = {}
        self._load()

    # ----- mutation -----------------------------------------------------
    def update_on_step(self, current_positions: Dict[str, float],
                         current_prices: Dict[str, float],
                         scores_by_symbol: Dict[str, float],
                         ts: pd.Timestamp,
                         entry_prices: Dict[str, float]) -> None:
        """Refresh trackers for currently-held positions.

        - if a position is new (we held none last step), record entry
        - if held, update high_mark + bars_held + last_score
        - drop trackers for positions that no longer exist
        """
        active = set()
        for sym, qty in current_positions.items():
            if qty == 0:
                continue
            active.add(sym)
            px = current_prices.get(sym)
            if px is None or px <= 0:
                continue
            entry = entry_prices.get(sym, px)
            if sym not in self.positions:
                self.positions[sym] = TrackedPosition(
                    symbol=sym, entry_ts=str(ts), entry_price=float(entry),
                    high_mark=float(px), bars_held=0,
                    last_score=float(scores_by_symbol.get(sym, 0.5)),
                )
            else:
                tp = self.positions[sym]
                if float(px) > tp.high_mark:
                    tp.high_mark = float(px)
                tp.bars_held += 1
                tp.last_score = float(scores_by_symbol.get(sym, tp.last_score))
        # Drop closed positions
        for sym in list(self.positions):
            if sym not in active:
                self.positions.pop(sym)
        self._persist()

    # ----- evaluation ---------------------------------------------------
    def check(self, symbol: str, current_price: float, cfg) -> Optional[ExitDecision]:
        tp = self.positions.get(symbol)
        if tp is None or current_price <= 0:
            return None

        # 1. Stop loss (from entry)
        pl_from_entry = (current_price - tp.entry_price) / max(tp.entry_price, 1e-9)
        if cfg.enable_stops and pl_from_entry <= cfg.stop_loss_pct:
            return ExitDecision("stop_loss",
                                  f"P&L {pl_from_entry:+.2%} from entry "
                                  f"${tp.entry_price:.2f} → ${current_price:.2f}",
                                  pl_from_entry)
        # 2. Profit take (from entry)
        if cfg.enable_stops and pl_from_entry >= cfg.profit_take_pct:
            return ExitDecision("profit_take",
                                  f"P&L {pl_from_entry:+.2%} from entry "
                                  f"${tp.entry_price:.2f} → ${current_price:.2f}",
                                  pl_from_entry)
        # 3. Trailing stop (from high mark)
        if cfg.enable_trailing_stop:
            pl_from_high = (current_price - tp.high_mark) / max(tp.high_mark, 1e-9)
            if pl_from_high <= cfg.trailing_stop_pct:
                return ExitDecision("trailing_stop",
                                      f"P&L {pl_from_high:+.2%} from peak "
                                      f"${tp.high_mark:.2f} → ${current_price:.2f}",
                                      pl_from_high)
        # 4. Score decay (model lost conviction)
        if cfg.enable_score_decay_exit and tp.last_score < cfg.decay_score_threshold:
            return ExitDecision("score_decay",
                                  f"score {tp.last_score:.4f} < {cfg.decay_score_threshold}",
                                  pl_from_entry)
        # 5. Time in trade
        if cfg.enable_time_exit and tp.bars_held >= cfg.max_hold_bars:
            return ExitDecision("time_exit",
                                  f"held {tp.bars_held} bars >= max {cfg.max_hold_bars}",
                                  pl_from_entry)
        return None

    # ----- persistence --------------------------------------------------
    def _persist(self) -> None:
        if not self.state_path:
            return
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {s: asdict(tp) for s, tp in self.positions.items()},
            indent=2, default=str,
        ))

    def _load(self) -> None:
        if not self.state_path:
            return
        p = Path(self.state_path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            for s, d in data.items():
                self.positions[s] = TrackedPosition(**d)
        except Exception:
            pass
