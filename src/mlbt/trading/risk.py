"""Risk management — hard caps + kill-switch.

Three layers:
  1. Pre-trade: position-size cap, gross-exposure cap.
  2. Per-bar: max single-bar loss kill-switch.
  3. Stateful: trailing max drawdown kill-switch (halt strategy if equity
     drops > X% from session high; emit FLATTEN order).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RiskBreach:
    kind: str             # "max_dd" | "single_loss" | "gross" | "position_cap"
    detail: str
    severity: str = "halt"  # "halt" | "warn" | "trim"


class RiskManager:
    def __init__(self, cfg, state_path: Optional[str] = None):
        self.cfg = cfg
        self.state_path = state_path
        self.session_peak_equity: float = -1.0
        self.halted: bool = False
        self.halt_reason: Optional[str] = None
        self._load()

    # ----- ledger ---------------------------------------------------------
    def update(self, equity: float, last_bar_pnl: float) -> Optional[RiskBreach]:
        if equity > self.session_peak_equity:
            self.session_peak_equity = equity
        breach = None
        # Max-DD kill switch
        peak = self.session_peak_equity
        dd = (equity - peak) / max(peak, 1e-9)
        if self.cfg.enable_kill_switch and dd < self.cfg.max_drawdown_kill:
            breach = RiskBreach(kind="max_dd",
                                  detail=f"dd={dd:.2%} < {self.cfg.max_drawdown_kill:.2%}",
                                  severity="halt")
            self.halted = True
            self.halt_reason = breach.detail
        # Single-bar loss
        if self.cfg.enable_kill_switch and last_bar_pnl < 0:
            bar_pct = last_bar_pnl / max(self.session_peak_equity, 1e-9)
            if bar_pct < self.cfg.max_single_loss_pct:
                breach = RiskBreach(kind="single_loss",
                                      detail=f"bar_pnl={bar_pct:.2%} < {self.cfg.max_single_loss_pct:.2%}",
                                      severity="halt")
                self.halted = True
                self.halt_reason = breach.detail
        self._persist()
        return breach

    def check_pretrade(self, target_weights: dict[str, float],
                        gross_exposure: float) -> Optional[RiskBreach]:
        # Cap per-symbol weight
        for s, w in target_weights.items():
            if abs(w) > self.cfg.max_position_pct:
                return RiskBreach(kind="position_cap",
                                    detail=f"{s} target_w={w:.2f} > {self.cfg.max_position_pct:.2f}",
                                    severity="trim")
        # Cap gross
        gross_target = sum(abs(w) for w in target_weights.values())
        if gross_target > self.cfg.max_gross_exposure:
            return RiskBreach(kind="gross",
                                detail=f"gross={gross_target:.2f} > {self.cfg.max_gross_exposure:.2f}",
                                severity="trim")
        return None

    def trim_to_caps(self, target_weights: dict[str, float]) -> dict[str, float]:
        cap = self.cfg.max_position_pct
        trimmed = {s: max(-cap, min(cap, w)) for s, w in target_weights.items()}
        gross = sum(abs(w) for w in trimmed.values())
        if gross > self.cfg.max_gross_exposure:
            scale = self.cfg.max_gross_exposure / gross
            trimmed = {s: w * scale for s, w in trimmed.items()}
        return trimmed

    def flatten_signal(self) -> bool:
        return self.halted

    # ----- persistence ----------------------------------------------------
    def _persist(self) -> None:
        if not self.state_path:
            return
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "session_peak_equity": self.session_peak_equity,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }, indent=2))

    def _load(self) -> None:
        if not self.state_path:
            return
        p = Path(self.state_path)
        if not p.exists():
            return
        d = json.loads(p.read_text())
        self.session_peak_equity = float(d.get("session_peak_equity", -1.0))
        self.halted = bool(d.get("halted", False))
        self.halt_reason = d.get("halt_reason")
