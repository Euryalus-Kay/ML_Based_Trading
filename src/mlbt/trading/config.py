"""Trading configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TradingConfig:
    # Model + universe
    model_dir: str = "data/cand_xs8_v1"
    universe_path: str = "config/universe.yaml"
    bar: str = "1h"
    horizon: int = 8

    # Portfolio
    top_k: int = 10
    bottom_k: int = 0
    market_neutral: bool = False
    gross_leverage: float = 1.0
    max_position_pct: float = 0.15  # cap per name

    # Execution
    entry_lag_bars: int = 1
    bps_slippage_estimate: float = 5.0  # for pre-trade ETA only

    # Risk
    max_gross_exposure: float = 1.5
    max_drawdown_kill: float = -0.15        # halt if equity DD < this
    max_single_loss_pct: float = -0.05       # halt if single bar net P&L < this
    min_universe_size: int = 6
    enable_kill_switch: bool = True

    # Broker / mode
    paper: bool = True
    broker: str = "alpaca"                   # "alpaca" or "paper"
    alpaca_base_url: Optional[str] = None     # auto-resolved from paper flag
    alpaca_key_env: str = "ALPACA_KEY"
    alpaca_secret_env: str = "ALPACA_SECRET"

    # Inference
    use_coreml: bool = False                  # if mlpackage exists prefer it
    inference_device: str = "auto"            # "auto", "mps", "cpu", "ane"

    # Loop
    poll_seconds: int = 30                    # how often to wake up
    market_hours_only: bool = True
    log_every_bar: bool = True

    # State persistence
    state_dir: str = "data/trading_state"
