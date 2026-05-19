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
    sizing_mode: str = "equal"      # "equal" | "confidence" | "conf_vol"
    vol_floor: float = 0.005         # min realised vol used when conf_vol-sizing

    # Execution
    entry_lag_bars: int = 1
    bps_slippage_estimate: float = 5.0  # for pre-trade ETA only
    rebalance_every_bars: int = 1       # 1 = every bar; h = match horizon-h backtest

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
    use_signal_cache: bool = True             # cache built dataset across cycles

    # Risk: per-position exits (tuned on xs4 1h micro — Sharpe 1.39 with these)
    stop_loss_pct: float = -0.10              # close if drawn down 10% from entry
    profit_take_pct: float = 0.30             # close if up 30% from entry
    trailing_stop_pct: float = -0.08          # close if down 8% from highest-mark-since-entry
    enable_stops: bool = True
    enable_trailing_stop: bool = True

    # Score-decay exit: backtested poorly with this model — model scores
    # cluster around 0.5 so name-by-name "lost conviction" reads as noise.
    # Disabled by default; re-enable if a model produces well-calibrated scores.
    enable_score_decay_exit: bool = False
    decay_score_threshold: float = 0.490

    # Time-in-trade exit: close if held more than max_hold_bars (regardless of model)
    enable_time_exit: bool = False
    max_hold_bars: int = 32                    # ≈ 5 trading days

    # Loop
    poll_seconds: int = 30                    # how often to wake up
    market_hours_only: bool = True
    log_every_bar: bool = True

    # State persistence
    state_dir: str = "data/trading_state"
