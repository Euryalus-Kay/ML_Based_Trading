"""Live trading system: signal → OMS → positions → risk → broker.

End-to-end pipeline that turns a trained model on disk into actual orders.
Designed for paper-trading first (Alpaca paper); promote to live by flipping
ALPACA_PAPER=false in .env.

  signal.LiveSignalGenerator   — load model + score universe at each bar
  oms.OrderManagementSystem    — convert scores to target weights, then orders
  positions.PositionTracker    — track open positions and realised P&L
  risk.RiskManager             — hard caps, kill-switch, vol-targeted sizing
  broker.AlpacaBrokerAdapter   — order submission + position query
  runner.TradingRunner         — the main loop tying it all together

CLI:
  python -m mlbt.trading.runner --model-dir data/cand_xs8_v1 --bar 1h --paper
"""
from mlbt.trading.config import TradingConfig
from mlbt.trading.signal import LiveSignalGenerator
from mlbt.trading.oms import OrderManagementSystem, TargetWeights
from mlbt.trading.positions import PositionTracker, Position
from mlbt.trading.risk import RiskManager, RiskBreach
from mlbt.trading.broker import BrokerAdapter, PaperBrokerAdapter

__all__ = [
    "TradingConfig",
    "LiveSignalGenerator",
    "OrderManagementSystem", "TargetWeights",
    "PositionTracker", "Position",
    "RiskManager", "RiskBreach",
    "BrokerAdapter", "PaperBrokerAdapter",
]
