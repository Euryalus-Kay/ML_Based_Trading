"""Main trading-loop runner.

Wakes up every `poll_seconds`, checks market hours, pulls the latest data,
runs the model to produce per-symbol scores, converts to target weights,
diffs against the position book, sends orders via the configured broker,
and persists state so we can resume cleanly.

Usage (paper):
    python -m mlbt.trading.runner --model-dir data/cand_xs8_v1 --paper

Usage (live, ALPACA_KEY/SECRET in .env):
    python -m mlbt.trading.runner --model-dir data/cand_xs8_v1 --live
"""
from __future__ import annotations

import argparse
import json
import os
import signal as posix_signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from mlbt.core.log import get_logger
from mlbt.trading.broker import make_broker, PaperBrokerAdapter
from mlbt.trading.config import TradingConfig
from mlbt.trading.oms import OrderManagementSystem
from mlbt.trading.positions import PositionTracker
from mlbt.trading.risk import RiskManager
from mlbt.trading.signal import LiveSignalGenerator

log = get_logger("runner")


class TradingRunner:
    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg
        self.state_dir = Path(cfg.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        load_dotenv()

        self.tracker = PositionTracker.load(
            state_path=str(self.state_dir / "positions.json"),
            default_equity=float(os.environ.get("MLBT_START_EQUITY", 100_000.0)),
        )
        self.broker = make_broker(cfg, self.tracker)
        self.risk = RiskManager(cfg, state_path=str(self.state_dir / "risk.json"))
        self.signal = LiveSignalGenerator(
            model_dir=cfg.model_dir, bar=cfg.bar,
            universe_path=cfg.universe_path,
            use_coreml=cfg.use_coreml,
            inference_device=cfg.inference_device,
        )
        self.oms = OrderManagementSystem(cfg)
        self._last_equity = self.broker.get_equity() if not isinstance(self.broker, PaperBrokerAdapter) else self.tracker.start_equity
        self._stop = False
        posix_signal.signal(posix_signal.SIGINT, lambda *_: self._request_stop("SIGINT"))
        posix_signal.signal(posix_signal.SIGTERM, lambda *_: self._request_stop("SIGTERM"))

    def _request_stop(self, why: str) -> None:
        log.warning("stop requested: %s", why)
        self._stop = True

    # ----- one cycle ------------------------------------------------------
    def step(self) -> dict:
        ts = pd.Timestamp.utcnow().tz_convert("UTC")
        # 1. Score
        scores = self.signal.score_now(ts)
        if scores.empty:
            return {"ts": str(ts), "skipped": "no scores"}

        # 2. Target weights
        targets = self.oms.scores_to_targets(scores, ts)

        # 3. Pre-trade risk check
        breach = self.risk.check_pretrade(targets.weights,
                                            self.tracker.gross_exposure({}))
        if breach and breach.severity == "halt":
            log.warning("PRE-TRADE HALT: %s", breach.detail)
            return {"ts": str(ts), "halt": breach.detail}
        if breach and breach.severity == "trim":
            targets.weights = self.risk.trim_to_caps(targets.weights)

        # 4. Translate to orders
        prices = {s: self.broker.get_last_price(s) or 0.0 for s in targets.weights}
        if isinstance(self.broker, PaperBrokerAdapter):
            # Paper broker uses last prices we feed it from the scoring step
            self.broker.update_prices(prices)
        equity = self.broker.get_equity() or self.tracker.total_equity(prices)
        current_positions = self.broker.get_positions()
        orders = self.oms.targets_to_orders(targets, current_positions,
                                              prices, equity,
                                              min_order_dollars=50.0)

        # 5. Optional kill-switch
        bar_pnl = equity - self._last_equity
        breach2 = self.risk.update(equity, bar_pnl)
        if breach2 and breach2.severity == "halt":
            log.warning("RISK HALT: %s — flattening", breach2.detail)
            orders = self._flatten_orders(current_positions, prices)

        # 6. Submit orders
        results = []
        for o in orders:
            if self.risk.flatten_signal() and o.side == "buy":
                continue  # don't add risk during a halt
            res = self.broker.submit_order(o.symbol, o.qty, o.side, ref_price=o.ref_price)
            results.append({"symbol": o.symbol, "side": o.side, "qty": o.qty,
                              "accepted": res.accepted, "reason": res.reason})

        self._last_equity = equity
        report = {
            "ts": str(ts), "equity": equity, "bar_pnl": bar_pnl,
            "n_scores": len(scores), "n_orders": len(orders),
            "gross_exposure": targets.gross(),
            "halted": self.risk.halted,
        }
        if self.cfg.log_every_bar:
            log.info("step: eq=$%.0f pnl=$%.2f orders=%d gross=%.2f halted=%s",
                     equity, bar_pnl, len(orders), targets.gross(), self.risk.halted)
        # Persist last cycle report
        (self.state_dir / "last_cycle.json").write_text(
            json.dumps({"report": report, "orders": results}, indent=2))
        return report

    def _flatten_orders(self, positions, prices):
        from mlbt.trading.oms import Order
        orders = []
        for s, q in positions.items():
            if q == 0:
                continue
            side = "sell" if q > 0 else "buy"
            orders.append(Order(symbol=s, qty=abs(q), side=side,
                                  target_weight=0.0, ref_price=prices.get(s, 0.0)))
        return orders

    # ----- main loop ------------------------------------------------------
    def run(self) -> None:
        log.info("trading runner starting (paper=%s, broker=%s, model=%s)",
                 self.cfg.paper, self.cfg.broker, self.cfg.model_dir)
        while not self._stop:
            try:
                if self.cfg.market_hours_only and not self.broker.is_market_open():
                    log.info("market closed; sleeping")
                else:
                    self.step()
            except Exception as e:  # noqa: BLE001
                log.exception("step failed: %s", e)
            for _ in range(self.cfg.poll_seconds):
                if self._stop:
                    break
                time.sleep(1)
        log.info("runner stopped")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="data/cand_xs8_v1")
    p.add_argument("--bar", default="1h")
    p.add_argument("--universe", default="config/universe.yaml")
    p.add_argument("--paper", action="store_true", default=True)
    p.add_argument("--live", action="store_true",
                    help="trade for real (requires ALPACA_KEY/SECRET, ALPACA_PAPER=false)")
    p.add_argument("--broker", choices=["alpaca", "paper"], default="alpaca")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--use-coreml", action="store_true")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "ane"], default="auto")
    p.add_argument("--once", action="store_true",
                    help="run a single step and exit (for cron mode)")
    args = p.parse_args()

    cfg = TradingConfig(
        model_dir=args.model_dir, bar=args.bar, universe_path=args.universe,
        paper=not args.live, broker=args.broker, poll_seconds=args.poll_seconds,
        top_k=args.top_k, use_coreml=args.use_coreml,
        inference_device=args.device,
    )
    runner = TradingRunner(cfg)
    if args.once:
        report = runner.step()
        print(json.dumps(report, indent=2, default=str))
    else:
        runner.run()


if __name__ == "__main__":
    cli()
