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
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from mlbt.core.log import get_logger
from mlbt.trading.broker import make_broker, PaperBrokerAdapter
from mlbt.trading.config import TradingConfig
from mlbt.trading.exits import ExitTracker
from mlbt.trading.oms import OrderManagementSystem
from mlbt.trading.positions import PositionTracker
from mlbt.trading.risk import RiskManager
from mlbt.trading.signal import LiveSignalGenerator
from mlbt.trading.signal_cached import CachedSignalGenerator

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
        self.exits = ExitTracker(state_path=str(self.state_dir / "exits.json"))
        # Cached signal generator: ~2 sec per cycle after the first
        # ~35-sec warm-up (vs 35 sec every cycle with the vanilla one).
        SignalCls = CachedSignalGenerator if getattr(cfg, "use_signal_cache", True) else LiveSignalGenerator
        self.signal = SignalCls(
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
        # 0. Cancel any orders still pending from prior cycles so we have a
        # clean slate. Avoids the bug where successive cycles submitted the
        # full top-10 again because pending orders weren't counted as
        # in-flight positions.
        if hasattr(self.broker, "cancel_all_open_orders"):
            n_cancelled = self.broker.cancel_all_open_orders()
            if n_cancelled:
                log.info("cancelled %d pending orders from prior cycle", n_cancelled)

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
        # Pull prices from the scored frame (signal builds them along with scores)
        # so the paper broker can size orders without an external quote feed.
        signal_prices = {}
        if "close" in scores.columns:
            signal_prices = {row.symbol: float(row.close)
                              for row in scores.itertuples()
                              if pd.notna(row.close) and row.close > 0}
        if isinstance(self.broker, PaperBrokerAdapter) and signal_prices:
            self.broker.update_prices(signal_prices)
        prices = {s: self.broker.get_last_price(s) or signal_prices.get(s) or 0.0
                  for s in targets.weights}
        equity = self.broker.get_equity() or self.tracker.total_equity(prices)
        current_positions = self.broker.get_positions()
        orders = self.oms.targets_to_orders(targets, current_positions,
                                              prices, equity,
                                              min_order_dollars=50.0)

        # 4b. Refresh per-position exit-tracker state (high-mark, bars-held,
        # latest score) and check each open position against the full exit
        # rules: stop-loss, profit-take, trailing-stop, score-decay, time-exit.
        # Any triggered exit overrides whatever the model thinks.
        from mlbt.trading.oms import Order
        scores_by_sym = {row.symbol: float(row.y_score)
                         for row in scores.itertuples()}
        entry_prices = {s: (self._position_entry_price(s) or prices.get(s, 0.0))
                        for s in current_positions if current_positions[s] != 0}
        self.exits.update_on_step(current_positions, prices, scores_by_sym, ts, entry_prices)

        stops_triggered = []
        for s, qty in current_positions.items():
            if qty == 0 or s not in prices:
                continue
            decision = self.exits.check(s, prices[s], self.cfg)
            if decision is None:
                continue
            side = "sell" if qty > 0 else "buy"
            orders = [o for o in orders if o.symbol != s]
            orders.insert(0, Order(symbol=s, qty=abs(qty), side=side,
                                      target_weight=0.0, ref_price=prices[s]))
            stops_triggered.append({"symbol": s, "kind": decision.reason,
                                      "detail": decision.detail,
                                      "pl_pct": decision.pl_pct})
            log.warning("EXIT %s on %s: %s", decision.reason.upper(), s, decision.detail)

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

        # Persist the latest top-10 signal + weights for the dashboard
        try:
            top_n = scores.head(20).copy()
            sig_rows = []
            for _, r in top_n.iterrows():
                sig_rows.append({
                    "symbol": str(r.symbol),
                    "score": float(r.y_score),
                    "edge": float(r.get("edge", r.y_score - 0.5)),
                    "weight": float(targets.weights.get(r.symbol, 0.0)),
                    "close": float(r.close) if pd.notna(r.get("close")) else None,
                })
            (self.state_dir / "last_signal.json").write_text(json.dumps({
                "ts": str(ts), "scores": sig_rows,
                "top_k": self.cfg.top_k, "model_dir": self.cfg.model_dir,
            }, indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            log.warning("could not write last_signal.json: %s", e)

        # Append to the equity ledger so the dashboard can plot the curve
        try:
            eq_path = self.state_dir / "equity_ledger.csv"
            line = f"{ts.isoformat()},{equity:.4f},{bar_pnl:.4f},{len(orders)},{int(self.risk.halted)}\n"
            if not eq_path.exists():
                eq_path.write_text("ts,equity,bar_pnl,n_orders,halted\n" + line)
            else:
                with eq_path.open("a") as f:
                    f.write(line)
        except Exception as e:  # noqa: BLE001
            log.warning("could not append equity_ledger: %s", e)

        return report

    def _position_entry_price(self, symbol: str) -> Optional[float]:
        """Look up avg entry price for a symbol from broker or local tracker."""
        # PaperBrokerAdapter routes through tracker
        if symbol in self.tracker.positions:
            return self.tracker.positions[symbol].avg_price
        # AlpacaBrokerAdapter — query Alpaca directly
        try:
            from alpaca.trading.client import TradingClient
            if hasattr(self.broker, "client"):
                positions = self.broker.client.get_all_positions()
                for p in positions:
                    if p.symbol == symbol:
                        return float(p.avg_entry_price)
        except Exception:
            pass
        return None

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
