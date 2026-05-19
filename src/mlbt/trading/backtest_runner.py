"""Backtest the live trading stack against historical data.

Replays the predictions.parquet + dataset against the same OMS/Risk/Tracker
chain used live, so we get a like-for-like result instead of having a
separate backtest engine that diverges from production behavior.

Differences from the live runner:
  - no model inference (we use the saved predictions.parquet)
  - no real broker — uses PaperBrokerAdapter fed with prices from the dataset
  - no sleep loop — walks bar-by-bar through history as fast as the CPU goes

Usage:
    python -m mlbt.trading.backtest_runner \\
        --model-dir data/cand_xs8_v1 \\
        --dataset data/dataset_1h_micro.parquet \\
        --top-k 10 --out data/cand_xs8_v1/live_replay.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.core.log import get_logger
from mlbt.trading.broker import PaperBrokerAdapter
from mlbt.trading.config import TradingConfig
from mlbt.trading.oms import OrderManagementSystem
from mlbt.trading.positions import PositionTracker
from mlbt.trading.risk import RiskManager

log = get_logger("bt_runner")


def replay(model_dir: str, dataset_path: str, out_path: Optional[str] = None,
            top_k: int = 10, bottom_k: int = 0, market_neutral: bool = False,
            bps_per_trade: float = 5.0, slippage_bps: float = 2.0,
            min_universe_size: int = 6,
            entry_lag_bars: int = 1,
            rebalance_every_bars: int = 1,
            sizing_mode: str = "equal",
            enable_kill_switch: bool = False,
            max_drawdown_kill: float = -0.30,
            # Live-mode exit rules (off by default to preserve alpha-only mode)
            enable_stops: bool = False,
            stop_loss_pct: float = -0.08,
            profit_take_pct: float = 0.20,
            enable_trailing_stop: bool = False,
            trailing_stop_pct: float = -0.05,
            enable_score_decay_exit: bool = False,
            decay_score_threshold: float = 0.495,
            enable_time_exit: bool = False,
            max_hold_bars: int = 32,
            start_equity: float = 100_000.0) -> dict:
    model_p = Path(model_dir)
    preds = pd.read_parquet(model_p / "predictions.parquet")
    ds = pd.read_parquet(dataset_path)
    if "symbol" not in preds.columns:
        raise ValueError("predictions need a 'symbol' column")

    # Pull close prices per (ts, symbol) from the dataset
    if "close" not in ds.columns:
        raise ValueError("dataset must contain a 'close' column")
    prices_df = (ds.reset_index().rename(columns={"index": "ts"})
                    [["ts", "symbol", "close"]]
                    .pivot_table(index="ts", columns="symbol", values="close"))
    prices_df = prices_df.sort_index().ffill()

    # Also build a vol_20 lookup so the OMS conf_vol sizing has data
    vol_col = "vol_20" if "vol_20" in ds.columns else None
    vol_df = None
    if vol_col is not None:
        vol_df = (ds.reset_index().rename(columns={"index": "ts"})
                    [["ts", "symbol", vol_col]]
                    .pivot_table(index="ts", columns="symbol", values=vol_col)
                    .sort_index().ffill())

    # Merge predictions w/ next-bar realised return so we know mark-to-market P&L
    # at the next bar.
    cfg = TradingConfig(top_k=top_k, bottom_k=bottom_k,
                          market_neutral=market_neutral,
                          gross_leverage=1.0, max_position_pct=0.15,
                          min_universe_size=min_universe_size,
                          bps_slippage_estimate=slippage_bps,
                          rebalance_every_bars=rebalance_every_bars,
                          sizing_mode=sizing_mode,
                          enable_kill_switch=enable_kill_switch,
                          max_drawdown_kill=max_drawdown_kill,
                          enable_stops=enable_stops,
                          stop_loss_pct=stop_loss_pct,
                          profit_take_pct=profit_take_pct,
                          enable_trailing_stop=enable_trailing_stop,
                          trailing_stop_pct=trailing_stop_pct,
                          enable_score_decay_exit=enable_score_decay_exit,
                          decay_score_threshold=decay_score_threshold,
                          enable_time_exit=enable_time_exit,
                          max_hold_bars=max_hold_bars,
                          poll_seconds=0)
    tracker = PositionTracker(equity=start_equity)
    broker = PaperBrokerAdapter(tracker, slippage_bps=bps_per_trade + slippage_bps)
    oms = OrderManagementSystem(cfg)
    risk = RiskManager(cfg)
    # Per-position exit tracker (lazy import to avoid cycles)
    from mlbt.trading.exits import ExitTracker
    exits = ExitTracker()

    # Group predictions by timestamp so we step the same way as the live runner
    preds = preds.reset_index().rename(columns={"index": "ts"})
    preds["ts"] = pd.to_datetime(preds["ts"])
    if preds["ts"].dt.tz is None:
        preds["ts"] = preds["ts"].dt.tz_localize("UTC")
    timestamps = sorted(preds["ts"].unique())
    log.info("replay starts: %d timestamps, %d preds", len(timestamps), len(preds))

    equity_curve = []
    n_orders_total = 0
    halted_at = None
    bars_since_rebalance = cfg.rebalance_every_bars  # rebalance on first bar
    last_targets: Optional[dict] = None
    for ts in timestamps:
        # Look up prices at this bar (or last-known)
        if ts in prices_df.index:
            ts_prices = prices_df.loc[ts].to_dict()
        else:
            # nearest earlier bar
            sub = prices_df.loc[:ts]
            if sub.empty:
                continue
            ts_prices = sub.iloc[-1].to_dict()
        ts_prices = {s: float(p) for s, p in ts_prices.items() if pd.notna(p)}
        broker.update_prices(ts_prices)
        equity_before = broker.get_equity()

        if risk.flatten_signal():
            # Already halted — flatten remaining positions if any
            positions_now = broker.get_positions()
            if positions_now:
                from mlbt.trading.oms import Order
                for s, q in positions_now.items():
                    side = "sell" if q > 0 else "buy"
                    broker.submit_order(s, abs(q), side, ref_price=ts_prices.get(s))
            halted_at = halted_at or str(ts)
            equity_curve.append({"ts": str(ts), "equity": broker.get_equity(),
                                   "halted": True})
            continue

        # Score frame for this bar (and vol_20 if available, for conf_vol sizing)
        scored = preds[preds["ts"] == ts][["symbol", "y_score"]].dropna()
        if vol_df is not None and ts in vol_df.index:
            vols_at_ts = vol_df.loc[ts]
            scored = scored.assign(vol_20=scored["symbol"].map(vols_at_ts))

        # Rebalance gate: only refresh target weights AND submit orders every
        # `rebalance_every_bars`. Between rebalances we let positions drift —
        # this matches how the offline backtest implicitly holds for h bars.
        do_rebalance = (bars_since_rebalance >= cfg.rebalance_every_bars
                          and len(scored) >= cfg.min_universe_size)
        if do_rebalance:
            targets = oms.scores_to_targets(scored, pd.Timestamp(ts))
            targets.weights = risk.trim_to_caps(targets.weights)
            last_targets = targets
            bars_since_rebalance = 0
        else:
            bars_since_rebalance += 1

        # Exit-tracker bookkeeping every bar (not just rebalance bars)
        current_positions = broker.get_positions()
        scores_by_sym = dict(zip(scored["symbol"], scored["y_score"]))
        # Entry prices: tracker.positions[sym].avg_price (broker is in-process)
        entry_prices = {s: (tracker.positions[s].avg_price
                            if s in tracker.positions else ts_prices.get(s, 0))
                        for s, q in current_positions.items() if q != 0}
        exits.update_on_step(current_positions, ts_prices, scores_by_sym,
                              pd.Timestamp(ts), entry_prices)

        orders = []
        if do_rebalance and last_targets is not None:
            orders = oms.targets_to_orders(last_targets, current_positions,
                                              ts_prices, equity_before)

        # Exit overrides every bar — force-close any position whose exit
        # rule has tripped, regardless of whether this is a rebalance bar
        from mlbt.trading.oms import Order
        n_exits = 0
        for s, qty in list(current_positions.items()):
            if qty == 0 or s not in ts_prices:
                continue
            decision = exits.check(s, ts_prices[s], cfg)
            if decision is None:
                continue
            orders = [o for o in orders if o.symbol != s]
            side = "sell" if qty > 0 else "buy"
            orders.insert(0, Order(symbol=s, qty=abs(qty), side=side,
                                      target_weight=0.0,
                                      ref_price=ts_prices[s]))
            n_exits += 1

        for o in orders:
            broker.submit_order(o.symbol, o.qty, o.side, ref_price=o.ref_price)
            n_orders_total += 1

        equity_after = broker.get_equity()
        bar_pnl = equity_after - equity_before
        breach = risk.update(equity_after, bar_pnl)
        equity_curve.append({"ts": str(ts), "equity": equity_after,
                               "bar_pnl": bar_pnl, "n_orders": len(orders),
                               "n_exits": n_exits,
                               "n_held": sum(1 for q in current_positions.values() if q != 0),
                               "gross": targets.gross() if 'targets' in dir() and targets else 0.0,
                               "halted": bool(breach and breach.severity == "halt")})

    # Compute metrics
    eq = pd.DataFrame(equity_curve)
    if eq.empty:
        return {"error": "no bars"}
    eq["ts"] = pd.to_datetime(eq["ts"])
    eq = eq.set_index("ts").sort_index()
    eq["ret"] = eq["equity"].pct_change().fillna(0)
    n_bars = len(eq)
    bpy = 252 * 6.5 if "1h" in dataset_path or "_1h_" in dataset_path else 252
    mu = eq["ret"].mean(); sd = eq["ret"].std()
    sharpe = (mu / sd) * np.sqrt(bpy) if sd > 0 else 0
    drawdown = (eq["equity"] / eq["equity"].cummax() - 1.0).min()
    total_ret = (eq["equity"].iloc[-1] / start_equity) - 1.0
    ann_ret = mu * bpy
    calmar = ann_ret / abs(drawdown) if drawdown < 0 else float("inf")

    # Production metrics
    universe_size = preds["symbol"].nunique()
    bars_per_day = bpy / 252.0
    trading_days = n_bars / bars_per_day
    rebalances = n_bars / cfg.rebalance_every_bars
    signals_per_day = universe_size * bars_per_day        # all symbols scored every bar
    opportunities_per_day = top_k * (rebalances / trading_days)  # top-k rebalances/day
    trades_per_day = n_orders_total / max(trading_days, 1)
    daily_pnl_mean = eq["ret"].sum() * start_equity / max(trading_days, 1)
    n_exit_events = int(eq["n_exits"].sum()) if "n_exits" in eq.columns else 0
    # Capital deployment: average gross exposure
    avg_gross = float(eq.get("gross", pd.Series([0])).mean())

    summary = {
        "n_bars": n_bars,
        "trading_days": round(trading_days, 1),
        "universe_size": universe_size,
        "start_equity": start_equity,
        "end_equity": float(eq["equity"].iloc[-1]),
        "total_return": float(total_ret), "ann_return": float(ann_ret),
        "sharpe": float(sharpe), "max_drawdown": float(drawdown),
        "calmar": float(calmar),
        # Opportunity / activity metrics
        "n_orders_total": n_orders_total,
        "trades_per_day": round(trades_per_day, 1),
        "signals_per_day": round(signals_per_day, 1),
        "opportunities_per_day": round(opportunities_per_day, 1),
        "rebalances_total": int(rebalances),
        "n_exit_events": n_exit_events,
        "avg_gross_exposure": round(avg_gross, 3),
        "daily_pnl_mean": round(daily_pnl_mean, 2),
        "halted_at": halted_at,
        "cfg": dict(top_k=top_k, bottom_k=bottom_k,
                     market_neutral=market_neutral,
                     bps_per_trade=bps_per_trade, slippage_bps=slippage_bps,
                     entry_lag_bars=entry_lag_bars,
                     enable_stops=enable_stops,
                     enable_trailing_stop=enable_trailing_stop,
                     enable_score_decay_exit=enable_score_decay_exit,
                     enable_time_exit=enable_time_exit),
    }

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2, default=str))
        eq.to_parquet(out_p.with_suffix(".parquet"))
        log.info("wrote %s and %s", out_p, out_p.with_suffix(".parquet"))

    print(f"\n=== live-replay backtest ===")
    print(f"Universe:               {universe_size} stocks  |  {trading_days:.0f} trading days  |  {n_bars} bars")
    print(f"Sharpe:                 {sharpe:.2f}")
    print(f"Ann return:             {ann_ret:.2%}")
    print(f"Max DD:                 {drawdown:.2%}")
    print(f"Total return:           {total_ret:.2%}")
    print(f"End equity:             ${eq['equity'].iloc[-1]:,.0f}")
    print(f"Avg daily P&L:          ${daily_pnl_mean:.0f}")
    print(f"Signals/day evaluated:  {signals_per_day:.0f}")
    print(f"Opportunities/day picked: {opportunities_per_day:.1f}")
    print(f"Trades/day:             {trades_per_day:.1f}")
    print(f"Total orders:           {n_orders_total}")
    print(f"Exit events fired:      {n_exit_events}")
    print(f"Avg gross exposure:     {avg_gross:.0%}")
    return summary


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="data/cand_xs8_v1")
    p.add_argument("--dataset", default="data/dataset_1h_micro.parquet")
    p.add_argument("--out", default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bottom-k", type=int, default=0)
    p.add_argument("--market-neutral", action="store_true")
    p.add_argument("--bps", type=float, default=5.0)
    p.add_argument("--slippage", type=float, default=2.0)
    p.add_argument("--rebalance-every", type=int, default=1)
    p.add_argument("--sizing", choices=["equal", "confidence", "conf_vol"],
                    default="equal")
    p.add_argument("--kill-switch", action="store_true",
                    help="Enable trailing max-DD kill switch (default off in backtest)")
    p.add_argument("--max-dd-kill", type=float, default=-0.30)
    p.add_argument("--start-equity", type=float, default=100_000.0)
    args = p.parse_args()
    replay(model_dir=args.model_dir, dataset_path=args.dataset,
           out_path=args.out, top_k=args.top_k, bottom_k=args.bottom_k,
           market_neutral=args.market_neutral,
           bps_per_trade=args.bps, slippage_bps=args.slippage,
           rebalance_every_bars=args.rebalance_every,
           sizing_mode=args.sizing,
           enable_kill_switch=args.kill_switch,
           max_drawdown_kill=args.max_dd_kill,
           start_equity=args.start_equity)


if __name__ == "__main__":
    cli()
