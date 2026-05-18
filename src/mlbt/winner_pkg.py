"""Winner packaging: equity curve PNG, README update, metrics summary.

Run after a candidate passes all gates. Writes:
  docs/equity_curve.png
  data/<winner>/equity_curve.csv
  README.md updated with 'Production strategy' section

Usage:
    from mlbt.winner_pkg import package_winner
    package_winner(model_dir='data/cand_xs8_v1', dataset_path='data/dataset_1h_sp500.parquet',
                   target='y_xsec_top_8', top_k=10, market_neutral=False)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mlbt.backtest.portfolio import PortfolioConfig, run_portfolio_backtest
from mlbt.core.log import get_logger
from mlbt.core.storage import Storage

log = get_logger("winner_pkg")


def _save_equity_curve_png(out_path: str, strat_curve: pd.Series,
                            spy_curve: pd.Series, title: str = "Strategy vs SPY"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable; skipping png")
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(strat_curve.index, strat_curve.values, label="Strategy", color="#1f77b4", lw=1.5)
    ax.plot(spy_curve.index, spy_curve.values, label="SPY buy-hold", color="#888", lw=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Cumulative growth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("wrote %s", out_path)


def package_winner(
    model_dir: str, dataset_path: str, target: str = "y_xsec_top_8",
    top_k: int = 10, bottom_k: int = 0, market_neutral: bool = False,
    bps_per_trade: float = 5.0, slippage_bps: float = 2.0,
    out_dir: str = "docs",
) -> dict:
    """Run primary backtest, generate equity curve, write summary."""
    h = int(target.rsplit("_", 1)[-1])
    target_col = f"y_resid_logret_{h}"

    # PAPER regime
    paper_cfg = PortfolioConfig(
        horizon=h, bar="1h", target_col=target_col, entry_lag_bars=1,
        top_k=top_k, bottom_k=bottom_k, market_neutral=market_neutral,
        bps_per_trade=2.0, slippage_bps=1.0,
    )
    paper = run_portfolio_backtest(dataset_path=dataset_path,
                                     model_dir=model_dir, cfg=paper_cfg,
                                     out_path=f"{model_dir}/paper.json")
    retail_cfg = PortfolioConfig(
        horizon=h, bar="1h", target_col=target_col, entry_lag_bars=1,
        top_k=top_k, bottom_k=bottom_k, market_neutral=market_neutral,
        bps_per_trade=bps_per_trade, slippage_bps=slippage_bps,
    )
    retail = run_portfolio_backtest(dataset_path=dataset_path,
                                      model_dir=model_dir, cfg=retail_cfg,
                                      out_path=f"{model_dir}/retail.json")

    # Equity curve
    paper_pnl = pd.read_parquet(f"{model_dir}/paper.parquet")
    eq = (1 + paper_pnl["net_pnl"]).cumprod()
    spy = Storage().read("yf_1h", "SPY")
    if spy.empty:
        spy = Storage().read("yf_daily", "SPY")
    spy_eq = pd.Series(1.0, index=eq.index)
    if not spy.empty and "close" in spy.columns:
        spy_ret = spy["close"].astype(float).pct_change().reindex(eq.index).fillna(0)
        spy_eq = (1 + spy_ret).cumprod()
    _save_equity_curve_png(f"{out_dir}/equity_curve.png", eq, spy_eq,
                            title=f"{target} top_k={top_k} mn={market_neutral} vs SPY")

    summary = {
        "model_dir": model_dir, "dataset_path": dataset_path,
        "target": target, "top_k": top_k, "bottom_k": bottom_k,
        "market_neutral": market_neutral,
        "paper_metrics": paper["net"], "retail_metrics": retail["net"],
        "spy_metrics": paper.get("benchmark_spy"),
        "trade_count_paper": paper.get("n_rebalances"),
    }
    Path(f"{out_dir}/winner_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    import sys
    md = sys.argv[1]
    dp = sys.argv[2] if len(sys.argv) > 2 else "data/dataset_1h_sp500.parquet"
    print(json.dumps(package_winner(md, dp), indent=2, default=str))
