"""Live portfolio dashboard.

Single-page web UI that polls /api/state every 30 sec and renders:
  - account equity, cash, buying power
  - open positions with unrealised P&L
  - recent fills
  - latest top-10 signal scores (from data/trading_state/last_signal.json
    written by the runner each cycle)
  - equity curve over time (from data/trading_state/equity_ledger.csv)
  - risk status (halted / not halted, current trailing DD)

Run:
    PYTHONPATH=src python -m mlbt.dashboard.app
Then open: http://localhost:8765
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template


# Resolve paths relative to repo root (assumes module is in src/mlbt/dashboard/)
REPO = Path(__file__).resolve().parents[3]
STATE_DIR = REPO / "data" / "trading_state"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

load_dotenv()


def _alpaca() -> Any:
    """Lazy-init the Alpaca client. None if keys missing."""
    if not os.environ.get("ALPACA_KEY"):
        return None
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        return None
    return TradingClient(
        os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"],
        paper=os.environ.get("ALPACA_PAPER", "true").lower() != "false",
    )


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    client = _alpaca()
    if client is None:
        return jsonify({"error": "ALPACA_KEY missing — paste into .env"}), 200

    try:
        acct = client.get_account()
    except Exception as e:
        return jsonify({"error": f"alpaca: {e}"}), 200

    positions = client.get_all_positions()
    pos_list = []
    for p in positions:
        pos_list.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price) if p.current_price else None,
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
            "side": p.side.value,
        })

    # Last 20 closed orders
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        recent = client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL, limit=20))
        fills = []
        for o in recent:
            fills.append({
                "ts": str(o.submitted_at)[:19] if o.submitted_at else "",
                "symbol": o.symbol,
                "side": o.side.value,
                "qty": float(o.qty) if o.qty else 0,
                "status": o.status.value,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            })
    except Exception:
        fills = []

    # Last-known signal scores (written by runner each cycle)
    sig_path = STATE_DIR / "last_signal.json"
    last_signal = None
    if sig_path.exists():
        try:
            last_signal = json.loads(sig_path.read_text())
        except Exception:
            pass

    # Risk ledger
    risk = {}
    risk_path = STATE_DIR / "risk.json"
    if risk_path.exists():
        try:
            risk = json.loads(risk_path.read_text())
        except Exception:
            pass

    # Equity curve
    eq_path = STATE_DIR / "equity_ledger.csv"
    equity_curve = []
    if eq_path.exists():
        try:
            eq = pd.read_csv(eq_path)
            if len(eq) > 500:
                eq = eq.tail(500)
            equity_curve = eq.to_dict("records")
        except Exception:
            pass

    market_open = False
    try:
        market_open = bool(client.get_clock().is_open)
    except Exception:
        pass

    return jsonify({
        "account": {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": acct.status.value if hasattr(acct.status, "value") else str(acct.status),
            "account_number": acct.account_number,
            "currency": acct.currency,
        },
        "positions": pos_list,
        "recent_orders": fills,
        "last_signal": last_signal,
        "risk": risk,
        "equity_curve": equity_curve,
        "market_open": market_open,
        "now": pd.Timestamp.utcnow().isoformat(),
    })


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
