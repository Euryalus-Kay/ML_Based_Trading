"""Pull the alternative data sources we haven't been using:
  - finra_short  (per-stock daily short volume / total volume)
  - wiki_pageviews  (per-company daily attention)
  - gdelt  (global news tone aggregates)

For 2024-05-20 → 2026-05-17 covering the 49 mega-caps.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/Users/zainzaidi/Desktop/ML MODEL/ML_Based_Trading/src")
import pandas as pd

from mlbt.core.storage import Storage
from mlbt.sources.finra_short import FinraShort
from mlbt.sources.wiki_pageviews import WikiPageviews
from mlbt.sources.gdelt import GdeltDaily


# Hand-curated ticker → wikipedia page name. From config/universe.yaml.
TICKER_TO_WIKI = {
    "AAPL": "Apple_Inc.", "MSFT": "Microsoft", "GOOGL": "Alphabet_Inc.",
    "AMZN": "Amazon_(company)", "NVDA": "Nvidia", "META": "Meta_Platforms",
    "TSLA": "Tesla,_Inc.", "AVGO": "Broadcom", "ORCL": "Oracle_Corporation",
    "JPM": "JPMorgan_Chase", "BAC": "Bank_of_America", "GS": "Goldman_Sachs",
    "MS": "Morgan_Stanley", "C": "Citigroup", "WFC": "Wells_Fargo",
    "AMD": "AMD", "INTC": "Intel", "MU": "Micron_Technology",
    "AMAT": "Applied_Materials", "LRCX": "Lam_Research", "KLAC": "KLA_Corporation",
    "XOM": "ExxonMobil", "CVX": "Chevron_Corporation", "COP": "ConocoPhillips",
    "SLB": "SLB", "EOG": "EOG_Resources", "OXY": "Occidental_Petroleum",
    "WMT": "Walmart", "COST": "Costco", "HD": "The_Home_Depot",
    "NKE": "Nike,_Inc.", "MCD": "McDonald%27s", "SBUX": "Starbucks",
    "TGT": "Target_Corporation",
    "UNH": "UnitedHealth_Group", "JNJ": "Johnson_%26_Johnson",
    "LLY": "Eli_Lilly_and_Company", "PFE": "Pfizer", "ABBV": "AbbVie",
    "MRK": "Merck_%26_Co.",
    "PLTR": "Palantir_Technologies", "COIN": "Coinbase",
}


def pull_finra(start, end) -> None:
    """One file per weekday, all symbols inside."""
    print(f"=== finra_short ({start.date()} → {end.date()}) ===")
    fs = FinraShort()
    df = fs.fetch_safe(start, end)
    if df.empty:
        print("  no data"); return
    print(f"  {len(df)} rows, {df['symbol'].nunique()} symbols")
    # Save per-symbol
    st = Storage()
    n_written = 0
    for sym, sub in df.groupby("symbol"):
        st.write("finra_short", sym, sub.drop(columns=["symbol"]))
        n_written += 1
    print(f"  wrote {n_written} symbol files")


def pull_wiki(start, end) -> None:
    print(f"=== wiki_pageviews ({start.date()} → {end.date()}) ===")
    w = WikiPageviews()
    pages = list(TICKER_TO_WIKI.values())
    df = w.fetch_safe(start, end, pages=pages)
    if df.empty:
        print("  no data"); return
    print(f"  {len(df)} rows, {df.shape[1]} pages")
    st = Storage()
    for col in df.columns:
        sub = df[[col]].dropna()
        if not sub.empty:
            st.write("wiki_pageviews", col, sub)
    print(f"  wrote {df.shape[1]} page files")


def pull_gdelt(start, end) -> None:
    print(f"=== gdelt ({start.date()} → {end.date()}) ===")
    g = GdeltDaily()
    df = g.fetch_safe(start, end)
    if df.empty:
        print("  no data"); return
    print(f"  {len(df)} rows, cols: {list(df.columns)}")
    Storage().write("gdelt", "_default", df)
    print(f"  wrote 1 file")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-05-20")
    p.add_argument("--end", default="2026-05-17")
    p.add_argument("--source", choices=["finra", "wiki", "gdelt", "all"], default="all")
    args = p.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    t0 = time.monotonic()
    if args.source in ("finra", "all"):
        pull_finra(start, end)
    if args.source in ("wiki", "all"):
        pull_wiki(start, end)
    if args.source in ("gdelt", "all"):
        pull_gdelt(start, end)
    print(f"\ndone in {time.monotonic()-t0:.0f} sec")
