"""SEC EDGAR filings — recent filings flow per ticker (no key).

Pulls the submissions JSON for each CIK and emits an event row per filing.
Used downstream as a categorical event feature (binary: "had_filing_today")
and a counter (form-type counts in last N days).
"""
from __future__ import annotations

from typing import Any, List

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get
from mlbt.core.log import get_logger

log = get_logger("edgar")

_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"


def _load_ticker_to_cik() -> dict[str, str]:
    resp = http_get(_TICKER_MAP, min_interval=0.5,
                    headers={"User-Agent": "mlbt research@example.com"},
                    raise_on_error=False)
    if resp.status_code != 200:
        return {}
    payload = resp.json()
    out = {}
    for v in payload.values():
        out[v["ticker"].upper()] = str(v["cik_str"]).zfill(10)
    return out


def _fetch_submissions(cik: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = http_get(url, min_interval=0.3,
                    headers={"User-Agent": "mlbt research@example.com"},
                    raise_on_error=False)
    if resp.status_code != 200:
        return pd.DataFrame()
    payload = resp.json()
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame({
        "filing_date": recent.get("filingDate", []),
        "form": recent.get("form", []),
        "accession": recent.get("accessionNumber", []),
    })
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["filing_date"]).set_index("filing_date")
    # publish at filing_date + 1 day @ 12:00 UTC (filings often arrive after close)
    df.index = df.index.normalize() + pd.Timedelta(hours=22)
    return df


@register("sec_edgar")
class SecEdgar(DataSource):
    name: str = "sec_edgar"
    frequency: str = "event"
    schema = {"form": "object", "accession": "object", "symbol": "object",
              "filing_event": "float64"}
    publish_lag = pd.Timedelta(0)  # we already shifted to publish time

    def fetch(self, start, end, *, symbols: List[str] | None = None, **kw: Any) -> pd.DataFrame:
        symbols = symbols or []
        if not symbols:
            return pd.DataFrame()
        mapping = _load_ticker_to_cik()
        out = []
        for s in symbols:
            cik = mapping.get(s.upper())
            if not cik:
                continue
            try:
                df = _fetch_submissions(cik)
                if df.empty:
                    continue
                df["symbol"] = s
                df["filing_event"] = 1.0
                out.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("edgar %s failed: %s", s, e)
        if not out:
            return pd.DataFrame()
        df = pd.concat(out).sort_index()
        s_ts = pd.Timestamp(start, tz="UTC")
        e_ts = pd.Timestamp(end, tz="UTC")
        return df.loc[(df.index >= s_ts) & (df.index <= e_ts)]
