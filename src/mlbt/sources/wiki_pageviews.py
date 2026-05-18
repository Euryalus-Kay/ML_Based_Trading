"""Wikipedia pageviews — daily attention proxy for any article.

Public REST API, no key. Useful as an alt-data feature: spikes in pageviews
for a ticker's company page often precede or accompany abnormal volume.
"""
from __future__ import annotations

from typing import Any, List

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get
from mlbt.core.log import get_logger

log = get_logger("wiki_pv")

_BASE = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
         "en.wikipedia.org/all-access/all-agents/{title}/daily/{start}/{end}")


def _fetch_article(title: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = _BASE.format(
        title=title,
        start=start.strftime("%Y%m%d"),
        end=end.strftime("%Y%m%d"),
    )
    resp = http_get(url, min_interval=0.2, raise_on_error=False)
    if resp.status_code != 200:
        return pd.DataFrame()
    items = resp.json().get("items", [])
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df["ts"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H", utc=True)
    df = df.set_index("ts")[["views"]].astype(float)
    df = df.rename(columns={"views": f"wikipv_{title}"})
    return df


@register("wiki_pageviews")
class WikiPageviews(DataSource):
    name: str = "wiki_pageviews"
    frequency: str = "1d"
    schema = {}
    publish_lag = pd.Timedelta(hours=12)

    def fetch(self, start, end, *, pages: List[str] | None = None, **kw: Any) -> pd.DataFrame:
        pages = pages or []
        frames = []
        for p in pages:
            try:
                df = _fetch_article(p, pd.Timestamp(start), pd.Timestamp(end))
                if not df.empty:
                    frames.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("wiki %s failed: %s", p, e)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()
