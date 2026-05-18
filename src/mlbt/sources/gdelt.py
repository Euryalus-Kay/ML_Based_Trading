"""GDELT 2.0 — global news tone, no key.

Per 15-minute interval, GDELT emits an aggregated tone CSV of every news
article worldwide. We pull the daily summary file `mentions.CSV.zip` and
aggregate tone per day. As an alt-data feature this captures global news
sentiment shifts that often lead equity moves by hours.
"""
from __future__ import annotations

from typing import Any

import io
import zipfile

import pandas as pd

from mlbt.core.base import DataSource
from mlbt.core.registry import register
from mlbt.core.http import http_get
from mlbt.core.log import get_logger

log = get_logger("gdelt")

# GDELT v1 Daily Updates: simpler than v2, daily CSVs of country-level tone
# We use the GKG (Global Knowledge Graph) summary endpoint
_GKG_DAILY = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"


def _fetch_day(day: pd.Timestamp) -> pd.DataFrame:
    url = _GKG_DAILY.format(date=day.strftime("%Y%m%d"))
    resp = http_get(url, min_interval=0.2, raise_on_error=False)
    if resp.status_code != 200 or not resp.content:
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                df = pd.read_csv(f, sep="\t", header=None, low_memory=False,
                                 on_bad_lines="skip")
    except Exception as e:  # noqa: BLE001
        log.warning("gdelt %s parse failed: %s", day, e)
        return pd.DataFrame()
    if df.empty or df.shape[1] < 8:
        return pd.DataFrame()
    # Column layout for GKG v1: see http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook.pdf
    # We only need tone (col idx 7). Tone is "tone,positive,negative,polarity,actref,grpref,wordcount"
    tone = df.iloc[:, 7].dropna().astype(str)
    parsed = tone.str.split(",", expand=True).iloc[:, 0]
    parsed = pd.to_numeric(parsed, errors="coerce").dropna()
    if parsed.empty:
        return pd.DataFrame()
    summary = {
        "gdelt_tone_mean": float(parsed.mean()),
        "gdelt_tone_std": float(parsed.std()),
        "gdelt_tone_p10": float(parsed.quantile(0.10)),
        "gdelt_tone_p90": float(parsed.quantile(0.90)),
        "gdelt_article_count": float(len(parsed)),
    }
    return pd.DataFrame([summary], index=pd.DatetimeIndex(
        [day.normalize() + pd.Timedelta(hours=23)], tz="UTC"))


@register("gdelt")
class GdeltDaily(DataSource):
    name: str = "gdelt"
    frequency: str = "1d"
    schema = {
        "gdelt_tone_mean": "float64", "gdelt_tone_std": "float64",
        "gdelt_tone_p10": "float64", "gdelt_tone_p90": "float64",
        "gdelt_article_count": "float64",
    }
    publish_lag = pd.Timedelta(hours=2)

    def fetch(self, start, end, **kw: Any) -> pd.DataFrame:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if start_ts < pd.Timestamp("2013-04-01"):
            start_ts = pd.Timestamp("2013-04-01")  # GKG starts here
        dates = pd.date_range(start_ts, end_ts, freq="1D")
        frames = []
        for d in dates:
            try:
                df = _fetch_day(d)
                if not df.empty:
                    frames.append(df)
            except Exception as e:  # noqa: BLE001
                log.warning("gdelt %s failed: %s", d, e)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()
