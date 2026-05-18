"""Parquet-on-disk storage with simple month partitioning.

Layout:
  data/raw/<source>/<symbol_or_id>/year=YYYY/month=MM/part.parquet

For non-symbol sources (FRED series, sentiment indices), <symbol_or_id> = the
metric/series id. Re-running a backfill is idempotent: we merge on the index.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from mlbt.core.log import get_logger

log = get_logger("storage")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(s: str) -> str:
    return _SAFE.sub("_", s)


class Storage:
    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root or os.environ.get("MLBT_DATA_DIR", "./data"))
        self.raw = self.root / "raw"
        self.aligned = self.root / "aligned"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.aligned.mkdir(parents=True, exist_ok=True)

    # -------- raw store -------------------------------------------------------
    def _path(self, source: str, key: str) -> Path:
        return self.raw / _safe(source) / _safe(key)

    def write(self, source: str, key: str, df: pd.DataFrame) -> Path:
        if df is None or df.empty:
            return self._path(source, key)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df must have a DatetimeIndex")
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
        existing = self.read(source, key)
        if not existing.empty:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        out = self._path(source, key)
        out.mkdir(parents=True, exist_ok=True)
        for (year, month), part in df.groupby([df.index.year, df.index.month]):
            sub = out / f"year={year}" / f"month={month:02d}"
            sub.mkdir(parents=True, exist_ok=True)
            part.to_parquet(sub / "part.parquet", engine="pyarrow")
        return out

    def read(self, source: str, key: str) -> pd.DataFrame:
        p = self._path(source, key)
        if not p.exists():
            return pd.DataFrame()
        files = sorted(p.rglob("part.parquet"))
        if not files:
            return pd.DataFrame()
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as e:  # noqa: BLE001
                log.warning("read failed %s :: %s", f, e)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def list_keys(self, source: str) -> Iterable[str]:
        p = self.raw / _safe(source)
        if not p.exists():
            return []
        return sorted(child.name for child in p.iterdir() if child.is_dir())

    # -------- aligned store ---------------------------------------------------
    def write_aligned(self, name: str, df: pd.DataFrame) -> Path:
        out = self.aligned / f"{_safe(name)}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, engine="pyarrow")
        return out

    def read_aligned(self, name: str) -> pd.DataFrame:
        p = self.aligned / f"{_safe(name)}.parquet"
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)
