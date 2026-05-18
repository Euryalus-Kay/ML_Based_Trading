"""mlbt — multi-source, time-aligned market data collector."""
__version__ = "0.1.0"

from mlbt.core.base import DataSource  # noqa: F401
from mlbt.core.registry import register, get_source, all_sources  # noqa: F401
