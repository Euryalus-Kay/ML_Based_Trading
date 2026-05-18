"""Auto-import every source module so they self-register on `import mlbt.sources`.

Adding a new source: drop a file in this directory that decorates a class
with @register("name"). It will appear in the CLI and orchestrator with no
further wiring.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from mlbt.core.log import get_logger

log = get_logger("sources")

_pkg_path = Path(__file__).parent
for mod in pkgutil.iter_modules([str(_pkg_path)]):
    if mod.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"mlbt.sources.{mod.name}")
    except Exception as e:  # noqa: BLE001
        # Don't let one broken source kill imports of the rest
        log.warning("failed to import source %s: %s", mod.name, e)
