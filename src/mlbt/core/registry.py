"""Source plugin registry.

@register("name") decorates a DataSource subclass; the instance lives in the
global registry so the orchestrator and CLI can enumerate sources without
hard-coding imports. Source modules self-register on import.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from mlbt.core.base import DataSource

_REGISTRY: Dict[str, DataSource] = {}


def register(name: str) -> Callable[[Type[DataSource]], Type[DataSource]]:
    def deco(cls: Type[DataSource]) -> Type[DataSource]:
        inst = cls()
        if not inst.name:
            inst.name = name
        if name in _REGISTRY:
            raise ValueError(f"duplicate source name: {name}")
        _REGISTRY[name] = inst
        return cls
    return deco


def get_source(name: str) -> DataSource:
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; have: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_sources() -> List[DataSource]:
    # ensure all source modules are imported so they register
    import mlbt.sources  # noqa: F401
    return list(_REGISTRY.values())


def enabled_sources() -> List[DataSource]:
    return [s for s in all_sources() if s.is_available()]
