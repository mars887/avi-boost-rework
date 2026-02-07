"""Metric registry and decorators."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

MetricFn = Callable[[object], float]
ResolverFn = Callable[[str], Optional[MetricFn]]


class MetricRegistry:
    """Registry for metrics used in rule scripts."""

    def __init__(self) -> None:
        self._items: Dict[str, MetricFn] = {}
        self._resolvers: List[ResolverFn] = []

    def register(self, name: str, fn: MetricFn) -> None:
        key = self._normalize(name)
        self._items[key] = fn

    def register_resolver(self, fn: ResolverFn) -> None:
        self._resolvers.append(fn)

    def get(self, name: str) -> MetricFn:
        key = self._normalize(name)
        if key in self._items:
            return self._items[key]
        for resolver in self._resolvers:
            found = resolver(key)
            if found is not None:
                return found
        raise KeyError(f"Unknown metric name: {name}")

    @staticmethod
    def _normalize(name: str) -> str:
        return str(name).strip().lower()


registry = MetricRegistry()


def metric(name: str):
    """Decorator to register a metric under a name."""

    def deco(fn: MetricFn) -> MetricFn:
        registry.register(name, fn)
        return fn

    return deco
