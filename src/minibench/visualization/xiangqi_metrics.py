"""Numeric views of current Xiangqi results; missing evidence is never zero."""
from __future__ import annotations

from math import nan
import statistics


def metric_value(result, field: str) -> float:
    value = getattr(result, field)
    return nan if result.status == "error" or value is None else float(value)


def metric_mean(results, field: str) -> float:
    values = [float(getattr(result, field)) for result in results
              if result.status != "error" and getattr(result, field) is not None]
    return statistics.mean(values) if values else nan
