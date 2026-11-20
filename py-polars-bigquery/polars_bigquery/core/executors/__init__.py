"""Shared executor abstractions and BigQuery scan backends."""

from __future__ import annotations

from polars_bigquery.core.executors.base import PlanExecutor, apply_local_plan
from polars_bigquery.core.executors.query import (
    CompiledQueryPlan,
    QueryExecutor,
    compile_query_plan,
)
from polars_bigquery.core.executors.read_api import (
    CompiledReadApiPlan,
    ReadApiExecutor,
    compile_read_api_plan,
)

__all__ = [
    "CompiledQueryPlan",
    "CompiledReadApiPlan",
    "PlanExecutor",
    "QueryExecutor",
    "ReadApiExecutor",
    "apply_local_plan",
    "compile_query_plan",
    "compile_read_api_plan",
]
