"""Shared executor abstractions and BigQuery scan backends."""

from __future__ import annotations

from polars_bigquery.core.executors.base import (
    ArrowQueryClient,
    ArrowTableClient,
    PlanExecutor,
    apply_local_plan,
    execute_local_plan_batches,
)
from polars_bigquery.core.executors.query import (
    CompiledQueryPlan,
    QueryExecutor,
    compile_query_plan,
    has_non_comment_sql,
)
from polars_bigquery.core.executors.read_api import (
    CompiledReadApiPlan,
    ReadApiDegradedReason,
    ReadApiExecutor,
    compile_read_api_plan,
)

__all__ = [
    "ArrowQueryClient",
    "ArrowTableClient",
    "CompiledQueryPlan",
    "CompiledReadApiPlan",
    "PlanExecutor",
    "QueryExecutor",
    "ReadApiDegradedReason",
    "ReadApiExecutor",
    "apply_local_plan",
    "compile_query_plan",
    "compile_read_api_plan",
    "execute_local_plan_batches",
    "has_non_comment_sql",
]
