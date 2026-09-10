"""Resource reference utilities for BigQuery."""

from __future__ import annotations

from arrow_bigquery._native import BigQueryTableId

parse_table_id = BigQueryTableId.parse

__all__ = [
    "BigQueryTableId",
    "parse_table_id",
]
