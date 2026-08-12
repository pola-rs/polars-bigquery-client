from __future__ import annotations

from polars_bigquery.core.version import __version__
from polars_bigquery._read_bigquery import read_bigquery_table, read_bigquery_query, scan_bigquery_table

__all__ = ["read_bigquery_table", "read_bigquery_query", "scan_bigquery_table", "__version__"]
