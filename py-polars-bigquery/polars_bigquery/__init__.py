from __future__ import annotations

from polars_bigquery.core.version import __version__
from polars_bigquery._read_bigquery import read_bigquery, read_bigquery_query

__all__ = ["read_bigquery", "read_bigquery_query", "__version__"]
