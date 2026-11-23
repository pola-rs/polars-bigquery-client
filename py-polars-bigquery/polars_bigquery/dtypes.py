"""Polars Arrow extension data types for BigQuery scalar `JSON` and `GEOGRAPHY` columns.

These constants represent the default unregistered `polars.Extension` dtypes
produced by Polars 2.0+ when reading BigQuery `JSON` and `GEOGRAPHY` columns
without custom extension registration via `polars.register_extension_type`.
"""

from __future__ import annotations

import polars as pl

from polars_bigquery.core.types import (
    GEOGRAPHY_EXTENSION_METADATA,
    GEOGRAPHY_EXTENSION_NAME,
    JSON_EXTENSION_NAME,
)

JSON: pl.DataType = pl.Extension(JSON_EXTENSION_NAME, pl.String())
"""Default unregistered `polars.Extension` dtype for BigQuery scalar `JSON` columns."""

GEOGRAPHY: pl.DataType = pl.Extension(
    GEOGRAPHY_EXTENSION_NAME,
    pl.String(),
    GEOGRAPHY_EXTENSION_METADATA,
)
"""Default unregistered `polars.Extension` dtype for BigQuery scalar `GEOGRAPHY` columns (WKT)."""

__all__ = ["GEOGRAPHY", "JSON"]
