from __future__ import annotations

from polars_bigquery.core.version import __version__
from polars_bigquery._read_bigquery import (
    Client,
)

__all__ = [
    "Client",
    "__version__",
]
