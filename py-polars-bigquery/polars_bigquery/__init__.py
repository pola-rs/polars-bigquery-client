from __future__ import annotations

from polars_bigquery._read_bigquery import (
    Client,
)
from polars_bigquery.core.version import __version__

__all__ = [
    "Client",
    "__version__",
]
