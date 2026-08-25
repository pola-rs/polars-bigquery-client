from __future__ import annotations

from arrow_bigquery.core.version import __version__
from arrow_bigquery._read_bigquery import Client, read_bigquery_table

__all__ = ["Client", "read_bigquery_table", "__version__"]
