from __future__ import annotations

from arrow_bigquery._native import BigQueryTableId
from arrow_bigquery._read_bigquery import Client
from arrow_bigquery.core.version import __version__

__all__ = ["BigQueryTableId", "Client", "__version__"]
