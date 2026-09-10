from __future__ import annotations

from arrow_bigquery._read_bigquery import Client
from arrow_bigquery.api.resources import BigQueryTableId, parse_table_id
from arrow_bigquery.core.version import __version__

__all__ = ["BigQueryTableId", "Client", "__version__", "parse_table_id"]
