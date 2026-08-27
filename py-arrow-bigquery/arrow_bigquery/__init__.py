from __future__ import annotations

from arrow_bigquery.core.version import __version__
from arrow_bigquery._read_bigquery import Client

__all__ = ["Client", "__version__"]
