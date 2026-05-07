from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from . import polars_bigquery
from .core.run_query import run_query


def read_bigquery(
    *,
    table: str | None = None,
    query: str | None = None,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP | None = None,
    is_ordered: bool = False,
) -> pl.DataFrame:
    if not table and not query:
        raise ValueError("One of `table` or `query` must be provided.")

    if table and query:
        raise ValueError("Only one of `table` or `query` can be provided.")

    if not credentials_provider:
        credentials_provider = pl.CredentialProviderGCP(quota_project_id=quota_project_id)

    if query:
        table = run_query(query, quota_project_id, credentials_provider)

    return polars_bigquery.read_bigquery(table, quota_project_id, is_ordered, credentials_provider)

