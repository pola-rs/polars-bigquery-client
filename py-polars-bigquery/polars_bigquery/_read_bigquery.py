from __future__ import annotations

from typing import Any, Dict

import polars as pl

from . import polars_bigquery
from .core.run_query import run_query

_DEFAULT_CREDENTIAL_PROVIDERS: Dict[str, pl.CredentialProviderGCP] = {}


def _get_default_provider(quota_project_id: str) -> pl.CredentialProviderGCP:
    if quota_project_id not in _DEFAULT_CREDENTIAL_PROVIDERS:
        _DEFAULT_CREDENTIAL_PROVIDERS[quota_project_id] = pl.CredentialProviderGCP(
            quota_project_id=quota_project_id
        )
    return _DEFAULT_CREDENTIAL_PROVIDERS[quota_project_id]


def _parse_table_id(table_id: Any) -> str:
    if not isinstance(table_id, str):
        if (
            hasattr(table_id, "project")
            and hasattr(table_id, "dataset_id")
            and hasattr(table_id, "table_id")
        ):
            return f"{table_id.project}.{table_id.dataset_id}.{table_id.table_id}"
        raise TypeError(f"Expected table_id to be a string, got {type(table_id)}")

    parts = table_id.split(".")
    if len(parts) < 3:
        raise ValueError("Invalid table ID")
    if len(parts) > 3 and not any(":" in part for part in parts[:-2]):
        raise TypeError("BigLake tables are not supported yet")

    # Let's just follow the rust regex logic:
    # it must have at least two dots, and the last two parts must not have dots.
    if len(parts) >= 3:
        return table_id

    raise ValueError("Invalid table ID")


def read_bigquery(
    table: str | None = None,
    *,
    query: str | None = None,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP | None = None,
    maintain_order: bool = False,
    user_agent: str | None = None,
) -> pl.DataFrame:
    if table is None and query is None:
        raise ValueError("Either table or query must be provided")
    if table is not None and query is not None:
        raise ValueError("Only one of table or query can be provided")

    if query is not None:
        return read_bigquery_query(
            query,
            quota_project_id=quota_project_id,
            credentials_provider=credentials_provider,
            maintain_order=maintain_order,
            user_agent=user_agent,
        )

    if not credentials_provider:
        credentials_provider = _get_default_provider(quota_project_id)

    table_ref = _parse_table_id(table)
    res = polars_bigquery.read_bigquery(
        table_ref, quota_project_id, maintain_order, credentials_provider, user_agent
    )
    return pl.DataFrame(res)


def read_bigquery_query(
    query: str,
    *,
    quota_project_id: str,
    credentials_provider: pl.CredentialProviderGCP | None = None,
    maintain_order: bool = False,
    user_agent: str | None = None,
) -> pl.DataFrame:
    if not credentials_provider:
        credentials_provider = _get_default_provider(quota_project_id)

    table = run_query(
        query,
        quota_project_id,
        credentials_provider,
        user_agent=user_agent,
    )
    table_ref = _parse_table_id(table)
    res = polars_bigquery.read_bigquery(
        table_ref, quota_project_id, maintain_order, credentials_provider, user_agent
    )
    return pl.DataFrame(res)
