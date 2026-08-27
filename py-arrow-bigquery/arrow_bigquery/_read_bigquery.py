from __future__ import annotations

from typing import Any

import arrow_bigquery.core.version
import arrow_bigquery._native


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"arrow-bigquery/{arrow_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


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


class Client:
    """Client for reading BigQuery tables as Arrow streams.

    Keeps the underlying gRPC connection channel open across multiple read operations,
    and caches the OAuth2 token in Rust.
    """

    def __init__(
        self,
        *,
        credentials_provider: Any = None,
        user_agent: str | None = None,
    ) -> None:
        self._user_agent = user_agent
        full_user_agent = _get_user_agent(user_agent)
        self._client = arrow_bigquery._native.Client(
            credentials_provider=credentials_provider,
            user_agent=full_user_agent,
        )

    def read_table(
        self,
        table: Any,
        *,
        quota_project_id: str,
        maintain_order: bool = False,
        user_agent: str | None = None,
    ) -> arrow_bigquery._native.ArrowStreamExporter:
        user_agent = _get_user_agent(user_agent or self._user_agent)
        table_ref = _parse_table_id(table)
        return self._client.read_table(
            table_ref,
            quota_project_id,
            maintain_order,
            user_agent,
        )
