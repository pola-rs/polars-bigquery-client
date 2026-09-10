from __future__ import annotations

import datetime
from typing import Any

import arrow_bigquery._native
import arrow_bigquery.core.version
from arrow_bigquery.api.resources import parse_table_id

_parse_table_id = parse_table_id


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"arrow-bigquery/{arrow_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


class Client:
    """Client for reading BigQuery tables as Arrow streams.

    Keeps the underlying gRPC connection channel open across multiple read operations,
    and caches the OAuth2 token in Rust.
    """

    def __init__(
        self,
        *,
        quota_project_id: str,
        credentials_provider: Any = None,
        user_agent: str | None = None,
    ) -> None:
        self._quota_project_id = quota_project_id
        self._user_agent = user_agent
        full_user_agent = _get_user_agent(user_agent)
        self._client = arrow_bigquery._native.Client(
            quota_project_id=quota_project_id,
            credentials_provider=credentials_provider,
            user_agent=full_user_agent,
        )

    @property
    def quota_project_id(self) -> str:
        return self._quota_project_id

    def read_table(
        self,
        table: arrow_bigquery._native.BigQueryTableId | str | Any,
        *,
        arrow_buffer_compression: str = "lz4frame",
        maintain_order: bool = False,
        max_stream_count: int | None = None,
        row_restriction: str = "",
        sample_percentage: float | None = None,
        selected_fields: list[str] | None = None,
        snapshot_time: datetime.datetime | None = None,
    ) -> arrow_bigquery._native.ArrowStreamExporter:
        table_id = _parse_table_id(table)
        return self._client.read_table(
            table_id,
            arrow_buffer_compression=arrow_buffer_compression,
            maintain_order=maintain_order,
            max_stream_count=max_stream_count,
            row_restriction=row_restriction,
            sample_percentage=sample_percentage,
            selected_fields=selected_fields,
            snapshot_time=snapshot_time,
        )
