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
        bq_endpoint: str | None = None,
        bqstorage_endpoint: str | None = None,
    ) -> None:
        self._quota_project_id = quota_project_id
        self._user_agent = user_agent
        full_user_agent = _get_user_agent(user_agent)
        self._client: arrow_bigquery._native.Client | None = (
            arrow_bigquery._native.Client(
                quota_project_id=quota_project_id,
                credentials_provider=credentials_provider,
                user_agent=full_user_agent,
                bq_endpoint=bq_endpoint,
                bqstorage_endpoint=bqstorage_endpoint,
            )
        )

    def close(self) -> None:
        """Release the underlying native client and cached network channels."""
        self._client = None

    def __enter__(self) -> Client:  # noqa: PYI034
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_open(self) -> arrow_bigquery._native.Client:
        if self._client is None:
            msg = "Cannot use a closed arrow_bigquery.Client instance."
            raise RuntimeError(msg)
        return self._client

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
        native_client = self._ensure_open()
        table_id = _parse_table_id(table)
        return native_client.read_table(
            table_id,
            arrow_buffer_compression=arrow_buffer_compression,
            maintain_order=maintain_order,
            max_stream_count=max_stream_count,
            row_restriction=row_restriction,
            sample_percentage=sample_percentage,
            selected_fields=selected_fields,
            snapshot_time=snapshot_time,
        )

    def read_query(
        self,
        query: str,
        *,
        arrow_buffer_compression: str = "lz4frame",
        dry_run: bool = False,
        maintain_order: bool = True,
        use_query_cache: bool | None = None,
        maximum_bytes_billed: int | None = None,
    ) -> arrow_bigquery._native.ArrowStreamExporter:
        native_client = self._ensure_open()
        kwargs: dict[str, Any] = {
            "arrow_buffer_compression": arrow_buffer_compression,
            "dry_run": dry_run,
            "maintain_order": maintain_order,
            "use_query_cache": use_query_cache,
        }
        if maximum_bytes_billed is not None:
            if isinstance(maximum_bytes_billed, bool) or not isinstance(
                maximum_bytes_billed, int
            ):
                msg = (
                    "maximum_bytes_billed must be an int or None, "
                    f"got {type(maximum_bytes_billed).__name__}"
                )
                raise TypeError(msg)
            if maximum_bytes_billed < 0:
                msg = (
                    "maximum_bytes_billed must be non-negative, "
                    f"got {maximum_bytes_billed}"
                )
                raise ValueError(msg)
            kwargs["maximum_bytes_billed"] = maximum_bytes_billed
        return native_client.read_query(query, **kwargs)
