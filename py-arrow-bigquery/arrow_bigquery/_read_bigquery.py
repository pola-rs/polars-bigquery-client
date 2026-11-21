from __future__ import annotations

import datetime
from typing import Any, Self

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
    """Client for reading BigQuery tables and query results as Arrow streams.

    Keeps the underlying gRPC/HTTP connection channels open across multiple
    read operations, and caches the OAuth2 token in Rust.

    Supports use as a context manager (``with Client(...) as client:``).
    Exiting the ``with`` block calls :meth:`close`, releasing the underlying
    native client and rejecting subsequent reads with :class:`RuntimeError`.
    Any lazy read initiated *after* exiting the ``with`` block will fail.
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
        """Close the client and release the underlying native client reference.

        Idempotent. Releasing the native client drops cached network channels
        and clears the Rust-side credentials reference once any previously
        returned ``ArrowStreamExporter`` streams finish or are dropped (note
        that an aborted/early-dropped incomplete query stream may spawn a
        detached best-effort ``jobs.cancel`` task that briefly retains the REST
        client and credentials provider for up to 10 s until it completes).

        Warning:
            Calling ``close()`` (or exiting a ``with Client(...) as client:``
            block) invalidates this ``Client`` for all future reads: any
            subsequent :meth:`read_table` or :meth:`read_query` call—including
            lazy/deferred reads started after exiting the ``with`` block—raises
            :class:`RuntimeError`.

        Thread safety:
            :meth:`read_table` and :meth:`read_query` snapshot the underlying
            native client reference once before starting I/O, so a concurrent
            ``close()`` will not interrupt an in-flight call or already-created
            ``ArrowStreamExporter``. Callers sharing a ``Client`` across threads
            should still finish dispatching reads before closing it.
        """
        native_client = self._client
        self._client = None
        if native_client is not None:
            native_client.close()

    @property
    def closed(self) -> bool:
        """Return ``True`` if :meth:`close` has been called on this client."""
        return self._client is None

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_open(self) -> arrow_bigquery._native.Client:
        native_client = self._client
        if native_client is None:
            msg = "Cannot use a closed arrow_bigquery.Client instance."
            raise RuntimeError(msg)
        return native_client

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
        """Execute a BigQuery SQL query (or dry run) as an Arrow stream.

        Args:
            query: GoogleSQL query string.
            arrow_buffer_compression: Arrow IPC buffer compression codec
                (``"lz4frame"``, ``"zstd"``, or ``"unspecified"``).
            dry_run: When ``True``, issue a dry-run request that returns a
                single-row metadata batch without executing or billing.
            maintain_order: Preserve query result ordering.
            use_query_cache: Override BigQuery query result caching.
            maximum_bytes_billed: Optional upper bound on bytes billed **per
                ``jobs.query`` attempt** (``0 <= maximum_bytes_billed <= (1 << 63) - 1``).
                When set, also forwarded on dry runs for consistent limit
                checking. Transient REST retries reuse a single ``request_id``,
                which guarantees deduplication for *mutating* (DML/DDL/script)
                queries within BigQuery's 15-minute window; however, per the
                BigQuery v2 API contract, read-only ``SELECT`` queries may ignore
                ``request_id`` and re-execute (and re-bill) on a mid-flight retry
                if not yet served from the query cache. Because
                ``maximum_bytes_billed`` applies per attempt rather than
                cumulatively across retries, keep this in mind for expensive
                uncached queries.

        Note:
            Interrupting a query via ``Ctrl+C`` (``KeyboardInterrupt``) triggers
            a detached, best-effort server-side ``jobs.cancel`` RPC once a job ID
            has been returned (``jobComplete=false``). If the Python process exits
            immediately upon an unhandled ``KeyboardInterrupt``, background
            threads may terminate before the detached ``jobs.cancel`` HTTP request
            flushes. Interrupting during the initial ``jobs.query`` HTTP wait
            (default 200 s) closes only the client socket because the
            server-generated job ID is not yet available. On non-main Python
            threads, blocking waits cannot be interrupted by ``Ctrl+C`` (CPython
            only delivers signals on the main thread).

        Raises:
            RuntimeError: If the client has already been closed via
                :meth:`close` or exited a ``with`` block.
            TypeError: If ``maximum_bytes_billed`` is a ``bool`` or not an
                ``int`` (or ``None``).
            ValueError: If ``maximum_bytes_billed`` is negative or exceeds
                signed 64-bit integer range (``(1 << 63) - 1``).
        """
        native_client = self._ensure_open()
        return native_client.read_query(
            query,
            arrow_buffer_compression=arrow_buffer_compression,
            dry_run=dry_run,
            maintain_order=maintain_order,
            use_query_cache=use_query_cache,
            maximum_bytes_billed=maximum_bytes_billed,
        )
