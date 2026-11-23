from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import arrow_bigquery
import arrow_bigquery.api.resources
import polars as pl
import polars.io.plugins
from polars.datatypes import DataType, DataTypeClass

import polars_bigquery.core.schema
import polars_bigquery.core.version
import polars_bigquery.exceptions
from polars_bigquery.core import bigquery_rest
from polars_bigquery.core.executors import (
    QueryExecutor,
    ReadApiExecutor,
    has_non_comment_sql,
)
from polars_bigquery.core.executors.base import close_exporter as _close_exporter
from polars_bigquery.core.plan import (
    ArrowBufferCompression,
    QueryScan,
    TableScan,
    build_io_source_plan,
    validate_maximum_bytes_billed,
)

logger = logging.getLogger(__name__)

_VALID_ARROW_BUFFER_COMPRESSIONS = frozenset({"lz4frame", "zstd", "unspecified"})
_EXPLAIN_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f\u2028\u2029]+")


def _sanitize_explain_value(value: str) -> str:
    """Normalize an `explain_detail` token value so output remains single-line."""
    return _EXPLAIN_CONTROL_CHARS_RE.sub(" ", value).strip()


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"polars-bigquery/{polars_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


def _validate_compression(arrow_buffer_compression: Any) -> None:
    if (
        not isinstance(arrow_buffer_compression, str)
        or arrow_buffer_compression not in _VALID_ARROW_BUFFER_COMPRESSIONS
    ):
        allowed = ", ".join(sorted(_VALID_ARROW_BUFFER_COMPRESSIONS))
        msg = (
            f"arrow_buffer_compression must be one of ({allowed}), "
            f"got {arrow_buffer_compression!r}"
        )
        raise ValueError(msg)


def _validate_query_str(query: Any) -> str:
    if not isinstance(query, str):
        msg = f"query must be a str, got {type(query).__name__}"
        raise TypeError(msg)
    return query


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{name} must be a bool, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _validate_optional_bool(value: Any, name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        msg = f"{name} must be a bool or None, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _normalize_schema(
    schema: pl.Schema | Mapping[str, DataType | DataTypeClass],
    *,
    require_non_empty: bool = False,
) -> pl.Schema:
    polars_schema = pl.Schema(schema)
    if require_non_empty and len(polars_schema) == 0:
        msg = "schema must contain at least one column"
        raise ValueError(msg)
    return polars_schema


def _dry_run_scalar(metadata: pl.DataFrame, column: str) -> Any:
    if column not in metadata.columns or metadata.height == 0:
        return None
    return metadata[column][0]


def _extract_bignumeric_columns(schema: pl.Schema) -> list[str]:
    return [
        name
        for name, dtype in schema.items()
        if polars_bigquery.core.schema.has_bignumeric_dtype(dtype)
    ]


def _register_io_source_safe(
    *,
    io_source: Any,
    schema: pl.Schema,
    explain_name: str,
    explain_detail: str,
) -> pl.LazyFrame:
    try:
        return polars.io.plugins.register_io_source(
            io_source=io_source,
            schema=schema,
            explain_name=explain_name,
            explain_detail=explain_detail,
        )
    except pl.exceptions.InvalidOperationError as exc:
        bignum_cols = _extract_bignumeric_columns(schema)
        if bignum_cols:
            cols_repr = ", ".join(repr(col) for col in bignum_cols)
            msg = (
                f"Polars does not support BigQuery BIGNUMERIC/BIGDECIMAL columns "
                f"(Decimal precision > 38) in LazyFrame schema for column(s): {cols_repr}. "
                "Cast them to NUMERIC, FLOAT64, or STRING in SQL or exclude them."
            )
            raise TypeError(msg) from exc
        raise


class Client:
    """Client for reading data from Google BigQuery into Polars.

    Wraps an `arrow_bigquery.Client` and a lazily created REST session to keep
    connections open and reuse credentials across multiple operations. Can be
    used as a context manager to automatically close both the REST session and
    the underlying `arrow_bigquery.Client` on exit; collect or sink any
    returned `LazyFrame` inside the `with Client(...) as client:` block.

    Examples
    --------
    ```python
    import polars as pl
    import polars_bigquery

    with polars_bigquery.Client(quota_project_id="my-project") as client:
        lf = client.scan_table("bigquery-public-data.usa_names.usa_1910_2013")
        df = (
            lf.filter((pl.col("state") == "WA") & (pl.col("year") >= 2000))
            .select(["name", "year", "number"])
            .collect()
        )
    ```
    """

    def __init__(
        self,
        *,
        quota_project_id: str,
        credentials_provider: pl.CredentialProviderGCP | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Initialize a BigQuery client.

        Parameters
        ----------
        quota_project_id
            Google Cloud project ID used for quota and billing of BigQuery API
            requests and query jobs.
        credentials_provider
            Polars GCP credential provider used to obtain OAuth2 bearer tokens.
            If `None`, a default `polars.CredentialProviderGCP(quota_project_id=quota_project_id)`
            instance is created using Application Default Credentials (ADC).
        user_agent
            Optional custom user-agent suffix appended to the client's default
            `polars-bigquery/<version>` user-agent header.
        """
        if credentials_provider is None:
            credentials_provider = pl.CredentialProviderGCP(
                quota_project_id=quota_project_id
            )
        self._credentials_provider = credentials_provider
        self._user_agent = _get_user_agent(user_agent)
        self._quota_project_id = quota_project_id
        self._rest_client_instance: bigquery_rest.BigQueryRestClient | None = None
        self._lifecycle_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._arrow_client = arrow_bigquery.Client(
            quota_project_id=quota_project_id,
            credentials_provider=credentials_provider,
            user_agent=self._user_agent,
        )

    @property
    def _rest_client(self) -> bigquery_rest.BigQueryRestClient:
        """Lazily initialize the REST client pool only when table metadata is needed."""
        with self._lifecycle_lock:
            if self._closed_event.is_set():
                msg = "Cannot use a closed polars_bigquery.Client instance."
                raise RuntimeError(msg)
            if self._rest_client_instance is None:
                self._rest_client_instance = bigquery_rest.BigQueryRestClient(
                    quota_project_id=self._quota_project_id,
                    credentials_provider=self._credentials_provider,
                    user_agent=self._user_agent,
                )
            return self._rest_client_instance

    @_rest_client.setter
    def _rest_client(self, client: bigquery_rest.BigQueryRestClient | None) -> None:
        with self._lifecycle_lock:
            self._rest_client_instance = client

    def _ensure_open(self) -> None:
        if self._closed_event.is_set():
            msg = "Cannot use a closed polars_bigquery.Client instance."
            raise RuntimeError(msg)

    @property
    def closed(self) -> bool:
        """Return `True` if [`close`][polars_bigquery.Client.close] has been called."""
        return self._closed_event.is_set()

    def close(self) -> None:
        """Close the underlying REST session and native Arrow client.

        Idempotent. Releases pooled REST connections (if allocated), cached
        native gRPC/HTTP channels, and the Rust-side credentials provider
        reference.

        Calling `close()` (or exiting a `with Client(...) as client:` block)
        invalidates this `Client` for subsequent operations: later calls to
        [`read_table`][polars_bigquery.Client.read_table],
        [`read_query`][polars_bigquery.Client.read_query],
        [`scan_table`][polars_bigquery.Client.scan_table], or
        [`scan_query`][polars_bigquery.Client.scan_query]—including collecting a
        `LazyFrame` created before closing—raise `RuntimeError`.
        """
        with self._lifecycle_lock:
            if self._closed_event.is_set():
                return
            self._closed_event.set()
            rest_client = self._rest_client_instance
            self._rest_client_instance = None
            arrow_client = self._arrow_client
        try:
            if rest_client is not None:
                rest_client.close()
        finally:
            close_arrow = getattr(arrow_client, "close", None)
            if callable(close_arrow):
                close_arrow()

    def __enter__(self) -> Client:  # noqa: PYI034
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def credentials_provider(self) -> pl.CredentialProviderGCP:
        """The `polars.CredentialProviderGCP` used to authenticate requests."""
        return self._credentials_provider

    @property
    def quota_project_id(self) -> str:
        """The Google Cloud project ID used for quota and billing."""
        return self._quota_project_id

    def _dry_run_query(
        self,
        query: str,
        *,
        arrow_buffer_compression: ArrowBufferCompression,
    ) -> pl.DataFrame:
        # Never forward maximum_bytes_billed on scan_query schema discovery:
        # dry runs are unbilled, and downstream .filter()/.select() pushdown
        # may prune the query below the caller's byte cap at .collect() time.
        arrow_stream_exporter = self._arrow_client.read_query(
            query,
            arrow_buffer_compression=arrow_buffer_compression,
            dry_run=True,
            maintain_order=True,
            use_query_cache=None,
            maximum_bytes_billed=None,
        )
        try:
            return pl.DataFrame(arrow_stream_exporter)
        except BaseException:
            _close_exporter(arrow_stream_exporter)
            raise

    def read_table(
        self,
        table: Any,
        *,
        maintain_order: bool = False,
    ) -> pl.DataFrame:
        """Eagerly read a BigQuery table into a Polars `DataFrame`.

        Reads the entire table directly via the BigQuery Storage Read API. For
        projection and filter pushdown, prefer [`Client.scan_table`][polars_bigquery.Client.scan_table].

        Parameters
        ----------
        table
            The BigQuery table to read. Accepts a `"project.dataset.table"`
            string, an `arrow_bigquery.BigQueryTableId`, or any object with
            `project` (or `project_id`), `dataset_id`, and `table_id` attributes
            (such as `google.cloud.bigquery.TableReference`).
        maintain_order
            Whether to preserve table row order by reading from a single
            Storage Read API stream. Defaults to `False`, which enables parallel
            multi-stream reads for higher throughput.

        Returns
        -------
        pl.DataFrame
            A Polars `DataFrame` containing the table data.

        Examples
        --------
        ```python
        df = client.read_table("bigquery-public-data.utility_us.country_code_iso")
        ```
        """
        self._ensure_open()
        _validate_bool(maintain_order, "maintain_order")
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_table(
            table_ref,
            maintain_order=maintain_order,
        )
        try:
            return pl.DataFrame(arrow_stream_exporter)
        except BaseException:
            _close_exporter(arrow_stream_exporter)
            raise

    def read_query(
        self,
        query: str,
        *,
        arrow_buffer_compression: ArrowBufferCompression = "lz4frame",
        dry_run: bool = False,
        maintain_order: bool = True,
        use_query_cache: bool | None = None,
        maximum_bytes_billed: int | None = None,
    ) -> pl.DataFrame:
        """Execute a GoogleSQL query and read the result into a Polars `DataFrame`.

        Executes the query via the BigQuery `jobs.query` API with
        `JOB_CREATION_OPTIONAL` and streams the Arrow results via the BigQuery
        Storage Read API.

        The whole result is read into memory. For lazy streaming of query
        results up to about 10 GB compressed, prefer
        [`Client.scan_query`][polars_bigquery.Client.scan_query]. For results
        over ~10 GB compressed (or materialized results scanned multiple times),
        see [Scanning large query results](../guides/large-query-results.md).

        Parameters
        ----------
        query
            GoogleSQL query string to execute in BigQuery.
        arrow_buffer_compression
            Compression codec for Arrow record batch buffers (`"lz4frame"`,
            `"zstd"`, or `"unspecified"`). Defaults to `"lz4frame"`.
        dry_run
            If `True`, validates the query without executing it and returns a
            1-row `DataFrame` containing job statistics and schema metadata
            (`bigquerySchema`, `projectId`, `jobId`, `location`,
            `totalBytesProcessed`, `cacheHit`, `statementType`, `creationTime`).
            Defaults to `False`.
        maintain_order
            Whether to preserve the row order produced by an `ORDER BY` clause
            by reading from a single Storage Read API stream. Defaults to
            `True`. Set to `False` to enable adaptive parallel reads for large
            result sets when row order does not matter.
        use_query_cache
            Whether to look for the result in the query cache. If `None`, the
            BigQuery API default (`True`) is used.
        maximum_bytes_billed
            Optional ceiling on bytes billed **per `jobs.query` attempt**
            (`0 <= maximum_bytes_billed <= 2**63 - 1`). Queries whose estimated
            billed bytes exceed this limit fail in BigQuery without incurring a
            charge. When `dry_run=True`, this limit is also forwarded to
            BigQuery. Note that because BigQuery may re-execute an uncached
            read-only `SELECT` on a transient retry, this limit applies per
            attempt rather than cumulatively across retries. Defaults to `None`
            (no client-side limit).

        Returns
        -------
        pl.DataFrame
            A Polars `DataFrame` containing the query results (or dry-run
            metadata when `dry_run=True`).

        Raises
        ------
        ValueError
            If `arrow_buffer_compression` is invalid, or if
            `maximum_bytes_billed` is negative or exceeds `(1 << 63) - 1`.
        TypeError
            If `query`, `dry_run`, `maintain_order`, `use_query_cache`, or
            `maximum_bytes_billed` have invalid types.
        polars_bigquery.exceptions.BigQueryError
            If the BigQuery query fails.
        RuntimeError
            If called on a closed `Client` instance.

        Examples
        --------
        ```python
        df = client.read_query(
            \"\"\"
            SELECT name, SUM(number) AS total_born
            FROM `bigquery-public-data.usa_names.usa_1910_2013`
            GROUP BY name
            ORDER BY total_born DESC
            LIMIT 100
            \"\"\"
        )
        ```
        """
        self._ensure_open()
        query = _validate_query_str(query)
        _validate_compression(arrow_buffer_compression)
        _validate_bool(dry_run, "dry_run")
        _validate_bool(maintain_order, "maintain_order")
        _validate_optional_bool(use_query_cache, "use_query_cache")
        validate_maximum_bytes_billed(maximum_bytes_billed)
        arrow_stream_exporter = self._arrow_client.read_query(
            query,
            arrow_buffer_compression=arrow_buffer_compression,
            dry_run=dry_run,
            maintain_order=maintain_order,
            use_query_cache=use_query_cache,
            maximum_bytes_billed=maximum_bytes_billed,
        )
        try:
            return pl.DataFrame(arrow_stream_exporter)
        except BaseException:
            _close_exporter(arrow_stream_exporter)
            raise

    def scan_table(
        self,
        table: Any,
        *,
        schema: pl.Schema | Mapping[str, DataType | DataTypeClass] | None = None,
    ) -> pl.LazyFrame:
        """Lazily scan a BigQuery table into a Polars `LazyFrame`.

        Fetches table schema and partitioning metadata via the BigQuery REST API
        when constructing the `LazyFrame` (unless `schema` is provided), and
        defers reading rows from the BigQuery Storage Read API until query
        execution (`.collect()`, `.collect_batches()`, or a sink). Collect or
        sink the returned `LazyFrame` inside the `with Client(...) as client:`
        block.

        Supports both **projection pushdown** (only reading referenced columns)
        and **row predicate pushdown** (translating `.filter()` expressions into
        BigQuery Storage Read API `row_restriction` SQL clauses). See the
        [Row predicate pushdown](predicate-pushdown.md) page for the complete
        list of supported Polars expressions and data types.

        For ingestion-time partitioned tables, `_PARTITIONDATE` (`pl.Date`) and
        `_PARTITIONTIME` (`pl.Datetime("us", "utc")`) pseudo-columns are
        automatically added to the `LazyFrame` schema when `schema` is not
        provided, so partition pruning filters can be pushed down.

        Parameters
        ----------
        table
            The BigQuery table to scan. Accepts a `"project.dataset.table"`
            string, an `arrow_bigquery.BigQueryTableId`, or any object with
            `project` (or `project_id`), `dataset_id`, and `table_id` attributes
            (such as `google.cloud.bigquery.TableReference`).
        schema
            Optional non-empty Polars schema (`polars.Schema` or a mapping of
            column names to Polars data types) for the table. When provided,
            `scan_table` avoids calling the BigQuery `tables.get` REST API to
            fetch table metadata. For ingestion-time partitioned tables, include
            `"_PARTITIONDATE": pl.Date` and/or
            `"_PARTITIONTIME": pl.Datetime("us", "utc")` in `schema` if you want
            to filter or select those pseudo-columns.

        Returns
        -------
        pl.LazyFrame
            A Polars `LazyFrame` backed by the BigQuery Storage Read API.

        Raises
        ------
        ValueError
            If an explicit `schema` is empty, or if the table metadata contains
            exact duplicate column names.
        polars_bigquery.exceptions.BigQueryError
            If `schema` is `None` and fetching the table metadata from the
            BigQuery REST API fails or returns an empty field list.
        TypeError
            If the table metadata contains an unsupported BigQuery field type,
            or if an explicit `schema` contains an unspecified Polars data type
            class (such as bare `pl.Datetime`).

        Examples
        --------
        ```python
        import polars as pl

        lf = client.scan_table("bigquery-public-data.usa_names.usa_1910_2013")
        df = (
            lf.filter(pl.col("name").str.starts_with("A") & (pl.col("year") >= 2010))
            .select(["name", "state", "year", "number"])
            .collect()
        )
        ```
        """
        self._ensure_open()
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        if schema is None:
            table_metadata = self._rest_client.get_table_metadata(table_ref)
            polars_schema = polars_bigquery.core.schema.extract_polars_schema(
                table_metadata,
                require_non_empty=True,
            )
            schema_mode = "metadata"
        else:
            polars_schema = _normalize_schema(schema, require_non_empty=True)
            schema_mode = "explicit"

        source = TableScan(
            table_ref=table_ref,
            schema=polars_schema,
            maintain_order=False,
        )
        executor = ReadApiExecutor(self._arrow_client)
        explain_detail = f"table={table_ref}, schema={schema_mode}"
        closed_event = self._closed_event

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            if closed_event.is_set():
                msg = "Cannot use a closed polars_bigquery.Client instance."
                raise RuntimeError(msg)
            plan = build_io_source_plan(
                source,
                with_columns=with_columns,
                predicate=predicate,
                n_rows=n_rows,
            )
            return executor.execute(plan, batch_size=batch_size)

        return _register_io_source_safe(
            io_source=source_generator,
            schema=polars_schema,
            explain_name="BIGQUERY_TABLE",
            explain_detail=explain_detail,
        )

    def scan_query(
        self,
        query: str,
        *,
        schema: pl.Schema | Mapping[str, DataType | DataTypeClass] | None = None,
        arrow_buffer_compression: ArrowBufferCompression = "lz4frame",
        maintain_order: bool = True,
        use_query_cache: bool | None = None,
        maximum_bytes_billed: int | None = None,
    ) -> pl.LazyFrame:
        """Lazily execute a GoogleSQL `SELECT` query into a Polars `LazyFrame`.

        Discovers the query's result schema with a single verbatim BigQuery dry
        run when constructing the `LazyFrame` (unless `schema` is provided),
        and defers running the query via `jobs.query` (`JOB_CREATION_OPTIONAL`)
        until `.collect()`, `.collect_batches()`, or a sink is invoked. Streams
        results up to BigQuery's ~10 GB compressed anonymous destination table
        limit; for larger results or repeated reuse of a materialized table, see
        [Scanning large query results](../guides/large-query-results.md).
        Collect or sink the returned `LazyFrame` inside the
        `with Client(...) as client:` block.

        When `maintain_order=True` (the default), pushdown is disabled, `query`
        is sent verbatim (preserving leading/trailing whitespace, trailing
        semicolons, `ORDER BY` ordering, and exact-string BigQuery query cache
        keys), and results stream sequentially over a single stream. Note that
        `LazyFrame.explain()` prints Polars `SELECTION:` and `PROJECT:` lines
        under `PYTHON[BIGQUERY_QUERY] SCAN` even when `maintain_order=True`
        keeps those operations client-side (`pushdown=disabled`).

        When `maintain_order=False`, adaptive parallel reads are enabled and
        supported projections, [row predicate pushdown](predicate-pushdown.md),
        and `LIMIT` slices (`n_rows`) are appended using GoogleSQL pipe syntax
        (`|> WHERE`, `|> SELECT`, `|> LIMIT`). Leading whitespace and newlines
        are preserved so BigQuery `[line:col]` error coordinates stay unshifted.
        Projected columns are emitted in schema declaration order so reordered
        selections share the same BigQuery query cache entry.

        Important caveats under `maintain_order=False`:

        - **Explicit `schema` vs. dry run (`schema=None`)**: Providing `schema`
          skips the construction-time dry run, `SELECT`-only `statementType`
          guard (so DDL/DML/scripts would execute at `.collect()`), and
          duplicate field checks; supplied dtypes are trusted unchecked
          (`validate_schema=False`). Invalid SQL raises
          `polars.exceptions.ComputeError` at `.collect()` instead of
          `BigQueryError` at construction time. For `schema=None`, `|> SELECT`
          is omitted when all discovered columns are required. For an explicit
          `schema` (which may be a **partial** subset of the query's columns),
          `|> SELECT` is **always** emitted so unmentioned query columns are
          still pruned server-side.
        - **Anonymous (`f0_`) and duplicate / case-colliding columns**: For
          `schema=None`, all-column scans omit `|> SELECT` so unaliased columns
          (`f0_`, `f1_`, ...) flow through unfiltered or filtered/sliced scans.
          Explicitly projecting or filtering on an anonymous column (or using
          an explicit `schema` naming `f0_`) fails at `.collect()` with
          `Unrecognized name: f0_`. Duplicate (`SELECT 1 AS a, 2 AS a`) or
          case-colliding (`SELECT 1 AS a, 2 AS A`) aliases are auto-suffixed by
          BigQuery to `a`/`a_1` and `a`/`A_1`; filtering on `a` fails as
          ambiguous, and selecting `a_1`/`A_1` raises `Unrecognized name`.
          Always give output columns unique case-insensitive `AS` aliases, or
          keep `maintain_order=True`.
        - **Trailing comments after a semicolon**: A single bare statement-terminating
          `;` is stripped when appending pipe operators, including when followed by
          trailing whitespace or GoogleSQL comments (`; -- note`, `; # note`,
          `; /* note */`), while **preserving** the trailing comments themselves
          (e.g., SQLCommenter/dbt tracing tags). Unterminated block comments
          (`; /* ...`) and invalid double semicolons (`;;`) retain `;` so BigQuery
          rejects them as syntax errors.
        - **`lf.filter(p).limit(n)` vs. `lf.limit(n)`**: Because Polars retains
          a post-filter `.limit(n)` above the I/O scan node, BigQuery executes
          the query with `|> WHERE` and *without* SQL `LIMIT`. The full filtered
          query executes and bills, and remains subject to the ~10 GB anonymous
          table cap. In the initial `jobs.query` wait (~200 s default), Python
          signals (`Ctrl+C`) interrupt the client wait promptly, though queries
          interrupted *before* BigQuery returns a job reference continue running
          server-side; incomplete jobs dropped after a job reference is received
          trigger a best-effort `jobs.cancel`. Put a `LIMIT` inside the SQL query
          or use `lf.limit(n)` when a server-side row cap is needed.
        - **Read API / Query length guard**: Oversized `row_restriction` filters
          in `scan_table` (and `|> WHERE` clauses in `scan_query`) degrade
          automatically in tiers (pruning largest top-level `AND` conjuncts first)
          and log one `WARNING` without exposing raw SQL literals.
        - **Polars `2.0.0–2.1.0` upstream note**: In `polars 2.0.0–2.1.0`, two
          upstream `pl.scan_arrow_c_stream` issues affect default
          `collect_batches()` execution: `lf.limit(n).filter(p)` may evaluate
          `filter` before `limit`
          ([pola-rs/polars#30916](https://github.com/pola-rs/polars/issues/30916)),
          and zero-column `lf.drop(pl.all()).collect()` can yield zero rows.
          Both issues are fixed on the upstream Polars `main` branch.

        Collection-time errors from BigQuery, query compilation, or mid-stream
        readers (including exceeding ~10 GB compressed, disallowed column
        identifiers, or SQL errors when `schema` is explicit) may be propagated
        directly (e.g., `BigQueryError`, `ValueError`) or wrapped in
        `polars.exceptions.ComputeError` depending on the Polars version and
        whether the failure occurs inside the Arrow C Stream reader.

        Parameters
        ----------
        query
            GoogleSQL `SELECT` query.
        schema
            Optional non-empty Polars schema (`polars.Schema` or a mapping of
            column names to Polars data types) for the query result. Passing a
            full or partial `schema` skips the construction-time dry-run RPC and
            its `SELECT`-only guard. Under `maintain_order=False`, `|> SELECT`
            is always emitted for the needed columns declared in `schema`.
        arrow_buffer_compression
            Compression codec for Arrow record batch buffers (`"lz4frame"`,
            `"zstd"`, or `"unspecified"`). Defaults to `"lz4frame"`.
        maintain_order
            Whether to preserve query result order by reading sequentially over
            a single stream and executing `query` verbatim without pushdown.
            Defaults to `True`. Set to `False` to enable adaptive parallel
            reads and GoogleSQL pipe pushdown (`|> WHERE`, `|> SELECT`,
            `|> LIMIT`) when row order does not matter.
        use_query_cache
            Whether BigQuery should use the query cache. If `None`, BigQuery's
            API default (`True`) is used.
        maximum_bytes_billed
            Optional ceiling on bytes billed **per `jobs.query` attempt** at
            execution time (`0 <= maximum_bytes_billed <= 2**63 - 1`). Not
            applied to the construction-time dry run used for schema discovery
            (`schema=None`) so downstream `.filter()` / `.select()` pushdown can
            prune the query below this limit before BigQuery executes it. Useful
            as a cost guardrail when `maintain_order=False` degrades oversized
            predicates locally. Because BigQuery may re-execute an uncached
            read-only `SELECT` on a transient retry, this limit applies per
            attempt rather than cumulatively across retries. Defaults to `None`
            (no client-side limit).

        Returns
        -------
        pl.LazyFrame
            A lazy query scan backed by BigQuery `jobs.query` and the Storage
            Read API.

        Raises
        ------
        ValueError
            If `arrow_buffer_compression` is invalid, `query` is empty (or contains
            only comments/semicolons), `maximum_bytes_billed` is negative or
            exceeds `(1 << 63) - 1`, an explicit `schema` is empty, the dry run
            reports a non-`SELECT` statement type, or the dry-run result schema
            contains exact duplicate column names (or duplicate nested `STRUCT`
            subfield names).
        polars_bigquery.exceptions.BigQueryError
            If the BigQuery dry run fails (for example, due to invalid SQL,
            unshifted `[line:col]` positions on `query`) or returns a missing,
            malformed, or empty schema response.
        TypeError
            If `query`, `maintain_order`, `use_query_cache`, or
            `maximum_bytes_billed` have invalid types, if the dry-run result
            schema contains an unsupported BigQuery field type (including
            `BIGNUMERIC` / `BIGDECIMAL`), or if an explicit `schema` contains an
            unspecified Polars data type class (such as bare `pl.Datetime`).
        RuntimeError
            If called on a closed `Client` instance.

        Examples
        --------
        ```python
        import polars as pl

        lf = client.scan_query(
            \"\"\"
            SELECT name, state, year, number
            FROM `bigquery-public-data.usa_names.usa_1910_2013`
            \"\"\",
            maintain_order=False,
        )
        df = (
            lf.filter((pl.col("state") == "WA") & (pl.col("year") >= 2000))
            .select(["name", "year", "number"])
            .limit(10)
            .collect()
        )
        ```
        """
        self._ensure_open()
        _validate_compression(arrow_buffer_compression)
        raw_query = _validate_query_str(query)
        _validate_bool(maintain_order, "maintain_order")
        _validate_optional_bool(use_query_cache, "use_query_cache")
        validate_maximum_bytes_billed(maximum_bytes_billed)

        if not has_non_comment_sql(raw_query):
            raise ValueError("query must be a non-empty SELECT statement")

        if schema is not None:
            polars_schema = _normalize_schema(schema, require_non_empty=True)
            location: Any = None
            schema_mode = "explicit"
        else:
            metadata = self._dry_run_query(
                raw_query,
                arrow_buffer_compression=arrow_buffer_compression,
            )
            if (
                metadata.height == 0
                or "statementType" not in metadata.columns
                or "bigquerySchema" not in metadata.columns
            ):
                msg = (
                    "Invalid response from BigQuery: dry run returned an "
                    "incomplete metadata payload."
                )
                raise polars_bigquery.exceptions.BigQueryError(msg)

            statement_type = metadata["statementType"][0]
            if statement_type not in (None, "SELECT"):
                msg = f"scan_query only supports SELECT statements, got {statement_type!r}"
                raise ValueError(msg)

            schema_json = metadata["bigquerySchema"][0]
            if schema_json is None:
                if statement_type == "SELECT":
                    msg = (
                        "Invalid response from BigQuery: dry run for SELECT "
                        "query did not return a result schema."
                    )
                else:
                    msg = (
                        "Invalid response from BigQuery: dry run did not return "
                        "a statement type or result schema."
                    )
                raise polars_bigquery.exceptions.BigQueryError(msg)

            if not isinstance(schema_json, str):
                msg = (
                    "Invalid response from BigQuery: dry run 'bigquerySchema' "
                    f"must be a JSON string, got {type(schema_json).__name__}."
                )
                raise polars_bigquery.exceptions.BigQueryError(msg)

            try:
                parsed_schema = json.loads(schema_json)
            except json.JSONDecodeError as exc:
                msg = "Invalid response from BigQuery: dry run returned malformed schema JSON."
                raise polars_bigquery.exceptions.BigQueryError(msg) from exc

            if not isinstance(parsed_schema, Mapping):
                msg = (
                    "Invalid response from BigQuery: dry run schema JSON must "
                    f"be an object, got {type(parsed_schema).__name__}."
                )
                raise polars_bigquery.exceptions.BigQueryError(msg)

            fields = parsed_schema.get("fields")
            if fields is not None and (
                not isinstance(fields, Sequence) or isinstance(fields, (str, bytes))
            ):
                msg = (
                    "Invalid response from BigQuery: dry run schema 'fields' "
                    f"must be a list, got {type(fields).__name__}."
                )
                raise polars_bigquery.exceptions.BigQueryError(msg)
            if not fields:
                msg = (
                    "Invalid response from BigQuery: dry run for SELECT "
                    "query returned an empty schema."
                )
                raise polars_bigquery.exceptions.BigQueryError(msg)

            polars_schema = (
                polars_bigquery.core.schema.extract_polars_schema_from_fields(fields)
            )
            location = _dry_run_scalar(metadata, "location")
            if logger.isEnabledFor(logging.DEBUG):
                total_bytes_processed = _dry_run_scalar(metadata, "totalBytesProcessed")
                cache_hit = _dry_run_scalar(metadata, "cacheHit")
                logger.debug(
                    "Dry run completed for scan_query: statement_type=%s, location=%s, "
                    "total_bytes_processed=%s, cache_hit=%s, columns=%d",
                    statement_type,
                    location,
                    total_bytes_processed,
                    cache_hit,
                    len(polars_schema),
                )
            schema_mode = "dry_run"

        pushdown_state = "disabled" if maintain_order else "enabled"
        explain_parts = [
            f"maintain_order={maintain_order}",
            f"pushdown={pushdown_state}",
            f"schema={schema_mode}",
        ]
        if isinstance(location, str):
            safe_location = _sanitize_explain_value(location)
            if safe_location:
                explain_parts.append(f"location={safe_location}")
        if maximum_bytes_billed is not None:
            explain_parts.append(f"maximum_bytes_billed={maximum_bytes_billed}")
        explain_detail = ", ".join(explain_parts)

        source = QueryScan(
            raw_query=raw_query,
            columns=tuple(polars_schema.keys()),
            maintain_order=maintain_order,
            arrow_buffer_compression=arrow_buffer_compression,
            use_query_cache=use_query_cache,
            explicit_schema=schema is not None,
            maximum_bytes_billed=maximum_bytes_billed,
        )
        executor = QueryExecutor(self._arrow_client)
        closed_event = self._closed_event

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            if closed_event.is_set():
                msg = "Cannot use a closed polars_bigquery.Client instance."
                raise RuntimeError(msg)
            plan = build_io_source_plan(
                source,
                with_columns=with_columns,
                predicate=predicate,
                n_rows=n_rows,
            )
            return executor.execute(plan, batch_size=batch_size)

        return _register_io_source_safe(
            io_source=source_generator,
            schema=polars_schema,
            explain_name="BIGQUERY_QUERY",
            explain_detail=explain_detail,
        )
