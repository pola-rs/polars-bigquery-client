from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import arrow_bigquery
import arrow_bigquery.api.resources
import polars as pl
import polars.io.plugins
from polars.datatypes import DataType, DataTypeClass

import polars_bigquery.core.schema
import polars_bigquery.core.types
import polars_bigquery.core.version
import polars_bigquery.exceptions
from polars_bigquery.core import bigquery_rest, compiler

logger = logging.getLogger(__name__)

_MAX_SKIPPED_COLUMNS_DISPLAY = 5
# Cap `read_table` fanout to a single stream when Polars pushes a small row
# limit (`n_rows`), avoiding spinning up `available_parallelism()` parallel gRPC
# streams and prefetch buffers only to discard nearly all rows.
_SINGLE_STREAM_ROW_LIMIT = 10_000


def _format_skipped_columns(
    skipped_columns: Sequence[polars_bigquery.core.schema.SkippedColumn],
) -> str:
    """Format omitted top-level columns with reasons/paths, capped at 5 entries."""
    entries = [
        f"{col.name} ({col.reason})"
        if col.path == col.name
        else f"{col.name} ({col.reason} at {col.path})"
        for col in skipped_columns[:_MAX_SKIPPED_COLUMNS_DISPLAY]
    ]
    if len(skipped_columns) > _MAX_SKIPPED_COLUMNS_DISPLAY:
        entries.append(
            f"... and {len(skipped_columns) - _MAX_SKIPPED_COLUMNS_DISPLAY} more"
        )
    return ", ".join(entries)


def _drop_columns_iter(batches: Iterator[pl.DataFrame]) -> Iterator[pl.DataFrame]:
    """Yield height-preserving zero-width frames and close the iterator if supported.

    Note: Polars 2.x's ``_CollectBatches`` iterator does not expose a ``.close()``
    method; its underlying native stream is released when this generator finishes
    or is garbage-collected. Calling ``.close()`` when present ensures prompt
    cleanup for generator-backed or custom iterators.
    """
    try:
        for batch in batches:
            yield batch.drop(batch.columns)
    finally:
        close = getattr(batches, "close", None)
        if callable(close):
            close()


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"polars-bigquery/{polars_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


class Client:
    """Client for reading data from Google BigQuery into Polars.

    Wraps an `arrow_bigquery.Client` and a REST session to keep connections open
    and reuse credentials across multiple operations. Can be used as a context
    manager to automatically close the underlying REST session on exit.

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
        self._rest_client = bigquery_rest.BigQueryRestClient(
            quota_project_id=quota_project_id,
            credentials_provider=credentials_provider,
            user_agent=self._user_agent,
        )
        try:
            self._arrow_client = arrow_bigquery.Client(
                quota_project_id=quota_project_id,
                credentials_provider=credentials_provider,
                user_agent=self._user_agent,
            )
        except Exception:
            self._rest_client.close()
            raise

    def close(self) -> None:
        """Close underlying REST session resources."""
        self._rest_client.close()

    def __enter__(self) -> Client:  # noqa: PYI034
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

    def read_table(
        self,
        table: Any,
        *,
        maintain_order: bool = False,
    ) -> pl.DataFrame:
        """Eagerly read a BigQuery table into a Polars `DataFrame`.

        Reads the entire table directly via the BigQuery Storage Read API. For
        projection and filter pushdown, prefer [`Client.scan_table`][polars_bigquery.Client.scan_table].

        Unlike [`Client.scan_table`][polars_bigquery.Client.scan_table] (which
        omits unsupported BigQuery columns with an
        [`UnsupportedTypeWarning`][polars_bigquery.exceptions.UnsupportedTypeWarning]),
        `read_table` streams all table columns directly from the Storage Read
        API. Tables containing BigQuery `INTERVAL` or `BIGNUMERIC` columns fail
        immediately with [`UnsupportedTypeError`][polars_bigquery.exceptions.UnsupportedTypeError]
        (a subclass of [`BigQueryError`][polars_bigquery.exceptions.BigQueryError])
        before Arrow decoding / FFI export. See [Data types](data-types.md) for the complete mapping and
        [`Client.read_query`][polars_bigquery.Client.read_query] workarounds.

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

        Raises
        ------
        polars_bigquery.exceptions.BigQueryError
            If the Storage Read API call fails or the table contains an Arrow
            type rejected prior to FFI export (such as BigQuery `INTERVAL`).

        Examples
        --------
        ```python
        df = client.read_table("bigquery-public-data.utility_us.country_code_iso")
        ```
        """
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_table(
            table_ref,
            maintain_order=maintain_order,
        )
        return pl.DataFrame(arrow_stream_exporter)

    def read_query(
        self,
        query: str,
        *,
        arrow_buffer_compression: str = "lz4frame",
        dry_run: bool = False,
        maintain_order: bool = True,
        use_query_cache: bool | None = None,
    ) -> pl.DataFrame:
        """Execute a GoogleSQL query and read the result into a Polars `DataFrame`.

        Executes the query via the BigQuery `jobs.query` API with
        `JOB_CREATION_OPTIONAL` and streams the Arrow results via the BigQuery
        Storage Read API.

        The whole result is read into memory. To process larger results lazily,
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
            `True`. Set to `False` to switch to parallel multi-stream reads for
            large result sets when row order does not matter.
        use_query_cache
            Whether to look for the result in the query cache. If `None`, the
            BigQuery API default (`True`) is used.

        Returns
        -------
        pl.DataFrame
            A Polars `DataFrame` containing the query results (or dry-run
            metadata when `dry_run=True`).

        Raises
        ------
        polars_bigquery.exceptions.BigQueryError
            If the BigQuery query fails.

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
        arrow_stream_exporter = self._arrow_client.read_query(
            query,
            arrow_buffer_compression=arrow_buffer_compression,
            dry_run=dry_run,
            maintain_order=maintain_order,
            use_query_cache=use_query_cache,
        )
        return pl.DataFrame(arrow_stream_exporter)

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
        execution (`.collect()` or `.collect_batches()`).

        Supports both **projection pushdown** (only reading referenced columns)
        and **schema-aware row predicate pushdown** (translating `.filter()`
        expressions into BigQuery Storage Read API `row_restriction` SQL
        clauses). See [Data types](data-types.md) for the BigQuery-to-Polars
        schema mapping (including `TIMESTAMP` `"UTC"`, `JSON`/`GEOGRAPHY`
        extension types, and unsupported columns) and
        [Row predicate pushdown](predicate-pushdown.md) for the supported
        expressions per type.

        When `schema` is `None`, top-level columns whose type (or any nested
        `STRUCT`/`ARRAY` subfield) is unsupported (`BIGNUMERIC`, `INTERVAL`,
        `RANGE<T>`, picosecond `TIMESTAMP`, or empty/unnamed/duplicate `STRUCT`)
        are omitted with an
        [`UnsupportedTypeWarning`][polars_bigquery.exceptions.UnsupportedTypeWarning],
        rather than failing the entire scan as [`Client.read_table`][polars_bigquery.Client.read_table]
        can.

        For ingestion-time partitioned tables, `_PARTITIONTIME`
        (`pl.Datetime("us", "UTC")`) is always added to the `LazyFrame` schema
        when `schema` is `None`; `_PARTITIONDATE` (`pl.Date`) is additionally
        added when the table uses daily (`DAY`) partitioning.

        Parameters
        ----------
        table
            The BigQuery table to scan. Accepts a `"project.dataset.table"`
            string, an `arrow_bigquery.BigQueryTableId`, or any object with
            `project` (or `project_id`), `dataset_id`, and `table_id` attributes
            (such as `google.cloud.bigquery.TableReference`).
        schema
            Optional Polars schema (`polars.Schema` or a mapping of column names
            to Polars data types) for the table. When provided, `scan_table`
            avoids calling the BigQuery `tables.get` REST API to fetch table
            metadata. For ingestion-time partitioned tables, include
            `"_PARTITIONTIME": pl.Datetime("us", "UTC")` and (for `DAY`
            partitions) `"_PARTITIONDATE": pl.Date` in `schema` if you want to
            filter or select those pseudo-columns. For `JSON` and `GEOGRAPHY`
            columns, use [`polars_bigquery.dtypes.JSON`][polars_bigquery.dtypes.JSON]
            and [`polars_bigquery.dtypes.GEOGRAPHY`][polars_bigquery.dtypes.GEOGRAPHY]
            (or a registered `polars.BaseExtension` subclass). Loading
            extension types as plain storage (`as_storage=True` or
            `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`) is not
            supported with an explicit `schema` because `pl.String` cannot be
            distinguished from BigQuery `STRING` without table metadata.
            Passing an empty mapping (`schema={}`) skips both `tables.get` and
            `CreateReadSession`, yielding an empty 0-row, 0-column `DataFrame`.

        Returns
        -------
        pl.LazyFrame
            A Polars `LazyFrame` backed by the BigQuery Storage Read API.

        Warns
        -----
        polars_bigquery.exceptions.UnsupportedTypeWarning
            If `schema` is `None` and one or more unsupported columns are
            omitted while at least one supported column (or partition
            pseudo-column) remains.

        Raises
        ------
        polars_bigquery.exceptions.BigQueryError
            If `schema` is `None` and fetching the table metadata from the
            BigQuery REST API fails, or if reading from the Storage Read API
            fails during collection.
        ValueError
            If `schema` is `None` and a non-empty table has zero supported
            columns remaining after omitting unsupported columns.

        Examples
        --------
        ```python
        import polars as pl
        import polars_bigquery

        lf = client.scan_table("bigquery-public-data.usa_names.usa_1910_2013")
        df = (
            lf.filter(pl.col("name").str.starts_with("A") & (pl.col("year") >= 2010))
            .select(["name", "state", "year", "number"])
            .collect()
        )

        # Skip the tables.get metadata call with an explicit schema:
        lf_typed = client.scan_table(
            "my-project.my_dataset.events",
            schema={
                "ts": pl.Datetime("us", "UTC"),
                "payload": polars_bigquery.dtypes.JSON,
                "location": polars_bigquery.dtypes.GEOGRAPHY,
            },
        )
        ```
        """
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        bq_types: Mapping[str, polars_bigquery.core.types.BQType] | None
        if schema is None:
            table_metadata = self._rest_client.get_table_metadata(table_ref)
            try:
                extracted = polars_bigquery.core.schema.extract_schema_metadata(
                    table_metadata
                )
            except (TypeError, ValueError) as exc:
                msg = f"Invalid BigQuery table metadata for '{table_ref}': {exc}"
                raise polars_bigquery.exceptions.BigQueryError(msg) from exc
            polars_schema = extracted.polars_schema
            bq_types = extracted.bq_types
            if extracted.skipped_columns:
                skipped_summary = _format_skipped_columns(extracted.skipped_columns)
                if len(polars_schema) == 0:
                    msg = (
                        f"Table '{table_ref}' has no supported columns to scan. "
                        f"Skipped {len(extracted.skipped_columns)} unsupported "
                        f"column(s): {skipped_summary}. "
                        "Use client.read_query() to cast or exclude unsupported "
                        "columns in SQL."
                    )
                    raise ValueError(msg)
                logger.debug(
                    "Omitting %d unsupported column(s) from table '%s': %s",
                    len(extracted.skipped_columns),
                    table_ref,
                    skipped_summary,
                )
                warnings.warn(
                    f"Skipping {len(extracted.skipped_columns)} unsupported column(s) "
                    f"in table '{table_ref}': {skipped_summary}. "
                    "Use client.read_query() to cast or select these columns in SQL.",
                    polars_bigquery.exceptions.UnsupportedTypeWarning,
                    stacklevel=2,
                )
        else:
            polars_schema = (
                schema if isinstance(schema, pl.Schema) else pl.Schema(schema)
            )
            bq_types = None

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            # Explicit empty `schema={}` is treated identically to a 0-column table:
            # short-circuit without calling `CreateReadSession` (BigQuery rejects empty
            # schemas with FAILED_PRECONDITION).
            if len(polars_schema) == 0:
                logger.debug(
                    "Short-circuiting empty schema scan for table '%s' without "
                    "CreateReadSession",
                    table_ref,
                )
                return iter([pl.DataFrame(schema=polars_schema)])

            columns = (
                with_columns if with_columns is not None else list(polars_schema.keys())
            )
            predicate_cols = (
                predicate.meta.root_names() if predicate is not None else []
            )
            selected_fields = list(dict.fromkeys([*columns, *predicate_cols]))
            if not selected_fields:
                narrowest_col = min(
                    polars_schema.items(),
                    key=lambda item: polars_bigquery.core.types.column_width_rank(
                        item[1]
                    ),
                )[0]
                selected_fields = [narrowest_col]

            row_restriction = (
                compiler.predicate_to_row_restriction(
                    predicate=predicate,
                    schema=polars_schema,
                    bq_types=bq_types,
                )
                if predicate is not None
                else ""
            )
            logger.debug(
                "Reading BigQuery table '%s' with selected_fields=%r, "
                "row_restriction=%r",
                table_ref,
                selected_fields,
                row_restriction,
            )

            # Note: even when `row_restriction == "FALSE"` (e.g. `int_col.is_nan()`),
            # we intentionally open a Storage Read API session so server-side table
            # existence and IAM authorization errors surface consistently.
            read_kwargs: dict[str, Any] = {
                "maintain_order": False,
                "selected_fields": selected_fields,
                "row_restriction": row_restriction,
            }
            if n_rows is not None and n_rows <= _SINGLE_STREAM_ROW_LIMIT:
                read_kwargs["max_stream_count"] = 1
            arrow_stream_exporter = self._arrow_client.read_table(
                table_ref,
                **read_kwargs,
            )
            lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)

            if predicate is not None:
                lazyframe = lazyframe.filter(predicate)

            if n_rows is not None:
                lazyframe = lazyframe.limit(n_rows)

            if columns == []:
                return _drop_columns_iter(
                    lazyframe.collect_batches(chunk_size=batch_size)
                )

            return lazyframe.select(columns).collect_batches(chunk_size=batch_size)

        return polars.io.plugins.register_io_source(
            io_source=source_generator,
            schema=polars_schema,
        )
