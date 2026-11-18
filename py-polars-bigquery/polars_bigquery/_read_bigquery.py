from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import arrow_bigquery
import arrow_bigquery.api.resources
import polars as pl
import polars.io.plugins
from polars.datatypes import DataType, DataTypeClass

import polars_bigquery.core.schema
import polars_bigquery.core.version
import polars_bigquery.exceptions
from polars_bigquery.core import bigquery_rest, compiler


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"polars-bigquery/{polars_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


def _wrap_query_for_pushdown(
    query: str,
    *,
    query_columns: list[str],
    all_columns: set[str],
    row_restriction: str,
) -> str:
    """Wrap `query` in an outer SELECT only when pruning columns or filtering."""
    needs_projection = bool(query_columns) and set(query_columns) < all_columns
    if not needs_projection and not row_restriction:
        return query

    select_clause = (
        ", ".join(
            compiler.sql.column_to_sql_identifier(column) for column in query_columns
        )
        if needs_projection
        else "*"
    )
    wrapped_query = (
        f"SELECT {select_clause} FROM (\n{query}\n) AS _polars_bigquery_source"
    )
    if row_restriction:
        wrapped_query += f" WHERE {row_restriction}"
    return wrapped_query


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
        self._arrow_client = arrow_bigquery.Client(
            quota_project_id=quota_project_id,
            credentials_provider=credentials_provider,
            user_agent=self._user_agent,
        )

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

    def scan_query(
        self,
        query: str,
        *,
        schema: pl.Schema | Mapping[str, DataType | DataTypeClass] | None = None,
        arrow_buffer_compression: str = "lz4frame",
        maintain_order: bool = True,
        use_query_cache: bool | None = None,
    ) -> pl.LazyFrame:
        """Lazily execute a GoogleSQL SELECT query into a Polars `LazyFrame`.

        Discovers the query's result schema with a BigQuery dry run when
        constructing the `LazyFrame` (unless `schema` is provided), and defers
        executing the query until `.collect()`, `.collect_batches()`, or a sink
        is invoked. Only `SELECT` statements are supported.

        When `maintain_order=True` (the default), pushdown is disabled and the
        original query is executed verbatim using a single stream so any
        `ORDER BY` ordering is preserved.

        When `maintain_order=False`, reads use parallel multi-stream execution
        and supported **projection pushdown** and **[row predicate pushdown](predicate-pushdown.md)**
        are pushed into an outer `SELECT` around the supplied SQL whenever
        columns are pruned or a supported filter is applied (regardless of
        whether `schema` was supplied or discovered via dry run). When no
        columns are pruned and no filter is pushed down, the original query runs
        verbatim. When only a filter is pushed down (all columns are needed),
        `SELECT *` is used so anonymous computed columns remain accessible.

        !!! note "Anonymous columns and pushdown"
            In GoogleSQL, unaliased expressions (which BigQuery names `f0_`,
            `f1_`, … in top-level query results) are anonymous inside a
            subquery and cannot be referenced by name from an outer query.
            When `maintain_order=False`, explicitly projecting or filtering on
            such a column will fail with `Unrecognized name`. Give computed
            expressions explicit aliases (`AS col_name`) in the SQL query, or
            keep `maintain_order=True`.

        Parameters
        ----------
        query
            GoogleSQL `SELECT` query.
        schema
            Optional Polars schema (`polars.Schema` or a mapping of column names
            to Polars data types) for the query result. Purely an optimization:
            when provided, `scan_query` skips the BigQuery dry run used for
            schema discovery.
        arrow_buffer_compression
            Compression codec for Arrow record batch buffers (`"lz4frame"`,
            `"zstd"`, or `"unspecified"`). Defaults to `"lz4frame"`.
        maintain_order
            Whether to preserve query result order by using a single stream and
            avoiding subquery wrapping. Defaults to `True`. Set to `False` to
            enable parallel multi-stream reads as well as projection and
            predicate pushdown when row order does not matter.
        use_query_cache
            Whether BigQuery should use the query cache. If `None`, BigQuery's
            API default (`True`) is used.

        Returns
        -------
        pl.LazyFrame
            A lazy query-result scan backed by the BigQuery Storage Read API.

        Raises
        ------
        ValueError
            If `query` is empty, if the dry run reports a non-`SELECT`
            statement type, or if the dry-run result schema contains duplicate
            column names.
        polars_bigquery.exceptions.BigQueryError
            If the BigQuery dry run fails (for example, due to invalid SQL) or
            returns an invalid/missing schema response.
        """
        query = query.strip().removesuffix(";").strip()
        if not query:
            raise ValueError("query must be a non-empty SELECT statement")

        if schema is None:
            metadata = self.read_query(query, dry_run=True)
            schema_json = metadata["bigquerySchema"][0]
            statement_type = metadata["statementType"][0]
            if statement_type not in (None, "SELECT"):
                msg = f"scan_query only supports SELECT statements, got {statement_type!r}"
                raise ValueError(msg)
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
            polars_schema = polars_bigquery.core.schema.extract_polars_schema(
                {"schema": json.loads(schema_json)}
            )
        else:
            polars_schema = (
                schema if isinstance(schema, pl.Schema) else pl.Schema(schema)
            )

        all_columns = list(polars_schema.keys())
        all_columns_set = set(all_columns)

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            output_columns = with_columns if with_columns is not None else all_columns

            if not maintain_order:
                predicate_columns = (
                    predicate.meta.root_names() if predicate is not None else []
                )
                query_columns = list(
                    dict.fromkeys([*output_columns, *predicate_columns])
                )
                row_restriction = (
                    compiler.predicate_to_row_restriction(predicate)
                    if predicate is not None
                    else ""
                )
                wrapped_query = _wrap_query_for_pushdown(
                    query,
                    query_columns=query_columns,
                    all_columns=all_columns_set,
                    row_restriction=row_restriction,
                )
            else:
                wrapped_query = query

            arrow_stream_exporter = self._arrow_client.read_query(
                wrapped_query,
                arrow_buffer_compression=arrow_buffer_compression,
                dry_run=False,
                maintain_order=maintain_order,
                use_query_cache=use_query_cache,
            )
            lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)
            if predicate is not None:
                lazyframe = lazyframe.filter(predicate)
            lazyframe = lazyframe.select(output_columns)
            if n_rows is not None:
                lazyframe = lazyframe.limit(n_rows)
            yield from lazyframe.collect_batches(chunk_size=batch_size)

        return polars.io.plugins.register_io_source(
            io_source=source_generator,
            schema=polars_schema,
        )

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
            Optional Polars schema (`polars.Schema` or a mapping of column names
            to Polars data types) for the table. When provided, `scan_table`
            avoids calling the BigQuery `tables.get` REST API to fetch table
            metadata. For ingestion-time partitioned tables, include
            `"_PARTITIONDATE": pl.Date` and/or
            `"_PARTITIONTIME": pl.Datetime("us", "utc")` in `schema` if you want
            to filter or select those pseudo-columns.

        Returns
        -------
        pl.LazyFrame
            A Polars `LazyFrame` backed by the BigQuery Storage Read API.

        Raises
        ------
        polars_bigquery.exceptions.BigQueryError
            If `schema` is `None` and fetching the table metadata from the
            BigQuery REST API fails.

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
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        if schema is None:
            table_metadata = self._rest_client.get_table_metadata(table_ref)
            polars_schema = polars_bigquery.core.schema.extract_polars_schema(
                table_metadata
            )
        else:
            polars_schema = (
                schema if isinstance(schema, pl.Schema) else pl.Schema(schema)
            )

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            columns = (
                with_columns if with_columns is not None else list(polars_schema.keys())
            )
            predicate_cols = (
                predicate.meta.root_names() if predicate is not None else []
            )
            selected_fields = list(dict.fromkeys([*columns, *predicate_cols]))

            arrow_stream_exporter = self._arrow_client.read_table(
                table_ref,
                maintain_order=False,
                selected_fields=selected_fields,
                row_restriction=compiler.predicate_to_row_restriction(
                    predicate=predicate
                )
                if predicate is not None
                else "",
            )
            lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)

            if predicate is not None:
                lazyframe = lazyframe.filter(predicate)

            lazyframe = lazyframe.select(columns)

            if n_rows is not None:
                lazyframe = lazyframe.limit(n_rows)

            return lazyframe.collect_batches(chunk_size=batch_size)

        return polars.io.plugins.register_io_source(
            io_source=source_generator,
            schema=polars_schema,
        )
