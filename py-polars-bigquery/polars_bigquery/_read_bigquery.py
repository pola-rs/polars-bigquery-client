from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import arrow_bigquery
import arrow_bigquery.api.resources
import polars as pl
import polars.io.plugins

import polars_bigquery.core.schema
import polars_bigquery.core.version
from polars_bigquery.core import bigquery_rest, compiler


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
        maintain_order: bool = False,
    ) -> pl.DataFrame:
        """Execute a GoogleSQL query and read the result into a Polars `DataFrame`.

        Submits a query job via the BigQuery Jobs REST API, waits for job
        completion, and streams the destination table via the BigQuery Storage
        Read API.

        Parameters
        ----------
        query
            GoogleSQL query string to execute in BigQuery.
        maintain_order
            Whether to preserve the row order produced by an `ORDER BY` clause
            by reading from a single Storage Read API stream. Defaults to
            `False`, which enables parallel multi-stream reads.

        Returns
        -------
        pl.DataFrame
            A Polars `DataFrame` containing the query results.

        Raises
        ------
        polars_bigquery.exceptions.BigQueryError
            If the BigQuery query job fails or times out.

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
            \"\"\",
            maintain_order=True,
        )
        ```
        """
        table = self._rest_client.run_query(query)
        table_ref = arrow_bigquery.api.resources.parse_table_id(table)
        arrow_stream_exporter = self._arrow_client.read_table(
            table_ref,
            maintain_order=maintain_order,
        )
        return pl.DataFrame(arrow_stream_exporter)

    def scan_table(
        self,
        table: Any,
    ) -> pl.LazyFrame:
        """Lazily scan a BigQuery table into a Polars `LazyFrame`.

        Fetches table schema and partitioning metadata via the BigQuery REST API
        when constructing the `LazyFrame`, and defers reading rows from the
        BigQuery Storage Read API until query execution (`.collect()` or
        `.collect_batches()`).

        Supports both **projection pushdown** (only reading referenced columns)
        and **row predicate pushdown** (translating `.filter()` expressions into
        BigQuery Storage Read API `row_restriction` SQL clauses). See the
        [Row predicate pushdown](predicate-pushdown.md) page for the complete
        list of supported Polars expressions and data types.

        For ingestion-time partitioned tables, `_PARTITIONDATE` (`pl.Date`) and
        `_PARTITIONTIME` (`pl.Datetime("us", "utc")`) pseudo-columns are
        automatically added to the `LazyFrame` schema so partition pruning
        filters can be pushed down.

        Parameters
        ----------
        table
            The BigQuery table to scan. Accepts a `"project.dataset.table"`
            string, an `arrow_bigquery.BigQueryTableId`, or any object with
            `project` (or `project_id`), `dataset_id`, and `table_id` attributes
            (such as `google.cloud.bigquery.TableReference`).

        Returns
        -------
        pl.LazyFrame
            A Polars `LazyFrame` backed by the BigQuery Storage Read API.

        Raises
        ------
        polars_bigquery.exceptions.BigQueryError
            If fetching the table metadata from the BigQuery REST API fails.

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
        table_metadata = self._rest_client.get_table_metadata(table_ref)
        schema = polars_bigquery.core.schema.extract_polars_schema(table_metadata)

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            columns = with_columns if with_columns is not None else list(schema.keys())
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
            schema=schema,
        )
