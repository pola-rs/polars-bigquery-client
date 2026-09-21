from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import arrow_bigquery
import arrow_bigquery.api.resources
import polars as pl
import polars.io.plugins

import polars_bigquery.core.schema
import polars_bigquery.core.version
from polars_bigquery.core import bigquery_rest, predicates


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"polars-bigquery/{polars_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


class Client:
    """Client for reading data from BigQuery into Polars.

    Wraps an arrow_bigquery Client to keep connections open and reuse credentials across operations.
    """

    def __init__(
        self,
        *,
        quota_project_id: str,
        credentials_provider: pl.CredentialProviderGCP | None = None,
        user_agent: str | None = None,
    ) -> None:
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
        return self._credentials_provider

    @property
    def quota_project_id(self) -> str:
        return self._quota_project_id

    def read_table(
        self,
        table: Any,
        *,
        maintain_order: bool = False,
    ) -> pl.DataFrame:
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
                row_restriction=predicates.predicate_to_row_restriction(
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
