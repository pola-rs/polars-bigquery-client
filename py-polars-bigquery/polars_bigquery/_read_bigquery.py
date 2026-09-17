from __future__ import annotations

from collections.abc import Iterator
import dataclasses
from typing import Any

import arrow_bigquery
import arrow_bigquery.api.resources
import polars as pl
import polars.io.plugins

import polars_bigquery.core.schema
import polars_bigquery.core.version
from polars_bigquery.core import bigquery_rest, predicates

_PSEUDO_COLUMNS = ("_PARTITIONDATE", "_PARTITIONTIME")


def _get_user_agent(user_agent: str | None) -> str:
    ua = f"polars-bigquery/{polars_bigquery.core.version.__version__}"

    if user_agent:
        return f"{ua} {user_agent}"
    else:
        return ua


@dataclasses.dataclass(frozen=True)
class ScanPlan:
    """Pure execution plan for a BigQuery table scan."""

    api_selected_fields: list[str]
    row_restriction: str
    residual_predicate: pl.Expr | None
    synthesize_pseudo_columns: list[tuple[str, pl.DataType]]


def plan_table_scan(
    *,
    schema: dict[str, pl.DataType],
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
) -> ScanPlan:
    """Compute column projection, server-side row restriction, and client-side residual filter."""
    compiled = (
        predicates.compile_predicate(predicate, pseudo_columns=_PSEUDO_COLUMNS)
        if predicate is not None
        else None
    )

    if with_columns is not None:
        predicate_cols = compiled.referenced_columns if compiled is not None else ()
        selected_fields = list(dict.fromkeys([*with_columns, *predicate_cols]))
    else:
        selected_fields = []

    api_selected_fields = [col for col in selected_fields if col not in _PSEUDO_COLUMNS]

    synthesize_pseudo_columns = [
        (pseudo_col, schema[pseudo_col])
        for pseudo_col in _PSEUDO_COLUMNS
        if pseudo_col in schema and (with_columns is None or pseudo_col in with_columns)
    ]

    return ScanPlan(
        api_selected_fields=api_selected_fields,
        row_restriction=compiled.row_restriction if compiled is not None else "",
        residual_predicate=compiled.residual_predicate
        if compiled is not None
        else None,
        synthesize_pseudo_columns=synthesize_pseudo_columns,
    )


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
            plan = plan_table_scan(
                schema=schema,
                with_columns=with_columns,
                predicate=predicate,
            )

            arrow_stream_exporter = self._arrow_client.read_table(
                table_ref,
                maintain_order=False,
                selected_fields=plan.api_selected_fields,
                row_restriction=plan.row_restriction,
            )
            lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)

            if plan.residual_predicate is not None:
                lazyframe = lazyframe.filter(plan.residual_predicate)

            for pseudo_col, dtype in plan.synthesize_pseudo_columns:
                lazyframe = lazyframe.with_columns(
                    pl.lit(None, dtype=dtype).alias(pseudo_col)
                )

            if with_columns is not None:
                lazyframe = lazyframe.select(with_columns)

            if n_rows is not None:
                lazyframe = lazyframe.limit(n_rows)

            return lazyframe.collect_batches(chunk_size=batch_size)

        return polars.io.plugins.register_io_source(
            io_source=source_generator,
            schema=schema,
        )
