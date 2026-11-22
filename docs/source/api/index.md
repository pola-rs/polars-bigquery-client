# API Reference

The `polars_bigquery` package provides an I/O connector for reading Google BigQuery tables and queries into [Polars](https://docs.pola.rs/) `DataFrame` and `LazyFrame` objects.

See [Data types](data-types.md) for the BigQuery-to-Polars schema mapping, extension dtypes, and unsupported-column handling, and [Row predicate pushdown](predicate-pushdown.md) for details on which Polars expressions are pushed down to BigQuery via Storage Read API `row_restriction` in [`Client.scan_table`][polars_bigquery.Client.scan_table] or via GoogleSQL pipe syntax (`|> WHERE`, `|> SELECT`, `|> LIMIT`) in [`Client.scan_query`][polars_bigquery.Client.scan_query] (with `maintain_order=False`).

::: polars_bigquery.Client

::: polars_bigquery.dtypes

::: polars_bigquery.exceptions.BigQueryError

::: polars_bigquery.exceptions.UnsupportedTypeError

::: polars_bigquery.exceptions.UnsupportedTypeWarning
