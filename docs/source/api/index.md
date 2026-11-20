# API Reference

The `polars_bigquery` package provides an I/O connector for reading Google BigQuery tables and queries into [Polars](https://docs.pola.rs/) `DataFrame` and `LazyFrame` objects.

See [Data types](data-types.md) for the BigQuery-to-Polars schema mapping, extension dtypes, and unsupported-column handling, and [Row predicate pushdown](predicate-pushdown.md) for details on which Polars expressions are pushed down to BigQuery when using [`Client.scan_table`][polars_bigquery.Client.scan_table].

::: polars_bigquery.Client

::: polars_bigquery.dtypes

::: polars_bigquery.exceptions.BigQueryError

::: polars_bigquery.exceptions.UnsupportedTypeWarning
