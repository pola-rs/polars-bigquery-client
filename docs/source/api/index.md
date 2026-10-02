# API Reference

The `polars_bigquery` package provides an I/O connector for reading Google BigQuery tables and queries into [Polars](https://docs.pola.rs/) `DataFrame` and `LazyFrame` objects.

See [Row predicate pushdown](predicate-pushdown.md) for details on which Polars expressions and data types are pushed down to BigQuery when using [`Client.scan_table`][polars_bigquery.Client.scan_table].

::: polars_bigquery.Client

::: polars_bigquery.exceptions.BigQueryError
