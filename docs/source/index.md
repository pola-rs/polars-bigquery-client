# Polars BigQuery

`polars-bigquery` is a fast I/O connector that brings [Google BigQuery](https://docs.cloud.google.com/bigquery/docs/introduction) tables and queries directly into [Polars](https://docs.pola.rs/) `LazyFrame` and `DataFrame` workflows.

Under the hood, it streams Apache Arrow record batches from the **BigQuery Storage Read API** in Rust and hands them off to Polars via the zero-copy Arrow C Stream interface.

## Key Features

- **Lazy scanning with pushdown**: [`Client.scan_table`](api/index.md#polars_bigquery.Client.scan_table) integrates with the Polars query optimizer to push down **column selections** and **[row filter predicates](api/predicate-pushdown.md)** to BigQuery Storage Read API—reducing both network transfer and BigQuery bytes scanned.
- **Lazy GoogleSQL query scans**: [`Client.scan_query`](api/index.md#polars_bigquery.Client.scan_query) defers executing `SELECT` queries until collected or sunk (for results up to ~10 GB compressed), preserving `ORDER BY` ordering by default (`maintain_order=True`), or pushing down projections, predicates, and `LIMIT` via GoogleSQL pipe operators with `maintain_order=False`.
- **High-throughput Rust streaming**: Reads from BigQuery Storage Read API streams with LZ4 Arrow compression, adaptive parallel reads, and zero-copy transfer into Polars.
- **Eager GoogleSQL query execution**: [`Client.read_query`](api/index.md#polars_bigquery.Client.read_query) runs arbitrary GoogleSQL queries and returns the result table in memory as a Polars `DataFrame`.
- **Connection & credential reuse**: [`Client`](api/index.md#polars_bigquery.Client) pools gRPC and HTTP connections and authenticates using Google Cloud Application Default Credentials (ADC) via `polars.CredentialProviderGCP`.

## Getting Started

### 1. Installation

Install `polars-bigquery` alongside `polars`:

=== "pip"

    ```shell
    pip install polars-bigquery
    ```

=== "uv"

    ```shell
    uv add polars-bigquery
    ```

### 2. Authentication

By default, `polars-bigquery` uses [Application Default Credentials (ADC)](https://docs.cloud.google.com/docs/authentication/provide-credentials-adc). When developing locally, authenticate with the Google Cloud CLI:

```shell
gcloud auth application-default login
```

You will also need a Google Cloud project ID (`quota_project_id`) with the appropriate IAM roles:

- `roles/bigquery.readSessionUser` to create and bill BigQuery Storage Read API sessions.
- `roles/bigquery.jobUser` (`bigquery.jobs.create`) to run query jobs and dry runs with `read_query` and `scan_query`.
- `roles/bigquery.dataViewer` (`bigquery.tables.getData`) on referenced private tables and datasets.

### 3. Quick Example

Use [`Client.scan_table`](api/index.md#polars_bigquery.Client.scan_table) to lazily scan a table—Polars will push both `.select()` columns and `.filter()` predicates down to BigQuery when `.collect()` is called:

```python
import polars as pl
import polars_bigquery

with polars_bigquery.Client(quota_project_id="my-gcp-project") as client:
    # Lazily scan a BigQuery table with column and predicate pushdown
    lf = client.scan_table("bigquery-public-data.usa_names.usa_1910_2013")
    df = (
        lf.filter((pl.col("state") == "WA") & (pl.col("year") >= 2000))
        .select(["name", "year", "number"])
        .group_by("name")
        .agg(pl.col("number").sum().alias("total"))
        .sort("total", descending=True)
        .head(10)
        .collect()
    )
    print(df)
```

To lazily stream a GoogleSQL `SELECT` query (up to ~10 GB compressed) to Parquet or another sink, use [`Client.scan_query`](api/index.md#polars_bigquery.Client.scan_query). Keep the default `maintain_order=True` to execute the SQL verbatim and preserve `ORDER BY` ordering and exact-string query cache hits. Set `maintain_order=False` when row order does not matter to enable adaptive parallel reads as well as supported projection, [row predicate pushdown](api/predicate-pushdown.md), and `LIMIT` pushdown via GoogleSQL pipe operators (`|> WHERE`, `|> SELECT`, `|> LIMIT`). Optionally pass `schema` to skip the initial BigQuery dry run used for schema discovery:

```python
with polars_bigquery.Client(quota_project_id="my-gcp-project") as client:
    lf = client.scan_query(
        """
        SELECT name, state, year, number
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        """,
        maintain_order=False,
    )
    lf.filter(pl.col("year") >= 2000).sink_parquet("names.parquet")
```

To execute a custom SQL query eagerly into an in-memory `DataFrame`, use [`Client.read_query`](api/index.md#polars_bigquery.Client.read_query):

```python
with polars_bigquery.Client(quota_project_id="my-gcp-project") as client:
    df = client.read_query(
        """
        SELECT name, SUM(number) AS total_born
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        GROUP BY name
        ORDER BY total_born DESC
        LIMIT 10
        """
    )
```

For query results exceeding ~10 GB compressed (the anonymous destination table limit) or materialized results scanned multiple times, see [Scanning large query results](guides/large-query-results.md).

## Documentation

- **[Python API Reference (`polars_bigquery`)](api/index.md)**: Full reference for [`Client`](api/index.md#polars_bigquery.Client) (`read_table`, `read_query`, `scan_table`, `scan_query`) and [`BigQueryError`](api/index.md#polars_bigquery.exceptions.BigQueryError).
- **[Row Predicate Pushdown](api/predicate-pushdown.md)**: Complete list of Polars expressions and data types supported for server-side filtering in [`Client.scan_table`](api/index.md#polars_bigquery.Client.scan_table) and [`Client.scan_query`](api/index.md#polars_bigquery.Client.scan_query).
- **[Scanning Large Query Results](guides/large-query-results.md)**: Lazily scan query results exceeding ~10 GB compressed (or materialized results reused across multiple scans) via an explicit destination table and [`Client.scan_table`](api/index.md#polars_bigquery.Client.scan_table).

## Development

Interested in contributing or running the test suite?

- [Contributing Overview](development/contributing/index.md)
- [Development Tasks (`maskfile.md`)](development/contributing/maskfile.md)
