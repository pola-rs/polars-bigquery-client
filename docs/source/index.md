# Polars BigQuery

`polars-bigquery` is a fast I/O connector that brings [Google BigQuery](https://docs.cloud.google.com/bigquery/docs/introduction) tables and queries directly into [Polars](https://docs.pola.rs/) `LazyFrame` and `DataFrame` workflows.

Under the hood, it streams Apache Arrow record batches from the **BigQuery Storage Read API** in Rust and hands them off to Polars via the zero-copy Arrow C Stream interface.

## Key Features

- **Lazy scanning with pushdown**: [`Client.scan_table`](api/index.md#polars_bigquery.Client.scan_table) integrates with the Polars query optimizer to push down **column selections** and **[row filter predicates](api/predicate-pushdown.md)** to BigQuery—reducing both network transfer and BigQuery bytes scanned.
- **High-throughput Rust streaming**: Reads from multiple BigQuery Storage Read API streams in parallel with LZ4 Arrow compression and zero-copy transfer into Polars.
- **GoogleSQL query execution**: [`Client.read_query`](api/index.md#polars_bigquery.Client.read_query) runs arbitrary GoogleSQL queries and streams the result table back into a Polars `DataFrame`.
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

You will also need a Google Cloud project ID (`quota_project_id`) with the BigQuery Read Session User (`roles/bigquery.readSessionUser`) role to bill Storage Read API sessions and query jobs.

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

Need to run a custom SQL query? Use [`Client.read_query`](api/index.md#polars_bigquery.Client.read_query):

```python
with polars_bigquery.Client(quota_project_id="my-gcp-project") as client:
    df = client.read_query(
        """
        SELECT name, SUM(number) AS total_born
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        GROUP BY name
        ORDER BY total_born DESC
        LIMIT 10
        """,
        maintain_order=True,
    )
```

## Documentation

- **[Python API Reference (`polars_bigquery`)](api/index.md)**: Full reference for [`Client`](api/index.md#polars_bigquery.Client), its methods, and [`BigQueryError`](api/index.md#polars_bigquery.exceptions.BigQueryError).
- **[Row Predicate Pushdown](api/predicate-pushdown.md)**: Complete list of Polars expressions and data types supported for server-side filtering in [`Client.scan_table`](api/index.md#polars_bigquery.Client.scan_table).

## Development

Interested in contributing or running the test suite?

- [Contributing Overview](development/contributing/index.md)
- [Development Tasks (`maskfile.md`)](development/contributing/maskfile.md)
