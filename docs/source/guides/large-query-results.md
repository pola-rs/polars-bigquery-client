# Scanning Large Query Results

[`Client.read_query`](../api/index.md#polars_bigquery.Client.read_query) runs a GoogleSQL query and reads the whole result into a Polars `DataFrame`. When the result is larger than memory, or you want to keep processing it with the Polars lazy API and stream it to a sink such as Parquet, write the query result to a table and scan that table with [`Client.scan_table`](../api/index.md#polars_bigquery.Client.scan_table).

This is also required for large results: the anonymous table that BigQuery writes query results to is [limited to about 10 GB](https://docs.cloud.google.com/bigquery/docs/writing-results#large-results). Larger results need an explicit destination table.

These examples use the [`google-cloud-bigquery`](https://pypi.org/project/google-cloud-bigquery/) client to run the query:

=== "pip"

    ```shell
    pip install polars-bigquery google-cloud-bigquery
    ```

=== "uv"

    ```shell
    uv add polars-bigquery google-cloud-bigquery
    ```

## Scan the Result of a Query Job

A completed `SELECT` query job has a `destination` table, which `scan_table` accepts directly:

```python
import polars as pl
import polars_bigquery
from google.cloud import bigquery

sql = """
SELECT name, state, year, number
FROM `bigquery-public-data.usa_names.usa_1910_2013`
WHERE year >= 2000
"""

bq_client = bigquery.Client(project="my-gcp-project")
job = bq_client.query(sql)
job.result()

with polars_bigquery.Client(quota_project_id="my-gcp-project") as client:
    lf = client.scan_table(job.destination)
    lf.filter(pl.col("state") == "WA").sink_parquet("names_wa.parquet")
```

Without a destination in the job configuration, BigQuery writes the result to an anonymous table that is kept for about 24 hours and is subject to the 10 GB limit. Use this form for smaller results that you want to process lazily.

## Write to an Explicit Destination Table

For results over 10 GB, set a destination table in the job configuration.

The example below creates a temporary dataset for the result and deletes it when the `with` block exits, also on error.

```python
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import polars as pl
import polars_bigquery
from google.cloud import bigquery


@contextmanager
def scan_query(
    sql: str,
    *,
    project: str,
    location: str = "US",
) -> Iterator[pl.LazyFrame]:
    bq_client = bigquery.Client(project=project)
    dataset = bigquery.Dataset(f"{project}.tmp_{uuid.uuid4().hex}")
    dataset.location = location
    dataset.default_table_expiration_ms = 24 * 3600 * 1000
    dataset = bq_client.create_dataset(dataset)
    try:
        job = bq_client.query(
            sql,
            job_config=bigquery.QueryJobConfig(destination=dataset.table("result")),
            location=location,
        )
        job.result()
        with polars_bigquery.Client(quota_project_id=project) as client:
            yield client.scan_table(job.destination)
    finally:
        bq_client.delete_dataset(dataset, delete_contents=True, not_found_ok=True)


with scan_query(sql, project="my-gcp-project") as lf:
    lf.sink_parquet(pl.PartitionBy("data/", max_rows_per_file=1_000_000))
```

The `LazyFrame` reads from the destination table, so collect or sink it inside the `with` block, before the table is deleted.

The table expiration (`default_table_expiration_ms`) is a safety net for when the dataset is not deleted, for example if the Python process is killed or crashes, or the connection to BigQuery breaks. The result table then expires after 24 hours, while the empty dataset remains.

The dataset location must match the location of the tables that the query reads. Creating and deleting datasets requires the `bigquery.datasets.create` and `bigquery.datasets.delete` permissions in `project`. If you can't create datasets, write to a table in an existing dataset instead and delete it with `bq_client.delete_table`.

## Behavior to Be Aware Of

- **The query runs eagerly.** The query job runs to completion before `scan_table` is called, so you pay for the full query regardless of the Polars operations that follow.
- **Pushdown applies to the result table.** Column selections and [row filter predicates](../api/predicate-pushdown.md) on the `LazyFrame` are pushed down to the Storage Read API when reading the result table. They reduce data transferred, but not the bytes processed by the query. Put filters that should reduce query cost in the SQL.
- **Row order is not preserved.** `scan_table` reads from multiple Storage Read API streams in parallel, so an `ORDER BY` in the query does not determine the order of the scanned rows. Sort in Polars, or use [`Client.read_query`](../api/index.md#polars_bigquery.Client.read_query) with `maintain_order=True` for results that fit in memory.
- **Storage Read API billing.** Reading the result table is billed to the `quota_project_id` of the [`Client`](../api/index.md#polars_bigquery.Client), separately from the query job.
