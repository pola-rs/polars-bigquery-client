import os

import polars as pl
import polars_bigquery
import pytest


@pytest.fixture(scope="module")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    return polars_bigquery.Client(quota_project_id=project)


def test_read_bigquery_public_data_ordered(client):
    # Use a query so that the test can run using BigQuery sandbox quota.
    df = client.read_query(
        query="""
        SELECT SUM(number) AS total_born,
        name
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        GROUP BY name
        ORDER BY total_born DESC
        LIMIT 100
        """,
    )
    assert isinstance(df, pl.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height == 100  # rows
    assert df.width > 0  # columns
    assert df["total_born"].is_sorted(descending=True)


def test_read_bigquery_public_data_unordered(client):
    # Use a query so that the test can run using BigQuery sandbox quota.
    df = client.read_query(
        query="""
        SELECT * FROM `bigquery-public-data.utility_us.country_code_iso`
        """,
        maintain_order=False,
    )
    assert isinstance(df, pl.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height > 200  # rows
    assert df.width > 0  # columns


def test_read_bigquery_dry_run(client):
    df = client.read_query(
        query="""
        SELECT name, number
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        WHERE state = 'WA'
        """,
        dry_run=True,
    )
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert df.columns == [
        "bigquerySchema",
        "projectId",
        "jobId",
        "location",
        "totalBytesProcessed",
        "cacheHit",
        "statementType",
        "creationTime",
    ]
    assert df["totalBytesProcessed"][0] > 0
    assert df["statementType"][0] == "SELECT"


def test_read_bigquery_large_results(client):
    """Verify completed >1 MB compressed LARGE_RESULTS queries stream all rows end-to-end."""
    sql = """
    SELECT
        id,
        CONCAT(
            TO_HEX(SHA256(CONCAT('a:', CAST(id AS STRING)))),
            TO_HEX(SHA256(CONCAT('b:', CAST(id AS STRING))))
        ) AS payload
    FROM UNNEST(GENERATE_ARRAY(1, 50000)) AS id
    """
    df = client.read_query(sql, maintain_order=False, use_query_cache=False)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 50_000
    assert df.schema == pl.Schema({"id": pl.Int64, "payload": pl.String})


def test_scan_query_public_data_unordered(client):
    lf = client.scan_query(
        query="""
        SELECT * FROM `bigquery-public-data.utility_us.country_code_iso`
        """,
        maintain_order=False,
    )
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert isinstance(df, pl.DataFrame)
    assert df.height > 200
    assert df.width > 0


def test_scan_query_public_data_ordered(client):
    lf = client.scan_query(
        query="""
        SELECT SUM(number) AS total_born,
        name
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        GROUP BY name
        ORDER BY total_born DESC
        LIMIT 100
        """,
        maintain_order=True,
    )
    df = lf.collect()
    assert df.height == 100
    assert df.width > 0
    assert df["total_born"].is_sorted(descending=True)


def test_scan_query_with_schema_pushes_projection_and_filter(client):
    lf = client.scan_query(
        query="""
        SELECT name, number, state
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        WHERE year = 2013
        """,
        schema={
            "name": pl.String,
            "number": pl.Int64,
            "state": pl.String,
        },
    )
    df = lf.filter(pl.col("state") == "WA").select("name").limit(10).collect()
    assert df.height == 10
    assert df.columns == ["name"]
    assert df["name"].null_count() == 0


def test_scan_query_without_schema_infers_schema_and_filters_locally(client):
    lf = client.scan_query(
        query="""
        SELECT * FROM `bigquery-public-data.utility_us.country_code_iso`
        """
    )
    df = lf.select(pl.first()).limit(5).collect()
    assert df.height == 5
    assert df.width == 1
    assert df.schema.names()[0] in lf.collect_schema().names()


def test_scan_query_empty_results(client):
    lf = client.scan_query(
        query="SELECT 1 AS value LIMIT 0",
        schema={"value": pl.Int64},
    )
    df = lf.collect()
    assert df.schema == pl.Schema({"value": pl.Int64})
    assert df.is_empty()
