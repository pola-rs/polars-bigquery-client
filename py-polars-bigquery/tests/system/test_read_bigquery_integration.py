import os

import polars
import pytest

import polars_bigquery


@pytest.fixture(scope="module")
def client():
    return polars_bigquery.Client()


def test_read_bigquery_public_data_ordered(client):
    project = os.environ["GOOGLE_CLOUD_PROJECT"]

    # Use a query so that the test can run using BigQuery sandbox quota.
    df = client.read_bigquery_query(
        query="""
        SELECT SUM(number) AS total_born,
        name
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        GROUP BY name
        ORDER BY total_born DESC
        LIMIT 100
        """,
        quota_project_id=project,
        maintain_order=True,
    )
    assert isinstance(df, polars.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height == 100  # rows
    assert df.width > 0  # columns
    assert df["total_born"].is_sorted(descending=True)


def test_read_bigquery_public_data_unordered(client):
    project = os.environ["GOOGLE_CLOUD_PROJECT"]

    # Use a query so that the test can run using BigQuery sandbox quota.
    df = client.read_bigquery_query(
        query="""
        SELECT * FROM `bigquery-public-data.utility_us.country_code_iso`
        """,
        quota_project_id=project,
        maintain_order=False,
    )
    assert isinstance(df, polars.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height > 200  # rows
    assert df.width > 0  # columns
