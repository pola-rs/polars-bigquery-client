import os

import polars as pl
import polars_bigquery
import pytest


@pytest.fixture(scope="session")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    return polars_bigquery.Client(quota_project_id=project)


@pytest.mark.benchmark(min_rounds=10, warmup=True)
def test_scan_bigquery_public_data(client, benchmark):
    def scan_bigquery_and_collect():
        ldf = client.scan_table(
            table="bigquery-public-data.usa_names.usa_1910_2013",
        )
        # add some filters to push down
        ldf = ldf.select(pl.col("name"), pl.col("number"), pl.col("year"))
        df = ldf.filter(
            pl.col("name").str.starts_with("T")
            & (pl.col("number") > 10)
            & (pl.col("year") == 2000)
        ).collect()
        return df

    df = benchmark(scan_bigquery_and_collect)
    assert isinstance(df, pl.DataFrame)
    assert df.height > 2_000  # rows
    assert df.width > 0  # columns
