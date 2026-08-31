import os

import polars
import pytest

import polars_bigquery

TABLE_IDS = [
    "bigquery-public-data.usa_names.usa_1910_2013",
]


@pytest.fixture(scope="session")
def client():
    return polars_bigquery.Client()


@pytest.mark.benchmark(min_rounds=10, warmup=True)
@pytest.mark.parametrize("table_id", TABLE_IDS)
def test_read_bigquery_public_data(client, table_id, benchmark):
    project = os.environ["GOOGLE_CLOUD_PROJECT"]

    df = benchmark(
        client.read_table,
        table=table_id,
        quota_project_id=project,
    )
    assert isinstance(df, polars.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height > 5_000_000  # rows
    assert df.width > 0  # columns
