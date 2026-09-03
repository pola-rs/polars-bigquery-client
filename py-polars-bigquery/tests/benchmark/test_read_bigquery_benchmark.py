import os

import polars
import polars_bigquery
import pytest

TABLE_IDS = [
    "bigquery-public-data.usa_names.usa_1910_2013",
]


@pytest.fixture(scope="session")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    return polars_bigquery.Client(quota_project_id=project)


@pytest.mark.benchmark(min_rounds=10, warmup=True)
@pytest.mark.parametrize("table_id", TABLE_IDS)
def test_read_bigquery_public_data(client, table_id, benchmark):
    df = benchmark(
        client.read_table,
        table=table_id,
    )
    assert isinstance(df, polars.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height > 5_000_000  # rows
    assert df.width > 0  # columns
