import os

import polars
import polars_bigquery
import pytest

TABLE_IDS = [
    "bigquery-public-data.usa_names.usa_1910_2013",
]

SEC_TABLES = [
    "sic_codes",
    "submission",
]

# The portion of the "numbers" table we are reading is about 5 GB total, so
# these reads vary from about 500 MB to 5 GB.
PROPORTIONS = [0.1, 0.5, 1.0]


def _format_sec_quarterly_financials_query(table_id: str) -> str:
    return f"SELECT * FROM `bigquery-public-data.sec_quarterly_financials.{table_id}`"


def _format_numbers_query(proportion: float) -> str:
    return f"""
        SELECT *
        FROM `bigquery-public-data.sec_quarterly_financials.numbers`
        WHERE
            TIMESTAMP_TRUNC(_PARTITIONTIME, DAY)
                BETWEEN TIMESTAMP("2019-01-01") AND TIMESTAMP("2019-12-31")
            AND ((FARM_FINGERPRINT(submission_number) / POW(2.0, 64)) + 0.5) < {proportion}
    """


@pytest.fixture(scope="session")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    return polars_bigquery.Client(quota_project_id=project)


@pytest.mark.benchmark(group="read_table", min_rounds=10, warmup=True)
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


@pytest.mark.benchmark(group="read_table", min_rounds=10, warmup=True)
@pytest.mark.parametrize("table_id", SEC_TABLES)
def test_read_table_sec_quarterly_financials(client, table_id, benchmark):
    df = benchmark(
        client.read_table,
        table=f"bigquery-public-data.sec_quarterly_financials.{table_id}",
        maintain_order=False,
    )
    assert isinstance(df, polars.DataFrame)
    assert df.height > 0
    assert df.width > 0


@pytest.mark.benchmark(group="read_query", min_rounds=10, warmup=True)
def test_read_query_select_1(client, benchmark):
    df = benchmark(
        client.read_query,
        query="SELECT 1",
        maintain_order=True,
    )
    assert isinstance(df, polars.DataFrame)
    assert df.height == 1
    assert df.width == 1


@pytest.mark.benchmark(group="read_query", min_rounds=10, warmup=True)
@pytest.mark.parametrize("table_id", SEC_TABLES)
def test_read_query_sec_quarterly_financials(client, table_id, benchmark):
    sql = _format_sec_quarterly_financials_query(table_id)
    df = benchmark(
        client.read_query,
        query=sql,
        maintain_order=True,
    )
    assert isinstance(df, polars.DataFrame)
    assert df.height > 0
    assert df.width > 0


@pytest.mark.benchmark(group="read_query", min_rounds=10, warmup=True)
@pytest.mark.parametrize("proportion", PROPORTIONS)
def test_read_query_sec_quarterly_financials_numbers(client, proportion, benchmark):
    sql = _format_numbers_query(proportion)
    df = benchmark(
        client.read_query,
        query=sql,
        maintain_order=False,
    )
    assert isinstance(df, polars.DataFrame)
    assert df.height > 0
    assert df.width > 0
