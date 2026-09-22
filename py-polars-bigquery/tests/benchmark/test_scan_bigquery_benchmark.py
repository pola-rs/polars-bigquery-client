import datetime
import os

import polars as pl
import pytest

import polars_bigquery


@pytest.fixture(scope="session")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    return polars_bigquery.Client(quota_project_id=project)


@pytest.mark.benchmark(min_rounds=10, warmup=True)
def test_scan_usa_names(client, benchmark):
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


@pytest.mark.benchmark(min_rounds=10, warmup=True)
def test_scan_sec_quarterly_financials_submission(client, benchmark):
    def scan_bigquery_and_collect():
        return (
            client.scan_table(
                "bigquery-public-data.sec_quarterly_financials.submission"
            )
            .filter(
                (pl.col("_PARTITIONDATE") >= datetime.date(2019, 9, 1))
                & (pl.col("_PARTITIONDATE") <= datetime.date(2019, 12, 31))
                & (
                    pl.col("central_index_key").is_in([1652044, 1288776])
                    | pl.col("company_name")
                    .str.to_uppercase()
                    .str.contains("ALPHABET INC")
                )
            )
            .select(
                "submission_number",
                "company_name",
                pl.col("central_index_key").alias("cik"),
                "form",
                "fiscal_year",
                pl.col("fiscal_period_focus").alias("fiscal_period"),
                pl.col("date_filed").alias("filing_date"),
            )
        ).collect()

    df = benchmark(scan_bigquery_and_collect)
    assert df.height > 0
    assert df.columns == [
        "submission_number",
        "company_name",
        "cik",
        "form",
        "fiscal_year",
        "fiscal_period",
        "filing_date",
    ]
