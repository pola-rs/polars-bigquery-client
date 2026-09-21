import os
from datetime import date

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
        maintain_order=True,
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


def test_scan_bigquery_sec_quarterly_financials_partition_pushdown(client):
    # Note: bigquery-public-data.sec_quarterly_financials contains partitions up through
    # 2020-12-31, so 2020 is the last year of data in the dataset.
    google_submissions = (
        client.scan_table("bigquery-public-data.sec_quarterly_financials.submission")
        .filter(
            (pl.col("_PARTITIONDATE") >= date(2020, 1, 1))
            & (pl.col("_PARTITIONDATE") <= date(2020, 12, 31))
            & (
                pl.col("central_index_key").is_in([1652044, 1288776])
                | pl.col("company_name").str.to_uppercase().str.contains("ALPHABET INC")
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
    )

    numbers = client.scan_table(
        "bigquery-public-data.sec_quarterly_financials.numbers"
    ).filter(
        (pl.col("_PARTITIONDATE") >= date(2020, 1, 1))
        & (pl.col("_PARTITIONDATE") <= date(2020, 12, 31))
        & pl.col("measure_tag").is_in(
            [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "OperatingIncomeLoss",
                "NetIncomeLoss",
                "ResearchAndDevelopmentExpense",
                "EarningsPerShareDiluted",
            ]
        )
        & pl.col("units").is_in(["USD", "shares"])
    )

    df = (
        numbers.join(google_submissions, on="submission_number", how="inner")
        .select(
            pl.col("_PARTITIONDATE").alias("ingestion_date"),
            pl.col("company_name_right").alias("company_name"),
            "form",
            "fiscal_year",
            "fiscal_period",
            pl.col("measure_tag").alias("tag"),
            "period_end_date",
            "number_of_quarters",
            "units",
            "value",
        )
        .sort(
            by=["period_end_date", "fiscal_year", "fiscal_period", "tag"],
            descending=[True, True, True, False],
        )
        .collect()
    )

    assert isinstance(df, pl.DataFrame)
    assert df.height == 386
    assert df.columns == [
        "ingestion_date",
        "company_name",
        "form",
        "fiscal_year",
        "fiscal_period",
        "tag",
        "period_end_date",
        "number_of_quarters",
        "units",
        "value",
    ]
    assert (df["ingestion_date"] >= date(2020, 1, 1)).all()
    assert (df["ingestion_date"] <= date(2020, 12, 31)).all()
    assert set(df["company_name"].unique().to_list()) == {"ALPHABET INC."}
