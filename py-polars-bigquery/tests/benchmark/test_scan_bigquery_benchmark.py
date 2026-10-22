import datetime
import os

import polars as pl
import polars_bigquery
import pytest

USA_NAMES_SCHEMA = pl.Schema(
    {
        "state": pl.String(),
        "gender": pl.String(),
        "year": pl.Int64(),
        "name": pl.String(),
        "number": pl.Int64(),
    }
)

SEC_SUBMISSION_SCHEMA = pl.Schema(
    {
        "submission_number": pl.String(),
        "central_index_key": pl.Int64(),
        "company_name": pl.String(),
        "sic": pl.String(),
        "countryba": pl.String(),
        "stprba": pl.String(),
        "cityba": pl.String(),
        "zipba": pl.String(),
        "ba_street1": pl.String(),
        "ba_street2": pl.String(),
        "ba_phone": pl.String(),
        "countryma": pl.String(),
        "stprma": pl.String(),
        "cityma": pl.String(),
        "zipma": pl.String(),
        "ma_street1": pl.String(),
        "ma_street2": pl.String(),
        "country_inc": pl.String(),
        "state_province_inc": pl.String(),
        "ein": pl.String(),
        "former_name": pl.String(),
        "date_changed": pl.Int64(),
        "filer_status": pl.String(),
        "well_known_seasoned_issuer": pl.Boolean(),
        "fiscal_year_end": pl.String(),
        "form": pl.String(),
        "period": pl.Int64(),
        "fiscal_year": pl.Int64(),
        "fiscal_period_focus": pl.String(),
        "date_filed": pl.Int64(),
        "date_accepted": pl.Datetime("us", "utc"),
        "has_previous_report": pl.Boolean(),
        "has_required_detail": pl.Boolean(),
        "instance": pl.String(),
        "num_ciks": pl.Int64(),
        "additional_ciks": pl.String(),
        "public_float_usd": pl.Float64(),
        "float_date": pl.Int64(),
        "float_axis": pl.String(),
        "float_terms": pl.Int64(),
        "_PARTITIONDATE": pl.Date(),
        "_PARTITIONTIME": pl.Datetime("us", "utc"),
    }
)


@pytest.fixture(scope="session")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    return polars_bigquery.Client(quota_project_id=project)


@pytest.mark.benchmark(min_rounds=10, warmup=True)
@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(None, id="no_schema"),
        pytest.param(USA_NAMES_SCHEMA, id="with_schema"),
    ],
)
def test_scan_usa_names(client, benchmark, schema):
    def scan_bigquery_and_collect():
        ldf = client.scan_table(
            table="bigquery-public-data.usa_names.usa_1910_2013",
            schema=schema,
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
@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(None, id="no_schema"),
        pytest.param(SEC_SUBMISSION_SCHEMA, id="with_schema"),
    ],
)
def test_scan_sec_quarterly_financials_submission(client, benchmark, schema):
    def scan_bigquery_and_collect():
        return (
            client.scan_table(
                "bigquery-public-data.sec_quarterly_financials.submission",
                schema=schema,
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
