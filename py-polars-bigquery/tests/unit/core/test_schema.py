import polars as pl
import pytest

from polars_bigquery.core.schema import extract_polars_schema


def test_extract_polars_schema_primitive_types():
    metadata = {
        "schema": {
            "fields": [
                {"name": "bool_col", "type": "BOOLEAN"},
                {"name": "bytes_col", "type": "BYTES"},
                {"name": "date_col", "type": "DATE"},
                {"name": "datetime_col", "type": "DATETIME"},
                {"name": "geo_col", "type": "GEOGRAPHY"},
                {"name": "float_col", "type": "FLOAT"},
                {"name": "int_col", "type": "INTEGER"},
                {"name": "numeric_col", "type": "NUMERIC"},
                {"name": "string_col", "type": "STRING"},
                {"name": "time_col", "type": "TIME"},
                {"name": "timestamp_col", "type": "TIMESTAMP"},
            ]
        }
    }
    schema = extract_polars_schema(metadata)
    assert schema["bool_col"] == pl.Boolean()
    assert schema["bytes_col"] == pl.Binary()
    assert schema["date_col"] == pl.Date()
    assert schema["datetime_col"] == pl.Datetime(time_unit="us")
    assert schema["geo_col"] == pl.String()
    assert schema["float_col"] == pl.Float64()
    assert schema["int_col"] == pl.Int64()
    assert schema["numeric_col"] == pl.Decimal(precision=38, scale=9)
    assert schema["string_col"] == pl.String()
    assert schema["time_col"] == pl.Time()
    assert schema["timestamp_col"] == pl.Datetime(time_unit="us", time_zone="utc")


def test_extract_polars_schema_repeated_type():
    metadata = {
        "schema": {
            "fields": [
                {"name": "int_array", "type": "INTEGER", "mode": "REPEATED"},
            ]
        }
    }
    schema = extract_polars_schema(metadata)
    assert schema["int_array"] == pl.List(pl.Int64())


def test_extract_polars_schema_struct_type():
    metadata = {
        "schema": {
            "fields": [
                {
                    "name": "record_col",
                    "type": "RECORD",
                    "fields": [
                        {"name": "sub_int", "type": "INT64"},
                        {"name": "sub_str", "type": "STRING"},
                    ],
                }
            ]
        }
    }
    schema = extract_polars_schema(metadata)
    assert schema["record_col"] == pl.Struct(
        [
            pl.Field("sub_int", pl.Int64()),
            pl.Field("sub_str", pl.String()),
        ]
    )


def test_extract_polars_schema_ingestion_time_partitioning():
    metadata = {
        "schema": {"fields": [{"name": "val", "type": "INTEGER"}]},
        "timePartitioning": {"type": "DAY"},
    }
    schema = extract_polars_schema(metadata)
    assert schema["val"] == pl.Int64()
    assert schema["_PARTITIONDATE"] == pl.Date()


def test_extract_polars_schema_column_partitioning():
    metadata = {
        "schema": {
            "fields": [
                {"name": "val", "type": "INTEGER"},
                {"name": "date_col", "type": "DATE"},
            ]
        },
        "timePartitioning": {"type": "DAY", "field": "date_col"},
    }
    schema = extract_polars_schema(metadata)
    assert "_PARTITIONDATE" not in schema


def test_extract_polars_schema_unexpected_type():
    metadata = {"schema": {"fields": [{"name": "bad", "type": "UNKNOWN_TYPE"}]}}
    with pytest.raises(TypeError, match="got unexpected BigQuery type: unknown_type"):
        extract_polars_schema(metadata)
