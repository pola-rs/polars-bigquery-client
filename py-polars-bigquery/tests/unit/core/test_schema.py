import polars as pl
import pytest
from polars_bigquery.core.schema import (
    extract_polars_schema,
    extract_polars_schema_from_fields,
)


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
    assert schema.names() == ["val", "_PARTITIONDATE", "_PARTITIONTIME"]
    assert schema["val"] == pl.Int64()
    assert schema["_PARTITIONDATE"] == pl.Date()
    assert schema["_PARTITIONTIME"] == pl.Datetime(time_unit="us", time_zone="utc")


def test_extract_polars_schema_extended_types():
    metadata = {
        "schema": {
            "fields": [
                {"name": "bignumeric_col", "type": "BIGNUMERIC"},
                {"name": "bigdecimal_col", "type": "BIGDECIMAL"},
                {"name": "json_col", "type": "JSON"},
                {"name": "interval_col", "type": "INTERVAL"},
            ]
        }
    }
    schema = extract_polars_schema(metadata)
    assert schema["bignumeric_col"] == pl.Decimal(precision=76, scale=38)
    assert schema["bigdecimal_col"] == pl.Decimal(precision=76, scale=38)
    assert schema["json_col"] == pl.String()
    assert schema["interval_col"] == pl.Duration(time_unit="us")


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


def test_extract_polars_schema_duplicate_column_names():
    metadata = {
        "schema": {
            "fields": [
                {"name": "a", "type": "INTEGER"},
                {"name": "b", "type": "STRING"},
                {"name": "a", "type": "STRING"},
                {"name": "a", "type": "FLOAT"},
                {"name": "b", "type": "BOOLEAN"},
            ]
        }
    }
    with pytest.raises(
        ValueError,
        match=r"BigQuery schema contains duplicate column name\(s\): 'a', 'b'\.",
    ):
        extract_polars_schema(metadata)


def test_extract_polars_schema_from_fields_standard_and_nested():
    fields = [
        {"name": "id", "type": "INT64"},
        {"name": "tags", "type": "STRING", "mode": "REPEATED"},
        {
            "name": "info",
            "type": "STRUCT",
            "fields": [{"name": "score", "type": "FLOAT64"}],
        },
    ]
    schema = extract_polars_schema_from_fields(fields)
    assert schema == pl.Schema(
        {
            "id": pl.Int64(),
            "tags": pl.List(pl.String()),
            "info": pl.Struct([pl.Field("score", pl.Float64())]),
        }
    )
    assert "_PARTITIONDATE" not in schema
    assert "_PARTITIONTIME" not in schema


def test_extract_polars_schema_from_fields_duplicate_names_raises():
    fields = [
        {"name": "a", "type": "INT64"},
        {"name": "a", "type": "STRING"},
    ]
    with pytest.raises(
        ValueError,
        match=r"BigQuery schema contains duplicate column name\(s\): 'a'\.",
    ):
        extract_polars_schema_from_fields(fields)


def test_extract_polars_schema_from_fields_case_differing_names_allowed():
    fields = [
        {"name": "a", "type": "INT64"},
        {"name": "A", "type": "STRING"},
    ]
    schema = extract_polars_schema_from_fields(fields)
    assert schema == pl.Schema({"a": pl.Int64(), "A": pl.String()})


def test_extract_polars_schema_from_fields_empty():
    schema = extract_polars_schema_from_fields([])
    assert schema == pl.Schema()
