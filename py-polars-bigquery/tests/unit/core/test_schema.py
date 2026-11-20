from __future__ import annotations

import polars as pl
import polars_bigquery
import pytest
from polars_bigquery.core.schema import (
    ExtractedSchema,
    SkippedColumn,
    extract_polars_schema,
    extract_schema_metadata,
)
from polars_bigquery.core.types import (
    GEOGRAPHY_EXTENSION_METADATA,
    GEOGRAPHY_EXTENSION_NAME,
    JSON_EXTENSION_NAME,
    BQType,
)


def test_extract_polars_schema_primitive_types(monkeypatch):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    metadata = {
        "schema": {
            "fields": [
                {"name": "bool_col", "type": "BOOLEAN"},
                {"name": "bool_alias", "type": "BOOL"},
                {"name": "bytes_col", "type": "BYTES"},
                {"name": "date_col", "type": "DATE"},
                {"name": "datetime_col", "type": "DATETIME"},
                {"name": "geo_col", "type": "GEOGRAPHY"},
                {"name": "json_col", "type": "JSON"},
                {"name": "float_col", "type": "FLOAT"},
                {"name": "float64_col", "type": "FLOAT64"},
                {"name": "int_col", "type": "INTEGER"},
                {"name": "int64_col", "type": "INT64"},
                {"name": "numeric_col", "type": "NUMERIC"},
                {"name": "decimal_col", "type": "DECIMAL"},
                {"name": "string_col", "type": "STRING"},
                {"name": "time_col", "type": "TIME"},
                {"name": "timestamp_col", "type": "TIMESTAMP"},
            ]
        }
    }
    extracted = extract_schema_metadata(metadata)
    assert isinstance(extracted, ExtractedSchema)
    assert extracted.skipped_columns == ()

    schema = extracted.polars_schema
    assert schema == extract_polars_schema(metadata)
    assert schema["bool_col"] == pl.Boolean()
    assert schema["bool_alias"] == pl.Boolean()
    assert schema["bytes_col"] == pl.Binary()
    assert schema["date_col"] == pl.Date()
    assert schema["datetime_col"] == pl.Datetime(time_unit="us")
    assert schema["geo_col"] == polars_bigquery.dtypes.GEOGRAPHY
    assert schema["json_col"] == polars_bigquery.dtypes.JSON
    assert schema["float_col"] == pl.Float64()
    assert schema["float64_col"] == pl.Float64()
    assert schema["int_col"] == pl.Int64()
    assert schema["int64_col"] == pl.Int64()
    assert schema["numeric_col"] == pl.Decimal(precision=38, scale=9)
    assert schema["decimal_col"] == pl.Decimal(precision=38, scale=9)
    assert schema["string_col"] == pl.String()
    assert schema["time_col"] == pl.Time()
    assert schema["timestamp_col"] == pl.Datetime(time_unit="us", time_zone="UTC")

    assert extracted.bq_types == {
        "bool_col": BQType.BOOL,
        "bool_alias": BQType.BOOL,
        "bytes_col": BQType.BYTES,
        "date_col": BQType.DATE,
        "datetime_col": BQType.DATETIME,
        "geo_col": BQType.GEOGRAPHY,
        "json_col": BQType.JSON,
        "float_col": BQType.FLOAT64,
        "float64_col": BQType.FLOAT64,
        "int_col": BQType.INT64,
        "int64_col": BQType.INT64,
        "numeric_col": BQType.NUMERIC,
        "decimal_col": BQType.NUMERIC,
        "string_col": BQType.STRING,
        "time_col": BQType.TIME,
        "timestamp_col": BQType.TIMESTAMP,
    }


def test_extract_polars_schema_repeated_and_extension_nesting(monkeypatch):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    metadata = {
        "schema": {
            "fields": [
                {"name": "int_array", "type": "INTEGER", "mode": "REPEATED"},
                {"name": "json_array", "type": "JSON", "mode": "REPEATED"},
                {"name": "geo_array", "type": "GEOGRAPHY", "mode": "REPEATED"},
                {
                    "name": "struct_with_ext_arrays",
                    "type": "STRUCT",
                    "fields": [
                        {"name": "j", "type": "JSON"},
                        {"name": "g", "type": "GEOGRAPHY"},
                        {"name": "j_arr", "type": "JSON", "mode": "REPEATED"},
                        {"name": "g_arr", "type": "GEOGRAPHY", "mode": "REPEATED"},
                    ],
                },
                {
                    "name": "array_of_structs",
                    "type": "RECORD",
                    "mode": "REPEATED",
                    "fields": [
                        {"name": "j", "type": "JSON"},
                        {"name": "g", "type": "GEOGRAPHY"},
                    ],
                },
            ]
        }
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.skipped_columns == ()
    schema = extracted.polars_schema

    expected_json_arr = pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.String()))
    expected_geo_arr = pl.Extension(
        GEOGRAPHY_EXTENSION_NAME,
        pl.List(pl.String()),
        GEOGRAPHY_EXTENSION_METADATA,
    )

    assert schema["int_array"] == pl.List(pl.Int64())
    assert schema["json_array"] == expected_json_arr
    assert schema["geo_array"] == expected_geo_arr
    assert schema["struct_with_ext_arrays"] == pl.Struct(
        [
            pl.Field("j", polars_bigquery.dtypes.JSON),
            pl.Field("g", polars_bigquery.dtypes.GEOGRAPHY),
            pl.Field("j_arr", expected_json_arr),
            pl.Field("g_arr", expected_geo_arr),
        ]
    )
    assert schema["array_of_structs"] == pl.List(
        pl.Struct(
            [
                pl.Field("j", polars_bigquery.dtypes.JSON),
                pl.Field("g", polars_bigquery.dtypes.GEOGRAPHY),
            ]
        )
    )
    assert extracted.bq_types == {
        "int_array": BQType.ARRAY,
        "json_array": BQType.ARRAY,
        "geo_array": BQType.ARRAY,
        "struct_with_ext_arrays": BQType.STRUCT,
        "array_of_structs": BQType.ARRAY,
    }


def test_extract_schema_preserves_bq_types_under_mode_b(monkeypatch):
    monkeypatch.setenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    metadata = {
        "schema": {
            "fields": [
                {"name": "j", "type": "JSON"},
                {"name": "g", "type": "GEOGRAPHY"},
                {"name": "j_arr", "type": "JSON", "mode": "REPEATED"},
                {"name": "g_arr", "type": "GEOGRAPHY", "mode": "REPEATED"},
            ]
        }
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema(
        {
            "j": pl.String(),
            "g": pl.String(),
            "j_arr": pl.List(pl.String()),
            "g_arr": pl.List(pl.String()),
        }
    )
    assert extracted.bq_types == {
        "j": BQType.JSON,
        "g": BQType.GEOGRAPHY,
        "j_arr": BQType.ARRAY,
        "g_arr": BQType.ARRAY,
    }


def test_extract_schema_preserves_bq_types_under_registered_as_storage(monkeypatch):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    pl.register_extension_type(JSON_EXTENSION_NAME, as_storage=True)
    try:
        metadata = {
            "schema": {
                "fields": [
                    {"name": "j", "type": "JSON"},
                    {"name": "j_arr", "type": "JSON", "mode": "REPEATED"},
                ]
            }
        }
        extracted = extract_schema_metadata(metadata)
        assert extracted.polars_schema == pl.Schema(
            {"j": pl.String(), "j_arr": pl.List(pl.String())}
        )
        assert extracted.bq_types == {"j": BQType.JSON, "j_arr": BQType.ARRAY}
    finally:
        pl.unregister_extension_type(JSON_EXTENSION_NAME)


def test_extract_schema_timestamp_precision_and_repeated_parameter_preservation():
    metadata = {
        "schema": {
            "fields": [
                {"name": "ts_default", "type": "TIMESTAMP"},
                {"name": "ts_int6", "type": "TIMESTAMP", "timestampPrecision": 6},
                {"name": "ts_str6", "type": "TIMESTAMP", "timestampPrecision": "6"},
                {
                    "name": "ts_arr_ok",
                    "type": "TIMESTAMP",
                    "mode": "REPEATED",
                    "timestampPrecision": "6",
                },
                {"name": "ts_int12", "type": "TIMESTAMP", "timestampPrecision": 12},
                {"name": "ts_str12", "type": "TIMESTAMP", "timestampPrecision": "12"},
                {"name": "ts_str9", "type": "TIMESTAMP", "timestampPrecision": "9"},
                {"name": "ts_bool", "type": "TIMESTAMP", "timestampPrecision": True},
                {
                    "name": "ts_arr_pico",
                    "type": "TIMESTAMP",
                    "mode": "REPEATED",
                    "timestampPrecision": "12",
                },
            ]
        }
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema(
        {
            "ts_default": pl.Datetime("us", "UTC"),
            "ts_int6": pl.Datetime("us", "UTC"),
            "ts_str6": pl.Datetime("us", "UTC"),
            "ts_arr_ok": pl.List(pl.Datetime("us", "UTC")),
        }
    )
    assert extracted.skipped_columns == (
        SkippedColumn("ts_int12", "TIMESTAMP(precision=12)", "ts_int12"),
        SkippedColumn("ts_str12", "TIMESTAMP(precision=12)", "ts_str12"),
        SkippedColumn("ts_str9", "TIMESTAMP(precision=9)", "ts_str9"),
        SkippedColumn("ts_bool", "TIMESTAMP(precision=True)", "ts_bool"),
        SkippedColumn("ts_arr_pico", "TIMESTAMP(precision=12)", "ts_arr_pico"),
    )


def test_extract_schema_omits_unsupported_top_level_and_nested_columns():
    metadata = {
        "schema": {
            "fields": [
                {"name": "id", "type": "INT64"},
                {"name": "bn", "type": "BIGNUMERIC"},
                {"name": "bd", "type": "BIGDECIMAL"},
                {"name": "dur", "type": "INTERVAL"},
                {"name": "rng", "type": "RANGE"},
                {"name": "unk", "type": "UNKNOWN_TYPE"},
                {"name": "missing_type", "type": None},
                {"name": "empty_struct", "type": "STRUCT", "fields": []},
                {"name": "null_struct", "type": "RECORD", "fields": None},
                {
                    "name": "unnamed_sub",
                    "type": "STRUCT",
                    "fields": [
                        {"name": "ok", "type": "INT64"},
                        {"type": "STRING"},
                    ],
                },
                {
                    "name": "dup_sub",
                    "type": "STRUCT",
                    "fields": [
                        {"name": "x", "type": "INT64"},
                        {"name": "x", "type": "STRING"},
                    ],
                },
                {
                    "name": "nested_bad",
                    "type": "RECORD",
                    "fields": [
                        {"name": "ok", "type": "INT64"},
                        {
                            "name": "inner",
                            "type": "STRUCT",
                            "fields": [{"name": "bad", "type": "INTERVAL"}],
                        },
                    ],
                },
                {"type": "INT64"},
                {"name": "", "type": "STRING"},
            ]
        }
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema({"id": pl.Int64()})
    assert extracted.bq_types == {"id": BQType.INT64}
    assert extracted.skipped_columns == (
        SkippedColumn("bn", "BIGNUMERIC", "bn"),
        SkippedColumn("bd", "BIGNUMERIC", "bd"),
        SkippedColumn("dur", "INTERVAL", "dur"),
        SkippedColumn("rng", "RANGE", "rng"),
        SkippedColumn("unk", "unsupported type unknown_type", "unk"),
        SkippedColumn("missing_type", "unsupported type <missing>", "missing_type"),
        SkippedColumn("empty_struct", "empty STRUCT", "empty_struct"),
        SkippedColumn("null_struct", "empty STRUCT", "null_struct"),
        SkippedColumn("unnamed_sub", "unnamed STRUCT subfield", "unnamed_sub[1]"),
        SkippedColumn("dup_sub", "duplicate STRUCT subfield", "dup_sub.x"),
        SkippedColumn("nested_bad", "INTERVAL", "nested_bad.inner.bad"),
        SkippedColumn("<unnamed[12]>", "unnamed column", "<unnamed[12]>"),
        SkippedColumn("<unnamed[13]>", "unnamed column", "<unnamed[13]>"),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"schema": None},
        {"schema": {}},
        {"schema": {"fields": None}},
        {"schema": {"fields": []}},
    ],
)
def test_extract_schema_empty_and_null_safe_metadata(metadata):
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema({})
    assert extracted.bq_types == {}
    assert extracted.skipped_columns == ()


def test_extract_schema_null_mode_is_treated_as_scalar():
    metadata = {
        "schema": {
            "fields": [{"name": "col", "type": "STRING", "mode": None}],
        }
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema({"col": pl.String()})
    assert extracted.bq_types == {"col": BQType.STRING}


@pytest.mark.parametrize(
    ("time_partitioning", "expected_pseudo_names"),
    [
        ({"type": "DAY"}, ["_PARTITIONDATE", "_PARTITIONTIME"]),
        ({}, ["_PARTITIONDATE", "_PARTITIONTIME"]),
        ({"type": None}, ["_PARTITIONDATE", "_PARTITIONTIME"]),
        ({"type": "HOUR"}, ["_PARTITIONTIME"]),
        ({"type": "MONTH"}, ["_PARTITIONTIME"]),
        ({"type": "YEAR"}, ["_PARTITIONTIME"]),
        ({"type": "DAY", "field": "date_col"}, []),
    ],
)
def test_extract_schema_ingestion_time_partitioning_granularity(
    time_partitioning, expected_pseudo_names
):
    metadata = {
        "schema": {"fields": [{"name": "val", "type": "INTEGER"}]},
        "timePartitioning": time_partitioning,
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema.names() == ["val", *expected_pseudo_names]
    if "_PARTITIONDATE" in expected_pseudo_names:
        assert extracted.polars_schema["_PARTITIONDATE"] == pl.Date()
        assert extracted.bq_types["_PARTITIONDATE"] is BQType.DATE
    if "_PARTITIONTIME" in expected_pseudo_names:
        assert extracted.polars_schema["_PARTITIONTIME"] == pl.Datetime("us", "UTC")
        assert extracted.bq_types["_PARTITIONTIME"] is BQType.TIMESTAMP


def test_extract_schema_all_unsupported_physical_with_ingestion_time_partitioning():
    metadata = {
        "schema": {"fields": [{"name": "dur", "type": "INTERVAL"}]},
        "timePartitioning": {"type": "HOUR"},
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema(
        {"_PARTITIONTIME": pl.Datetime("us", "UTC")}
    )
    assert extracted.bq_types == {"_PARTITIONTIME": BQType.TIMESTAMP}
    assert extracted.skipped_columns == (SkippedColumn("dur", "INTERVAL", "dur"),)
