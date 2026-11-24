from __future__ import annotations

import polars as pl
import polars_bigquery
import pytest
from polars_bigquery.core.schema import (
    ExtractedSchema,
    SkippedColumn,
    extract_polars_schema,
    extract_polars_schema_from_fields,
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


def test_extract_schema_deeply_nested_structs_with_datetime_subfields(monkeypatch):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)

    # Multi-level STRUCT -> STRUCT -> ARRAY<STRUCT<dt DATETIME, dt_arr ARRAY<DATETIME>, ...>>
    nested_field: dict = {
        "name": "outer",
        "type": "STRUCT",
        "fields": [
            {
                "name": "mid",
                "type": "RECORD",
                "fields": [
                    {
                        "name": "items",
                        "type": "RECORD",
                        "mode": "REPEATED",
                        "fields": [
                            {"name": "dt", "type": "DATETIME"},
                            {"name": "dt_arr", "type": "DATETIME", "mode": "REPEATED"},
                            {"name": "j", "type": "JSON"},
                            {"name": "g", "type": "GEOGRAPHY"},
                        ],
                    }
                ],
            }
        ],
    }

    # 15-level deeply nested STRUCT (BigQuery's maximum RECORD nesting depth)
    deep_field: dict = {
        "name": "leaf",
        "type": "STRUCT",
        "fields": [
            {"name": "dt", "type": "DATETIME"},
            {"name": "dt_arr", "type": "DATETIME", "mode": "REPEATED"},
        ],
    }
    expected_deep_dtype: pl.DataType = pl.Struct(
        [
            pl.Field("dt", pl.Datetime(time_unit="us")),
            pl.Field("dt_arr", pl.List(pl.Datetime(time_unit="us"))),
        ]
    )
    for level in range(13, -1, -1):
        sub_name = f"level_{level}"
        deep_field = {
            "name": sub_name,
            "type": "STRUCT",
            "fields": [{**deep_field, "name": sub_name}],
        }
        expected_deep_dtype = pl.Struct([pl.Field(sub_name, expected_deep_dtype)])
    deep_field["name"] = "deep"

    extracted = extract_schema_metadata(
        {"schema": {"fields": [nested_field, deep_field]}}
    )
    assert extracted.skipped_columns == ()
    assert extracted.bq_types == {"outer": BQType.STRUCT, "deep": BQType.STRUCT}
    assert extracted.polars_schema["outer"] == pl.Struct(
        [
            pl.Field(
                "mid",
                pl.Struct(
                    [
                        pl.Field(
                            "items",
                            pl.List(
                                pl.Struct(
                                    [
                                        pl.Field("dt", pl.Datetime(time_unit="us")),
                                        pl.Field(
                                            "dt_arr",
                                            pl.List(pl.Datetime(time_unit="us")),
                                        ),
                                        pl.Field("j", polars_bigquery.dtypes.JSON),
                                        pl.Field("g", polars_bigquery.dtypes.GEOGRAPHY),
                                    ]
                                )
                            ),
                        )
                    ]
                ),
            )
        ]
    )
    assert extracted.polars_schema["deep"] == expected_deep_dtype


def test_extract_schema_preserves_bq_types_under_mode_b(monkeypatch):
    import polars_bigquery.core.types as types_mod

    monkeypatch.setattr(
        types_mod, "_INITIAL_UNKNOWN_EXTENSION_BEHAVIOR", "load_as_storage"
    )
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


def test_extract_schema_hardens_malformed_deep_and_duplicate_metadata():
    # Malformed non-mapping / wrong-typed envelopes fail fast with ValueError so
    # corrupted HTTP responses never silently turn into 0-row empty table scans.
    with pytest.raises(TypeError, match="expected a mapping"):
        extract_schema_metadata("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected 'schema' to be a mapping"):
        extract_schema_metadata({"schema": "bad"})
    with pytest.raises(ValueError, match="expected 'schema.fields' to be a list"):
        extract_schema_metadata({"schema": {"fields": "bad"}})
    with pytest.raises(ValueError, match="expected 'timePartitioning' to be a mapping"):
        extract_schema_metadata({"timePartitioning": "bad"})
    with pytest.raises(
        ValueError, match=r"expected 'timePartitioning\.field' to be a string"
    ):
        extract_schema_metadata({"timePartitioning": {"field": 123}})
    with pytest.raises(
        ValueError, match=r"expected 'timePartitioning\.type' to be a string"
    ):
        extract_schema_metadata({"timePartitioning": {"type": 123}})
    with pytest.raises(ValueError, match="unsupported 'timePartitioning.type'"):
        extract_schema_metadata({"timePartitioning": {"type": "WEEK"}})

    # Deeply nested STRUCT beyond MAX_SCHEMA_NESTING_DEPTH skips without RecursionError
    from polars_bigquery.core.schema import MAX_SCHEMA_NESTING_DEPTH

    deep: dict = {"name": "leaf", "type": "INT64"}
    for idx in range(MAX_SCHEMA_NESTING_DEPTH + 2):
        deep = {"name": f"s{idx}", "type": "STRUCT", "fields": [deep]}

    metadata = {
        "schema": {
            "fields": [
                {"name": "ok", "type": "INT64"},
                {"name": "ok", "type": "STRING"},
                {"name": "OK", "type": "STRING"},
                "not-a-mapping",
                {"name": 123, "type": "STRING"},
                {"name": "bad_mode", "type": "STRING", "mode": 42},
                {"name": "bad_type", "type": 99},
                {"name": "ts_float6", "type": "TIMESTAMP", "timestampPrecision": 6.0},
                {"name": "bad_fields", "type": "STRUCT", "fields": "not-a-list"},
                {"name": "bad_sub", "type": "STRUCT", "fields": ["not-a-dict"]},
                {
                    "name": "dup_case_sub",
                    "type": "STRUCT",
                    "fields": [
                        {"name": "x", "type": "INT64"},
                        {"name": "X", "type": "STRING"},
                    ],
                },
                deep,
            ]
        }
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema({"ok": pl.Int64()})
    reasons = {col.name: col.reason for col in extracted.skipped_columns}
    assert reasons["ok"] == "duplicate column"
    assert reasons["OK"] == "duplicate column"
    assert reasons["bad_mode"] == "invalid mode 42"
    assert reasons["bad_type"] == "invalid type 99"
    assert reasons["ts_float6"] == "TIMESTAMP(precision=6.0)"
    assert reasons["bad_fields"] == "malformed STRUCT fields"
    assert reasons["bad_sub"] == "malformed STRUCT subfield"
    assert reasons["dup_case_sub"] == "duplicate STRUCT subfield"
    assert f"exceeds {MAX_SCHEMA_NESTING_DEPTH}" in reasons[deep["name"]]


def test_extract_schema_extension_builder_non_datatype_and_lookup_error():
    metadata = {"schema": {"fields": [{"name": "j", "type": "JSON"}]}}

    extracted_none = extract_schema_metadata(
        metadata,
        extension_dtype_builder=lambda _n, _s, _m: None,  # type: ignore[return-value]
    )
    assert extracted_none.polars_schema == pl.Schema({})
    assert len(extracted_none.skipped_columns) == 1
    assert "non-DataType NoneType" in extracted_none.skipped_columns[0].reason

    def _raise_key_error(_n, _s, _m):
        raise KeyError("missing config")

    extracted_err = extract_schema_metadata(
        metadata,
        extension_dtype_builder=_raise_key_error,
    )
    assert extracted_err.polars_schema == pl.Schema({})
    assert len(extracted_err.skipped_columns) == 1
    assert "rejected storage" in extracted_err.skipped_columns[0].reason


def test_extract_schema_custom_extension_rejection_becomes_skipped_column():
    class _ScalarOnlyJson(pl.BaseExtension):
        @classmethod
        def ext_from_params(cls, name, storage, metadata):
            if isinstance(storage, pl.List):
                raise TypeError("scalar String storage required")
            return cls(name, storage, metadata)

    pl.register_extension_type(JSON_EXTENSION_NAME, _ScalarOnlyJson)
    try:
        metadata = {
            "schema": {
                "fields": [
                    {"name": "j_scalar", "type": "JSON"},
                    {"name": "j_array", "type": "JSON", "mode": "REPEATED"},
                ]
            }
        }
        extracted = extract_schema_metadata(metadata)
        assert list(extracted.polars_schema.keys()) == ["j_scalar"]
        assert len(extracted.skipped_columns) == 1
        assert extracted.skipped_columns[0].name == "j_array"
        assert "rejected storage" in extracted.skipped_columns[0].reason
    finally:
        pl.unregister_extension_type(JSON_EXTENSION_NAME)


def test_extract_schema_injected_extension_dtype_builder():
    metadata = {
        "schema": {
            "fields": [
                {"name": "j", "type": "JSON"},
                {"name": "j_arr", "type": "JSON", "mode": "REPEATED"},
            ]
        }
    }
    extracted = extract_schema_metadata(
        metadata,
        extension_dtype_builder=lambda _name, storage, _meta: storage,
    )
    assert extracted.polars_schema == pl.Schema(
        {"j": pl.String(), "j_arr": pl.List(pl.String())}
    )
    assert extracted.bq_types == {"j": BQType.JSON, "j_arr": BQType.ARRAY}


def test_extract_schema_empty_partition_field_and_no_overwrite():
    metadata = {
        "schema": {
            "fields": [
                {"name": "val", "type": "INT64"},
                {"name": "_PARTITIONDATE", "type": "STRING"},
            ]
        },
        "timePartitioning": {"type": "DAY", "field": ""},
    }
    extracted = extract_schema_metadata(metadata)
    # Empty field "" means ingestion-time partitioned, so _PARTITIONTIME is added,
    # while an existing physical column named _PARTITIONDATE is not overwritten.
    assert extracted.polars_schema == pl.Schema(
        {
            "val": pl.Int64(),
            "_PARTITIONDATE": pl.String(),
            "_PARTITIONTIME": pl.Datetime("us", "UTC"),
        }
    )


def test_extract_schema_duplicate_after_skipped_and_unknown_mode():
    metadata = {
        "schema": {
            "fields": [
                {"name": "ok", "type": "INT64"},
                # Unsupported first occurrence must still reserve the name against duplicates.
                {"name": "dup", "type": "INTERVAL"},
                {"name": "DUP", "type": "INT64"},
                # Skipped column named _PARTITIONTIME must block pseudo-column shadowing.
                {"name": "_PARTITIONTIME", "type": "INTERVAL"},
                # Unrecognized mode fails closed as skipped instead of defaulting to scalar.
                {"name": "bad_mode", "type": "STRING", "mode": "STREAMING"},
            ]
        },
        "timePartitioning": {"type": "DAY"},
    }
    extracted = extract_schema_metadata(metadata)
    assert extracted.polars_schema == pl.Schema(
        {"ok": pl.Int64(), "_PARTITIONDATE": pl.Date()}
    )
    assert "_PARTITIONTIME" not in extracted.polars_schema
    assert [(c.name, c.reason) for c in extracted.skipped_columns] == [
        ("dup", "INTERVAL"),
        ("DUP", "duplicate column"),
        ("_PARTITIONTIME", "INTERVAL"),
        ("bad_mode", "unsupported mode 'STREAMING'"),
    ]


def test_extract_polars_schema_from_fields_unexpected_type():
    fields = [{"name": "bad", "type": "UNKNOWN_TYPE"}]
    with pytest.raises(TypeError, match="got unexpected BigQuery type: unknown_type"):
        extract_polars_schema_from_fields(fields)


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
        {"name": "b", "type": "STRING"},
        {"name": "a", "type": "STRING"},
        {"name": "a", "type": "FLOAT"},
        {"name": "b", "type": "BOOLEAN"},
    ]
    with pytest.raises(
        ValueError,
        match=r"BigQuery schema contains duplicate column name\(s\): 'a', 'b'\.",
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


def test_extract_polars_schema_rejects_duplicate_struct_subfields():
    fields = [
        {
            "name": "info",
            "type": "STRUCT",
            "fields": [
                {"name": "x", "type": "INT64"},
                {"name": "x", "type": "STRING"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="duplicate subfield name.*'x'"):
        extract_polars_schema_from_fields(fields)

    unnamed_fields = [
        {
            "name": "anon_struct",
            "type": "RECORD",
            "fields": [
                {"name": "", "type": "INT64"},
                {"name": "", "type": "STRING"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="duplicate subfield name.*''"):
        extract_polars_schema_from_fields(unnamed_fields)


def test_extract_polars_schema_handles_null_and_malformed_metadata():
    from polars_bigquery.exceptions import BigQueryError

    # None mode / schema / fields degrade cleanly instead of AttributeError.
    assert extract_polars_schema({"schema": None}) == pl.Schema()
    assert extract_polars_schema({"schema": {"fields": None}}) == pl.Schema()
    assert extract_polars_schema_from_fields(
        [{"name": "a", "type": "INT64", "mode": None}]
    ) == pl.Schema({"a": pl.Int64()})
    assert extract_polars_schema_from_fields(
        [{"name": "s", "type": "STRUCT", "fields": None}]
    ) == pl.Schema({"s": pl.Struct([])})

    with pytest.raises(TypeError, match="got unexpected BigQuery type: "):
        extract_polars_schema_from_fields([{"name": "a", "type": None}])

    with pytest.raises(TypeError, match="expected a mapping"):
        extract_polars_schema(["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expected 'schema' to be a mapping"):
        extract_polars_schema({"schema": "bad"})

    with pytest.raises(ValueError, match="expected 'schema.fields' to be a list"):
        extract_polars_schema({"schema": {"fields": "bad"}})

    with pytest.raises(BigQueryError, match="expected schema field object"):
        extract_polars_schema_from_fields(["not_a_dict"])  # type: ignore[list-item]

    with pytest.raises(BigQueryError, match="field 'name' must be a string"):
        extract_polars_schema_from_fields([{"name": 123, "type": "INT64"}])

    with pytest.raises(BigQueryError, match="field 'mode' must be a string"):
        extract_polars_schema_from_fields([{"name": "a", "type": "INT64", "mode": 42}])

    with pytest.raises(BigQueryError, match="field 'type' must be a string"):
        extract_polars_schema_from_fields([{"name": "a", "type": 42}])

    with pytest.raises(BigQueryError, match="STRUCT 'fields' must be a list"):
        extract_polars_schema_from_fields(
            [{"name": "s", "type": "STRUCT", "fields": "bad"}]
        )

    with pytest.raises(BigQueryError, match="expected STRUCT subfield object"):
        extract_polars_schema_from_fields(
            [{"name": "s", "type": "STRUCT", "fields": ["bad"]}]
        )

    with pytest.raises(BigQueryError, match="STRUCT subfield 'name' must be a string"):
        extract_polars_schema_from_fields(
            [
                {
                    "name": "s",
                    "type": "STRUCT",
                    "fields": [{"name": 7, "type": "INT64"}],
                }
            ]
        )

    with pytest.raises(BigQueryError, match="field 'name' must be a string"):
        extract_polars_schema_from_fields([{"type": "INT64"}])

    with pytest.raises(BigQueryError, match="must be a non-empty string"):
        extract_polars_schema_from_fields([{"name": "", "type": "INT64"}])

    with pytest.raises(BigQueryError, match="did not contain any schema fields"):
        extract_polars_schema({"schema": {"fields": []}}, require_non_empty=True)


def test_extract_polars_schema_rejects_excessive_nesting_depth():
    from polars_bigquery.core.schema import MAX_SCHEMA_NESTING_DEPTH
    from polars_bigquery.exceptions import BigQueryError

    node: dict = {"name": "leaf", "type": "INT64"}
    for _ in range(MAX_SCHEMA_NESTING_DEPTH + 2):
        node = {"name": "s", "type": "STRUCT", "fields": [node]}

    with pytest.raises(BigQueryError, match="nesting depth exceeds maximum"):
        extract_polars_schema_from_fields([node])


def test_has_bignumeric_dtype_nested():
    from polars_bigquery.core.schema import has_bignumeric_dtype

    bn = pl.Decimal(precision=76, scale=38)
    num = pl.Decimal(precision=38, scale=9)
    assert has_bignumeric_dtype(bn) is True
    assert has_bignumeric_dtype(pl.List(bn)) is True
    assert has_bignumeric_dtype(pl.Array(bn, 2)) is True
    assert has_bignumeric_dtype(pl.Struct([pl.Field("x", pl.List(bn))])) is True
    assert has_bignumeric_dtype(num) is False
    assert has_bignumeric_dtype(pl.List(num)) is False
    assert has_bignumeric_dtype(pl.Struct([pl.Field("x", pl.Int64())])) is False
    if hasattr(pl, "Map"):
        assert has_bignumeric_dtype(pl.Map(pl.String(), bn)) is True
        assert has_bignumeric_dtype(pl.Map(pl.String(), pl.Int64())) is False


def test_extract_polars_schema_from_fields_types_and_strict_rejections(monkeypatch):
    from polars_bigquery.exceptions import BigQueryError

    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    schema = extract_polars_schema_from_fields(
        [
            {"name": "b", "type": "BOOL"},
            {"name": "ts", "type": "TIMESTAMP", "timestampPrecision": 6},
            {
                "name": "ts_arr",
                "type": "TIMESTAMP",
                "mode": "REPEATED",
                "timestampPrecision": "6",
            },
            {"name": "j", "type": "JSON"},
            {"name": "g", "type": "GEOGRAPHY"},
            {"name": "j_arr", "type": "JSON", "mode": "REPEATED"},
            {"name": "g_arr", "type": "GEOGRAPHY", "mode": "REPEATED"},
        ]
    )
    assert schema == pl.Schema(
        {
            "b": pl.Boolean(),
            "ts": pl.Datetime("us", "UTC"),
            "ts_arr": pl.List(pl.Datetime("us", "UTC")),
            "j": polars_bigquery.dtypes.JSON,
            "g": polars_bigquery.dtypes.GEOGRAPHY,
            "j_arr": pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.String())),
            "g_arr": pl.Extension(
                GEOGRAPHY_EXTENSION_NAME,
                pl.List(pl.String()),
                GEOGRAPHY_EXTENSION_METADATA,
            ),
        }
    )

    with pytest.raises(TypeError, match="INTERVAL column 'dur'"):
        extract_polars_schema_from_fields([{"name": "dur", "type": "INTERVAL"}])
    with pytest.raises(TypeError, match="RANGE column 'rng'"):
        extract_polars_schema_from_fields([{"name": "rng", "type": "RANGE"}])
    with pytest.raises(TypeError, match="BIGNUMERIC/BIGDECIMAL.*'bd'"):
        extract_polars_schema_from_fields([{"name": "bd", "type": "BIGDECIMAL"}])
    with pytest.raises(TypeError, match=r"TIMESTAMP\(precision=12\).*'ts_pico'"):
        extract_polars_schema_from_fields(
            [
                {
                    "name": "ts_pico",
                    "type": "TIMESTAMP",
                    "mode": "REPEATED",
                    "timestampPrecision": "12",
                }
            ]
        )
    with pytest.raises(BigQueryError, match="unsupported field 'mode' 'STREAMING'"):
        extract_polars_schema_from_fields(
            [{"name": "bad", "type": "STRING", "mode": "STREAMING"}]
        )
