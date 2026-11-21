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

    with pytest.raises(BigQueryError, match="expected table metadata object"):
        extract_polars_schema(["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(BigQueryError, match="must be an object"):
        extract_polars_schema({"schema": "bad"})

    with pytest.raises(BigQueryError, match="must be a list"):
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
