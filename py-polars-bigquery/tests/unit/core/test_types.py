from __future__ import annotations

import polars as pl
import polars_bigquery
import pytest
from polars_bigquery.core.types import (
    GEOGRAPHY_EXTENSION_METADATA,
    GEOGRAPHY_EXTENSION_NAME,
    JSON_EXTENSION_NAME,
    OPAQUE_BQ_TYPES,
    BQType,
    build_extension_dtype,
    column_width_rank,
    polars_datetime_time_unit,
    polars_dtype_to_bq_type,
)
from polars_bigquery.exceptions import UnsupportedTypeWarning


class _CustomJsonExtension(pl.BaseExtension):
    pass


class _CustomGeographyExtension(pl.BaseExtension):
    pass


def test_public_dtypes_and_warning_exports():
    assert issubclass(UnsupportedTypeWarning, UserWarning)

    json_dtype = polars_bigquery.dtypes.JSON
    assert isinstance(json_dtype, pl.Extension)
    assert json_dtype.ext_name() == JSON_EXTENSION_NAME
    assert json_dtype.ext_storage() == pl.String()
    assert json_dtype.ext_metadata() is None

    geo_dtype = polars_bigquery.dtypes.GEOGRAPHY
    assert isinstance(geo_dtype, pl.Extension)
    assert geo_dtype.ext_name() == GEOGRAPHY_EXTENSION_NAME
    assert geo_dtype.ext_storage() == pl.String()
    assert geo_dtype.ext_metadata() == GEOGRAPHY_EXTENSION_METADATA

    assert len(BQType) == 14
    assert OPAQUE_BQ_TYPES == frozenset(
        {BQType.JSON, BQType.GEOGRAPHY, BQType.ARRAY, BQType.STRUCT}
    )


def test_build_extension_dtype_default(monkeypatch):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)

    json_dtype = build_extension_dtype(JSON_EXTENSION_NAME, pl.String())
    assert json_dtype == pl.Extension(JSON_EXTENSION_NAME, pl.String())

    geo_dtype = build_extension_dtype(
        GEOGRAPHY_EXTENSION_NAME,
        pl.List(pl.String()),
        GEOGRAPHY_EXTENSION_METADATA,
    )
    assert geo_dtype == pl.Extension(
        GEOGRAPHY_EXTENSION_NAME,
        pl.List(pl.String()),
        GEOGRAPHY_EXTENSION_METADATA,
    )


def test_build_extension_dtype_mode_a_registered_subclass(monkeypatch):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    ext_name = "test:polars_bq:mode_a_json"
    pl.register_extension_type(ext_name, _CustomJsonExtension)
    try:
        dtype = build_extension_dtype(ext_name, pl.String(), '{"v": 1}')
        assert isinstance(dtype, _CustomJsonExtension)
        assert dtype.ext_name() == ext_name
        assert dtype.ext_storage() == pl.String()
        assert dtype.ext_metadata() == '{"v": 1}'
    finally:
        pl.unregister_extension_type(ext_name)


def test_build_extension_dtype_mode_b_as_storage(monkeypatch):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    ext_name = "test:polars_bq:mode_b_storage"
    pl.register_extension_type(ext_name, as_storage=True)
    try:
        dtype = build_extension_dtype(ext_name, pl.List(pl.String()))
        assert dtype == pl.List(pl.String())
    finally:
        pl.unregister_extension_type(ext_name)


def test_build_extension_dtype_mode_b_env_var(monkeypatch):
    monkeypatch.setenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    dtype = build_extension_dtype(
        GEOGRAPHY_EXTENSION_NAME,
        pl.String(),
        GEOGRAPHY_EXTENSION_METADATA,
    )
    assert dtype == pl.String()


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        # Uninstantiated scalar and container DataTypeClass objects
        (pl.Boolean, BQType.BOOL),
        (pl.Int8, BQType.INT64),
        (pl.Int16, BQType.INT64),
        (pl.Int32, BQType.INT64),
        (pl.Int64, BQType.INT64),
        (pl.Int128, BQType.INT64),
        (pl.UInt8, BQType.INT64),
        (pl.UInt16, BQType.INT64),
        (pl.UInt32, BQType.INT64),
        (pl.UInt64, BQType.INT64),
        (pl.UInt128, None),
        (pl.Float16, None),
        (pl.Float32, BQType.FLOAT64),
        (pl.Float64, BQType.FLOAT64),
        (pl.String, BQType.STRING),
        (pl.Binary, BQType.BYTES),
        (pl.Date, BQType.DATE),
        (pl.Datetime, BQType.DATETIME),
        (pl.Time, BQType.TIME),
        (pl.List, BQType.ARRAY),
        (pl.Struct, BQType.STRUCT),
        # Uninstantiated parameterized or unsupported classes must not raise
        (pl.Decimal, None),
        (pl.Array, None),
        (pl.Enum, None),
        (pl.Extension, None),
        (pl.BaseExtension, None),
        (_CustomJsonExtension, None),
        (pl.Duration, None),
        (pl.Categorical, None),
        (pl.Null, None),
        (pl.Object, None),
        # Instantiated supported DataTypes
        (pl.Boolean(), BQType.BOOL),
        (pl.Int8(), BQType.INT64),
        (pl.Int16(), BQType.INT64),
        (pl.Int32(), BQType.INT64),
        (pl.Int64(), BQType.INT64),
        (pl.Int128(), BQType.INT64),
        (pl.UInt8(), BQType.INT64),
        (pl.UInt16(), BQType.INT64),
        (pl.UInt32(), BQType.INT64),
        (pl.UInt64(), BQType.INT64),
        (pl.Float32(), BQType.FLOAT64),
        (pl.Float64(), BQType.FLOAT64),
        (pl.Decimal(38, 9), BQType.NUMERIC),
        (pl.Decimal(10, 2), BQType.NUMERIC),
        (pl.String(), BQType.STRING),
        (pl.Binary(), BQType.BYTES),
        (pl.Date(), BQType.DATE),
        (pl.Datetime("us"), BQType.DATETIME),
        (pl.Datetime("ms", None), BQType.DATETIME),
        (pl.Datetime("us", "UTC"), BQType.TIMESTAMP),
        (pl.Datetime("ns", "America/New_York"), BQType.TIMESTAMP),
        (pl.Time(), BQType.TIME),
        (pl.List(pl.Int64()), BQType.ARRAY),
        (pl.Struct({"a": pl.Int64()}), BQType.STRUCT),
        # Extension instances (scalar and outer-wrapped ARRAY)
        (polars_bigquery.dtypes.JSON, BQType.JSON),
        (polars_bigquery.dtypes.GEOGRAPHY, BQType.GEOGRAPHY),
        (
            _CustomJsonExtension(JSON_EXTENSION_NAME, pl.String()),
            BQType.JSON,
        ),
        (
            _CustomGeographyExtension(
                GEOGRAPHY_EXTENSION_NAME,
                pl.String(),
                GEOGRAPHY_EXTENSION_METADATA,
            ),
            BQType.GEOGRAPHY,
        ),
        (
            pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.String())),
            BQType.ARRAY,
        ),
        (
            pl.Extension(
                GEOGRAPHY_EXTENSION_NAME,
                pl.List(pl.String()),
                GEOGRAPHY_EXTENSION_METADATA,
            ),
            BQType.ARRAY,
        ),
        (pl.Extension("custom:unknown", pl.String()), None),
        # Instantiated unsupported DataTypes
        (pl.UInt128(), None),
        (pl.Float16(), None),
        (pl.Decimal(38, 10), None),
        (pl.Decimal(76, 38), None),
        (pl.Duration("us"), None),
        (pl.Array(pl.Int64(), 2), None),
        (pl.Categorical(), None),
        (pl.Enum(["a", "b"]), None),
        (pl.Null(), None),
        (pl.Object(), None),
    ],
)
def test_polars_dtype_to_bq_type(dtype, expected):
    assert polars_dtype_to_bq_type(dtype) is expected


def test_polars_dtype_to_bq_type_decimal_precision_none():
    # In Polars 2.x, Decimal(precision=None, scale=2) may normalize precision to 38
    # or retain None depending on the Polars release.
    defaulted = pl.Decimal(precision=None, scale=2)
    if defaulted.precision is None:
        assert polars_dtype_to_bq_type(defaulted) is None
    else:
        assert polars_dtype_to_bq_type(defaulted) is BQType.NUMERIC

    # Explicitly verify precision=None on a Decimal instance returns None.
    unspecified_precision = pl.Decimal(38, 2)
    unspecified_precision.precision = None
    assert polars_dtype_to_bq_type(unspecified_precision) is None


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Datetime, "us"),
        (pl.Datetime("ms"), "ms"),
        (pl.Datetime("us", "UTC"), "us"),
        (pl.Datetime("ns"), "ns"),
        (pl.Date, None),
        (pl.Date(), None),
        (pl.Int64, None),
        (pl.Int64(), None),
        (pl.Time(), None),
    ],
)
def test_polars_datetime_time_unit(dtype, expected):
    assert polars_datetime_time_unit(dtype) == expected


@pytest.mark.parametrize(
    ("dtype", "expected_rank"),
    [
        (pl.Boolean(), 1),
        (pl.Date(), 2),
        (pl.Int64(), 3),
        (pl.Float64(), 3),
        (pl.Time(), 3),
        (pl.Datetime("us"), 3),
        (pl.Datetime("us", "UTC"), 3),
        (pl.Decimal(38, 9), 4),
        (pl.String(), 5),
        (pl.Binary(), 5),
        (polars_bigquery.dtypes.JSON, 5),
        (polars_bigquery.dtypes.GEOGRAPHY, 5),
        (pl.List(pl.Int64()), 6),
        (pl.Struct({"a": pl.Int64()}), 6),
        (pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.String())), 6),
        (
            pl.Extension(
                GEOGRAPHY_EXTENSION_NAME,
                pl.List(pl.String()),
                GEOGRAPHY_EXTENSION_METADATA,
            ),
            6,
        ),
        (pl.Duration("us"), 6),
        (pl.Float16(), 6),
        (pl.Decimal(76, 38), 6),
    ],
)
def test_column_width_rank(dtype, expected_rank):
    assert column_width_rank(dtype) == expected_rank
