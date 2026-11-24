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
    has_bignumeric_dtype,
    polars_datetime_time_unit,
    polars_dtype_to_bq_type,
)
from polars_bigquery.exceptions import UnsupportedTypeWarning


class _CustomJsonExtension(pl.BaseExtension):
    pass


class _CustomGeographyExtension(pl.BaseExtension):
    pass


def test_public_dtypes_and_warning_exports():
    assert "dtypes" in polars_bigquery.__all__
    assert polars_bigquery.dtypes.__all__ == ["GEOGRAPHY", "JSON"]
    assert "UnsupportedTypeWarning" in polars_bigquery.exceptions.__all__
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
    import polars_bigquery.core.types as types_mod

    monkeypatch.setattr(
        types_mod, "_INITIAL_UNKNOWN_EXTENSION_BEHAVIOR", "load_as_storage"
    )
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
    # (36 integer digits > BigQuery NUMERIC max of 29 -> None), while scale=9 normalizes
    # to Decimal(38, 9) (29 integer digits -> BQType.NUMERIC).
    defaulted_scale2 = pl.Decimal(precision=None, scale=2)
    assert polars_dtype_to_bq_type(defaulted_scale2) is None

    defaulted_scale9 = pl.Decimal(precision=None, scale=9)
    if defaulted_scale9.precision is None:
        assert polars_dtype_to_bq_type(defaulted_scale9) is None
    else:
        assert polars_dtype_to_bq_type(defaulted_scale9) is BQType.NUMERIC

    # Explicitly verify precision=None on a Decimal instance returns None.
    unspecified_precision = pl.Decimal(38, 9)
    unspecified_precision.precision = None
    assert polars_dtype_to_bq_type(unspecified_precision) is None


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Decimal(29, 0), BQType.NUMERIC),
        (pl.Decimal(38, 9), BQType.NUMERIC),
        (pl.Decimal(10, 2), BQType.NUMERIC),
        (pl.Decimal(3, 3), BQType.NUMERIC),
        (pl.Decimal(30, 0), None),
        (pl.Decimal(38, 0), None),
        (pl.Decimal(38, 2), None),
        (pl.Decimal(38, 8), None),
        (pl.Decimal(38, 10), None),
    ],
)
def test_polars_dtype_to_bq_type_decimal_integer_digit_bounds(dtype, expected):
    assert polars_dtype_to_bq_type(dtype) is expected


def test_polars_decimal_supports_one():
    from polars_bigquery.core.types import polars_decimal_supports_one

    assert polars_decimal_supports_one(pl.Decimal(3, 3)) is False
    assert polars_decimal_supports_one(pl.Decimal(9, 9)) is False
    assert polars_decimal_supports_one(pl.Decimal(4, 3)) is True
    assert polars_decimal_supports_one(pl.Decimal(38, 9)) is True
    assert polars_decimal_supports_one(pl.Int64()) is True


def test_build_extension_dtype_explicit_load_as_storage_and_env_warning(
    monkeypatch, caplog
):
    # Explicit override is independent of POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR
    monkeypatch.setenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    assert build_extension_dtype(
        JSON_EXTENSION_NAME, pl.String(), load_as_storage=False
    ) == pl.Extension(JSON_EXTENSION_NAME, pl.String())
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    assert (
        build_extension_dtype(JSON_EXTENSION_NAME, pl.String(), load_as_storage=True)
        == pl.String()
    )

    # Changing POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR post-import logs a warning
    # and keeps the latched startup behavior so schemas stay aligned with Polars 2.x.
    import polars_bigquery.core.types as types_mod

    monkeypatch.setattr(types_mod, "_INITIAL_UNKNOWN_EXTENSION_BEHAVIOR", None)
    monkeypatch.setattr(types_mod, "_last_warned_env_drift", None)
    monkeypatch.setenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    with caplog.at_level("WARNING", logger=types_mod.__name__):
        assert build_extension_dtype(JSON_EXTENSION_NAME, pl.String()) == pl.Extension(
            JSON_EXTENSION_NAME, pl.String()
        )
        # Subsequent calls with identical drift must be throttled (1 warning total).
        assert build_extension_dtype(
            GEOGRAPHY_EXTENSION_NAME, pl.String(), GEOGRAPHY_EXTENSION_METADATA
        ) == pl.Extension(
            GEOGRAPHY_EXTENSION_NAME, pl.String(), GEOGRAPHY_EXTENSION_METADATA
        )
    assert len(caplog.records) == 1
    assert "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR" in caplog.text

    # Restoring the initial value clears the latch; drifting again warns once.
    caplog.clear()
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    with caplog.at_level("WARNING", logger=types_mod.__name__):
        assert build_extension_dtype(JSON_EXTENSION_NAME, pl.String()) == pl.Extension(
            JSON_EXTENSION_NAME, pl.String()
        )
    assert caplog.records == []
    monkeypatch.setenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    with caplog.at_level("WARNING", logger=types_mod.__name__):
        build_extension_dtype(JSON_EXTENSION_NAME, pl.String())
    assert len(caplog.records) == 1

    caplog.clear()
    monkeypatch.setattr(
        types_mod, "_INITIAL_UNKNOWN_EXTENSION_BEHAVIOR", "load_as_storage"
    )
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    with caplog.at_level("WARNING", logger=types_mod.__name__):
        assert build_extension_dtype(JSON_EXTENSION_NAME, pl.String()) == pl.String()
    assert len(caplog.records) == 1
    assert "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR" in caplog.text


def test_subprocess_unknown_extension_env_var_matches_arrow_c_stream():
    """Verify startup env var `load_as_storage` makes `build_extension_dtype` match Arrow stream."""
    import os
    import subprocess
    import sys

    code = """
import polars as pl
from polars_bigquery.core.types import (
    GEOGRAPHY_EXTENSION_METADATA,
    GEOGRAPHY_EXTENSION_NAME,
    JSON_EXTENSION_NAME,
    build_extension_dtype,
)

expected_schema = pl.Schema(
    {
        "j": build_extension_dtype(JSON_EXTENSION_NAME, pl.String()),
        "g": build_extension_dtype(
            GEOGRAPHY_EXTENSION_NAME,
            pl.String(),
            GEOGRAPHY_EXTENSION_METADATA,
        ),
        "j_arr": build_extension_dtype(JSON_EXTENSION_NAME, pl.List(pl.String())),
    }
)
assert expected_schema == pl.Schema(
    {"j": pl.String(), "g": pl.String(), "j_arr": pl.List(pl.String())}
), expected_schema

src = pl.DataFrame({
    "j": pl.Series(['{"a": 1}'], dtype=pl.String()).ext.to(
        pl.Extension(JSON_EXTENSION_NAME, pl.String())
    ),
    "g": pl.Series(["POINT(0 0)"], dtype=pl.String()).ext.to(
        pl.Extension(GEOGRAPHY_EXTENSION_NAME, pl.String(), GEOGRAPHY_EXTENSION_METADATA)
    ),
    "j_arr": pl.Series([['{"a": 1}']], dtype=pl.List(pl.String())).ext.to(
        pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.String()))
    ),
})
roundtrip = pl.scan_arrow_c_stream(src).collect()
assert roundtrip.schema == expected_schema, (roundtrip.schema, expected_schema)
"""
    env = {**os.environ, "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    # Verify mutating POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR mid-process still aligns
    # `build_extension_dtype()` with what Polars' Rust importer actually latched.
    drift_unset_to_storage = """
import os
import polars as pl
from polars_bigquery.core.types import JSON_EXTENSION_NAME, build_extension_dtype

os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
expected_schema = pl.Schema(
    {"j": build_extension_dtype(JSON_EXTENSION_NAME, pl.String())}
)
src = pl.DataFrame({
    "j": pl.Series(['{"a": 1}'], dtype=pl.String()).ext.to(
        pl.Extension(JSON_EXTENSION_NAME, pl.String())
    ),
})
roundtrip = pl.scan_arrow_c_stream(src).collect()
assert roundtrip.schema == expected_schema, (roundtrip.schema, expected_schema)
"""
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k != "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"
    }
    res_drift = subprocess.run(
        [sys.executable, "-c", drift_unset_to_storage],
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_drift.returncode == 0, (
        f"stdout={res_drift.stdout}\nstderr={res_drift.stderr}"
    )


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
        (pl.Duration("us"), 7),
        (pl.Float16(), 7),
        (pl.Decimal(76, 38), 7),
        (pl.List(pl.Decimal(76, 38)), 7),
        (pl.Struct({"x": pl.Decimal(76, 38)}), 7),
        (pl.List(pl.Duration("us")), 7),
        (pl.Struct({"x": pl.Duration("us")}), 7),
        (pl.Extension(JSON_EXTENSION_NAME, pl.Decimal(76, 38)), 7),
        (pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.Decimal(76, 38))), 7),
        # Invalid storage on BigQuery extension names MUST rank 7 (never unwrapped).
        (pl.Extension(JSON_EXTENSION_NAME, pl.Boolean()), 7),
        (pl.Extension(JSON_EXTENSION_NAME, pl.Int64()), 7),
        (pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.Int64())), 7),
        (pl.Extension(GEOGRAPHY_EXTENSION_NAME, pl.Boolean()), 7),
        (pl.Extension(GEOGRAPHY_EXTENSION_NAME, pl.Int64()), 7),
        (pl.Extension(GEOGRAPHY_EXTENSION_NAME, pl.List(pl.Int64())), 7),
        (pl.List(pl.Extension(JSON_EXTENSION_NAME, pl.Int64())), 7),
        (pl.Struct({"x": pl.Extension(JSON_EXTENSION_NAME, pl.Boolean())}), 7),
        (
            pl.Extension(
                "custom:outer",
                pl.Extension(JSON_EXTENSION_NAME, pl.Boolean()),
            ),
            7,
        ),
        (
            pl.Struct(
                {
                    "x": pl.Extension(
                        "custom:outer",
                        pl.Extension(GEOGRAPHY_EXTENSION_NAME, pl.Int64()),
                    )
                }
            ),
            7,
        ),
        # Valid BigQuery extensions wrapped by a custom outer extension keep BQ rank.
        (
            pl.Extension(
                "custom:outer",
                pl.Extension(JSON_EXTENSION_NAME, pl.String()),
            ),
            5,
        ),
        (
            pl.Extension(
                "custom:outer",
                pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.String())),
            ),
            6,
        ),
        (pl.Extension("custom:bn", pl.Decimal(76, 38)), 7),
        (pl.Extension("custom:bool", pl.Boolean()), 1),
        (pl.Extension("custom:bin", pl.Binary()), 5),
        (
            pl.Extension(
                "custom:outer",
                pl.Extension("custom:inner", pl.Boolean()),
            ),
            1,
        ),
    ],
)
def test_column_width_rank(dtype, expected_rank):
    assert column_width_rank(dtype) == expected_rank


def test_column_width_rank_shared_struct_field_dtype_and_invalid_bq_extensions():
    shared_int = pl.Int64()
    shared_struct = pl.Struct([pl.Field("a", shared_int), pl.Field("b", shared_int)])
    assert column_width_rank(shared_struct) == 6
    assert column_width_rank(pl.Struct([])) == 6

    assert (
        polars_dtype_to_bq_type(pl.Extension(JSON_EXTENSION_NAME, pl.Decimal(76, 38)))
        is None
    )
    assert (
        polars_dtype_to_bq_type(pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.Int64())))
        is None
    )
    assert (
        polars_dtype_to_bq_type(pl.Extension(GEOGRAPHY_EXTENSION_NAME, pl.Int64()))
        is None
    )


def test_extension_cycle_and_depth_guards():
    class _SelfCycleExt(pl.BaseExtension):
        def ext_storage(self):
            return self

    class _FreshWrapperCycleExt(pl.BaseExtension):
        def ext_storage(self):
            return _FreshWrapperCycleExt("custom:fresh", pl.String())

    self_cycle = _SelfCycleExt("custom:self", pl.String())
    assert has_bignumeric_dtype(self_cycle) is False
    assert column_width_rank(self_cycle) == 7
    assert column_width_rank(pl.List(self_cycle)) == 7

    fresh_cycle = _FreshWrapperCycleExt("custom:fresh", pl.String())
    assert has_bignumeric_dtype(fresh_cycle) is False
    assert column_width_rank(fresh_cycle) == 7
    assert column_width_rank(pl.Struct({"x": fresh_cycle})) == 7


def test_dtype_traversals_prevent_id_reuse_on_fresh_ext_storage():
    """Verify strong-reference tracking prevents CPython id() aliasing across loops."""

    class _ChainExt(pl.BaseExtension):
        def __init__(self, remaining: int, leaf_factory):
            self._rem = remaining
            self._leaf_factory = leaf_factory
            super().__init__("custom:chain", pl.Int64())

        def ext_storage(self):
            if self._rem > 0:
                return _ChainExt(self._rem - 1, self._leaf_factory)
            return self._leaf_factory()

    # Repeated runs exercise pymalloc free-list address reuse.
    for _ in range(50):
        assert column_width_rank(_ChainExt(5, pl.Boolean)) == 1
        assert has_bignumeric_dtype(_ChainExt(5, lambda: pl.Decimal(76, 38))) is True

        struct_with_dec = pl.Struct(
            [
                pl.Field("bad", _ChainExt(0, lambda: pl.Decimal(76, 38))),
                pl.Field("ok1", _ChainExt(0, lambda: pl.Decimal(38, 9))),
                pl.Field("ok2", _ChainExt(0, lambda: pl.Decimal(38, 9))),
                pl.Field("ok3", _ChainExt(0, lambda: pl.Decimal(38, 9))),
                pl.Field("ok4", _ChainExt(0, lambda: pl.Decimal(38, 9))),
            ]
        )
        assert column_width_rank(struct_with_dec) == 7


def test_has_bignumeric_dtype_deep_containers_and_extension_diamond_dag():
    from polars_bigquery.core.types import _MAX_TYPE_DEPTH

    # Finite Polars List containers deeper than _MAX_TYPE_DEPTH still detect BIGNUMERIC.
    deep_bn: pl.DataType = pl.Decimal(precision=76, scale=38)
    for _ in range(_MAX_TYPE_DEPTH + 5):
        deep_bn = pl.List(deep_bn)
    assert has_bignumeric_dtype(deep_bn) is True

    # Shared extension DAG reached at depth _MAX_TYPE_DEPTH AND depth 0 must detect
    # BIGNUMERIC regardless of field order (minimum ext_depth tracking).
    shared_ext = pl.Extension("custom:shared", pl.Decimal(precision=76, scale=38))
    deep_chain: pl.DataType = shared_ext
    for idx in range(_MAX_TYPE_DEPTH):
        deep_chain = pl.Extension(f"custom:w{idx}", deep_chain)

    dag_deep_last = pl.Struct(
        [pl.Field("shallow", shared_ext), pl.Field("deep", deep_chain)]
    )
    dag_deep_first = pl.Struct(
        [pl.Field("deep", deep_chain), pl.Field("shallow", shared_ext)]
    )
    assert has_bignumeric_dtype(dag_deep_last) is True
    assert has_bignumeric_dtype(dag_deep_first) is True


def test_has_bignumeric_dtype_nested():
    bn = pl.Decimal(precision=76, scale=38)
    num = pl.Decimal(precision=38, scale=9)
    unparam = pl.Decimal(precision=38, scale=9)
    unparam.precision = None  # type: ignore[misc]

    assert has_bignumeric_dtype(bn) is True
    assert has_bignumeric_dtype(pl.List(bn)) is True
    assert has_bignumeric_dtype(pl.Array(bn, 2)) is True
    assert has_bignumeric_dtype(pl.Struct([pl.Field("x", pl.List(bn))])) is True
    assert has_bignumeric_dtype(pl.Extension("custom:bn", bn)) is True
    assert has_bignumeric_dtype(pl.Extension(JSON_EXTENSION_NAME, bn)) is True
    assert has_bignumeric_dtype(num) is False
    assert has_bignumeric_dtype(unparam) is False
    assert has_bignumeric_dtype(pl.Extension("custom:num", num)) is False
    assert has_bignumeric_dtype(pl.List(num)) is False
    assert has_bignumeric_dtype(pl.Struct([pl.Field("x", pl.Int64())])) is False
    if hasattr(pl, "Map"):
        assert has_bignumeric_dtype(pl.Map(pl.String(), bn)) is True
        assert has_bignumeric_dtype(pl.Map(pl.String(), pl.Int64())) is False


def test_column_width_rank_unparameterized_decimal_and_bq_type_override():
    unparam_decimal = pl.Decimal(38, 9)
    unparam_decimal.precision = None  # type: ignore[misc]
    assert column_width_rank(unparam_decimal) == 7

    # When authoritative bq_type is passed, it overrides dtype inference.
    assert column_width_rank(pl.String(), bq_type=BQType.ARRAY) == 6
    assert column_width_rank(pl.String(), bq_type=BQType.BOOL) == 1
    assert column_width_rank(pl.Int64(), bq_type=BQType.STRUCT) == 6
