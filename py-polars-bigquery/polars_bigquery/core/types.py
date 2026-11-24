"""Canonical BigQuery type system and Polars dtype classification helpers."""

from __future__ import annotations

import enum
import logging
import os
import threading
from typing import Literal

import polars as pl
from polars.datatypes import DataType, DataTypeClass

logger = logging.getLogger(__name__)

JSON_EXTENSION_NAME = "google:sqlType:json"
GEOGRAPHY_EXTENSION_NAME = "google:sqlType:geography"
GEOGRAPHY_EXTENSION_METADATA = '{"encoding": "WKT"}'
_BIGQUERY_EXTENSION_NAMES = frozenset({JSON_EXTENSION_NAME, GEOGRAPHY_EXTENSION_NAME})

# BigQuery NUMERIC supports up to 38 total digits, up to 9 fractional digits,
# and at most 29 integer digits (precision - scale <= 29).
BQ_NUMERIC_MAX_PRECISION = 38
BQ_NUMERIC_MAX_SCALE = 9
BQ_NUMERIC_MAX_INTEGER_DIGITS = 29

# Polars 2.x latches `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` once per process
# on import (even `pl.Config.reload_env_vars()` does not refresh it). Latch the
# same startup value here so schema inference matches Arrow C Stream decoding,
# and warn if `os.environ` is mutated mid-process.
_UNKNOWN_EXTENSION_ENV_VAR = "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"
_INITIAL_UNKNOWN_EXTENSION_BEHAVIOR: str | None = os.environ.get(
    _UNKNOWN_EXTENSION_ENV_VAR
)
_WARNED_ENV_DRIFT_LOCK = threading.Lock()
_last_warned_env_drift: tuple[str | None, str | None] | None = None


class BQType(enum.Enum):
    """Closed enumeration of the 14 supported BigQuery data types."""

    BOOL = "BOOL"
    INT64 = "INT64"
    FLOAT64 = "FLOAT64"
    NUMERIC = "NUMERIC"
    STRING = "STRING"
    BYTES = "BYTES"
    DATE = "DATE"
    DATETIME = "DATETIME"
    TIME = "TIME"
    TIMESTAMP = "TIMESTAMP"
    JSON = "JSON"
    GEOGRAPHY = "GEOGRAPHY"
    ARRAY = "ARRAY"
    STRUCT = "STRUCT"


OPAQUE_BQ_TYPES: frozenset[BQType] = frozenset(
    {
        BQType.JSON,
        BQType.GEOGRAPHY,
        BQType.ARRAY,
        BQType.STRUCT,
    }
)

INT_DTYPE_CLASSES: tuple[type[DataType], ...] = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)

FLOAT_DTYPE_CLASSES: tuple[type[DataType], ...] = (
    pl.Float32,
    pl.Float64,
)

_SIMPLE_DTYPE_TO_BQ_TYPE: tuple[
    tuple[type[DataType] | tuple[type[DataType], ...], BQType], ...
] = (
    (pl.Boolean, BQType.BOOL),
    (INT_DTYPE_CLASSES, BQType.INT64),
    (FLOAT_DTYPE_CLASSES, BQType.FLOAT64),
    (pl.String, BQType.STRING),
    (pl.Binary, BQType.BYTES),
    (pl.Date, BQType.DATE),
    (pl.Time, BQType.TIME),
    (pl.List, BQType.ARRAY),
    (pl.Struct, BQType.STRUCT),
)

_BQ_TYPE_WIDTH_RANK: dict[BQType | None, int] = {
    BQType.BOOL: 1,
    BQType.DATE: 2,
    BQType.INT64: 3,
    BQType.FLOAT64: 3,
    BQType.TIME: 3,
    BQType.DATETIME: 3,
    BQType.TIMESTAMP: 3,
    BQType.NUMERIC: 4,
    BQType.STRING: 5,
    BQType.BYTES: 5,
    BQType.JSON: 5,
    BQType.GEOGRAPHY: 5,
    BQType.ARRAY: 6,
    BQType.STRUCT: 6,
}


def build_extension_dtype(
    name: str,
    storage: DataType,
    metadata: str | None = None,
    *,
    load_as_storage: bool | None = None,
) -> DataType:
    """Build a Polars DataType for a BigQuery Arrow extension type.

    Respects Polars' runtime extension registry and environment configuration:
    - Mode A (registered ``BaseExtension`` subclass): instantiates via
      ``ext_from_params(name, storage, metadata)``.
    - Mode B (registered ``as_storage=True`` or
      ``POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage``): returns
      ``storage``.
    - Default (``load_as_extension``): returns ``pl.Extension(name, storage, metadata)``.

    Note: In Polars 2.x, the Rust Arrow C Stream importer latches
    ``POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR`` once per process on import
    (``pl.Config.reload_env_vars()`` does not refresh it). This helper follows
    the value latched at module import time and logs one warning per
    ``(initial, current)`` drift transition if the environment variable is
    mutated mid-process; pass ``load_as_storage`` explicitly when deterministic
    resolution independent of process environment is needed.
    """
    registered = pl.get_extension_type(name)
    if isinstance(registered, type) and issubclass(registered, pl.BaseExtension):
        return registered.ext_from_params(name, storage, metadata)
    if registered == "storage":
        return storage
    if load_as_storage is not None:
        return storage if load_as_storage else pl.Extension(name, storage, metadata)
    global _last_warned_env_drift
    current_behavior = os.environ.get(_UNKNOWN_EXTENSION_ENV_VAR)
    if current_behavior != _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR:
        should_warn = False
        drift = (_INITIAL_UNKNOWN_EXTENSION_BEHAVIOR, current_behavior)
        with _WARNED_ENV_DRIFT_LOCK:
            if _last_warned_env_drift != drift:
                _last_warned_env_drift = drift
                should_warn = True
        if should_warn:
            logger.warning(
                "%s changed from %r to %r after module import; Polars 2.x latches "
                "this setting once per process, so keeping the initial value %r to "
                "stay aligned with Arrow stream decoding. Prefer "
                "polars.register_extension_type(..., as_storage=True) "
                "(e.g. %r and %r) or setting the environment variable before "
                "launching Python.",
                _UNKNOWN_EXTENSION_ENV_VAR,
                _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR,
                current_behavior,
                _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR,
                JSON_EXTENSION_NAME,
                GEOGRAPHY_EXTENSION_NAME,
            )
    elif _last_warned_env_drift is not None:
        with _WARNED_ENV_DRIFT_LOCK:
            if (
                os.environ.get(_UNKNOWN_EXTENSION_ENV_VAR)
                == _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR
            ):
                _last_warned_env_drift = None
    if _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR == "load_as_storage":
        return storage
    return pl.Extension(name, storage, metadata)


def _is_dtype_or_class(value: object, cls: type[DataType] = DataType) -> bool:
    """Return `True` if `value` is an instance or `DataTypeClass` subclass of `cls`."""
    return isinstance(value, cls) or (
        isinstance(value, type) and issubclass(value, cls)
    )


def _unwrap_extension_storage(
    dtype: pl.BaseExtension,
) -> DataType | DataTypeClass | None:
    """Recursively unwrap custom `pl.BaseExtension` storage types.

    Stops at BigQuery Arrow extension types (`google:sqlType:json` and
    `google:sqlType:geography`): returns them intact when valid and returns
    `None` when their storage dtype is invalid so they cannot fall through to a
    narrower storage width rank. Also returns `None` if a custom extension
    returns a non-`DataType`/`DataTypeClass`. Self-referential extensions
    terminate with `RecursionError`.
    """
    if dtype.ext_name() in _BIGQUERY_EXTENSION_NAMES:
        return dtype if polars_dtype_to_bq_type(dtype) is not None else None
    storage = dtype.ext_storage()
    if not _is_dtype_or_class(storage):
        return None
    if isinstance(storage, pl.BaseExtension):
        return _unwrap_extension_storage(storage)
    return storage


def has_bignumeric_dtype(dtype: DataType | DataTypeClass) -> bool:
    """Return `True` if `dtype` is or nests a `BIGNUMERIC` (`Decimal(precision > 38)`).

    Uses a simple recursive depth-first walk (BigQuery caps schema nesting at
    15 levels). Self-referential extensions terminate with `RecursionError`.
    """
    if isinstance(dtype, pl.Decimal):
        return (
            dtype.precision is not None and dtype.precision > BQ_NUMERIC_MAX_PRECISION
        )
    if isinstance(dtype, pl.BaseExtension):
        storage = dtype.ext_storage()
        return isinstance(storage, DataType) and has_bignumeric_dtype(storage)
    if isinstance(dtype, (pl.List, pl.Array)):
        return isinstance(dtype.inner, DataType) and has_bignumeric_dtype(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(has_bignumeric_dtype(field.dtype) for field in dtype.fields)
    map_cls = getattr(pl, "Map", None)
    if map_cls is not None and isinstance(dtype, map_cls):
        return (
            isinstance(dtype.key, DataType) and has_bignumeric_dtype(dtype.key)
        ) or (isinstance(dtype.value, DataType) and has_bignumeric_dtype(dtype.value))
    return False


def _has_unsupported_nested_dtype(dtype: DataType | DataTypeClass) -> bool:
    """Return `True` if a container/extension nests any dtype outside BigQuery's 14 `BQType`s."""
    if isinstance(dtype, type):
        return polars_dtype_to_bq_type(dtype) is None
    if isinstance(dtype, pl.BaseExtension):
        if dtype.ext_name() in _BIGQUERY_EXTENSION_NAMES:
            return polars_dtype_to_bq_type(dtype) is None
        storage = dtype.ext_storage()
        if not _is_dtype_or_class(storage):
            return True
        return _has_unsupported_nested_dtype(storage)
    if isinstance(dtype, pl.List):
        return dtype.inner is None or _has_unsupported_nested_dtype(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_has_unsupported_nested_dtype(field.dtype) for field in dtype.fields)
    return polars_dtype_to_bq_type(dtype) is None


def polars_dtype_to_bq_type(dtype: DataType | DataTypeClass) -> BQType | None:
    """Classify a Polars ``DataType`` or ``DataTypeClass`` into a ``BQType`` or ``None``."""
    if isinstance(dtype, type):
        if issubclass(dtype, pl.Datetime):
            return BQType.DATETIME
        for dtype_cls, bq_type in _SIMPLE_DTYPE_TO_BQ_TYPE:
            if issubclass(dtype, dtype_cls):
                return bq_type
        return None

    if isinstance(dtype, pl.BaseExtension):
        ext_name = dtype.ext_name()
        if ext_name in (JSON_EXTENSION_NAME, GEOGRAPHY_EXTENSION_NAME):
            storage = dtype.ext_storage()
            if _is_dtype_or_class(storage, pl.String):
                return (
                    BQType.JSON if ext_name == JSON_EXTENSION_NAME else BQType.GEOGRAPHY
                )
            if isinstance(storage, pl.List) and _is_dtype_or_class(
                storage.inner, pl.String
            ):
                return BQType.ARRAY
        return None

    if isinstance(dtype, pl.Datetime):
        return BQType.DATETIME if dtype.time_zone is None else BQType.TIMESTAMP

    if isinstance(dtype, pl.Decimal):
        if (
            dtype.precision is not None
            and 0 <= dtype.scale <= BQ_NUMERIC_MAX_SCALE
            and dtype.scale <= dtype.precision <= BQ_NUMERIC_MAX_PRECISION
            and (dtype.precision - dtype.scale) <= BQ_NUMERIC_MAX_INTEGER_DIGITS
        ):
            return BQType.NUMERIC
        return None

    for dtype_cls, bq_type in _SIMPLE_DTYPE_TO_BQ_TYPE:
        if isinstance(dtype, dtype_cls):
            return bq_type

    return None


def polars_datetime_time_unit(
    dtype: DataType | DataTypeClass,
) -> Literal["ms", "us", "ns"] | None:
    """Return the time unit of a Polars ``Datetime`` dtype, or ``None`` for other dtypes."""
    if isinstance(dtype, type):
        if issubclass(dtype, pl.Datetime):
            return "us"
        return None
    if isinstance(dtype, pl.Datetime):
        return dtype.time_unit
    return None


def polars_decimal_supports_one(dtype: DataType | DataTypeClass) -> bool:
    """Return True if a Polars ``Decimal`` dtype can represent integer value ``1``."""
    if isinstance(dtype, pl.Decimal) and dtype.precision is not None:
        return (dtype.precision - dtype.scale) >= 1
    return True


def column_width_rank(
    dtype: DataType | DataTypeClass,
    *,
    bq_type: BQType | None = None,
) -> int:
    """Return a deterministic storage-width rank (1=narrowest, 6=nested, 7=unknown).

    Ranks follow BigQuery wire widths across the 14 supported `BQType` values
    (`1` BOOL, `2` DATE, `3` 8-byte scalars, `4` NUMERIC, `5` variable-length
    STRING/BYTES/JSON/GEOGRAPHY, `6` ARRAY/STRUCT), leaving `7` for unknown or
    unsupported dtypes (including nested `BIGNUMERIC` / `Duration` / `Float16`
    inside `List`/`Struct`/`BaseExtension`) so zero-column fallback prioritizes
    every valid BigQuery type first.

    When `bq_type` is supplied (e.g. from authoritative BigQuery REST metadata),
    it takes precedence over inferring the BigQuery type from `dtype`.
    """
    if bq_type is not None:
        return _BQ_TYPE_WIDTH_RANK.get(bq_type, 7)
    if isinstance(dtype, type):
        resolved = polars_dtype_to_bq_type(dtype)
        return _BQ_TYPE_WIDTH_RANK[resolved] if resolved is not None else 7
    resolved = polars_dtype_to_bq_type(dtype)
    if resolved is None:
        if isinstance(dtype, pl.BaseExtension):
            unwrapped = _unwrap_extension_storage(dtype)
            if unwrapped is None:
                return 7
            unwrapped_bq = polars_dtype_to_bq_type(unwrapped)
            if unwrapped_bq is None:
                return 7
            if isinstance(
                unwrapped, (pl.List, pl.Struct)
            ) and _has_unsupported_nested_dtype(unwrapped):
                return 7
            return _BQ_TYPE_WIDTH_RANK[unwrapped_bq]
        return 7
    if isinstance(dtype, (pl.List, pl.Struct)) and _has_unsupported_nested_dtype(dtype):
        return 7
    return _BQ_TYPE_WIDTH_RANK[resolved]


__all__ = [
    "BQ_NUMERIC_MAX_INTEGER_DIGITS",
    "BQ_NUMERIC_MAX_PRECISION",
    "BQ_NUMERIC_MAX_SCALE",
    "FLOAT_DTYPE_CLASSES",
    "GEOGRAPHY_EXTENSION_METADATA",
    "GEOGRAPHY_EXTENSION_NAME",
    "INT_DTYPE_CLASSES",
    "JSON_EXTENSION_NAME",
    "OPAQUE_BQ_TYPES",
    "BQType",
    "build_extension_dtype",
    "column_width_rank",
    "has_bignumeric_dtype",
    "polars_datetime_time_unit",
    "polars_decimal_supports_one",
    "polars_dtype_to_bq_type",
]
