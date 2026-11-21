"""Canonical BigQuery type system and Polars dtype classification helpers."""

from __future__ import annotations

import enum
import logging
import os
from typing import Literal

import polars as pl
from polars.datatypes import DataType, DataTypeClass

logger = logging.getLogger(__name__)

JSON_EXTENSION_NAME = "google:sqlType:json"
GEOGRAPHY_EXTENSION_NAME = "google:sqlType:geography"
GEOGRAPHY_EXTENSION_METADATA = '{"encoding": "WKT"}'

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
    the value latched at module import time and logs a warning if the
    environment variable is mutated mid-process; pass ``load_as_storage``
    explicitly when deterministic resolution independent of process environment
    is needed.
    """
    registered = pl.get_extension_type(name)
    if isinstance(registered, type) and issubclass(registered, pl.BaseExtension):
        return registered.ext_from_params(name, storage, metadata)
    if registered == "storage":
        return storage
    if load_as_storage is not None:
        return storage if load_as_storage else pl.Extension(name, storage, metadata)
    current_behavior = os.environ.get(_UNKNOWN_EXTENSION_ENV_VAR)
    if current_behavior != _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR:
        logger.warning(
            "%s changed from %r to %r after module import; Polars 2.x latches "
            "this setting once per process, so keeping the initial value %r to "
            "stay aligned with Arrow stream decoding. Prefer "
            "polars.register_extension_type(%r, as_storage=True) or setting the "
            "environment variable before launching Python.",
            _UNKNOWN_EXTENSION_ENV_VAR,
            _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR,
            current_behavior,
            _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR,
            name,
        )
    if _INITIAL_UNKNOWN_EXTENSION_BEHAVIOR == "load_as_storage":
        return storage
    return pl.Extension(name, storage, metadata)


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
        if ext_name == JSON_EXTENSION_NAME:
            return (
                BQType.ARRAY
                if isinstance(dtype.ext_storage(), pl.List)
                else BQType.JSON
            )
        if ext_name == GEOGRAPHY_EXTENSION_NAME:
            return (
                BQType.ARRAY
                if isinstance(dtype.ext_storage(), pl.List)
                else BQType.GEOGRAPHY
            )
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
    """Return a deterministic storage-width rank (1=narrowest, 6=widest/unknown).

    When `bq_type` is supplied (e.g. from authoritative BigQuery REST metadata),
    it takes precedence over inferring the BigQuery type from `dtype`.
    """
    resolved = bq_type if bq_type is not None else polars_dtype_to_bq_type(dtype)
    return _BQ_TYPE_WIDTH_RANK.get(resolved, 6)


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
    "polars_datetime_time_unit",
    "polars_decimal_supports_one",
    "polars_dtype_to_bq_type",
]
