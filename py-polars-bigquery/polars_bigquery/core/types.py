"""Canonical BigQuery type system and Polars dtype classification helpers."""

from __future__ import annotations

import enum
import os
from typing import Literal

import polars as pl
from polars.datatypes import DataType, DataTypeClass

JSON_EXTENSION_NAME = "google:sqlType:json"
GEOGRAPHY_EXTENSION_NAME = "google:sqlType:geography"
GEOGRAPHY_EXTENSION_METADATA = '{"encoding": "WKT"}'


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

_INT_DTYPE_CLASSES: tuple[type[DataType], ...] = (
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

_FLOAT_DTYPE_CLASSES: tuple[type[DataType], ...] = (
    pl.Float32,
    pl.Float64,
)

_SIMPLE_DTYPE_TO_BQ_TYPE: tuple[
    tuple[type[DataType] | tuple[type[DataType], ...], BQType], ...
] = (
    (pl.Boolean, BQType.BOOL),
    (_INT_DTYPE_CLASSES, BQType.INT64),
    (_FLOAT_DTYPE_CLASSES, BQType.FLOAT64),
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
) -> DataType:
    """Build a Polars DataType for a BigQuery Arrow extension type.

    Respects Polars' runtime extension registry and environment configuration:
    - Mode A (registered ``BaseExtension`` subclass): instantiates via
      ``ext_from_params(name, storage, metadata)``.
    - Mode B (registered ``as_storage=True`` or
      ``POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage``): returns
      ``storage``.
    - Default (``load_as_extension``): returns ``pl.Extension(name, storage, metadata)``.
    """
    registered = pl.get_extension_type(name)
    if isinstance(registered, type) and issubclass(registered, pl.BaseExtension):
        return registered.ext_from_params(name, storage, metadata)
    if registered == "storage":
        return storage
    if os.environ.get("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR") == "load_as_storage":
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
        if dtype.precision is not None and dtype.precision <= 38 and dtype.scale <= 9:
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


def column_width_rank(dtype: DataType | DataTypeClass) -> int:
    """Return a deterministic storage-width rank (1=narrowest, 6=widest/unknown)."""
    bq_type = polars_dtype_to_bq_type(dtype)
    return _BQ_TYPE_WIDTH_RANK.get(bq_type, 6)


__all__ = [
    "GEOGRAPHY_EXTENSION_METADATA",
    "GEOGRAPHY_EXTENSION_NAME",
    "JSON_EXTENSION_NAME",
    "OPAQUE_BQ_TYPES",
    "BQType",
    "build_extension_dtype",
    "column_width_rank",
    "polars_datetime_time_unit",
    "polars_dtype_to_bq_type",
]
