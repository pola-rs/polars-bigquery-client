from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import polars as pl
from polars.datatypes import (
    Binary,
    Boolean,
    DataType,
    Date,
    Datetime,
    Decimal,
    Field,
    Float64,
    Int64,
    List,
    String,
    Struct,
    Time,
)

from polars_bigquery.core.types import (
    GEOGRAPHY_EXTENSION_METADATA,
    GEOGRAPHY_EXTENSION_NAME,
    JSON_EXTENSION_NAME,
    BQType,
    build_extension_dtype,
)

# BigQuery limits RECORD/STRUCT nesting depth to 15; cap recursion well above
# BigQuery's limit to guard against `RecursionError` on malformed REST payloads.
MAX_SCHEMA_NESTING_DEPTH = 64

ExtensionDtypeBuilder = Callable[[str, DataType, str | None], DataType]


@dataclasses.dataclass(frozen=True)
class SkippedColumn:
    """Metadata describing a top-level BigQuery column omitted during schema extraction."""

    name: str
    reason: str
    path: str


@dataclasses.dataclass(frozen=True)
class ExtractedSchema:
    """Structured result of extracting a Polars schema and BigQuery types from REST metadata."""

    polars_schema: pl.Schema
    bq_types: dict[str, BQType]
    skipped_columns: tuple[SkippedColumn, ...]


class _UnsupportedBigQueryType(Exception):
    """Internal sentinel raised when a field or nested subfield cannot be represented."""

    def __init__(self, reason: str, path: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.path = path


def _build_extension_dtype_checked(
    builder: ExtensionDtypeBuilder,
    name: str,
    storage: DataType,
    metadata: str | None,
    path: str,
) -> DataType:
    """Invoke `builder` within an explicit error boundary for custom `BaseExtension` hooks."""
    try:
        return builder(name, storage, metadata)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _UnsupportedBigQueryType(
            f"extension {name} rejected storage {storage}: {exc}",
            path,
        ) from exc


def _extract_data_type(
    field: Any,
    path: str,
    *,
    depth: int = 0,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> tuple[DataType, BQType]:
    """Convert a BigQuery REST TableFieldSchema dict into `(Polars DataType, BQType)`.

    Note: the REST API uses the names from the Legacy SQL data types, but if
    user-entered it may include the newer aliases.
    (https://cloud.google.com/bigquery/docs/data-types).

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    """
    if depth > MAX_SCHEMA_NESTING_DEPTH:
        raise _UnsupportedBigQueryType(
            f"STRUCT nesting depth exceeds {MAX_SCHEMA_NESTING_DEPTH}", path
        )
    if not isinstance(field, Mapping):
        raise _UnsupportedBigQueryType("malformed field metadata", path)

    raw_mode = field.get("mode")
    if raw_mode is None:
        mode = ""
    elif isinstance(raw_mode, str):
        mode = raw_mode.casefold()
    else:
        raise _UnsupportedBigQueryType(f"invalid mode {raw_mode!r}", path)

    raw_type = field.get("type")
    if raw_type is None:
        type_ = ""
    elif isinstance(raw_type, str):
        type_ = raw_type.casefold()
    else:
        raise _UnsupportedBigQueryType(f"invalid type {raw_type!r}", path)

    # Check for BQ ARRAY (Polars List) first because the REST API encodes ARRAY
    # via mode="REPEATED" rather than a separate type name.
    if mode == "repeated":
        # BigQuery attaches ARROW:extension:name to the outer List field for
        # ARRAY<JSON> and ARRAY<GEOGRAPHY> rather than the inner String element.
        if type_ == "json":
            return (
                _build_extension_dtype_checked(
                    extension_dtype_builder,
                    JSON_EXTENSION_NAME,
                    List(String()),
                    None,
                    path,
                ),
                BQType.ARRAY,
            )
        if type_ == "geography":
            return (
                _build_extension_dtype_checked(
                    extension_dtype_builder,
                    GEOGRAPHY_EXTENSION_NAME,
                    List(String()),
                    GEOGRAPHY_EXTENSION_METADATA,
                    path,
                ),
                BQType.ARRAY,
            )
        inner_type, _ = _extract_data_type(
            {**field, "mode": "NULLABLE"},
            path,
            depth=depth + 1,
            extension_dtype_builder=extension_dtype_builder,
        )
        return List(inner_type), BQType.ARRAY

    if type_ in ("bool", "boolean"):
        return Boolean(), BQType.BOOL
    if type_ == "bytes":
        return Binary(), BQType.BYTES
    if type_ == "date":
        return Date(), BQType.DATE
    if type_ == "datetime":
        # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
        # https://stackoverflow.com/a/47724366/101923
        return Datetime(time_unit="us"), BQType.DATETIME
    if type_ == "geography":
        return (
            _build_extension_dtype_checked(
                extension_dtype_builder,
                GEOGRAPHY_EXTENSION_NAME,
                String(),
                GEOGRAPHY_EXTENSION_METADATA,
                path,
            ),
            BQType.GEOGRAPHY,
        )
    if type_ in ("float", "float64"):
        return Float64(), BQType.FLOAT64
    if type_ in ("integer", "int64"):
        return Int64(), BQType.INT64
    if type_ in ("numeric", "decimal"):
        # BigQuery NUMERIC type has precision 38 and scale 9.
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
        return Decimal(precision=38, scale=9), BQType.NUMERIC
    if type_ in ("bignumeric", "bigdecimal"):
        raise _UnsupportedBigQueryType("BIGNUMERIC", path)
    if type_ == "json":
        return (
            _build_extension_dtype_checked(
                extension_dtype_builder,
                JSON_EXTENSION_NAME,
                String(),
                None,
                path,
            ),
            BQType.JSON,
        )
    if type_ == "interval":
        raise _UnsupportedBigQueryType("INTERVAL", path)
    if type_ == "range":
        raise _UnsupportedBigQueryType("RANGE", path)
    if type_ in ("record", "struct"):
        subfields = field.get("fields")
        if subfields is None:
            subfields = []
        if not isinstance(subfields, (list, tuple)):
            raise _UnsupportedBigQueryType("malformed STRUCT fields", path)
        if not subfields:
            raise _UnsupportedBigQueryType("empty STRUCT", path)
        seen: set[str] = set()
        polars_fields: list[Field] = []
        for idx, subfield in enumerate(subfields):
            if not isinstance(subfield, Mapping):
                raise _UnsupportedBigQueryType(
                    "malformed STRUCT subfield", f"{path}[{idx}]"
                )
            sub_name = subfield.get("name")
            if not isinstance(sub_name, str) or not sub_name:
                raise _UnsupportedBigQueryType(
                    "unnamed STRUCT subfield", f"{path}[{idx}]"
                )
            if sub_name in seen:
                raise _UnsupportedBigQueryType(
                    "duplicate STRUCT subfield", f"{path}.{sub_name}"
                )
            seen.add(sub_name)
            sub_dtype, _ = _extract_data_type(
                subfield,
                f"{path}.{sub_name}",
                depth=depth + 1,
                extension_dtype_builder=extension_dtype_builder,
            )
            polars_fields.append(Field(sub_name, sub_dtype))
        return Struct(polars_fields), BQType.STRUCT
    if type_ == "string":
        return String(), BQType.STRING
    if type_ == "time":
        return Time(), BQType.TIME
    if type_ == "timestamp":
        precision = field.get("timestampPrecision")
        if precision is None or (
            not isinstance(precision, bool) and precision in (6, "6")
        ):
            return Datetime(time_unit="us", time_zone="UTC"), BQType.TIMESTAMP
        raise _UnsupportedBigQueryType(f"TIMESTAMP(precision={precision})", path)

    raise _UnsupportedBigQueryType(f"unsupported type {type_ or '<missing>'}", path)


def extract_schema_metadata(
    table_metadata: Mapping[str, Any],
    *,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> ExtractedSchema:
    """Extract a Polars schema, authoritative BigQuery types, and skipped columns.

    Pure metadata transformation with no warning or exception side effects.
    Pass `extension_dtype_builder` to inject custom `JSON`/`GEOGRAPHY` dtype
    resolution independently of Polars' global extension registry or `os.environ`.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table
    """
    pl_schema: dict[str, DataType] = {}
    bq_types: dict[str, BQType] = {}
    skipped_columns: list[SkippedColumn] = []

    raw_schema = (
        table_metadata.get("schema") if isinstance(table_metadata, Mapping) else None
    )
    raw_fields = raw_schema.get("fields") if isinstance(raw_schema, Mapping) else None
    fields = raw_fields if isinstance(raw_fields, (list, tuple)) else []

    for idx, field in enumerate(fields):
        if not isinstance(field, Mapping):
            placeholder = f"<invalid[{idx}]>"
            skipped_columns.append(
                SkippedColumn(
                    name=placeholder,
                    reason="malformed field metadata",
                    path=placeholder,
                )
            )
            continue
        name = field.get("name")
        if not isinstance(name, str) or not name:
            placeholder = f"<unnamed[{idx}]>"
            skipped_columns.append(
                SkippedColumn(
                    name=placeholder,
                    reason="unnamed column",
                    path=placeholder,
                )
            )
            continue
        if name in pl_schema:
            skipped_columns.append(
                SkippedColumn(
                    name=name,
                    reason="duplicate column",
                    path=name,
                )
            )
            continue
        try:
            dtype, bq_type = _extract_data_type(
                field,
                name,
                extension_dtype_builder=extension_dtype_builder,
            )
        except _UnsupportedBigQueryType as exc:
            skipped_columns.append(
                SkippedColumn(name=name, reason=exc.reason, path=exc.path)
            )
            continue
        pl_schema[name] = dtype
        bq_types[name] = bq_type

    # If table is ingestion-time partitioned, add pseudo-columns for partition
    # filters. BigQuery exposes _PARTITIONDATE only on DAY-partitioned tables,
    # whereas _PARTITIONTIME is exposed on all ingestion-time partitioned tables.
    # Treat both `None` and `""` as an absent partitioning field (protobuf-JSON safe).
    # See: https://cloud.google.com/bigquery/docs/partitioned-tables#ingestion_time
    time_partitioning = (
        table_metadata.get("timePartitioning")
        if isinstance(table_metadata, Mapping)
        else None
    )
    if isinstance(time_partitioning, Mapping) and not time_partitioning.get("field"):
        raw_part_type = time_partitioning.get("type")
        part_type = (
            raw_part_type.casefold()
            if isinstance(raw_part_type, str) and raw_part_type
            else "day"
        )
        if part_type == "day" and "_PARTITIONDATE" not in pl_schema:
            pl_schema["_PARTITIONDATE"] = Date()
            bq_types["_PARTITIONDATE"] = BQType.DATE
        if "_PARTITIONTIME" not in pl_schema:
            pl_schema["_PARTITIONTIME"] = Datetime(time_unit="us", time_zone="UTC")
            bq_types["_PARTITIONTIME"] = BQType.TIMESTAMP

    return ExtractedSchema(
        polars_schema=pl.Schema(pl_schema),
        bq_types=bq_types,
        skipped_columns=tuple(skipped_columns),
    )


def extract_polars_schema(
    table_metadata: Mapping[str, Any],
    *,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> pl.Schema:
    """Convert a BigQuery table REST response into a Polars schema.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table
    """
    return extract_schema_metadata(
        table_metadata,
        extension_dtype_builder=extension_dtype_builder,
    ).polars_schema
