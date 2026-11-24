from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn

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

import polars_bigquery.exceptions
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


def has_bignumeric_dtype(dtype: pl.DataType) -> bool:
    """Return `True` if `dtype` is or nests a `BIGNUMERIC` (`Decimal(precision > 38)`)."""
    map_cls = getattr(pl, "Map", None)
    stack: list[pl.DataType] = [dtype]
    while stack:
        current = stack.pop()
        if isinstance(current, pl.Decimal):
            if current.precision is not None and current.precision > 38:
                return True
        elif isinstance(current, (pl.List, pl.Array)):
            if current.inner is not None:
                stack.append(current.inner)
        elif isinstance(current, pl.Struct):
            for field in current.fields:
                stack.append(field.dtype)
        elif map_cls is not None and isinstance(current, map_cls):
            if current.key is not None:
                stack.append(current.key)
            if current.value is not None:
                stack.append(current.value)
    return False


def _build_extension_dtype_checked(
    builder: ExtensionDtypeBuilder,
    name: str,
    storage: DataType,
    metadata: str | None,
    path: str,
) -> DataType:
    """Invoke `builder` within an explicit error boundary for custom `BaseExtension` hooks."""
    try:
        dtype = builder(name, storage, metadata)
    except (TypeError, ValueError, RuntimeError, AttributeError, LookupError) as exc:
        raise _UnsupportedBigQueryType(
            f"extension {name} rejected storage {storage}: {exc}",
            path,
        ) from exc
    if not isinstance(dtype, DataType):
        raise _UnsupportedBigQueryType(
            f"extension {name} returned non-DataType {type(dtype).__name__}",
            path,
        )
    return dtype


_VALID_FIELD_MODES = frozenset({"", "nullable", "required", "repeated"})


@dataclasses.dataclass(frozen=True, slots=True)
class _ExtensionTypeSpec:
    """Arrow extension mapping specification for a BigQuery SQL extension type."""

    extension_name: str
    metadata: str | None
    bq_type: BQType


# Explicit mapping of casefolded BigQuery REST type names (and legacy SQL aliases)
# to `(Polars dtype factory, canonical BQType)`. Factories ensure every lookup
# returns a fresh `DataType` instance rather than sharing a mutable instance.
# https://cloud.google.com/bigquery/docs/data-types
_SCALAR_TYPE_MAP: dict[str, tuple[Callable[[], DataType], BQType]] = {
    "bool": (Boolean, BQType.BOOL),
    "boolean": (Boolean, BQType.BOOL),
    "bytes": (Binary, BQType.BYTES),
    "date": (Date, BQType.DATE),
    # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
    # https://stackoverflow.com/a/47724366/101923
    "datetime": (lambda: Datetime(time_unit="us"), BQType.DATETIME),
    "float": (Float64, BQType.FLOAT64),
    "float64": (Float64, BQType.FLOAT64),
    "integer": (Int64, BQType.INT64),
    "int64": (Int64, BQType.INT64),
    # BigQuery NUMERIC type has precision 38 and scale 9.
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
    "numeric": (lambda: Decimal(precision=38, scale=9), BQType.NUMERIC),
    "decimal": (lambda: Decimal(precision=38, scale=9), BQType.NUMERIC),
    "string": (String, BQType.STRING),
    "time": (Time, BQType.TIME),
}

# BigQuery Arrow extension types. For `mode="REPEATED"` (`ARRAY<JSON>` /
# `ARRAY<GEOGRAPHY>`), BigQuery attaches `ARROW:extension:name` to the outer
# `List(String())` field rather than the inner `String()` element.
_EXTENSION_TYPE_MAP: dict[str, _ExtensionTypeSpec] = {
    "json": _ExtensionTypeSpec(
        extension_name=JSON_EXTENSION_NAME,
        metadata=None,
        bq_type=BQType.JSON,
    ),
    "geography": _ExtensionTypeSpec(
        extension_name=GEOGRAPHY_EXTENSION_NAME,
        metadata=GEOGRAPHY_EXTENSION_METADATA,
        bq_type=BQType.GEOGRAPHY,
    ),
}

_BIGNUMERIC_UNSUPPORTED = (
    "BIGNUMERIC",
    (
        "Polars does not support BigQuery BIGNUMERIC/BIGDECIMAL columns "
        "(Decimal precision > 38) in LazyFrame schema for column(s): {col!r}. "
        "Cast them to NUMERIC, FLOAT64, or STRING in SQL or exclude them."
    ),
)

# Known BigQuery types that cannot be represented in Polars schemas or exported
# over Arrow C Data Interface FFI: REST alias -> `(skip_reason, strict_message)`.
_UNSUPPORTED_TYPE_MAP: dict[str, tuple[str, str]] = {
    "bignumeric": _BIGNUMERIC_UNSUPPORTED,
    "bigdecimal": _BIGNUMERIC_UNSUPPORTED,
    "interval": (
        "INTERVAL",
        (
            "BigQuery INTERVAL column {col!r} is not supported over "
            "Arrow C Data Interface FFI. Cast it in SQL or exclude it."
        ),
    ),
    "range": (
        "RANGE",
        (
            "BigQuery RANGE column {col!r} is not supported in Polars. "
            "Cast it in SQL or exclude it."
        ),
    ),
}


def _is_supported_timestamp_precision(precision: Any) -> bool:
    return (
        precision is None
        or (
            isinstance(precision, int)
            and not isinstance(precision, bool)
            and precision == 6
        )
        or precision == "6"
    )


def _build_extension_for_spec(
    builder: ExtensionDtypeBuilder,
    spec: _ExtensionTypeSpec,
    storage: DataType,
    *,
    path: str,
    strict: bool,
) -> DataType:
    if strict:
        return builder(spec.extension_name, storage, spec.metadata)
    return _build_extension_dtype_checked(
        builder,
        spec.extension_name,
        storage,
        spec.metadata,
        path,
    )


def _extract_skip_struct(
    field: Mapping[str, Any],
    *,
    path: str,
    depth: int,
    extension_dtype_builder: ExtensionDtypeBuilder,
) -> tuple[DataType, BQType]:
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
            raise _UnsupportedBigQueryType("unnamed STRUCT subfield", f"{path}[{idx}]")
        normalized_sub_name = sub_name.casefold()
        if normalized_sub_name in seen:
            raise _UnsupportedBigQueryType(
                "duplicate STRUCT subfield", f"{path}.{sub_name}"
            )
        seen.add(normalized_sub_name)
        sub_dtype, _ = _extract_field_dtype(
            subfield,
            path=f"{path}.{sub_name}",
            strict=False,
            depth=depth + 1,
            extension_dtype_builder=extension_dtype_builder,
        )
        polars_fields.append(Field(sub_name, sub_dtype))
    return Struct(polars_fields), BQType.STRUCT


def _extract_strict_struct(
    field: Mapping[str, Any],
    *,
    top_col_name: str,
    depth: int,
    extension_dtype_builder: ExtensionDtypeBuilder,
) -> tuple[DataType, BQType]:
    raw_subfields = field.get("fields")
    if raw_subfields is None:
        raw_subfields = []
    elif not isinstance(raw_subfields, Sequence) or isinstance(
        raw_subfields, (str, bytes)
    ):
        msg = (
            "Invalid response from BigQuery: STRUCT 'fields' must be a list, "
            f"got {type(raw_subfields).__name__}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)

    polars_fields: list[Field] = []
    seen_names: set[str] = set()
    dup_names: list[str] = []
    dup_seen: set[str] = set()
    for subfield in raw_subfields:
        if not isinstance(subfield, Mapping):
            msg = (
                "Invalid response from BigQuery: expected STRUCT subfield object, "
                f"got {type(subfield).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        raw_sub_name = subfield.get("name", "")
        sub_name = "" if raw_sub_name is None else raw_sub_name
        if not isinstance(sub_name, str):
            msg = (
                "Invalid response from BigQuery: STRUCT subfield 'name' must be a string, "
                f"got {type(sub_name).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        if sub_name in seen_names:
            if sub_name not in dup_seen:
                dup_names.append(sub_name)
                dup_seen.add(sub_name)
            continue
        seen_names.add(sub_name)
        sub_dtype, _ = _extract_field_dtype(
            subfield,
            path=f"{top_col_name}.{sub_name}",
            strict=True,
            top_col_name=top_col_name,
            depth=depth + 1,
            extension_dtype_builder=extension_dtype_builder,
        )
        polars_fields.append(Field(sub_name, sub_dtype))

    if dup_names:
        dup_list = ", ".join(repr(name) for name in dup_names)
        struct_name = field.get("name") or "<unnamed>"
        hint = (
            " Alias unnamed STRUCT fields explicitly with `AS`."
            if "" in dup_seen
            else ""
        )
        msg = (
            f"BigQuery STRUCT field {struct_name!r} contains duplicate "
            f"subfield name(s): {dup_list}.{hint}"
        )
        raise ValueError(msg)
    return Struct(polars_fields), BQType.STRUCT


def _raise_malformed_field(
    *,
    strict: bool,
    path: str,
    skip_reason: str,
    strict_message: str,
) -> NoReturn:
    if strict:
        raise polars_bigquery.exceptions.BigQueryError(strict_message)
    raise _UnsupportedBigQueryType(skip_reason, path)


def _extract_field_dtype(
    field: Any,
    *,
    path: str,
    strict: bool,
    top_col_name: str = "",
    depth: int = 0,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> tuple[DataType, BQType]:
    """Convert a BigQuery REST `TableFieldSchema` dict into `(DataType, BQType)`.

    Shared by both table-metadata extraction (`strict=False`, which signals
    omitted columns via `_UnsupportedBigQueryType`) and dry-run query schema
    extraction (`strict=True`, which fails eagerly with `BigQueryError`,
    `ValueError`, or `TypeError` since verbatim queries cannot skip selected
    columns).

    Note: the REST API uses Legacy SQL data type names, but user-entered schemas
    may include the newer aliases (https://cloud.google.com/bigquery/docs/data-types).

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    """
    if depth > MAX_SCHEMA_NESTING_DEPTH:
        _raise_malformed_field(
            strict=strict,
            path=path,
            skip_reason=f"STRUCT nesting depth exceeds {MAX_SCHEMA_NESTING_DEPTH}",
            strict_message=(
                "Invalid response from BigQuery: schema nesting depth exceeds "
                f"maximum supported depth ({MAX_SCHEMA_NESTING_DEPTH})."
            ),
        )

    if not isinstance(field, Mapping):
        _raise_malformed_field(
            strict=strict,
            path=path,
            skip_reason="malformed field metadata",
            strict_message=(
                "Invalid response from BigQuery: expected schema field object, "
                f"got {type(field).__name__}."
            ),
        )

    raw_mode = field.get("mode")
    if raw_mode is not None and not isinstance(raw_mode, str):
        _raise_malformed_field(
            strict=strict,
            path=path,
            skip_reason=f"invalid mode {raw_mode!r}",
            strict_message=(
                "Invalid response from BigQuery: field 'mode' must be a string, "
                f"got {type(raw_mode).__name__}."
            ),
        )

    raw_type = field.get("type")
    if raw_type is not None and not isinstance(raw_type, str):
        _raise_malformed_field(
            strict=strict,
            path=path,
            skip_reason=f"invalid type {raw_type!r}",
            strict_message=(
                "Invalid response from BigQuery: field 'type' must be a string, "
                f"got {type(raw_type).__name__}."
            ),
        )

    mode = (raw_mode or "").casefold()
    if mode not in _VALID_FIELD_MODES:
        _raise_malformed_field(
            strict=strict,
            path=path,
            skip_reason=f"unsupported mode {raw_mode!r}",
            strict_message=(
                "Invalid response from BigQuery: unsupported field 'mode' "
                f"{raw_mode!r}."
            ),
        )

    type_ = (raw_type or "").casefold()

    # Check for BQ ARRAY (Polars List) first because the REST API encodes ARRAY
    # via mode="REPEATED" rather than a separate type name.
    if mode == "repeated":
        ext_spec = _EXTENSION_TYPE_MAP.get(type_)
        if ext_spec is not None:
            dtype = _build_extension_for_spec(
                extension_dtype_builder,
                ext_spec,
                List(String()),
                path=path,
                strict=strict,
            )
            return dtype, BQType.ARRAY
        inner_type, _ = _extract_field_dtype(
            {**field, "mode": "NULLABLE"},
            path=path,
            strict=strict,
            top_col_name=top_col_name,
            depth=depth + 1,
            extension_dtype_builder=extension_dtype_builder,
        )
        return List(inner_type), BQType.ARRAY

    if (scalar_entry := _SCALAR_TYPE_MAP.get(type_)) is not None:
        dtype_factory, bq_type = scalar_entry
        return dtype_factory(), bq_type

    if (ext_spec := _EXTENSION_TYPE_MAP.get(type_)) is not None:
        dtype = _build_extension_for_spec(
            extension_dtype_builder,
            ext_spec,
            String(),
            path=path,
            strict=strict,
        )
        return dtype, ext_spec.bq_type

    if type_ == "timestamp":
        precision = field.get("timestampPrecision")
        if _is_supported_timestamp_precision(precision):
            return Datetime(time_unit="us", time_zone="UTC"), BQType.TIMESTAMP
        if strict:
            msg = (
                f"BigQuery TIMESTAMP(precision={precision}) column {top_col_name!r} "
                "is not supported in Polars. Cast to microsecond TIMESTAMP in SQL."
            )
            raise TypeError(msg)
        raise _UnsupportedBigQueryType(f"TIMESTAMP(precision={precision})", path)

    if type_ in ("record", "struct"):
        if strict:
            return _extract_strict_struct(
                field,
                top_col_name=top_col_name,
                depth=depth,
                extension_dtype_builder=extension_dtype_builder,
            )
        return _extract_skip_struct(
            field,
            path=path,
            depth=depth,
            extension_dtype_builder=extension_dtype_builder,
        )

    if (unsupported := _UNSUPPORTED_TYPE_MAP.get(type_)) is not None:
        skip_reason, strict_template = unsupported
        if strict:
            raise TypeError(strict_template.format(col=top_col_name))
        raise _UnsupportedBigQueryType(skip_reason, path)

    if strict:
        msg = f"got unexpected BigQuery type: {type_}"
        raise TypeError(msg)
    raise _UnsupportedBigQueryType(f"unsupported type {type_ or '<missing>'}", path)


def extract_polars_schema_from_fields(
    fields: Sequence[Mapping[str, Any]],
    *,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> pl.Schema:
    """Convert BigQuery `TableFieldSchema` entries into a Polars schema.

    Strictly validates every entry (no silent column skipping) for use with
    `scan_query` dry-run schema discovery.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    """
    pl_schema: dict[str, DataType] = {}
    duplicates: list[str] = []
    seen_duplicates: set[str] = set()
    for field in fields:
        if not isinstance(field, Mapping):
            msg = (
                "Invalid response from BigQuery: expected schema field object, "
                f"got {type(field).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        raw_name = field.get("name")
        if not isinstance(raw_name, str):
            msg = (
                "Invalid response from BigQuery: schema field 'name' must be a string, "
                f"got {type(raw_name).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        if raw_name == "":
            msg = (
                "Invalid response from BigQuery: schema field 'name' must be a "
                "non-empty string."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        if raw_name in pl_schema:
            if raw_name not in seen_duplicates:
                duplicates.append(raw_name)
                seen_duplicates.add(raw_name)
            continue
        dtype, _ = _extract_field_dtype(
            field,
            path=raw_name,
            strict=True,
            top_col_name=raw_name,
            extension_dtype_builder=extension_dtype_builder,
        )
        pl_schema[raw_name] = dtype

    if duplicates:
        dup_list = ", ".join(repr(name) for name in duplicates)
        msg = f"BigQuery schema contains duplicate column name(s): {dup_list}."
        raise ValueError(msg)

    return pl.Schema(pl_schema)


_VALID_TIME_PARTITIONING_TYPES = frozenset({"day", "hour", "month", "year"})


def extract_schema_metadata(
    table_metadata: Mapping[str, Any],
    *,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> ExtractedSchema:
    """Extract a Polars schema, authoritative BigQuery types, and skipped columns.

    Pure metadata transformation with no warning side effects. Raises
    ``TypeError`` when the top-level ``table_metadata`` argument is not a
    mapping, and ``ValueError`` when nested REST metadata fields are malformed
    (e.g., non-mapping ``schema`` / ``timePartitioning``, non-list ``fields``,
    or invalid ``timePartitioning.type``) so corrupted responses never silently
    degrade into a 0-row empty table scan.

    Pass `extension_dtype_builder` to inject custom `JSON`/`GEOGRAPHY` dtype
    resolution independently of Polars' global extension registry or `os.environ`.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table
    """
    if not isinstance(table_metadata, Mapping):
        msg = (
            "Malformed BigQuery table metadata: expected a mapping, "
            f"got {type(table_metadata).__name__}"
        )
        raise TypeError(msg)

    raw_schema = table_metadata.get("schema")
    if raw_schema is None:
        fields: list[Any] | tuple[Any, ...] = []
    elif not isinstance(raw_schema, Mapping):
        msg = (
            "Malformed BigQuery table metadata: expected 'schema' to be a mapping, "
            f"got {type(raw_schema).__name__}"
        )
        raise ValueError(msg)
    else:
        raw_fields = raw_schema.get("fields")
        if raw_fields is None:
            fields = []
        elif not isinstance(raw_fields, (list, tuple)):
            msg = (
                "Malformed BigQuery table metadata: expected 'schema.fields' to be a list, "
                f"got {type(raw_fields).__name__}"
            )
            raise ValueError(msg)
        else:
            fields = raw_fields

    pl_schema: dict[str, DataType] = {}
    seen_names: set[str] = set()
    bq_types: dict[str, BQType] = {}
    skipped_columns: list[SkippedColumn] = []

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
        normalized_name = name.casefold()
        if normalized_name in seen_names:
            skipped_columns.append(
                SkippedColumn(
                    name=name,
                    reason="duplicate column",
                    path=name,
                )
            )
            continue
        # Record the name before validating its type so a skipped column cannot be
        # shadowed by a later duplicate or overwritten by an ingestion-time pseudo-column.
        seen_names.add(normalized_name)
        try:
            dtype, bq_type = _extract_field_dtype(
                field,
                path=name,
                strict=False,
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
    time_partitioning = table_metadata.get("timePartitioning")
    if time_partitioning is not None:
        if not isinstance(time_partitioning, Mapping):
            msg = (
                "Malformed BigQuery table metadata: expected 'timePartitioning' "
                f"to be a mapping, got {type(time_partitioning).__name__}"
            )
            raise ValueError(msg)
        part_field = time_partitioning.get("field")
        if part_field is not None and not isinstance(part_field, str):
            msg = (
                "Malformed BigQuery table metadata: expected 'timePartitioning.field' "
                f"to be a string, got {type(part_field).__name__}"
            )
            raise ValueError(msg)
        if not part_field:
            raw_part_type = time_partitioning.get("type")
            if raw_part_type is None or raw_part_type == "":
                part_type = "day"
            elif isinstance(raw_part_type, str):
                part_type = raw_part_type.casefold()
                if part_type not in _VALID_TIME_PARTITIONING_TYPES:
                    msg = (
                        "Malformed BigQuery table metadata: unsupported "
                        f"'timePartitioning.type' value {raw_part_type!r}"
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    "Malformed BigQuery table metadata: expected "
                    f"'timePartitioning.type' to be a string, got {type(raw_part_type).__name__}"
                )
                raise ValueError(msg)
            if part_type == "day" and "_partitiondate" not in seen_names:
                pl_schema["_PARTITIONDATE"] = Date()
                bq_types["_PARTITIONDATE"] = BQType.DATE
                seen_names.add("_partitiondate")
            if "_partitiontime" not in seen_names:
                pl_schema["_PARTITIONTIME"] = Datetime(time_unit="us", time_zone="UTC")
                bq_types["_PARTITIONTIME"] = BQType.TIMESTAMP
                seen_names.add("_partitiontime")

    return ExtractedSchema(
        polars_schema=pl.Schema(pl_schema),
        bq_types=bq_types,
        skipped_columns=tuple(skipped_columns),
    )


def extract_polars_schema(
    table_metadata: Mapping[str, Any],
    *,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
    require_non_empty: bool = False,
) -> pl.Schema:
    """Convert a BigQuery table REST response into a Polars schema.

    When `require_non_empty=True`, raises `BigQueryError` if `schema.fields` is
    missing or empty *before* appending ingestion-time partitioning pseudo-columns.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table
    """
    extracted = extract_schema_metadata(
        table_metadata,
        extension_dtype_builder=extension_dtype_builder,
    )
    if require_non_empty:
        raw_schema = table_metadata.get("schema")
        raw_fields = (
            raw_schema.get("fields") if isinstance(raw_schema, Mapping) else None
        )
        if not raw_fields:
            msg = (
                "Invalid response from BigQuery: table metadata did not "
                "contain any schema fields."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
    return extracted.polars_schema
