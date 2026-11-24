from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
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

import polars_bigquery.exceptions
from polars_bigquery.core.types import (
    BQ_NUMERIC_MAX_PRECISION,
    BQ_NUMERIC_MAX_SCALE,
    GEOGRAPHY_EXTENSION_METADATA,
    GEOGRAPHY_EXTENSION_NAME,
    JSON_EXTENSION_NAME,
    build_extension_dtype,
)

# BigQuery enforces a maximum STRUCT nesting depth of 15. Cap schema parsing
# well above that ceiling so pathological payloads fail with BigQueryError
# instead of Python RecursionError.
MAX_SCHEMA_NESTING_DEPTH = 64

ExtensionDtypeBuilder = Callable[[str, DataType, str | None], DataType]

_VALID_FIELD_MODES = frozenset({"", "nullable", "required", "repeated"})


@dataclasses.dataclass(frozen=True)
class SkippedColumn:
    """Metadata describing a top-level BigQuery column omitted during schema extraction."""

    name: str
    reason: str
    path: str


@dataclasses.dataclass(frozen=True)
class ExtractedSchema:
    """Result of extracting a Polars schema and skipped columns from table metadata."""

    polars_schema: pl.Schema
    skipped_columns: tuple[SkippedColumn, ...]


class _UnsupportedBigQueryType(Exception):
    """Internal exception raised when a field cannot be represented in `scan_table`."""

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
            if (
                current.precision is not None
                and current.precision > BQ_NUMERIC_MAX_PRECISION
            ):
                return True
        elif isinstance(current, pl.BaseExtension):
            stack.append(current.ext_storage())
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


@dataclasses.dataclass(frozen=True, slots=True)
class _ExtensionTypeSpec:
    """Arrow extension mapping specification for a BigQuery SQL extension type."""

    extension_name: str
    metadata: str | None


# Mapping of casefolded BigQuery REST scalar type names (and legacy SQL aliases)
# to Polars dtype factories so every lookup produces a fresh `DataType` instance.
# https://cloud.google.com/bigquery/docs/data-types
_SCALAR_TYPE_MAP: dict[str, Callable[[], DataType]] = {
    "bool": Boolean,
    "boolean": Boolean,
    "bytes": Binary,
    "date": Date,
    # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
    # https://stackoverflow.com/a/47724366/101923
    "datetime": lambda: Datetime(time_unit="us"),
    "float": Float64,
    "float64": Float64,
    "integer": Int64,
    "int64": Int64,
    # BigQuery NUMERIC type has precision 38 and scale 9.
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
    "numeric": lambda: Decimal(
        precision=BQ_NUMERIC_MAX_PRECISION, scale=BQ_NUMERIC_MAX_SCALE
    ),
    "decimal": lambda: Decimal(
        precision=BQ_NUMERIC_MAX_PRECISION, scale=BQ_NUMERIC_MAX_SCALE
    ),
    "string": String,
    "time": Time,
}

# BigQuery Arrow extension types. For `mode="REPEATED"` (`ARRAY<JSON>` /
# `ARRAY<GEOGRAPHY>`), BigQuery attaches `ARROW:extension:name` to the outer
# `List(String())` field rather than the inner `String()` element.
_EXTENSION_TYPE_MAP: dict[str, _ExtensionTypeSpec] = {
    "json": _ExtensionTypeSpec(extension_name=JSON_EXTENSION_NAME, metadata=None),
    "geography": _ExtensionTypeSpec(
        extension_name=GEOGRAPHY_EXTENSION_NAME,
        metadata=GEOGRAPHY_EXTENSION_METADATA,
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


def _extract_struct_dtype(
    field: Mapping[str, Any],
    *,
    path: str,
    strict: bool,
    top_col_name: str,
    depth: int,
    extension_dtype_builder: ExtensionDtypeBuilder,
) -> DataType:
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
        raw_sub_name = subfield.get("name", "")
        sub_name = "" if raw_sub_name is None else raw_sub_name
        if sub_name in seen_names:
            if sub_name not in dup_seen:
                dup_names.append(sub_name)
                dup_seen.add(sub_name)
            continue
        seen_names.add(sub_name)
        sub_dtype = _extract_field_dtype(
            subfield,
            path=f"{path}.{sub_name}",
            strict=strict,
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
    return Struct(polars_fields)


def _extract_field_dtype(
    field: Mapping[str, Any],
    *,
    path: str,
    strict: bool,
    top_col_name: str = "",
    depth: int = 0,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> DataType:
    """Convert a BigQuery REST `TableFieldSchema` dict into a Polars `DataType`.

    Shared by table-metadata extraction (`strict=False`, which omits
    unsupported columns via `_UnsupportedBigQueryType`) and dry-run query
    schema extraction (`strict=True`, which fails eagerly with `TypeError` or
    `ValueError` since verbatim queries cannot skip selected columns).

    Note: the REST API uses Legacy SQL data type names, but user-entered schemas
    may include the newer aliases (https://cloud.google.com/bigquery/docs/data-types).

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    """
    if depth > MAX_SCHEMA_NESTING_DEPTH:
        msg = (
            "Invalid response from BigQuery: schema nesting depth exceeds "
            f"maximum supported depth ({MAX_SCHEMA_NESTING_DEPTH})."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)

    raw_mode = field.get("mode")
    mode = (raw_mode or "").casefold()
    if mode not in _VALID_FIELD_MODES:
        if strict:
            msg = (
                f"BigQuery column {top_col_name!r} uses unsupported field mode "
                f"{raw_mode!r}."
            )
            raise TypeError(msg)
        raise _UnsupportedBigQueryType(f"unsupported mode {raw_mode!r}", path)

    type_ = (field.get("type") or "").casefold()

    # Check for BQ ARRAY (Polars List) first because the REST API encodes ARRAY
    # via mode="REPEATED" rather than a separate type name.
    if mode == "repeated":
        ext_spec = _EXTENSION_TYPE_MAP.get(type_)
        if ext_spec is not None:
            return extension_dtype_builder(
                ext_spec.extension_name,
                List(String()),
                ext_spec.metadata,
            )
        inner_type = _extract_field_dtype(
            {**field, "mode": "NULLABLE"},
            path=path,
            strict=strict,
            top_col_name=top_col_name,
            depth=depth + 1,
            extension_dtype_builder=extension_dtype_builder,
        )
        return List(inner_type)

    if (dtype_factory := _SCALAR_TYPE_MAP.get(type_)) is not None:
        return dtype_factory()

    if (ext_spec := _EXTENSION_TYPE_MAP.get(type_)) is not None:
        return extension_dtype_builder(
            ext_spec.extension_name,
            String(),
            ext_spec.metadata,
        )

    if type_ == "timestamp":
        precision = field.get("timestampPrecision")
        if _is_supported_timestamp_precision(precision):
            return Datetime(time_unit="us", time_zone="UTC")
        if strict:
            msg = (
                f"BigQuery TIMESTAMP(precision={precision!r}) column {top_col_name!r} "
                "is not supported in Polars. Cast to microsecond TIMESTAMP in SQL."
            )
            raise TypeError(msg)
        raise _UnsupportedBigQueryType(f"TIMESTAMP(precision={precision!r})", path)

    if type_ in ("record", "struct"):
        return _extract_struct_dtype(
            field,
            path=path,
            strict=strict,
            top_col_name=top_col_name,
            depth=depth,
            extension_dtype_builder=extension_dtype_builder,
        )

    if (unsupported := _UNSUPPORTED_TYPE_MAP.get(type_)) is not None:
        skip_reason, strict_template = unsupported
        if strict:
            raise TypeError(strict_template.format(col=top_col_name))
        raise _UnsupportedBigQueryType(skip_reason, path)

    if strict:
        msg = f"got unexpected BigQuery type: {type_} (column {top_col_name!r})"
        raise TypeError(msg)
    raise _UnsupportedBigQueryType(f"unsupported type {type_ or '<missing>'}", path)


def extract_polars_schema_from_fields(
    fields: Sequence[Mapping[str, Any]],
    *,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> pl.Schema:
    """Convert BigQuery `TableFieldSchema` entries into a Polars schema.

    Strictly validates every entry (no column skipping) for use with
    `scan_query` dry-run schema discovery.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    """
    pl_schema: dict[str, DataType] = {}
    duplicates: list[str] = []
    seen_duplicates: set[str] = set()
    for field in fields:
        name = field["name"]
        if name in pl_schema:
            if name not in seen_duplicates:
                duplicates.append(name)
                seen_duplicates.add(name)
            continue
        pl_schema[name] = _extract_field_dtype(
            field,
            path=name,
            strict=True,
            top_col_name=name,
            extension_dtype_builder=extension_dtype_builder,
        )

    if duplicates:
        dup_list = ", ".join(repr(name) for name in duplicates)
        msg = f"BigQuery schema contains duplicate column name(s): {dup_list}."
        raise ValueError(msg)

    return pl.Schema(pl_schema)


def extract_schema_metadata(
    table_metadata: Mapping[str, Any],
    *,
    extension_dtype_builder: ExtensionDtypeBuilder = build_extension_dtype,
) -> ExtractedSchema:
    """Extract a Polars schema and skipped columns from BigQuery REST metadata.

    Pure given a deterministic `extension_dtype_builder`; the default
    `build_extension_dtype` consults Polars' global extension registry and
    startup `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` latch (and logs a warning
    if that environment variable is mutated after import). Raises
    `BigQueryError` when the top-level metadata envelope (`table_metadata`,
    `schema`, `schema.fields`, `timePartitioning`) is malformed or nesting
    exceeds `MAX_SCHEMA_NESTING_DEPTH`.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table
    """
    if not isinstance(table_metadata, Mapping):
        msg = (
            "Invalid response from BigQuery: expected table metadata object, "
            f"got {type(table_metadata).__name__}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)

    raw_schema = table_metadata.get("schema")
    if raw_schema is None:
        fields: Sequence[Mapping[str, Any]] = ()
    elif not isinstance(raw_schema, Mapping):
        msg = (
            "Invalid response from BigQuery: table metadata 'schema' must be "
            f"an object, got {type(raw_schema).__name__}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)
    else:
        raw_fields = raw_schema.get("fields")
        if raw_fields is None:
            fields = ()
        elif not isinstance(raw_fields, Sequence) or isinstance(
            raw_fields, (str, bytes)
        ):
            msg = (
                "Invalid response from BigQuery: table metadata 'schema.fields' "
                f"must be a list, got {type(raw_fields).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        else:
            fields = raw_fields

    pl_schema: dict[str, DataType] = {}
    seen_names: set[str] = set()
    skipped_columns: list[SkippedColumn] = []

    for field in fields:
        name = field["name"]
        # Record the casefolded name before evaluating the column type so a
        # skipped column prevents appending a colliding pseudo-column.
        seen_names.add(name.casefold())
        try:
            dtype = _extract_field_dtype(
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

    # For ingestion-time partitioned tables with at least one physical field,
    # append partition pseudo-columns. Schemaless (0-column) partitioned tables
    # are skipped here because BigQuery's CreateReadSession rejects them with
    # FAILED_PRECONDITION even when requesting only pseudo-columns.
    # BigQuery exposes _PARTITIONDATE only on DAY-partitioned tables, whereas
    # _PARTITIONTIME is exposed on all ingestion-time partitioned tables.
    # See: https://cloud.google.com/bigquery/docs/partitioned-tables#ingestion_time
    time_partitioning = table_metadata.get("timePartitioning")
    if fields and time_partitioning is not None:
        if not isinstance(time_partitioning, Mapping):
            msg = (
                "Invalid response from BigQuery: table metadata 'timePartitioning' "
                f"must be an object, got {type(time_partitioning).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        if not time_partitioning.get("field"):
            raw_part_type = time_partitioning.get("type")
            is_daily = not raw_part_type or raw_part_type.casefold() == "day"
            if is_daily and "_partitiondate" not in seen_names:
                pl_schema["_PARTITIONDATE"] = Date()
                seen_names.add("_partitiondate")
            if "_partitiontime" not in seen_names:
                pl_schema["_PARTITIONTIME"] = Datetime(time_unit="us", time_zone="UTC")
                seen_names.add("_partitiontime")

    return ExtractedSchema(
        polars_schema=pl.Schema(pl_schema),
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
