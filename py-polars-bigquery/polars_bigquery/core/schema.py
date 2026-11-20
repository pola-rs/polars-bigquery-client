from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
from polars.datatypes import (
    Binary,
    Boolean,
    DataType,
    Date,
    Datetime,
    Decimal,
    Duration,
    Field,
    Float64,
    Int64,
    List,
    String,
    Struct,
    Time,
)

import polars_bigquery.exceptions


def _extract_data_type(field: Mapping[str, Any]) -> DataType:
    """Convert a BigQuery type into a Polars type.

    Note: the REST API uses the names from the Legacy SQL data types, but if
    user-entered it may include the newer aliases.
    (https://cloud.google.com/bigquery/docs/data-types).

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    """
    if not isinstance(field, Mapping):
        msg = (
            "Invalid response from BigQuery: expected schema field object, "
            f"got {type(field).__name__}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)

    raw_mode = field.get("mode")
    if raw_mode is not None and not isinstance(raw_mode, str):
        msg = (
            "Invalid response from BigQuery: field 'mode' must be a string, "
            f"got {type(raw_mode).__name__}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)

    raw_type = field.get("type")
    if raw_type is not None and not isinstance(raw_type, str):
        msg = (
            "Invalid response from BigQuery: field 'type' must be a string, "
            f"got {type(raw_type).__name__}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)

    # Check for BQ ARRAY (polars List) type first because it's not returned as
    # a separate type in the BQ REST API. Instead, it uses the 'mode' field to
    # indicate an ARRAY type.
    if (raw_mode or "").casefold() == "repeated":
        inner_type = _extract_data_type(
            {
                "name": field.get("name", ""),
                "type": raw_type or "",
                "fields": field.get("fields", []),
                "mode": "NULLABLE",
            }
        )
        return List(inner_type)

    type_ = (raw_type or "").casefold()

    if type_ in ("bool", "boolean"):
        return Boolean()
    if type_ == "bytes":
        return Binary()
    if type_ == "date":
        return Date()
    if type_ == "datetime":
        # In BigQuery DATETIME is naive (no associated timezone) and TIMESTAMP is UTC.
        # https://stackoverflow.com/a/47724366/101923
        return Datetime(time_unit="us")
    if type_ == "geography":
        # TODO(https://github.com/pola-rs/polars-bigquery-client/issues/40):
        # Support GeoPolars data types, only if the package is installed.
        return String()
    if type_ in ("float", "float64"):
        return Float64()
    if type_ in ("integer", "int64"):
        return Int64()
    if type_ in ("numeric", "decimal"):
        # BigQuery NUMERIC type has precision 38 and scale 9.
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
        return Decimal(precision=38, scale=9)
    if type_ in ("bignumeric", "bigdecimal"):
        # BigQuery BIGNUMERIC type has precision 76 and scale 38.
        return Decimal(precision=76, scale=38)
    if type_ == "json":
        return String()
    if type_ == "interval":
        return Duration(time_unit="us")
    if type_ in ("record", "struct"):
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
            polars_fields.append(Field(sub_name, _extract_data_type(subfield)))
        if dup_names:
            dup_list = ", ".join(repr(name) for name in dup_names)
            struct_name = field.get("name") or "<unnamed>"
            msg = (
                f"BigQuery STRUCT field {struct_name!r} contains duplicate "
                f"subfield name(s): {dup_list}."
            )
            raise ValueError(msg)
        return Struct(polars_fields)
    if type_ == "string":
        return String()
    if type_ == "time":
        return Time()
    if type_ == "timestamp":
        return Datetime(time_unit="us", time_zone="utc")

    message = f"got unexpected BigQuery type: {type_}"
    raise TypeError(message)


def extract_polars_schema_from_fields(
    fields: Sequence[Mapping[str, Any]],
) -> pl.Schema:
    """Convert BigQuery `TableFieldSchema` entries into a Polars schema.

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
        raw_name = field.get("name", "")
        if not isinstance(raw_name, str):
            msg = (
                "Invalid response from BigQuery: schema field 'name' must be a string, "
                f"got {type(raw_name).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
        if raw_name in pl_schema:
            if raw_name not in seen_duplicates:
                duplicates.append(raw_name)
                seen_duplicates.add(raw_name)
            continue
        pl_schema[raw_name] = _extract_data_type(field)

    if duplicates:
        dup_list = ", ".join(repr(name) for name in duplicates)
        msg = f"BigQuery schema contains duplicate column name(s): {dup_list}."
        raise ValueError(msg)

    return pl.Schema(pl_schema)


def extract_polars_schema(table_metadata: Mapping[str, Any]) -> pl.Schema:
    """Convert a BigQuery table REST response into a Polars schema.

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
        raw_fields: Sequence[Mapping[str, Any]] = []
    elif isinstance(raw_schema, Mapping):
        fields_val = raw_schema.get("fields")
        if fields_val is None:
            raw_fields = []
        elif isinstance(fields_val, Sequence) and not isinstance(
            fields_val, (str, bytes)
        ):
            raw_fields = fields_val
        else:
            msg = (
                "Invalid response from BigQuery: table metadata 'schema.fields' "
                f"must be a list, got {type(fields_val).__name__}."
            )
            raise polars_bigquery.exceptions.BigQueryError(msg)
    else:
        msg = (
            "Invalid response from BigQuery: table metadata 'schema' must be an object, "
            f"got {type(raw_schema).__name__}."
        )
        raise polars_bigquery.exceptions.BigQueryError(msg)

    pl_schema = extract_polars_schema_from_fields(raw_fields)

    # If table is ingestion time partitioned, add pseudocolumns for _PARTITIONDATE
    # and _PARTITIONTIME to allow for partition filters. See:
    # https://cloud.google.com/bigquery/docs/partitioned-tables#ingestion_time
    time_partitioning = table_metadata.get("timePartitioning")
    if (
        isinstance(time_partitioning, Mapping)
        and time_partitioning.get("field") is None
    ):
        pl_schema["_PARTITIONDATE"] = Date()
        pl_schema["_PARTITIONTIME"] = Datetime(time_unit="us", time_zone="utc")

    return pl_schema
