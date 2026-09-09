from __future__ import annotations

import polars._reexport as pl
from polars.datatypes import (
    Binary,
    Boolean,
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
from polars.datatypes import DataType


def _extract_data_type(field: dict) -> DataType:
    """Convert a BigQuery type into a Polars type.

    Note: the REST API uses the names from the Legacy SQL data types, but if
    user-entered it may include the newer aliases.
    (https://cloud.google.com/bigquery/docs/data-types).

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema
    """
    # Check for BQ ARRAY (polars List) type first because it's not returned as
    # a separate type in the BQ REST API. Instead, it uses the 'mode' field to
    # indicate an ARRAY type.
    if field.get("mode", "").casefold() == "repeated":
        inner_type = _extract_data_type(
            {
                "name": field.get("name", ""),
                "type": field.get("type", ""),
                "fields": field.get("fields", []),
                "mode": "NULLABLE",
            }
        )
        return List(inner_type)

    type_ = field.get("type", "").casefold()

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
        # TODO: support geopolars data types, if available
        return String()
    if type_ in ("float", "float64"):
        return Float64()
    if type_ in ("integer", "int64"):
        return Int64()
    if type_ in ("numeric", "decimal"):
        # BigQuery NUMERIC type has precision 38 and scale 9.
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
        return Decimal(precision=38, scale=9)
    if type_ in ("record", "struct"):
        polars_fields = [
            Field(field.name, _extract_data_type(field)) for field in field.get("fields", [])
        ]
        return Struct(polars_fields)
    if type_ == "string":
        return String()
    if type_ == "time":
        return Time()
    if type_ == "timestamp":
        return Datetime(time_unit="us", time_zone="utc")

    message = f"got unexpected BigQuery type: {type_}"
    raise TypeError(message)


def extract_polars_schema(table_metadata: dict) -> pl.Schema:
    """Convert a BigQuery table REST response into a Polars schema.

    See: https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table
    """
    pl_schema = {}
    for field in table_metadata.get("schema", {}).get("fields", []):
        pl_schema[field.get("name", "")] = _extract_data_type(field)

    # If table is ingestion time partitioned, add pseudocolumn for _PARTITIONDATE
    # to allow for partition filters. See:
    # https://cloud.google.com/bigquery/docs/partitioned-tables#ingestion_time
    if (
        time_partitioning := table_metadata.get("timePartitioning")
    ) is not None and time_partitioning.field is None:
        pl_schema["_PARTITIONDATE"] = Date()

    return pl.Schema(pl_schema)
