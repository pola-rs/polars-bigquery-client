from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import polars as pl

from polars_bigquery.core.compiler.ir.base import (
    Expr,
    Literal,
    NullLiteral,
    Unsupported,
)
from polars_bigquery.core.compiler.ir.boolean import BoolLiteral
from polars_bigquery.core.compiler.ir.list_ import IsIn, ListLiteral
from polars_bigquery.core.compiler.ir.numeric import FloatLiteral, IntLiteral
from polars_bigquery.core.compiler.ir.string import StringLiteral
from polars_bigquery.core.compiler.ir.temporal import DateLiteral, TimestampLiteral
from polars_bigquery.core.compiler.parser.base import (
    NULL_LITERAL_PARSERS,
    UNSUPPORTED_RECORD,
    unwrap_literal_json,
)
from polars_bigquery.core.compiler.parser.boolean import BOOLEAN_LITERAL_PARSERS
from polars_bigquery.core.compiler.parser.numeric import NUMERIC_LITERAL_PARSERS
from polars_bigquery.core.compiler.parser.string import STRING_LITERAL_PARSERS
from polars_bigquery.core.compiler.parser.temporal import (
    TEMPORAL_LITERAL_PARSERS,
    parse_datetime_literal,
)

_DATETIME_UNIT_NAMES: dict[str, str] = {
    "us": "Microseconds",
    "ms": "Milliseconds",
    "ns": "Nanoseconds",
}

_SCALAR_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    **NULL_LITERAL_PARSERS,
    **BOOLEAN_LITERAL_PARSERS,
    **NUMERIC_LITERAL_PARSERS,
    **STRING_LITERAL_PARSERS,
    **TEMPORAL_LITERAL_PARSERS,
}


def parse_ipc_series_to_list_literal(raw_bytes: bytes) -> Expr:
    """Decode an Arrow IPC stream representing a 1-column Polars Series into a ListLiteral."""
    try:
        df = pl.read_ipc_stream(io.BytesIO(raw_bytes))
    except (pl.exceptions.PolarsError, OSError, ValueError):
        return Unsupported()

    if df.width != 1 or df.height == 0:
        return Unsupported()

    series = df.to_series()
    if series.null_count() > 0:
        return Unsupported()

    if series.dtype.is_integer():
        return ListLiteral(
            values=tuple(IntLiteral(value=int(v)) for v in series.to_list())
        )
    if series.dtype.is_float():
        if series.is_nan().any():
            return Unsupported()
        return ListLiteral(
            values=tuple(FloatLiteral(value=float(v)) for v in series.to_list())
        )
    if series.dtype == pl.String:
        return ListLiteral(
            values=tuple(StringLiteral(value=str(v)) for v in series.to_list())
        )
    if series.dtype == pl.Boolean:
        return ListLiteral(
            values=tuple(BoolLiteral(value=bool(v)) for v in series.to_list())
        )
    if series.dtype == pl.Date:
        return ListLiteral(
            values=tuple(
                DateLiteral(days=int(d)) for d in series.to_physical().to_list()
            )
        )
    if isinstance(series.dtype, pl.Datetime):
        unit_str = _DATETIME_UNIT_NAMES.get(series.dtype.time_unit)
        if unit_str is None:
            return Unsupported()
        parsed_items = [
            parse_datetime_literal([int(t), unit_str, series.dtype.time_zone])
            for t in series.to_physical().to_list()
        ]
        if all(isinstance(item, TimestampLiteral) for item in parsed_items):
            return ListLiteral(
                values=tuple(
                    item for item in parsed_items if isinstance(item, TimestampLiteral)
                )
            )
    return Unsupported()


def parse_list_literal(value: Any) -> Expr:
    """Parse a List or Series literal from Polars JSON into a ListLiteral or Unsupported."""
    if not isinstance(value, list) or not value:
        return Unsupported()

    if all(
        isinstance(b, int) and not isinstance(b, bool) and 0 <= b <= 255 for b in value
    ):
        return parse_ipc_series_to_list_literal(bytes(value))

    parsed_literals: list[Literal] = []
    for elem in value:
        unwrapped = unwrap_literal_json(elem)
        if not isinstance(unwrapped, dict) or len(unwrapped) != 1:
            return Unsupported()
        polars_type, elem_val = next(iter(unwrapped.items()))
        parser = _SCALAR_LITERAL_PARSERS.get(polars_type)
        if parser is None:
            return Unsupported()
        ir_elem = parser(elem_val)
        if not isinstance(ir_elem, Literal) or isinstance(
            ir_elem, (NullLiteral, ListLiteral)
        ):
            return Unsupported()
        parsed_literals.append(ir_elem)
    return ListLiteral(values=tuple(parsed_literals))


# Keys represent Polars Rust AST `LiteralValue` / `AnyValue` list/series variant names
# emitted inside `{"Literal": ...}` in serialized Polars JSON.
LIST_LITERAL_PARSERS: dict[str, Callable[[Any], Expr]] = {
    "List": parse_list_literal,
    "Series": parse_list_literal,
}


def parse_is_in_function(
    isin_spec: Any, inputs: list[Any]
) -> tuple[str, Any, tuple[Any, ...]]:
    """Extract IR constructor and child JSONs for a Polars `IsIn` function node."""
    if len(inputs) != 2:
        return UNSUPPORTED_RECORD
    isin_opts = (
        isin_spec["IsIn"]
        if isinstance(isin_spec, dict) and "IsIn" in isin_spec
        else isin_spec
    )
    nulls_equal = (
        isin_opts.get("nulls_equal", False)
        if isinstance(isin_opts, dict)
        else bool(isin_opts)
    )
    if not nulls_equal:
        return ("Binary", IsIn, (inputs[0], inputs[1]))
    return UNSUPPORTED_RECORD


LIST_FUNCTION_PARSERS: dict[
    str, Callable[[Any, list[Any]], tuple[str, Any, tuple[Any, ...]]]
] = {
    "IsIn": parse_is_in_function,
}
