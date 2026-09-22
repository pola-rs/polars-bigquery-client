from __future__ import annotations

import io
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
    UNSUPPORTED_RECORD,
    ParseRecord,
    dispatch_literal_parser,
    extract_function_inputs,
    register_parser,
)
from polars_bigquery.core.compiler.parser.temporal import (
    parse_ticks_and_unit,
    parse_timezone,
)

MAX_IN_LIST_ELEMENTS = 10_000
MAX_IPC_BYTES = 1_048_576

_DATETIME_UNIT_NAMES: dict[str, str] = {
    "us": "Microseconds",
    "ms": "Milliseconds",
    "ns": "Nanoseconds",
}


def parse_ipc_series_to_list_literal(raw_bytes: bytes) -> Expr:
    """Decode an Arrow IPC stream representing a 1-column Polars Series into a ListLiteral."""
    if not raw_bytes or len(raw_bytes) > MAX_IPC_BYTES:
        return Unsupported()

    try:
        df = pl.read_ipc_stream(io.BytesIO(raw_bytes))
    except (pl.exceptions.PolarsError, OSError, ValueError):
        return Unsupported()

    if df.width != 1 or df.height == 0 or df.height > MAX_IN_LIST_ELEMENTS:
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
        try:
            tz = parse_timezone(series.dtype.time_zone)
            timestamps = tuple(
                TimestampLiteral(
                    ticks=ticks,
                    unit=unit,
                    tz=tz,
                )
                for raw_t in series.to_physical().to_list()
                for ticks, unit in (parse_ticks_and_unit(raw_t, unit_str),)
            )
        except (TypeError, ValueError):
            return Unsupported()
        return ListLiteral(values=timestamps)
    return Unsupported()


def _try_extract_ipc_bytes(value: list[Any]) -> bytes | None:
    """Convert an integer byte list to `bytes` without redundant full-list scans."""
    first = value[0]
    if not isinstance(first, int) or isinstance(first, bool):
        return None
    if len(value) > MAX_IPC_BYTES:
        return b""
    try:
        if any(isinstance(b, bool) for b in value):
            return None
        return bytes(value)
    except (TypeError, ValueError):
        return None


@register_parser("$.Literal..List", "$.Literal..Series")
def parse_list_literal(value: Any) -> Expr:
    """Parse a List or Series literal from Polars JSON into a ListLiteral or Unsupported."""
    if not isinstance(value, list) or not value:
        return Unsupported()

    ipc_bytes = _try_extract_ipc_bytes(value)
    if ipc_bytes is not None:
        return parse_ipc_series_to_list_literal(ipc_bytes)

    if len(value) > MAX_IN_LIST_ELEMENTS:
        return Unsupported()

    parsed_literals: list[Literal] = []
    for elem in value:
        kind, ir_elem, _ = dispatch_literal_parser(elem)
        if (
            kind != "Leaf"
            or not isinstance(ir_elem, Literal)
            or isinstance(ir_elem, (NullLiteral, ListLiteral))
        ):
            return Unsupported()
        parsed_literals.append(ir_elem)
    return ListLiteral(values=tuple(parsed_literals))


@register_parser("$.Function.function.Boolean.IsIn")
def parse_is_in_function(isin_spec: Any, expr_json: Any) -> ParseRecord:
    """Extract IR constructor and child JSONs for a Polars `IsIn` function node."""
    inputs = extract_function_inputs(expr_json)
    if inputs is None or len(inputs) != 2:
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
