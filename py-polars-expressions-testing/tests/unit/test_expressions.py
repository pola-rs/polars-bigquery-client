from __future__ import annotations

from datetime import date, datetime

import polars as pl
from polars.testing import assert_frame_equal
from polars_expressions_testing import unsupported


def test_unsupported_is_noop_across_dtypes() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2, None],
            "float": [1.5, 2.5, None],
            "str": ["a", "b", None],
            "bool": [True, False, None],
            "date": [date(2024, 1, 1), date(2024, 1, 2), None],
            "datetime": [
                datetime(2024, 1, 1, 12, 0),  # noqa: DTZ001
                datetime(2024, 1, 2, 12, 0),  # noqa: DTZ001
                None,
            ],
            "list": [[1, 2], [3], None],
        }
    )
    result = df.select(
        unsupported(pl.col("int")),
        unsupported("float"),
        unsupported(pl.col("str")),
        unsupported(pl.col("bool")),
        unsupported(pl.col("date")),
        unsupported(pl.col("datetime")),
        unsupported(pl.col("list")),
    )
    assert_frame_equal(result, df)


def test_unsupported_in_filter_expression() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    result = lf.filter(unsupported(pl.col("a")) > 1).collect()
    expected = pl.DataFrame({"a": [2, 3], "b": [20, 30]})
    assert_frame_equal(result, expected)
