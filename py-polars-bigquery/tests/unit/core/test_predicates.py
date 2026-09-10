"""Test coverage for `bigquery` expressions comprehension."""

from __future__ import annotations

import polars as pl
from polars_bigquery.core import predicates


def test_is_null_expression() -> None:
    expr = pl.col("id").is_null()
    assert predicates.predicate_to_row_restriction(expr) == "(`id` IS NULL)"


def test_is_not_null_expression() -> None:
    expr = pl.col("id").is_not_null()
    assert predicates.predicate_to_row_restriction(expr) == "(`id` IS NOT NULL)"


def test_parse_combined_expression() -> None:
    expr = (pl.col("str") == "2") & ((pl.col("id") > 10) | (pl.col("id") < 7.0))
    assert (
        predicates.predicate_to_row_restriction(expr)
        == "((`str` = '2') AND ((`id` > 10) OR (`id` < 7.0)))"
    )


def test_parse_gt() -> None:
    expr = pl.col("ts") > "2023-08-08"
    assert predicates.predicate_to_row_restriction(expr) == "(`ts` > '2023-08-08')"


def test_parse_gteq() -> None:
    expr = pl.col("ts") >= "2023-08-08"
    assert predicates.predicate_to_row_restriction(expr) == "(`ts` >= '2023-08-08')"


def test_parse_eq() -> None:
    expr = pl.col("ts") == "2023-08-08"
    assert predicates.predicate_to_row_restriction(expr) == "(`ts` = '2023-08-08')"


def test_parse_lt() -> None:
    expr = pl.col("ts") < "2023-08-08"
    assert predicates.predicate_to_row_restriction(expr) == "(`ts` < '2023-08-08')"


def test_parse_lteq() -> None:
    expr = pl.col("ts") <= "2023-08-08"
    assert predicates.predicate_to_row_restriction(expr) == "(`ts` <= '2023-08-08')"


def test_starts_with_expression() -> None:
    expr = pl.col("name").str.starts_with("T")
    assert predicates.predicate_to_row_restriction(expr) == "STARTS_WITH(`name`, 'T')"


def test_starts_with_combined_expression() -> None:
    expr = (
        pl.col("name").str.starts_with("T")
        & (pl.col("number") > 10)
        & (pl.col("year") == 2000)
    )
    assert (
        predicates.predicate_to_row_restriction(expr)
        == "((STARTS_WITH(`name`, 'T') AND (`number` > 10)) AND (`year` = 2000))"
    )


def test_starts_with_column_prefix() -> None:
    expr = pl.col("name").str.starts_with(pl.col("prefix"))
    assert (
        predicates.predicate_to_row_restriction(expr) == "STARTS_WITH(`name`, `prefix`)"
    )


def test_ends_with_expression() -> None:
    expr = pl.col("name").str.ends_with("xyz")
    assert predicates.predicate_to_row_restriction(expr) == "ENDS_WITH(`name`, 'xyz')"


def test_not_starts_with_expression() -> None:
    expr = ~pl.col("name").str.starts_with("T")
    assert (
        predicates.predicate_to_row_restriction(expr)
        == "(NOT STARTS_WITH(`name`, 'T'))"
    )
