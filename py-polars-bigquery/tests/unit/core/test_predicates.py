"""Test coverage for `bigquery` expressions comprehension."""

from __future__ import annotations

import io
from typing import Any

import polars as pl
import pytest
from polars_bigquery.core import compiler


def _json_literal_to_sql(literal_json: dict[str, Any]) -> str | None:
    """Convert a literal from a Polars expression JSON into a BigQuery SQL string."""
    ir_node = compiler.json_to_ir({"Literal": literal_json})
    if isinstance(ir_node, compiler.ir.Literal):
        return compiler.ir_to_sql(ir_node)
    return None


def test_is_null_expression() -> None:
    expr = pl.col("id").is_null()
    assert compiler.predicate_to_row_restriction(expr) == "(`id` IS NULL)"


def test_is_not_null_expression() -> None:
    expr = pl.col("id").is_not_null()
    assert compiler.predicate_to_row_restriction(expr) == "(`id` IS NOT NULL)"


def test_parse_combined_expression() -> None:
    expr = (pl.col("str") == "2") & ((pl.col("id") > 10) | (pl.col("id") < 7.0))
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "((`str` = '2') AND ((`id` > 10) OR (`id` < 7.0)))"
    )


def test_parse_gt() -> None:
    expr = pl.col("ts") > "2023-08-08"
    assert compiler.predicate_to_row_restriction(expr) == "(`ts` > '2023-08-08')"


def test_parse_gteq() -> None:
    expr = pl.col("ts") >= "2023-08-08"
    assert compiler.predicate_to_row_restriction(expr) == "(`ts` >= '2023-08-08')"


def test_parse_eq() -> None:
    expr = pl.col("ts") == "2023-08-08"
    assert compiler.predicate_to_row_restriction(expr) == "(`ts` = '2023-08-08')"


def test_parse_lt() -> None:
    expr = pl.col("ts") < "2023-08-08"
    assert compiler.predicate_to_row_restriction(expr) == "(`ts` < '2023-08-08')"


def test_parse_lteq() -> None:
    expr = pl.col("ts") <= "2023-08-08"
    assert compiler.predicate_to_row_restriction(expr) == "(`ts` <= '2023-08-08')"


def test_starts_with_expression() -> None:
    expr = pl.col("name").str.starts_with("T")
    assert compiler.predicate_to_row_restriction(expr) == "STARTS_WITH(`name`, 'T')"


def test_starts_with_combined_expression() -> None:
    expr = (
        pl.col("name").str.starts_with("T")
        & (pl.col("number") > 10)
        & (pl.col("year") == 2000)
    )
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "((STARTS_WITH(`name`, 'T') AND (`number` > 10)) AND (`year` = 2000))"
    )


def test_starts_with_column_prefix() -> None:
    expr = pl.col("name").str.starts_with(pl.col("prefix"))
    assert (
        compiler.predicate_to_row_restriction(expr) == "STARTS_WITH(`name`, `prefix`)"
    )


def test_ends_with_expression() -> None:
    expr = pl.col("name").str.ends_with("xyz")
    assert compiler.predicate_to_row_restriction(expr) == "ENDS_WITH(`name`, 'xyz')"


def test_not_starts_with_expression() -> None:
    expr = ~pl.col("name").str.starts_with("T")
    assert (
        compiler.predicate_to_row_restriction(expr) == "(NOT STARTS_WITH(`name`, 'T'))"
    )


def test_not_eq_and_boolean_literal() -> None:
    expr = (pl.col("status") != "DELETED") & (pl.col("active") == True)
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "((`status` != 'DELETED') AND (`active` = TRUE))"
    )


def test_flexible_column_names_and_string_escaping() -> None:
    # Flexible column names (e.g. with apostrophe, spaces, hyphens, unicode) are
    # enclosed in backticks without string-style backslash escaping, whereas string
    # literals are single-quoted and escaped.
    expr = pl.col("user's-name 1") == "O'Reilly"
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "(`user's-name 1` = 'O\\'Reilly')"
    )


def test_japanese_and_multilingual_column_names() -> None:
    # Japanese column names using Kanji, Hiragana, Katakana, and mixed scripts
    # are valid BigQuery flexible column names (\p{L}, \p{N}, \p{Pc}, \p{Pd}).
    expr = (pl.col("ユーザー名") == "田中") & (pl.col("価格_円") > 1000)
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "((`ユーザー名` = '田中') AND (`価格_円` > 1000))"
    )

    # Test individual identifiers across Japanese and other languages with diacritics/marks
    assert (
        compiler.sql._column_to_sql_identifier("ユーザー名") == "`ユーザー名`"
    )  # Katakana + Kanji
    assert compiler.sql._column_to_sql_identifier("なまえ") == "`なまえ`"  # Hiragana
    assert (
        compiler.sql._column_to_sql_identifier("顧客コード_テスト#1")
        == "`顧客コード_テスト#1`"
    )
    assert (
        compiler.sql._column_to_sql_identifier("注文-番号:2026") == "`注文-番号:2026`"
    )
    assert (
        compiler.sql._column_to_sql_identifier("café_au_lait") == "`café_au_lait`"
    )  # French (\p{M})
    assert (
        compiler.sql._column_to_sql_identifier("Größe_in_cm") == "`Größe_in_cm`"
    )  # German (\p{M})
    assert (
        compiler.sql._column_to_sql_identifier("año_fiscal") == "`año_fiscal`"
    )  # Spanish (\p{M})


def test_unsupported_column_names_raise_value_error() -> None:
    # Column names containing characters disallowed by BigQuery (such as backtick,
    # backslash, dot, dollar, exclamation mark, or empty string) raise ValueError.
    for invalid_col in ("bad`col", "bad\\col", "bad.col", "bad$col", ""):
        expr = pl.col(invalid_col) == 1
        with pytest.raises(ValueError, match="Invalid BigQuery column name"):
            compiler.predicate_to_row_restriction(expr)

    # Length limit (>300 chars) is not enforced client-side so relaxed server limits work
    long_col = "a" * 301
    assert compiler.sql._column_to_sql_identifier(long_col) == f"`{long_col}`"


def test_nan_and_inf_floats_cast_to_float64() -> None:
    # Polars expressions with float("nan") or float("inf") serialize with Float: null,
    # which cannot safely distinguish nan vs inf vs -inf. They safely fallback to in-memory filtering.
    expr_nan = pl.col("val") == float("nan")  # noqa: PLW0177
    assert compiler.predicate_to_row_restriction(expr_nan) == ""

    expr_inf = pl.col("val") < float("inf")
    assert compiler.predicate_to_row_restriction(expr_inf) == ""

    expr_neg_inf = pl.col("val") > float("-inf")
    assert compiler.predicate_to_row_restriction(expr_neg_inf) == ""

    # When JSON literals contain nan, inf, and -inf, they are correctly cast to FLOAT64
    assert _json_literal_to_sql({"Float64": "nan"}) == "CAST('nan' AS FLOAT64)"
    assert _json_literal_to_sql({"Float64": "inf"}) == "CAST('inf' AS FLOAT64)"
    assert _json_literal_to_sql({"Float64": "-inf"}) == "CAST('-inf' AS FLOAT64)"
    assert _json_literal_to_sql({"Float64": "infinity"}) == "CAST('inf' AS FLOAT64)"
    assert _json_literal_to_sql({"Float64": "-infinity"}) == "CAST('-inf' AS FLOAT64)"

    # Boolean functions for NaN and Infinity
    assert (
        compiler.predicate_to_row_restriction(pl.col("val").is_nan()) == "IS_NAN(`val`)"
    )
    assert (
        compiler.predicate_to_row_restriction(pl.col("val").is_not_nan())
        == "(NOT IS_NAN(`val`))"
    )
    assert (
        compiler.predicate_to_row_restriction(pl.col("val").is_infinite())
        == "IS_INF(`val`)"
    )
    assert (
        compiler.predicate_to_row_restriction(pl.col("val").is_finite())
        == "(NOT IS_INF(`val`) AND NOT IS_NAN(`val`))"
    )


def test_escape_sql_and_column_identifier_helpers() -> None:
    assert compiler.sql._escape_sql_string("a\\b'c\nd\re") == "'a\\\\b\\'c\\nd\\re'"

    # Test Unicode category validation helper
    assert compiler.sql._is_allowed_unicode_category("a")  # \p{L} Letter
    assert compiler.sql._is_allowed_unicode_category("1")  # \p{N} Number
    assert compiler.sql._is_allowed_unicode_category(
        "_"
    )  # \p{Pc} Connector punctuation
    assert compiler.sql._is_allowed_unicode_category("-")  # \p{Pd} Dash punctuation
    assert compiler.sql._is_allowed_unicode_category("é")  # Letter / Mark
    assert not compiler.sql._is_allowed_unicode_category("$")  # Currency symbol
    assert not compiler.sql._is_allowed_unicode_category("@")  # Other punctuation

    # Test column character validation helper
    assert compiler.sql._is_valid_column_character("a")
    assert compiler.sql._is_valid_column_character(" ")
    assert compiler.sql._is_valid_column_character("&")
    assert not compiler.sql._is_valid_column_character("`")
    assert not compiler.sql._is_valid_column_character("\\")
    assert not compiler.sql._is_valid_column_character("\n")

    assert compiler.sql._column_to_sql_identifier("valid_col_1") == "`valid_col_1`"
    assert (
        compiler.sql._column_to_sql_identifier("café & tea: 100%")
        == "`café & tea: 100%`"
    )
    with pytest.raises(ValueError, match="Invalid BigQuery column name"):
        compiler.sql._column_to_sql_identifier("bad`col")
    with pytest.raises(ValueError, match="Invalid BigQuery column name"):
        compiler.sql._column_to_sql_identifier("bad\\col")


def test_json_literal_to_sql_all_scalar_types_and_immutability() -> None:
    # Verify input dictionary is not mutated by _json_literal_to_sql
    lit_dict = {"Int64": 42}
    assert _json_literal_to_sql(lit_dict) == "42"
    assert lit_dict == {"Int64": 42}

    # Empty, multi-key, or non-dict inputs
    assert _json_literal_to_sql({}) is None
    assert _json_literal_to_sql({"Int64": 42, "Extra": 1}) is None
    assert _json_literal_to_sql(None) is None  # type: ignore[arg-type]

    # Integer and Unsigned Integer scalar variants
    for int_type in (
        "Int",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    ):
        assert _json_literal_to_sql({int_type: 99}) == "99"

    # Float scalar variants including nan, inf, and -inf
    for float_type in ("Float", "Float32", "Float64"):
        assert _json_literal_to_sql({float_type: 3.5}) == "3.5"
        assert (
            _json_literal_to_sql({float_type: float("inf")}) == "CAST('inf' AS FLOAT64)"
        )
        assert (
            _json_literal_to_sql({float_type: float("-inf")})
            == "CAST('-inf' AS FLOAT64)"
        )
        assert (
            _json_literal_to_sql({float_type: float("nan")}) == "CAST('nan' AS FLOAT64)"
        )
        assert _json_literal_to_sql({float_type: "nan"}) == "CAST('nan' AS FLOAT64)"
        assert _json_literal_to_sql({float_type: "+nan"}) == "CAST('nan' AS FLOAT64)"
        assert _json_literal_to_sql({float_type: "-nan"}) == "CAST('nan' AS FLOAT64)"
        assert _json_literal_to_sql({float_type: "inf"}) == "CAST('inf' AS FLOAT64)"
        assert _json_literal_to_sql({float_type: "+inf"}) == "CAST('inf' AS FLOAT64)"
        assert (
            _json_literal_to_sql({float_type: "infinity"}) == "CAST('inf' AS FLOAT64)"
        )
        assert (
            _json_literal_to_sql({float_type: "+infinity"}) == "CAST('inf' AS FLOAT64)"
        )
        assert _json_literal_to_sql({float_type: "-inf"}) == "CAST('-inf' AS FLOAT64)"
        assert (
            _json_literal_to_sql({float_type: "-infinity"}) == "CAST('-inf' AS FLOAT64)"
        )
        assert _json_literal_to_sql({float_type: None}) is None

    # String variants
    assert _json_literal_to_sql({"String": "hello"}) == "'hello'"
    assert _json_literal_to_sql({"StringOwned": "world"}) == "'world'"

    # Boolean and Null
    assert _json_literal_to_sql({"Boolean": True}) == "TRUE"
    assert _json_literal_to_sql({"Boolean": False}) == "FALSE"
    assert _json_literal_to_sql({"Null": None}) == "NULL"

    # DateTime units: Microseconds, Milliseconds, Nanoseconds (exact and inexact)
    assert (
        _json_literal_to_sql({"DateTime": [1000, "Microseconds"]})
        == "TIMESTAMP_MICROS(1000)"
    )
    assert (
        _json_literal_to_sql({"DateTime": [2000, "Milliseconds"]})
        == "TIMESTAMP_MILLIS(2000)"
    )
    assert (
        _json_literal_to_sql({"DateTime": [5000, "Nanoseconds"]})
        == "TIMESTAMP_MICROS(5)"
    )
    assert _json_literal_to_sql({"DateTime": [5555, "Nanoseconds"]}) is None
    assert _json_literal_to_sql({"DateTime": [10, "Seconds"]}) is None

    # Malformed literal values catch ValueError/TypeError locally and return None
    assert _json_literal_to_sql({"Int64": "not-an-int"}) is None
    assert _json_literal_to_sql({"Float64": "not-a-float"}) is None
    assert _json_literal_to_sql({"DateTime": ["not-an-int", "Microseconds"]}) is None
    assert _json_literal_to_sql({"DateTime": []}) is None
    assert _json_literal_to_sql({"Date": "not-a-date"}) is None


def test_and_expression_preserves_valid_branch_with_unsupported_branch() -> None:
    expr = (pl.col("keep_col") == 1) & (pl.col("other_col").sin() > 0.5)
    assert compiler.predicate_to_row_restriction(expr) == "(`keep_col` = 1)"


def test_deep_expression_avoids_stack_overflow() -> None:
    # Build a deeply nested JSON expression tree (depth > 2500, well above Python's default
    # recursion limit of 1000) to verify that json_to_ir and ir_to_sql walk trees
    # breadth-first using a queue without hitting RecursionError.
    leaf: dict[str, object] = {
        "BinaryExpr": {
            "left": {"Column": "x"},
            "op": "Eq",
            "right": {"Literal": {"Int64": 0}},
        }
    }
    curr = leaf
    depth = 2500
    for i in range(1, depth):
        right_leaf = {
            "BinaryExpr": {
                "left": {"Column": "x"},
                "op": "Eq",
                "right": {"Literal": {"Int64": i}},
            }
        }
        curr = {
            "BinaryExpr": {
                "left": curr,
                "op": "And",
                "right": right_leaf,
            }
        }

    ir_tree = compiler.json_to_ir(curr)
    assert isinstance(ir_tree, compiler.ir.And)

    sql_result = compiler.ir_to_sql(ir_tree)
    assert sql_result is not None
    assert sql_result.startswith("(" * depth + "`x` = 0)")
    assert sql_result.endswith(f"AND (`x` = {depth - 1}))")


def test_ir_frozen_dataclasses_and_submodules() -> None:
    from dataclasses import FrozenInstanceError

    col_a = compiler.ir.base.Column("a")
    lit_10 = compiler.ir.numeric.IntLiteral(10)
    eq_node = compiler.ir.comparison.Eq(left=col_a, right=lit_10)

    assert col_a.children() == ()
    assert lit_10.children() == ()
    assert eq_node.children() == (col_a, lit_10)

    # Verify Expr is abstract and requires children() implementation
    with pytest.raises(TypeError, match="Can't instantiate abstract class Expr"):
        compiler.ir.base.Expr()  # type: ignore[abstract]

    # Verify frozen dataclass immutability
    with pytest.raises(FrozenInstanceError):
        col_a.name = "b"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        eq_node.left = col_a  # type: ignore[misc]

    # Verify submodule organization and direct IR -> SQL compilation
    str_starts = compiler.ir.string.StartsWith(
        left=compiler.ir.base.Column("name"),
        right=compiler.ir.string.StringLiteral("pre"),
    )
    assert str_starts.expr == compiler.ir.base.Column("name")
    assert str_starts.prefix == compiler.ir.string.StringLiteral("pre")

    date_cmp = compiler.ir.comparison.GtEq(
        left=compiler.ir.base.Column("dt"),
        right=compiler.ir.temporal.DateLiteral(days=10),
    )
    not_nan = compiler.ir.numeric.IsNotNan(expr=compiler.ir.base.Column("score"))

    combined = compiler.ir.boolean.And(
        left=compiler.ir.boolean.Or(left=eq_node, right=str_starts),
        right=compiler.ir.boolean.And(left=date_cmp, right=not_nan),
    )

    expected_sql = (
        "(((`a` = 10) OR STARTS_WITH(`name`, 'pre')) AND "
        "((`dt` >= DATE(TIMESTAMP_SECONDS(10 * 86400))) AND (NOT IS_NAN(`score`))))"
    )
    assert compiler.ir_to_sql(combined) == expected_sql

    # Verify parser submodules
    assert (
        compiler.parser.base.parse_null_literal(None) == compiler.ir.base.NullLiteral()
    )
    assert compiler.parser.boolean.parse_bool_literal(
        True
    ) == compiler.ir.boolean.BoolLiteral(True)
    assert compiler.parser.comparison.parse_eq(
        "Eq", {"left": {"Column": "a"}, "op": "Eq", "right": {"Literal": {"Int64": 10}}}
    ) == (
        "Binary",
        compiler.ir.comparison.Eq,
        ({"Column": "a"}, {"Literal": {"Int64": 10}}),
    )
    assert (
        compiler.parser.PARSERS["$.BinaryExpr.op.Eq"]
        is compiler.parser.comparison.parse_eq
    )
    assert compiler.parser.list_.parse_list_literal(
        [{"Int64": 1}, {"Int64": 2}]
    ) == compiler.ir.list_.ListLiteral(
        values=(compiler.ir.IntLiteral(1), compiler.ir.IntLiteral(2))
    )
    assert compiler.parser.list_.parse_is_in_function(
        {"nulls_equal": False}, [{"Column": "a"}, {"Literal": {"List": []}}]
    ) == (
        "Binary",
        compiler.ir.list_.IsIn,
        ({"Column": "a"}, {"Literal": {"List": []}}),
    )
    assert compiler.parser.numeric.parse_int_literal(42) == compiler.ir.IntLiteral(42)
    assert compiler.parser.string.parse_string_literal(
        "abc"
    ) == compiler.ir.StringLiteral("abc")
    assert compiler.parser.temporal.parse_date_literal(10) == compiler.ir.DateLiteral(
        10
    )

    # Verify sql submodules
    assert compiler.sql.base.column_to_sql_identifier("col") == "`col`"
    assert (
        compiler.sql.boolean.format_bool_literal(compiler.ir.BoolLiteral(True))
        == "TRUE"
    )
    assert (
        compiler.sql.comparison.format_eq(
            compiler.ir.Eq(left=col_a, right=lit_10), "`a`", "10"
        )
        == "(`a` = 10)"
    )
    assert (
        compiler.sql.list_.format_list_literal(
            compiler.ir.list_.ListLiteral(values=(lit_10,))
        )
        == "(10)"
    )
    assert (
        compiler.sql.list_.format_is_in(
            compiler.ir.list_.IsIn(
                left=col_a, right=compiler.ir.list_.ListLiteral(values=(lit_10,))
            ),
            "`a`",
            "(10)",
        )
        == "(`a` IN (10))"
    )
    assert compiler.sql.numeric.format_int_literal(lit_10) == "10"
    assert compiler.sql.string.escape_sql_string("abc") == "'abc'"
    assert (
        compiler.sql.temporal.format_date_literal(compiler.ir.DateLiteral(10))
        == "DATE(TIMESTAMP_SECONDS(10 * 86400))"
    )


def test_predicate_to_row_restriction_root_recursion_error_fallback(
    monkeypatch,
) -> None:
    def raise_recursion(_expr_json):
        raise RecursionError("Maximum recursion depth exceeded")

    monkeypatch.setattr(compiler, "_json_expr_to_row_restriction", raise_recursion)
    assert compiler.predicate_to_row_restriction(pl.col("a") == 1) == ""


def test_escape_sql_string_tabs_and_null_bytes() -> None:
    assert compiler.sql._escape_sql_string("a\tb\0c") == "'a\\tb\\x00c'"


def test_json_to_ir_prunes_unsupported_subtrees() -> None:
    unsupported_json = {
        "BinaryExpr": {
            "left": {"Column": "a"},
            "op": "Plus",
            "right": {"Column": "b"},
        }
    }
    ir_node = compiler.json_to_ir(unsupported_json)
    assert isinstance(ir_node, compiler.ir.Unsupported)
    assert ir_node.children() == ()


def test_predicate_to_row_restriction_pseudo_column_and_physical_column() -> None:
    from datetime import date

    expr = (pl.col("_PARTITIONDATE") == date(2024, 1, 1)) & (pl.col("val").sin() > 0.5)
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "(`_PARTITIONDATE` = DATE(TIMESTAMP_SECONDS(19723 * 86400)))"
    )

    expr_or = (pl.col("_PARTITIONDATE") == date(2024, 1, 1)) | (
        pl.col("val").sin() > 0.5
    )
    assert compiler.predicate_to_row_restriction(expr_or) == ""


def test_negation_over_partial_and_does_not_drop_data() -> None:
    # ~((a == 1) & (sin(b) > 0.5)) must NOT relax to NOT (a = 1), which would
    # drop rows where a == 1 and sin(b) <= 0.5.
    expr = ~((pl.col("a") == 1) & (pl.col("b").sin() > 0.5))
    assert compiler.predicate_to_row_restriction(expr) == ""

    # Negation over 100% exact AND is safe to push down
    exact_expr = ~((pl.col("a") == 1) & (pl.col("b") == 2))
    assert (
        compiler.predicate_to_row_restriction(exact_expr)
        == "(NOT ((`a` = 1) AND (`b` = 2)))"
    )

    # Positive monotone OR over partial AND safely relaxes to superset ((a = 1) OR (c = 3))
    or_expr = ((pl.col("a") == 1) & (pl.col("b").sin() > 0.5)) | (pl.col("c") == 3)
    assert compiler.predicate_to_row_restriction(or_expr) == "((`a` = 1) OR (`c` = 3))"

    # Negation over that OR must NOT push down because its child is a relaxed superset
    neg_or_expr = ~or_expr
    assert compiler.predicate_to_row_restriction(neg_or_expr) == ""


def test_bare_null_literal_comparison_not_pushed_down() -> None:
    expr = pl.col("a") == pl.lit(None)
    assert compiler.predicate_to_row_restriction(expr) == ""


def test_datetime_naive_vs_timezone_aware_pushdown() -> None:
    from datetime import datetime, timezone

    # Timezone-aware datetime -> BigQuery TIMESTAMP
    ts_expr = pl.col("_PARTITIONTIME") == datetime(
        2024, 1, 1, 12, 0, tzinfo=timezone.utc
    )
    assert (
        compiler.predicate_to_row_restriction(ts_expr)
        == "(`_PARTITIONTIME` = TIMESTAMP_MICROS(1704110400000000))"
    )

    # Timezone-naive datetime -> BigQuery DATETIME
    dt_expr = pl.col("created_dt") == datetime(2024, 1, 1, 12, 0)  # noqa: DTZ001
    assert (
        compiler.predicate_to_row_restriction(dt_expr)
        == "(`created_dt` = DATETIME(TIMESTAMP_MICROS(1704110400000000)))"
    )


def test_sec_quarterly_financials_submission_partition_pushdown() -> None:
    from datetime import date

    expr = (
        (pl.col("_PARTITIONDATE") >= date(2020, 1, 1))
        & (pl.col("_PARTITIONDATE") <= date(2020, 12, 31))
        & (
            pl.col("central_index_key").is_in([1652044, 1288776])
            | pl.col("company_name").str.to_uppercase().str.contains("ALPHABET INC")
        )
    )
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "(((`_PARTITIONDATE` >= DATE(TIMESTAMP_SECONDS(18262 * 86400))) "
        "AND (`_PARTITIONDATE` <= DATE(TIMESTAMP_SECONDS(18627 * 86400)))) "
        "AND ((`central_index_key` IN (1652044, 1288776)) "
        "OR REGEXP_CONTAINS(UPPER(`company_name`), 'ALPHABET INC')))"
    )


def test_sec_quarterly_financials_numbers_partition_pushdown() -> None:
    from datetime import date

    expr = (
        (pl.col("_PARTITIONDATE") >= date(2020, 1, 1))
        & (pl.col("_PARTITIONDATE") <= date(2020, 12, 31))
        & pl.col("measure_tag").is_in(
            [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "OperatingIncomeLoss",
                "NetIncomeLoss",
                "ResearchAndDevelopmentExpense",
                "EarningsPerShareDiluted",
            ]
        )
        & pl.col("units").is_in(["USD", "shares"])
    )
    assert (
        compiler.predicate_to_row_restriction(expr)
        == "((((`_PARTITIONDATE` >= DATE(TIMESTAMP_SECONDS(18262 * 86400))) "
        "AND (`_PARTITIONDATE` <= DATE(TIMESTAMP_SECONDS(18627 * 86400)))) "
        "AND (`measure_tag` IN ('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', "
        "'OperatingIncomeLoss', 'NetIncomeLoss', 'ResearchAndDevelopmentExpense', "
        "'EarningsPerShareDiluted'))) "
        "AND (`units` IN ('USD', 'shares')))"
    )


def test_is_in_and_string_contains_case_expressions() -> None:
    from datetime import date, datetime, timezone

    # Literal vs regex contains and lowercase
    assert (
        compiler.predicate_to_row_restriction(
            pl.col("name").str.to_lowercase().str.contains("alphabet", literal=True)
        )
        == "(STRPOS(LOWER(`name`), 'alphabet') > 0)"
    )

    # Series, Float, Boolean, Date, and Datetime lists in is_in
    assert (
        compiler.predicate_to_row_restriction(
            pl.col("tag").is_in(pl.Series(["a", "b"]))
        )
        == "(`tag` IN ('a', 'b'))"
    )
    assert (
        compiler.predicate_to_row_restriction(pl.col("score").is_in([1.5, 2.5]))
        == "(`score` IN (1.5, 2.5))"
    )
    assert (
        compiler.predicate_to_row_restriction(pl.col("flag").is_in([True, False]))
        == "(`flag` IN (TRUE, FALSE))"
    )
    assert (
        compiler.predicate_to_row_restriction(
            pl.col("dt").is_in([date(2020, 1, 1), date(2020, 12, 31)])
        )
        == "(`dt` IN (DATE(TIMESTAMP_SECONDS(18262 * 86400)), DATE(TIMESTAMP_SECONDS(18627 * 86400))))"
    )
    assert (
        compiler.predicate_to_row_restriction(
            pl.col("ts").is_in([datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)])
        )
        == "(`ts` IN (TIMESTAMP_MICROS(1704110400000000)))"
    )

    # Edge cases that must not push down (empty list, nulls_equal=True, NaN in float list, column rhs)
    assert compiler.predicate_to_row_restriction(pl.col("a").is_in([])) == ""
    assert (
        compiler.predicate_to_row_restriction(
            pl.col("a").is_in([1, 2], nulls_equal=True)
        )
        == ""
    )
    assert (
        compiler.predicate_to_row_restriction(pl.col("a").is_in([1.0, float("nan")]))
        == ""
    )
    assert compiler.predicate_to_row_restriction(pl.col("a").is_in(pl.col("b"))) == ""


def test_json_path_parser_registration_and_dispatch() -> None:
    assert compiler.parser.parse_json_path("$.Column") == ((".", "Column"),)
    assert compiler.parser.parse_json_path("$.BinaryExpr.op.Eq") == (
        (".", "BinaryExpr"),
        (".", "op"),
        (".", "Eq"),
    )
    assert compiler.parser.parse_json_path("$.Literal..Int64") == (
        (".", "Literal"),
        ("..", "Int64"),
    )

    with pytest.raises(ValueError, match="Invalid JSON path"):
        compiler.parser.parse_json_path("BinaryExpr.op.Eq")
    with pytest.raises(ValueError, match="Invalid JSON path"):
        compiler.parser.parse_json_path("$.BinaryExpr.")
    with pytest.raises(
        ValueError, match="register_parser requires at least one JSON Path string"
    ):
        compiler.parser.register_parser()

    expected_paths = {
        "$.Column",
        "$.Literal..Null",
        "$.Literal..Boolean",
        "$.BinaryExpr.op.And",
        "$.BinaryExpr.op.Or",
        "$.BinaryExpr.op.Eq",
        "$.BinaryExpr.op.NotEq",
        "$.BinaryExpr.op.Gt",
        "$.BinaryExpr.op.GtEq",
        "$.BinaryExpr.op.Lt",
        "$.BinaryExpr.op.LtEq",
        "$.Function.function.Boolean.Not",
        "$.Function.function.Boolean.IsNull",
        "$.Function.function.Boolean.IsNotNull",
        "$.Function.function.Boolean.IsNan",
        "$.Function.function.Boolean.IsNotNan",
        "$.Function.function.Boolean.IsInfinite",
        "$.Function.function.Boolean.IsFinite",
        "$.Function.function.Boolean.IsIn",
        "$.Function.function.StringExpr.Uppercase",
        "$.Function.function.StringExpr.Lowercase",
        "$.Function.function.StringExpr.StartsWith",
        "$.Function.function.StringExpr.EndsWith",
        "$.Function.function.StringExpr.Contains",
        "$.Literal..Int64",
        "$.Literal..Float64",
        "$.Literal..String",
        "$.Literal..Date",
        "$.Literal..DateTime",
        "$.Literal..List",
    }
    assert expected_paths.issubset(compiler.parser.PARSERS.keys())

    # Unmatched JSON paths dispatch to an Unsupported IR leaf record
    assert compiler.parser.dispatch_parser({"UnknownNode": 123}) == (
        "Leaf",
        compiler.ir.Unsupported(),
        (),
    )
    assert compiler.json_to_ir({"UnknownNode": 123}) == compiler.ir.Unsupported()

    # Duplicate path registration raises ValueError
    with pytest.raises(ValueError, match="Duplicate parser registration"):
        compiler.parser.register_parser("$.Column")(lambda x: compiler.ir.Unsupported())

    # Strict type boundary checks (preventing str()/bool()/int(bool) coercion bugs)
    assert compiler.parser.base.parse_column_expr(None) == (
        "Leaf",
        compiler.ir.Unsupported(),
        (),
    )
    assert compiler.parser.base.parse_column_expr({"name": "a"}) == (
        "Leaf",
        compiler.ir.Unsupported(),
        (),
    )
    assert (
        compiler.parser.boolean.parse_bool_literal("false") == compiler.ir.Unsupported()
    )
    assert compiler.parser.numeric.parse_int_literal(True) == compiler.ir.Unsupported()
    assert (
        compiler.parser.numeric.parse_float_literal(False) == compiler.ir.Unsupported()
    )
    assert (
        compiler.parser.temporal.parse_date_literal(True) == compiler.ir.Unsupported()
    )
    assert (
        compiler.parser.string.parse_string_literal(None) == compiler.ir.Unsupported()
    )

    # Unit-string Null variant and multi-arg unary function rejection
    assert compiler.json_to_ir({"Literal": "Null"}) == compiler.ir.NullLiteral()
    assert compiler.parser.boolean.parse_not(
        "Not",
        {"input": [{"Column": "a"}, {"Column": "b"}], "function": {"Boolean": "Not"}},
    ) == ("Leaf", compiler.ir.Unsupported(), ())


def test_json_to_ir_partial_ast_degradation_preserves_shallow_and_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(compiler.parser, "MAX_AST_NODES", 7)
    oversized_and_json = {
        "BinaryExpr": {
            "left": {
                "BinaryExpr": {
                    "left": {"Column": "_PARTITIONDATE"},
                    "op": "Eq",
                    "right": {"Literal": {"Date": 19723}},
                }
            },
            "op": "And",
            "right": {
                "BinaryExpr": {
                    "left": {
                        "BinaryExpr": {
                            "left": {"Column": "x"},
                            "op": "Eq",
                            "right": {"Literal": {"Int64": 1}},
                        }
                    },
                    "op": "Or",
                    "right": {
                        "BinaryExpr": {
                            "left": {"Column": "y"},
                            "op": "Eq",
                            "right": {"Literal": {"Int64": 2}},
                        }
                    },
                }
            },
        }
    }

    # Act
    ir_tree = compiler.json_to_ir(oversized_and_json)
    sql = compiler.ir_to_sql(ir_tree)

    # Assert
    assert sql == "(`_PARTITIONDATE` = DATE(TIMESTAMP_SECONDS(19723 * 86400)))"


def test_parse_int_and_date_literals_reject_float_truncation() -> None:
    # Arrange
    float_int_input = 3.9
    float_date_input = 10.5
    float_ticks_input = 100.5

    # Act
    int_ir = compiler.parser.numeric.parse_int_literal(float_int_input)
    date_ir = compiler.parser.temporal.parse_date_literal(float_date_input)

    # Assert
    assert int_ir == compiler.ir.Unsupported()
    assert date_ir == compiler.ir.Unsupported()
    with pytest.raises(TypeError, match="Invalid timestamp tick count type"):
        compiler.parser.temporal.parse_ticks_and_unit(float_ticks_input, "Microseconds")


def test_parse_int_literal_enforces_bigquery_int64_bounds() -> None:
    # Arrange
    max_int64 = (1 << 63) - 1
    overflow_uint64 = 1 << 63
    underflow_int128 = -(1 << 63) - 1
    overflow_series_expr = pl.col("id").is_in(
        pl.Series([1, overflow_uint64], dtype=pl.UInt64)
    )

    # Act
    valid_max_ir = compiler.parser.numeric.parse_int_literal(max_int64)
    overflow_ir = compiler.parser.numeric.parse_int_literal(overflow_uint64)
    underflow_ir = compiler.parser.numeric.parse_int_literal(underflow_int128)
    series_sql = compiler.predicate_to_row_restriction(overflow_series_expr)

    # Assert
    assert valid_max_ir == compiler.ir.IntLiteral(max_int64)
    assert overflow_ir == compiler.ir.Unsupported()
    assert underflow_ir == compiler.ir.Unsupported()
    assert series_sql == ""


def test_parse_temporal_literals_enforce_bigquery_date_and_timestamp_bounds() -> None:
    # Arrange
    ipc_buf = io.BytesIO()
    pl.Series("d", [-800_000], dtype=pl.Int32).cast(
        pl.Date
    ).to_frame().write_ipc_stream(ipc_buf)
    out_of_range_date_ipc = ipc_buf.getvalue()

    # Act
    scalar_date_ir = compiler.parser.temporal.parse_date_literal(3_000_000)
    ipc_date_ir = compiler.parser.list_.parse_ipc_series_to_list_literal(
        out_of_range_date_ipc
    )

    # Assert
    assert scalar_date_ir == compiler.ir.Unsupported()
    assert ipc_date_ir == compiler.ir.Unsupported()
    with pytest.raises(
        ValueError, match="Timestamp microseconds out of BigQuery bounds"
    ):
        compiler.parser.temporal.parse_ticks_and_unit(
            300_000_000_000_000_000, "Microseconds"
        )
    with pytest.raises(
        ValueError, match="Timestamp milliseconds out of BigQuery bounds"
    ):
        compiler.parser.temporal.parse_ticks_and_unit(
            300_000_000_000_000, "Milliseconds"
        )
    with pytest.raises(
        ValueError, match="Timestamp nanoseconds out of BigQuery bounds"
    ):
        compiler.parser.temporal.parse_ticks_and_unit(
            300_000_000_000_000_000_000, "Nanoseconds"
        )


def test_parse_is_in_and_contains_validate_unit_and_boolean_options() -> None:
    # Arrange
    binary_inputs = [{"Column": "a"}, {"Literal": {"List": [{"Int64": 1}]}}]
    string_inputs = [{"Column": "a"}, {"Literal": {"String": "x"}}]

    # Act
    unit_isin = compiler.parser.list_.parse_is_in_function("IsIn", binary_inputs)
    bool_false_isin = compiler.parser.list_.parse_is_in_function(False, binary_inputs)
    bool_true_isin = compiler.parser.list_.parse_is_in_function(True, binary_inputs)
    invalid_spec_isin = compiler.parser.list_.parse_is_in_function(
        "InvalidSpec", binary_inputs
    )
    non_bool_opt_isin = compiler.parser.list_.parse_is_in_function(
        {"nulls_equal": "false"}, binary_inputs
    )
    non_bool_contains = compiler.parser.string.parse_contains(
        {"literal": "false"}, string_inputs
    )
    invalid_spec_contains = compiler.parser.string.parse_contains(
        "InvalidContainsSpec", string_inputs
    )
    wrong_arity_contains = compiler.parser.string.parse_contains(
        {"literal": True}, [{"Column": "a"}]
    )

    # Assert
    assert isinstance(unit_isin, compiler.parser.base.ParseRecord)
    assert unit_isin.kind == "Binary"
    assert bool_false_isin.kind == "Binary"
    assert bool_true_isin == compiler.parser.base.UNSUPPORTED_RECORD
    assert invalid_spec_isin == compiler.parser.base.UNSUPPORTED_RECORD
    assert non_bool_opt_isin == compiler.parser.base.UNSUPPORTED_RECORD
    assert non_bool_contains == compiler.parser.base.UNSUPPORTED_RECORD
    assert invalid_spec_contains == compiler.parser.base.UNSUPPORTED_RECORD
    assert wrong_arity_contains == compiler.parser.base.UNSUPPORTED_RECORD


def test_trie_wrapper_unwrapping_enforces_unified_depth_and_record_contract() -> None:
    # Arrange
    wrapped_literal = {"Dyn": {"Scalar": {"dtype": "Int64", "value": {"Int64": 7}}}}
    deep_wrapped: dict[str, Any] = {"Int64": 1}
    for _ in range(compiler.parser.base.MAX_TRIE_DEPTH + 2):
        deep_wrapped = {"Dyn": {"CustomWrapper": deep_wrapped}}
    bad_node = compiler.parser.base._PathTrieNode(parser=lambda _x: "not-a-record")  # type: ignore[arg-type,return-value]

    # Act
    unwrapped = compiler.parser.base.unwrap_literal_json(wrapped_literal)
    deep_ir = compiler.json_to_ir({"Literal": deep_wrapped})
    unknown_unit_ir = compiler.json_to_ir({"Literal": "UnknownUnitLiteral"})
    fallback_record = compiler.parser.base._invoke_matched_parser(bad_node, 1, {})

    # Assert
    assert unwrapped == {"Int64": 7}
    assert deep_ir == compiler.ir.Unsupported()
    assert unknown_unit_ir == compiler.ir.Unsupported()
    assert fallback_record == compiler.parser.base.UNSUPPORTED_RECORD
