"""Test coverage for `bigquery` expressions comprehension."""

from __future__ import annotations

import polars as pl
import pytest
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


def test_not_eq_and_boolean_literal() -> None:
    expr = (pl.col("status") != "DELETED") & (pl.col("active") == True)
    assert (
        predicates.predicate_to_row_restriction(expr)
        == "((`status` != 'DELETED') AND (`active` = TRUE))"
    )


def test_flexible_column_names_and_string_escaping() -> None:
    # Flexible column names (e.g. with apostrophe, spaces, hyphens, unicode) are
    # enclosed in backticks without string-style backslash escaping, whereas string
    # literals are single-quoted and escaped.
    expr = pl.col("user's-name 1") == "O'Reilly"
    assert (
        predicates.predicate_to_row_restriction(expr)
        == "(`user's-name 1` = 'O\\'Reilly')"
    )


def test_japanese_and_multilingual_column_names() -> None:
    # Japanese column names using Kanji, Hiragana, Katakana, and mixed scripts
    # are valid BigQuery flexible column names (\p{L}, \p{N}, \p{Pc}, \p{Pd}).
    expr = (pl.col("ユーザー名") == "田中") & (pl.col("価格_円") > 1000)
    assert (
        predicates.predicate_to_row_restriction(expr)
        == "((`ユーザー名` = '田中') AND (`価格_円` > 1000))"
    )

    # Test individual identifiers across Japanese and other languages with diacritics/marks
    assert (
        predicates._column_to_sql_identifier("ユーザー名") == "`ユーザー名`"
    )  # Katakana + Kanji
    assert predicates._column_to_sql_identifier("なまえ") == "`なまえ`"  # Hiragana
    assert (
        predicates._column_to_sql_identifier("顧客コード_テスト#1")
        == "`顧客コード_テスト#1`"
    )
    assert predicates._column_to_sql_identifier("注文-番号:2026") == "`注文-番号:2026`"
    assert (
        predicates._column_to_sql_identifier("café_au_lait") == "`café_au_lait`"
    )  # French (\p{M})
    assert (
        predicates._column_to_sql_identifier("Größe_in_cm") == "`Größe_in_cm`"
    )  # German (\p{M})
    assert (
        predicates._column_to_sql_identifier("año_fiscal") == "`año_fiscal`"
    )  # Spanish (\p{M})


def test_unsupported_column_names_raise_value_error() -> None:
    # Column names containing characters disallowed by BigQuery (such as backtick,
    # backslash, dot, dollar, exclamation mark, or >300 chars) raise ValueError.
    for invalid_col in ("bad`col", "bad\\col", "bad.col", "bad$col", "", "a" * 301):
        expr = pl.col(invalid_col) == 1
        with pytest.raises(ValueError, match="Invalid BigQuery column name"):
            predicates.predicate_to_row_restriction(expr)


def test_nan_and_inf_floats_cast_to_float64() -> None:
    # Polars expressions with float("nan") or float("inf") serialize with Float: null,
    # which cannot safely distinguish nan vs inf vs -inf. They safely fallback to in-memory filtering.
    expr_nan = pl.col("val") == float("nan")  # noqa: PLW0177
    assert predicates.predicate_to_row_restriction(expr_nan) == ""

    expr_inf = pl.col("val") < float("inf")
    assert predicates.predicate_to_row_restriction(expr_inf) == ""

    expr_neg_inf = pl.col("val") > float("-inf")
    assert predicates.predicate_to_row_restriction(expr_neg_inf) == ""

    # When JSON literals contain nan, inf, and -inf, they are correctly cast to FLOAT64
    assert (
        predicates._json_literal_to_sql({"Float64": "nan"}) == "CAST('nan' AS FLOAT64)"
    )
    assert (
        predicates._json_literal_to_sql({"Float64": "inf"}) == "CAST('inf' AS FLOAT64)"
    )
    assert (
        predicates._json_literal_to_sql({"Float64": "-inf"})
        == "CAST('-inf' AS FLOAT64)"
    )
    assert (
        predicates._json_literal_to_sql({"Float64": "infinity"})
        == "CAST('inf' AS FLOAT64)"
    )
    assert (
        predicates._json_literal_to_sql({"Float64": "-infinity"})
        == "CAST('-inf' AS FLOAT64)"
    )

    # Boolean functions for NaN and Infinity
    assert (
        predicates.predicate_to_row_restriction(pl.col("val").is_nan())
        == "IS_NAN(`val`)"
    )
    assert (
        predicates.predicate_to_row_restriction(pl.col("val").is_not_nan())
        == "(NOT IS_NAN(`val`))"
    )
    assert (
        predicates.predicate_to_row_restriction(pl.col("val").is_infinite())
        == "IS_INF(`val`)"
    )
    assert (
        predicates.predicate_to_row_restriction(pl.col("val").is_finite())
        == "(NOT IS_INF(`val`) AND NOT IS_NAN(`val`))"
    )


def test_escape_sql_and_column_identifier_helpers() -> None:
    assert predicates._escape_sql_string("a\\b'c\nd\re") == "'a\\\\b\\'c\\nd\\re'"

    # Test Unicode category validation helper
    assert predicates._is_allowed_unicode_category("a")  # \p{L} Letter
    assert predicates._is_allowed_unicode_category("1")  # \p{N} Number
    assert predicates._is_allowed_unicode_category("_")  # \p{Pc} Connector punctuation
    assert predicates._is_allowed_unicode_category("-")  # \p{Pd} Dash punctuation
    assert predicates._is_allowed_unicode_category("é")  # Letter / Mark
    assert not predicates._is_allowed_unicode_category("$")  # Currency symbol
    assert not predicates._is_allowed_unicode_category("@")  # Other punctuation

    # Test column character validation helper
    assert predicates._is_valid_column_character("a")
    assert predicates._is_valid_column_character(" ")
    assert predicates._is_valid_column_character("&")
    assert not predicates._is_valid_column_character("`")
    assert not predicates._is_valid_column_character("\\")
    assert not predicates._is_valid_column_character("\n")

    assert predicates._column_to_sql_identifier("valid_col_1") == "`valid_col_1`"
    assert (
        predicates._column_to_sql_identifier("café & tea: 100%") == "`café & tea: 100%`"
    )
    with pytest.raises(ValueError, match="Invalid BigQuery column name"):
        predicates._column_to_sql_identifier("bad`col")
    with pytest.raises(ValueError, match="Invalid BigQuery column name"):
        predicates._column_to_sql_identifier("bad\\col")


def test_json_literal_to_sql_all_scalar_types_and_immutability() -> None:
    # Verify input dictionary is not mutated by _json_literal_to_sql
    lit_dict = {"Int64": 42}
    assert predicates._json_literal_to_sql(lit_dict) == "42"
    assert lit_dict == {"Int64": 42}

    # Empty or non-dict inputs
    assert predicates._json_literal_to_sql({}) is None
    assert predicates._json_literal_to_sql(None) is None  # type: ignore[arg-type]

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
        assert predicates._json_literal_to_sql({int_type: 99}) == "99"

    # Float scalar variants including nan, inf, and -inf
    for float_type in ("Float", "Float32", "Float64"):
        assert predicates._json_literal_to_sql({float_type: 3.5}) == "3.5"
        assert (
            predicates._json_literal_to_sql({float_type: float("inf")})
            == "CAST('inf' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: float("-inf")})
            == "CAST('-inf' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: float("nan")})
            == "CAST('nan' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "nan"})
            == "CAST('nan' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "+nan"})
            == "CAST('nan' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "-nan"})
            == "CAST('nan' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "inf"})
            == "CAST('inf' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "+inf"})
            == "CAST('inf' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "infinity"})
            == "CAST('inf' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "+infinity"})
            == "CAST('inf' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "-inf"})
            == "CAST('-inf' AS FLOAT64)"
        )
        assert (
            predicates._json_literal_to_sql({float_type: "-infinity"})
            == "CAST('-inf' AS FLOAT64)"
        )
        assert predicates._json_literal_to_sql({float_type: None}) is None

    # String variants
    assert predicates._json_literal_to_sql({"String": "hello"}) == "'hello'"
    assert predicates._json_literal_to_sql({"StringOwned": "world"}) == "'world'"

    # Boolean and Null
    assert predicates._json_literal_to_sql({"Boolean": True}) == "TRUE"
    assert predicates._json_literal_to_sql({"Boolean": False}) == "FALSE"
    assert predicates._json_literal_to_sql({"Null": None}) == "NULL"

    # DateTime units: Microseconds, Milliseconds, Nanoseconds (exact and inexact)
    assert (
        predicates._json_literal_to_sql({"DateTime": [1000, "Microseconds"]})
        == "TIMESTAMP_MICROS(1000)"
    )
    assert (
        predicates._json_literal_to_sql({"DateTime": [2000, "Milliseconds"]})
        == "TIMESTAMP_MILLIS(2000)"
    )
    assert (
        predicates._json_literal_to_sql({"DateTime": [5000, "Nanoseconds"]})
        == "TIMESTAMP_MICROS(5)"
    )
    assert predicates._json_literal_to_sql({"DateTime": [5555, "Nanoseconds"]}) is None
    assert predicates._json_literal_to_sql({"DateTime": [10, "Seconds"]}) is None

    # Malformed literal values catch ValueError/TypeError locally and return None
    assert predicates._json_literal_to_sql({"Int64": "not-an-int"}) is None
    assert predicates._json_literal_to_sql({"Float64": "not-a-float"}) is None
    assert (
        predicates._json_literal_to_sql({"DateTime": ["not-an-int", "Microseconds"]})
        is None
    )
    assert predicates._json_literal_to_sql({"DateTime": []}) is None
    assert predicates._json_literal_to_sql({"Date": "not-a-date"}) is None


def test_and_expression_preserves_valid_branch_with_unsupported_branch() -> None:
    expr = (pl.col("keep_col") == 1) & (pl.col("other_col").sin() > 0.5)
    assert predicates.predicate_to_row_restriction(expr) == "(`keep_col` = 1)"


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

    ir_tree = predicates.json_to_ir(curr)
    assert isinstance(ir_tree, predicates.ir.And)

    sql_result = predicates.ir_to_sql(ir_tree)
    assert sql_result is not None
    assert sql_result.startswith("(" * depth + "`x` = 0)")
    assert sql_result.endswith(f"AND (`x` = {depth - 1}))")


def test_ir_frozen_dataclasses_and_submodules() -> None:
    from dataclasses import FrozenInstanceError

    col_a = predicates.ir.base.Column("a")
    lit_10 = predicates.ir.numeric.IntLiteral(10)
    eq_node = predicates.ir.comparison.Eq(left=col_a, right=lit_10)

    # Verify frozen dataclass immutability
    with pytest.raises(FrozenInstanceError):
        col_a.name = "b"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        eq_node.left = col_a  # type: ignore[misc]

    # Verify submodule organization and direct IR -> SQL compilation
    str_starts = predicates.ir.string.StartsWith(
        left=predicates.ir.base.Column("name"),
        right=predicates.ir.string.StringLiteral("pre"),
    )
    assert str_starts.expr == predicates.ir.base.Column("name")
    assert str_starts.prefix == predicates.ir.string.StringLiteral("pre")

    date_cmp = predicates.ir.comparison.GtEq(
        left=predicates.ir.base.Column("dt"),
        right=predicates.ir.temporal.DateLiteral(days=10),
    )
    not_nan = predicates.ir.numeric.IsNotNan(expr=predicates.ir.base.Column("score"))

    combined = predicates.ir.boolean.And(
        left=predicates.ir.boolean.Or(left=eq_node, right=str_starts),
        right=predicates.ir.boolean.And(left=date_cmp, right=not_nan),
    )

    expected_sql = (
        "(((`a` = 10) OR STARTS_WITH(`name`, 'pre')) AND "
        "((`dt` >= DATE(TIMESTAMP_SECONDS(10 * 86400))) AND (NOT IS_NAN(`score`))))"
    )
    assert predicates.ir_to_sql(combined) == expected_sql


def test_predicate_to_row_restriction_root_recursion_error_fallback(
    monkeypatch,
) -> None:
    def raise_recursion(_expr_json):
        raise RecursionError("Maximum recursion depth exceeded")

    monkeypatch.setattr(predicates, "_json_expr_to_row_restriction", raise_recursion)
    assert predicates.predicate_to_row_restriction(pl.col("a") == 1) == ""


def test_escape_sql_string_tabs_and_null_bytes() -> None:
    assert predicates._escape_sql_string("a\tb\0c") == "'a\\tb\\x00c'"


def test_json_to_ir_prunes_unsupported_subtrees() -> None:
    unsupported_json = {
        "BinaryExpr": {
            "left": {"Column": "a"},
            "op": "Plus",
            "right": {"Column": "b"},
        }
    }
    ir_node = predicates.json_to_ir(unsupported_json)
    assert isinstance(ir_node, predicates.ir.Unsupported)
    assert ir_node.children() == ()


def test_compile_predicate_compound_pseudo_column_and_physical_column() -> None:
    from datetime import date

    expr = (pl.col("_PARTITIONDATE") == date(2024, 1, 1)) & (pl.col("val").sin() > 0.5)
    compiled = predicates.compile_predicate(
        expr, pseudo_columns=("_PARTITIONDATE", "_PARTITIONTIME")
    )
    assert (
        compiled.row_restriction
        == "(`_PARTITIONDATE` = DATE(TIMESTAMP_SECONDS(19723 * 86400)))"
    )
    assert compiled.residual_predicate is not None
    df = pl.DataFrame({"val": [0.0, 1.0]})
    filtered = df.filter(compiled.residual_predicate)
    assert filtered["val"].to_list() == [1.0]


def test_compile_predicate_unpushable_pseudo_column_raises_bigquery_error() -> None:
    from datetime import date

    import polars_bigquery.exceptions

    expr_or = (pl.col("_PARTITIONDATE") == date(2024, 1, 1)) | (
        pl.col("val").sin() > 0.5
    )
    with pytest.raises(
        polars_bigquery.exceptions.BigQueryError,
        match="Predicate referencing BigQuery pseudo-column",
    ):
        predicates.compile_predicate(
            expr_or, pseudo_columns=("_PARTITIONDATE", "_PARTITIONTIME")
        )
