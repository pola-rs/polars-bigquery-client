"""Unit tests for `polars_bigquery.core.executors.read_api`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import arrow_bigquery
import polars as pl
import pytest
from polars_bigquery.core.executors import (
    ReadApiExecutor,
    compile_read_api_plan,
)
from polars_bigquery.core.plan import (
    Filter,
    Project,
    QueryScan,
    Slice,
    TableScan,
    decompose_linear_plan,
)


def _table_id() -> arrow_bigquery.BigQueryTableId:
    return arrow_bigquery.BigQueryTableId("proj", "ds", "tbl")


def _tscan(
    schema: dict[str, pl.DataType] | pl.Schema,
    *,
    maintain_order: bool = False,
) -> TableScan:
    return TableScan(
        table_ref=_table_id(),
        schema=pl.Schema(schema),
        maintain_order=maintain_order,
    )


def test_compile_read_api_plan_rejects_non_table_scan_leaf() -> None:
    qscan = QueryScan(raw_query="SELECT a FROM t", columns=("a",))
    with pytest.raises(TypeError, match="expects a TableScan leaf, got QueryScan"):
        compile_read_api_plan(qscan)


def test_compile_read_api_plan_without_project_uses_all_columns() -> None:
    scan = _tscan({"a": pl.Int64(), "b": pl.String()})
    compiled = compile_read_api_plan(scan)
    assert compiled.selected_fields == ("a", "b")
    assert compiled.row_restriction == ""
    assert compiled.residual_plan == scan


def test_compile_read_api_plan_visibility_and_multi_project_collapse() -> None:
    scan = _tscan({"a": pl.Int64(), "b": pl.String(), "c": pl.Float64()})

    # Invalid reference fails visibility check
    with pytest.raises(ValueError, match="'c'"):
        compile_read_api_plan(
            Filter(
                input=Project(input=scan, columns=("a", "b")),
                predicate=pl.col("c") > 0,
            )
        )

    # Multi-Project preserves dict.fromkeys([*output_cols, *filter_cols]) ordering
    # and collapses superseded intermediate Project node.
    plan = Project(
        input=Filter(
            input=Project(input=scan, columns=("b", "a")),
            predicate=pl.col("b") == "ok",
        ),
        columns=("a",),
    )
    compiled = compile_read_api_plan(plan)
    assert compiled.selected_fields == ("a", "b")
    assert compiled.row_restriction == "(`b` = 'ok')"
    _, ops = decompose_linear_plan(compiled.residual_plan)
    assert [type(op) for op in ops] == [Filter, Project]


def test_compile_read_api_plan_zero_column_fallback_priority_tiers() -> None:
    # Check each rank tier strictly preferred over all higher-rank tiers.
    nested_map = (
        pl.Map(pl.String(), pl.Int64())
        if hasattr(pl, "Map")
        else pl.Struct({"k": pl.String()})
    )
    ordered_candidates: list[tuple[str, pl.DataType]] = [
        ("tier0_bool", pl.Boolean()),
        ("tier1_i8", pl.Int8()),
        ("tier1_u8", pl.UInt8()),
        ("tier2_i16", pl.Int16()),
        ("tier2_u16", pl.UInt16()),
        ("tier2_f16", pl.Float16()),
        ("tier3_date", pl.Date()),
        ("tier3_i32", pl.Int32()),
        ("tier3_u32", pl.UInt32()),
        ("tier3_f32", pl.Float32()),
        ("tier4_i64", pl.Int64()),
        ("tier4_u64", pl.UInt64()),
        ("tier4_f64", pl.Float64()),
        ("tier4_time", pl.Time()),
        ("tier4_dt", pl.Datetime("us")),
        ("tier5_i128", pl.Int128()),
        ("tier5_u128", pl.UInt128()),
        ("tier5_dec38", pl.Decimal(precision=38, scale=9)),
        ("tier6_str", pl.String()),
        ("tier6_bin", pl.Binary()),
        ("tier6_cat", pl.Categorical()),
        ("tier6_enum", pl.Enum(["x"])),
        ("tier7_list", pl.List(pl.Int64())),
        ("tier7_arr", pl.Array(pl.Int64(), 2)),
        ("tier7_struct", pl.Struct({"x": pl.Int64()})),
        ("tier7_map", nested_map),
        ("tier8_dur", pl.Duration("us")),
        ("tier8_bignum", pl.Decimal(precision=76, scale=38)),
        ("tier8_null", pl.Null()),
        ("tier8_unknown", pl.Unknown()),
        ("tier8_obj", pl.Object()),
    ]
    # Add in reverse order so the narrowest type wins by rank, not position.
    for i in range(len(ordered_candidates)):
        slice_map = dict(reversed(ordered_candidates[i:]))
        # Include ingestion-time pseudo-columns to confirm physical column wins.
        slice_map["_PARTITIONDATE"] = pl.Date()
        slice_map["_PARTITIONTIME"] = pl.Datetime("us", "utc")
        scan = _tscan(slice_map)
        compiled = compile_read_api_plan(Project(input=scan, columns=()))
        assert len(compiled.selected_fields) == 1
        winner = compiled.selected_fields[0]
        assert winner not in {"_PARTITIONDATE", "_PARTITIONTIME"}
        # Verify strictly matching rank tier of expected front candidate
        expected_name = ordered_candidates[i][0]
        expected_tier = expected_name.split("_")[0]
        assert winner.startswith(expected_tier), (
            f"For slice starting at {expected_name}, got {winner}"
        )

    # Tie breaking within the same tier picks earliest schema index.
    tie_scan = _tscan({"second_tier4": pl.Float64(), "first_i64": pl.Int64()})
    assert compile_read_api_plan(
        Project(input=tie_scan, columns=())
    ).selected_fields == ("second_tier4",)

    # Pseudo-columns only: prefer _PARTITIONTIME over _PARTITIONDATE
    pseudo_both = _tscan(
        {"_PARTITIONDATE": pl.Date(), "_PARTITIONTIME": pl.Datetime("us", "utc")}
    )
    assert compile_read_api_plan(
        Project(input=pseudo_both, columns=())
    ).selected_fields == ("_PARTITIONTIME",)

    # Only _PARTITIONDATE present
    pseudo_date_only = _tscan({"_PARTITIONDATE": pl.Date()})
    assert compile_read_api_plan(
        Project(input=pseudo_date_only, columns=())
    ).selected_fields == ("_PARTITIONDATE",)


def test_compile_read_api_plan_zero_column_with_filter_selects_filter_col() -> None:
    scan = _tscan({"flag": pl.Boolean(), "name": pl.String()})
    plan = Project(
        input=Filter(input=scan, predicate=pl.col("name") == "Alice"),
        columns=(),
    )
    compiled = compile_read_api_plan(plan)
    assert compiled.selected_fields == ("name",)
    assert compiled.row_restriction == "(`name` = 'Alice')"


def test_compile_read_api_plan_multiple_filters_and_true_omission() -> None:
    scan = _tscan({"a": pl.Int64(), "b": pl.Int64(), "c": pl.Int64()})
    plan = Filter(
        input=Filter(
            input=Filter(
                input=Filter(input=scan, predicate=pl.col("a") == 1),
                predicate=pl.lit(True),
            ),
            predicate=pl.col("b") == 2,
        ),
        predicate=pl.col("c") == 3,
    )
    compiled = compile_read_api_plan(plan)
    assert compiled.row_restriction == "((`a` = 1) AND (`b` = 2) AND (`c` = 3))"


def test_compile_read_api_plan_slice_barriers() -> None:
    # Unordered Slice(0, n) allows filter pushdown across the slice
    unordered = _tscan({"a": pl.Int64(), "b": pl.Int64()}, maintain_order=False)
    c_u = compile_read_api_plan(
        Filter(
            input=Slice(input=unordered, offset=0, length=10),
            predicate=pl.col("a") == 1,
        )
    )
    assert c_u.row_restriction == "(`a` = 1)"

    # Unordered Slice(offset > 0, n) stops filter pushdown
    c_off = compile_read_api_plan(
        Filter(
            input=Slice(
                input=Filter(input=unordered, predicate=pl.col("a") == 1),
                offset=5,
                length=10,
            ),
            predicate=pl.col("b") == 2,
        )
    )
    assert c_off.row_restriction == "(`a` = 1)"
    assert c_off.selected_fields == ("a", "b")

    # Ordered Slice(0, n) stops filter pushdown (maintain_order=True)
    ordered = _tscan({"a": pl.Int64(), "b": pl.Int64()}, maintain_order=True)
    c_ord = compile_read_api_plan(
        Filter(
            input=Slice(
                input=Filter(input=ordered, predicate=pl.col("a") == 1),
                offset=0,
                length=10,
            ),
            predicate=pl.col("b") == 2,
        )
    )
    assert c_ord.row_restriction == "(`a` = 1)"


def test_read_api_executor_execute() -> None:
    mock_arrow_client = MagicMock()
    mock_arrow_client.read_table.return_value = MagicMock(name="exporter")
    source_lf = pl.LazyFrame({"visible": ["a", "b", "c"], "hidden": [5, 15, 25]})

    scan = _tscan({"visible": pl.String(), "hidden": pl.Int64(), "unused": pl.String()})
    plan = Project(
        input=Filter(
            input=Slice(input=scan, offset=0, length=2),
            predicate=pl.col("hidden") > 10,
        ),
        columns=("visible",),
    )
    executor = ReadApiExecutor(mock_arrow_client)
    with patch("polars.scan_arrow_c_stream", return_value=source_lf):
        batches = list(executor.execute(plan, batch_size=1))

    mock_arrow_client.read_table.assert_called_once_with(
        _table_id(),
        maintain_order=False,
        selected_fields=["visible", "hidden"],
        row_restriction="(`hidden` > 10)",
    )
    assert pl.concat(batches).to_dict(as_series=False) == {"visible": ["b"]}


def test_compile_read_api_plan_case_insensitive_pseudo_column_fallback() -> None:
    scan = _tscan({"_partitiondate": pl.Date(), "note": pl.String()})
    compiled = compile_read_api_plan(Project(input=scan, columns=()))
    assert compiled.selected_fields == ("note",)

    pseudo_lower = _tscan(
        {"_partitiondate": pl.Date(), "_PartitionTime": pl.Datetime("us", "utc")}
    )
    compiled_pseudo = compile_read_api_plan(Project(input=pseudo_lower, columns=()))
    assert compiled_pseudo.selected_fields == ("_PartitionTime",)


def test_compile_read_api_plan_row_restriction_length_guard(caplog) -> None:
    import logging

    scan = _tscan({"a": pl.Int64(), "b": pl.String()})
    pred = (pl.col("a") == 1) & (pl.col("b") == "x" * 60)

    # Validates max_row_restriction_bytes argument.
    with pytest.raises(TypeError, match="positive int"):
        compile_read_api_plan(scan, max_row_restriction_bytes=True)
    with pytest.raises(ValueError, match="must be positive"):
        compile_read_api_plan(scan, max_row_restriction_bytes=0)
    with pytest.raises(ValueError, match="must be positive"):
        ReadApiExecutor(MagicMock(), max_row_restriction_bytes=-5)

    # Prunes largest conjunct while preserving smaller leftmost filter.
    plan = Filter(input=scan, predicate=pred)
    pruned = compile_read_api_plan(plan, max_row_restriction_bytes=20)
    assert pruned.row_restriction == "(`a` = 1)"
    assert pruned.degraded_reason == "row_restriction_length_pruned_conjuncts"

    # Drops all conjuncts when not even the smallest fits.
    dropped = compile_read_api_plan(plan, max_row_restriction_bytes=4)
    assert dropped.row_restriction == ""
    assert dropped.degraded_reason == "row_restriction_length_dropped_all"

    # Executor emits single PII-safe WARNING on degradation and DEBUG summary.
    mock_arrow_client = MagicMock()
    mock_arrow_client.read_table.return_value = MagicMock(name="exporter")
    executor = ReadApiExecutor(mock_arrow_client, max_row_restriction_bytes=20)
    with (
        caplog.at_level(
            logging.DEBUG, logger="polars_bigquery.core.executors.read_api"
        ),
        patch(
            "polars.scan_arrow_c_stream",
            return_value=pl.LazyFrame({"a": [1], "b": ["secret"]}),
        ),
    ):
        list(executor.execute(plan))
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "row_restriction_length_pruned_conjuncts" in warnings[0]
    assert "xxx" not in warnings[0]
    debugs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debugs) == 1
    assert "selected_fields=2/2" in debugs[0]
