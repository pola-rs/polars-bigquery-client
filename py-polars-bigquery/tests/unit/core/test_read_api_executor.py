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
    PlanNode,
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
    # Check each BigQuery width tier (via core.types.column_width_rank) is strictly
    # preferred over all higher-rank tiers: 1=BOOL, 2=DATE, 3=8-byte scalars,
    # 4=NUMERIC, 5=STRING/BYTES/JSON/GEOGRAPHY, 6=ARRAY/STRUCT, 7=unknown/unsupported.
    unparam_decimal = pl.Decimal(precision=38, scale=9)
    unparam_decimal.precision = None  # type: ignore[misc]
    ordered_candidates: list[tuple[str, pl.DataType]] = [
        ("tier1_bool", pl.Boolean()),
        ("tier2_date", pl.Date()),
        ("tier3_i64", pl.Int64()),
        ("tier3_i32", pl.Int32()),
        ("tier3_f64", pl.Float64()),
        ("tier3_time", pl.Time()),
        ("tier3_dt", pl.Datetime("us")),
        ("tier3_ts", pl.Datetime("us", "UTC")),
        ("tier4_dec38", pl.Decimal(precision=38, scale=9)),
        ("tier5_str", pl.String()),
        ("tier5_bin", pl.Binary()),
        ("tier5_json", pl.Extension("google:sqlType:json", pl.String())),
        (
            "tier5_geo",
            pl.Extension(
                "google:sqlType:geography", pl.String(), '{"encoding": "WKT"}'
            ),
        ),
        ("tier6_list", pl.List(pl.Int64())),
        ("tier6_struct", pl.Struct({"x": pl.Int64()})),
        (
            "tier6_json_arr",
            pl.Extension("google:sqlType:json", pl.List(pl.String())),
        ),
        ("tier7_dur", pl.Duration("us")),
        ("tier7_f16", pl.Float16()),
        ("tier7_bignum", pl.Decimal(precision=76, scale=38)),
        ("tier7_dec_none", unparam_decimal),
        ("tier7_null", pl.Null()),
        ("tier7_unknown", pl.Unknown()),
        ("tier7_obj", pl.Object()),
    ]
    # Add in reverse order so the narrowest type wins by rank, not position.
    for i in range(len(ordered_candidates)):
        expected_name = ordered_candidates[i][0]
        expected_tier = expected_name.split("_")[0]
        slice_map = dict(reversed(ordered_candidates[i:]))
        if expected_tier == "tier7":
            # Without pseudo-columns, rank-7 physical columns tie-break by schema order.
            scan = _tscan(slice_map)
            compiled = compile_read_api_plan(Project(input=scan, columns=()))
            assert compiled.selected_fields == (next(iter(slice_map)),)
            # With valid ingestion-time pseudo-columns, _PARTITIONDATE (rank 2)
            # takes precedence over rank-7 unsupported physical columns.
            slice_with_pseudo = {
                **slice_map,
                "_PARTITIONDATE": pl.Date(),
                "_PARTITIONTIME": pl.Datetime("us", "UTC"),
            }
            compiled_pseudo = compile_read_api_plan(
                Project(input=_tscan(slice_with_pseudo), columns=())
            )
            assert compiled_pseudo.selected_fields == ("_PARTITIONDATE",)
            continue

        # Supported physical columns (tiers 1–6) take priority over pseudo-columns.
        slice_map["_PARTITIONDATE"] = pl.Date()
        slice_map["_PARTITIONTIME"] = pl.Datetime("us", "UTC")
        scan = _tscan(slice_map)
        compiled = compile_read_api_plan(Project(input=scan, columns=()))
        assert len(compiled.selected_fields) == 1
        winner = compiled.selected_fields[0]
        assert winner not in {"_PARTITIONDATE", "_PARTITIONTIME"}
        # Verify strictly matching rank tier of expected front candidate
        assert winner.startswith(expected_tier), (
            f"For slice starting at {expected_name}, got {winner}"
        )

    # Unsupported physical column + _PARTITIONTIME (non-DAY) in both orders picks _PARTITIONTIME.
    for schema_map in (
        {"bad_dur": pl.Duration("us"), "_PARTITIONTIME": pl.Datetime("us", "UTC")},
        {"_PARTITIONTIME": pl.Datetime("us", "UTC"), "bad_dur": pl.Duration("us")},
    ):
        assert compile_read_api_plan(
            Project(input=_tscan(schema_map), columns=())
        ).selected_fields == ("_PARTITIONTIME",)

    # Invalid BigQuery extension storage (rank 7) never wins ahead of String or pseudo-columns.
    bad_ext = pl.Extension("google:sqlType:json", pl.Boolean())
    assert compile_read_api_plan(
        Project(input=_tscan({"bad_json": bad_ext, "note": pl.String()}), columns=())
    ).selected_fields == ("note",)
    assert compile_read_api_plan(
        Project(
            input=_tscan({"bad_json": bad_ext, "_PARTITIONDATE": pl.Date()}),
            columns=(),
        )
    ).selected_fields == ("_PARTITIONDATE",)

    # If both physical and pseudo-columns are rank 7, physical column wins.
    assert compile_read_api_plan(
        Project(
            input=_tscan(
                {"bad_dur": pl.Duration("us"), "_PARTITIONTIME": pl.Duration("us")}
            ),
            columns=(),
        )
    ).selected_fields == ("bad_dur",)

    # Custom non-BigQuery BaseExtension unwraps its storage dtype for ranking.
    ext_scan = _tscan(
        {
            "note": pl.String(),
            "flag_ext": pl.Extension("custom:flag", pl.Boolean()),
        }
    )
    assert compile_read_api_plan(
        Project(input=ext_scan, columns=())
    ).selected_fields == ("flag_ext",)

    # Tie breaking within the same tier picks earliest schema index.
    tie_scan = _tscan({"second_tier3": pl.Float64(), "first_i64": pl.Int64()})
    assert compile_read_api_plan(
        Project(input=tie_scan, columns=())
    ).selected_fields == ("second_tier3",)

    # Pseudo-columns only: _PARTITIONDATE (DATE, rank 2) wins over _PARTITIONTIME
    # (TIMESTAMP, rank 3) regardless of declaration order.
    pseudo_both = _tscan(
        {"_PARTITIONDATE": pl.Date(), "_PARTITIONTIME": pl.Datetime("us", "UTC")}
    )
    assert compile_read_api_plan(
        Project(input=pseudo_both, columns=())
    ).selected_fields == ("_PARTITIONDATE",)

    pseudo_reversed = _tscan(
        {"_PARTITIONTIME": pl.Datetime("us", "UTC"), "_PARTITIONDATE": pl.Date()}
    )
    assert compile_read_api_plan(
        Project(input=pseudo_reversed, columns=())
    ).selected_fields == ("_PARTITIONDATE",)

    # Non-DAY ingestion-time table: only _PARTITIONTIME present.
    pseudo_time_only = _tscan({"_PARTITIONTIME": pl.Datetime("us", "UTC")})
    assert compile_read_api_plan(
        Project(input=pseudo_time_only, columns=())
    ).selected_fields == ("_PARTITIONTIME",)

    # Only _PARTITIONDATE present.
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
        {"_PartitionTime": pl.Datetime("us", "UTC"), "_partitiondate": pl.Date()}
    )
    compiled_pseudo = compile_read_api_plan(Project(input=pseudo_lower, columns=()))
    assert compiled_pseudo.selected_fields == ("_partitiondate",)


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


def test_zero_column_fallback_deprioritizes_nested_unsupported_dtypes() -> None:
    scan = _tscan(
        {
            "arr_bn": pl.List(pl.Decimal(precision=76, scale=38)),
            "struct_bn": pl.Struct([pl.Field("x", pl.Decimal(precision=76, scale=38))]),
            "ext_bn": pl.Extension("custom:bn", pl.Decimal(precision=76, scale=38)),
            "arr_dur": pl.List(pl.Duration("us")),
            "struct_f16": pl.Struct([pl.Field("f", pl.Float16())]),
            "nested_ok": pl.List(pl.String()),
        }
    )
    compiled = compile_read_api_plan(Project(input=scan, columns=()))
    assert compiled.selected_fields == ("nested_ok",)


def test_zero_column_fallback_short_circuits_on_rank_one_bool() -> None:
    scan = _tscan(
        {
            "flag": pl.Boolean(),
            "later_str": pl.String(),
            "later_struct": pl.Struct([pl.Field("x", pl.Int64())]),
        }
    )
    with patch(
        "polars_bigquery.core.executors.read_api.column_width_rank",
        side_effect=[1],
    ) as mock_rank:
        compiled = compile_read_api_plan(Project(input=scan, columns=()))
    assert compiled.selected_fields == ("flag",)
    mock_rank.assert_called_once_with(pl.Boolean())


def test_zero_column_fallback_skips_nested_dfs_when_better_rank_known() -> None:
    scan = _tscan(
        {
            "dt": pl.Date(),
            "wide_list": pl.List(pl.Struct([pl.Field("a", pl.Int64())])),
            "wide_struct": pl.Struct([pl.Field("b", pl.String())]),
        }
    )
    with patch(
        "polars_bigquery.core.executors.read_api.column_width_rank",
        return_value=2,
    ) as mock_rank:
        compiled = compile_read_api_plan(Project(input=scan, columns=()))
    assert compiled.selected_fields == ("dt",)
    mock_rank.assert_called_once_with(pl.Date())


def test_zero_column_fallback_handles_decimal_none_precision() -> None:
    from polars_bigquery.core.types import column_width_rank

    dec_unspec = pl.Decimal(38, 2)
    dec_unspec.precision = None  # type: ignore[misc]
    assert column_width_rank(dec_unspec) == 7
    assert column_width_rank(pl.Decimal(38, 9)) == 4
    assert column_width_rank(pl.Decimal(76, 38)) == 7


def test_read_api_executor_max_stream_count_boundaries() -> None:
    from polars_bigquery.core.executors.read_api import _SINGLE_STREAM_ROW_LIMIT

    scan = _tscan({"id": pl.Int64()})
    cases: list[tuple[str, PlanNode, int | None]] = [
        ("zero_limit", Slice(input=scan, offset=0, length=0), 1),
        (
            "at_threshold",
            Slice(input=scan, offset=0, length=_SINGLE_STREAM_ROW_LIMIT),
            1,
        ),
        (
            "above_threshold",
            Slice(input=scan, offset=0, length=_SINGLE_STREAM_ROW_LIMIT + 1),
            None,
        ),
        ("nonzero_offset", Slice(input=scan, offset=100_000, length=10), None),
        (
            "nonzero_offset_then_small_slice",
            Slice(
                input=Slice(input=scan, offset=100_000, length=50_000),
                offset=0,
                length=10,
            ),
            None,
        ),
        (
            "large_leading_slice_then_small_slice",
            Slice(
                input=Slice(
                    input=scan, offset=0, length=_SINGLE_STREAM_ROW_LIMIT + 100
                ),
                offset=0,
                length=10,
            ),
            1,
        ),
    ]
    for label, plan, expected_max_streams in cases:
        mock_arrow_client = MagicMock()
        mock_arrow_client.read_table.return_value = MagicMock(name="exporter")
        executor = ReadApiExecutor(mock_arrow_client)
        with patch(
            "polars.scan_arrow_c_stream",
            return_value=pl.LazyFrame({"id": [1, 2, 3]}),
        ):
            list(executor.execute(plan))
        kwargs = mock_arrow_client.read_table.call_args.kwargs
        if expected_max_streams is None:
            assert "max_stream_count" not in kwargs, f"unexpected cap for {label}"
        else:
            assert kwargs.get("max_stream_count") == expected_max_streams, (
                f"wrong cap for {label}"
            )
