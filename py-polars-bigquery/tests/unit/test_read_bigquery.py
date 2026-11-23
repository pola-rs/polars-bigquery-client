import gc
import logging
import weakref
from datetime import date
from unittest.mock import MagicMock, patch

import arrow_bigquery
import polars as pl
import pytest
from polars_bigquery import (
    Client,
    __version__,
)
from polars_bigquery._read_bigquery import _get_user_agent
from polars_bigquery.core import bigquery_rest
from polars_bigquery.core.executors import query as query_executor_mod
from polars_bigquery.exceptions import BigQueryError


@pytest.fixture
def mock_arrow_client():
    with patch("arrow_bigquery.Client") as mocked_cls:
        mock_instance = MagicMock()
        mocked_cls.return_value = mock_instance
        yield mock_instance


def test_get_user_agent():
    assert _get_user_agent(None) == f"polars-bigquery/{__version__}"
    assert _get_user_agent("") == f"polars-bigquery/{__version__}"
    assert (
        _get_user_agent("custom-extension/1.0")
        == f"polars-bigquery/{__version__} custom-extension/1.0"
    )


def test_client_custom_credentials_provider():
    custom_creds = MagicMock()
    with patch("arrow_bigquery.Client") as mock_arrow_cls:
        client = Client(quota_project_id="test-proj", credentials_provider=custom_creds)
        assert client.credentials_provider is custom_creds
        assert client.quota_project_id == "test-proj"
        mock_arrow_cls.assert_called_once_with(
            quota_project_id="test-proj",
            credentials_provider=custom_creds,
            user_agent=f"polars-bigquery/{__version__}",
        )


def test_client_init_requires_quota_project_id():
    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'quota_project_id'"
    ):
        Client()


def test_client_read_bigquery_calls_arrow_with_parsed_id(mock_arrow_client):
    mock_exporter = MagicMock()
    mock_arrow_client.read_table.return_value = mock_exporter

    with patch("polars.DataFrame") as mock_df_cls:
        mock_df = MagicMock()
        mock_df_cls.return_value = mock_df

        client = Client(quota_project_id="q")
        result = client.read_table(table="my-project.my_dataset.my_table")

        mock_arrow_client.read_table.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            maintain_order=False,
        )
        mock_df_cls.assert_called_once_with(mock_exporter)
        assert result is mock_df


def test_client_read_query(mock_arrow_client):
    mock_exporter = MagicMock()
    mock_arrow_client.read_query.return_value = mock_exporter

    with patch("polars.DataFrame") as mock_df_cls:
        mock_df = MagicMock()
        mock_df_cls.return_value = mock_df

        client = Client(quota_project_id="q")
        result = client.read_query(query="SELECT 1")

        mock_arrow_client.read_query.assert_called_once_with(
            "SELECT 1",
            arrow_buffer_compression="lz4frame",
            dry_run=False,
            maintain_order=True,
            use_query_cache=None,
            maximum_bytes_billed=None,
        )
        mock_df_cls.assert_called_once_with(mock_exporter)
        assert result is mock_df


def test_client_read_query_passes_additional_parameters(mock_arrow_client):
    mock_exporter = MagicMock()
    mock_arrow_client.read_query.return_value = mock_exporter

    with patch("polars.DataFrame") as mock_df_cls:
        mock_df = MagicMock()
        mock_df_cls.return_value = mock_df

        client = Client(quota_project_id="q")
        result = client.read_query(
            query="SELECT 1",
            arrow_buffer_compression="zstd",
            dry_run=True,
            maintain_order=False,
            use_query_cache=False,
        )

        mock_arrow_client.read_query.assert_called_once_with(
            "SELECT 1",
            arrow_buffer_compression="zstd",
            dry_run=True,
            maintain_order=False,
            use_query_cache=False,
            maximum_bytes_billed=None,
        )
        mock_df_cls.assert_called_once_with(mock_exporter)
        assert result is mock_df


def test_client_read_query_propagates_errors(mock_arrow_client):
    mock_arrow_client.read_query.side_effect = Exception("Rust query error")

    client = Client(quota_project_id="q")
    with pytest.raises(Exception, match="Rust query error"):
        client.read_query(query="SELECT 1")


def test_client_scan_query_with_schema_pushes_projection_and_filter(mock_arrow_client):
    mock_arrow_client.read_query.return_value = MagicMock(name="arrow_stream")
    source_frame = pl.LazyFrame({"visible": ["a", "b"], "hidden": [5, 15]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT visible, hidden, unused FROM `p.d.t`;",
            schema={
                "visible": pl.String(),
                "hidden": pl.Int64(),
                "unused": pl.String(),
            },
            maintain_order=False,
        )
        result = lf.filter(pl.col("hidden") > 10).select("visible").collect()

    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT visible, hidden, unused FROM `p.d.t`\n"
        "|> WHERE (`hidden` > 10)\n"
        "|> SELECT `visible`, `hidden`",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=None,
    )
    assert result.to_dict(as_series=False) == {"visible": ["b"]}


def test_client_scan_query_without_schema_pushes_projection_and_filter_when_unordered(
    mock_arrow_client,
):
    exporter = MagicMock(name="arrow_stream")
    schema_json = (
        '{"fields": [{"name": "visible", "type": "STRING"}, '
        '{"name": "hidden", "type": "INTEGER"}, '
        '{"name": "unused", "type": "STRING"}]}'
    )
    mock_arrow_client.read_query.side_effect = [
        pl.DataFrame(
            {
                "bigquerySchema": [schema_json],
                "statementType": ["SELECT"],
            }
        ),
        exporter,
    ]
    source_frame = pl.LazyFrame({"visible": ["a", "b"], "hidden": [5, 15]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT visible, hidden, unused FROM `p.d.t`",
            maintain_order=False,
        )
        result = lf.filter(pl.col("hidden") > 10).select("visible").collect()

    assert mock_arrow_client.read_query.call_args_list[0].args == (
        "SELECT visible, hidden, unused FROM `p.d.t`",
    )
    assert mock_arrow_client.read_query.call_args_list[0].kwargs == {
        "arrow_buffer_compression": "lz4frame",
        "dry_run": True,
        "maintain_order": True,
        "use_query_cache": None,
        "maximum_bytes_billed": None,
    }
    expected_sql = (
        "SELECT visible, hidden, unused FROM `p.d.t`\n"
        "|> WHERE (`hidden` > 10)\n"
        "|> SELECT `visible`, `hidden`"
    )
    assert mock_arrow_client.read_query.call_args_list[1].args == (expected_sql,)
    assert mock_arrow_client.read_query.call_args_list[1].kwargs == {
        "arrow_buffer_compression": "lz4frame",
        "dry_run": False,
        "maintain_order": False,
        "use_query_cache": None,
        "maximum_bytes_billed": None,
    }
    assert result.to_dict(as_series=False) == {"visible": ["b"]}


def test_client_scan_query_filter_without_column_pruning_omits_pipe_select(
    mock_arrow_client,
):
    exporter = MagicMock(name="arrow_stream")
    schema_json = (
        '{"fields": [{"name": "f0_", "type": "INTEGER"}, '
        '{"name": "state", "type": "STRING"}]}'
    )
    mock_arrow_client.read_query.side_effect = [
        pl.DataFrame(
            {
                "bigquerySchema": [schema_json],
                "statementType": ["SELECT"],
            }
        ),
        exporter,
    ]
    source_frame = pl.LazyFrame({"f0_": [10, 20], "state": ["OR", "WA"]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT COUNT(*), state FROM `p.d.t` GROUP BY state",
            maintain_order=False,
        )
        result = lf.filter(pl.col("state") == "WA").collect()

    expected_sql = (
        "SELECT COUNT(*), state FROM `p.d.t` GROUP BY state\n|> WHERE (`state` = 'WA')"
    )
    assert mock_arrow_client.read_query.call_args_list[1].args == (expected_sql,)
    assert result.to_dict(as_series=False) == {"f0_": [20], "state": ["WA"]}


def test_client_scan_query_dry_run_skips_pipe_when_all_columns_needed(
    mock_arrow_client,
):
    schema_json = (
        '{"fields": [{"name": "a", "type": "INT64"}, {"name": "b", "type": "STRING"}]}'
    )
    exporter = MagicMock(name="arrow_stream")
    mock_arrow_client.read_query.side_effect = [
        pl.DataFrame(
            {
                "bigquerySchema": [schema_json],
                "statementType": ["SELECT"],
            }
        ),
        exporter,
        exporter,
    ]
    source_frame = pl.LazyFrame({"a": [1, 2], "b": ["x", "y"]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "  SELECT 1 AS a, 'x' AS b;  ",
            maintain_order=False,
        )
        assert lf.collect().to_dict(as_series=False) == {"a": [1, 2], "b": ["x", "y"]}
        # Column reorder alone does not prune columns, so raw query runs verbatim.
        assert lf.select(["b", "a"]).collect().columns == ["b", "a"]

    assert [call.args[0] for call in mock_arrow_client.read_query.call_args_list] == [
        "  SELECT 1 AS a, 'x' AS b;  ",
        "  SELECT 1 AS a, 'x' AS b;  ",
        "  SELECT 1 AS a, 'x' AS b;  ",
    ]


def test_client_scan_query_explicit_schema_always_emits_pipe_select_unordered(
    mock_arrow_client,
):
    mock_arrow_client.read_query.return_value = MagicMock(name="arrow_stream")
    source_frame = pl.LazyFrame({"a": [1, 2], "b": ["x", "y"]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT * FROM wide_table;",
            schema={"a": pl.Int64, "b": pl.String},
            maintain_order=False,
        )
        mock_arrow_client.read_query.assert_not_called()
        assert lf.collect().to_dict(as_series=False) == {"a": [1, 2], "b": ["x", "y"]}
        assert lf.select(["b", "a"]).collect().columns == ["b", "a"]

    assert [call.args[0] for call in mock_arrow_client.read_query.call_args_list] == [
        "SELECT * FROM wide_table\n|> SELECT `a`, `b`",
        "SELECT * FROM wide_table\n|> SELECT `a`, `b`",
    ]


def test_client_scan_query_dry_run_skips_pipe_for_unpushable_predicate_and_all_columns(
    mock_arrow_client,
):
    schema_json = '{"fields": [{"name": "a", "type": "INT64"}]}'
    exporter = MagicMock(name="arrow_stream")
    mock_arrow_client.read_query.side_effect = [
        pl.DataFrame(
            {
                "bigquerySchema": [schema_json],
                "statementType": ["SELECT"],
            }
        ),
        exporter,
    ]
    source_frame = pl.LazyFrame({"a": [1, 2, 1]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT a FROM `p.d.t`",
            maintain_order=False,
        )
        result = lf.filter(pl.col("a").is_duplicated()).collect()

    assert mock_arrow_client.read_query.call_args_list[1].args == (
        "SELECT a FROM `p.d.t`",
    )
    assert mock_arrow_client.read_query.call_args_list[1].kwargs == {
        "arrow_buffer_compression": "lz4frame",
        "dry_run": False,
        "maintain_order": False,
        "use_query_cache": None,
        "maximum_bytes_billed": None,
    }
    assert result.to_dict(as_series=False) == {"a": [1, 1]}


def test_client_scan_query_projection_only_without_filter(mock_arrow_client):
    mock_arrow_client.read_query.return_value = MagicMock(name="arrow_stream")
    source_frame = pl.LazyFrame({"visible": ["a", "b"]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT visible, unused FROM `p.d.t` -- trailing comment",
            schema={"visible": pl.String, "unused": pl.String},
            maintain_order=False,
        )
        result = lf.select("visible").collect()

    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT visible, unused FROM `p.d.t` -- trailing comment\n|> SELECT `visible`",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=None,
    )
    assert result.to_dict(as_series=False) == {"visible": ["a", "b"]}


def test_client_scan_query_validates_column_identifiers_on_projection(
    mock_arrow_client,
):
    client = Client(quota_project_id="q")
    lf = client.scan_query(
        "SELECT `a` FROM `p.d.t`",
        schema={"bad`col": pl.Int64, "unused": pl.Int64},
        maintain_order=False,
    )

    with pytest.raises(
        (ValueError, pl.exceptions.ComputeError), match="Invalid BigQuery column name"
    ):
        lf.select("bad`col").collect()

    mock_arrow_client.read_query.assert_not_called()


def test_client_scan_query_maintain_order_true_runs_verbatim_and_filters_locally(
    mock_arrow_client,
):
    exporter = MagicMock(name="arrow_stream")
    schema_json = (
        '{"fields": [{"name": "visible", "type": "STRING"}, '
        '{"name": "hidden", "type": "INTEGER"}]}'
    )
    mock_arrow_client.read_query.side_effect = [
        pl.DataFrame(
            {
                "bigquerySchema": [schema_json],
                "statementType": ["SELECT"],
            }
        ),
        exporter,
    ]
    source_frame = pl.LazyFrame({"visible": ["a", "b"], "hidden": [5, 15]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query("SELECT visible, hidden FROM `p.d.t`")
        result = lf.filter(pl.col("hidden") > 10).select("visible").collect()

    assert mock_arrow_client.read_query.call_args_list[0].args == (
        "SELECT visible, hidden FROM `p.d.t`",
    )
    assert mock_arrow_client.read_query.call_args_list[0].kwargs == {
        "arrow_buffer_compression": "lz4frame",
        "dry_run": True,
        "maintain_order": True,
        "use_query_cache": None,
        "maximum_bytes_billed": None,
    }
    assert mock_arrow_client.read_query.call_args_list[1].args == (
        "SELECT visible, hidden FROM `p.d.t`",
    )
    assert mock_arrow_client.read_query.call_args_list[1].kwargs == {
        "arrow_buffer_compression": "lz4frame",
        "dry_run": False,
        "maintain_order": True,
        "use_query_cache": None,
        "maximum_bytes_billed": None,
    }
    assert result.to_dict(as_series=False) == {"visible": ["b"]}


def test_client_scan_query_with_schema_handles_empty_results(mock_arrow_client):
    mock_arrow_client.read_query.return_value = MagicMock(name="arrow_stream")
    source_frame = pl.LazyFrame(schema={"value": pl.Int64})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        result = client.scan_query(
            "SELECT value FROM `p.d.t`", schema={"value": pl.Int64}
        ).collect()

    assert result.schema == pl.Schema({"value": pl.Int64})
    assert result.is_empty()


def test_client_scan_query_rejects_empty_query(mock_arrow_client):
    client = Client(quota_project_id="q")

    for bad_query in ("", "   ", " ; "):
        with pytest.raises(ValueError, match="non-empty SELECT statement"):
            client.scan_query(bad_query)


@pytest.mark.parametrize("stmt_type", ["INSERT", "CREATE_TABLE", "SCRIPT"])
def test_client_scan_query_rejects_non_select_statement_type(
    mock_arrow_client, stmt_type
):
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": pl.Series([None], dtype=pl.String),
            "statementType": [stmt_type],
        }
    )
    client = Client(quota_project_id="q")

    with pytest.raises(
        ValueError,
        match=f"scan_query only supports SELECT statements, got {stmt_type!r}",
    ):
        client.scan_query("CREATE TABLE p.d.t (x INT64)")


def test_client_scan_query_raises_bigquery_error_on_missing_schema_for_select(
    mock_arrow_client,
):
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": pl.Series([None], dtype=pl.String),
            "statementType": ["SELECT"],
        }
    )
    client = Client(quota_project_id="q")

    with pytest.raises(
        BigQueryError,
        match="Invalid response from BigQuery: dry run for SELECT query did not return a result schema",
    ):
        client.scan_query("SELECT 1")


def test_client_scan_query_raises_bigquery_error_on_missing_schema_and_statement_type(
    mock_arrow_client,
):
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": pl.Series([None], dtype=pl.String),
            "statementType": pl.Series([None], dtype=pl.String),
        }
    )
    client = Client(quota_project_id="q")

    with pytest.raises(
        BigQueryError,
        match="Invalid response from BigQuery: dry run did not return a statement type or result schema",
    ):
        client.scan_query("SELECT 1")


def test_client_scan_query_rejects_duplicate_column_names(mock_arrow_client):
    schema_json = (
        '{"fields": [{"name": "a", "type": "INTEGER"}, '
        '{"name": "b", "type": "STRING"}, '
        '{"name": "a", "type": "STRING"}]}'
    )
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": [schema_json],
            "statementType": ["SELECT"],
        }
    )
    client = Client(quota_project_id="q")

    with pytest.raises(
        ValueError,
        match=r"BigQuery schema contains duplicate column name\(s\): 'a'\.",
    ):
        client.scan_query("SELECT 1 AS a, 'x' AS b, 'y' AS a")


def test_client_scan_query_propagates_dry_run_errors(mock_arrow_client):
    mock_arrow_client.read_query.side_effect = BigQueryError("invalid query")
    client = Client(quota_project_id="q")

    with pytest.raises(BigQueryError, match="invalid query"):
        client.scan_query("SELECT broken")


def test_client_scan_query_invalid_sql_raises_polars_bigquery_error():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            payload = json.dumps(
                {
                    "error": {
                        "code": 400,
                        "message": 'Syntax error: Unexpected identifier "BROKEN"',
                        "status": "INVALID_ARGUMENT",
                    }
                }
            ).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 2.0
    port = server.server_port
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    try:
        creds = lambda: ({"bearer_token": "fake-tok"}, None)
        client = Client(quota_project_id="test-proj", credentials_provider=creds)
        client._arrow_client = arrow_bigquery.Client(
            quota_project_id="test-proj",
            credentials_provider=creds,
            bq_endpoint=f"http://127.0.0.1:{port}",
        )
        with pytest.raises(BigQueryError, match="Syntax error"):
            client.scan_query("SELECT BROKEN")
    finally:
        server_thread.join(timeout=2.0)
        server.server_close()


def test_client_read_bigquery_handles_bigquery_objects(mock_arrow_client):
    mock_exporter = MagicMock()
    mock_arrow_client.read_table.return_value = mock_exporter
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"

    with patch("polars.DataFrame"):
        client = Client(quota_project_id="q")
        client.read_table(table=mock_ref)

        mock_arrow_client.read_table.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("p", "d", "t"),
            maintain_order=False,
        )


def test_client_read_bigquery_propagates_errors(mock_arrow_client):
    mock_arrow_client.read_table.side_effect = Exception("Rust error")

    client = Client(quota_project_id="q")
    with pytest.raises(Exception, match="Rust error"):
        client.read_table(table="p.d.t")


def test_client_scan_bigquery_calls_arrow_with_parsed_id(mock_arrow_client):
    mock_stream = MagicMock()
    mock_arrow_client.read_table.return_value = mock_stream

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata"
        ) as mock_get_metadata,
        patch("polars.scan_arrow_c_stream") as mock_scan,
    ):
        mock_get_metadata.return_value = {
            "schema": {"fields": [{"name": "col1", "type": "INTEGER"}]}
        }
        mock_lazy_df = pl.LazyFrame({"col1": [1, 2]})
        mock_scan.return_value = mock_lazy_df

        client = Client(quota_project_id="q")
        result = client.scan_table(table="my-project.my_dataset.my_table")

        mock_get_metadata.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            quota_project_id="q",
            credentials_provider=client.credentials_provider,
            user_agent=f"polars-bigquery/{__version__}",
            session=client._rest_client.session,
        )

        mock_arrow_client.read_table.assert_not_called()
        mock_scan.assert_not_called()

        df = result.collect()

        mock_arrow_client.read_table.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            maintain_order=False,
            selected_fields=["col1"],
            row_restriction="",
        )
        mock_scan.assert_called_once_with(mock_stream)
        assert df.equals(mock_lazy_df.collect())


def test_client_scan_bigquery_with_schema_dict_skips_get_table_metadata(
    mock_arrow_client,
):
    mock_stream = MagicMock()
    mock_arrow_client.read_table.return_value = mock_stream

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata"
        ) as mock_get_metadata,
        patch("polars.scan_arrow_c_stream") as mock_scan,
    ):
        mock_lazy_df = pl.LazyFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_scan.return_value = mock_lazy_df

        client = Client(quota_project_id="q")
        result = client.scan_table(
            table="my-project.my_dataset.my_table",
            schema={"col1": pl.Int64, "col2": pl.String()},
        )

        mock_get_metadata.assert_not_called()
        assert result.collect_schema() == pl.Schema(
            {"col1": pl.Int64(), "col2": pl.String()}
        )

        df = result.collect()

        mock_arrow_client.read_table.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            maintain_order=False,
            selected_fields=["col1", "col2"],
            row_restriction="",
        )
        assert df.equals(mock_lazy_df.collect())


def test_client_scan_bigquery_with_polars_schema_pushes_down_projection_and_filter(
    mock_arrow_client,
):
    mock_stream = MagicMock()
    mock_arrow_client.read_table.return_value = mock_stream

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata"
        ) as mock_get_metadata,
        patch("polars.scan_arrow_c_stream") as mock_scan,
    ):
        mock_scan.return_value = pl.LazyFrame(
            {"visible": ["a", "b"], "hidden": [5, 15]}
        )

        client = Client(quota_project_id="q")
        lf = client.scan_table(
            table="my-project.my_dataset.my_table",
            schema=pl.Schema(
                {
                    "visible": pl.String(),
                    "hidden": pl.Int64(),
                    "unused": pl.String(),
                }
            ),
        )
        mock_get_metadata.assert_not_called()

        df = lf.filter(pl.col("hidden") > 10).select(["visible"]).collect()

        mock_arrow_client.read_table.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            maintain_order=False,
            selected_fields=["visible", "hidden"],
            row_restriction="(`hidden` > 10)",
        )
        assert df.equals(pl.DataFrame({"visible": ["b"]}))


def test_client_scan_table_drop_nulls_pushes_down_is_not_null(mock_arrow_client):
    mock_arrow_client.read_table.return_value = MagicMock()

    with patch("polars.scan_arrow_c_stream") as mock_scan:
        mock_scan.return_value = pl.LazyFrame(
            {"id": [1, 2, None], "flag": [None, True, False]}
        )
        client = Client(quota_project_id="q")
        schema = pl.Schema({"id": pl.Int64(), "flag": pl.Boolean()})

        # `lf.drop_nulls()` is lowered by Polars to `is_not_null()` conjuncts.
        result_all = (
            client.scan_table(table="p.d.t", schema=schema).drop_nulls().collect()
        )
        assert mock_arrow_client.read_table.call_args.kwargs == {
            "maintain_order": False,
            "selected_fields": ["id", "flag"],
            "row_restriction": "((`flag` IS NOT NULL) AND (`id` IS NOT NULL))",
        }
        assert result_all.to_dict(as_series=False) == {"id": [2], "flag": [True]}

        mock_arrow_client.read_table.reset_mock()
        mock_scan.return_value = pl.LazyFrame(
            {"id": [1, 2, 3], "flag": [None, True, None]}
        )

        # `lf.drop_nulls(subset=[...])` pushes down only that column's `IS NOT NULL`.
        result_subset = (
            client.scan_table(table="p.d.t", schema=schema)
            .drop_nulls(subset=["flag"])
            .collect()
        )
        assert mock_arrow_client.read_table.call_args.kwargs == {
            "maintain_order": False,
            "selected_fields": ["id", "flag"],
            "row_restriction": "(`flag` IS NOT NULL)",
        }
        assert result_subset.to_dict(as_series=False) == {"id": [2], "flag": [True]}


def test_client_scan_table_expr_drop_nulls_in_filter_not_pushed_down(mock_arrow_client):
    mock_arrow_client.read_table.return_value = MagicMock()
    client = Client(quota_project_id="q")
    schema = pl.Schema({"id": pl.Int64(), "flag": pl.Boolean()})

    # Single non-null value broadcasts across all rows including nulls.
    with patch(
        "polars.scan_arrow_c_stream",
        return_value=pl.LazyFrame({"id": [1, 2, 3], "flag": [None, True, None]}),
    ):
        result = (
            client.scan_table(table="p.d.t", schema=schema)
            .filter(pl.col("flag").drop_nulls())
            .collect()
        )
    assert mock_arrow_client.read_table.call_args.kwargs["row_restriction"] == ""
    assert result.to_dict(as_series=False) == {
        "id": [1, 2, 3],
        "flag": [None, True, None],
    }

    # Multiple non-null values alongside a null raise ShapeError in Polars.
    mock_arrow_client.read_table.reset_mock()
    with (
        patch(
            "polars.scan_arrow_c_stream",
            return_value=pl.LazyFrame({"id": [1, 2, 3], "flag": [None, True, False]}),
        ),
        pytest.raises(pl.exceptions.ShapeError),
    ):
        (
            client.scan_table(table="p.d.t", schema=schema)
            .filter(pl.col("flag").drop_nulls())
            .collect()
        )
    assert mock_arrow_client.read_table.call_args.kwargs["row_restriction"] == ""


def test_client_scan_query_drop_nulls_pushes_down_is_not_null(mock_arrow_client):
    mock_arrow_client.read_query.return_value = MagicMock(name="arrow_stream")
    source_frame = pl.LazyFrame({"id": [1, 2, 3], "flag": [None, True, None]})

    with patch("polars.scan_arrow_c_stream", return_value=source_frame):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT id, flag FROM `p.d.t`",
            schema={"id": pl.Int64(), "flag": pl.Boolean()},
            maintain_order=False,
        )
        result = lf.drop_nulls(subset=["flag"]).collect()

    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT id, flag FROM `p.d.t`\n"
        "|> WHERE (`flag` IS NOT NULL)\n"
        "|> SELECT `id`, `flag`",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=None,
    )
    assert result.to_dict(as_series=False) == {"id": [2], "flag": [True]}


def test_client_scan_bigquery_handles_bigquery_objects(mock_arrow_client):
    mock_stream = MagicMock()
    mock_arrow_client.read_table.return_value = mock_stream
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata"
        ) as mock_get_metadata,
        patch("polars.scan_arrow_c_stream") as mock_scan,
    ):
        mock_get_metadata.return_value = {
            "schema": {"fields": [{"name": "col1", "type": "INTEGER"}]}
        }
        mock_lazy_df = pl.LazyFrame({"col1": [1, 2]})
        mock_scan.return_value = mock_lazy_df

        client = Client(quota_project_id="q")
        result = client.scan_table(table=mock_ref)
        df = result.collect()

        mock_arrow_client.read_table.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("p", "d", "t"),
            maintain_order=False,
            selected_fields=["col1"],
            row_restriction="",
        )
        assert df.equals(mock_lazy_df.collect())


def test_client_scan_bigquery_projects_filter_cols(
    mock_arrow_client,
):
    mock_stream = MagicMock()
    mock_arrow_client.read_table.return_value = mock_stream

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata"
        ) as mock_get_metadata,
        patch("polars.scan_arrow_c_stream") as mock_scan,
    ):
        mock_get_metadata.return_value = {
            "schema": {
                "fields": [
                    {"name": "visible", "type": "STRING"},
                    {"name": "hidden", "type": "INTEGER"},
                    {"name": "unused", "type": "STRING"},
                ]
            }
        }
        mock_scan.return_value = pl.LazyFrame(
            {"visible": ["a", "b"], "hidden": [5, 15]}
        )

        client = Client(quota_project_id="q")
        lf = client.scan_table(table="my-project.my_dataset.my_table")
        mock_get_metadata.assert_called_once()

        # Filter on 'hidden' (not in select projection) and project 'visible'
        df = lf.filter(pl.col("hidden") > 10).select(["visible"]).collect()

        mock_arrow_client.read_table.assert_called_once_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            maintain_order=False,
            selected_fields=["visible", "hidden"],
            row_restriction="(`hidden` > 10)",
        )
        assert df.equals(pl.DataFrame({"visible": ["b"]}))


def test_client_scan_ingestion_time_partitioned_table(mock_arrow_client):
    from datetime import datetime, timezone

    mock_stream = MagicMock()
    mock_arrow_client.read_table.return_value = mock_stream

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata"
        ) as mock_get_metadata,
        patch("polars.scan_arrow_c_stream") as mock_scan,
        patch(
            "polars.io.plugins.register_io_source",
            wraps=pl.io.plugins.register_io_source,
        ) as mock_register,
    ):
        mock_get_metadata.return_value = {
            "schema": {"fields": [{"name": "val", "type": "INTEGER"}]},
            "timePartitioning": {"type": "DAY"},
        }
        mock_scan.return_value = pl.LazyFrame(
            {
                "val": [10, 20],
                "_PARTITIONDATE": [date(2024, 1, 1), date(2024, 1, 2)],
                "_PARTITIONTIME": [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                ],
            }
        )

        client = Client(quota_project_id="q")
        lf = client.scan_table(table="my-project.my_dataset.my_table")

        # Verify the underlying io_source yields batches with deterministic schema column order
        io_source = mock_register.call_args.kwargs["io_source"]
        raw_df = next(io_source(None, None, None, None))
        assert raw_df.columns == ["val", "_PARTITIONDATE", "_PARTITIONTIME"]

        # 1. Verify unfiltered collect includes pseudo-columns in selected_fields
        df_all = lf.collect()
        mock_arrow_client.read_table.assert_called_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            maintain_order=False,
            selected_fields=["val", "_PARTITIONDATE", "_PARTITIONTIME"],
            row_restriction="",
        )
        assert df_all.columns == ["val", "_PARTITIONDATE", "_PARTITIONTIME"]
        assert df_all["_PARTITIONDATE"].to_list() == [
            date(2024, 1, 1),
            date(2024, 1, 2),
        ]

        # 2. Verify filter on _PARTITIONDATE includes _PARTITIONDATE in selected_fields
        df_filtered = (
            lf.filter(pl.col("_PARTITIONDATE") == date(2024, 1, 1))
            .select("val")
            .collect()
        )
        mock_arrow_client.read_table.assert_called_with(
            arrow_bigquery.BigQueryTableId("my-project", "my_dataset", "my_table"),
            maintain_order=False,
            selected_fields=["val", "_PARTITIONDATE"],
            row_restriction="(`_PARTITIONDATE` = DATE(TIMESTAMP_SECONDS(19723 * 86400)))",
        )
        assert df_filtered.columns == ["val"]
        assert df_filtered["val"].to_list() == [10]

        # 3. Verify compound filter with _PARTITIONDATE and physical column
        df_compound = (
            lf.filter(
                (pl.col("_PARTITIONDATE") == date(2024, 1, 2)) & (pl.col("val") > 15)
            )
            .select("val")
            .collect()
        )
        assert df_compound["val"].to_list() == [20]


def test_client_context_manager_closes_rest_session(mock_arrow_client):
    with patch.object(
        Client, "close", wraps=Client(quota_project_id="q").close
    ) as mock_close:
        with Client(quota_project_id="q") as client:
            assert client.quota_project_id == "q"
        mock_close.assert_called_once()


def test_client_read_query_dry_run_returns_polars_dataframe():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            assert self.path.startswith("/bigquery/v2/projects/test-proj/queries")
            assert body["query"] == "SELECT 1 AS one"
            assert body.get("dryRun") is True

            resp = {
                "jobReference": {
                    "projectId": "test-proj",
                    "location": "US",
                },
                "totalBytesProcessed": "128",
                "cacheHit": True,
                "statementType": "SELECT",
                "creationTime": "1767037135155",
                "schema": {"fields": [{"name": "one", "type": "INT64"}]},
            }
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 2.0
    port = server.server_port
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    try:
        creds = lambda: ({"bearer_token": "fake-tok"}, None)
        client = Client(quota_project_id="test-proj", credentials_provider=creds)
        client._arrow_client = arrow_bigquery.Client(
            quota_project_id="test-proj",
            credentials_provider=creds,
            bq_endpoint=f"http://127.0.0.1:{port}",
        )
        df = client.read_query("SELECT 1 AS one", dry_run=True)
    finally:
        server_thread.join(timeout=2.0)
        server.server_close()

    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert df.columns == [
        "bigquerySchema",
        "projectId",
        "jobId",
        "location",
        "totalBytesProcessed",
        "cacheHit",
        "statementType",
        "creationTime",
    ]
    assert df["projectId"].to_list() == ["test-proj"]
    assert df["location"].to_list() == ["US"]
    assert df["totalBytesProcessed"].to_list() == [128]
    assert df["cacheHit"].to_list() == [True]
    assert df["statementType"].to_list() == ["SELECT"]


@pytest.mark.parametrize("bad_codec", ["snappy", "gzip", 123, ["lz4frame"], None])
def test_client_scan_query_rejects_invalid_compression(mock_arrow_client, bad_codec):
    client = Client(quota_project_id="q")
    with pytest.raises(ValueError, match="arrow_buffer_compression must be one of"):
        client.scan_query("SELECT 1", arrow_buffer_compression=bad_codec)
    mock_arrow_client.read_query.assert_not_called()


def test_client_scan_query_eager_schema_validation(mock_arrow_client):
    client = Client(quota_project_id="q")

    # Empty dict or empty pl.Schema raises ValueError without dry-run RPC.
    with pytest.raises(ValueError, match="at least one column"):
        client.scan_query("SELECT 1", schema={})
    with pytest.raises(ValueError, match="at least one column"):
        client.scan_query("SELECT 1", schema=pl.Schema())

    # Unspecified dtype class raises TypeError eagerly.
    with pytest.raises(TypeError):
        client.scan_query("SELECT 1", schema={"ts": pl.Datetime})

    # Empty dry-run "fields" list raises BigQueryError.
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": ['{"fields": []}'],
            "statementType": ["SELECT"],
        }
    )
    with pytest.raises(BigQueryError, match="empty schema"):
        client.scan_query("SELECT 1")

    # Unknown BigQuery type in dry-run raises TypeError eagerly.
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": ['{"fields": [{"name": "x", "type": "BOGUS"}]}'],
            "statementType": ["SELECT"],
        }
    )
    with pytest.raises(TypeError, match="got unexpected BigQuery type: bogus"):
        client.scan_query("SELECT x")


def test_client_scan_table_eager_validation(mock_arrow_client):
    client = Client(quota_project_id="q")

    # Empty explicit schema raises ValueError without REST call.
    with patch("polars_bigquery.core.bigquery_rest.get_table_metadata") as mock_meta:
        with pytest.raises(ValueError, match="at least one column"):
            client.scan_table("p.d.t", schema={})
        with pytest.raises(ValueError, match="at least one column"):
            client.scan_table("p.d.t", schema=pl.Schema())
        with pytest.raises(TypeError):
            client.scan_table("p.d.t", schema={"ts": pl.Datetime})
        mock_meta.assert_not_called()

    # Empty table metadata fields raises BigQueryError even when ingestion-time partitioned.
    with patch("polars_bigquery.core.bigquery_rest.get_table_metadata") as mock_meta:
        mock_meta.return_value = {
            "schema": {"fields": []},
            "timePartitioning": {"type": "DAY"},
        }
        with pytest.raises(BigQueryError, match="did not contain any schema fields"):
            client.scan_table("p.d.t")

    # Duplicate table metadata column names raise ValueError.
    with patch("polars_bigquery.core.bigquery_rest.get_table_metadata") as mock_meta:
        mock_meta.return_value = {
            "schema": {
                "fields": [
                    {"name": "a", "type": "INT64"},
                    {"name": "a", "type": "STRING"},
                ]
            }
        }
        with pytest.raises(ValueError, match="duplicate column name"):
            client.scan_table("p.d.t")

    # Unknown BigQuery field type in metadata raises TypeError.
    with patch("polars_bigquery.core.bigquery_rest.get_table_metadata") as mock_meta:
        mock_meta.return_value = {
            "schema": {"fields": [{"name": "bad", "type": "MYSTERY"}]}
        }
        with pytest.raises(TypeError, match="got unexpected BigQuery type: mystery"):
            client.scan_table("p.d.t")


def test_client_scan_query_dry_run_isolation_and_post_init_override(mock_arrow_client):
    client = Client(quota_project_id="q")
    replacement_arrow = MagicMock()
    schema_json = '{"fields": [{"name": "a", "type": "INT64"}]}'
    replacement_arrow.read_query.side_effect = [
        pl.DataFrame(
            {
                "bigquerySchema": [schema_json],
                "statementType": ["SELECT"],
                "location": ["US"],
                "totalBytesProcessed": [1024],
                "cacheHit": [False],
            }
        ),
        MagicMock(name="stream"),
    ]
    client._arrow_client = replacement_arrow

    # Patching public Client.read_query must NOT intercept scan_query's internal dry run.
    with (
        patch.object(
            Client, "read_query", side_effect=AssertionError("should not be called")
        ),
        patch("polars.scan_arrow_c_stream", return_value=pl.LazyFrame({"a": [42]})),
    ):
        lf = client.scan_query("SELECT a FROM t")
        df = lf.collect()

    assert df.to_dict(as_series=False) == {"a": [42]}
    mock_arrow_client.read_query.assert_not_called()
    assert replacement_arrow.read_query.call_count == 2


def test_client_scan_table_post_init_arrow_client_override(mock_arrow_client):
    client = Client(quota_project_id="q")
    replacement_arrow = MagicMock()
    replacement_arrow.read_table.return_value = MagicMock(name="stream")
    client._arrow_client = replacement_arrow

    with patch("polars.scan_arrow_c_stream", return_value=pl.LazyFrame({"a": [7]})):
        lf = client.scan_table("p.d.t", schema={"a": pl.Int64})
        df = lf.collect()

    assert df.to_dict(as_series=False) == {"a": [7]}
    mock_arrow_client.read_table.assert_not_called()
    replacement_arrow.read_table.assert_called_once()


def test_io_source_closures_do_not_capture_client_or_rest_client(mock_arrow_client):
    client = Client(quota_project_id="q")

    with patch(
        "polars.io.plugins.register_io_source",
        wraps=pl.io.plugins.register_io_source,
    ) as mock_register:
        client.scan_query("SELECT a FROM t", schema={"a": pl.Int64})
        query_source_fn = mock_register.call_args.kwargs["io_source"]
        captured_query = [
            cell.cell_contents for cell in query_source_fn.__closure__ or ()
        ]
        assert not any(
            isinstance(obj, (Client, bigquery_rest.BigQueryRestClient))
            for obj in captured_query
        )

        client.scan_table("p.d.t", schema={"a": pl.Int64})
        table_source_fn = mock_register.call_args.kwargs["io_source"]
        captured_table = [
            cell.cell_contents for cell in table_source_fn.__closure__ or ()
        ]
        assert not any(
            isinstance(obj, (Client, bigquery_rest.BigQueryRestClient))
            for obj in captured_table
        )


def test_client_scan_query_and_table_explain_metadata(mock_arrow_client):
    client = Client(quota_project_id="q")
    secret_sql = (
        "SELECT col FROM secret_dataset.secret_table WHERE token = 'top_secret'"
    )
    schema_json = '{"fields": [{"name": "col", "type": "STRING"}]}'

    # Ordered dry run with location
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": [schema_json],
            "statementType": ["SELECT"],
            "location": ["US"],
        }
    )
    lf_ord = client.scan_query(secret_sql)
    plan_ord = lf_ord.explain()
    assert "PYTHON[BIGQUERY_QUERY] SCAN" in plan_ord
    assert (
        "maintain_order=True, pushdown=disabled, schema=dry_run, location=US"
        in plan_ord
    )
    assert "secret_dataset" not in plan_ord
    assert "top_secret" not in plan_ord

    # Unordered dry run without location
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": [schema_json],
            "statementType": ["SELECT"],
            "location": pl.Series([None], dtype=pl.String),
        }
    )
    lf_unord = client.scan_query(secret_sql, maintain_order=False)
    plan_unord = lf_unord.explain()
    assert "maintain_order=False, pushdown=enabled, schema=dry_run" in plan_unord
    assert "location=" not in plan_unord
    assert "top_secret" not in plan_unord

    # Explicit schema omits location
    lf_exp = client.scan_query(
        secret_sql,
        schema={"col": pl.String},
        maintain_order=False,
    )
    plan_exp = lf_exp.explain()
    assert "maintain_order=False, pushdown=enabled, schema=explicit" in plan_exp
    assert "location=" not in plan_exp
    assert "top_secret" not in plan_exp

    # scan_table explicit & metadata explain_detail
    lf_tbl_exp = client.scan_table("my-proj.my_ds.my_tbl", schema={"col": pl.String})
    plan_tbl_exp = lf_tbl_exp.explain()
    assert "PYTHON[BIGQUERY_TABLE] SCAN" in plan_tbl_exp
    assert "table=my-proj.my_ds.my_tbl, schema=explicit" in plan_tbl_exp

    with patch("polars_bigquery.core.bigquery_rest.get_table_metadata") as mock_meta:
        mock_meta.return_value = {
            "schema": {"fields": [{"name": "col", "type": "STRING"}]}
        }
        lf_tbl_meta = client.scan_table("my-proj.my_ds.my_tbl")
    plan_tbl_meta = lf_tbl_meta.explain()
    assert "PYTHON[BIGQUERY_TABLE] SCAN" in plan_tbl_meta
    assert "table=my-proj.my_ds.my_tbl, schema=metadata" in plan_tbl_meta


def test_client_scan_query_logging_and_degradation_warning(
    mock_arrow_client, monkeypatch, caplog
):
    schema_json = (
        '{"fields": [{"name": "a", "type": "INT64"}, {"name": "b", "type": "STRING"}]}'
    )
    dry_run_df = pl.DataFrame(
        {
            "bigquerySchema": [schema_json],
            "statementType": ["SELECT"],
            "location": ["EU"],
            "totalBytesProcessed": [4096],
            "cacheHit": [True],
        }
    )
    secret_sql = "SELECT a, b FROM private_ds.secret_table"
    mock_arrow_client.read_query.side_effect = [
        dry_run_df,
        MagicMock(name="stream"),
    ]

    monkeypatch.setattr(query_executor_mod, "MAX_UNRESOLVED_SQL_BYTES", 60)

    client = Client(quota_project_id="q")
    with (
        caplog.at_level(logging.DEBUG, logger="polars_bigquery"),
        patch(
            "polars.scan_arrow_c_stream",
            return_value=pl.LazyFrame({"a": [1, 2], "b": ["secret_lit", "other"]}),
        ),
    ):
        lf = client.scan_query(secret_sql, maintain_order=False)
        df = lf.filter(
            (pl.col("a") == 1) & (pl.col("b") == "secret_lit_" + "z" * 40)
        ).collect()

    assert df.is_empty()
    dry_run_logs = [
        r.message
        for r in caplog.records
        if r.name == "polars_bigquery._read_bigquery" and r.levelno == logging.DEBUG
    ]
    assert len(dry_run_logs) == 1
    assert "location=EU" in dry_run_logs[0]
    assert "total_bytes_processed=4096" in dry_run_logs[0]
    assert "cache_hit=True" in dry_run_logs[0]
    assert "columns=2" in dry_run_logs[0]
    assert "secret_table" not in dry_run_logs[0]

    warn_logs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_logs) == 1
    assert "sql_length_pruned_conjuncts" in warn_logs[0]
    assert "max_sql_bytes=60" in warn_logs[0]
    expected_sql = f"{secret_sql}\n|> WHERE (`a` = 1)"
    assert f"sql_bytes={len(expected_sql.encode('utf-8'))}" in warn_logs[0]
    assert "secret_table" not in warn_logs[0]
    assert "secret_lit" not in warn_logs[0]


def test_client_scan_query_and_table_real_arrow_c_stream_and_teardown(
    mock_arrow_client,
):
    """Verify end-to-end execution via __arrow_c_stream__ with zero scan_arrow_c_stream mocks."""
    call_log: list[str] = []

    class _DataFrameStreamExporter:
        def __init__(self, frame: pl.DataFrame) -> None:
            self._frame = frame

        def __arrow_c_stream__(self, requested_schema=None):
            call_log.append("exported")
            return self._frame.__arrow_c_stream__(requested_schema)

    query_frame = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    exporter_q = _DataFrameStreamExporter(query_frame)
    exporter_q_ref = weakref.ref(exporter_q)
    mock_arrow_client.read_query.return_value = exporter_q

    client = Client(quota_project_id="q")
    lf_q = client.scan_query(
        "SELECT a, b FROM t",
        schema={"a": pl.Int64, "b": pl.String},
        maintain_order=False,
    )
    res_q = lf_q.filter(pl.col("a") >= 2).select(["b", "a"]).collect()
    assert res_q.columns == ["b", "a"]
    assert res_q.to_dict(as_series=False) == {
        "b": ["x", "y", "z"],
        "a": [2, 3, 4],
    }
    assert call_log == ["exported"]

    # Release references and confirm exporter is garbage-collected cleanly.
    mock_arrow_client.read_query.return_value = None
    del exporter_q
    gc.collect()
    assert exporter_q_ref() is None

    # Early-stop batch generator teardown via .close()
    table_frame = pl.DataFrame({"x": [10, 20, 30, 40], "y": [1, 2, 3, 4]})
    exporter_t = _DataFrameStreamExporter(table_frame)
    exporter_t_ref = weakref.ref(exporter_t)
    mock_arrow_client.read_table.return_value = exporter_t

    lf_t = client.scan_table("p.d.t", schema={"x": pl.Int64, "y": pl.Int64})
    batches_iter = lf_t.select(["y", "x"]).collect_batches(chunk_size=1)
    first_batch = next(iter(batches_iter))
    assert first_batch.columns == ["y", "x"]
    assert first_batch.height >= 1
    if hasattr(batches_iter, "close"):
        batches_iter.close()
    mock_arrow_client.read_table.return_value = None
    del batches_iter, exporter_t
    gc.collect()
    assert exporter_t_ref() is None


def test_client_scan_query_limit_pushdown_unordered(mock_arrow_client):
    mock_arrow_client.read_query.return_value = MagicMock(name="arrow_stream")
    with patch(
        "polars.scan_arrow_c_stream",
        return_value=pl.LazyFrame({"a": [1, 2, 3]}),
    ):
        client = Client(quota_project_id="q")
        lf = client.scan_query(
            "SELECT a, b FROM t",
            schema={"a": pl.Int64, "b": pl.String},
            maintain_order=False,
        )
        df = lf.limit(2).select("a").collect()

    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT a, b FROM t\n|> SELECT `a`\n|> LIMIT 2",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=None,
    )
    assert df.to_dict(as_series=False) == {"a": [1, 2]}


def test_client_scan_table_zero_column_selects_narrowest_physical_column(
    mock_arrow_client,
):
    mock_arrow_client.read_table.return_value = MagicMock(name="arrow_stream")
    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata",
            return_value={
                "schema": {
                    "fields": [
                        {"name": "payload", "type": "STRING"},
                        {"name": "flag", "type": "BOOLEAN"},
                    ]
                },
                "timePartitioning": {"type": "HOUR"},
            },
        ),
        patch(
            "polars.scan_arrow_c_stream",
            return_value=pl.LazyFrame({"flag": [True, False]}),
        ),
    ):
        client = Client(quota_project_id="q")
        lf = client.scan_table("p.d.t")
        df = lf.drop(pl.all()).collect()

    mock_arrow_client.read_table.assert_called_once_with(
        arrow_bigquery.BigQueryTableId("p", "d", "t"),
        maintain_order=False,
        selected_fields=["flag"],
        row_restriction="",
    )
    assert df.width == 0


def test_client_scan_query_preserves_caller_schema_copy(mock_arrow_client):
    caller_schema = pl.Schema({"a": pl.Int64(), "b": pl.String()})
    client = Client(quota_project_id="q")
    lf = client.scan_query("SELECT a, b FROM t", schema=caller_schema)
    caller_schema["c"] = pl.Boolean()
    assert lf.collect_schema() == pl.Schema({"a": pl.Int64(), "b": pl.String()})


def test_client_scan_query_forwards_compression_and_cache_options(mock_arrow_client):
    schema_json = '{"fields": [{"name": "a", "type": "INT64"}]}'
    mock_arrow_client.read_query.side_effect = [
        pl.DataFrame(
            {
                "bigquerySchema": [schema_json],
                "statementType": ["SELECT"],
            }
        ),
        MagicMock(name="stream"),
    ]
    raw = "\n  SELECT a FROM `p.d.t`;\n"
    client = Client(quota_project_id="q")
    with patch("polars.scan_arrow_c_stream", return_value=pl.LazyFrame({"a": [1]})):
        df = client.scan_query(
            raw,
            arrow_buffer_compression="zstd",
            maintain_order=True,
            use_query_cache=False,
        ).collect()

    assert df.to_dict(as_series=False) == {"a": [1]}
    assert mock_arrow_client.read_query.call_args_list[0].args == (raw,)
    assert mock_arrow_client.read_query.call_args_list[0].kwargs == {
        "arrow_buffer_compression": "zstd",
        "dry_run": True,
        "maintain_order": True,
        "use_query_cache": None,
        "maximum_bytes_billed": None,
    }
    assert mock_arrow_client.read_query.call_args_list[1].args == (raw,)
    assert mock_arrow_client.read_query.call_args_list[1].kwargs == {
        "arrow_buffer_compression": "zstd",
        "dry_run": False,
        "maintain_order": True,
        "use_query_cache": False,
        "maximum_bytes_billed": None,
    }


def test_client_read_and_scan_reject_malicious_table_ids(mock_arrow_client):
    class FakeTable:
        project = "proj/../../evil"
        dataset_id = "ds"
        table_id = "tab?fields=all#"

    with patch(
        "polars_bigquery.core.bigquery_rest.get_table_metadata"
    ) as mock_get_metadata:
        client = Client(
            quota_project_id="q",
            credentials_provider=lambda: ({"bearer_token": "tok"}, None),
        )

        for bad_table in ("proj.ds.tab?fields=etag#frag", FakeTable()):
            with pytest.raises(ValueError, match="Invalid table ID"):
                client.read_table(bad_table)
            with pytest.raises(ValueError, match="Invalid table ID"):
                client.scan_table(bad_table)

        mock_arrow_client.read_table.assert_not_called()
        mock_get_metadata.assert_not_called()


def test_client_lazy_rest_client_and_lifecycle_guards(mock_arrow_client):
    with patch(
        "polars_bigquery.core.bigquery_rest.BigQueryRestClient"
    ) as mock_rest_cls:
        client = Client(quota_project_id="q")
        # Query-only initialization must NOT allocate an HTTP session pool eagerly.
        mock_rest_cls.assert_not_called()

        mock_arrow_client.read_query.return_value = MagicMock()
        with patch("polars.DataFrame"):
            client.read_query("SELECT 1")
        mock_rest_cls.assert_not_called()

        # close() closes _arrow_client and is idempotent without opening REST session.
        client.close()
        client.close()
        mock_rest_cls.assert_not_called()
        mock_arrow_client.close.assert_called_once()

        # Subsequent operations (including direct _rest_client access) raise RuntimeError when closed.
        with pytest.raises(RuntimeError, match="closed"):
            _ = client._rest_client
        mock_rest_cls.assert_not_called()
        with pytest.raises(RuntimeError, match="closed"):
            client.read_table("p.d.t")
        with pytest.raises(RuntimeError, match="closed"):
            client.read_query("SELECT 1")
        with pytest.raises(RuntimeError, match="closed"):
            client.scan_table("p.d.t", schema={"a": pl.Int64})
        with pytest.raises(RuntimeError, match="closed"):
            client.scan_query("SELECT 1", schema={"a": pl.Int64})


def test_client_rest_client_lazy_init_is_thread_safe(mock_arrow_client):
    import threading

    with patch(
        "polars_bigquery.core.bigquery_rest.BigQueryRestClient"
    ) as mock_rest_cls:
        sentinel = MagicMock(name="rest_client")
        mock_rest_cls.return_value = sentinel
        client = Client(quota_project_id="q")

        barrier = threading.Barrier(8)
        results: list[object] = []
        results_lock = threading.Lock()

        def worker():
            barrier.wait()
            rc = client._rest_client
            with results_lock:
                results.append(rc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is sentinel for r in results)
        mock_rest_cls.assert_called_once()


def test_client_input_type_and_wire_payload_validation(mock_arrow_client):
    client = Client(quota_project_id="q")

    # Non-str query rejected eagerly
    with pytest.raises(TypeError, match="query must be a str"):
        client.scan_query(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="query must be a str"):
        client.read_query(None)  # type: ignore[arg-type]

    # read_query shares _validate_compression parity with scan_query
    with pytest.raises(ValueError, match="arrow_buffer_compression must be one of"):
        client.read_query("SELECT 1", arrow_buffer_compression="gzip")  # type: ignore[arg-type]

    # Non-bool flags rejected eagerly (no silent truthiness conversion)
    with pytest.raises(TypeError, match="maintain_order must be a bool"):
        client.scan_query("SELECT 1", maintain_order="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use_query_cache must be a bool or None"):
        client.scan_query("SELECT 1", use_query_cache="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dry_run must be a bool"):
        client.read_query("SELECT 1", dry_run=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="maintain_order must be a bool"):
        client.read_table("p.d.t", maintain_order="false")  # type: ignore[arg-type]

    # Semicolon-only / comment-only queries rejected eagerly without RPC
    for empty_sql in (";;", " ; ; ", "-- just a comment", "/* comment only */"):
        with pytest.raises(ValueError, match="non-empty SELECT statement"):
            client.scan_query(empty_sql)
    mock_arrow_client.read_query.assert_not_called()

    # Wire-payload error boundaries on dry-run response
    bad_payloads = [
        pl.DataFrame(schema={"bigquerySchema": pl.String, "statementType": pl.String}),
        pl.DataFrame({"statementType": ["SELECT"]}),
        pl.DataFrame({"bigquerySchema": ["{not json"], "statementType": ["SELECT"]}),
        pl.DataFrame({"bigquerySchema": ["null"], "statementType": ["SELECT"]}),
        pl.DataFrame({"bigquerySchema": ["[]"], "statementType": ["SELECT"]}),
        pl.DataFrame(
            {"bigquerySchema": ['{"fields": "bad"}'], "statementType": ["SELECT"]}
        ),
        pl.DataFrame({"bigquerySchema": [123], "statementType": ["SELECT"]}),
    ]
    for bad_df in bad_payloads:
        mock_arrow_client.read_query.return_value = bad_df
        with pytest.raises(BigQueryError, match="Invalid response from BigQuery"):
            client.scan_query("SELECT 1")

    # scan_table wire-payload error boundaries
    for bad_meta in (
        ["not-a-dict"],
        {"schema": "bad"},
        {"schema": None},
        {"schema": {"fields": "bad"}},
    ):
        with (
            patch(
                "polars_bigquery.core.bigquery_rest.get_table_metadata",
                return_value=bad_meta,
            ),
            pytest.raises(BigQueryError, match="Invalid response from BigQuery"),
        ):
            client.scan_table("p.d.t")


def test_client_scan_bignumeric_raises_clear_type_error(mock_arrow_client):
    client = Client(quota_project_id="q")
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": ['{"fields": [{"name": "bn", "type": "BIGNUMERIC"}]}'],
            "statementType": ["SELECT"],
        }
    )
    with pytest.raises(TypeError, match="BIGNUMERIC/BIGDECIMAL"):
        client.scan_query("SELECT bn FROM t")

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata",
            return_value={"schema": {"fields": [{"name": "bn", "type": "BIGNUMERIC"}]}},
        ),
        pytest.raises(TypeError, match="BIGNUMERIC/BIGDECIMAL"),
    ):
        client.scan_table("p.d.t")

    # Nested BIGNUMERIC inside ARRAY (`mode="REPEATED"`) and STRUCT also raise TypeError.
    for schema_json, col_name in (
        (
            '{"fields": [{"name": "arr_bn", "type": "BIGNUMERIC", "mode": "REPEATED"}]}',
            "arr_bn",
        ),
        (
            '{"fields": [{"name": "st_bn", "type": "STRUCT", "fields": [{"name": "x", "type": "BIGNUMERIC"}]}]}',
            "st_bn",
        ),
    ):
        mock_arrow_client.read_query.return_value = pl.DataFrame(
            {"bigquerySchema": [schema_json], "statementType": ["SELECT"]}
        )
        with pytest.raises(TypeError, match=f"BIGNUMERIC/BIGDECIMAL.*{col_name}"):
            client.scan_query("SELECT col FROM t")

    # Explicit schema with nested BIGNUMERIC also produces a clear TypeError.
    with pytest.raises(TypeError, match="BIGNUMERIC/BIGDECIMAL.*'s'"):
        client.scan_query(
            "SELECT s FROM t",
            schema={"s": pl.Struct([pl.Field("x", pl.Decimal(76, 38))])},
        )


def test_client_scan_query_sanitizes_control_chars_in_explain_location(
    mock_arrow_client,
):
    client = Client(quota_project_id="q")
    mock_arrow_client.read_query.return_value = pl.DataFrame(
        {
            "bigquerySchema": ['{"fields": [{"name": "a", "type": "INT64"}]}'],
            "statementType": ["SELECT"],
            "location": ["US\nINJECTED\r\t"],
        }
    )
    lf = client.scan_query("SELECT 1 AS a")
    explain = lf.explain()
    info_lines = [line for line in explain.splitlines() if "INFO:" in line]
    assert len(info_lines) == 1
    assert "location=US INJECTED" in info_lines[0]
    assert "INJECTED\n" not in explain


def test_client_scan_query_with_udf_filter_degrades_gracefully(mock_arrow_client):
    import warnings

    from polars_bigquery.core.executors import QueryExecutor
    from polars_bigquery.core.plan import Filter, Project, QueryScan

    mock_arrow_client.read_query.return_value = MagicMock(name="stream")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pl.exceptions.PolarsInefficientMapWarning)
        udf_pred = pl.col("a").map_elements(lambda x: x > 1, return_dtype=pl.Boolean)
    scan = QueryScan(
        raw_query="SELECT a FROM t",
        columns=("a",),
        maintain_order=False,
        explicit_schema=True,
    )
    plan = Project(input=Filter(input=scan, predicate=udf_pred), columns=("a",))
    executor = QueryExecutor(mock_arrow_client)
    with patch(
        "polars.scan_arrow_c_stream",
        return_value=pl.LazyFrame({"a": [1, 2, 3]}),
    ):
        batches = list(executor.execute(plan))
    assert sorted(pl.concat(batches)["a"].to_list()) == [2, 3]
    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT a FROM t\n|> SELECT `a`",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=None,
    )


def test_client_forwards_and_validates_maximum_bytes_billed(mock_arrow_client):
    from polars_bigquery.core.plan import MAX_BIGQUERY_INT64

    client = Client(quota_project_id="q")

    # Eager type and range checks on both read_query and scan_query (including
    # huge integers that would exceed sys.get_int_max_str_digits() if formatted).
    for bad_type in (True, False, 1.5, "100"):
        with pytest.raises(TypeError, match="maximum_bytes_billed must be an int"):
            client.read_query("SELECT 1", maximum_bytes_billed=bad_type)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="maximum_bytes_billed must be an int"):
            client.scan_query(
                "SELECT 1",
                schema={"a": pl.Int64},
                maximum_bytes_billed=bad_type,  # type: ignore[arg-type]
            )

    for bad_neg in (-1, -100, -(10**5000)):
        with pytest.raises(
            ValueError, match="maximum_bytes_billed must be non-negative"
        ):
            client.read_query("SELECT 1", maximum_bytes_billed=bad_neg)
        with pytest.raises(
            ValueError, match="maximum_bytes_billed must be non-negative"
        ):
            client.scan_query(
                "SELECT 1",
                schema={"a": pl.Int64},
                maximum_bytes_billed=bad_neg,
            )

    for bad_huge in (MAX_BIGQUERY_INT64 + 1, 10**5000):
        with pytest.raises(
            ValueError, match=f"maximum_bytes_billed must be <= {MAX_BIGQUERY_INT64}"
        ):
            client.read_query("SELECT 1", maximum_bytes_billed=bad_huge)
        with pytest.raises(
            ValueError, match=f"maximum_bytes_billed must be <= {MAX_BIGQUERY_INT64}"
        ):
            client.scan_query(
                "SELECT 1",
                schema={"a": pl.Int64},
                maximum_bytes_billed=bad_huge,
            )
    mock_arrow_client.read_query.assert_not_called()

    # read_query forwards maximum_bytes_billed for both live and dry-run calls.
    mock_arrow_client.read_query.return_value = MagicMock()
    with patch("polars.DataFrame"):
        client.read_query("SELECT 1", maximum_bytes_billed=1_000_000)
        client.read_query("SELECT 1", dry_run=True, maximum_bytes_billed=0)
    assert (
        mock_arrow_client.read_query.call_args_list[0].kwargs["maximum_bytes_billed"]
        == 1_000_000
    )
    assert (
        mock_arrow_client.read_query.call_args_list[1].kwargs["maximum_bytes_billed"]
        == 0
    )

    # scan_query does NOT forward maximum_bytes_billed on its construction-time
    # dry run (so pruning can shrink the query before execution), DOES surface it
    # in explain_detail, and DOES forward it at .collect() time.
    mock_arrow_client.read_query.reset_mock()
    dry_run_df = pl.DataFrame(
        {
            "bigquerySchema": ['{"fields": [{"name": "a", "type": "INT64"}]}'],
            "statementType": ["SELECT"],
            "location": ["US"],
        }
    )
    mock_arrow_client.read_query.return_value = dry_run_df
    lf = client.scan_query(
        "SELECT a FROM t",
        maintain_order=False,
        maximum_bytes_billed=500_000,
    )
    assert mock_arrow_client.read_query.call_args_list[0].kwargs["dry_run"] is True
    assert (
        mock_arrow_client.read_query.call_args_list[0].kwargs["maximum_bytes_billed"]
        is None
    )
    explain = lf.explain()
    assert "maximum_bytes_billed=500000" in explain

    mock_arrow_client.read_query.reset_mock()
    mock_arrow_client.read_query.return_value = MagicMock(name="stream")
    with patch("polars.scan_arrow_c_stream", return_value=pl.LazyFrame({"a": [1]})):
        assert lf.collect().to_dict(as_series=False) == {"a": [1]}
    mock_arrow_client.read_query.assert_called_once_with(
        "SELECT a FROM t",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=False,
        use_query_cache=None,
        maximum_bytes_billed=500_000,
    )

    # Default maximum_bytes_billed=None omits it from explain_detail.
    lf_default = client.scan_query("SELECT a FROM t", schema={"a": pl.Int64})
    assert "maximum_bytes_billed" not in lf_default.explain()


def test_client_maximum_bytes_billed_wire_serialization():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured_bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            captured_bodies.append(body)
            if body.get("dryRun"):
                resp = {
                    "jobReference": {
                        "projectId": "test-proj",
                        "location": "US",
                    },
                    "totalBytesProcessed": "16",
                    "statementType": "SELECT",
                    "schema": {"fields": [{"name": "one", "type": "INT64"}]},
                }
                status = 200
            else:
                resp = {
                    "error": {
                        "code": 400,
                        "message": "Query exceeded limit for bytes billed: 500000.",
                        "status": "INVALID_ARGUMENT",
                    }
                }
                status = 400
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 2.0
    port = server.server_port

    def serve_three():
        for _ in range(3):
            server.handle_request()

    server_thread = threading.Thread(target=serve_three, daemon=True)
    server_thread.start()

    try:
        creds = lambda: ({"bearer_token": "fake-tok"}, None)
        client = Client(quota_project_id="test-proj", credentials_provider=creds)
        client._arrow_client = arrow_bigquery.Client(
            quota_project_id="test-proj",
            credentials_provider=creds,
            bq_endpoint=f"http://127.0.0.1:{port}",
        )
        # 1. Explicit dry-run read_query forwards maximumBytesBilled on the wire.
        client.read_query(
            "SELECT 1 AS one",
            dry_run=True,
            maximum_bytes_billed=1_048_576,
        )
        # 2. scan_query schema discovery omits maximumBytesBilled on its dry run.
        lf = client.scan_query(
            "SELECT 1 AS one",
            maintain_order=False,
            maximum_bytes_billed=500_000,
        )
        assert "maximum_bytes_billed=500000" in lf.explain()
        # 3. scan_query execution (.collect()) forwards maximumBytesBilled on the wire.
        with pytest.raises(
            (BigQueryError, pl.exceptions.ComputeError),
            match="exceeded limit for bytes billed",
        ):
            lf.collect()
    finally:
        server_thread.join(timeout=2.0)
        server.server_close()

    assert len(captured_bodies) == 3
    assert captured_bodies[0].get("maximumBytesBilled") == "1048576"
    assert captured_bodies[0].get("dryRun") is True
    assert "maximumBytesBilled" not in captured_bodies[1]
    assert captured_bodies[1].get("dryRun") is True
    assert captured_bodies[2].get("maximumBytesBilled") == "500000"
    assert not captured_bodies[2].get("dryRun")
