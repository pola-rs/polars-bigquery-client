import warnings
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import arrow_bigquery
import polars as pl
import polars_bigquery
import pytest
from polars_bigquery import (
    Client,
    __version__,
)
from polars_bigquery._read_bigquery import _get_user_agent
from polars_bigquery.core.types import (
    GEOGRAPHY_EXTENSION_METADATA,
    GEOGRAPHY_EXTENSION_NAME,
    JSON_EXTENSION_NAME,
)
from polars_bigquery.exceptions import BigQueryError, UnsupportedTypeWarning


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
        )
        mock_df_cls.assert_called_once_with(mock_exporter)
        assert result is mock_df


def test_client_read_query_propagates_errors(mock_arrow_client):
    mock_arrow_client.read_query.side_effect = Exception("Rust query error")

    client = Client(quota_project_id="q")
    with pytest.raises(Exception, match="Rust query error"):
        client.read_query(query="SELECT 1")


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
    port = server.server_port
    server_thread = threading.Thread(target=server.handle_request)
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
        server_thread.join()
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


@pytest.mark.parametrize("dry_run", [False, True])
def test_client_read_query_invalid_sql_raises_polars_bigquery_error(dry_run):
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
    port = server.server_port
    server_thread = threading.Thread(target=server.handle_request)
    server_thread.start()

    try:

        def creds():
            return ({"bearer_token": "fake-tok"}, None)

        with Client(quota_project_id="test-proj", credentials_provider=creds) as client:
            client._arrow_client = arrow_bigquery.Client(
                quota_project_id="test-proj",
                credentials_provider=creds,
                bq_endpoint=f"http://127.0.0.1:{port}",
            )
            with pytest.raises(BigQueryError, match="Syntax error"):
                client.read_query("SELECT BROKEN", dry_run=dry_run)
    finally:
        server_thread.join()
        server.server_close()


class _FakeArrowStreamExporter:
    """Single-use Arrow C Stream exporter backed by a real Polars DataFrame."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def __arrow_c_stream__(self, requested_schema=None):
        return self._df.__arrow_c_stream__(requested_schema=requested_schema)


def test_scan_table_emits_unsupported_type_warning_and_formats_paths(mock_arrow_client):
    metadata = {
        "schema": {
            "fields": [
                {"name": "id", "type": "INT64"},
                {"name": "dur", "type": "INTERVAL"},
                {
                    "name": "meta",
                    "type": "STRUCT",
                    "fields": [
                        {"name": "ok", "type": "INT64"},
                        {"name": "bad", "type": "BIGNUMERIC"},
                    ],
                },
            ]
        }
    }
    with patch(
        "polars_bigquery.core.bigquery_rest.get_table_metadata", return_value=metadata
    ):
        client = Client(quota_project_id="q")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lf = client.scan_table("p.d.t")

    assert lf.collect_schema() == pl.Schema({"id": pl.Int64()})
    assert len(caught) == 1
    w = caught[0]
    assert issubclass(w.category, UnsupportedTypeWarning)
    assert w.filename == __file__
    msg = str(w.message)
    assert "Skipping 2 unsupported column(s)" in msg
    assert "dur (INTERVAL)" in msg
    assert "meta (BIGNUMERIC at meta.bad)" in msg
    assert "client.read_query()" in msg


def test_scan_table_unsupported_type_warning_caps_at_five_entries(mock_arrow_client):
    metadata = {
        "schema": {
            "fields": [
                {"name": "id", "type": "INT64"},
                *[{"name": f"bad_{i}", "type": "INTERVAL"} for i in range(7)],
            ]
        }
    }
    with patch(
        "polars_bigquery.core.bigquery_rest.get_table_metadata", return_value=metadata
    ):
        client = Client(quota_project_id="q")
        with pytest.warns(UnsupportedTypeWarning) as record:
            lf = client.scan_table("p.d.t")

    assert lf.collect_schema() == pl.Schema({"id": pl.Int64()})
    msg = str(record[0].message)
    assert "Skipping 7 unsupported column(s)" in msg
    for i in range(5):
        assert f"bad_{i} (INTERVAL)" in msg
    assert "bad_5" not in msg
    assert "... and 2 more" in msg
    assert "client.read_query()" in msg


def test_scan_table_all_unsupported_non_partitioned_raises_value_error_without_warning(
    mock_arrow_client,
):
    metadata = {
        "schema": {
            "fields": [
                {"name": "dur", "type": "INTERVAL"},
                {"name": "amt", "type": "BIGNUMERIC"},
            ]
        }
    }
    with patch(
        "polars_bigquery.core.bigquery_rest.get_table_metadata", return_value=metadata
    ):
        client = Client(quota_project_id="q")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(
                ValueError, match="has no supported columns to scan"
            ) as exc_info:
                client.scan_table("p.d.t")

    assert caught == []
    msg = str(exc_info.value)
    assert "dur (INTERVAL)" in msg
    assert "amt (BIGNUMERIC)" in msg
    assert "client.read_query()" in msg


@pytest.mark.parametrize(
    ("partition_type", "expected_cols", "expected_fallback_field"),
    [
        ("DAY", ["_PARTITIONDATE", "_PARTITIONTIME"], "_PARTITIONDATE"),
        ("HOUR", ["_PARTITIONTIME"], "_PARTITIONTIME"),
    ],
)
def test_scan_table_all_unsupported_physical_retains_ingestion_partition_pseudo_columns(
    mock_arrow_client, partition_type, expected_cols, expected_fallback_field
):
    metadata = {
        "schema": {"fields": [{"name": "dur", "type": "INTERVAL"}]},
        "timePartitioning": {"type": partition_type},
    }
    if expected_fallback_field == "_PARTITIONDATE":
        stream_df = pl.DataFrame(
            {"_PARTITIONDATE": [date(2024, 1, 1), date(2024, 1, 2)]}
        )
    else:
        stream_df = pl.DataFrame(
            {
                "_PARTITIONTIME": pl.Series(
                    [
                        datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                    ],
                    dtype=pl.Datetime("us", "UTC"),
                )
            }
        )
    mock_arrow_client.read_table.return_value = _FakeArrowStreamExporter(stream_df)

    with patch(
        "polars_bigquery.core.bigquery_rest.get_table_metadata", return_value=metadata
    ):
        client = Client(quota_project_id="q")
        with pytest.warns(UnsupportedTypeWarning, match=r"dur \(INTERVAL\)"):
            lf = client.scan_table("p.d.t")

    assert lf.collect_schema().names() == expected_cols
    count_df = lf.select(pl.len()).collect()
    assert count_df.item() == 2
    mock_arrow_client.read_table.assert_called_once_with(
        arrow_bigquery.BigQueryTableId("p", "d", "t"),
        maintain_order=False,
        selected_fields=[expected_fallback_field],
        row_restriction="",
    )


@pytest.mark.parametrize("use_explicit_schema", [False, True])
def test_scan_table_empty_schema_short_circuits_without_read_table(
    mock_arrow_client, use_explicit_schema
):
    with patch(
        "polars_bigquery.core.bigquery_rest.get_table_metadata", return_value={}
    ) as mock_get_metadata:
        client = Client(quota_project_id="q")
        lf = (
            client.scan_table("p.d.t", schema={})
            if use_explicit_schema
            else client.scan_table("p.d.t")
        )
        assert lf.collect_schema() == pl.Schema({})
        df = lf.collect()

    assert df.shape == (0, 0)
    mock_arrow_client.read_table.assert_not_called()
    if use_explicit_schema:
        mock_get_metadata.assert_not_called()
    else:
        mock_get_metadata.assert_called_once()


def test_scan_table_zero_column_projection_picks_narrowest_column_and_preserves_height(
    mock_arrow_client,
):
    metadata = {
        "schema": {
            "fields": [
                {"name": "payload", "type": "STRING"},
                {"name": "num", "type": "INT64"},
                {"name": "dt", "type": "DATE"},
                {"name": "flag", "type": "BOOL"},
                {"name": "flag2", "type": "BOOL"},
            ]
        }
    }
    full_df = pl.DataFrame(
        {
            "payload": ["a", "b", "c", "d"],
            "num": [1, 20, 30, 4],
            "dt": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ],
            "flag": [True, False, True, True],
            "flag2": [False, True, False, True],
        }
    )

    def fake_read_table(table_ref, *, selected_fields, **kwargs):
        return _FakeArrowStreamExporter(full_df.select(selected_fields))

    mock_arrow_client.read_table.side_effect = fake_read_table

    with (
        patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata",
            return_value=metadata,
        ),
        patch(
            "polars.io.plugins.register_io_source",
            wraps=pl.io.plugins.register_io_source,
        ) as mock_register,
    ):
        client = Client(quota_project_id="q")
        lf = client.scan_table("p.d.t")
        io_source = mock_register.call_args.kwargs["io_source"]

        # 1. Direct zero-column invocation picks narrowest column (tie-broken to first BOOL: "flag")
        #    and preserves row height without collapsing to 0 rows.
        batches = list(io_source([], None, None, None))
        assert sum(b.height for b in batches) == 4
        assert all(b.width == 0 for b in batches)
        assert mock_arrow_client.read_table.call_args_list[0].kwargs[
            "selected_fields"
        ] == ["flag"]

        # 2. Direct zero-column invocation with n_rows limit preserves limited height.
        limited_batches = list(io_source([], None, 2, None))
        assert sum(b.height for b in limited_batches) == 2
        assert all(b.width == 0 for b in limited_batches)
        assert mock_arrow_client.read_table.call_args_list[1].kwargs[
            "selected_fields"
        ] == ["flag"]

        # 3. Direct zero-column invocation with predicate selects only the predicate column.
        filtered_batches = list(io_source([], pl.col("num") > 10, None, None))
        assert sum(b.height for b in filtered_batches) == 2
        assert all(b.width == 0 for b in filtered_batches)
        assert mock_arrow_client.read_table.call_args_list[2].kwargs[
            "selected_fields"
        ] == ["num"]

        # 4. End-to-end Polars LazyFrame row counts with limit and filter
        assert lf.select(pl.len()).collect().item() == 4
        assert lf.limit(2).select(pl.len()).collect().item() == 2
        assert lf.filter(pl.col("num") > 10).select(pl.len()).collect().item() == 2


def test_scan_table_non_mocked_arrow_c_stream_utc_and_extension_types(
    mock_arrow_client, monkeypatch
):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    metadata = {
        "schema": {
            "fields": [
                {"name": "ts", "type": "TIMESTAMP"},
                {"name": "j", "type": "JSON"},
                {"name": "g", "type": "GEOGRAPHY"},
                {"name": "j_arr", "type": "JSON", "mode": "REPEATED"},
                {"name": "g_arr", "type": "GEOGRAPHY", "mode": "REPEATED"},
            ]
        }
    }
    json_arr_dtype = pl.Extension(JSON_EXTENSION_NAME, pl.List(pl.String()))
    geo_arr_dtype = pl.Extension(
        GEOGRAPHY_EXTENSION_NAME,
        pl.List(pl.String()),
        GEOGRAPHY_EXTENSION_METADATA,
    )
    source_df = pl.DataFrame(
        {
            "ts": pl.Series(
                [datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)],
                dtype=pl.Datetime("us", "UTC"),
            ),
            "j": pl.Series(['{"k": 1}'], dtype=pl.String()).ext.to(
                polars_bigquery.dtypes.JSON
            ),
            "g": pl.Series(["POINT(1 2)"], dtype=pl.String()).ext.to(
                polars_bigquery.dtypes.GEOGRAPHY
            ),
            "j_arr": pl.Series(
                [['{"a": 1}', "null"]], dtype=pl.List(pl.String())
            ).ext.to(json_arr_dtype),
            "g_arr": pl.Series([["POINT(0 0)"]], dtype=pl.List(pl.String())).ext.to(
                geo_arr_dtype
            ),
        }
    )
    mock_arrow_client.read_table.return_value = _FakeArrowStreamExporter(source_df)

    with patch(
        "polars_bigquery.core.bigquery_rest.get_table_metadata", return_value=metadata
    ):
        client = Client(quota_project_id="q")
        lf = client.scan_table("p.d.t")
        assert lf.collect_schema() == source_df.schema
        collected = lf.collect()

    assert collected.schema == source_df.schema
    assert collected.equals(source_df)


def test_scan_table_explicit_schema_with_dtypes_and_mismatch_error(
    mock_arrow_client, monkeypatch
):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    source_df = pl.DataFrame(
        {
            "ts": pl.Series(
                [datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)],
                dtype=pl.Datetime("us", "UTC"),
            ),
            "j": pl.Series(['{"k": 1}'], dtype=pl.String()).ext.to(
                polars_bigquery.dtypes.JSON
            ),
            "g": pl.Series(["POINT(1 2)"], dtype=pl.String()).ext.to(
                polars_bigquery.dtypes.GEOGRAPHY
            ),
        }
    )

    client = Client(quota_project_id="q")

    # Matching explicit schema succeeds through real Arrow C stream FFI
    mock_arrow_client.read_table.return_value = _FakeArrowStreamExporter(source_df)
    lf_ok = client.scan_table(
        "p.d.t",
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "j": polars_bigquery.dtypes.JSON,
            "g": polars_bigquery.dtypes.GEOGRAPHY,
        },
    )
    assert lf_ok.collect().equals(source_df)

    # Mismatched explicit schema (plain pl.String for JSON) raises native Polars SchemaError
    mock_arrow_client.read_table.return_value = _FakeArrowStreamExporter(
        source_df.select("j")
    )
    lf_bad = client.scan_table("p.d.t", schema={"j": pl.String()})
    with pytest.raises(pl.exceptions.SchemaError, match="google:sqlType:json"):
        lf_bad.collect()


def test_scan_table_threads_authoritative_bq_types_under_mode_b(
    mock_arrow_client, monkeypatch
):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    metadata = {
        "schema": {
            "fields": [
                {"name": "id", "type": "INT64"},
                {"name": "j", "type": "JSON"},
            ]
        }
    }
    full_df = pl.DataFrame({"id": [1, 2], "j": ['{"a": 1}', None]})
    mock_arrow_client.read_table.side_effect = (
        lambda table_ref, *, selected_fields, **kwargs: _FakeArrowStreamExporter(
            full_df.select(selected_fields)
        )
    )

    pl.register_extension_type(JSON_EXTENSION_NAME, as_storage=True)
    try:
        with patch(
            "polars_bigquery.core.bigquery_rest.get_table_metadata",
            return_value=metadata,
        ):
            client = Client(quota_project_id="q")
            lf = client.scan_table("p.d.t")

            # Under Mode B, j is pl.String in polars_schema, yet bq_types keeps it JSON
            assert lf.collect_schema() == pl.Schema(
                {"id": pl.Int64(), "j": pl.String()}
            )

            # String method on JSON column must NOT push down STARTS_WITH to BigQuery,
            # while sibling INT64 filter still pushes down cleanly.
            df = lf.filter(
                pl.col("j").str.starts_with("{") & (pl.col("id") == 1)
            ).collect()
            assert df.to_dict(as_series=False) == {"id": [1], "j": ['{"a": 1}']}
            assert (
                mock_arrow_client.read_table.call_args_list[0].kwargs["row_restriction"]
                == "(`id` = 1)"
            )

            # Null check on JSON column pushes down even under Mode B
            df_nn = lf.filter(pl.col("j").is_not_null()).collect()
            assert df_nn.height == 1
            assert (
                mock_arrow_client.read_table.call_args_list[1].kwargs["row_restriction"]
                == "(`j` IS NOT NULL)"
            )
    finally:
        pl.unregister_extension_type(JSON_EXTENSION_NAME)


def test_scan_table_explicit_schema_derives_compiler_types(
    mock_arrow_client, monkeypatch
):
    monkeypatch.delenv("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", raising=False)
    source_df = pl.DataFrame(
        {
            "id": [1, 2],
            "j": pl.Series(['{"a": 1}', None], dtype=pl.String()).ext.to(
                polars_bigquery.dtypes.JSON
            ),
        }
    )
    mock_arrow_client.read_table.side_effect = (
        lambda table_ref, *, selected_fields, **kwargs: _FakeArrowStreamExporter(
            source_df.select(selected_fields)
        )
    )

    client = Client(quota_project_id="q")
    lf = client.scan_table(
        "p.d.t",
        schema={"id": pl.Int64, "j": polars_bigquery.dtypes.JSON},
    )

    df = lf.filter(pl.col("j").is_not_null() & (pl.col("id") >= 1)).collect()
    assert df.height == 1
    assert mock_arrow_client.read_table.call_args_list[0].kwargs["row_restriction"] in (
        "((`j` IS NOT NULL) AND (`id` >= 1))",
        "((`id` >= 1) AND (`j` IS NOT NULL))",
    )

    # .ext.storage().str.starts_with(...) falls back cleanly without breaking sibling pushdown
    df2 = lf.filter(
        pl.col("j").ext.storage().str.starts_with("{") & (pl.col("id") == 1)
    ).collect()
    assert df2.height == 1
    assert (
        mock_arrow_client.read_table.call_args_list[1].kwargs["row_restriction"]
        == "(`id` = 1)"
    )


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


def test_scan_table_udf_predicate_falls_back_cleanly(mock_arrow_client):
    source_df = pl.DataFrame({"id": [1, 2, 3]})
    mock_arrow_client.read_table.side_effect = (
        lambda table_ref, *, selected_fields, **kwargs: _FakeArrowStreamExporter(
            source_df.select(selected_fields)
        )
    )
    client = Client(quota_project_id="q")

    # Pass a predicate that cannot be JSON-serialized via Polars `meta.serialize()`
    # without requiring optional `cloudpickle` in `.collect()`.
    def is_gt_one(val: int) -> bool:
        return val > 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        udf_expr = pl.col("id").map_elements(is_gt_one, return_dtype=pl.Boolean)

    with patch(
        "polars_bigquery.core.compiler.predicate_to_row_restriction",
        wraps=polars_bigquery.core.compiler.predicate_to_row_restriction,
    ) as spy:
        # Exercise compiler.predicate_to_row_restriction directly with the UDF
        assert (
            polars_bigquery.core.compiler.predicate_to_row_restriction(
                udf_expr, schema={"id": pl.Int64}
            )
            == ""
        )
        # Also verify scan_table collects without raising if predicate pushdown
        # raises PolarsError during meta.serialize() (e.g. local UDF fallback).
        lf = client.scan_table("p.d.t", schema={"id": pl.Int64}).filter(
            pl.col("id") > 1
        )
        with patch.object(
            pl.expr.expr.ExprMetaNameSpace,
            "serialize",
            side_effect=pl.exceptions.ComputeError("cannot serialize UDF"),
        ):
            df = lf.collect()
    assert df["id"].to_list() == [2, 3]
    assert mock_arrow_client.read_table.call_args.kwargs["row_restriction"] == ""
    assert spy.call_count >= 1


def test_drop_columns_iter_closes_underlying_batch_iterator():
    from polars_bigquery._read_bigquery import _drop_columns_iter

    closed = False

    def batch_gen():
        nonlocal closed
        try:
            yield pl.DataFrame({"a": [1, 2]})
            yield pl.DataFrame({"a": [3, 4]})
        finally:
            closed = True

    it = _drop_columns_iter(batch_gen())
    first = next(it)
    assert first.shape == (2, 0)
    assert not closed
    it.close()
    assert closed
