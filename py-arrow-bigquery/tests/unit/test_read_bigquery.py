import _thread
import datetime
import threading
import time
from unittest.mock import MagicMock, patch

import nanoarrow
import pytest
from arrow_bigquery import (
    BigQueryTableId,
    Client,
    __version__,
    _native,
)
from arrow_bigquery._read_bigquery import _get_user_agent


@pytest.fixture
def mock_rust_client():
    with patch("arrow_bigquery._native.Client") as mocked_cls:
        mock_instance = MagicMock()
        mocked_cls.return_value = mock_instance
        yield mock_instance


def test_get_user_agent():
    assert _get_user_agent(None) == f"arrow-bigquery/{__version__}"
    assert _get_user_agent("") == f"arrow-bigquery/{__version__}"
    assert (
        _get_user_agent("custom-extension/1.0")
        == f"arrow-bigquery/{__version__} custom-extension/1.0"
    )


def test_bigquery_table_id_str_and_properties():
    table_id = BigQueryTableId("proj", "ds", "tab")
    assert str(table_id) == "proj.ds.tab"
    assert table_id.project == "proj"
    assert table_id.project_id == "proj"
    assert table_id.dataset_id == "ds"
    assert table_id.table_id == "tab"


def test_bigquery_table_id_from_string():
    id1 = BigQueryTableId.from_string("proj.ds.tab")
    assert id1 == BigQueryTableId("proj", "ds", "tab")

    id2 = BigQueryTableId.from_str("google.com:proj.ds.tab")
    assert id2 == BigQueryTableId("google.com:proj", "ds", "tab")

    id3 = BigQueryTableId.from_string("too.many.parts.here")
    assert id3 == BigQueryTableId("too", "many.parts", "here")

    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId.from_string("just_a_string")

    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId.from_string("`proj.ds.tab`")

    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId.from_string("proj..tab")


def test_client_init_passes_credentials_provider():
    mock_cp = MagicMock()
    with patch("arrow_bigquery._native.Client") as mock_native_cls:
        client = Client(quota_project_id="test-proj", credentials_provider=mock_cp)
        assert client.quota_project_id == "test-proj"
        mock_native_cls.assert_called_once_with(
            quota_project_id="test-proj",
            credentials_provider=mock_cp,
            user_agent=f"arrow-bigquery/{__version__}",
            bq_endpoint=None,
            bqstorage_endpoint=None,
        )


def test_client_init_passes_endpoints():
    with patch("arrow_bigquery._native.Client") as mock_native_cls:
        Client(
            quota_project_id="test-proj",
            bq_endpoint="https://custom.bq.endpoint.com",
            bqstorage_endpoint="https://custom.bqstorage.endpoint.com",
        )
        mock_native_cls.assert_called_once_with(
            quota_project_id="test-proj",
            credentials_provider=None,
            user_agent=f"arrow-bigquery/{__version__}",
            bq_endpoint="https://custom.bq.endpoint.com",
            bqstorage_endpoint="https://custom.bqstorage.endpoint.com",
        )


def test_client_init_requires_quota_project_id():
    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'quota_project_id'"
    ):
        Client()


def test_client_read_bigquery_calls_rust(mock_rust_client):
    placeholder = object()
    mock_rust_client.read_table.return_value = placeholder

    table = BigQueryTableId("my-project", "my_dataset", "my_table")
    client = Client(quota_project_id="q")
    result = client.read_table(table=table)

    mock_rust_client.read_table.assert_called_once_with(
        table,
        arrow_buffer_compression="lz4frame",
        maintain_order=False,
        max_stream_count=None,
        row_restriction="",
        sample_percentage=None,
        selected_fields=None,
        snapshot_time=None,
    )
    assert result is placeholder


def test_client_read_bigquery_passes_additional_parameters(mock_rust_client):
    placeholder = object()
    mock_rust_client.read_table.return_value = placeholder

    table = BigQueryTableId("my-project", "my_dataset", "my_table")
    snapshot_dt = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    client = Client(quota_project_id="q")
    result = client.read_table(
        table=table,
        arrow_buffer_compression="zstd",
        maintain_order=True,
        max_stream_count=4,
        row_restriction="foo > 1",
        sample_percentage=10.0,
        selected_fields=["foo", "bar"],
        snapshot_time=snapshot_dt,
    )

    mock_rust_client.read_table.assert_called_once_with(
        table,
        arrow_buffer_compression="zstd",
        maintain_order=True,
        max_stream_count=4,
        row_restriction="foo > 1",
        sample_percentage=10.0,
        selected_fields=["foo", "bar"],
        snapshot_time=snapshot_dt,
    )
    assert result is placeholder


def test_client_read_bigquery_with_str_table(mock_rust_client):
    placeholder = object()
    mock_rust_client.read_table.return_value = placeholder

    client = Client(quota_project_id="q")
    result = client.read_table(table="my-project.my_dataset.my_table")

    mock_rust_client.read_table.assert_called_once_with(
        BigQueryTableId("my-project", "my_dataset", "my_table"),
        arrow_buffer_compression="lz4frame",
        maintain_order=False,
        max_stream_count=None,
        row_restriction="",
        sample_percentage=None,
        selected_fields=None,
        snapshot_time=None,
    )
    assert result is placeholder


def test_client_read_bigquery_with_table_object(mock_rust_client):
    placeholder = object()
    mock_rust_client.read_table.return_value = placeholder

    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"

    client = Client(quota_project_id="q")
    result = client.read_table(table=mock_ref)

    mock_rust_client.read_table.assert_called_once_with(
        BigQueryTableId("p", "d", "t"),
        arrow_buffer_compression="lz4frame",
        maintain_order=False,
        max_stream_count=None,
        row_restriction="",
        sample_percentage=None,
        selected_fields=None,
        snapshot_time=None,
    )
    assert result is placeholder


def test_client_read_bigquery_invalid_table_type(mock_rust_client):
    client = Client(quota_project_id="q")
    with pytest.raises(TypeError, match="Expected table_id to be a string"):
        client.read_table(table=123)


def test_client_read_bigquery_invalid_table_str(mock_rust_client):
    client = Client(quota_project_id="q")
    with pytest.raises(ValueError, match="Invalid table ID"):
        client.read_table(table="just_a_string")
    with pytest.raises(ValueError, match="Invalid table ID"):
        client.read_table(table="proj.ds.tab?fields=etag#frag")

    class FakeTable:
        project = "proj/../../evil"
        dataset_id = "ds"
        table_id = "tab?fields=all#"

    with pytest.raises(ValueError, match="Invalid table ID"):
        client.read_table(table=FakeTable())
    mock_rust_client.read_table.assert_not_called()


def test_client_read_bigquery_propagates_errors(mock_rust_client):
    mock_rust_client.read_table.side_effect = Exception("Rust error")

    client = Client(quota_project_id="q")
    with pytest.raises(Exception, match="Rust error"):
        client.read_table(table=BigQueryTableId("p", "d", "t"))


def test_native_client_default_maintain_order_signatures():
    import inspect

    read_table_sig = inspect.signature(_native.Client.read_table)
    assert read_table_sig.parameters["maintain_order"].default is False

    read_query_sig = inspect.signature(_native.Client.read_query)
    assert read_query_sig.parameters["maintain_order"].default is True


def test_client_read_query_calls_rust(mock_rust_client):
    placeholder = object()
    mock_rust_client.read_query.return_value = placeholder

    client = Client(quota_project_id="q")
    result = client.read_query("SELECT 1")

    mock_rust_client.read_query.assert_called_once_with(
        "SELECT 1",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=True,
        use_query_cache=None,
    )
    assert result is placeholder


def test_client_read_query_passes_additional_parameters(mock_rust_client):
    placeholder = object()
    mock_rust_client.read_query.return_value = placeholder

    client = Client(quota_project_id="q")
    result = client.read_query(
        "SELECT 1",
        arrow_buffer_compression="zstd",
        dry_run=True,
        maintain_order=False,
        use_query_cache=False,
    )

    mock_rust_client.read_query.assert_called_once_with(
        "SELECT 1",
        arrow_buffer_compression="zstd",
        dry_run=True,
        maintain_order=False,
        use_query_cache=False,
    )
    assert result is placeholder


def test_client_read_query_propagates_errors(mock_rust_client):
    mock_rust_client.read_query.side_effect = Exception("Rust query error")

    client = Client(quota_project_id="q")
    with pytest.raises(Exception, match="Rust query error"):
        client.read_query("SELECT 1")


def test_client_read_query_dry_run_and_read_with_mock_server():
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            assert self.path.startswith("/bigquery/v2/projects/test-proj/queries")
            assert body["query"] == "SELECT 1 AS one"
            assert body["useLegacySql"] is False
            assert body.get("dryRun") is True

            resp = {
                "jobReference": {
                    "projectId": "test-proj",
                    "location": "US",
                },
                "totalBytesProcessed": "42",
                "cacheHit": False,
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
        client = Client(
            quota_project_id="test-proj",
            credentials_provider=lambda: ({"bearer_token": "fake-tok"}, None),
            bq_endpoint=f"http://127.0.0.1:{port}",
        )
        exporter = client.read_query("SELECT 1 AS one", dry_run=True)
    finally:
        server_thread.join()
        server.server_close()

    arr = nanoarrow.ArrayStream(exporter).read_all()
    assert len(arr) == 1
    col_names = [field.name for field in arr.schema.fields]
    assert col_names == [
        "bigquerySchema",
        "projectId",
        "jobId",
        "location",
        "totalBytesProcessed",
        "cacheHit",
        "statementType",
        "creationTime",
    ]
    assert arr.child(4).to_pylist() == [42]

    # Consuming a second time raises ValueError
    with pytest.raises(ValueError, match="Stream already consumed"):
        nanoarrow.c_array_stream(exporter)


def test_receiver_iterator_interrupt():
    exporter = _native._create_test_exporter()

    interrupted = False

    def trigger_interrupt():
        # Wait a bit to ensure we are blocking in the iterator
        time.sleep(0.3)
        _thread.interrupt_main()

    # Start the interrupt thread
    thread = threading.Thread(target=trigger_interrupt)
    thread.start()

    try:
        # Consuming the stream will block because the channel is empty and kept open.
        # The interrupt should break it.
        stream = nanoarrow.c_array_stream(exporter)
        for _ in stream:
            pass
    except BaseException as err:
        if (
            isinstance(err, KeyboardInterrupt)
            or isinstance(getattr(err, "__cause__", None), KeyboardInterrupt)
            or isinstance(getattr(err, "__context__", None), KeyboardInterrupt)
            or "Python interrupt" in str(err)
        ):
            interrupted = True
        else:
            raise
    finally:
        thread.join()

    assert interrupted, "The C-stream consumption was not interrupted"


def test_exporter_drop_direct():
    from arrow_bigquery._testing import run_exporter_drop_test

    assert run_exporter_drop_test(), "The background task was not aborted within 1s"


def test_exporter_drop_after_stream_created():
    from arrow_bigquery._testing import run_exporter_drop_after_stream_created_test

    assert run_exporter_drop_after_stream_created_test(), (
        "The background task was not aborted within 1s"
    )


def test_bigquery_error_hierarchy_and_empty_quota_project():
    from arrow_bigquery import BigQueryError
    from arrow_bigquery.exceptions import BigQueryError as ExceptionsBigQueryError

    assert BigQueryError is ExceptionsBigQueryError
    assert issubclass(BigQueryError, RuntimeError)

    with pytest.raises(ValueError, match="quota_project_id is required"):
        Client(quota_project_id="   ")


def test_client_close_and_context_manager(mock_rust_client):
    with Client(quota_project_id="q") as client:
        assert client.quota_project_id == "q"

    # Calling close() repeatedly is idempotent.
    client.close()

    with pytest.raises(RuntimeError, match="closed"):
        client.read_table("p.d.t")
    with pytest.raises(RuntimeError, match="closed"):
        client.read_query("SELECT 1")


def test_client_read_query_maximum_bytes_billed(mock_rust_client):
    client = Client(quota_project_id="q")
    client.read_query("SELECT 1", maximum_bytes_billed=10_000)
    mock_rust_client.read_query.assert_called_once_with(
        "SELECT 1",
        arrow_buffer_compression="lz4frame",
        dry_run=False,
        maintain_order=True,
        use_query_cache=None,
        maximum_bytes_billed=10_000,
    )

    max_i64 = (1 << 63) - 1
    client.read_query("SELECT 1", maximum_bytes_billed=max_i64)
    assert (
        mock_rust_client.read_query.call_args.kwargs["maximum_bytes_billed"] == max_i64
    )

    with pytest.raises(TypeError, match="maximum_bytes_billed must be an int or None"):
        client.read_query("SELECT 1", maximum_bytes_billed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="maximum_bytes_billed must be non-negative"):
        client.read_query("SELECT 1", maximum_bytes_billed=-1)
    with pytest.raises(ValueError, match=f"maximum_bytes_billed must be <= {max_i64}"):
        client.read_query("SELECT 1", maximum_bytes_billed=1 << 63)


def test_client_dry_run_forwards_maximum_bytes_billed_and_worker_thread():
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received_bodies: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            received_bodies.append(body)
            resp = {
                "jobReference": {"projectId": "test-proj", "location": "US"},
                "totalBytesProcessed": "16",
                "cacheHit": False,
                "statementType": "SELECT",
                "creationTime": "1767037135155",
                "schema": {"fields": [{"name": "x", "type": "INT64"}]},
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
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        client = Client(
            quota_project_id="test-proj",
            credentials_provider=lambda: ({"bearer_token": "tok"}, None),
            bq_endpoint=f"http://127.0.0.1:{port}",
        )
        # Dry run on the main thread forwards maximumBytesBilled.
        exporter = client.read_query(
            "SELECT 1 AS x", dry_run=True, maximum_bytes_billed=1024
        )
        assert len(nanoarrow.ArrayStream(exporter).read_all()) == 1

        # Dry run on a worker (non-main) thread exercises the direct wait path.
        from concurrent.futures import ThreadPoolExecutor

        def run_in_worker() -> int:
            exp = client.read_query("SELECT 1 AS x", dry_run=True)
            return len(nanoarrow.ArrayStream(exp).read_all())

        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(run_in_worker).result(timeout=5.0) == 1
    finally:
        server.shutdown()
        server.server_close()

    assert len(received_bodies) == 2
    assert received_bodies[0].get("dryRun") is True
    assert str(received_bodies[0].get("maximumBytesBilled")) == "1024"
    assert received_bodies[1].get("dryRun") is True
    assert "maximumBytesBilled" not in received_bodies[1]


def test_read_query_interruptible_during_wait():
    import socket

    # Hold a TCP connection open without responding so `jobs.query` blocks waiting.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    accepted: list[socket.socket] = []
    stop_event = threading.Event()

    def accept_and_hold():
        try:
            conn, _ = srv.accept()
            accepted.append(conn)
            stop_event.wait(timeout=10.0)
        except OSError:
            pass

    server_thread = threading.Thread(target=accept_and_hold, daemon=True)
    server_thread.start()

    def send_interrupt():
        time.sleep(0.25)
        _thread.interrupt_main()

    irq_thread = threading.Thread(target=send_interrupt, daemon=True)
    client = Client(
        quota_project_id="test-proj",
        credentials_provider=lambda: ({"bearer_token": "tok"}, None),
        bq_endpoint=f"http://127.0.0.1:{port}",
    )
    irq_thread.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            client.read_query("SELECT 1", dry_run=True)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, (
            f"read_query took too long to notice KeyboardInterrupt: {elapsed:.2f}s"
        )
    finally:
        stop_event.set()
        for conn in accepted:
            conn.close()
        srv.close()
        irq_thread.join(timeout=1.0)
        server_thread.join(timeout=1.0)
