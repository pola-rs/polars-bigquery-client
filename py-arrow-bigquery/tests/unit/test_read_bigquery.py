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


def test_client_read_bigquery_propagates_errors(mock_rust_client):
    mock_rust_client.read_table.side_effect = Exception("Rust error")

    client = Client(quota_project_id="q")
    with pytest.raises(Exception, match="Rust error"):
        client.read_table(table=BigQueryTableId("p", "d", "t"))


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
