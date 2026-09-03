import _thread
import threading
import time
from unittest.mock import MagicMock, patch

import nanoarrow
import pytest
from arrow_bigquery import (
    Client,
    __version__,
    _native,
)
from arrow_bigquery._read_bigquery import _get_user_agent, _parse_table_id


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


def test_parse_table_id_valid_string():
    assert _parse_table_id("proj.ds.tab") == "proj.ds.tab"


def test_parse_table_id_with_colon():
    assert _parse_table_id("google.com:project.ds.tab") == "google.com:project.ds.tab"


def test_parse_table_id_table_reference():
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"
    assert _parse_table_id(mock_ref) == "p.d.t"


def test_parse_table_id_table_object():
    mock_table = MagicMock()
    mock_table.project = "proj-obj"
    mock_table.dataset_id = "ds-obj"
    mock_table.table_id = "tab-obj"
    assert _parse_table_id(mock_table) == "proj-obj.ds-obj.tab-obj"


def test_parse_table_id_invalid_format():
    with pytest.raises(ValueError, match="Invalid table ID"):
        _parse_table_id("just_a_string")
    with pytest.raises(TypeError, match="BigLake tables are not supported yet"):
        _parse_table_id("too.many.parts.here")


def test_parse_table_id_invalid_type():
    with pytest.raises(TypeError, match="Expected table_id to be a string"):
        _parse_table_id(123)


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


def test_client_read_bigquery_calls_rust_with_parsed_id(mock_rust_client):
    placeholder = object()
    mock_rust_client.read_table.return_value = placeholder

    client = Client(quota_project_id="q")
    result = client.read_table(table="my-project.my_dataset.my_table")

    mock_rust_client.read_table.assert_called_once_with(
        "my-project.my_dataset.my_table",
        maintain_order=False,
    )
    assert result is placeholder


def test_client_read_bigquery_handles_bigquery_objects(mock_rust_client):
    mock_rust_client.read_table.return_value = MagicMock()
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"

    client = Client(quota_project_id="q")
    client.read_table(table=mock_ref)

    mock_rust_client.read_table.assert_called_once_with("p.d.t", maintain_order=False)


def test_client_read_bigquery_propagates_errors(mock_rust_client):
    mock_rust_client.read_table.side_effect = Exception("Rust error")

    client = Client(quota_project_id="q")
    with pytest.raises(Exception, match="Rust error"):
        client.read_table(table="p.d.t")


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
