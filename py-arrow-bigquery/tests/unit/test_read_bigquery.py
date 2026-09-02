import _thread
import datetime
import inspect
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


def test_native_client_read_table_signature():
    sig = inspect.signature(_native.Client.read_table)
    params = sig.parameters

    assert "table" in params
    assert params["table"].default is inspect.Parameter.empty

    assert "maintain_order" in params
    assert params["maintain_order"].default is False

    assert "snapshot_time" in params
    assert params["snapshot_time"].default is None

    assert "selected_fields" in params
    assert params["selected_fields"].default is None

    assert "row_restriction" in params
    assert params["row_restriction"].default == ""

    assert "arrow_buffer_compression" in params
    assert params["arrow_buffer_compression"].default == "lz4frame"

    assert "sample_percentage" in params
    assert params["sample_percentage"].default is None


def test_native_client_read_table_compression_validation():
    client = _native.Client(quota_project_id="test-quota-project")

    with pytest.raises(
        ValueError, match="got unexpected compression codec invalid_codec"
    ):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            arrow_buffer_compression="invalid_codec",
        )

    with pytest.raises(ValueError, match="got unexpected compression codec snappy"):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            arrow_buffer_compression="snappy",
        )

    for codec in ("unspecified", "lz4frame", "zstd"):
        with pytest.raises(RuntimeError):
            client.read_table(
                "projects/p/datasets/d/tables/t",
                arrow_buffer_compression=codec,
            )


def test_native_client_read_table_snapshot_time_validation():
    client = _native.Client(quota_project_id="test-quota-project")

    naive_dt = datetime.datetime(2023, 1, 1, 12, 0, 0)  # noqa: DTZ001
    with pytest.raises(ValueError, match="failed to extract snapshot_time"):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            snapshot_time=naive_dt,
        )

    with pytest.raises(TypeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            snapshot_time="not-a-datetime",
        )

    with pytest.raises(TypeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            snapshot_time=12345,
        )

    utc_dt = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    with pytest.raises(RuntimeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            snapshot_time=utc_dt,
        )


def test_native_client_read_table_type_validations():
    client = _native.Client(quota_project_id="test-quota-project")

    with pytest.raises(TypeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            selected_fields=123,
        )

    with pytest.raises(TypeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            selected_fields=[123],
        )

    with pytest.raises(TypeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            row_restriction=123,
        )

    with pytest.raises(TypeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            sample_percentage="fifty",
        )

    with pytest.raises(TypeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            maintain_order="true",
        )


def test_native_client_read_table_all_parameters_accepted():
    client = _native.Client(quota_project_id="test-quota-project")
    utc_dt = datetime.datetime(
        2023, 11, 14, 22, 13, 20, 500000, tzinfo=datetime.timezone.utc
    )

    with pytest.raises(RuntimeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            maintain_order=True,
            snapshot_time=utc_dt,
            selected_fields=["col1", "col2"],
            row_restriction="col1 > 100",
            arrow_buffer_compression="zstd",
            sample_percentage=42.5,
        )

    with pytest.raises(RuntimeError):
        client.read_table(
            "projects/p/datasets/d/tables/t",
            True,
            utc_dt,
            ["col1", "col2"],
            "col1 > 100",
            "zstd",
            42.5,
        )
