import threading
import _thread
import time

from unittest.mock import patch, MagicMock, ANY
import polars as pl
import pytest

from arrow_bigquery import _native
from polars_bigquery import (
    read_bigquery_table,
    read_bigquery_query,
    scan_bigquery_table,
    __version__,
)
from polars_bigquery._read_bigquery import _get_user_agent, _parse_table_id


@pytest.fixture
def mock_rust_read():
    with patch("arrow_bigquery._native.read_bigquery_table") as mocked:
        yield mocked


def test_get_user_agent():
    assert _get_user_agent(None) == f"polars-bigquery/{__version__}"
    assert _get_user_agent("") == f"polars-bigquery/{__version__}"
    assert (
        _get_user_agent("custom-extension/1.0")
        == f"polars-bigquery/{__version__} custom-extension/1.0"
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


def test_read_bigquery_calls_rust_with_parsed_id(mock_rust_read):
    # Prepare
    mock_df = pl.DataFrame({"col1": [1, 2]})
    mock_rust_read.return_value = mock_df

    # Execute
    result = read_bigquery_table(table="my-project.my_dataset.my_table", quota_project_id="q")

    # Assert
    mock_rust_read.assert_called_once_with(
        "my-project.my_dataset.my_table",
        "q",
        False,
        ANY,
        f"polars-bigquery/{__version__}",
    )
    assert result.equals(mock_df)


def test_read_bigquery_query(mock_rust_read):
    # Prepare
    mock_df = pl.DataFrame({"col1": [1, 2]})
    mock_rust_read.return_value = mock_df

    with patch("polars_bigquery._read_bigquery.run_query") as mock_run_query:
        mock_run_query.return_value = "project.dataset.temp_table"

        # Execute
        result = read_bigquery_query(query="SELECT 1", quota_project_id="q")

        # Assert
        expected_ua = f"polars-bigquery/{__version__}"
        mock_run_query.assert_called_once_with("SELECT 1", "q", ANY, user_agent=expected_ua)
        mock_rust_read.assert_called_once_with(
            "project.dataset.temp_table", "q", False, ANY, expected_ua
        )
        assert result.equals(mock_df)


def test_read_bigquery_query_with_user_agent(mock_rust_read):
    # Prepare
    mock_df = pl.DataFrame({"col1": [1, 2]})
    mock_rust_read.return_value = mock_df

    with patch("polars_bigquery._read_bigquery.run_query") as mock_run_query:
        mock_run_query.return_value = "project.dataset.temp_table"

        # Execute
        result = read_bigquery_query(
            query="SELECT 1", quota_project_id="q", user_agent="custom-ua/1.0"
        )

        # Assert
        expected_ua = f"polars-bigquery/{__version__} custom-ua/1.0"
        mock_run_query.assert_called_once_with(
            "SELECT 1", "q", ANY, user_agent=expected_ua
        )
        mock_rust_read.assert_called_once_with(
            "project.dataset.temp_table", "q", False, ANY, expected_ua
        )
        assert result.equals(mock_df)


def test_read_bigquery_handles_bigquery_objects(mock_rust_read):
    # Prepare
    mock_rust_read.return_value = pl.DataFrame()
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"

    # Execute
    read_bigquery_table(table=mock_ref, quota_project_id="q")

    # Assert
    mock_rust_read.assert_called_once_with(
        "p.d.t", "q", False, ANY, f"polars-bigquery/{__version__}"
    )


def test_read_bigquery_propagates_errors(mock_rust_read):
    # Prepare
    mock_rust_read.side_effect = Exception("Rust error")

    # Execute & Assert
    with pytest.raises(Exception, match="Rust error"):
        read_bigquery_table(table="p.d.t", quota_project_id="q")


def test_read_bigquery_with_user_agent(mock_rust_read):
    # Prepare
    mock_rust_read.return_value = pl.DataFrame()

    # Execute
    read_bigquery_table(
        table="p.d.t", quota_project_id="q", user_agent="custom-extension/1.0"
    )

    # Assert
    mock_rust_read.assert_called_once_with(
        "p.d.t",
        "q",
        False,
        ANY,
        f"polars-bigquery/{__version__} custom-extension/1.0",
    )


def test_scan_bigquery_calls_rust_with_parsed_id(mock_rust_read):
    # Prepare
    mock_stream = MagicMock()
    mock_rust_read.return_value = mock_stream

    with patch("polars.scan_arrow_c_stream") as mock_scan:
        mock_lazy_df = pl.LazyFrame({"col1": [1, 2]})
        mock_scan.return_value = mock_lazy_df

        # Execute
        result = scan_bigquery_table(
            table="my-project.my_dataset.my_table", quota_project_id="q"
        )

        # Assert
        mock_rust_read.assert_called_once_with(
            "my-project.my_dataset.my_table",
            "q",
            False,
            ANY,
            f"polars-bigquery/{__version__}",
        )
        mock_scan.assert_called_once_with(mock_stream)
        assert result.collect().equals(mock_lazy_df.collect())


def test_scan_bigquery_with_user_agent(mock_rust_read):
    # Prepare
    mock_stream = MagicMock()
    mock_rust_read.return_value = mock_stream

    with patch("polars.scan_arrow_c_stream") as mock_scan:
        mock_lazy_df = pl.LazyFrame({"col1": [1, 2]})
        mock_scan.return_value = mock_lazy_df

        # Execute
        result = scan_bigquery_table(
            table="my-project.my_dataset.my_table",
            quota_project_id="q",
            user_agent="custom-extension/1.0",
        )

        # Assert
        mock_rust_read.assert_called_once_with(
            "my-project.my_dataset.my_table",
            "q",
            False,
            ANY,
            f"polars-bigquery/{__version__} custom-extension/1.0",
        )
        mock_scan.assert_called_once_with(mock_stream)


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
        # Constructing the DataFrame will consume the C-stream.
        # Since the channel is empty and kept open, it will block.
        # The interrupt should break it.
        pl.DataFrame(exporter)
    except BaseException as err:
        # This catches KeyboardInterrupt, ComputeError, and PanicException (from Polars unwrap).
        if isinstance(err, KeyboardInterrupt) or "Python interrupt" in str(err):
            interrupted = True
        else:
            raise

    thread.join()
    assert interrupted, "The C-stream consumption was not interrupted"
