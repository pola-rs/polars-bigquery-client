from unittest.mock import patch, MagicMock, ANY
import polars as pl
import pytest
from polars_bigquery import read_bigquery
from polars_bigquery._read_bigquery import _parse_table_id


@pytest.fixture
def mock_rust_read():
    with patch("polars_bigquery.polars_bigquery.read_bigquery") as mocked:
        yield mocked


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
    result = read_bigquery(table="my-project.my_dataset.my_table", quota_project_id="q")

    # Assert
    mock_rust_read.assert_called_once_with("my-project.my_dataset.my_table")
    assert result.equals(mock_df)


def test_read_bigquery_with_query(mock_rust_read):
    # Prepare
    mock_df = pl.DataFrame({"col1": [1, 2]})
    mock_rust_read.return_value = mock_df
    
    with patch("polars_bigquery._read_bigquery.run_query") as mock_run_query:
        mock_run_query.return_value = "project.dataset.temp_table"
        
        # Execute
        result = read_bigquery(query="SELECT 1", quota_project_id="q")
        
        # Assert
        mock_run_query.assert_called_once_with("SELECT 1", "q", ANY)
        mock_rust_read.assert_called_once_with("project.dataset.temp_table")
        assert result.equals(mock_df)


def test_read_bigquery_handles_bigquery_objects(mock_rust_read):
    # Prepare
    mock_rust_read.return_value = pl.DataFrame()
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"

    # Execute
    read_bigquery(table=mock_ref, quota_project_id="q")

    # Assert
    mock_rust_read.assert_called_once_with("p.d.t")


def test_read_bigquery_propagates_errors(mock_rust_read):
    # Prepare
    mock_rust_read.side_effect = Exception("Rust error")

    # Execute & Assert
    with pytest.raises(Exception, match="Rust error"):
        read_bigquery(table="p.d.t", quota_project_id="q")
