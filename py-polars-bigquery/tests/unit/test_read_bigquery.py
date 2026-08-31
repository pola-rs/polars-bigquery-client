from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from polars_bigquery import (
    Client,
    __version__,
)
from polars_bigquery._read_bigquery import _get_user_agent, _parse_table_id


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
            "my-project.my_dataset.my_table",
            maintain_order=False,
        )
        mock_df_cls.assert_called_once_with(mock_exporter)
        assert result is mock_df


def test_client_read_query(mock_arrow_client):
    mock_exporter = MagicMock()
    mock_arrow_client.read_table.return_value = mock_exporter

    with (
        patch("polars_bigquery._read_bigquery.run_query") as mock_run_query,
        patch("polars.DataFrame") as mock_df_cls,
    ):
        mock_run_query.return_value = "project.dataset.temp_table"
        mock_df = MagicMock()
        mock_df_cls.return_value = mock_df

        client = Client(quota_project_id="q")
        result = client.read_query(query="SELECT 1")

        expected_ua = f"polars-bigquery/{__version__}"
        mock_run_query.assert_called_once_with(
            "SELECT 1", "q", client.credentials_provider, user_agent=expected_ua
        )
        mock_arrow_client.read_table.assert_called_once_with(
            "project.dataset.temp_table",
            maintain_order=False,
        )
        assert result is mock_df


def test_client_read_query_with_user_agent(mock_arrow_client):
    mock_exporter = MagicMock()
    mock_arrow_client.read_table.return_value = mock_exporter

    with (
        patch("polars_bigquery._read_bigquery.run_query") as mock_run_query,
        patch("polars.DataFrame") as mock_df_cls,
    ):
        mock_run_query.return_value = "project.dataset.temp_table"
        mock_df = MagicMock()
        mock_df_cls.return_value = mock_df

        client = Client(user_agent="custom-ua/1.0", quota_project_id="q")
        result = client.read_query(query="SELECT 1")

        assert result is not None
        expected_ua = f"polars-bigquery/{__version__} custom-ua/1.0"
        mock_run_query.assert_called_once_with(
            "SELECT 1", "q", client.credentials_provider, user_agent=expected_ua
        )


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
            "p.d.t",
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

    with patch("polars.scan_arrow_c_stream") as mock_scan:
        mock_lazy_df = pl.LazyFrame({"col1": [1, 2]})
        mock_scan.return_value = mock_lazy_df

        client = Client(quota_project_id="q")
        result = client.scan_table(table="my-project.my_dataset.my_table")

        mock_arrow_client.read_table.assert_called_once_with(
            "my-project.my_dataset.my_table",
            maintain_order=False,
        )
        mock_scan.assert_called_once_with(mock_stream)
        assert result.collect().equals(mock_lazy_df.collect())
