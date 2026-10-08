from unittest.mock import MagicMock

import pytest
from arrow_bigquery.api.resources import BigQueryTableId, parse_table_id


def test_parse_table_id_valid_string():
    assert parse_table_id("proj.ds.tab") == BigQueryTableId("proj", "ds", "tab")


def test_parse_table_id_with_colon():
    assert parse_table_id("google.com:project.ds.tab") == BigQueryTableId(
        "google.com:project", "ds", "tab"
    )


def test_parse_table_id_multipart():
    assert parse_table_id("too.many.parts.here") == BigQueryTableId(
        "too", "many.parts", "here"
    )


def test_parse_table_id_table_reference():
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"
    assert parse_table_id(mock_ref) == BigQueryTableId("p", "d", "t")


def test_parse_table_id_table_object():
    mock_table = MagicMock()
    mock_table.project = "proj-obj"
    mock_table.dataset_id = "ds-obj"
    mock_table.table_id = "tab-obj"
    assert parse_table_id(mock_table) == BigQueryTableId(
        "proj-obj", "ds-obj", "tab-obj"
    )


def test_parse_table_id_with_project_id_attribute():
    mock_table = MagicMock(spec=["project_id", "dataset_id", "table_id"])
    mock_table.project_id = "proj"
    mock_table.dataset_id = "ds"
    mock_table.table_id = "tab"
    assert parse_table_id(mock_table) == BigQueryTableId("proj", "ds", "tab")


def test_parse_table_id_idempotent():
    table_id = BigQueryTableId("proj", "ds", "tab")
    assert parse_table_id(table_id) == table_id


def test_parse_table_id_invalid_format():
    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id("just_a_string")
    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id("`proj.ds.tab`")
    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id("proj..tab")


def test_parse_table_id_invalid_type():
    with pytest.raises(TypeError, match="Expected table_id to be a string"):
        parse_table_id(123)


def test_bigquery_table_id_parse():
    assert BigQueryTableId.parse("proj.ds.tab") == BigQueryTableId("proj", "ds", "tab")
    assert BigQueryTableId.parse("google.com:proj.ds.tab") == BigQueryTableId(
        "google.com:proj", "ds", "tab"
    )
    mock_ref = MagicMock()
    mock_ref.project = "p"
    mock_ref.dataset_id = "d"
    mock_ref.table_id = "t"
    assert BigQueryTableId.parse(mock_ref) == BigQueryTableId("p", "d", "t")
    assert parse_table_id == BigQueryTableId.parse
