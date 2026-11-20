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


def test_bigquery_table_id_immutable_and_hashable():
    t1 = BigQueryTableId("proj", "ds", "tab")
    t2 = BigQueryTableId("proj", "ds", "tab")
    assert hash(t1) == hash(t2)
    assert {t1: "ok"}[t2] == "ok"

    with pytest.raises(AttributeError):
        t1.project_id = "mutated"


def test_parse_table_id_propagates_property_exception():
    class FailingTableObj:
        @property
        def project(self):
            raise RuntimeError("boom in property")

    with pytest.raises(RuntimeError, match="boom in property"):
        parse_table_id(FailingTableObj())


@pytest.mark.parametrize(
    ("project_id", "dataset_id", "table_id"),
    [
        ("proj", "ds", "tab"),
        ("my-project-123", "my_dataset", "my_table"),
        ("google.com:my-project", "my_dataset", "my_table"),
        ("sub.example.com:proj-1", "cat.ns1.ns2", "tab-1"),
        ("proj", "_temp_dataset", "anon_table_123"),
        ("proj", "region-us.INFORMATION_SCHEMA", "JOBS_BY_PROJECT"),
        ("proj", "ds", "my_table$20200101"),
        ("proj", "ds", "my_table$__NULL__"),
        ("proj", "ds", "my_table@1700000000000"),
        ("proj", "ds", "my_table@-3600000"),
        ("proj", "ds", "my_table@-3600000-"),
        ("proj", "ds", "my_table@1000-2000"),
        ("proj", "ds", "table 01"),
        ("proj", "ds", "my  table"),
        ("proj", "ds", " table 01 "),
        ("proj", "ds", "table\u00a001"),
        ("proj", "ds", "テーブル\u300001"),
        ("proj", "ds", "table\u201301"),
        ("proj", "ds", "table\uff3f01"),
        ("proj", "ds", "table 01$20200101"),
        ("proj", "ds", "table 01@1700000000000"),
        ("google.com:my-project", "ds", "table 01"),
        ("proj", "ds", "étudiant-01"),
        ("proj", "ds", "00_お客様"),
        ("proj", "ds", "ग्राहक"),
        ("proj", "ds", "e\u0301tudiant"),
        ("proj", "数据_集", "étudiant-01$20240101"),
    ],
)
def test_table_id_valid_edge_cases_round_trip(project_id, dataset_id, table_id):
    tid = BigQueryTableId(project_id, dataset_id, table_id)
    assert tid.project_id == project_id
    assert tid.dataset_id == dataset_id
    assert tid.table_id == table_id
    assert BigQueryTableId.from_string(str(tid)) == tid
    assert BigQueryTableId.from_str(str(tid)) == tid
    assert parse_table_id(str(tid)) == tid


@pytest.mark.parametrize(
    "bad_project",
    [
        "",
        ".",
        "..",
        "proj.extra",
        ":proj",
        "google.com:",
        "google..com:proj",
        ".google.com:proj",
        "google.com.:proj",
        "..:proj",
        "google.com:proj:extra",
        "google.com:proj.extra",
        "proj/../../evil",
        "proj/evil",
        "proj\\evil",
        "proj?fields=all",
        "proj#frag",
        "proj%2f",
        "proj%2e%2e",
        "`proj`",
        "proj ",
        " proj",
        "proj name",
        "proj\t",
        "proj\n",
        "proj\r\n",
        "proj\x00",
        "proj\x7f",
        "proj\u00a0",
        "proj\u3000",
        "proj;evil=1",
        "proj&x=1",
        "proj$1",
        "proj@1",
    ],
)
def test_table_id_rejects_invalid_project_id(bad_project):
    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId(bad_project, "ds", "tab")

    class FakeTableWithProject:
        project = bad_project
        dataset_id = "ds"
        table_id = "tab"

    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id(FakeTableWithProject())

    class FakeTableWithProjectId:
        project_id = bad_project
        dataset_id = "ds"
        table_id = "tab"

    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id(FakeTableWithProjectId())


@pytest.mark.parametrize(
    "bad_dataset",
    [
        "",
        ".",
        "..",
        ".ds",
        "ds.",
        "a..b",
        "ds:extra",
        "ds/../../evil",
        "ds\\evil",
        "ds?fields=all",
        "ds#frag",
        "ds%2f",
        "`ds`",
        "ds ",
        " ds",
        "ds name",
        "ds\t",
        "ds\n",
        "ds\r\n",
        "ds\x00",
        "ds\x7f",
        "ds\u0085",
        "ds\u00a0",
        "ds\u2028",
        "ds\u3000",
        "ds;evil=1",
        "ds&x=1",
        "ds$1",
        "ds@1",
    ],
)
def test_table_id_rejects_invalid_dataset_id(bad_dataset):
    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId("proj", bad_dataset, "tab")

    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId.from_string(f"proj.{bad_dataset}.tab")

    class FakeTable:
        project = "proj"
        dataset_id = bad_dataset
        table_id = "tab"

    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id(FakeTable())


@pytest.mark.parametrize(
    "bad_table",
    [
        "",
        " ",
        "   ",
        "\u00a0",
        "\u3000",
        "   $20200101",
        ".",
        "..",
        "a.b",
        "tab:delete",
        "tab/../../evil",
        "tab\\evil",
        "tab?fields=all#",
        "tab?fields=etag#frag",
        "tab#frag",
        "tab%2f",
        "tab%2e%2e",
        "`tab`",
        "tab\t",
        "tab\n",
        "tab\r\n",
        "tab\x0b",
        "tab\x0c",
        "tab\x00",
        "tab\x7f",
        "tab\u0085",
        "tab\u009f",
        "tab\u2028",
        "tab\u2029",
        "tab HTTP/1.1",
        "tab;evil=1",
        "tab&fields=all",
        "tab=1",
        "tab'quote",
        'tab"quote',
        "$",
        "@",
        "$20200101",
        "@1700000000000",
        "tab$",
        "tab@",
        "tab$$20200101",
        "tab$2020$2021",
        "tab@@1234",
        "tab@1@2",
        "tab$@1234",
        "tab@$1234",
        "tab$2020 01",
        "tab@123 456",
    ],
)
def test_table_id_rejects_invalid_table_id(bad_table):
    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId("proj", "ds", bad_table)

    if bad_table != "a.b":
        with pytest.raises(ValueError, match="Invalid table ID"):
            BigQueryTableId.from_string(f"proj.ds.{bad_table}")

    class FakeTable:
        project = "proj"
        dataset_id = "ds"
        table_id = bad_table

    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id(FakeTable())


def test_vuln_03_repro_cases_rejected():
    class FakeTable:
        project = "proj/../../evil"
        dataset_id = "ds"
        table_id = "tab?fields=all#"

    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id(FakeTable())

    with pytest.raises(ValueError, match="Invalid table ID"):
        parse_table_id("proj.ds.tab?fields=etag#frag")

    with pytest.raises(ValueError, match="Invalid table ID"):
        BigQueryTableId("proj/../../evil", "ds", "tab?fields=all#")


def test_parse_table_id_escapes_control_chars_in_error_message():
    with pytest.raises(ValueError) as exc_info:
        parse_table_id("proj.ds.tab\r\nInjected: header")
    message = str(exc_info.value)
    assert "Invalid table ID" in message
    assert "\r" not in message
    assert "\n" not in message


def test_parse_table_id_rejects_surrogate_string_as_value_error():
    with pytest.raises(UnicodeEncodeError):
        parse_table_id("proj.ds.\ud800")
