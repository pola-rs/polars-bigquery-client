import urllib.parse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import arrow_bigquery.api.resources
import arrow_bigquery.exceptions
import pytest
import requests
from polars_bigquery.core.bigquery_rest import (
    BigQueryRestClient,
    _get_table_metadata_url,
    _raise_for_bigquery_error,
    create_resilient_session,
    get_table_metadata,
)
from polars_bigquery.exceptions import BigQueryError


def test_bigquery_error_is_unified_with_arrow_bigquery():
    assert BigQueryError is arrow_bigquery.exceptions.BigQueryError
    assert issubclass(BigQueryError, RuntimeError)


def test_create_resilient_session_configures_retries_and_pooling():
    from email.utils import formatdate

    from polars_bigquery.core.bigquery_rest import MAX_RETRY_AFTER_SECONDS
    from urllib3.exceptions import InvalidHeader

    session = create_resilient_session()
    try:
        adapter = session.get_adapter("https://bigquery.googleapis.com")
        retry = adapter.max_retries
        assert retry.total == 5
        assert retry.backoff_factor == 0.5
        assert set(retry.status_forcelist) == {429, 500, 502, 503, 504}
        assert retry.respect_retry_after_header is True

        # Retry-After durations are bounded to prevent multi-hour/indefinite sleeps.
        assert retry.parse_retry_after("5") == 5.0
        assert retry.parse_retry_after("0") == 0.0
        assert retry.parse_retry_after("86400") == float(MAX_RETRY_AFTER_SECONDS)
        far_future = formatdate(timeval=4_000_000_000, usegmt=True)
        assert retry.parse_retry_after(far_future) == float(MAX_RETRY_AFTER_SECONDS)
        past_date = formatdate(timeval=0, usegmt=True)
        assert retry.parse_retry_after(past_date) == 0.0
        with pytest.raises(InvalidHeader):
            retry.parse_retry_after("not-a-valid-retry-after")

        # Verify the cap stays enforced across `Retry.increment()` / `Retry.new()`.
        next_retry = retry.increment(
            method="GET", url="https://bigquery.googleapis.com"
        )
        assert type(next_retry) is type(retry)
        assert next_retry.parse_retry_after("86400") == float(MAX_RETRY_AFTER_SECONDS)
    finally:
        session.close()


def test_raise_for_bigquery_error_extracts_structured_json_message():
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
    resp.json.return_value = {
        "error": {"message": "Table not found: my_dataset.my_table"}
    }

    with pytest.raises(BigQueryError, match="Table not found: my_dataset.my_table"):
        _raise_for_bigquery_error(resp)


def test_raise_for_bigquery_error_falls_back_when_json_invalid():
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("502 Bad Gateway")
    resp.json.side_effect = ValueError("No JSON object could be decoded")

    with pytest.raises(BigQueryError, match="502 Bad Gateway"):
        _raise_for_bigquery_error(resp)


def test_get_table_metadata_uses_field_mask_and_session():
    mock_cp = MagicMock(return_value=({"bearer_token": "tok"}, 123))
    mock_session = MagicMock()
    mock_session.get.return_value.json.return_value = {"schema": {"fields": []}}

    table_ref = arrow_bigquery.api.resources.BigQueryTableId("p", "d", "t")
    metadata = get_table_metadata(
        table_ref,
        quota_project_id="quota-project",
        credentials_provider=mock_cp,
        user_agent="polars-bigquery/0.1.0",
        session=mock_session,
    )

    assert metadata == {"schema": {"fields": []}}
    called_url = mock_session.get.call_args.args[0]
    assert called_url.endswith(
        "/projects/p/datasets/d/tables/t?fields=schema,timePartitioning"
    )


def test_bigquery_rest_client_context_manager_closes_session():
    mock_cp = MagicMock(return_value=({"bearer_token": "tok"}, 123))
    client = BigQueryRestClient(
        quota_project_id="q",
        credentials_provider=mock_cp,
        user_agent="ua",
    )
    with patch.object(client.session, "close") as mock_close:
        with client:
            pass
        mock_close.assert_called_once()


@pytest.mark.parametrize(
    ("project_id", "dataset_id", "table_id", "expected_path"),
    [
        (
            "my-project",
            "my_dataset",
            "my_table",
            "/projects/my-project/datasets/my_dataset/tables/my_table",
        ),
        (
            "google.com:my-project",
            "my_dataset",
            "my_table",
            "/projects/google.com%3Amy-project/datasets/my_dataset/tables/my_table",
        ),
        (
            "proj",
            "cat.ns1.ns2",
            "tab",
            "/projects/proj/datasets/cat.ns1.ns2/tables/tab",
        ),
        (
            "proj",
            "ds",
            "my_table$20200101",
            "/projects/proj/datasets/ds/tables/my_table%2420200101",
        ),
        (
            "proj",
            "ds",
            "my_table@1700000000000",
            "/projects/proj/datasets/ds/tables/my_table%401700000000000",
        ),
        (
            "proj",
            "ds",
            "table 01",
            "/projects/proj/datasets/ds/tables/table%2001",
        ),
        (
            "proj",
            "ds",
            "table 01$20200101",
            "/projects/proj/datasets/ds/tables/table%2001%2420200101",
        ),
        (
            "proj",
            "ds",
            "テーブル\u300001",
            "/projects/proj/datasets/ds/tables/%E3%83%86%E3%83%BC%E3%83%96%E3%83%AB%E3%80%8001",
        ),
        (
            "proj",
            "ds",
            "étudiant-01",
            "/projects/proj/datasets/ds/tables/%C3%A9tudiant-01",
        ),
        (
            "proj",
            "ds",
            "00_お客様",
            "/projects/proj/datasets/ds/tables/00_%E3%81%8A%E5%AE%A2%E6%A7%98",
        ),
        (
            "proj",
            "ds",
            "ग्राहक",
            "/projects/proj/datasets/ds/tables/%E0%A4%97%E0%A5%8D%E0%A4%B0%E0%A4%BE%E0%A4%B9%E0%A4%95",
        ),
    ],
)
def test_get_table_metadata_url_encodes_valid_segments(
    project_id, dataset_id, table_id, expected_path
):
    table_ref = arrow_bigquery.api.resources.BigQueryTableId(
        project_id, dataset_id, table_id
    )
    url = _get_table_metadata_url(table_ref=table_ref)
    assert (
        url
        == f"https://bigquery.googleapis.com/bigquery/v2{expected_path}?fields=schema,timePartitioning"
    )
    # Verify requests does not double-encode or unquote reserved characters.
    prepared_url = requests.Request("GET", url).prepare().url
    assert prepared_url == url


def test_get_table_metadata_url_defense_in_depth_encodes_metacharacters():
    fake_ref = SimpleNamespace(
        project_id="proj/../../evil",
        dataset_id="ds%2f\\bad",
        table_id="tab?fields=all#frag\r\n ",
    )
    url = _get_table_metadata_url(table_ref=fake_ref)
    parsed = urllib.parse.urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "bigquery.googleapis.com"
    assert parsed.query == "fields=schema,timePartitioning"
    assert parsed.fragment == ""
    assert (
        parsed.path
        == "/bigquery/v2/projects/proj%2F..%2F..%2Fevil/datasets/ds%252f%5Cbad/tables/tab%3Ffields%3Dall%23frag%0D%0A%20"
    )
    assert requests.Request("GET", url).prepare().url == url


@pytest.mark.parametrize(
    ("project_id", "dataset_id", "table_id"),
    [
        ("", "ds", "tab"),
        (".", "ds", "tab"),
        ("..", "ds", "tab"),
        ("proj", "", "tab"),
        ("proj", ".", "tab"),
        ("proj", "..", "tab"),
        ("proj", "ds", ""),
        ("proj", "ds", "."),
        ("proj", "ds", ".."),
    ],
)
def test_get_table_metadata_url_rejects_empty_and_dot_segments(
    project_id, dataset_id, table_id
):
    fake_ref = SimpleNamespace(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
    )
    with pytest.raises(ValueError, match="URL path segment"):
        _get_table_metadata_url(table_ref=fake_ref)


def test_get_table_metadata_url_rejects_non_string_segments():
    fake_ref = SimpleNamespace(project_id=None, dataset_id="ds", table_id="tab")
    with pytest.raises(TypeError, match="Expected project_id to be a str"):
        _get_table_metadata_url(table_ref=fake_ref)


def test_get_table_metadata_rejects_invalid_json_or_non_object():
    mock_cp = MagicMock(return_value=({"bearer_token": "tok"}, 123))
    table_ref = arrow_bigquery.api.resources.BigQueryTableId("p", "d", "t")

    mock_session = MagicMock()
    mock_session.get.return_value.json.side_effect = ValueError("bad json")
    with pytest.raises(BigQueryError, match="invalid JSON"):
        get_table_metadata(
            table_ref,
            quota_project_id="q",
            credentials_provider=mock_cp,
            user_agent="ua",
            session=mock_session,
        )

    mock_session.get.return_value.json.side_effect = None
    mock_session.get.return_value.json.return_value = ["unexpected", "list"]
    with pytest.raises(BigQueryError, match="unexpected metadata payload type list"):
        get_table_metadata(
            table_ref,
            quota_project_id="q",
            credentials_provider=mock_cp,
            user_agent="ua",
            session=mock_session,
        )


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.Timeout("timed out"),
        requests.exceptions.ConnectionError("connection reset"),
        requests.exceptions.RetryError("max retries exceeded"),
    ],
)
def test_get_table_metadata_wraps_request_exceptions_as_bigquery_error(exc):
    mock_cp = MagicMock(return_value=({"bearer_token": "tok"}, 123))
    table_ref = arrow_bigquery.api.resources.BigQueryTableId("p", "d", "t")
    mock_session = MagicMock()
    mock_session.get.side_effect = exc

    with pytest.raises(BigQueryError, match="Failed to fetch table metadata") as raised:
        get_table_metadata(
            table_ref,
            quota_project_id="q",
            credentials_provider=mock_cp,
            user_agent="ua",
            session=mock_session,
        )
    assert raised.value.__cause__ is exc
