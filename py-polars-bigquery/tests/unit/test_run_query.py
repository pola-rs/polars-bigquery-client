from unittest.mock import MagicMock, patch

import arrow_bigquery.api.resources
import pytest
import requests
from polars_bigquery.core.bigquery_rest import (
    BigQueryRestClient,
    _raise_for_bigquery_error,
    create_resilient_session,
    get_table_metadata,
)
from polars_bigquery.exceptions import BigQueryError


def test_create_resilient_session_configures_retries_and_pooling():
    session = create_resilient_session()
    adapter = session.get_adapter("https://bigquery.googleapis.com")
    assert adapter.max_retries.total == 5
    assert adapter.max_retries.backoff_factor == 0.5
    assert set(adapter.max_retries.status_forcelist) == {429, 500, 502, 503, 504}
    assert adapter.max_retries.respect_retry_after_header is True


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
