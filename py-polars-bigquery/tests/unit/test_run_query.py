from unittest.mock import MagicMock, patch

import arrow_bigquery.api.resources
import pytest
import freezegun
import requests

from polars_bigquery.core.bigquery_rest import (
    BigQueryRestClient,
    _raise_for_bigquery_error,
    _wait_for_job,
    create_resilient_session,
    get_table_metadata,
    run_query,
)
from polars_bigquery.exceptions import BigQueryError


@pytest.fixture(autouse=True)
def freeze_time():
    with freezegun.freeze_time("2026-09-21", auto_tick_seconds=1):
        yield


def test_run_query_success():
    mock_cp = MagicMock()
    mock_cp.return_value = ({"bearer_token": "fake-token"}, 12345)

    with (
        patch("requests.post") as mock_post,
        patch("requests.get") as mock_get,
        patch("time.sleep"),
    ):
        # Mock job insertion
        mock_post.return_value.json.return_value = {
            "jobReference": {"jobId": "job-123"}
        }
        mock_post.return_value.raise_for_status = MagicMock()

        # Mock job polling (first RUNNING, then DONE)
        m1 = MagicMock()
        m1.json.return_value = {"status": {"state": "RUNNING"}}

        m2 = MagicMock()
        m2.json.return_value = {
            "status": {"state": "DONE"},
            "configuration": {
                "query": {
                    "destinationTable": {
                        "projectId": "p",
                        "datasetId": "d",
                        "tableId": "t",
                    }
                }
            },
        }

        m_poll = MagicMock()
        m_poll.json.return_value = {}

        mock_get.side_effect = [m1, m_poll, m2]

        result = run_query(
            "SELECT 1",
            quota_project_id="quota-project",
            credentials_provider=mock_cp,
            user_agent="polars-bigquery/0.1.0",
        )

        assert result == "p.d.t"
        mock_post.assert_called_once()
        assert mock_get.call_count == 3

        # Verify headers
        called_headers = mock_post.call_args.kwargs["headers"]
        assert called_headers["Authorization"] == "Bearer fake-token"
        assert called_headers["x-goog-user-project"] == "quota-project"
        assert called_headers["User-Agent"] == "polars-bigquery/0.1.0"


def test_run_query_error():
    mock_cp = MagicMock()
    mock_cp.return_value = ({"bearer_token": "fake-token"}, 12345)

    with (
        patch("requests.post") as mock_post,
        patch("requests.get") as mock_get,
        patch("time.sleep"),
    ):
        mock_post.return_value.json.return_value = {
            "jobReference": {"jobId": "job-123"}
        }
        mock_post.return_value.raise_for_status = MagicMock()

        mock_get.return_value.json.return_value = {
            "status": {
                "state": "DONE",
                "errorResult": {"message": "Something went wrong"},
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()

        with pytest.raises(BigQueryError, match="Something went wrong"):
            run_query(
                "SELECT 1",
                quota_project_id="quota-project",
                credentials_provider=mock_cp,
                user_agent="polars-bigquery/0.1.0",
            )


def test_run_query_with_user_agent():
    mock_cp = MagicMock()
    mock_cp.return_value = ({"bearer_token": "fake-token"}, 12345)

    with (
        patch("requests.post") as mock_post,
        patch("requests.get") as mock_get,
        patch("time.sleep"),
    ):
        mock_post.return_value.json.return_value = {
            "jobReference": {"jobId": "job-123"}
        }
        mock_post.return_value.raise_for_status = MagicMock()

        mock_get.return_value.json.return_value = {
            "status": {"state": "DONE"},
            "configuration": {
                "query": {
                    "destinationTable": {
                        "projectId": "p",
                        "datasetId": "d",
                        "tableId": "t",
                    }
                }
            },
        }

        run_query(
            "SELECT 1",
            quota_project_id="quota-project",
            credentials_provider=mock_cp,
            user_agent="polars-bigquery/0.1.0 custom-ua/1.0",
        )

        called_headers = mock_post.call_args.kwargs["headers"]
        assert called_headers["User-Agent"] == "polars-bigquery/0.1.0 custom-ua/1.0"


def test_run_query_requires_user_agent():
    mock_cp = MagicMock()
    with pytest.raises(TypeError):
        run_query(
            "SELECT 1",
            quota_project_id="quota-project",
            credentials_provider=mock_cp,
        )


def test_run_query_refreshes_token_during_polling_and_generates_job_id():
    tokens = iter(
        [
            ({"bearer_token": "initial-token"}, 12345),
            ({"bearer_token": "refreshed-token-1"}, 12346),
            ({"bearer_token": "refreshed-token-2"}, 12347),
        ]
    )
    mock_cp = MagicMock(side_effect=lambda: next(tokens))

    with (
        patch("requests.post") as mock_post,
        patch("requests.get") as mock_get,
    ):
        mock_post.return_value.json.return_value = {
            "jobReference": {"jobId": "job-123"}
        }
        mock_post.return_value.raise_for_status = MagicMock()

        m1 = MagicMock()
        m1.json.return_value = {"status": {"state": "RUNNING"}}
        m1.raise_for_status = MagicMock()

        m_poll = MagicMock()
        m_poll.json.return_value = {}
        m_poll.raise_for_status = MagicMock()

        m2 = MagicMock()
        m2.json.return_value = {
            "status": {"state": "DONE"},
            "configuration": {
                "query": {
                    "destinationTable": {
                        "projectId": "p",
                        "datasetId": "d",
                        "tableId": "t",
                    }
                }
            },
        }
        m2.raise_for_status = MagicMock()

        mock_get.side_effect = [m1, m_poll, m2]

        result = run_query(
            "SELECT 1",
            quota_project_id="quota-project",
            credentials_provider=mock_cp,
            user_agent="polars-bigquery/0.1.0",
        )
        assert result == "p.d.t"

        # Verify client-side idempotent jobId was generated in jobs.insert body
        insert_body = mock_post.call_args.kwargs["json"]
        assert insert_body["jobReference"]["jobId"].startswith("polars_bq_")
        assert insert_body["jobReference"]["projectId"] == "quota-project"

        # Verify credentials_provider was called on insert and on each poll cycle
        assert mock_cp.call_count == 3


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


def test_wait_for_job_timeout_raises_bigquery_error():
    mock_cp = MagicMock(return_value=({"bearer_token": "tok"}, 123))
    with (
        patch("time.monotonic", side_effect=[0.0, 100.0]),
        pytest.raises(
            BigQueryError,
            match="Timed out waiting for BigQuery job job-123 after 50.0s",
        ),
    ):
        _wait_for_job(
            "job-123",
            "quota-project",
            mock_cp,
            "polars-bigquery/0.1.0",
            max_poll_seconds=50.0,
        )


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


def test_run_query_propagates_regional_location():
    mock_cp = MagicMock(return_value=({"bearer_token": "tok"}, 123))
    mock_session = MagicMock()

    # Mock insert response containing regional location
    mock_insert_resp = MagicMock()
    mock_insert_resp.json.return_value = {
        "jobReference": {"jobId": "job-reg", "location": "europe-west1"}
    }
    mock_session.post.return_value = mock_insert_resp

    # Mock poll response
    mock_poll_resp = MagicMock()
    mock_poll_resp.json.return_value = {
        "status": {"state": "DONE"},
        "configuration": {
            "query": {
                "destinationTable": {"projectId": "p", "datasetId": "d", "tableId": "t"}
            }
        },
    }
    mock_session.get.return_value = mock_poll_resp

    dest = run_query(
        "SELECT 1",
        quota_project_id="q",
        credentials_provider=mock_cp,
        user_agent="ua",
        session=mock_session,
    )
    assert dest == "p.d.t"

    # Verify location query parameter was appended to jobs.get URL
    called_get_url = mock_session.get.call_args_list[0].args[0]
    assert "location=europe-west1" in called_get_url


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


def test_run_query_handles_http_409_conflict_on_insert():
    mock_cp = MagicMock(return_value=({"bearer_token": "tok"}, 123))
    mock_session = MagicMock()

    mock_insert_resp = MagicMock()
    mock_insert_resp.status_code = 409
    mock_session.post.return_value = mock_insert_resp

    mock_poll_resp = MagicMock()
    mock_poll_resp.status_code = 200
    mock_poll_resp.json.return_value = {
        "status": {"state": "DONE"},
        "configuration": {
            "query": {
                "destinationTable": {"projectId": "p", "datasetId": "d", "tableId": "t"}
            }
        },
    }
    mock_session.get.return_value = mock_poll_resp

    dest = run_query(
        "SELECT 1",
        quota_project_id="q",
        credentials_provider=mock_cp,
        user_agent="ua",
        session=mock_session,
    )
    assert dest == "p.d.t"
    mock_session.get.assert_called()
