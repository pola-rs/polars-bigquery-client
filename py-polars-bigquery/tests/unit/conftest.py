from unittest import mock

import pytest
import requests


@pytest.fixture(autouse=True)
def mock_google_auth_default():
    with mock.patch("google.auth.default") as mocked:
        mocked.return_value = (mock.MagicMock(), "mock-project")
        yield mocked


@pytest.fixture(autouse=True)
def no_real_http_calls(monkeypatch):
    """Ensure unit tests do not make real HTTP calls to the BigQuery API or external endpoints."""

    def guard_send(self, request, *args, **kwargs):
        raise RuntimeError(
            f"Unit test attempted to make a real HTTP request to {request.url}. "
            "Ensure all external API calls are properly mocked."
        )

    monkeypatch.setattr(requests.Session, "send", guard_send)
