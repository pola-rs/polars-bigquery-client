from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def mock_google_auth_default():
    with mock.patch("google.auth.default") as mocked:
        mocked.return_value = (mock.MagicMock(), "mock-project")
        yield mocked
