import polars as pl
import pytest
from unittest.mock import MagicMock, patch
from polars_bigquery.core.run_query import run_query
from polars_bigquery.exceptions import BigQueryError

def test_run_query_success():
    mock_cp = MagicMock()
    mock_cp.return_value = ({"bearer_token": "fake-token"}, 12345)
    
    with patch("requests.post") as mock_post, \
         patch("requests.get") as mock_get, \
         patch("time.sleep"):
        
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
                        "tableId": "t"
                    }
                }
            }
        }
        
        mock_get.side_effect = [m1, m2]

        result = run_query("SELECT 1", "quota-project", mock_cp)
        
        assert result == "p.d.t"
        mock_post.assert_called_once()
        assert mock_get.call_count == 2
        
        # Verify headers
        called_headers = mock_post.call_args.kwargs["headers"]
        assert called_headers["Authorization"] == "Bearer fake-token"
        assert called_headers["x-goog-user-project"] == "quota-project"

def test_run_query_error():
    mock_cp = MagicMock()
    mock_cp.return_value = ({"bearer_token": "fake-token"}, 12345)
    
    with patch("requests.post") as mock_post, \
         patch("requests.get") as mock_get, \
         patch("time.sleep"):
        
        mock_post.return_value.json.return_value = {
            "jobReference": {"jobId": "job-123"}
        }
        mock_post.return_value.raise_for_status = MagicMock()
        
        mock_get.return_value.json.return_value = {
            "status": {
                "state": "DONE",
                "errorResult": {"message": "Something went wrong"}
            }
        }
        mock_get.return_value.raise_for_status = MagicMock()
        
        with pytest.raises(BigQueryError, match="Something went wrong"):
            run_query("SELECT 1", "quota-project", mock_cp)
