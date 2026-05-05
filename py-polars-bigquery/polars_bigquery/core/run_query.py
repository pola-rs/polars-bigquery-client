"""Utilities for running a query in BigQuery and getting the results as a table.

TODO(tswast): Implement this in Rust using jobs.query and the
JOB_CREATION_OPTIONAL parameter to improve latency in small query results.
"""

import time

import polars as pl
import requests

import polars_bigquery.core.version
import polars_bigquery.exceptions


_BIGQUERY_ENDPOINT = "https://bigquery.googleapis.com/bigquery/v2"



def _get_user_agent() -> str:
    return f"polars-bigquery/{polars_bigquery.core.version.__version__}"


def _get_jobs_insert_url(quota_project_id: str) -> str:
    return f"{_BIGQUERY_ENDPOINT}/projects/{quota_project_id}/jobs"


def _get_jobs_insert_body(query: str) -> dict:
    # TODO(tswast): include a job ID for safer API-level retries.
    return {
        "configuration": {
            "query": {
                "query": query,
                "useLegacySql": False,
            }
        }
    }


def _get_jobs_get_url(job_id: str, quota_project_id: str) -> str:
    return f"{_BIGQUERY_ENDPOINT}/projects/{quota_project_id}/jobs/{job_id}"


def _get_query_results_url(job_id: str, quota_project_id: str) -> str:
    # Set maxResults=0 to only use this endpoint to wait for job to finish.
    return f"{_BIGQUERY_ENDPOINT}/projects/{quota_project_id}/queries/{job_id}/results?maxResults=0"


def _wait_for_job(job_id: str, quota_project_id: str, headers: dict) -> dict:
    while True:
        response = requests.get(
            _get_jobs_get_url(job_id, quota_project_id), headers=headers
        )
        response.raise_for_status()
        job = response.json()

        if job["status"]["state"] == "DONE":
            if "errorResult" in job["status"]:
                # TODO(tswast): Retry jobs that fail for a retriable reason.
                raise polars_bigquery.exceptions.BigQueryError(
                    job["status"]["errorResult"]["message"]
                )
            return job

        # jobs.getQueryResults waits about 10s or until the query finishes.
        requests.get(_get_query_results_url(job_id, quota_project_id), headers=headers).json()


def run_query(
    query: str, quota_project_id: str, credentials_provider: pl.CredentialProviderGCP
) -> str:
    """Run a query and return the destination table from the job resource."""
    token_data, _ = credentials_provider()
    token = token_data["bearer_token"]
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": _get_user_agent(),
        "Content-Type": "application/json",
        "x-goog-user-project": quota_project_id,
    }

    # 1. Insert the job
    insert_url = _get_jobs_insert_url(quota_project_id)
    body = _get_jobs_insert_body(query)
    response = requests.post(insert_url, headers=headers, json=body)
    response.raise_for_status()
    job_resource = response.json()
    job_id = job_resource["jobReference"]["jobId"]

    # 2. Wait for the job to complete
    job = _wait_for_job(job_id, quota_project_id, headers)

    # 3. Return the destination table ID
    dest = job["configuration"]["query"]["destinationTable"]
    return f"{dest['projectId']}.{dest['datasetId']}.{dest['tableId']}"
