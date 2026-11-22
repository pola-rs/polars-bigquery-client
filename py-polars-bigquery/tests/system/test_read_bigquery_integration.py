from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import sys
import time
import tracemalloc
import uuid
from typing import Any

import polars as pl
import polars_bigquery
import pytest
from polars_bigquery.core import bigquery_rest, compiler
from polars_bigquery.core.executors import QueryExecutor
from polars_bigquery.core.plan import Filter, Project, QueryScan, Slice
from polars_bigquery.core.types import JSON_EXTENSION_NAME
from polars_bigquery.exceptions import (
    BigQueryError,
    UnsupportedTypeError,
    UnsupportedTypeWarning,
)

if sys.platform != "win32":
    import resource

logger = logging.getLogger(__name__)

_REST_QUERY_TIMEOUT_SECONDS = 120.0
_DELETE_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclasses.dataclass(frozen=True)
class EphemeralDataset:
    project: str
    dataset_id: str
    client: polars_bigquery.Client

    def table_name(self, name: str) -> str:
        return f"{self.project}.{self.dataset_id}.{name}"

    def _headers(self) -> dict[str, str]:
        return bigquery_rest._get_headers(
            quota_project_id=self.client.quota_project_id,
            credentials_provider=self.client.credentials_provider,
            user_agent=self.client._user_agent,
        )

    def create_table(self, name: str, body: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "tableReference": {
                "projectId": self.project,
                "datasetId": self.dataset_id,
                "tableId": name,
            }
        }
        if body:
            payload.update(body)
        url = (
            f"{bigquery_rest._BIGQUERY_ENDPOINT}/projects/{self.project}"
            f"/datasets/{self.dataset_id}/tables"
        )
        resp = self.client._rest_client.session.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=bigquery_rest.DEFAULT_TIMEOUT,
        )
        bigquery_rest._raise_for_bigquery_error(resp)
        return self.table_name(name)

    def run_sql_rest(
        self, sql: str, *, timeout_seconds: float = _REST_QUERY_TIMEOUT_SECONDS
    ) -> None:
        """Execute a DDL/DML statement via BigQuery REST without Arrow decoding."""
        queries_url = (
            f"{bigquery_rest._BIGQUERY_ENDPOINT}/projects/{self.project}/queries"
        )
        deadline = time.monotonic() + timeout_seconds
        resp = self.client._rest_client.session.post(
            queries_url,
            headers=self._headers(),
            json={
                "query": sql,
                "useLegacySql": False,
                "useQueryCache": False,
                "location": "US",
                # Supply a requestId so any HTTPAdapter retry on POST is idempotent.
                "requestId": f"req_{uuid.uuid4().hex}",
            },
            timeout=bigquery_rest.DEFAULT_TIMEOUT,
        )
        bigquery_rest._raise_for_bigquery_error(resp)
        body = resp.json()
        backoff_seconds = 0.5
        while not body.get("jobComplete", False):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                job_id = (body.get("jobReference") or {}).get("jobId", "<unknown>")
                msg = (
                    f"Timed out after {timeout_seconds:.1f}s waiting for "
                    f"BigQuery job {job_id} to complete"
                )
                raise TimeoutError(msg)
            job_id = body["jobReference"]["jobId"]
            time.sleep(min(backoff_seconds, remaining))
            backoff_seconds = min(backoff_seconds * 2.0, 5.0)
            poll = self.client._rest_client.session.get(
                f"{queries_url}/{job_id}?location=US",
                headers=self._headers(),
                timeout=bigquery_rest.DEFAULT_TIMEOUT,
            )
            bigquery_rest._raise_for_bigquery_error(poll)
            body = poll.json()
        if errors := body.get("errors"):
            raise BigQueryError(f"BigQuery query failed: {errors}")


@pytest.fixture(scope="module")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    with polars_bigquery.Client(quota_project_id=project) as bq_client:
        yield bq_client


@pytest.fixture(scope="module")
def ephemeral_dataset(client: polars_bigquery.Client):
    project = client.quota_project_id
    dataset_id = f"polars_bq_test_{uuid.uuid4().hex}"
    headers = bigquery_rest._get_headers(
        quota_project_id=project,
        credentials_provider=client.credentials_provider,
        user_agent=client._user_agent,
    )
    datasets_url = f"{bigquery_rest._BIGQUERY_ENDPOINT}/projects/{project}/datasets"
    resp = client._rest_client.session.post(
        datasets_url,
        headers=headers,
        json={
            "datasetReference": {
                "projectId": project,
                "datasetId": dataset_id,
            },
            "location": "US",
            "defaultTableExpirationMs": "3600000",
        },
        timeout=bigquery_rest.DEFAULT_TIMEOUT,
    )
    bigquery_rest._raise_for_bigquery_error(resp)

    ds = EphemeralDataset(project=project, dataset_id=dataset_id, client=client)
    try:
        yield ds
    finally:
        delete_url = f"{datasets_url}/{dataset_id}?deleteContents=true"
        delay = 0.5
        for attempt in range(4):
            delete_headers = bigquery_rest._get_headers(
                quota_project_id=project,
                credentials_provider=client.credentials_provider,
                user_agent=client._user_agent,
            )
            try:
                delete_resp = client._rest_client.session.delete(
                    delete_url,
                    headers=delete_headers,
                    timeout=bigquery_rest.DEFAULT_TIMEOUT,
                )
            except Exception:
                logger.warning(
                    "Transient exception deleting ephemeral dataset %s (attempt %d/4)",
                    dataset_id,
                    attempt + 1,
                    exc_info=True,
                )
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2.0
                continue
            if delete_resp.ok or delete_resp.status_code == 404:
                break
            if (
                delete_resp.status_code in _DELETE_RETRYABLE_STATUS_CODES
                and attempt < 3
            ):
                logger.warning(
                    "Retrying delete of ephemeral dataset %s after HTTP %d (attempt %d/4)",
                    dataset_id,
                    delete_resp.status_code,
                    attempt + 1,
                )
                time.sleep(delay)
                delay *= 2.0
                continue
            logger.warning(
                "Failed to delete ephemeral dataset %s: HTTP %d %s",
                dataset_id,
                delete_resp.status_code,
                delete_resp.text[:500],
            )
            break


@pytest.fixture(scope="module")
def all_supported_table(ephemeral_dataset: EphemeralDataset) -> str:
    table_id = ephemeral_dataset.table_name("all_supported_types")
    ephemeral_dataset.client.read_query(
        f"""
        CREATE TABLE `{table_id}` (
            row_id INT64 NOT NULL,
            flag BOOL,
            i_col INT64,
            f1 FLOAT64,
            f2 FLOAT64,
            n_col NUMERIC,
            s_col STRING,
            b_col BYTES,
            d_col DATE,
            dt_col DATETIME,
            t_col TIME,
            ts_col TIMESTAMP,
            j_col JSON,
            g_col GEOGRAPHY,
            j_arr ARRAY<JSON>,
            g_arr ARRAY<GEOGRAPHY>,
            dt_arr ARRAY<DATETIME>,
            int_arr ARRAY<INT64>,
            meta_struct STRUCT<k STRING, v INT64, j JSON, g GEOGRAPHY, dt DATETIME>
        ) AS
        SELECT * FROM UNNEST([
            STRUCT(
                1 AS row_id,
                TRUE AS flag,
                5 AS i_col,
                1.0 AS f1,
                5.0 AS f2,
                NUMERIC '1.5' AS n_col,
                'hello' AS s_col,
                b'abc' AS b_col,
                DATE '1970-01-01' AS d_col,
                DATETIME '1970-01-01 00:00:00' AS dt_col,
                TIME '00:00:00' AS t_col,
                TIMESTAMP '1970-01-01 00:00:00 UTC' AS ts_col,
                JSON '{{"a": 1}}' AS j_col,
                ST_GEOGFROMTEXT('POINT(1 2)') AS g_col,
                [JSON '{{"x": 1}}', JSON 'null'] AS j_arr,
                [ST_GEOGFROMTEXT('POINT(0 0)')] AS g_arr,
                [DATETIME '2024-01-01 00:00:00'] AS dt_arr,
                [1, 2] AS int_arr,
                STRUCT('k1' AS k, 10 AS v, JSON '{{"nested": true}}' AS j, ST_GEOGFROMTEXT('POINT(3 4)') AS g, DATETIME '2024-01-01 12:00:00' AS dt) AS meta_struct
            ),
            STRUCT(
                2,
                FALSE,
                7,
                5.0,
                1.0,
                NUMERIC '0',
                'HELLO',
                b'',
                DATE '1970-01-02',
                DATETIME '1970-01-01 00:00:00.000001',
                TIME '00:00:00.000001',
                TIMESTAMP '1970-01-01 00:00:00.000001 UTC',
                JSON '[1, 2]',
                ST_GEOGFROMTEXT('LINESTRING(0 0, 1 1)'),
                CAST([] AS ARRAY<JSON>),
                CAST([] AS ARRAY<GEOGRAPHY>),
                CAST([] AS ARRAY<DATETIME>),
                CAST([] AS ARRAY<INT64>),
                CAST(NULL AS STRUCT<k STRING, v INT64, j JSON, g GEOGRAPHY, dt DATETIME>)
            ),
            STRUCT(
                3,
                NULL,
                NULL,
                CAST('NaN' AS FLOAT64),
                5.0,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                [JSON '123'],
                [ST_GEOGFROMTEXT('POINT(5 6)')],
                [DATETIME '2024-02-01 00:00:00'],
                [3],
                STRUCT('k3', 30, JSON 'null', ST_GEOGFROMTEXT('POINT(7 8)'), DATETIME '2024-02-01 12:00:00')
            ),
            STRUCT(
                4,
                TRUE,
                0,
                5.0,
                CAST('NaN' AS FLOAT64),
                NUMERIC '1',
                '',
                b'\\x00\\xff',
                DATE '2024-06-15',
                DATETIME '2024-06-15 12:30:00',
                TIME '12:30:45',
                TIMESTAMP '2024-06-15 12:30:00 UTC',
                JSON '{{"b": 2}}',
                ST_GEOGFROMTEXT('POINT(9 10)'),
                [JSON '{{"y": 2}}'],
                [ST_GEOGFROMTEXT('POINT(1 1)')],
                [DATETIME '2024-06-15 12:30:00'],
                [4, 5],
                STRUCT('k4', 40, JSON '{{"m": 4}}', ST_GEOGFROMTEXT('POINT(2 2)'), DATETIME '2024-06-15 12:30:00')
            ),
            STRUCT(
                5,
                FALSE,
                4,
                CAST('NaN' AS FLOAT64),
                CAST('NaN' AS FLOAT64),
                NUMERIC '9007199254740993',
                'prefix_match',
                b'xyz',
                DATE '2024-01-01',
                DATETIME '2024-01-01 00:00:00',
                TIME '23:59:59',
                TIMESTAMP '2024-01-01 00:00:00 UTC',
                JSON 'true',
                ST_GEOGFROMTEXT('POINT(-1 -2)'),
                CAST([] AS ARRAY<JSON>),
                CAST([] AS ARRAY<GEOGRAPHY>),
                CAST([] AS ARRAY<DATETIME>),
                [6],
                NULL
            ),
            STRUCT(
                6,
                NULL,
                9007199254740993,
                9007199254740992.0,
                9007199254740992.0,
                NUMERIC '9007199254740992',
                'world',
                b'123',
                DATE '2023-12-31',
                DATETIME '2023-12-31 23:59:59',
                TIME '06:00:00',
                TIMESTAMP '2023-12-31 23:59:59 UTC',
                JSON '{{"big": 1}}',
                ST_GEOGFROMTEXT('POINT(10 20)'),
                [JSON 'false'],
                [ST_GEOGFROMTEXT('POINT(3 3)')],
                [DATETIME '2023-12-31 23:59:59'],
                [7],
                STRUCT('k6', 60, JSON '1', ST_GEOGFROMTEXT('POINT(4 4)'), DATETIME '2023-12-31 23:59:59')
            ),
            STRUCT(
                7,
                TRUE,
                1,
                CAST('Infinity' AS FLOAT64),
                NULL,
                NUMERIC '2.5',
                'inf_row',
                b'inf',
                DATE '2024-02-01',
                DATETIME '2024-02-01 00:00:00',
                TIME '01:02:03',
                TIMESTAMP '2024-02-01 00:00:00 UTC',
                NULL,
                NULL,
                CAST([] AS ARRAY<JSON>),
                CAST([] AS ARRAY<GEOGRAPHY>),
                CAST([] AS ARRAY<DATETIME>),
                [8],
                NULL
            ),
            STRUCT(
                8,
                FALSE,
                NULL,
                CAST('-Infinity' AS FLOAT64),
                0.0,
                NUMERIC '-3.5',
                'neg_inf_row',
                b'neg',
                DATE '2024-03-01',
                DATETIME '2024-03-01 00:00:00',
                TIME '02:03:04',
                TIMESTAMP '2024-03-01 00:00:00 UTC',
                JSON 'null',
                ST_GEOGFROMTEXT('POINT(0 0)'),
                CAST([] AS ARRAY<JSON>),
                CAST([] AS ARRAY<GEOGRAPHY>),
                CAST([] AS ARRAY<DATETIME>),
                [9],
                NULL
            ),
            STRUCT(
                9,
                NULL,
                2,
                NULL,
                CAST('NaN' AS FLOAT64),
                NULL,
                'null_f1_nan_f2',
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                CAST([] AS ARRAY<JSON>),
                CAST([] AS ARRAY<GEOGRAPHY>),
                CAST([] AS ARRAY<DATETIME>),
                CAST([] AS ARRAY<INT64>),
                NULL
            ),
            STRUCT(
                10,
                NULL,
                NULL,
                CAST('NaN' AS FLOAT64),
                NULL,
                NULL,
                'nan_f1_null_f2',
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                CAST([] AS ARRAY<JSON>),
                CAST([] AS ARRAY<GEOGRAPHY>),
                CAST([] AS ARRAY<DATETIME>),
                CAST([] AS ARRAY<INT64>),
                NULL
            ),
            STRUCT(
                11,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                'both_null',
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                CAST([] AS ARRAY<JSON>),
                CAST([] AS ARRAY<GEOGRAPHY>),
                CAST([] AS ARRAY<DATETIME>),
                CAST([] AS ARRAY<INT64>),
                NULL
            ),
            STRUCT(
                12,
                TRUE,
                5,
                5.0,
                5.0,
                NUMERIC '5',
                'eq_row',
                b'eq',
                DATE '2024-04-01',
                DATETIME '2024-04-01 00:00:00',
                TIME '04:05:06',
                TIMESTAMP '2024-04-01 00:00:00 UTC',
                JSON '{{"eq": true}}',
                ST_GEOGFROMTEXT('POINT(5 5)'),
                [JSON '{{"eq": 1}}'],
                [ST_GEOGFROMTEXT('POINT(5 5)')],
                [DATETIME '2024-04-01 00:00:00'],
                [12],
                STRUCT('k12', 120, JSON '{{"eq": 1}}', ST_GEOGFROMTEXT('POINT(5 5)'), DATETIME '2024-04-01 00:00:00')
            )
        ])
        """,
        use_query_cache=False,
    )
    return table_id


def test_read_bigquery_public_data_ordered(client):
    # Use a query so that the test can run using BigQuery sandbox quota.
    df = client.read_query(
        query="""
        SELECT SUM(number) AS total_born,
        name
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        GROUP BY name
        ORDER BY total_born DESC
        LIMIT 100
        """,
    )
    assert isinstance(df, pl.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height == 100  # rows
    assert df.width > 0  # columns
    assert df["total_born"].is_sorted(descending=True)


def test_read_bigquery_public_data_unordered(client):
    # Use a query so that the test can run using BigQuery sandbox quota.
    df = client.read_query(
        query="""
        SELECT * FROM `bigquery-public-data.utility_us.country_code_iso`
        """,
        maintain_order=False,
    )
    assert isinstance(df, pl.DataFrame)
    # Make sure we got all of the expected data, not just a subset.
    assert df.height > 200  # rows
    assert df.width > 0  # columns


def test_read_bigquery_dry_run(client):
    df = client.read_query(
        query="""
        SELECT name, number
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        WHERE state = 'WA'
        """,
        dry_run=True,
    )
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert df.columns == [
        "bigquerySchema",
        "projectId",
        "jobId",
        "location",
        "totalBytesProcessed",
        "cacheHit",
        "statementType",
        "creationTime",
    ]
    assert df["totalBytesProcessed"][0] > 0
    assert df["statementType"][0] == "SELECT"


def test_read_bigquery_large_results(client):
    """Verify completed >1 MB compressed LARGE_RESULTS queries stream all rows end-to-end."""
    sql = """
    SELECT
        id,
        CONCAT(
            TO_HEX(SHA256(CONCAT('a:', CAST(id AS STRING)))),
            TO_HEX(SHA256(CONCAT('b:', CAST(id AS STRING))))
        ) AS payload
    FROM UNNEST(GENERATE_ARRAY(1, 50000)) AS id
    """
    df = client.read_query(sql, maintain_order=False, use_query_cache=False)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 50_000
    assert df.schema == pl.Schema({"id": pl.Int64, "payload": pl.String})


def test_scan_mixed_table_omits_unsupported_and_read_table_rejects_interval(
    client: polars_bigquery.Client,
    ephemeral_dataset: EphemeralDataset,
):
    fields: list[dict[str, Any]] = [
        {"name": "id", "type": "INT64"},
        {"name": "name", "type": "STRING"},
        {"name": "amount", "type": "BIGNUMERIC"},
        {"name": "dur", "type": "INTERVAL"},
        {
            "name": "window_range",
            "type": "RANGE",
            "rangeElementType": {"type": "DATE"},
        },
        {
            "name": "meta",
            "type": "STRUCT",
            "fields": [
                {"name": "ok", "type": "INT64"},
                {"name": "bad", "type": "INTERVAL"},
            ],
        },
    ]
    # Include TIMESTAMP(12) if the GCP project has picosecond timestamps enabled.
    has_pico = True
    try:
        table_id = ephemeral_dataset.create_table(
            "mixed_unsupported",
            {
                "schema": {
                    "fields": [
                        *fields,
                        {
                            "name": "ts_pico",
                            "type": "TIMESTAMP",
                            "timestampPrecision": "12",
                        },
                    ]
                }
            },
        )
    except BigQueryError as exc:
        if "picosecond" not in str(exc).lower():
            raise
        has_pico = False
        table_id = ephemeral_dataset.create_table(
            "mixed_unsupported",
            {"schema": {"fields": fields}},
        )

    if has_pico:
        ephemeral_dataset.run_sql_rest(
            f"""
            INSERT INTO `{table_id}` (id, name, amount, dur, window_range, meta, ts_pico)
            VALUES
                (1, 'alice', BIGNUMERIC '12345678901234567890.12345', INTERVAL 1 DAY, RANGE(DATE '2024-01-01', DATE '2024-01-10'), STRUCT(10, INTERVAL 2 HOUR), TIMESTAMP '2024-01-01 00:00:00.000000000001 UTC'),
                (2, 'bob', BIGNUMERIC '99.5', INTERVAL 3 MONTH, RANGE(DATE '2024-02-01', DATE '2024-02-10'), STRUCT(20, INTERVAL 4 MINUTE), TIMESTAMP '2024-02-01 00:00:00.000000000002 UTC')
            """
        )
    else:
        ephemeral_dataset.run_sql_rest(
            f"""
            INSERT INTO `{table_id}` (id, name, amount, dur, window_range, meta)
            VALUES
                (1, 'alice', BIGNUMERIC '12345678901234567890.12345', INTERVAL 1 DAY, RANGE(DATE '2024-01-01', DATE '2024-01-10'), STRUCT(10, INTERVAL 2 HOUR)),
                (2, 'bob', BIGNUMERIC '99.5', INTERVAL 3 MONTH, RANGE(DATE '2024-02-01', DATE '2024-02-10'), STRUCT(20, INTERVAL 4 MINUTE))
            """
        )

    with pytest.warns(UnsupportedTypeWarning) as record:
        lf = client.scan_table(table_id)

    warn_msg = str(record[0].message)
    expected_skipped = 5 if has_pico else 4
    assert f"Skipping {expected_skipped} unsupported column(s)" in warn_msg
    assert "amount (BIGNUMERIC)" in warn_msg
    assert "dur (INTERVAL)" in warn_msg
    assert "window_range (RANGE)" in warn_msg
    assert "meta (INTERVAL at meta.bad)" in warn_msg
    if has_pico:
        assert "ts_pico (TIMESTAMP(precision=12))" in warn_msg

    assert lf.collect_schema() == pl.Schema({"id": pl.Int64(), "name": pl.String()})
    assert lf.select(pl.len()).collect().item() == 2

    df = lf.collect().sort("id")
    assert df.to_dict(as_series=False) == {"id": [1, 2], "name": ["alice", "bob"]}

    # Calling read_table() on an INTERVAL/BIGNUMERIC-containing table raises
    # UnsupportedTypeError (a BigQueryError subclass) from crates/arrow-bigquery
    # rather than aborting or panicking the Python process.
    with pytest.raises(UnsupportedTypeError):
        client.read_table(table_id)

    # Calling read_query() returning an INTERVAL raises UnsupportedTypeError safely.
    with pytest.raises(UnsupportedTypeError) as q_exc_info:
        client.read_query("SELECT INTERVAL 1 DAY AS dur", use_query_cache=False)
    assert "30920" in str(q_exc_info.value)
    assert "30921" in str(q_exc_info.value)

    # Calling read_query() returning an isolated BIGNUMERIC (Arrow Decimal256)
    # also raises UnsupportedTypeError safely instead of panicking in Rayon IPC decoding.
    with pytest.raises(UnsupportedTypeError, match="BIGNUMERIC"):
        client.read_query("SELECT BIGNUMERIC '1.5' AS amount", use_query_cache=False)


def test_scan_ingestion_time_partitioned_interval_only_tables(
    client: polars_bigquery.Client,
    ephemeral_dataset: EphemeralDataset,
):
    daily_table = ephemeral_dataset.create_table(
        "daily_interval_only",
        {
            "schema": {"fields": [{"name": "dur", "type": "INTERVAL"}]},
            "timePartitioning": {"type": "DAY"},
        },
    )
    hourly_table = ephemeral_dataset.create_table(
        "hourly_interval_only",
        {
            "schema": {"fields": [{"name": "dur", "type": "INTERVAL"}]},
            "timePartitioning": {"type": "HOUR"},
        },
    )

    for tbl in (daily_table, hourly_table):
        ephemeral_dataset.run_sql_rest(
            f"INSERT INTO `{tbl}` (dur) VALUES (INTERVAL 1 DAY), (INTERVAL 2 HOUR)"
        )

    with pytest.warns(UnsupportedTypeWarning, match=r"dur \(INTERVAL\)"):
        daily_lf = client.scan_table(daily_table)
    assert daily_lf.collect_schema() == pl.Schema(
        {
            "_PARTITIONDATE": pl.Date(),
            "_PARTITIONTIME": pl.Datetime("us", "UTC"),
        }
    )
    assert daily_lf.select(pl.len()).collect().item() == 2
    assert daily_lf.select("_PARTITIONDATE").collect().height == 2

    with pytest.warns(UnsupportedTypeWarning, match=r"dur \(INTERVAL\)"):
        hourly_lf = client.scan_table(hourly_table)
    assert hourly_lf.collect_schema() == pl.Schema(
        {"_PARTITIONTIME": pl.Datetime("us", "UTC")}
    )
    assert hourly_lf.select(pl.len()).collect().item() == 2
    assert hourly_lf.select("_PARTITIONTIME").collect().height == 2


def test_scan_empty_schema_table_returns_zero_rows_and_columns(
    client: polars_bigquery.Client,
    ephemeral_dataset: EphemeralDataset,
):
    empty_table = ephemeral_dataset.create_table("empty_schema_table")
    lf = client.scan_table(empty_table)
    assert lf.collect_schema() == pl.Schema({})
    df = lf.collect()
    assert df.shape == (0, 0)


def test_scan_table_schema_parity_with_read_table(
    client: polars_bigquery.Client,
    all_supported_table: str,
):
    eager_df = client.read_table(all_supported_table)
    lf = client.scan_table(all_supported_table)
    assert lf.collect_schema() == eager_df.schema

    scanned_df = lf.collect()
    assert scanned_df.schema == eager_df.schema
    assert scanned_df.sort("row_id").equals(eager_df.sort("row_id"))


def test_scan_table_differential_pushdown_parity_all_14_types(
    client: polars_bigquery.Client,
    all_supported_table: str,
):
    eager_df = client.read_table(all_supported_table).sort("row_id")
    lf = client.scan_table(all_supported_table)
    schema = lf.collect_schema()

    test_predicates: list[tuple[str, pl.Expr, bool]] = [
        # (name, expr, expect_pushed_non_empty)
        ("bool_eq", pl.col("flag") == True, True),
        ("bool_fill_zero", pl.col("flag").fill_null(strategy="zero"), True),
        ("bool_fill_one", pl.col("flag").fill_null(strategy="one"), True),
        # INT64 float predicates (positive, negated, is_null, fill_null)
        ("int_is_nan", pl.col("i_col").is_nan(), True),
        ("int_is_not_nan", pl.col("i_col").is_not_nan(), True),
        ("int_is_inf", pl.col("i_col").is_infinite(), True),
        ("int_is_finite", pl.col("i_col").is_finite(), True),
        ("int_not_is_nan", ~pl.col("i_col").is_nan(), True),
        ("int_is_nan_is_null", pl.col("i_col").is_nan().is_null(), True),
        ("int_is_nan_fill_true", pl.col("i_col").is_nan().fill_null(True), True),
        # INT64 bitwise (skipped, sibling still pushes)
        ("int_bitwise_and", (pl.col("i_col") & 4) == 4, False),
        ("int_bitwise_or", (pl.col("i_col") | 1) == 5, False),
        (
            "int_bitwise_with_sibling",
            ((pl.col("i_col") & 4) == 4) & (pl.col("row_id") <= 6),
            True,
        ),
        # INT64(2^53+1) vs FLOAT64(2^53) (allowed and matches Polars)
        ("int_vs_float_2_53_eq", pl.col("i_col") == 9007199254740992.0, True),
        ("int_vs_float_2_53_gt", pl.col("i_col") > 9007199254740992.0, True),
        # NUMERIC(2^53+1) vs FLOAT64(2^53) (skipped to prevent silent row loss)
        ("numeric_vs_float_2_53_eq", pl.col("n_col") == 9007199254740992.0, False),
        ("numeric_vs_float_2_53_gt", pl.col("n_col") > 9007199254740992.0, False),
        ("numeric_vs_int_eq", pl.col("n_col") == 9007199254740993, True),
        ("numeric_fill_zero_eq", pl.col("n_col").fill_null(strategy="zero") == 0, True),
        ("numeric_fill_one_eq", pl.col("n_col").fill_null(strategy="one") == 1, True),
        # FLOAT64 NaN/Inf/NULL comparisons (Cases B, C, D across all 6 ops + negated)
        ("f1_eq_5", pl.col("f1") == 5.0, True),
        ("f1_ne_5", pl.col("f1") != 5.0, True),
        ("f1_gt_5", pl.col("f1") > 5.0, True),
        ("f1_ge_5", pl.col("f1") >= 5.0, True),
        ("f1_lt_5", pl.col("f1") < 5.0, True),
        ("f1_le_5", pl.col("f1") <= 5.0, True),
        ("5_lt_f1", pl.lit(5.0) < pl.col("f1"), True),
        ("5_le_f1", pl.lit(5.0) <= pl.col("f1"), True),
        ("not_f1_gt_5", ~(pl.col("f1") > 5.0), True),
        ("f1_gt_i", pl.col("f1") > pl.col("i_col"), True),
        ("f1_ge_i", pl.col("f1") >= pl.col("i_col"), True),
        ("i_lt_f1", pl.col("i_col") < pl.col("f1"), True),
        ("i_le_f1", pl.col("i_col") <= pl.col("f1"), True),
        ("f1_eq_f2", pl.col("f1") == pl.col("f2"), True),
        ("f1_ne_f2", pl.col("f1") != pl.col("f2"), True),
        ("f1_gt_f2", pl.col("f1") > pl.col("f2"), True),
        ("f1_ge_f2", pl.col("f1") >= pl.col("f2"), True),
        ("f1_lt_f2", pl.col("f1") < pl.col("f2"), True),
        ("f1_le_f2", pl.col("f1") <= pl.col("f2"), True),
        ("not_f1_eq_f2", ~(pl.col("f1") == pl.col("f2")), True),
        ("not_f1_ne_f2", ~(pl.col("f1") != pl.col("f2")), True),
        ("f1_fill_zero_ge_0", pl.col("f1").fill_null(strategy="zero") >= 0.0, True),
        ("f1_fill_one_gt_0", pl.col("f1").fill_null(strategy="one") > 0.0, True),
        # STRING & BYTES fill_null(strategy=...)
        ("str_fill_zero", pl.col("s_col").fill_null(strategy="zero") == "", True),
        ("str_starts_with", pl.col("s_col").str.starts_with("prefix"), True),
        (
            "bytes_fill_zero_eq_self",
            pl.col("b_col").fill_null(strategy="zero") == pl.col("b_col"),
            True,
        ),
        # Temporal IFNULL expressions emitted by sql/temporal.py
        (
            "date_fill_zero",
            pl.col("d_col").fill_null(strategy="zero") == dt.date(1970, 1, 1),
            True,
        ),
        (
            "date_fill_one",
            pl.col("d_col").fill_null(strategy="one") == dt.date(1970, 1, 2),
            True,
        ),
        (
            "dt_fill_zero",
            pl.col("dt_col").fill_null(strategy="zero")
            == dt.datetime(1970, 1, 1, 0, 0, 0),  # noqa: DTZ001
            True,
        ),
        (
            "dt_fill_one",
            pl.col("dt_col").fill_null(strategy="one")
            == dt.datetime(1970, 1, 1, 0, 0, 0, 1),  # noqa: DTZ001
            True,
        ),
        (
            "ts_fill_zero",
            pl.col("ts_col").fill_null(strategy="zero")
            == dt.datetime(1970, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            True,
        ),
        (
            "ts_fill_one",
            pl.col("ts_col").fill_null(strategy="one")
            == dt.datetime(1970, 1, 1, 0, 0, 0, 1, tzinfo=dt.timezone.utc),
            True,
        ),
        (
            "time_fill_zero_eq_self",
            pl.col("t_col").fill_null(strategy="zero") == pl.col("t_col"),
            True,
        ),
        (
            "time_fill_one_skipped",
            pl.col("t_col").fill_null(strategy="one") == dt.time(0, 0, 0, 0),
            False,
        ),
        # OPAQUE types: IS [NOT] NULL pushed, .ext.storage() falls back cleanly
        ("json_is_not_null", pl.col("j_col").is_not_null(), True),
        ("geo_is_null", pl.col("g_col").is_null(), True),
        ("array_is_not_null", pl.col("int_arr").is_not_null(), True),
        ("struct_is_null", pl.col("meta_struct").is_null(), True),
        (
            "json_ext_storage_starts_with",
            pl.col("j_col").ext.storage().str.starts_with("{"),
            False,
        ),
        (
            "geo_ext_storage_starts_with",
            pl.col("g_col").ext.storage().str.starts_with("POINT"),
            False,
        ),
    ]

    for name, expr, expect_pushed in test_predicates:
        sql = compiler.predicate_to_row_restriction(expr, schema=schema)
        if expect_pushed:
            assert sql != "", f"Expected non-empty row_restriction for {name}"
        else:
            assert sql == "", f"Expected empty row_restriction for {name}, got {sql!r}"

        expected = eager_df.filter(expr).sort("row_id")
        actual = lf.filter(expr).collect().sort("row_id")
        assert actual["row_id"].to_list() == expected["row_id"].to_list(), (
            f"Row mismatch for {name} (sql={sql!r}): "
            f"actual={actual['row_id'].to_list()} vs expected={expected['row_id'].to_list()}"
        )


def test_scan_table_mode_b_as_storage_live_parity(
    client: polars_bigquery.Client,
    all_supported_table: str,
):
    pl.register_extension_type(JSON_EXTENSION_NAME, as_storage=True)
    try:
        eager_df = client.read_table(all_supported_table).sort("row_id")
        lf = client.scan_table(all_supported_table)
        assert lf.collect_schema()["j_col"] == pl.String()
        assert lf.collect_schema()["j_arr"] == pl.List(pl.String())

        expr = pl.col("j_col").str.starts_with("{") & (pl.col("row_id") <= 6)
        actual = lf.filter(expr).collect().sort("row_id")
        expected = eager_df.filter(expr).sort("row_id")
        assert actual["row_id"].to_list() == expected["row_id"].to_list()
        assert actual.height > 0
    finally:
        pl.unregister_extension_type(JSON_EXTENSION_NAME)


def test_scan_query_public_data_unordered(client):
    lf = client.scan_query(
        query="""
        SELECT * FROM `bigquery-public-data.utility_us.country_code_iso`
        """,
        maintain_order=False,
    )
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert isinstance(df, pl.DataFrame)
    assert df.height > 200
    assert df.width > 0


def test_scan_query_public_data_ordered(client):
    lf = client.scan_query(
        query="""
        SELECT SUM(number) AS total_born,
        name
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        GROUP BY name
        ORDER BY total_born DESC
        LIMIT 100
        """,
        maintain_order=True,
    )
    df = lf.collect()
    assert df.height == 100
    assert df.width > 0
    assert df["total_born"].is_sorted(descending=True)


def test_scan_query_with_schema_pushes_projection_and_filter(client):
    lf = client.scan_query(
        query="""
        SELECT name, number, state
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        WHERE year = 2013
        """,
        schema={
            "name": pl.String,
            "number": pl.Int64,
            "state": pl.String,
        },
        maintain_order=False,
    )
    df = lf.filter(pl.col("state") == "WA").select("name").limit(10).collect()
    assert df.height == 10
    assert df.columns == ["name"]
    assert df["name"].null_count() == 0


def test_scan_query_without_schema_infers_schema_and_filters_locally(client):
    lf = client.scan_query(
        query="""
        SELECT * FROM `bigquery-public-data.utility_us.country_code_iso`
        """
    )
    df = lf.select(pl.first()).limit(5).collect()
    assert df.height == 5
    assert df.width == 1
    assert df.schema.names()[0] in lf.collect_schema().names()


def test_scan_query_empty_results(client):
    lf = client.scan_query(
        query="SELECT 1 AS value LIMIT 0",
        schema={"value": pl.Int64},
    )
    df = lf.collect()
    assert df.schema == pl.Schema({"value": pl.Int64})
    assert df.is_empty()


def test_scan_query_without_schema_pushes_projection_and_filter(client):
    lf = client.scan_query(
        query="""
        SELECT name, number, state
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        WHERE year = 2013 -- trailing comment
        """,
        maintain_order=False,
    )
    df = lf.filter(pl.col("state") == "WA").select("name").limit(10).collect()
    assert df.height == 10
    assert df.columns == ["name"]
    assert df["name"].null_count() == 0


def test_scan_query_anonymous_column_unordered(client):
    # Plain scan without pruning/filter should not wrap, so anonymous f0_ works.
    lf = client.scan_query(query="SELECT 1", maintain_order=False)
    df = lf.collect()
    assert df.to_dict(as_series=False) == {"f0_": [1]}

    # Filter without column pruning omits |> SELECT, preserving anonymous f0_.
    lf_filtered = client.scan_query(
        query="""
        SELECT COUNT(*), state
        FROM `bigquery-public-data.usa_names.usa_1910_2013`
        WHERE year = 2013
        GROUP BY state
        """,
        maintain_order=False,
    )
    df_wa = lf_filtered.filter(pl.col("state") == "WA").collect()
    assert df_wa.height == 1
    assert df_wa["state"].to_list() == ["WA"]
    assert df_wa["f0_"][0] > 0


def test_scan_query_invalid_sql_raises_bigquery_error(client):
    with pytest.raises(BigQueryError):
        client.scan_query("SELECT FROM WHERE")


def test_scan_query_anonymous_column_explicit_projection_raises(client):
    lf = client.scan_query(query="SELECT 1, 2 AS named", maintain_order=False)
    with pytest.raises(
        (BigQueryError, pl.exceptions.ComputeError), match="Unrecognized name"
    ):
        lf.select("f0_").collect()


def test_scan_query_with_and_recursive_ctes(client):
    cte_sql = """
    WITH nums AS (
        SELECT n, CAST(n AS STRING) AS label
        FROM UNNEST(GENERATE_ARRAY(1, 5)) AS n
    )
    SELECT n, label FROM nums
    """
    df_cte = (
        client.scan_query(cte_sql, maintain_order=False)
        .filter(pl.col("n") >= 3)
        .select("label")
        .limit(2)
        .collect()
    )
    assert df_cte.columns == ["label"]
    assert df_cte.height == 2
    assert set(df_cte["label"].to_list()) <= {"3", "4", "5"}

    rec_sql = """
    WITH RECURSIVE seq AS (
        SELECT 1 AS n
        UNION ALL
        SELECT n + 1 FROM seq WHERE n < 5
    )
    SELECT n, n * 10 AS scaled FROM seq
    """
    df_rec = (
        client.scan_query(rec_sql, maintain_order=False)
        .filter(pl.col("n") > 2)
        .select("scaled")
        .collect()
    )
    assert sorted(df_rec["scaled"].to_list()) == [30, 40, 50]


@pytest.mark.parametrize(
    ("set_op", "expected"),
    [
        ("UNION ALL", [3, 3, 4]),
        ("INTERSECT DISTINCT", [3]),
        ("EXCEPT DISTINCT", [4]),
    ],
)
def test_scan_query_set_operations_pipe_precedence(client, set_op, expected):
    sql = f"""
    SELECT x FROM UNNEST([1, 2, 3, 4]) AS x
    {set_op}
    SELECT x FROM UNNEST([1, 2, 3]) AS x
    """
    df = client.scan_query(sql, maintain_order=False).filter(pl.col("x") >= 3).collect()
    assert sorted(df["x"].to_list()) == expected


def test_scan_query_create_temp_function_pipe_pushdown(client):
    sql = """
    CREATE TEMP FUNCTION double_it(x INT64) RETURNS INT64 AS (x * 2);
    SELECT n, double_it(n) AS doubled FROM UNNEST([1, 2, 3, 4]) AS n
    """
    df = (
        client.scan_query(sql, maintain_order=False)
        .filter(pl.col("doubled") >= 6)
        .select("n")
        .collect()
    )
    assert sorted(df["n"].to_list()) == [3, 4]


def test_scan_query_trailing_order_by_limit(client):
    sql = """
    SELECT n, n * 2 AS doubled
    FROM UNNEST(GENERATE_ARRAY(1, 10)) AS n
    ORDER BY n DESC
    LIMIT 5
    """
    df = (
        client.scan_query(sql, maintain_order=False)
        .filter(pl.col("n") >= 8)
        .select("doubled")
        .limit(2)
        .collect()
    )
    assert df.columns == ["doubled"]
    assert df.height == 2
    assert set(df["doubled"].to_list()) <= {16, 18, 20}


def test_scan_query_user_pipe_syntax_composed(client):
    sql = """
    FROM UNNEST(GENERATE_ARRAY(1, 10)) AS n
    |> SELECT n, n * 10 AS scaled
    |> WHERE n >= 5
    """
    df = (
        client.scan_query(sql, maintain_order=False)
        .filter(pl.col("n") <= 7)
        .select("scaled")
        .collect()
    )
    assert sorted(df["scaled"].to_list()) == [50, 60, 70]


def test_scan_query_struct_value_tables(client):
    sql_as_struct = (
        "SELECT AS STRUCT 1 AS a, 'x' AS b UNION ALL SELECT AS STRUCT 2 AS a, 'y' AS b"
    )
    df_struct = (
        client.scan_query(sql_as_struct, maintain_order=False)
        .filter(pl.col("a") == 2)
        .select("b")
        .collect()
    )
    assert df_struct.to_dict(as_series=False) == {"b": ["y"]}

    sql_as_value = "SELECT AS VALUE STRUCT(1 AS a, 'x' AS b)"
    df_val = (
        client.scan_query(sql_as_value, maintain_order=False)
        .filter(pl.col("a") == 1)
        .select("b")
        .collect()
    )
    assert df_val.to_dict(as_series=False) == {"b": ["x"]}


@pytest.mark.parametrize(
    ("sql", "expected_cols", "second_col"),
    [
        ("SELECT 1 AS a, 2 AS a", ["a", "a_1"], "a_1"),
        ("SELECT 1 AS a, 2 AS A", ["a", "A_1"], "A_1"),
    ],
)
def test_scan_query_duplicate_and_case_colliding_aliases(
    client, sql, expected_cols, second_col
):
    lf = client.scan_query(sql, maintain_order=False)
    assert lf.collect_schema().names() == expected_cols

    # |> LIMIT 1 alone succeeds without naming duplicate columns.
    assert lf.limit(1).collect().height == 1

    # |> WHERE on duplicate alias raises ambiguous name error.
    with pytest.raises(
        (BigQueryError, pl.exceptions.ComputeError), match=r"(?i)ambiguous"
    ):
        lf.filter(pl.col("a") == 1).collect()

    # |> SELECT referencing auto-suffixed column raises Unrecognized name.
    with pytest.raises(
        (BigQueryError, pl.exceptions.ComputeError), match="Unrecognized name"
    ):
        lf.select(second_col).collect()


def test_scan_query_limit_and_executor_offset(client):
    sql = "SELECT n FROM UNNEST(GENERATE_ARRAY(1, 20)) AS n"
    lf = client.scan_query(sql, maintain_order=False)

    df_lim = lf.limit(5).collect()
    assert df_lim.height == 5

    df_lim_filt = lf.limit(10).filter(pl.col("n") > 5).collect()
    assert df_lim_filt.height <= 10
    assert all(val > 5 for val in df_lim_filt["n"].to_list())

    # Direct QueryExecutor verification of GoogleSQL |> LIMIT length OFFSET offset
    executor = QueryExecutor(client._arrow_client)
    scan = QueryScan(
        raw_query="SELECT n FROM UNNEST(GENERATE_ARRAY(1, 10)) AS n ORDER BY n",
        columns=("n",),
        maintain_order=False,
        schema=pl.Schema({"n": pl.Int64()}),
    )
    plan_slice_offset = Project(
        input=Filter(
            input=Slice(input=scan, offset=2, length=5),
            predicate=pl.col("n") >= 4,
        ),
        columns=("n",),
    )
    batches = list(executor.execute(plan_slice_offset))
    res = pl.concat(batches)
    assert res.columns == ["n"]
    assert all(val >= 4 for val in res["n"].to_list())
    assert res.height <= 5


@pytest.mark.parametrize("maintain_order", [True, False])
def test_scan_query_large_results_sink_parquet(client, tmp_path, maintain_order):
    sql = """
    SELECT
        id,
        CONCAT(
            TO_HEX(SHA256(CONCAT('a:', CAST(id AS STRING)))),
            TO_HEX(SHA256(CONCAT('b:', CAST(id AS STRING))))
        ) AS payload
    FROM UNNEST(GENERATE_ARRAY(1, 50000)) AS id
    """
    out_path = tmp_path / f"large_{maintain_order}.parquet"
    lf = client.scan_query(
        sql,
        maintain_order=maintain_order,
        use_query_cache=False,
    )

    tracemalloc.start()
    try:
        lf.filter(pl.col("id") <= 25_000).sink_parquet(out_path)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if sys.platform != "win32":
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info(
            "Peak RSS after sink_parquet (maintain_order=%s): %d", maintain_order, rss
        )

    assert peak < 64 * 1024 * 1024
    written = pl.read_parquet(out_path)
    assert written.height == 25_000
    assert written.schema == pl.Schema({"id": pl.Int64(), "payload": pl.String()})


def test_scan_query_partial_explicit_schema(client):
    df = (
        client.scan_query(
            "SELECT * FROM `bigquery-public-data.usa_names.usa_1910_2013`",
            schema={"state": pl.String},
            maintain_order=False,
        )
        .limit(10)
        .collect()
    )
    assert df.columns == ["state"]
    assert df.height == 10


def test_scan_query_dry_run_bytes_processed_pruning_parity(client):
    def dry_run_bytes(sql: str) -> int:
        return int(client.read_query(sql, dry_run=True)["totalBytesProcessed"][0])

    # Column pruning & zero-column parity on usa_names
    base_names = "SELECT * FROM `bigquery-public-data.usa_names.usa_1910_2013`"
    assert dry_run_bytes(f"{base_names}\n|> SELECT `name`") == dry_run_bytes(
        f"SELECT `name` FROM ({base_names})"
    )
    assert dry_run_bytes(f"{base_names}\n|> SELECT 1 AS `_polars_bigquery_empty`") == 0

    # Partition pruning & projection parity on pypi.file_downloads
    base_pypi = "SELECT * FROM `bigquery-public-data.pypi.file_downloads`"
    pred = "(`timestamp` >= TIMESTAMP('2024-01-01 00:00:00+00') AND `timestamp` < TIMESTAMP('2024-01-02 00:00:00+00'))"
    pipe_pypi = dry_run_bytes(
        f"{base_pypi}\n|> WHERE {pred}\n|> SELECT `file`, `timestamp`"
    )
    subquery_pypi = dry_run_bytes(
        f"SELECT `file`, `timestamp` FROM ({base_pypi}) WHERE {pred}"
    )
    assert pipe_pypi == subquery_pypi
    assert dry_run_bytes(f"{base_pypi}\n|> SELECT 1 AS `_polars_bigquery_empty`") == 0


def test_scan_table_zero_column_hour_ingestion_time_partitioned(client):
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    dataset = f"_polars_bq_test_{uuid.uuid4().hex[:12]}"
    table_id = f"{project}.{dataset}.hour_part_tbl"

    client.read_query(
        f"CREATE SCHEMA `{project}.{dataset}` OPTIONS(default_table_expiration_days=1)"
    )
    try:
        client.read_query(
            f"""
            CREATE TABLE `{table_id}` (id INT64, flag BOOL, note STRING)
            PARTITION BY TIMESTAMP_TRUNC(_PARTITIONTIME, HOUR)
            """
        )
        client.read_query(
            f"INSERT INTO `{table_id}` (id, flag, note) VALUES (1, TRUE, 'a'), (2, FALSE, 'b')"
        )
        df = client.scan_table(table_id).drop(pl.all()).collect()
        assert df.width == 0
    finally:
        client.read_query(f"DROP SCHEMA IF EXISTS `{project}.{dataset}` CASCADE")
