import logging
import os
import sys
import tracemalloc
import uuid

import polars as pl
import polars_bigquery
import pytest
from polars_bigquery.core.executors import QueryExecutor
from polars_bigquery.core.plan import Filter, Project, QueryScan, Slice
from polars_bigquery.exceptions import BigQueryError

if sys.platform != "win32":
    import resource

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    with polars_bigquery.Client(quota_project_id=project) as client:
        yield client


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
