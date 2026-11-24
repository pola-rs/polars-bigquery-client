# BigQuery Schema Mapping, Column Skipping, and Plan Type Propagation for Polars 2.0+

*   **Author:** @tswast
*   **Status:** Proposed
*   **Date:** 2026-11-24
*   **Related design docs:** [Row Predicate Pushdown](./polars_bigquery_scan_table_row_predicate_pushdown.md), [`Client.scan_query()`](./polars_bigquery_scan_query.md)

--------------------------------------------------------------------------------

## Overview

When calling `Client.scan_table()`, Polars requires the full output schema up front, before reading rows over the BigQuery Storage Read API. Unlike eager `Client.read_table()` (which infers the schema directly from BigQuery's Arrow IPC stream), `Client.scan_table()` builds a `polars.Schema` from the JSON response returned by the BigQuery `tables.get` REST API (unless the caller supplies an explicit `schema=`). Similarly, `Client.scan_query()` discovers its initial schema from the JSON `bigquerySchema` payload returned by a `jobs.query` dry run.

In Polars 2.0+, Arrow C Stream validation rejects collection when the registered schema disagrees with the Arrow IPC stream returned at execution time. This design covers the schema-mapping layer in `polars_bigquery`:

1.  Aligning extracted Polars dtypes with BigQuery's Arrow IPC output (uppercase `"UTC"` timezone, Arrow extension types for `JSON` and `GEOGRAPHY`, and outer-wrapped `List(String())` extensions for `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>`).
2.  Distinguishing table-scan extraction (which skips unsupported top-level columns with a warning so mixed tables remain readable) from query-scan extraction (which rejects unsupported columns immediately so verbatim `SELECT` outputs are never silently dropped).
3.  Preserving REST field parameters across `REPEATED` (`ARRAY`) recursion and validating `timestampPrecision`.
4.  Restricting the `_PARTITIONDATE` pseudo-column to daily (`DAY`) ingestion-time partitioned tables.
5.  Short-circuiting empty tables (`0` columns) and explicit `schema={}` without opening a Storage Read API session.
6.  Carrying an immutable column-name-to-`BQType` mapping on `TableScan` plan nodes so `ReadApiExecutor` retains authoritative BigQuery types even when Polars loads extension columns as plain storage (`pl.String` / `pl.List(pl.String)`).

## The Problem

This repository contains three layers: `crates/arrow-bigquery` (Rust Storage Read API and `jobs.query` Arrow IPC client), `py-arrow-bigquery` (PyO3 bindings exposing Arrow C Stream export), and `py-polars-bigquery` (Python Polars connector). Prior PRs introduced `UnsupportedTypeError` (`#116`), stripped `google:sqlType:datetime` extension metadata during Arrow decoding so `DATETIME` matches `pl.Datetime("us")` (`#123`), and added `BQType`, `build_extension_dtype`, and `polars_bigquery.dtypes.{JSON,GEOGRAPHY}` (`#121`).

Even with those prerequisites merged, `core/schema.py` broke on real-world BigQuery tables under Polars 2.0+ for five separate reasons:

1.  **`TIMESTAMP` timezone casing mismatch.** BigQuery's Arrow IPC stream marks `TIMESTAMP` columns and `_PARTITIONTIME` with uppercase `"UTC"`. The previous schema extractor emitted lowercase `time_zone="utc"`, causing `scan_table().collect_schema()` to disagree with `read_table().schema`.
2.  **`JSON` and `GEOGRAPHY` extension mismatch.** BigQuery tags `JSON` with Arrow extension name `google:sqlType:json` and `GEOGRAPHY` with `google:sqlType:geography` and metadata `'{"encoding": "WKT"}'`. For `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>`, BigQuery places the extension metadata on the outer Arrow `List` field rather than the inner `String` element. Polars 2.0+ loads unregistered Arrow extension types as `pl.Extension(...)` by default. Because `core/schema.py` previously mapped `JSON` and `GEOGRAPHY` to plain `pl.String`, collecting a `scan_table()` call on any table with those columns failed Polars schema validation (`SchemaError`).
3.  **Unscannable mixed tables.** A single unsupported column anywhere in a BigQuery table crashed the entire `scan_table()` call instead of allowing users to scan the remaining supported columns:
    *   `BIGNUMERIC` / `BIGDECIMAL` maps to precision 76 (exceeding Polars' maximum `Decimal` precision of 38).
    *   `INTERVAL` requires Arrow `Interval(MonthDayNano)`, which causes FFI failure (`UnsupportedTypeError`).
    *   `RANGE<T>` has no Polars representation.
    *   Picosecond / non-microsecond `TIMESTAMP` (`timestampPrecision` other than `6` / `"6"` / unset) exceeds microsecond precision assumptions.
    *   Future BigQuery data types that have no mapping yet would fail the entire scan unless omitted individually.
4.  **Dropped REST field parameters and incorrect partition pseudo-columns.**
    *   When recursing into `mode="REPEATED"` (`ARRAY`), the previous extractor rebuilt a synthetic dict containing only `name`, `type`, `fields`, and `mode`. That dropped `timestampPrecision` and any other field properties, letting `ARRAY<TIMESTAMP>` with non-microsecond precision slip through as microsecond `List(Datetime("us", "UTC"))`.
    *   The extractor unconditionally attached `_PARTITIONDATE` to every ingestion-time partitioned table. BigQuery only exposes `_PARTITIONDATE` on `DAY`-partitioned tables; `HOUR`, `MONTH`, and `YEAR` ingestion-time partitioned tables expose only `_PARTITIONTIME`.
5.  **Zero-column tables failing session creation.** BigQuery's `CreateReadSession` RPC rejects empty (`0`-column) tables with `FAILED_PRECONDITION`. Passing explicit `schema={}` raised `ValueError` eagerly, and scanning a discovered `0`-column table threw `BigQueryError` when `require_non_empty=True`.

A sixth structural hazard existed in downstream metadata handling: when a user configures Polars to load Arrow extension columns as plain storage (`register_extension_type(..., as_storage=True)` or `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`), `JSON` and `GEOGRAPHY` appear in `pl.Schema` as `pl.String()`, and `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` appear as `pl.List(pl.String())`. Inspecting `pl.Schema` alone loses the distinction between a real `STRING` column and an extension column loaded as storage, affecting zero-column fallback column ranking now and future SQL predicate compilation later.

## Goals

*   **Schema parity with Polars 2.0+ Arrow C Streams:** For all 14 supported `BQType` values (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`), ensure `scan_table().collect_schema()` matches the schema imported from BigQuery's Arrow IPC stream, including uppercase `"UTC"` and default/registered extension handling for scalar and `REPEATED` `JSON` and `GEOGRAPHY`.
*   **Graceful omission on mixed tables (`scan_table`):** Skip top-level columns whose own type or nested subfield is unsupported (`BIGNUMERIC`, `INTERVAL`, `RANGE`, non-microsecond `TIMESTAMP`, or unrecognized future BigQuery types), emitting one `UnsupportedTypeWarning` listing up to five omitted columns with their nested field paths (`col.sub.leaf`).
*   **Fail-fast on unusable table schemas (`scan_table`):** Raise `ValueError` without emitting a warning when a non-empty table has zero supported columns (and zero retained partition pseudo-columns) left after skipping.
*   **Avoid speculative client-side REST validation:** Trust the structural invariants enforced by BigQuery's REST API rather than guarding against impossible payloads (such as non-dict table metadata, integer `mode`/`type` values, or deeply nested `STRUCT`s beyond BigQuery's limit of 15) and avoid closed allowlists (`timePartitioning.type`, `field.mode`) that would break scanning if BigQuery introduces a new feature or granularity. Gate `_PARTITIONDATE` on `timePartitioning.type` being `"DAY"` (or omitted/`""`/`None`), and retain `_PARTITIONTIME` for any other `timePartitioning.type` without rejecting unknown granularities. *(Note: the existing `schema.py` code carries over-defensive checks and closed allowlists for `timePartitioning.type` and `field.mode` that require cleanup to match this design.)*
*   **Strict extraction for valid queries (`scan_query`):** Keep `extract_polars_schema_from_fields` strict so query outputs are never silently dropped when running verbatim SQL, while accepting valid query results that cannot occur in stored tables—including a single unnamed `STRUCT` subfield (`""`), empty `STRUCT([])`, and case-differing subfield names (`a` vs `A`). Raise `ValueError` on exact duplicate column or subfield names (including multiple unnamed subfields `""`, which Polars `Struct` cannot represent), and `TypeError` on unsupported BigQuery types (`BIGNUMERIC`, `INTERVAL`, `RANGE`, non-microsecond `TIMESTAMP`, unknown types).
*   **Single source of truth for type mapping:** Share declarative BigQuery-to-Polars type mapping tables (`_SCALAR_TYPE_MAP`, `_EXTENSION_TYPE_MAP`, `_UNSUPPORTED_TYPE_MAP`) and one recursive field extractor across table-scan and query-scan modes so supported types and aliases cannot drift out of sync.
*   **Fail-loudly on custom extension builder hooks:** Call `extension_dtype_builder` directly in both `scan_table` and `scan_query` without catching exceptions from user-registered `pl.BaseExtension.ext_from_params(...)` hooks, so defects in third-party extension implementations surface directly instead of being masked as skipped columns. *(Note: the existing `schema.py` code contains `_build_extension_dtype_checked` catching five exception types and requires a follow-up update to remove that wrapper.)*
*   **Preserve column parameters and accurate partition metadata:** Preserve extra REST keys across `REPEATED` recursion (`{**field, "mode": "NULLABLE"}`), guard `timestampPrecision`, avoid overwriting existing columns when appending pseudo-columns, and expose `_PARTITIONDATE` only on `DAY` ingestion-time partitioned tables (`_PARTITIONTIME` on all ingestion-time partitioned tables).
*   **Empty-schema short-circuit:** Support `0`-column tables and explicit `schema={}` in `scan_table` by returning a 0-row, 0-column `pl.DataFrame` without invoking `CreateReadSession`, while still enforcing `Client.close()` checks at collection time.
*   **Immutable `BQType` propagation on `TableScan` and narrowest-column fallback:** Store `bq_types` inside a read-only `MappingProxyType` wrapper on `TableScan` so `ReadApiExecutor`'s zero-column fallback ranks candidates using authoritative `BQType` sizes and unwraps `pl.BaseExtension`. On rank ties, pick the first column in schema order. When only ingestion-time pseudo-columns remain, apply the same size-then-order rule so `_PARTITIONDATE` (`DATE`, 4 bytes) beats `_PARTITIONTIME` (`TIMESTAMP`, 8 bytes) on daily tables. *(Note: the existing `read_api.py` code still hardcodes `_PARTITIONTIME` preference and requires a follow-up update to match this design.)*

## Non-Goals

*   **Schema-aware predicate pushdown (`core/compiler/`):** Wiring `source.schema` and `source.bq_types` into `compiler.compile_predicate(...)` is deferred to a separate PR.
*   **`QueryScan.schema` and `QueryScan.bq_types` plan fields:** Unconsumed metadata fields on `QueryScan` are in the process of being removed and are out of scope for this design.
*   **Zero-column batch height workaround in Polars 2.0–2.1:** Avoiding `lazyframe.drop(pl.all()).collect_batches()` row-height collapse in `core/executors/base.py` and capping `max_stream_count=1` on small `n_rows` limits belong to a separate PR.
*   **Full support for deferred BigQuery types:** Adding Arrow/Polars decoding for `BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, or parameterized `NUMERIC(P, S)` precision preservation is out of scope.
*   **Partial `STRUCT` pruning:** Extracting only the supported subfields of a `STRUCT` that contains an unsupported nested field is deferred; the entire top-level column is omitted.
*   **End-user documentation pages (`docs/source/api/data-types.md`):** Adding the MkDocs data-types reference page and registering extension docstrings in `docs/source/api/index.md` are deferred to a separate docs PR.
*   **Mode B (`load_as_storage`) with explicit `schema=`:** When callers supply explicit `schema=`, `tables.get` is bypassed and `pl.String` is indistinguishable from BigQuery `STRING`. Callers providing explicit `schema=` for `JSON`/`GEOGRAPHY` columns must use `polars_bigquery.dtypes.{JSON,GEOGRAPHY}` or a registered `BaseExtension` subclass.
*   **Backward compatibility for prior wrong mappings:** Retaining lowercase `"utc"`, plain `pl.String` for unregistered `JSON`/`GEOGRAPHY`, raising `TypeError` for one unsupported column in a mixed table, or raising `ValueError` for explicit `schema={}` is explicitly not preserved.

## Open Questions

None. All initial questions have been resolved by the owner:

*   Strict query extraction retains support for unnamed `STRUCT` subfields (single `""`), empty `STRUCT([])`, and case-differing subfield names (`a` vs `A`).
*   `QueryScan.schema` and `QueryScan.bq_types` are moved out of scope (`Non-Goals`).
*   Custom `BaseExtension.ext_from_params` exceptions propagate directly in both `scan_table` and `scan_query` (no `_build_extension_dtype_checked` catch boundary).
*   Zero-column pseudo-column fallback uses the same size-then-order rule as physical columns, selecting `_PARTITIONDATE` (`DATE`, 4 bytes) over `_PARTITIONTIME` (`TIMESTAMP`, 8 bytes) on `DAY`-partitioned tables.

## Critical User Journeys (CUJs)

### CUJ 1: Scanning a table with `TIMESTAMP`, `JSON`, `GEOGRAPHY`, and array variants

*   **Given:** A BigQuery table containing `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY<JSON>`, and `ARRAY<GEOGRAPHY>` columns and default Polars 2.0+ extension settings (`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` unset).
*   **When:** A user runs `lf = client.scan_table("project.dataset.tbl")` and collects with `lf.collect()`.
*   **Then:** `lf.collect_schema()` reports `pl.Datetime("us", "UTC")` for `TIMESTAMP`, `polars_bigquery.dtypes.JSON` / `GEOGRAPHY` for scalar extensions, and outer-wrapped `pl.Extension(..., pl.List(pl.String()))` for `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>`. Collection over an Arrow C Stream succeeds without `SchemaError`, matching `read_table().schema`.

### CUJ 2: Scanning a mixed table containing an unsupported column

*   **Given:** A BigQuery table with supported column `id INT64`, top-level `dur INTERVAL`, and nested `meta STRUCT<ok INT64, bad BIGNUMERIC>`.
*   **When:** A user runs `client.scan_table("project.dataset.mixed")`.
*   **Then:** The client omits `dur` and `meta` from the returned `LazyFrame` schema (retaining `id`), and emits one `UnsupportedTypeWarning` naming up to five skipped columns with nested paths (`dur (INTERVAL)`, `meta (BIGNUMERIC at meta.bad)`), directing the user to `client.read_query()` and formatting warnings at the caller's stack level.

### CUJ 3: Scanning a table with zero supported physical columns

*   **Given:** A non-empty BigQuery table whose physical columns are all unsupported (e.g., `INTERVAL` and `BIGNUMERIC`).
*   **When:** A user runs `client.scan_table("project.dataset.all_bad")`.
*   **Then:**
    *   If the table is **not** ingestion-time partitioned, `scan_table` raises `ValueError` listing the skipped columns and does **not** emit `UnsupportedTypeWarning`.
    *   If the table **is** ingestion-time partitioned, `scan_table` emits `UnsupportedTypeWarning` and returns a `LazyFrame` exposing only the valid pseudo-columns (`_PARTITIONDATE` + `_PARTITIONTIME` for `DAY`, or `_PARTITIONTIME` only for `HOUR`/`MONTH`/`YEAR`).

### CUJ 4: Scanning an empty (0-column) table or passing explicit `schema={}`

*   **Given:** A BigQuery table whose REST metadata defines no columns (`schema` absent/empty/`None`) or a caller passing `schema={}` to `scan_table`.
*   **When:** A user builds `lf = client.scan_table(...)` and calls `lf.collect()` (before or after `client.close()`).
*   **Then:** Before `client.close()`, `lf.collect()` returns a `(0, 0)` `pl.DataFrame` without issuing a `CreateReadSession` call (and with `schema={}`, without issuing a `tables.get` REST call). After `client.close()`, `lf.collect()` raises `RuntimeError`.

### CUJ 5: Providing an explicit schema for `JSON` / `GEOGRAPHY` / `TIMESTAMP`

*   **Given:** A user wants to skip the `tables.get` REST call and supplies `schema={"ts": pl.Datetime("us", "UTC"), "j": dtypes.JSON, "g": dtypes.GEOGRAPHY}` to `scan_table`.
*   **When:** The user collects `lf.collect()`.
*   **Then:** Matching extension schemas succeed through Arrow C Stream import without calling `tables.get`. Supplying plain `pl.String()` for an unregistered `JSON` column fails with Polars' `SchemaError` naming `google:sqlType:json`.

### CUJ 6: Running `scan_query()` against a query selecting an unsupported type

*   **Given:** A SQL query whose dry-run schema contains `BIGNUMERIC`, `INTERVAL`, `RANGE`, or picosecond `TIMESTAMP` (including inside `ARRAY` or `STRUCT`).
*   **When:** A user runs `client.scan_query(sql)`.
*   **Then:** `scan_query` fails immediately with `TypeError` referencing the top-level column name, without executing the query or silently omitting columns.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### High-Level Data Flow

```text
Client.scan_table(table, schema=None)
  |
  +--> BigQuery REST tables.get (JSON Table metadata)
  |      |
  |      v
  |    extract_schema_metadata(table_metadata)     [pure, no warnings]
  |      |-- for each field: _extract_field_dtype(strict=False)
  |      |-- appends _PARTITIONDATE (DAY/default only) & _PARTITIONTIME
  |      v
  |    ExtractedSchema(polars_schema, bq_types, skipped_columns)
  |      |-- if skipped_columns and len(polars_schema) == 0 -> raise ValueError
  |      |-- if skipped_columns and len(polars_schema) > 0  -> warn UnsupportedTypeWarning
  |      |
  +--> if len(polars_schema) == 0 (or explicit schema={}):
  |      register_io_source -> check Client.closed -> yield 0x0 DataFrame
  |      (no CreateReadSession RPC)
  |
  +--> if len(polars_schema) > 0:
         TableScan(table_ref, schema=polars_schema, bq_types=MappingProxyType)
           -> ReadApiExecutor
                -> zero-col fallback: _select_zero_column_fallback(schema, bq_types)
                -> arrow_client.read_table(selected_fields=...)
                -> Polars Arrow C Stream import

Client.scan_query(query, schema=None)
  |
  +--> jobs.query(dryRun=True) -> JSON bigquerySchema.fields
         |
         v
       extract_polars_schema_from_fields(fields)   [strict, raises on bad type/dup]
         |-- for each field: _extract_field_dtype(strict=True)
         v
       pl.Schema -> register_io_source -> QueryExecutor
```

### Component Design

#### 1. Shared Type Mapping & Dispatcher (`polars_bigquery/core/schema.py`)

Schema extraction is a pure transformation with no warning or logging side effects. To stop table-scan and query-scan paths from drifting out of sync, all type-name-to-Polars logic lives in module-level declarative tables consumed by one recursive function, `_extract_field_dtype`.

**Public/internal dataclasses:**

*   `SkippedColumn(name: str, reason: str, path: str)` (frozen) — records an omitted top-level column from `tables.get`, the human-readable reason (e.g., `"BIGNUMERIC"`, `"INTERVAL"`, `"TIMESTAMP(precision=12)"`), and the dotted path to the failing element (`"dur"` for top-level, `"meta.inner.bad"` for nested `STRUCT` fields).
*   `ExtractedSchema(polars_schema: pl.Schema, bq_types: dict[str, BQType], skipped_columns: tuple[SkippedColumn, ...])` (frozen) — output of `extract_schema_metadata()`.
*   `_UnsupportedBigQueryType(reason, path)` — internal exception raised inside skip mode (`strict=False`) and caught at the top-level field loop in `extract_schema_metadata()` so nested `STRUCT`/`ARRAY` failures unwind directly to omit the enclosing top-level column.
*   `_ExtensionTypeSpec(extension_name: str, metadata: str | None, bq_type: BQType)` (frozen, slotted) — pairs Arrow extension parameters with the canonical `BQType`.

**Declarative mapping tables:**

| BigQuery REST `type` (casefolded) | `mode` | Polars `DataType` | Canonical `BQType` | Source table |
| --- | --- | --- | --- | --- |
| `bool`, `boolean` | scalar | `Boolean()` | `BOOL` | `_SCALAR_TYPE_MAP` |
| `bytes` | scalar | `Binary()` | `BYTES` | `_SCALAR_TYPE_MAP` |
| `date` | scalar | `Date()` | `DATE` | `_SCALAR_TYPE_MAP` |
| `datetime` | scalar | `Datetime(time_unit="us")` | `DATETIME` | `_SCALAR_TYPE_MAP` |
| `float`, `float64` | scalar | `Float64()` | `FLOAT64` | `_SCALAR_TYPE_MAP` |
| `integer`, `int64` | scalar | `Int64()` | `INT64` | `_SCALAR_TYPE_MAP` |
| `numeric`, `decimal` | scalar | `Decimal(precision=38, scale=9)` | `NUMERIC` | `_SCALAR_TYPE_MAP` |
| `string` | scalar | `String()` | `STRING` | `_SCALAR_TYPE_MAP` |
| `time` | scalar | `Time()` | `TIME` | `_SCALAR_TYPE_MAP` |
| `json` | scalar | `builder("google:sqlType:json", String(), None)` | `JSON` | `_EXTENSION_TYPE_MAP` |
| `json` | `REPEATED` | `builder("google:sqlType:json", List(String()), None)` | `ARRAY` | `_EXTENSION_TYPE_MAP` |
| `geography` | scalar | `builder("google:sqlType:geography", String(), '{"encoding": "WKT"}')` | `GEOGRAPHY` | `_EXTENSION_TYPE_MAP` |
| `geography` | `REPEATED` | `builder("google:sqlType:geography", List(String()), '{"encoding": "WKT"}')` | `ARRAY` | `_EXTENSION_TYPE_MAP` |
| `timestamp` (`precision` in `None, 6, "6"`) | scalar | `Datetime(time_unit="us", time_zone="UTC")` | `TIMESTAMP` | timestamp branch |
| `record`, `struct` | scalar | `Struct([...])` | `STRUCT` | struct helper |
| Any other supported type | `REPEATED` | `List(inner_dtype)` | `ARRAY` | recursive branch |
| `bignumeric`, `bigdecimal` | any | Unsupported (`BIGNUMERIC`) | — | `_UNSUPPORTED_TYPE_MAP` |
| `interval` | any | Unsupported (`INTERVAL`) | — | `_UNSUPPORTED_TYPE_MAP` |
| `range` | any | Unsupported (`RANGE`) | — | `_UNSUPPORTED_TYPE_MAP` |
| `timestamp` (`precision` not in `None, 6, "6"`) | any | Unsupported (`TIMESTAMP(precision=...)`) | — | timestamp branch |
| Any unrecognized type | any | Unsupported (`unsupported type ...`) | — | fallback |

Notes on mapping choices:

*   `_SCALAR_TYPE_MAP` stores callables (`Boolean`, `lambda: Decimal(precision=38, scale=9)`) rather than singleton instances so every column receives a fresh `DataType` object.
*   **Outer-wrapped `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>`:** In Arrow IPC streams from the BigQuery Storage Read API, `ARROW:extension:name` is attached to the outer Arrow `List` field rather than the inner `String` child. Checking `_EXTENSION_TYPE_MAP` inside the `mode == "repeated"` branch before recursing produces `Extension(name, List(String()), metadata)`, matching the stream Polars imports. By contrast, `STRUCT<j JSON>` and `ARRAY<STRUCT<j JSON>>` keep the extension wrapper on the inner `Struct` field.
*   **Parameter-preserving `REPEATED` recursion:** When `mode == "repeated"` and `type_` is not in `_EXTENSION_TYPE_MAP`, recursion passes `{**field, "mode": "NULLABLE"}` so field attributes such as `timestampPrecision` and `fields` stay intact (for example, `ARRAY<TIMESTAMP>` with `timestampPrecision: "12"` is caught rather than misclassified as microsecond `TIMESTAMP`).
*   **`timestampPrecision` check:** Accepts `None` (absent / default microsecond), integer `6` (excluding `bool`, since `isinstance(True, int)`), or string `"6"` (proto-JSON encoding). Rejects `12`, `"12"`, `"9"`, `True`, floats, and other precisions.
*   **`extension_dtype_builder` injection and error propagation:** Both `extract_schema_metadata()` and `extract_polars_schema_from_fields()` default `extension_dtype_builder` to `core.types.build_extension_dtype`. No `try/except` catches errors from `extension_dtype_builder`; if a user-registered `pl.BaseExtension.ext_from_params(...)` fails or returns an invalid object, the exception propagates directly in both `scan_table` and `scan_query`.

**Mode resolution across Polars extension configurations:**

| Extension setting (`build_extension_dtype`) | Scalar `JSON` in `polars_schema` | `ARRAY<JSON>` in `polars_schema` | `bq_types` entry |
| --- | --- | --- | --- |
| Default (`load_as_extension`) | `Extension("google:sqlType:json", String())` | `Extension("google:sqlType:json", List(String()))` | `JSON` / `ARRAY` |
| Registered `BaseExtension` subclass | `MyExt("google:sqlType:json", String(), None)` | `MyExt("google:sqlType:json", List(String()), None)` | `JSON` / `ARRAY` |
| `as_storage=True` or `load_as_storage` | `String()` | `List(String())` | `JSON` / `ARRAY` |

**Why `scan_query` is strict (`strict=True`) while `scan_table` skips (`strict=False`):**

1.  **Server projection limitation on queries.** `scan_table` passes an explicit `selected_fields=[...]` allowlist to BigQuery Storage Read API, so omitted columns are never encoded or streamed over the wire. By contrast, `scan_query` executes the user's SQL verbatim by default (`maintain_order=True`) and also falls back to verbatim SQL on oversized pipe rewrites (`maintain_order=False`). If `scan_query` silently dropped an unsupported column like `INTERVAL` or `BIGNUMERIC` from its registered `pl.Schema`, BigQuery would still execute the query and return that column inside the Arrow IPC stream, failing immediately at `.collect()` time inside `arrow_bigquery`'s `UnsupportedTypeError` guard (or with a Polars `SchemaError` for an unexpected extra column).
2.  **Caller control over column list.** With `scan_table`, a user often scans a shared table whose DDL they do not own and wants the supported columns. With `scan_query`, the user authored the `SELECT` list directly and can drop or cast columns in SQL (`SELECT * EXCEPT (col)` or `CAST(col AS STRING)`).

**Table (`scan_table`) vs. query (`scan_query`) `STRUCT` divergence:**

The only structural rule that diverges between `strict=False` (`scan_table`) and `strict=True` (`scan_query`) is `STRUCT` subfield handling:

*   **`scan_table` (skip mode):** Mirrors BigQuery stored-table constraints. A field or nested subfield whose type is unsupported (e.g., `BIGNUMERIC`, `INTERVAL`, `RANGE`, picosecond `TIMESTAMP`, or unrecognized future type) raises `_UnsupportedBigQueryType(reason, path)` to skip the entire enclosing top-level column.
*   **`scan_query` (strict mode):** Cannot drop selected output columns from verbatim SQL. Accepts valid GoogleSQL query results that cannot occur in stored tables: empty `STRUCT([])`, a single unnamed subfield `""` (e.g., `SELECT STRUCT(1)`), and case-differing subfield names (`a` vs `A`). Raises `ValueError` only when exact duplicate names collide (such as `SELECT STRUCT(1, 2)`, which produces two `""` subfields that Polars `pl.Struct` rejects, hinting users to add `AS` aliases). Raises `TypeError` for unsupported types naming the top-level column.

**Ingestion-time partitioning pseudo-columns (`extract_schema_metadata`):**

```text
if table_metadata.get("timePartitioning") is not None:
    part_field = time_partitioning.get("field")
    if not part_field:   # None or "" => ingestion-time partitioned
        raw_type = time_partitioning.get("type")
        is_daily = not raw_type or raw_type.casefold() == "day"
        if is_daily and "_partitiondate" not in seen_names:
            add _PARTITIONDATE : Date()                   (BQType.DATE)
        if "_partitiontime" not in seen_names:
            add _PARTITIONTIME : Datetime("us", "UTC")    (BQType.TIMESTAMP)
```

No closed allowlist rejects unknown `timePartitioning.type` values. If BigQuery adds a new granularity (for example, `MINUTE` or `WEEK`), `_PARTITIONDATE` is omitted and `_PARTITIONTIME` is retained without breaking the scan.

Similarly, column `mode` checks only `(field.get("mode") or "").casefold() == "repeated"` rather than rejecting unknown modes through a closed allowlist, and impossible BigQuery REST payload checks (such as non-dict `table_metadata`, integer `mode`/`type`, duplicate physical table columns, or excessive nesting depth past BigQuery's hard limit of 15) are omitted.

**`has_bignumeric_dtype(dtype)` guard:**

Iteratively traverses `pl.Decimal`, `pl.List`, `pl.Array`, `pl.Struct`, and optional `pl.Map` (`getattr(pl, "Map", None)`). Checks `current.precision is not None and current.precision > 38` before comparing so unparameterized `Decimal` values (`precision=None`) do not raise `TypeError` during explicit-schema validation in `_register_io_source_safe`.

#### 2. Public Warning Class (`polars_bigquery/exceptions.py`)

Defines `UnsupportedTypeWarning(UserWarning)` exported in `__all__` alongside `BigQueryError` and `UnsupportedTypeError`. Callers can ignore, promote to an error (`warnings.filterwarnings("error", category=UnsupportedTypeWarning)`), or inspect these warnings with the standard Python `warnings` module.

#### 3. Logical Plan Propagation (`polars_bigquery/core/plan.py`)

`TableScan` gains one optional field:

*   `bq_types: Mapping[str, BQType] | None = None`

In `TableScan.__post_init__`, if `bq_types` is supplied, it is defensively copied into a `types.MappingProxyType(dict(self.bq_types))` so external caller mutations cannot modify an immutable frozen plan node. (`QueryScan` remains untouched in this design; `QueryScan.schema` and `QueryScan.bq_types` are out of scope.)

#### 4. Zero-Column Fallback Ranking (`polars_bigquery/core/executors/read_api.py`)

When Polars pushes down a zero-column projection (no columns selected and no columns referenced by a filter predicate), BigQuery Storage Read API still requires at least one entry in `selected_fields` (sending `selected_fields=[]` instructs the API to read *every* column in the table, pulling skipped unsupported columns and inflating scan cost).

`_BQ_TYPE_FALLBACK_RANK` maps all 14 `BQType` members to relative wire-size ranks:

*   `0`: `BOOL` (1 bit / 1 byte)
*   `3`: `DATE` (4 bytes)
*   `4`: `INT64`, `FLOAT64`, `TIME`, `DATETIME`, `TIMESTAMP` (8 bytes)
*   `5`: `NUMERIC` (16 bytes)
*   `6`: `STRING`, `BYTES`, `JSON`, `GEOGRAPHY` (variable-length bytes/text)
*   `7`: `ARRAY`, `STRUCT` (nested)

`_dtype_fallback_rank(dtype, bq_type=None)` resolves rank in this order:

1.  If authoritative `bq_type` is present, return `_BQ_TYPE_FALLBACK_RANK[bq_type]`. (This keeps a storage-loaded `JSON` / `ARRAY<JSON>` column ranked accurately as 6 / 7 even if Polars sees plain `String` / `List(String)`.)
2.  If `dtype` is a `pl.BaseExtension`, unwrap `dtype.ext_storage()` and recurse.
3.  Otherwise fall back to Polars `DataType` inspection (with `pl.Decimal` guarded by `dtype.precision is not None and dtype.precision <= 38`, else rank `8`).

`_select_zero_column_fallback(schema, bq_types=None)` splits candidates into physical columns vs. ingestion-time pseudo-columns (`_partitiondate`, `_partitiontime` matched via `.casefold()`):

1.  If any physical column exists, select `min(physical_columns, key=(rank, schema_index))` so the smallest column wins and rank ties pick the first column in schema order.
2.  If *only* ingestion-time pseudo-columns exist (all physical columns were omitted as unsupported), apply the **same** `min(..., key=(rank, schema_index))` selector over the pseudo-columns. On `DAY`-partitioned tables, `_PARTITIONDATE` (`DATE`, rank `3`, index `0`) wins over `_PARTITIONTIME` (`TIMESTAMP`, rank `4`, index `1`). On non-daily tables, only `_PARTITIONTIME` exists and is selected.

#### 5. Client Integration & Empty-Schema Short-Circuit (`polars_bigquery/_read_bigquery.py`)

*   **Warning formatting:** `_format_skipped_columns` formats up to `_MAX_SKIPPED_COLUMNS_DISPLAY = 5` entries (`"col (reason)"` when `path == name`, `"col (reason at path)"` for nested fields), appending `"... and N more"` when truncated.
*   **`Client.scan_table` warning/error contract (`schema=None`):**
    *   Calls `extract_schema_metadata(table_metadata)` directly (no broad `except (TypeError, ValueError)` wrapper masking user `BaseExtension` bugs as `BigQueryError`).
    *   If `skipped_columns` is non-empty **and** `len(polars_schema) == 0`: raises `ValueError` listing the skipped summary with a `client.read_query()` hint. Does **not** emit `UnsupportedTypeWarning`.
    *   If `skipped_columns` is non-empty **and** `len(polars_schema) > 0`: logs a `DEBUG` entry and emits one `UnsupportedTypeWarning` with `stacklevel=2` (pointing directly to the user's `client.scan_table(...)` line).
*   **Empty-schema short-circuit (`schema={}` or 0-column table):**
    *   `_normalize_schema(schema, require_non_empty=False)` allows empty `schema={}` / `pl.Schema()`.
    *   When `len(polars_schema) == 0`, skip instantiating `TableScan` and `ReadApiExecutor`. Inside `source_generator`, check `closed_event.is_set()` **first** (raising `RuntimeError` if the client was closed), then log at `DEBUG` and yield a single `pl.DataFrame(schema=polars_schema)` (`(0, 0)`) without calling `arrow_client.read_table(...)`.
*   **Docstring cross-links:** Inline code names (`UnsupportedTypeWarning`, `UnsupportedTypeError`, `polars_bigquery.dtypes.JSON`) avoid broken MkDocs cross-references until the dedicated API documentation page PR lands.

### Backward-Incompatible Behavioral Changes

| Case | Previous behavior | New behavior |
| --- | --- | --- |
| `TIMESTAMP` / `_PARTITIONTIME` timezone | `Datetime("us", "utc")` | `Datetime("us", "UTC")` |
| `JSON` / `GEOGRAPHY` default dtype | `pl.String()` | `pl.Extension("google:sqlType:json" \| "google:sqlType:geography", pl.String(), ...)` |
| `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>` default dtype | `pl.List(pl.String())` | `pl.Extension(..., pl.List(pl.String()), ...)` |
| Mixed table with an unsupported column | Fails entire `scan_table` (`TypeError` or Polars `InvalidOperationError`) | Skips top-level column with `UnsupportedTypeWarning` |
| Non-partitioned table with *only* unsupported columns | `TypeError` / `InvalidOperationError` | `ValueError` (no warning emitted) |
| Non-`DAY` (`HOUR`/`MONTH`/`YEAR`) ingestion-time table | Included `_PARTITIONDATE` and `_PARTITIONTIME` | Includes `_PARTITIONTIME` only |
| 0-column table or explicit `schema={}` | `BigQueryError` / `ValueError` | Short-circuits to `(0, 0)` `DataFrame` without `CreateReadSession` |
| `scan_query` with `INTERVAL` | Mapped to `Duration("us")` | Raises `TypeError` |

### Deltas Between Current Branch Code and This Design

Because this document was drafted alongside an existing branch implementation, four concrete code areas currently diverge from the design above and are tracked in the Detailed Implementation Plan:

1.  **`core/schema.py` — `_build_extension_dtype_checked` wrapper:** Branch currently wraps `extension_dtype_builder` in a 5-exception `try/except` boundary (`TypeError`, `ValueError`, `RuntimeError`, `AttributeError`, `LookupError`). Target design calls `extension_dtype_builder` directly so errors propagate.
2.  **`core/schema.py` — speculative client-side REST validation & closed allowlists:** Branch currently rejects unknown `timePartitioning.type` values (`_VALID_TIME_PARTITIONING_TYPES`), unknown `field.mode` values (`_VALID_FIELD_MODES`), `MAX_SCHEMA_NESTING_DEPTH` > 64, and impossible JSON shapes (`table_metadata` not a mapping, non-string `mode`/`type`, non-list `fields`, etc.). Target design removes closed allowlists and guards for impossible BigQuery payloads.
3.  **`_read_bigquery.py` — `except (TypeError, ValueError)` wrapper & `QueryScan` wiring:** Branch currently catches `(TypeError, ValueError)` around `extract_schema_metadata()` (which would mislabel propagated `BaseExtension` errors as `BigQueryError`), and constructs `QueryScan(..., schema=..., bq_types=...)`. Target design removes both.
4.  **`core/plan.py` & `core/executors/read_api.py`:** `QueryScan.schema` and `QueryScan.bq_types` are scheduled for removal from `plan.py`. In `read_api.py`, `_select_zero_column_fallback` currently prefers `_PARTITIONTIME` over `_PARTITIONDATE` in the pseudo-column loop instead of ranking by size (`_PARTITIONDATE` first on `DAY` tables).

--------------------------------------------------------------------------------

## Alternatives Considered

### Alternative A: Support Both Polars 1.x and 2.x Dynamically for `JSON` and `GEOGRAPHY`

*   **Pros:** Keeps existing Polars `1.x` installations working without requiring `polars >= 2.0.0`.
*   **Cons:** Polars `1.x` defaults to loading unknown Arrow extensions as storage (`pl.String`), while Polars `2.0+` defaults to `pl.Extension(...)`. Supporting both requires version-gated schema logic, separate documentation, and a split CI matrix.
*   **Reason for Rejection:** Polars `2.x` is the target runtime (`polars >= 2.0.0, < 3.0.0` in `pyproject.toml`), allowing a single default `pl.Extension` path matching `read_table()` and `read_query()`.

### Alternative B: Unwrap `JSON` and `GEOGRAPHY` to Plain `pl.String` Inside `scan_table()`

*   **Pros:** Lets callers invoke `.str.*` directly without calling `.ext.storage()`, and avoids exposing Polars' `@unstable` `pl.Extension` dtype by default.
*   **Cons:** Makes `scan_table().collect_schema()` disagree with `read_table().schema` and `read_query().schema`, erases BigQuery type identity on explicit `schema=` inputs, and risks semantic confusion because BigQuery rejects `CAST(JSON AS STRING)` / `CAST(GEOGRAPHY AS STRING)` inside Storage Read API filters.
*   **Reason for Rejection:** Keeping Polars 2.0+ native Arrow extension dtypes preserves parity across `read_table()`, `read_query()`, `scan_table()`, and `scan_query()`.

### Alternative C: Fail Fast on Any Unsupported Column in `scan_table()`, or Strip Nested `STRUCT` Subfields

*   **Pros:** Failing fast keeps `lf.collect_schema()` strictly identical to the BigQuery table DDL. Stripping only unsupported *nested* subfields would keep sibling subfields inside a `STRUCT` readable.
*   **Cons:** Failing fast blocks an entire mixed table because of a single `INTERVAL`, `BIGNUMERIC`, `RANGE`, or picosecond `TIMESTAMP` column even when the caller never selects it. Dropping a nested subfield from `polars_schema` does not work without extra machinery because Storage Read API `selected_fields` projects by top-level name and still returns the unmapped subfield in the Arrow stream; partial `STRUCT` reading requires nested field projection and client-side `STRUCT` reconstruction.
*   **Reason for Rejection:** Omitting the enclosing top-level column with `UnsupportedTypeWarning` unblocks mixed tables with bounded complexity; partial `STRUCT` subfield exclusion is left as a future improvement.

### Alternative D: Silently Skip Unsupported Columns in `scan_query()` Like `scan_table()`

*   **Pros:** Uniform warning behavior across both `scan_table()` and `scan_query()`.
*   **Cons:** Under default `maintain_order=True` (and under oversized pipe fallback on `maintain_order=False`), `scan_query()` executes verbatim SQL without appending `|> SELECT`. BigQuery still encodes and streams the unsupported column over Arrow IPC, causing collection to fail anyway with `UnsupportedTypeError` or Polars `SchemaError`. Users also write the query's `SELECT` list explicitly and expect errors rather than missing output columns.
*   **Reason for Rejection:** Fail-fast `strict=True` validation in `scan_query()` stops doomed queries before billing/execution and avoids surprising users with dropped `SELECT` expressions.

### Alternative E: Separate `_extract_data_type` and `_extract_strict_data_type` Functions with `if` Chains

*   **Pros:** Slightly shallower call depth inside each function with zero `strict: bool` parameter threading.
*   **Cons:** Duplicates the full 14-type mapping, `REPEATED` recursion, `timestampPrecision` check, and Arrow extension wrapping across two ~170-line functions. Adding a new BigQuery type or alias requires editing two parallel `if` ladders that can easily fall out of sync.
*   **Reason for Rejection:** Explicit declarative tables (`_SCALAR_TYPE_MAP`, `_EXTENSION_TYPE_MAP`, `_UNSUPPORTED_TYPE_MAP`) plus a single `_extract_field_dtype(..., strict=...)` dispatcher leave a single source of truth while isolating only the divergent `STRUCT` policy into small helpers.

### Alternative F: Catch Custom `BaseExtension.ext_from_params` Errors and Convert Them into `SkippedColumn`s

*   **Pros:** If a caller registers a scalar-only third-party `pl.BaseExtension` subclass for `google:sqlType:json` or `google:sqlType:geography` whose `ext_from_params` rejects `pl.List(pl.String())`, a mixed table containing an `ARRAY<JSON>` column could still skip just that array column during `scan_table()`.
*   **Cons:** Catching `(TypeError, ValueError, RuntimeError, AttributeError, LookupError)` cannot distinguish intentional `List(String())` rejection from bugs in a user's extension code (typos, missing dict keys, wrong constructor signature), silently downgrading a broken hook into a dropped column warning. It also behaves inconsistently with `read_table()` and `scan_query()`, which raise immediately.
*   **Reason for Rejection:** Propagating exceptions directly keeps failures loud and consistent across `read_table()`, `scan_table()`, and `scan_query()`. *(The current branch's `_build_extension_dtype_checked` wrapper is scheduled for removal.)*

### Alternative G: Closed Client-Side Allowlists and Exhaustive REST Metadata Type Validation

*   **Pros:** Guards unit tests or corrupted proxy responses against unexpected types (`mode: 42`, `table_metadata: "bad"`, `timePartitioning.type: "WEEK"`).
*   **Cons:** Adds client-side checks for conditions real BigQuery REST endpoints never produce, and closed allowlists (`_VALID_TIME_PARTITIONING_TYPES`, `_VALID_FIELD_MODES`) actively break valid user tables if BigQuery introduces a new partitioning granularity (e.g., `WEEK` or `MINUTE`) or a new field mode.
*   **Reason for Rejection:** Only validate structural facts that real BigQuery can emit and that actually affect Polars Arrow mapping (unknown `field.type`, `timestampPrecision` values other than microsecond, duplicate query columns/subfields). Check `timePartitioning.type` strictly for daily (`"DAY"` / default) instead of rejecting unrecognized granularities.

### Alternative H: Prefer `_PARTITIONTIME` Over `_PARTITIONDATE` in Zero-Column Pseudo-Column Fallback

*   **Pros:** Uniformly selects `_PARTITIONTIME` regardless of whether an ingestion-time partitioned table is `DAY`, `HOUR`, `MONTH`, or `YEAR`.
*   **Cons:** On `DAY`-partitioned tables whose physical columns were all skipped, `_PARTITIONDATE` is 4 bytes (`DATE`) while `_PARTITIONTIME` is 8 bytes (`TIMESTAMP`), and `_PARTITIONDATE` appears first in the schema (`["_PARTITIONDATE", "_PARTITIONTIME"]`). Hardcoding `_PARTITIONTIME` scans twice the width per row and breaks the project's general rule of picking the narrowest column (tie-broken to first in schema order).
*   **Reason for Rejection:** Using the same `(rank, schema_index)` rule for both physical and pseudo-columns selects `_PARTITIONDATE` on daily tables and `_PARTITIONTIME` on non-daily tables (matching Polars' own column selection for `lf.select(pl.len())`).

### Alternative I: Pre-Flight Schema Interception on Explicit `schema=` Mismatches

*   **Pros:** Could reword Polars' mismatch error when a caller passes `pl.String` for a `JSON` column to recommend `polars_bigquery.dtypes.JSON`.
*   **Cons:** Requires adding a new `__arrow_c_schema__` PyO3 export to `arrow_bigquery`'s `ArrowStreamExporter` (coupling this Python change to a Rust wheel release and version pin bump in `py-arrow-bigquery`), plus maintaining a Python-side schema comparator that risks drifting from Polars' Rust Arrow importer rules (`time_zone`, `BaseExtension`, `load_as_storage`). Without `__arrow_c_schema__`, calling the single-use `__arrow_c_stream__` ahead of `pl.scan_arrow_c_stream()` would consume the stream.
*   **Reason for Rejection:** Polars' native `SchemaError` already identifies both expected and actual dtypes, including the Arrow extension name (`google:sqlType:json` / `google:sqlType:geography`). Adding `__arrow_c_schema__` on `ArrowStreamExporter` independently for general Arrow PyCapsule ecosystem interop is tracked under [Future Improvements](#future-improvements).

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Privacy & logging:** `UnsupportedTypeWarning` messages and `DEBUG` logs record table IDs, column names, dotted schema paths, and BigQuery type names only. No row data or predicate literal values are logged. Warning summaries cap output at `_MAX_SKIPPED_COLUMNS_DISPLAY = 5` entries (`"... and N more"`) so wide tables with many unsupported columns do not flood logs.
*   **Network & cost impact:**
    *   Short-circuiting empty `0`-column tables and explicit `schema={}` avoids an unserviceable `CreateReadSession` RPC (and skips `tables.get` for explicit `schema={}`).
    *   Zero-column fallback requests a single narrowest safe column (`BOOL` before `DATE` before `INT64`, and `_PARTITIONDATE` before `_PARTITIONTIME` when only pseudo-columns remain) rather than `selected_fields=[]`, preventing full-table scans over Storage Read API.
*   **Immutability:** `TableScan.bq_types` is wrapped in `types.MappingProxyType(dict(...))` so external mutation of caller dictionaries cannot alter a cached or normalized logical plan.
*   **Rollout & backward compatibility:** Callers upgrading should inspect the behavioral changes table above (`"utc"` -> `"UTC"`, `JSON`/`GEOGRAPHY` extension types, `UnsupportedTypeWarning` on mixed tables, `ValueError` when no supported columns remain, `schema={}` short-circuit). Callers wanting the prior fail-fast behavior on mixed tables can run `warnings.filterwarnings("error", category=UnsupportedTypeWarning)`.

--------------------------------------------------------------------------------

### Future Improvements

*   **Partial `STRUCT` subfield exclusion:** Support omitting only unsupported nested subfields of a `STRUCT` and reconstructing the remaining subfields client-side, once Storage Read API nested field projection support is wired through `arrow-bigquery`.
*   **Schema-aware predicate pushdown:** Consume `TableScan.bq_types` (and query dry-run types) inside `core/compiler/` so SQL generation avoids invalid GoogleSQL expressions on `JSON`/`GEOGRAPHY`, integer bitwise operators, `NaN`/`Inf`, and cross-temporal comparisons.
*   **Dedicated API data-types documentation:** Publish `docs/source/api/data-types.md` and register `UnsupportedTypeWarning`, `UnsupportedTypeError`, and `polars_bigquery.dtypes` entries in `docs/source/api/index.md` so docstring links resolve under `mkdocs build --strict`.
*   **Zero-column row-height preservation (`Polars 2.0–2.1`):** Pair the narrowest-column fallback with incremental empty-`Project(())` execution in `core/executors/base.py` so direct `io_source([], ...)` yields keep exact row height (`shape == (N, 0)`) rather than dropping batches via `drop(pl.all()).collect_batches()`.
*   **Non-destructive `__arrow_c_schema__` on `arrow_bigquery.ArrowStreamExporter`:** `ArrowStreamExporter` in `py-arrow-bigquery/src/stream.rs` already retains `schema: ArrowSchemaRef` independently of the single-use `source` mutex. Implementing the Arrow PyCapsule `__arrow_c_schema__` hook (`export_field_to_c` gated by `schema_ffi_validated`) in a standalone `arrow-bigquery` change enables repeated, non-destructive schema inspection for ecosystem consumers (`pyarrow.schema(exporter)`, `arro3`, DuckDB) without consuming the stream.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Each entry below lists the complete design-state changes for the file (relative to `docs/designs/`). Where the existing branch implementation diverges from the finalized design choices in this document, the exact alignment change is labeled **[Alignment Update]**.

### Target Files to Modify

#### 1. [`py-polars-bigquery/polars_bigquery/core/schema.py`](../../py-polars-bigquery/polars_bigquery/core/schema.py) (Modify)

*   **Type / Scope:** Pure Python schema-extraction module (`polars_bigquery.core.schema`).
*   **Rationale:** Centralizes BigQuery-to-Polars schema mapping across both `scan_table` (skip/warn on unsupported types) and `scan_query` (strict extraction) so mapping tables and alias lists cannot drift out of sync, while stripping speculative REST payload checks and closed allowlists that break on new BigQuery features.
*   **Required Changes:**
    *   Define frozen dataclasses `SkippedColumn(name: str, reason: str, path: str)` and `ExtractedSchema(polars_schema: pl.Schema, bq_types: dict[str, BQType], skipped_columns: tuple[SkippedColumn, ...])`, internal sentinel `_UnsupportedBigQueryType(reason: str, path: str)`, internal `_ExtensionTypeSpec(extension_name: str, metadata: str | None, bq_type: BQType)`, and type alias `ExtensionDtypeBuilder = Callable[[str, DataType, str | None], DataType]`.
    *   Define explicit module-level lookup tables:
        *   `_SCALAR_TYPE_MAP`: casefolded aliases (`bool`, `boolean`, `bytes`, `date`, `datetime`, `float`, `float64`, `integer`, `int64`, `numeric`, `decimal`, `string`, `time`) mapping to `(dtype_factory, BQType)` callables producing fresh `DataType` instances.
        *   `_EXTENSION_TYPE_MAP`: `json` (`google:sqlType:json`, `metadata=None`, `BQType.JSON`) and `geography` (`google:sqlType:geography`, `metadata='{"encoding": "WKT"}'`, `BQType.GEOGRAPHY`).
        *   `_UNSUPPORTED_TYPE_MAP`: `bignumeric` / `bigdecimal` (`"BIGNUMERIC"`), `interval` (`"INTERVAL"`), and `range` (`"RANGE"`) mapped to `(skip_reason, strict_message_template)`.
    *   Define shared recursive `_extract_field_dtype(field, *, path, strict, top_col_name="", extension_dtype_builder=build_extension_dtype) -> tuple[DataType, BQType]`:
        *   Check `mode == "repeated"` first (`(field.get("mode") or "").casefold() == "repeated"`). For `type_` in `_EXTENSION_TYPE_MAP`, call `extension_dtype_builder(spec.extension_name, List(String()), spec.metadata)` and return `(dtype, BQType.ARRAY)`. Otherwise recurse on `{**field, "mode": "NULLABLE"}` so field parameters (including `timestampPrecision`) are preserved.
        *   Dispatch scalar types via `_SCALAR_TYPE_MAP` and `_EXTENSION_TYPE_MAP`.
        *   Map `timestamp` to `Datetime(time_unit="us", time_zone="UTC")`, validating `timestampPrecision` via `_is_supported_timestamp_precision(precision)` (`None`, integer `6` excluding `bool`, or string `"6"`). Non-microsecond precision raises `_UnsupportedBigQueryType` (`strict=False`) or `TypeError` (`strict=True`).
        *   Dispatch `record`/`struct` to small mode-specific helpers:
            *   Table-scan (`strict=False`): walk subfields recursively, tracking dotted paths (`f"{path}.{sub_name}"`) so nested unsupported types unwind and skip the top-level column.
            *   Query-scan (`strict=True`): keep empty `STRUCT([])`, permit a single unnamed subfield (`""`), allow case-differing subfield names (`a` vs `A`), collect exact duplicate names, and raise `ValueError` with the `AS` alias hint when duplicates collide.
        *   Dispatch known-unsupported aliases via `_UNSUPPORTED_TYPE_MAP`; treat unknown `type` values as skipped (`strict=False`) or `TypeError` (`strict=True`).
    *   Expose `extract_schema_metadata(table_metadata, *, extension_dtype_builder=build_extension_dtype) -> ExtractedSchema` (pure; no warning side effects):
        *   Read `(table_metadata.get("schema") or {}).get("fields") or []`.
        *   Track casefolded `seen_names` before checking each field's type so skipped columns do not collide with appended pseudo-columns.
        *   Handle `timePartitioning` when `field` is absent (`None` or `""`): append `_PARTITIONDATE` (`Date()`, `BQType.DATE`) only when `type` is `None`, `""`, or `"day"` (case-insensitive); always append `_PARTITIONTIME` (`Datetime("us", "UTC")`, `BQType.TIMESTAMP`) if not already in `seen_names`.
    *   Update `extract_polars_schema(table_metadata, *, extension_dtype_builder=build_extension_dtype, require_non_empty=False) -> pl.Schema` to delegate to `extract_schema_metadata()`.
    *   Update `has_bignumeric_dtype(dtype)` to guard `current.precision is not None and current.precision > 38`.
    *   **[Alignment Update]** Remove `_build_extension_dtype_checked` and `_build_extension_for_spec` exception catching so `extension_dtype_builder(...)` exceptions propagate directly in both `scan_table` and `scan_query`.
    *   **[Alignment Update]** Remove closed allowlists `_VALID_TIME_PARTITIONING_TYPES` and `_VALID_FIELD_MODES` so future BigQuery partitioning granularities (`WEEK`, `MINUTE`, etc.) retain `_PARTITIONTIME` without raising and future field modes do not break parsing.
    *   **[Alignment Update]** Remove speculative client-side REST checks for payloads real BigQuery cannot produce (`table_metadata` non-mapping guard, non-mapping `schema`/`timePartitioning`, non-list `fields`, non-string `name`/`mode`/`type`, duplicate/unnamed table columns, `MAX_SCHEMA_NESTING_DEPTH` > 64 guard). Keep genuine null-safety (`schema: None`, `fields: None`, `mode: None`, `type: None`, `field: ""`).
    *   **[Alignment Update]** Fold `extract_query_schema_from_fields` directly into `extract_polars_schema_from_fields(fields, *, extension_dtype_builder=build_extension_dtype) -> pl.Schema` (removing the unconsumed tuple return API until schema-aware query pushdown requires `bq_types`).

#### 2. [`py-polars-bigquery/polars_bigquery/exceptions.py`](../../py-polars-bigquery/polars_bigquery/exceptions.py) (Modify)

*   **Type / Scope:** Public exceptions and warnings module.
*   **Rationale:** Exposes a dedicated warning subclass so callers can programmatically ignore, capture, or promote skipped-column warnings to errors with Python's `warnings` module.
*   **Required Changes:**
    *   Define `UnsupportedTypeWarning(UserWarning)` with docstring detailing the omitted column categories and a `warnings.filterwarnings("error", category=UnsupportedTypeWarning)` example.
    *   Export `"UnsupportedTypeWarning"` in `__all__` alongside `"BigQueryError"` and `"UnsupportedTypeError"`.

#### 3. [`py-polars-bigquery/polars_bigquery/core/plan.py`](../../py-polars-bigquery/polars_bigquery/core/plan.py) (Modify)

*   **Type / Scope:** Immutable logical scan plan IR.
*   **Rationale:** Preserves authoritative BigQuery column types on `TableScan` even when Polars loads extension columns as plain storage (`load_as_storage` / `as_storage=True`), while keeping plan nodes protected against caller dict mutation.
*   **Required Changes:**
    *   Import `types`, `Mapping`, and `BQType`.
    *   Add optional `bq_types: Mapping[str, BQType] | None = None` to `TableScan`; in `TableScan.__post_init__`, copy any non-`None` mapping into `types.MappingProxyType(dict(self.bq_types))`.
    *   **[Alignment Update]** Remove the unconsumed `schema: pl.Schema | None = None` and `bq_types: Mapping[str, BQType] | None = None` fields (and associated `__post_init__` checks) from `QueryScan`, leaving `QueryScan` unchanged.

#### 4. [`py-polars-bigquery/polars_bigquery/core/executors/read_api.py`](../../py-polars-bigquery/polars_bigquery/core/executors/read_api.py) (Modify)

*   **Type / Scope:** Storage Read API plan compiler and executor.
*   **Rationale:** Avoids passing `selected_fields=[]` (which causes BigQuery to stream every column, including skipped unsupported ones) by choosing the single narrowest column using authoritative `BQType` widths and unwrapping `pl.BaseExtension`.
*   **Required Changes:**
    *   Add `_BQ_TYPE_FALLBACK_RANK: dict[BQType, int]` covering all 14 `BQType` enum members (`BOOL`: 0, `DATE`: 3, `INT64`/`FLOAT64`/`TIME`/`DATETIME`/`TIMESTAMP`: 4, `NUMERIC`: 5, `STRING`/`BYTES`/`JSON`/`GEOGRAPHY`: 6, `ARRAY`/`STRUCT`: 7).
    *   Update `_dtype_fallback_rank(dtype: pl.DataType, bq_type: BQType | None = None) -> int`: prefer `_BQ_TYPE_FALLBACK_RANK[bq_type]` when `bq_type` is supplied; unwrap `pl.BaseExtension` via `dtype.ext_storage()`; guard `pl.Decimal` precision with `dtype.precision is not None and dtype.precision <= 38`.
    *   Update `_select_zero_column_fallback(schema: pl.Schema, bq_types: Mapping[str, BQType] | None = None) -> str` and wire `source.bq_types` inside `compile_read_api_plan()`. Do **not** alter `compiler.compile_predicate(op.predicate)` arguments here.
    *   **[Alignment Update]** Replace the hardcoded `_partitiontime` loop in `_select_zero_column_fallback` with the same `min(..., key=(rank, schema_index))` comparator used for physical columns, so `_PARTITIONDATE` (`DATE`, 4 bytes, rank `3`) wins over `_PARTITIONTIME` (`TIMESTAMP`, 8 bytes, rank `4`) on daily tables and ties break by first column in schema order.

#### 5. [`py-polars-bigquery/polars_bigquery/_read_bigquery.py`](../../py-polars-bigquery/polars_bigquery/_read_bigquery.py) (Modify)

*   **Type / Scope:** Public `Client` implementation and docstrings.
*   **Rationale:** Coordinates `extract_schema_metadata()` warning emission, all-unsupported `ValueError` exits, empty-schema short-circuiting without `CreateReadSession`, `bq_types` handoff to `TableScan`, and updated API docstrings.
*   **Required Changes:**
    *   Add `_MAX_SKIPPED_COLUMNS_DISPLAY = 5` and `_format_skipped_columns(skipped_columns)` formatting `"col (reason)"`, `"col (reason at path)"`, and `"... and N more"` truncation.
    *   In `Client.scan_table`:
        *   Call `extract_schema_metadata(table_metadata)` directly when `schema is None`.
        *   If `extracted.skipped_columns` is non-empty and `len(polars_schema) == 0`, raise `ValueError` with the skipped summary and a `client.read_query()` hint (do **not** emit a warning).
        *   If `extracted.skipped_columns` is non-empty and `len(polars_schema) > 0`, log `DEBUG` and emit `warnings.warn(..., UnsupportedTypeWarning, stacklevel=2)`.
        *   For explicit `schema=`, call `_normalize_schema(schema, require_non_empty=False)` and leave `bq_types = None`.
        *   When `len(polars_schema) == 0` (empty table or `schema={}`), leave `source = None` and `executor = None`; inside `source_generator`, check `closed_event.is_set()` first, then short-circuit to `iter([pl.DataFrame(schema=polars_schema)])` without calling `read_table()`.
        *   When `len(polars_schema) > 0`, pass `bq_types=bq_types` into `TableScan(...)`.
    *   Update `read_table()` and `scan_table()` docstrings (UTC casing, `JSON`/`GEOGRAPHY` extension types, `UnsupportedTypeWarning`, `DAY`-gated `_PARTITIONDATE`, `schema={}` short-circuit) using inline code spans rather than unresolved cross-reference links (`data-types.md` does not exist yet).
    *   **[Alignment Update]** Remove the `except (TypeError, ValueError)` wrapper around `extract_schema_metadata(table_metadata)` inside `scan_table` so errors from custom `BaseExtension.ext_from_params(...)` hooks propagate directly with their original type and traceback instead of being wrapped as `BigQueryError("Invalid BigQuery table metadata...")`.
    *   **[Alignment Update]** Inside `Client.scan_query`, keep calling `extract_polars_schema_from_fields(fields)` and drop the `schema=polars_schema, bq_types=bq_types` keyword arguments to `QueryScan(...)`.

#### 6. [`py-polars-bigquery/tests/unit/core/test_schema.py`](../../py-polars-bigquery/tests/unit/core/test_schema.py) (Modify)

*   **Type / Scope:** Unit test suite for `core/schema.py`.
*   **Rationale:** Validates all 14 supported `BQType` mappings, outer-wrapped extension arrays, nesting depth, `timestampPrecision`, partitioning granularities, skipped column paths, and strict query extraction.
*   **Required Changes:**
    *   Cover primitive types, `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>`, deeply nested `STRUCT`s with `DATETIME` / `JSON` / `GEOGRAPHY` (15 levels), Mode B `load_as_storage` and `register_extension_type(..., as_storage=True)` preserving `bq_types`, `timestampPrecision` (`6`, `"6"`, `12`, `"12"`, `"9"`, `True`, inside `REPEATED`), ingestion-time partitioning granularity (`DAY`, omitted, `None`, `HOUR`, `MONTH`, `YEAR`, plus future unknown granularity such as `WEEK` retaining `_PARTITIONTIME` without `_PARTITIONDATE`), explicit `field=""`, non-overwriting pseudo-column checks, strict `extract_polars_schema_from_fields` type/duplicate errors (including single unnamed `STRUCT` subfield accepted, duplicate unnamed subfields rejected), and `has_bignumeric_dtype` (including `precision=None`).
    *   **[Alignment Update]** Replace `test_extract_schema_custom_extension_rejection_becomes_skipped_column` and `test_extract_schema_extension_builder_non_datatype_and_lookup_error` assertions to verify direct exception propagation instead of `SkippedColumn` omission.
    *   **[Alignment Update]** Remove test assertions for impossible BigQuery REST shapes and closed `timePartitioning.type`/`field.mode` allowlist rejections; verify unrecognized future `timePartitioning.type` values omit `_PARTITIONDATE` and retain `_PARTITIONTIME`.
    *   **[Alignment Update]** Update the strict-extraction test to invoke `extract_polars_schema_from_fields` directly if `extract_query_schema_from_fields` is folded in.

#### 7. [`py-polars-bigquery/tests/unit/core/test_types.py`](../../py-polars-bigquery/tests/unit/core/test_types.py) (Modify)

*   **Type / Scope:** Unit test suite for `core/types.py` and public exports.
*   **Rationale:** Guards the public export contract for `UnsupportedTypeWarning` in `polars_bigquery.exceptions`.
*   **Required Changes:**
    *   Assert `"UnsupportedTypeWarning" in polars_bigquery.exceptions.__all__` and `issubclass(UnsupportedTypeWarning, UserWarning)` inside `test_public_dtypes_and_warning_exports()`.

#### 8. [`py-polars-bigquery/tests/unit/core/test_plan.py`](../../py-polars-bigquery/tests/unit/core/test_plan.py) (Modify)

*   **Type / Scope:** Unit test suite for `core/plan.py`.
*   **Rationale:** Covers immutability of the new `TableScan.bq_types` field.
*   **Required Changes:**
    *   **[Alignment Update]** Add assertions inside `test_table_scan_validation_and_mutation_isolation` verifying `isinstance(scan.bq_types, types.MappingProxyType)`, rejecting item assignment (`TypeError`), and confirming caller dict mutation after construction does not modify `scan.bq_types`.

#### 9. [`py-polars-bigquery/tests/unit/core/test_read_api_executor.py`](../../py-polars-bigquery/tests/unit/core/test_read_api_executor.py) (Modify)

*   **Type / Scope:** Unit test suite for `core/executors/read_api.py`.
*   **Rationale:** Validates zero-column fallback ranking using `bq_types`, `pl.BaseExtension` unwrapping, `Decimal(precision=None)` guard, and the `_PARTITIONDATE` size-first tie-breaking rule.
*   **Required Changes:**
    *   **[Alignment Update]** Update `test_compile_read_api_plan_zero_column_fallback_priority_tiers` and `test_compile_read_api_plan_case_insensitive_pseudo_column_fallback` to use uppercase `"UTC"` and assert `("_PARTITIONDATE",)` / `("_partitiondate",)` win over `_PARTITIONTIME` when both pseudo-columns are present.
    *   **[Alignment Update]** Add focused assertions checking `_BQ_TYPE_FALLBACK_RANK` covers all `BQType` values, `bq_types` overrides `pl.String()` rank for `ARRAY`/`BOOL`, `pl.BaseExtension` unwraps `ext_storage()`, and `pl.Decimal` with `precision=None` ranks safely at `8`.

#### 10. [`py-polars-bigquery/tests/unit/test_read_bigquery.py`](../../py-polars-bigquery/tests/unit/test_read_bigquery.py) (Modify)

*   **Type / Scope:** Unit test suite for `polars_bigquery.Client`.
*   **Rationale:** Exercises end-to-end `Client.scan_table()` schema extraction, real non-mocked Arrow C Stream import via `_FakeArrowStreamExporter`, warning stack levels and formatting, empty-schema short-circuits, and lifecycle guards.
*   **Required Changes:**
    *   Add `_FakeArrowStreamExporter` and unit tests covering: warning emission with dotted paths and `stacklevel=2`, 5-entry truncation cap, all-unsupported non-partitioned `ValueError` without warning, all-unsupported ingestion-time partitioned tables (`DAY` vs `HOUR`), empty schema (`{}` / 0-column table) short-circuiting without `read_table`, zero-column narrowest-column fallback selection, non-mocked Arrow C Stream roundtrip for `"UTC"` `TIMESTAMP` and `JSON`/`GEOGRAPHY`/`ARRAY<JSON>`/`ARRAY<GEOGRAPHY>`, and explicit `schema=` with `polars_bigquery.dtypes` (including `SchemaError` mismatch on plain `pl.String`).
    *   Update `test_client_close_invalidates_pending_lazy_scan` to assert pending empty-schema scans (`schema={}`) also raise `RuntimeError` after `client.close()`.
    *   Update `test_client_scan_table_eager_validation` (`schema={}` short-circuit, unknown type `ValueError`).
    *   Update `test_client_scan_bignumeric_raises_clear_type_error` (`scan_table` metadata raises `ValueError` for all-`BIGNUMERIC`; explicit `schema={"bn": pl.Decimal(76, 38)}` raises `TypeError`).
    *   **[Alignment Update]** Remove `test_scan_table_wraps_malformed_metadata_as_bigquery_error` and the `bad_meta` loop in `test_client_input_type_and_wire_payload_validation` matching removal of impossible-payload wrappers; add a test verifying a custom `BaseExtension.ext_from_params` `TypeError` propagates directly out of `client.scan_table(...)` without being wrapped as `BigQueryError`.

### Verification Plan

*   **Rust / Build:** No Rust (`crates/arrow-bigquery`, `py-arrow-bigquery`) changes are required by this design.
*   **Python Unit Tests:**
    *   `uv run pytest py-polars-bigquery/tests/unit/core/test_types.py`
    *   `uv run pytest py-polars-bigquery/tests/unit/core/test_schema.py`
    *   `uv run pytest py-polars-bigquery/tests/unit/core/test_plan.py`
    *   `uv run pytest py-polars-bigquery/tests/unit/core/test_read_api_executor.py`
    *   `uv run pytest py-polars-bigquery/tests/unit/test_read_bigquery.py`
    *   Full suite: `uv run pytest py-polars-bigquery/tests/unit` and `uv run pytest py-arrow-bigquery/tests/unit`
*   **Static Analysis & Formatting:**
    *   `uv run --project py-polars-bigquery --group dev ruff check --no-fix py-polars-bigquery/`
    *   `uv run --project py-polars-bigquery --group dev ruff format --diff py-polars-bigquery/`
*   **Documentation Build (`strict: true`):**
    *   `uv run --with-requirements docs/source/requirements.txt mkdocs build --strict` (verifies updated `Client.read_table` / `Client.scan_table` docstrings contain zero unresolved autoref or relative links before `data-types.md` lands).
*   **Notes on Polars CI Matrix:** Verify both `--resolution lowest` (`polars 2.0.0`, Python 3.11) and latest `polars 2.x` (`--resolution highest --upgrade`) match `test-py-polars-unit.yml`.
*   **Integration / Live Verification:** Defer new live Storage Read API differential tests (`tests/system/`) to the paired integration test PR; verify manually when `GOOGLE_CLOUD_PROJECT` is configured with `uv run --project py-polars-bigquery --group test pytest py-polars-bigquery/tests/system`.

--------------------------------------------------------------------------------

*Drafted with EGM*
