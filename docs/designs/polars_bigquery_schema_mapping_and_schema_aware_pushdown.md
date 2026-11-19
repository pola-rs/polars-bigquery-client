# Schema Mapping Fixes and Schema-Aware Predicate Pushdown for `polars-bigquery`

*   **Author:** @tswast
*   **Status:** Proposed
*   **Date:** 2026-11-19

--------------------------------------------------------------------------------

## Overview

`polars_bigquery.Client.scan_table` lazily scans BigQuery tables through the BigQuery Storage Read API with column projection and row predicate pushdown. Today, two coupled defects prevent `scan_table` from working across BigQuery's type system on Polars 2.0+: schema extraction disagrees with BigQuery's Arrow output and fails on tables containing unsupported columns, and predicate pushdown emits GoogleSQL without knowing column types.

This design upgrades `polars-bigquery` to Polars 2.0+, aligns schema mapping and projection safety across all 14 supported BigQuery types, guards BigQuery `INTERVAL` (`MonthDayNano`) decoding in Rust (`crates/arrow-bigquery`), adds a public `polars_bigquery.dtypes` module and `UnsupportedTypeWarning`, and adds a schema-aware typing and lowering pass to `core/compiler/` so pushed predicates are valid GoogleSQL and preserve Polars semantics.

## The Problem

This repository is organized into three layers: `crates/arrow-bigquery` (Rust BigQuery Storage Read API and Arrow IPC decoder), `py-arrow-bigquery` (PyO3 Arrow C Stream bindings exposed as `self._arrow_client`), and `py-polars-bigquery` (Polars wrapper exposing eager `Client.read_table()` / `Client.read_query()` and lazy `Client.scan_table()`). Eager `read_table()` derives its Polars schema directly from BigQuery's Arrow IPC stream, whereas lazy `scan_table()` constructs a Polars schema up front from BigQuery's REST `tables.get` metadata (unless the caller supplies an explicit `schema=`).

Users adopting `Client.scan_table` hit four classes of failures:

*   **Broken scans on tables with `JSON`, `GEOGRAPHY`, or `TIMESTAMP` columns**: BigQuery's Arrow IPC stream encodes `TIMESTAMP` with uppercase `"UTC"`, `JSON` with `ARROW:extension:name = "google:sqlType:json"`, and `GEOGRAPHY` with `ARROW:extension:name = "google:sqlType:geography"` and `ARROW:extension:metadata = '{"encoding": "WKT"}'`. For `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>`, BigQuery attaches the extension metadata to the outer `List` field rather than the inner `String` element. Polars 2.0+ loads unknown Arrow extension types as `pl.Extension(...)` by default. Because `core/schema.py` maps `JSON` and `GEOGRAPHY` to plain `pl.String` and `TIMESTAMP` to lowercase `time_zone="utc"`, `scan_table().collect()` fails Polars schema validation on `JSON`/`GEOGRAPHY` tables and disagrees with `read_table().schema` on `TIMESTAMP` tables. In addition, `core/schema.py` currently reconstructs only `{name, type, fields, mode}` during `REPEATED` recursion, dropping parameters such as `timestampPrecision`.
*   **Unscannable mixed tables and process aborts**: BigQuery tables can include types deferred or unsupported by this library—`BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, or unnamed/duplicate `STRUCT` subfields—alongside supported scalar columns. Today, `core/schema.py` maps `BIGNUMERIC` to `Decimal(76, 38)` (exceeding Polars' 38-digit maximum precision and failing `register_io_source`), maps `INTERVAL` to `Duration(time_unit="us")`, raises `TypeError` on `RANGE<T>`, and ignores `timestampPrecision` and unnamed/duplicate `STRUCT` subfields. Meanwhile, BigQuery transmits `INTERVAL` over Arrow IPC as `Interval(MonthDayNano)`, which `polars 2.1.0` panics on during schema import (`PanicException`) and `polars-arrow 0.53.0` aborts with a non-unwinding `SIGABRT` during Arrow C Stream FFI export. Even when a caller requests only row count (`lf.select(pl.len())`), passing `selected_fields=[]` instructs Storage Read API to stream every column in the table—billing a full-table scan and pulling omitted `INTERVAL` columns.
*   **BigQuery SQL errors on valid Polars filters**: Because `core/compiler/` compiles Polars filter expressions without knowing column data types, valid Polars filters crash `CreateReadSession` with `400 INVALID_ARGUMENT` errors—including `IS_NAN`/`IS_INF` on `INT64`, `fill_null(strategy="zero"|"one")` eagerly emitting `IFNULL(col, 0)` on `BOOL`/`STRING`/`BYTES`/temporal columns, integer bitwise `&`/`|`/`~` emitted as logical `AND`/`OR`/`NOT`, and comparisons or string functions pushed onto `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`, or mismatched temporal operands (`DATE`/`DATETIME` vs. `TIMESTAMP`).
*   **Silent data loss on `FLOAT64` `NaN` and `NUMERIC` vs. `FLOAT64` comparisons**: Polars orders `NaN` greater than all other floats (`NaN == NaN`, `NaN > 5.0`, `NaN >= 5.0`, and `1.0 < NaN` all evaluate to `True`), whereas BigQuery evaluates IEEE-754 `NaN` comparisons as `FALSE`. Similarly, Polars casts `Decimal(38, 9)` to lossy `Float64` when comparing against `Float64` (`Decimal('9007199254740993.0') == 9007199254740992.0` is `True`), whereas BigQuery compares `NUMERIC` against `FLOAT64` exactly (`FALSE`). Both violate the predicate pushdown superset invariant and drop matching rows before Polars can re-filter them.

In this design, deferred BigQuery types are treated identically to unsupported types at runtime (referred to below simply as **unsupported**).

## Goals

*   **Polars 2.0+ alignment**: Raise the dependency pin in `py-polars-bigquery/pyproject.toml` to `polars >= 2.0.0, < 3.0.0` and expose Arrow-matching Polars 2.0+ schemas for all 14 supported BigQuery types (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`) plus ingestion-time pseudo-columns (`_PARTITIONDATE` for daily partitions; `_PARTITIONTIME` for all ingestion-time partitions).
*   **Public extension dtypes and warning class**: Publish `polars_bigquery.dtypes` exporting scalar `JSON` and `GEOGRAPHY` so callers avoid hand-writing Arrow extension metadata strings for scalar `JSON` and `GEOGRAPHY` columns, and export `UnsupportedTypeWarning` in `polars_bigquery.exceptions` so callers can filter or promote skip warnings programmatically.
*   **Safe handling of unsupported columns**:
    *   Omit top-level columns whose type (or any nested descendant inside `STRUCT`/`ARRAY`) is unsupported (`BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, unnamed/duplicate/empty `STRUCT`, or unrecognized types) with a concise `UnsupportedTypeWarning` and a `client.read_query()` workaround.
    *   Retain valid ingestion-time pseudo-columns (`_PARTITIONTIME`, plus `_PARTITIONDATE` on daily partitions) when physical columns are omitted; fail with `ValueError` at `scan_table()` when a non-empty table has zero supported columns remaining after appending pseudo-columns; short-circuit empty-schema (0-column) tables without calling `CreateReadSession`.
*   **Rust-layer `MonthDayNano` (`INTERVAL`) abort protection**: Reject `ArrowDataType::Interval(IntervalUnit::MonthDayNano)` in `crates/arrow-bigquery` (`parse_arrow_schema`) with a `BigQueryError` linking to the upstream Polars issues before FFI export so `read_table()`, `read_query()`, and explicit `scan_table(schema=...)` never abort Python with `SIGABRT`.
*   **Minimal-width zero-column projections**: Ensure Storage Read API never receives `selected_fields=[]` on a non-empty schema by selecting the narrowest supported column when Polars requests zero columns, and preserve batch row height without collapsing `lf.select(pl.len())` to zero rows.
*   **Eliminate compiler-induced GoogleSQL type errors and silent row loss under accurate schemas**:
    *   Make `predicate_to_row_restriction` schema-aware across all 14 supported BigQuery types while honoring authoritative BigQuery types on `scan_table(schema=None)` even when extensions are loaded as storage.
    *   Support typed `fill_null(strategy="zero"|"one")` rewrites, null-safe 3-valued `INT64` `is_nan`/`is_inf` rewrites, Polars-aligned `FLOAT64` `NaN` comparisons (with AST-duplication and 1 MB SQL byte caps), integer bitwise rejection, `OPAQUE` null-check-only handling (`JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`), and strict cross-type rules (`INT64` vs. `FLOAT64` and `INT64` vs. `NUMERIC` allowed; `NUMERIC` vs. `FLOAT64` and all cross-temporal mixes skipped) while preserving partial pushdown and valid `drop_nulls` sibling hoisting.
*   **Observability**: Emit `DEBUG` logs for `selected_fields`, compiled `row_restriction`, omitted columns, and type-driven pushdown fallbacks.
*   **End-user data type and pushdown documentation**:
    *   Publish `docs/source/api/data-types.md` (wired into `mkdocs.yml`) covering the 14-type mapping, `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` outer wrapping, `TIMESTAMP` `"UTC"`, ingestion-time pseudo-columns, `.ext.storage()`, `polars_bigquery.dtypes` explicit `schema=` examples, extension registration behavior, unsupported types with `read_query()` workarounds, and `read_table()` vs. `scan_table()` differences.
    *   Update `docs/source/api/predicate-pushdown.md` with the per-type operator pushdown matrix and common pushdown pitfalls.
*   **End-to-end verification**: Validate schema mapping, Rust `MonthDayNano` rejection, and pushdown parity via unit tests (including non-mocked Arrow C stream tests) and live BigQuery Storage Read API differential tests.

## Non-Goals

*   **`scan_query` integration**: Query-backed lazy scanning and standard SQL `WHERE` pushdown are tracked separately.
*   **Full support for deferred BigQuery types**: Adding Arrow/Polars representations for `BIGNUMERIC`, preserving narrow `(P, S)` Arrow metadata on parameterized `NUMERIC(P, S)`, picosecond `TIMESTAMP`, `INTERVAL`, `RANGE<T>`, or unnamed/duplicate `STRUCT` subfields is out of scope.
*   **Partial `STRUCT` subfield exclusion**: Projecting only supported subfields out of a `STRUCT` containing an unsupported descendant (requiring client-side `STRUCT` reconstruction for Storage Read API compatibility) is deferred.
*   **User-facing `BYTES`, `TIME`, and `Decimal` literal parsing**: Parsing user literals or `is_in` collections of `pl.Binary`, `pl.Time`, or `pl.Decimal` is deferred; internal literal IR nodes are added only for `fill_null(strategy="zero"|"one")` lowering.
*   **Third-party extension packages**: Integrating GeoPolars or custom JSON/WKT namespaces is out of scope beyond respecting extensions already registered via `pl.register_extension_type(...)` on `scan_table(schema=None)`.
*   **Mode B (`as_storage=True` / `load_as_storage`) with explicit `schema=`**: When a caller passes an explicit `schema=` (skipping `tables.get`), plain storage dtypes (`pl.String`, `pl.List(pl.String)`) cannot be distinguished from BigQuery `STRING` / `ARRAY<STRING>`. Mode B is explicitly unsupported with explicit `schema=` for ambiguous storage types; callers supplying `schema=` for `JSON`/`GEOGRAPHY` columns must use `pl.Extension` (`polars_bigquery.dtypes`) or a registered `BaseExtension` subclass.
*   **Array extension helper constants**: `polars_bigquery.dtypes` exports scalar `JSON` and `GEOGRAPHY` only (no array helper constants).
*   **Backward compatibility for previous mappings**: Retaining lowercase `"utc"`, plain `pl.String` for unregistered `JSON`/`GEOGRAPHY`, or `TypeError` on a single unsupported column is not required.

## Open Questions

None. Key technical behaviors were verified live against the BigQuery Storage Read API and locally against Polars 2.x (with one remaining live Storage Read API check for `sql/temporal.py`'s exact `IFNULL` output folded into Phase 5 integration tests):

*   **Resolved — `ARRAY`/`STRUCT` `IS [NOT] NULL` in `row_restriction`**: Live-tested on `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`. Storage Read API accepts `IS NULL` and `IS NOT NULL` on both `STRUCT` (`totals`) and `ARRAY<STRUCT>` (`hits`), rejects `hits = hits` (`Equality is not defined`), and evaluates `totals = totals` to `0` rows when subfields contain `NULL`. Confirms the null-check-only allowlist for `ARRAY` and `STRUCT`.
*   **Resolved — Pseudo-column-only `selected_fields` and ingestion-time granularity**: Live-tested on a daily ingestion-time partitioned table whose only physical column was `INTERVAL`. Both `selected_fields=["_PARTITIONDATE"]` and `selected_fields=["_PARTITIONTIME"]` succeeded without transmitting `INTERVAL`, returning `pl.Date` / `pl.Datetime("us", "UTC")` and the exact row count (`height=2`). Because BigQuery exposes `_PARTITIONDATE` on daily partitions while hourly ingestion-time partitions expose only `_PARTITIONTIME`, `_PARTITIONDATE` is gated on `DAY` granularity while `_PARTITIONTIME` is always retained.
*   **Resolved — Nested `ARRAY<JSON>` and `STRUCT<g GEOGRAPHY>` in Polars 2.x**: Live-tested via `pl.DataFrame` and `pl.scan_arrow_c_stream`. BigQuery attaches `ARROW:extension:name` to the outer `List` field for `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>` (top-level and inside `STRUCT`s), so Polars 2.x produces `pl.Extension("google:sqlType:json", pl.List(pl.String))` and `pl.Extension("google:sqlType:geography", pl.List(pl.String), '{"encoding": "WKT"}')` rather than `pl.List(pl.Extension(...))`, while `STRUCT<j JSON, g GEOGRAPHY>` and `ARRAY<STRUCT<j JSON, g GEOGRAPHY>>` keep `pl.Extension(..., pl.String)` on the struct fields.
*   **Resolved — GoogleSQL `NaN`, `INT64`/`NUMERIC` vs. `FLOAT64`, temporal cross-types, and `fill_null(strategy=...)`**:
    *   `CAST('nan' AS FLOAT64) != 5.0` evaluates to `TRUE` (matching Polars); `NaN != NaN` evaluates to `TRUE` in BigQuery vs. `FALSE` in Polars (confirming `((a != b) AND NOT (IS_NAN(a) AND IS_NAN(b)))` for 3-valued exactness).
    *   `INT64(2^53+1) = FLOAT64(2^53)` is `TRUE` and `>` is `FALSE` in both BigQuery and Polars 2.x.
    *   `NUMERIC(2^53+1) = FLOAT64(2^53)` is `FALSE` (`>` is `TRUE`) in BigQuery, whereas Polars evaluates `Decimal == Float64` to `TRUE` (`>` to `FALSE`)—confirming `NUMERIC` vs. `FLOAT64` pushdown must be skipped to prevent silent row loss. `NUMERIC` vs. `INT64` is exact in both.
    *   `DATE = TIMESTAMP` and `DATETIME = TIMESTAMP` fail in BigQuery with `INVALID_ARGUMENT`; strict same-type matching is enforced across all temporal operators.
    *   Verified locally on Polars `2.0.0-rc.1`: `Decimal(38, 9).fill_null(strategy="one")` produces `Decimal('1.000000000')` (matching `NUMERIC '1'`), and `Datetime("ms"|"us"|"ns").fill_null(strategy="one")` fills physical integer `1` (Unix epoch + 1 unit).
    *   Live Storage Read API tests verified `IFNULL` with `TIME '00:00:00'`, `b''`, `NUMERIC '0'`, `DATE '1970-01-01'`, `DATETIME '1970-01-01 00:00:00.000001'`, and `TIMESTAMP_MICROS(1)`. Phase 5 integration tests will additionally verify `sql/temporal.py`'s exact emitted forms (`DATE(TIMESTAMP_SECONDS(0 * 86400))`, `DATE(TIMESTAMP_SECONDS(1 * 86400))`, `DATETIME(TIMESTAMP_MICROS(0))`, `DATETIME(TIMESTAMP_MICROS(1))`, `TIMESTAMP_MICROS(0)`, `TIMESTAMP_MICROS(1)`) inside `IFNULL(...)`.
*   **Resolved — 0-column BigQuery table**: Live-tested via REST `tables.insert` with an empty schema. `tables.get` returns `{}` (no `schema` field), and Storage Read API `CreateReadSession` fails with `FAILED_PRECONDITION: Table or view (...) has empty schema, cannot create read session.` Short-circuiting empty schemas in `scan_table` returns a 0-row, 0-column `DataFrame` without invoking `CreateReadSession`.

## Critical User Journeys (CUJs)

### CUJ 1: Scanning a Mixed Table Containing Unsupported Columns

*   **Given**: A BigQuery table has supported columns (`id INT64`, `name STRING`) alongside unsupported columns (`amount BIGNUMERIC`, `dur INTERVAL`, `window RANGE<DATE>`, `ts_pico TIMESTAMP(12)`, and `meta STRUCT<ok INT64, bad INTERVAL>`).
*   **When**: A user calls `lf = client.scan_table(table)` and executes `lf.filter(pl.col("id") > 10).collect()` or `lf.select(pl.len()).collect()`.
*   **Then**:
    *   `scan_table()` emits one `UnsupportedTypeWarning` listing the omitted top-level columns with their reasons and nested paths (capped at 5 entries with `"... and N more"` and a `client.read_query()` workaround).
    *   `lf.collect_schema()` exposes only `id` and `name`.
    *   Storage Read API receives only supported `selected_fields` (selecting `id` rather than `[]` for `pl.len()`), preserves exact row count on `pl.len()`, and completes without `InvalidOperationError`, `TypeError`, or `MonthDayNano` crashes.

### CUJ 2: Scanning and Filtering Tables with `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, and `STRUCT`

*   **Given**: A BigQuery table contains `ts TIMESTAMP`, `payload JSON`, `geom GEOGRAPHY`, `json_list ARRAY<JSON>`, `tags ARRAY<STRING>`, and `info STRUCT<k STRING, v INT64, j JSON>` columns on Polars 2.0+.
*   **When**: The user calls `lf = client.scan_table(table)` and runs `lf.collect()`, `lf.filter(pl.col("payload").is_not_null()).collect()`, and `lf.filter(pl.col("payload").ext.storage().str.starts_with("{")).collect()`.
*   **Then**: `lf.collect_schema()` matches `client.read_table(table).schema` (`pl.Datetime("us", "UTC")`, `polars_bigquery.dtypes.JSON`, `polars_bigquery.dtypes.GEOGRAPHY`, and `pl.Extension("google:sqlType:json", pl.List(pl.String))` for `ARRAY<JSON>`), `is_not_null()` pushes down to BigQuery `row_restriction`, `.ext.storage().str.starts_with("{")` falls back to Polars without sending invalid SQL to BigQuery, and all queries return exact results.

### CUJ 3: Schema-Aware Filtering with Typed `fill_null`, Float Checks, Bitwise Ops, and `FLOAT64` `NaN`

*   **Given**: A BigQuery table contains `flag BOOL`, `count INT64`, `score FLOAT64` (including `NaN`, `±Inf`, and `NULL`), `label STRING`, `t TIME`, and `price NUMERIC`.
*   **When**: The user applies filters including `pl.col("flag").fill_null(strategy="zero")`, `pl.col("label").fill_null(strategy="zero") == ""`, `pl.col("t").fill_null(strategy="one") > ...`, `~pl.col("count").is_nan()`, `(pl.col("count") & 4) == 4`, `pl.col("price") == 1.5`, and `pl.col("score") > 5.0`.
*   **Then**: Representable expressions compile to Polars-aligned GoogleSQL (`IFNULL(flag, FALSE)`, `(IFNULL(label, '') = '')`, null-safe 3-valued `INT64` check, `((score > 5.0) OR IS_NAN(score))`), unrepresentable or divergent expressions (`TIME` `strategy="one"`, integer bitwise `&`, `NUMERIC == FLOAT64`) degrade to client-side filtering without breaking sibling `&` partial pushdown, no compiler-induced `400 INVALID_ARGUMENT` errors occur, and collected rows match eager `read_table().filter(...)`.

### CUJ 4: Scanning with an Explicit User-Supplied `schema=`

*   **Given**: A user skips the `tables.get` REST call by calling `client.scan_table(table, schema=...)` on a table with `TIMESTAMP`, `JSON`, and `GEOGRAPHY` columns.
*   **When**: The user supplies `schema={"ts": pl.Datetime("us", "UTC"), "j": polars_bigquery.dtypes.JSON, "g": polars_bigquery.dtypes.GEOGRAPHY}`—or mistakenly passes `schema={"j": pl.String}`.
*   **Then**: With `polars_bigquery.dtypes` and `"UTC"`, `scan_table` skips `tables.get`, pushes valid predicates using the supplied schema, and collects without error; with plain `pl.String` for a `JSON` column, Polars raises its native `SchemaError` showing expected `String` vs. actual `Extension("google:sqlType:json", String)` (or BigQuery raises `BigQueryError` if a string filter was pushed onto `JSON`), matching the guidance documented in `data-types.md` and `Client.scan_table`.

### CUJ 5: All-Unsupported Tables (Partitioned vs. Non-Partitioned), `INTERVAL` Reads, and 0-Column Tables

*   **Given**: Four tables: (A) a non-partitioned table with only an `INTERVAL` column, (B) a daily ingestion-time partitioned table whose only physical column is `INTERVAL`, (C) an hourly ingestion-time partitioned table whose only physical column is `INTERVAL`, and (D) a 0-column BigQuery table.
*   **When**: A user calls `client.scan_table(...)` on each table (or `client.read_table(A)`).
*   **Then**:
    *   Table (A) fails at `scan_table()` with `ValueError` listing the omitted column, reason, and `read_query()` workaround (without emitting `UnsupportedTypeWarning`); calling `client.read_table(A)` raises `BigQueryError` from `crates/arrow-bigquery` citing the upstream Polars `MonthDayNano` issues rather than aborting Python.
    *   Table (B) emits `UnsupportedTypeWarning` and retains `_PARTITIONDATE` and `_PARTITIONTIME` so partition row counts succeed via `selected_fields=["_PARTITIONDATE"]`.
    *   Table (C) emits `UnsupportedTypeWarning` and retains only `_PARTITIONTIME` so partition row counts succeed via `selected_fields=["_PARTITIONTIME"]`.
    *   Table (D) returns an empty-schema `LazyFrame` and collects a 0-row, 0-column `DataFrame` without invoking `CreateReadSession`.

## Technical Plan (Architecture)

All package paths below are rooted at the repository root (`crates/arrow-bigquery/`, `py-polars-bigquery/polars_bigquery/`, `docs/source/`). The design spans five cooperating runtime layers sharing a canonical BigQuery type model (`py-polars-bigquery/polars_bigquery/core/types.py`), plus end-user documentation updates (Section 6):

```text
                       +----------------------------+
                       |   Client.scan_table(...)   |
                       +-------------+--------------+
                                     |
                   schema is None    |    explicit schema=...
                   +-----------------+-----------------+
                   v                                   v
         +-------------------+               +-------------------+
         | BigQuery REST API |               | Normalize to      |
         |   (tables.get)    |               | pl.Schema         |
         +---------+---------+               +---------+---------+
                   |                                   |
                   v                                   |
    +--------------+----------------+                  |
    | core/schema.py                |                  |
    | extract_schema_metadata()     |<-- core/types.py |
    | - Polars 2.0+ mapping & "UTC" |   (BQType enum,  |
    | - Extension & registered ext  |    extension     |
    | - Whole-column skip + paths   |    resolution,   |
    | - Authoritative bq_types map  |    Polars->BQ    |
    | - Partition pseudo-columns    |    classifier)   |
    +--------------+----------------+                  |
                   |                                   |
      warn skipped / fail if non-empty                 |
      table has 0 supported columns                    |
                   +-----------------+-----------------+
                                     |
                                     v
                     +---------------+---------------+
                     | Polars register_io_source     |
                     +---------------+---------------+
                                     |  (on .collect() / .collect_batches())
                                     v
             +-----------------------+-----------------------+
             | Empty schema -> return empty DataFrame iter   |
             +-----------------------+-----------------------+
                                     | (non-empty)
                                     v
      +------------------------------+------------------------------+
      | core/compiler/ (4-stage iterative pipeline)                 |
      | 1. json_to_ir               [schema-blind parser]           |
      | 2. resolve_and_lower_types  [types, rewrites, NaN, bounds]  |
      | 3. rewrite_ir               [existing boolean rewriters]    |
      | 4. ir_to_sql + 1 MB guard   [mechanical SQL formatter]      |
      +------------------------------+------------------------------+
                                     |
                                     v
      +------------------------------+------------------------------+
      | Projection safeguard (narrowest column if selected_fields   |
      | is empty) -> arrow_client.read_table(...)                   |
      | -> crates/arrow-bigquery parse_arrow_schema (rejects        |
      |    Interval(MonthDayNano) before FFI export)                |
      | -> pl.scan_arrow_c_stream -> filter -> limit -> projection  |
      |    (height-preserving drop when requested columns == [])    |
      +-------------------------------------------------------------+
```

### 1. Rust-Layer `MonthDayNano` (`INTERVAL`) Guard (`crates/arrow-bigquery`)

*   Every Arrow IPC stream decoded by `crates/arrow-bigquery` (`read_table`, `read_query`, and `scan_table`) passes through `parse_arrow_schema` in `crates/arrow-bigquery/src/decode.rs`.
*   Add an `UnsupportedType(String)` variant to `BigQueryError` in `crates/arrow-bigquery/src/error.rs` (mapped to Python `BigQueryError` automatically by `py-arrow-bigquery/src/error.rs`).
*   In `parse_arrow_schema`, recursively inspect the decoded `ArrowSchema` (including nested `List`, `LargeList`, `FixedSizeList`, `Struct`, `Map`, and `Extension` types). If any field has `ArrowDataType::Interval(IntervalUnit::MonthDayNano)`, return `Err(BigQueryError::UnsupportedType(...))` naming the column path and linking to upstream Polars issues [#30920](https://github.com/pola-rs/polars/issues/30920) and [#30921](https://github.com/pola-rs/polars/issues/30921).
*   This eliminates the `SIGABRT` across `read_table()`, `read_query()`, and explicit `scan_table(schema=...)` without requiring a partial Python `pl.Duration` check.

### 2. Dependency Floor, Public `dtypes`, Warning Class, and Canonical Type Layer

*   **Polars 2.0+ dependency floor (`py-polars-bigquery/pyproject.toml`)**: Raise the pin from `polars >= 1.43.0, < 3.0.0` to `polars >= 2.0.0, < 3.0.0`. This removes Polars 1.x `load_as_storage` vs. Polars 2.0 `load_as_extension` version skew.
*   **Public `polars_bigquery.dtypes` module (`py-polars-bigquery/polars_bigquery/dtypes.py`)**: Export scalar `JSON` (`pl.Extension("google:sqlType:json", pl.String())`) and `GEOGRAPHY` (`pl.Extension("google:sqlType:geography", pl.String(), '{"encoding": "WKT"}')`), representing the default unregistered `pl.Extension` dtypes. Register `polars_bigquery.dtypes` in `docs/source/api/index.md`. Callers passing explicit `schema=` for `ARRAY<JSON>` or `ARRAY<GEOGRAPHY>` construct `pl.Extension(..., pl.List(pl.String()), ...)` directly, as documented in `data-types.md`.
*   **Public warning subclass (`py-polars-bigquery/polars_bigquery/exceptions.py`)**: Export `UnsupportedTypeWarning(UserWarning)` and document it alongside `BigQueryError` in `docs/source/api/index.md`.
*   **Canonical type model (`py-polars-bigquery/polars_bigquery/core/types.py`)**: Shared type layer used by `dtypes.py`, `core/schema.py`, `_read_bigquery.py`, and `core/compiler/`:
    *   Closed 14-member `BQType` enum covering all supported BigQuery types (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`), plus `OPAQUE_BQ_TYPES = frozenset({JSON, GEOGRAPHY, ARRAY, STRUCT})`. Unsupported or non-BigQuery dtypes map to `None`.
    *   Extension resolution (`build_extension_dtype`): respects Polars' runtime extension configuration when building schemas from REST metadata:
        *   **Mode A (registered `BaseExtension` subclass)**: instantiates the user's registered class via `ext_from_params(name, storage, metadata)` so custom extensions (e.g., GeoPolars) match Arrow decoding and preserve the extension type identity.
        *   **Mode B (`as_storage=True` or `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`)**: returns `storage` (`pl.String()` or `pl.List(pl.String())`). Supported on `scan_table(schema=None)` because `core/schema.py` passes authoritative `bq_types` from REST metadata into the compiler, keeping `JSON`/`GEOGRAPHY`/`ARRAY` opaque in SQL generation. Explicitly unsupported with explicit `schema=` for ambiguous storage types (`pl.String` / `pl.List(pl.String)` cannot be distinguished from BigQuery `STRING` / `ARRAY<STRING>` without REST metadata).
        *   **Default (`load_as_extension`)**: returns `pl.Extension(name, storage, metadata)`.
    *   Polars-to-BigQuery classifier (`polars_dtype_to_bq_type(dtype) -> BQType | None`): handles both instantiated `DataType`s and uninstantiated `DataTypeClass` objects safely (without instantiating parameterized classes). Maps `BaseExtension` by extension name (`ARRAY` if storage is `List`, else `JSON`/`GEOGRAPHY`; unknown extension names -> `None`), `Boolean` -> `BOOL`, integer types `Int8`..`Int128` and `UInt8`..`UInt64` (excluding `UInt128`) -> `INT64`, `Float32`/`Float64` (excluding `Float16`) -> `FLOAT64`, `Decimal` with explicit `precision <= 38` and `scale <= 9` -> `NUMERIC` (enabling compiler pushdown for any BigQuery-compatible decimal precision, while `Decimal(precision=None)` or larger scales map to `None`), `String` -> `STRING`, `Binary` -> `BYTES`, `Date` -> `DATE`, naive `Datetime` -> `DATETIME`, tz-aware `Datetime` -> `TIMESTAMP`, `Time` -> `TIME`, `List` -> `ARRAY`, `Struct` -> `STRUCT`, and all other Polars dtypes -> `None`.
    *   Time-unit extractor (`polars_datetime_time_unit`) and deterministic storage-width rank (`column_width_rank`) delegating to `polars_dtype_to_bq_type`: `BOOL` (1) < `DATE` (2) < `INT64`/`FLOAT64`/`TIME`/`DATETIME`/`TIMESTAMP` (3) < `NUMERIC` (4) < `STRING`/`BYTES`/`JSON`/`GEOGRAPHY` (5) < `ARRAY`/`STRUCT`/`None` (6).

### 3. Schema Extraction and Whole-Column Omission (`py-polars-bigquery/polars_bigquery/core/schema.py`)

#### BigQuery-to-Polars Schema Mapping

| BigQuery Type (`REST`) | Mode | Polars `2.0+` DataType |
| --- | --- | --- |
| `BOOL` / `BOOLEAN` | scalar | `pl.Boolean()` |
| `INT64` / `INTEGER` | scalar | `pl.Int64()` |
| `FLOAT64` / `FLOAT` | scalar | `pl.Float64()` |
| `NUMERIC` / `DECIMAL` (unparameterized or `NUMERIC(P, S)`) | scalar | `pl.Decimal(precision=38, scale=9)` |
| `STRING` | scalar | `pl.String()` |
| `BYTES` | scalar | `pl.Binary()` |
| `DATE` | scalar | `pl.Date()` |
| `DATETIME` | scalar | `pl.Datetime(time_unit="us")` |
| `TIME` | scalar | `pl.Time()` |
| `TIMESTAMP` (`timestampPrecision` in `None`, `6`, `"6"`) | scalar | `pl.Datetime(time_unit="us", time_zone="UTC")` |
| `JSON` | scalar | `build_extension_dtype("google:sqlType:json", pl.String())` |
| `JSON` | `REPEATED` | `build_extension_dtype("google:sqlType:json", pl.List(pl.String()))` |
| `GEOGRAPHY` | scalar | `build_extension_dtype("google:sqlType:geography", pl.String(), '{"encoding": "WKT"}')` |
| `GEOGRAPHY` | `REPEATED` | `build_extension_dtype("google:sqlType:geography", pl.List(pl.String()), '{"encoding": "WKT"}')` |
| `RECORD` / `STRUCT` (non-empty; all subfields named, unique, supported) | scalar | `pl.Struct([...])` |
| Any other supported type | `REPEATED` | `pl.List(...)` preserving all field parameters |
| `_PARTITIONDATE` (ingestion-time partitioned, `DAY` granularity) | pseudo | `pl.Date()` |
| `_PARTITIONTIME` (ingestion-time partitioned, any granularity) | pseudo | `pl.Datetime(time_unit="us", time_zone="UTC")` |

#### Omission, Metadata Hardening, and Partition Rules

*   **Metadata hardening**: Handle missing or JSON-`null` values for `schema`, `fields`, `mode`, `type`, and `name` safely. Preserve all REST field parameters during `REPEATED` recursion. Accept only microsecond `TIMESTAMP` precision (`None`, `6`, or `"6"`, rejecting booleans and picosecond/other precisions). Reject top-level fields with missing/empty names and reject `STRUCT` fields that are empty or contain unnamed or duplicate subfields.
*   **Whole-column omission**: Any unsupported column type (`BIGNUMERIC`/`BIGDECIMAL`, `INTERVAL`, `RANGE`, non-microsecond `TIMESTAMP`, invalid `STRUCT`, missing top-level name, or unrecognized type)—whether top-level or nested inside `STRUCT`/`ARRAY`—omits the enclosing top-level column and records a `SkippedColumn(name, reason, path)` entry.
*   **Ingestion-time pseudo-columns**: Per the [BigQuery REST Table resource](https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table), a table with `timePartitioning` present and `timePartitioning.field is None` is ingestion-time partitioned. Always append `_PARTITIONTIME`; append `_PARTITIONDATE` only when partition granularity is `DAY` (default when `type` is omitted).
*   **Structured result**: `extract_schema_metadata(table_metadata)` returns `ExtractedSchema(polars_schema, bq_types, skipped_columns)` with no warning or exception side effects. `bq_types` maps every supported column name (physical and pseudo-column) to its authoritative `BQType`. Keep `extract_polars_schema` as a wrapper returning `.polars_schema`.

### 4. Projection Safety, Height Preservation, and Warnings (`py-polars-bigquery/polars_bigquery/_read_bigquery.py`)

*   **Early error vs. warning policy (`schema is None`)**:
    *   If `extracted.skipped_columns` is non-empty and `len(extracted.polars_schema) == 0`: raise `ValueError` at `scan_table()` without emitting `UnsupportedTypeWarning`, listing up to 5 skipped columns with reasons/paths, `"... and N more"` when truncated, and a `client.read_query()` workaround.
    *   If `extracted.skipped_columns` is non-empty and `len(extracted.polars_schema) > 0` (including ingestion-time partitioned tables that retain `_PARTITIONTIME` / `_PARTITIONDATE`): emit `UnsupportedTypeWarning` with the total skipped count, up to 5 entries, `"... and N more"`, and the `client.read_query()` workaround.
*   **Explicit `schema=` contract**: Normalize via `pl.Schema(schema)` and derive compiler types via `polars_dtype_to_bq_type`. Rely on Polars' native `register_io_source` `SchemaError` and BigQuery's native `BigQueryError` (avoiding single-use stream consumption issues on `ArrowStreamExporter` and avoiding heuristic error-message rewriting).
*   **`source_generator` execution flow**: Polars invokes `source_generator(with_columns, predicate, n_rows, batch_size)` to obtain an `Iterator[pl.DataFrame]` (with `columns = with_columns if with_columns is not None else list(polars_schema.keys())` and `predicate_cols` from `predicate.meta.root_names()`). Keep `source_generator` as a regular function returning iterators across all branches (no mixed `yield`/`return`):
    *   **Empty-schema short-circuit (`len(polars_schema) == 0`)**: covers 0-column BigQuery tables and explicit `schema={}`. Return an iterator yielding a single 0-row, 0-column `DataFrame` without invoking `read_table()`.
    *   **Non-empty `selected_fields` safeguard**: combine `columns` and `predicate_cols` preserving order; if empty (e.g., `lf.select(pl.len())`), pick the single column in `polars_schema` with the lowest `column_width_rank` (tie-breaking by schema order).
    *   **Zero-width height preservation**: after `pl.scan_arrow_c_stream`, apply `filter(predicate)` and `limit(n_rows)`. When `columns == []`, do not call `lazyframe.select([])` (which collapses to 0 rows in Polars 2.x); return a generator dropping all columns from each batch via `batch.drop(batch.columns)` (verified to preserve batch height with stable Polars API). Otherwise select `columns` and return `collect_batches`.
*   **Observability**: Emit `DEBUG` logs recording table ID, `selected_fields`, compiled `row_restriction`, and skipped columns.

### 5. Schema-Aware Predicate Pushdown Compiler (`py-polars-bigquery/polars_bigquery/core/compiler/`)

#### Existing Pipeline Context and Partial Pushdown Contract

As detailed in [`polars_bigquery_scan_table_row_predicate_pushdown.md`](./polars_bigquery_scan_table_row_predicate_pushdown.md), `core/compiler/` currently runs a 3-stage schema-blind pipeline (`json_to_ir -> rewrite_ir -> ir_to_sql`). Any expression that cannot be safely translated degrades to an `Unsupported` IR node. The existing `rewrite_ir` passes rely on `Unsupported` to achieve **partial pushdown** while preserving the superset invariant:

1.  `rewrite_drop_nulls` unwraps `DropNulls(e)` (when `e` has no `Unsupported` descendant) and conjoins `IsNotNull(e)` at the root; it traverses `children()`, including `Unsupported.operands`, so a valid `DropNulls` sibling inside an unsupported expression still hoists `IsNotNull(e)`.
2.  `rewrite_boolean_when` converts boolean `Ternary(p, t, f)` (from Polars `pl.when(p).then(t).otherwise(f)`) into `And`/`Or` combinations—splitting partial-pushdown cases where `t` or `f` contains `Unsupported`—and wraps negated predicates in `COALESCE(..., FALSE)` when required by 3-valued logic.
3.  `rewrite_boolean_literals` folds constant boolean expressions.
4.  `rewrite_when` merges nested `Ternary` chains into multi-branch `When` nodes.
5.  `rewrite_de_morgan` pushes `Not` over `Or` into `And` so `emit_and_sql` in Stage 4 can drop `Unsupported` conjuncts while still pushing supported conjuncts.

#### 4-Stage Schema-Aware Pipeline

The compiler inserts an iterative typing and lowering stage between parsing and rewriting:

```text
Polars Expr JSON
   |
   v
1. json_to_ir(expr_json)                           [schema-blind parser; preserves FillNullWithStrategy]
   |
   v
2. resolve_and_lower_types(ir, schema, bq_types)   [New: iterative BFS + reverse-BFS typing & lowering]
   |
   v
3. rewrite_ir(ir)                                  [existing 5 boolean rewriters]
   |
   v
4. ir_to_sql(ir) + 1 MB UTF-8 byte guard           [mechanical SQL emitter + Bytes/Time/Decimal literals]
```

#### Public Compiler Entrypoint Contract

*   `predicate_to_row_restriction(predicate, *, schema, bq_types=None) -> str`: `schema` is a required keyword argument (`pl.Schema` or dtype mapping). When `bq_types` is provided (from `ExtractedSchema` on `scan_table(schema=None)`), it overrides `polars_dtype_to_bq_type` so Mode B `as_storage=True` / `load_as_storage` columns retain their true BigQuery type (`JSON`, `GEOGRAPHY`, `ARRAY`).
*   Enforce a final UTF-8 byte-length cap (`1_000_000` bytes) on the emitted SQL string inside `_json_expr_to_row_restriction`, falling back to `""` with a `DEBUG` log if exceeded so BigQuery Storage Read API's 1 MB `row_restriction` limit is never breached.

#### New IR Nodes and SQL Formatters

*   `FillNullWithStrategy(UnaryExpr)` with `strategy: typing.Literal["Zero", "One"]`: emitted by `parser/boolean.py` instead of eagerly lowering to `IntLiteral(0|1)`.
*   Internal-only literal IR nodes (produced by `resolve_and_lower_types` when lowering `FillNullWithStrategy`):
    *   `BytesLiteral(value: bytes)` -> hex-escaped `b'...'` (`b''` for empty bytes).
    *   `TimeLiteral(micros: int)` -> validated `0 <= micros < 86_400_000_000`, formatted as `TIME 'HH:MM:SS'` or `TIME 'HH:MM:SS.ffffff'`.
    *   `DecimalLiteral(value: str)` -> validated against `^[+-]?\d+(?:\.\d+)?$`, formatted as `NUMERIC '{value}'`.

#### Iterative Pre-Rewrite Typing and Lowering Policy (`core/compiler/typing.py`)

Evaluates iteratively via top-down BFS indexing followed by bottom-up reverse-BFS evaluation (zero Python recursion), tracking per-node state `(node, bq_type, time_unit, is_known_non_nan, is_known_non_null, weight)` and a node-expansion budget initialized from `MAX_AST_NODES - root_input_weight` (reusing `MAX_AST_NODES = 10_000` from `parser/__init__.py`, with `ListLiteral` weighted as `1 + len(values)` to match `parser/base.py:record_leaf_fanout`). Any unrecognized IR node fails closed to `Unsupported()`.

1.  **Valid `drop_nulls` sibling hoisting invariant**: Any invalid `DropNulls(...)` degrades bottom-up to bare `Unsupported()` (`operands=()`). Whenever a parent operator (or a pre-existing parser `Unsupported(operands=...)`) degrades to `Unsupported`, retain only individually type-valid children containing a `DropNulls` descendant inside `Unsupported(operands=(...))`; otherwise emit bare `Unsupported()`. Because every surviving `DropNulls` wraps a validated expression of one of the 14 supported BigQuery types—all of which accept `IS NOT NULL` in Storage Read API—`rewrite_drop_nulls` preserves valid sibling hoisting (such as `pl.col("a").drop_nulls() == unsupported(pl.col("b"))`) without ever hoisting invalid SQL.
2.  **Leaves and `ListLiteral` isolation**:
    *   `Column(name)` resolves `bq_type` from `bq_types` (falling back to `polars_dtype_to_bq_type`), and `time_unit` from `polars_datetime_time_unit`. Missing columns or `bq_type is None` degrade to bare `Unsupported()`.
    *   Scalar literals infer their corresponding `BQType` (`is_known_non_null = True`; `is_known_non_nan = True` except `FloatLiteral(NaN)`). `TimestampLiteral` also records its `time_unit`.
    *   `NullLiteral` has `bq_type = None`, `is_known_non_nan = True`, `is_known_non_null = False`. Operators distinguish `NullLiteral` from `Unsupported` via `isinstance` rather than `bq_type is None`: `NullLiteral` is accepted only in `And`, `Or`, `Not`, and `Ternary`/`When` value branches (at least one value branch must be non-`NullLiteral`; predicates cannot be `NullLiteral`), matching existing SQL emitter rules.
    *   `ListLiteral` carries `bq_type = BQType.ARRAY` (non-`BOOL` and `OPAQUE`), preventing boolean or numeric `ListLiteral`s from passing `And`, `Or`, `Not`, root validation, comparisons, `FillNull`, `Coalesce`, or `Ternary`. `ListLiteral` is also rejected in `IsNull`, `IsNotNull`, and `DropNulls`. Its homogeneous non-opaque scalar element type is inspected exclusively inside `IsIn`.
3.  **`BOOL` logical operators and root check**: `And`, `Or`, `Not` accept `BOOL`, `Unsupported`, and `NullLiteral`, rejecting `ListLiteral` or any known non-`BOOL` operand (catching bitwise integer `&`/`|`/`~` while preserving partial pushdown). At the root, reject `ListLiteral` or any known non-`BOOL` `bq_type`.
4.  **`OPAQUE` bucket (`JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`)**: Permitted only in `IsNull`, `IsNotNull`, and `DropNulls` (excluding `ListLiteral` and `NullLiteral`). All other operators reject `OPAQUE` operands.
5.  **String operators (`Uppercase`, `Lowercase`, `Contains`, `StartsWith`, `EndsWith`)**: Require strict `BQType.STRING` on all operands.
6.  **Float predicates (`IsNan`, `IsNotNan`, `IsInfinite`, `IsFinite`)**:
    *   `FLOAT64`: keep as-is (`BOOL`).
    *   `INT64`: lower to null-safe 3-valued `Ternary(IsNotNull(e), BoolLiteral(False|True), NullLiteral())` (guarded by remaining AST budget), folding to `FALSE` / `(e IS NOT NULL)` in positive filter context and emitting `IF((e IS NOT NULL), FALSE|TRUE, NULL)` inside `Not`, `IsNull`, `FillNull`, or `==`.
    *   `NUMERIC` and all other types: degrade to `Unsupported`.
7.  **`FillNullWithStrategy` lowering**: Polars fills temporal types with physical integer `0` or `1` (Unix epoch + 0 or 1 time unit), fills `Decimal(38, 9)` `"zero"`/`"one"` with `0`/`1`, and rejects `"one"` on `String`/`Binary`. Lower using the operand's inferred `(bq_type, time_unit)` (propagating `time_unit` through `DropNulls`/`FillNull`/`Coalesce`/`Ternary`, producing `None` on mismatch):
    *   `BOOL`: `False` / `True`; `INT64`: `0` / `1`; `FLOAT64`: `0.0` / `1.0`; `NUMERIC`: `DecimalLiteral("0")` / `DecimalLiteral("1")`; `DATE`: `DateLiteral(0)` / `DateLiteral(1)`.
    *   `STRING`: `"Zero"` -> `""`, `"One"` -> `Unsupported`; `BYTES`: `"Zero"` -> `BytesLiteral(b"")`, `"One"` -> `Unsupported`.
    *   `DATETIME` / `TIMESTAMP`: `"Zero"` -> `TimestampLiteral(0, MICROSECONDS, tz=None|"UTC")`; `"One"` -> `TimestampLiteral(1, MILLISECONDS if time_unit == "ms" else MICROSECONDS, tz=None|"UTC")` when `time_unit in ("ms", "us")`, else `Unsupported` (`"ns"` or unknown).
    *   `TIME`: `"Zero"` -> `TimeLiteral(0)`; `"One"` -> `Unsupported` (Polars fills `1 ns`, unrepresentable in microsecond `TIME`).
    *   `OPAQUE` / `None`: `Unsupported`.
8.  **Supertype unification (`FillNull`, `Coalesce`, `Ternary`, `When`, `IsIn`, comparisons)**:
    *   Valid pairwise supertypes over the 10 non-opaque scalar types: `T + T -> T`, `INT64 + FLOAT64 -> FLOAT64`, and `INT64 + NUMERIC -> NUMERIC`. All other pairs (`NUMERIC + FLOAT64`, cross-temporal mixes, `BYTES + STRING`, `BOOL + numeric`, any `OPAQUE` type) fail unification.
    *   Single-operand `Coalesce` and single-sided `Ternary` reject `OPAQUE`, `ListLiteral`, and all-`NullLiteral` value branches individually.
    *   `Ternary(p, t, f)` bottom-up partial pushdown: reject `NullLiteral`/`ListLiteral`/non-`BOOL` predicates (`p` may be `BOOL` or `Unsupported`); if either value branch is `Unsupported`, keep `Ternary` with `bq_type = BQType.BOOL` only when every remaining non-`NullLiteral`/non-`Unsupported` value branch has `BQType.BOOL` (including zero remaining known branches), enabling Stage 3 `rewrite_boolean_when` splitting; otherwise degrade. `When` is produced in Stage 3 (`rewrite_when`) and handled defensively with the same rules for direct IR callers.
    *   `IsIn(left, ListLiteral)`: strict same-type match between `left` and `_list_literal_element_type(right)` across the 10 non-opaque scalar types.
9.  **`FLOAT64` `NaN` comparison lowering**:
    *   After unifying `(left, right)` to `FLOAT64`, check **Case 0** (`left.is_known_non_nan and right.is_known_non_nan` -> unchanged).
    *   Otherwise apply exact 9-state 3-valued total-order lowering:
        *   **Case A — `FloatLiteral(NaN)`** (shown for `a op NaN`; symmetric for `NaN op b`): `Eq`/`GtEq` -> `IsNan(a)` (or `Ternary(IsNotNull(a), False, Null)` if `a.is_known_non_nan`); `NotEq`/`Lt` -> `Not(IsNan(a))` (or `Ternary(IsNotNull(a), True, Null)` if `a.is_known_non_nan`); `LtEq` -> `Ternary(IsNotNull(a), True, Null)`; `Gt` -> `Ternary(IsNotNull(a), False, Null)`.
        *   **Case B — `b` is `is_known_non_nan and is_known_non_null` and `a` can be `NaN`** (symmetric when `a` is non-NaN/non-null): `Eq`/`NotEq`/`Lt`/`LtEq` -> unchanged; `Gt` -> `Or(Gt(a, b), IsNan(a))`; `GtEq` -> `Or(GtEq(a, b), IsNan(a))`.
        *   **Case C — `b` is `is_known_non_nan` and nullable (`not b.is_known_non_null`) and `a` can be `NaN`** (symmetric): `Eq`/`NotEq`/`Lt`/`LtEq` -> unchanged; `Gt` -> `Or(Gt(a, b), And(IsNan(a), IsNotNull(b)))`; `GtEq` -> `Or(GtEq(a, b), And(IsNan(a), IsNotNull(b)))`.
        *   **Case D — Both `a` and `b` can be `NaN`**: `Eq` -> `Or(Eq(a, b), And(IsNan(a), IsNan(b)))`; `NotEq` -> `And(NotEq(a, b), Not(And(IsNan(a), IsNan(b))))`; `Gt` -> `Or(Gt(a, b), And(IsNan(a), Not(IsNan(b))))`; `GtEq` -> `Or(GtEq(a, b), And(IsNan(a), IsNotNull(b)))`; `Lt` -> `Or(Lt(a, b), And(Not(IsNan(a)), IsNan(b)))`; `LtEq` -> `Or(LtEq(a, b), And(IsNotNull(a), IsNan(b)))`.
    *   **Duplication budget**: Before applying any rewrite that duplicates subtree `x`, verify `x.weight <= 64` (`MAX_DUPLICATED_OPERAND_NODES`) and `extra_weight <= remaining_budget` (where `ListLiteral` contributes `1 + len(values)`). If exceeded, degrade to `Unsupported` and log at `DEBUG`.

### 6. End-User Documentation and API Reference

*   **New Data Types guide (`docs/source/api/data-types.md`, wired into `mkdocs.yml`)**: Document the 14-type mapping table (including `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` outer wrapping, `"UTC"` on `TIMESTAMP`, `Decimal(38, 9)` widening for parameterized `NUMERIC(P, S)`, and `_PARTITIONDATE` vs. `_PARTITIONTIME` granularity rules), `.ext.storage()` usage, `polars_bigquery.dtypes` explicit `schema=` examples, the extension registration contract (Mode A `BaseExtension` subclasses, Mode B `as_storage=True` / `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` on `schema=None` vs. explicit exclusion with explicit `schema=`), unsupported types with `UnsupportedTypeWarning` and `read_query()` workarounds, and `read_table()` vs. `scan_table()` differences.
*   **API index (`docs/source/api/index.md`)**: Link to `data-types.md` and register `polars_bigquery.dtypes` and `polars_bigquery.exceptions.UnsupportedTypeWarning`.
*   **Predicate pushdown reference (`docs/source/api/predicate-pushdown.md`)**:
    *   Replace the existing inline schema mapping table with a link to `data-types.md`.
    *   Add a per-type operator pushdown matrix covering all 14 supported types, typed `fill_null(strategy="zero"|"one")`, `FLOAT64` `NaN` total-order parity, `OPAQUE` null-check-only rules, and cross-type exclusions.
    *   Add a "Common pitfalls that disable pushdown" section (`NUMERIC` vs. `FLOAT64`, cross-temporal mixes, naive vs. UTC `_PARTITIONTIME`, integer bitwise `&`/`|`/`~`, and `OPAQUE` types).

## Alternatives Considered

### Alternative A: Support Both Polars 1.x and 2.x Dynamically for `JSON` and `GEOGRAPHY`

*   **Pros**: Keeps existing Polars `1.43.x` installs working without raising the dependency floor in `py-polars-bigquery/pyproject.toml`.
*   **Cons**: Polars 1.x defaults to `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` (`pl.String`), while Polars 2.0+ defaults to `load_as_extension` (`pl.Extension`). Maintaining both requires version-gated schema construction, dual documentation, and a split CI matrix.
*   **Reason for Rejection**: Polars 2.x is released and Polars 1.x compatibility is not required. Pinning `polars >= 2.0.0, < 3.0.0` leaves one `pl.Extension` default path aligned with `read_table()` and `read_query()`.

### Alternative B: Unwrap `JSON` and `GEOGRAPHY` to `pl.String` and Push `TO_JSON_STRING` / `ST_ASTEXT`

*   **Pros**: Lets users call `.str.*` directly on `JSON` and `GEOGRAPHY` columns without `.ext.storage()`, and avoids exposing Polars' `@unstable` `pl.Extension` dtype by default.
*   **Cons**: Live tests confirmed `CAST(JSON AS STRING)` and `CAST(GEOGRAPHY AS STRING)` fail in Storage Read API with `INVALID_ARGUMENT: Invalid cast`. Unwrapping extension columns inside `source_generator` makes `scan_table()` schemas disagree with `read_table()` and `read_query()`, erases BigQuery type identity in explicit `schema=` calls, and risks silent row loss if `TO_JSON_STRING` or `ST_ASTEXT` formats edge cases differently from BigQuery's Arrow serializer.
*   **Reason for Rejection**: Preserving native `pl.Extension` types in Polars 2.0+ keeps `scan_table()` consistent with `read_table()`/`read_query()` and avoids SQL-vs-Arrow string formatting drift.

### Alternative C: Skip Pushdown for Non-Numeric `fill_null(strategy=...)` and `INT64` Float Predicates

*   **Pros**: Smaller compiler footprint: avoids `FillNullWithStrategy`, internal `BytesLiteral`/`TimeLiteral`/`DecimalLiteral` emitters, `time_unit` propagation, and 3-valued `Ternary` rewrites for `INT64` `is_nan`/`is_inf`.
*   **Cons**: Leaves valid, representable filters (such as `pl.col("flag").fill_null(strategy="zero")`, `pl.col("label").fill_null(strategy="zero") == ""`, temporal `fill_null(strategy="zero"|"one")`, and `INT64` `is_nan`/`is_finite`) unpushed even though exact GoogleSQL equivalents exist.
*   **Reason for Rejection**: Performing exact type-aware rewrites pushes down all representable expressions while reserving fallback for unrepresentable or divergent cases (`TIME` `"one"`, `STRING`/`BYTES` `"one"`, `Datetime("ns")` `"one"`, `NUMERIC` `is_nan`/`is_inf`, `NUMERIC` vs. `FLOAT64`, and `OPAQUE` types).

### Alternative D: Fail Fast on Any Unsupported Column, or Strip Unsupported Nested `STRUCT` Subfields Now

*   **Pros**: Failing fast prevents schema differences between BigQuery and `lf.collect_schema()`; stripping only unsupported nested subfields would retain supported sibling fields inside a `STRUCT`.
*   **Cons**: Failing on a single unsupported column makes entire mixed tables unscannable even when the caller selects only `INT64`/`STRING` columns. Dropping a nested subfield from `polars_schema` breaks at runtime because BigQuery Storage Read API projects top-level columns in `selected_fields` and still transmits the omitted subfield. Supporting partial `STRUCT` exclusion requires subfield projection and client-side `STRUCT` reconstruction.
*   **Reason for Rejection**: Omitting the enclosing top-level column with `UnsupportedTypeWarning` unblocks mixed tables now without complex client-side `STRUCT` reconstruction; partial `STRUCT` subfield exclusion is deferred to Future Improvements.

### Alternative E: Add Schema Checks Only Inside `core/compiler/sql/` Without a Pre-Rewrite Pass

*   **Pros**: Keeps the existing 3-stage compiler structure (`json_to_ir -> rewrite_ir -> ir_to_sql`) by passing `schema` into `_compile_ir_to_sql`.
*   **Cons**: Runs after `rewrite_drop_nulls` (which would hoist `IsNotNull` for type-invalid subtrees), after `rewrite_boolean_when` (which inspects `has_descendant(Unsupported)` for partial pushdown and wraps predicates in `COALESCE(..., FALSE)`), and after `rewrite_de_morgan` (which would rewrite bitwise integer `~(a | b)` as logical negation). It also cannot distinguish `fill_null(strategy="zero")` from `fill_null(0)` if `parser/boolean.py` lowers strategies eagerly.
*   **Reason for Rejection**: Inserting `resolve_and_lower_types` between `json_to_ir` and `rewrite_ir` keeps `parser/` schema-blind, keeps `sql/` mechanical, and preserves partial-pushdown and boolean-simplification rewrites.

### Alternative F: Custom Upfront Explicit-`schema=` Validation and `BigQueryError` Interception in `_read_bigquery.py`

*   **Pros**: Could reword explicit-`schema=` mismatch errors (such as passing `pl.String` for a `JSON` column) to point directly to `polars_bigquery.dtypes`.
*   **Cons**: `py-arrow-bigquery/src/stream.rs` (`ArrowStreamExporter`) implements only a single-use `__arrow_c_stream__` (`self.source.lock().take()`) and does not implement `__arrow_c_schema__`, so calling `lazyframe.collect_schema()` before `collect_batches()` risks `ValueError("Stream already consumed")` unless a new Rust PyO3 method is added. Intercepting `BigQueryError` on string predicates also risks attaching misleading `JSON`/`GEOGRAPHY` hints to unrelated `403`/`404`/quota/network errors.
*   **Reason for Rejection**: Polars' native `register_io_source` `SchemaError` already reports expected vs. actual dtypes (including extension names) and BigQuery already reports SQL type errors; relying on native errors plus `data-types.md` avoids cross-language validation complexity and error-wrapping heuristics.

### Alternative G: Python-Only `pl.Duration` Check Instead of Rust `MonthDayNano` Guard

*   **Pros**: Keeps all changes inside `py-polars-bigquery` without touching `crates/arrow-bigquery`.
*   **Cons**: Checking `pl.Duration` in `scan_table(schema=...)` still leaves `read_table()`, `read_query()`, and `scan_table(schema={"dur": pl.String})` exposed to uncatchable `SIGABRT` process aborts inside `polars_arrow::ffi::export_iterator`.
*   **Reason for Rejection**: Guarding `ArrowDataType::Interval(IntervalUnit::MonthDayNano)` once inside `crates/arrow-bigquery/src/decode.rs:parse_arrow_schema` protects every Python and Rust reader uniformly in ~20 lines.

## Risks & Mitigations

| Risk / Limit | Impact | Mitigation |
| --- | --- | --- |
| **Polars 2.x `@unstable` API drift** (`pl.Extension`, `pl.get_extension_type`) under `polars >= 2.0.0, < 3.0.0` | Minor Polars `2.x` updates could alter extension constructor signatures or emit `UnstableWarning`. | Avoid `@unstable pl.DataFrame(height=...)` by using stable `batch.drop(batch.columns)`; isolate `pl.Extension` and `pl.get_extension_type` inside `core/types.py`; run unit tests in CI against both `--resolution lowest` (`2.0.0`) and latest `2.x`. |
| **BigQuery Storage Read API 1 MB `row_restriction` limit** | Duplicating compound float expressions (especially subtrees containing large `IsIn` lists) could exceed 1 MB and fail with `400 INVALID_ARGUMENT`. | Weight `ListLiteral` as `1 + len(values)` in `typing.py`, enforce `MAX_DUPLICATED_OPERAND_NODES = 64` and `MAX_AST_NODES = 10_000`, and add a final `1_000_000`-byte UTF-8 fallback guard in `_json_expr_to_row_restriction`. |
| **Mode B (`as_storage=True` / `load_as_storage`) with explicit `schema=`** | Without `tables.get` metadata, `pl.String` is indistinguishable from BigQuery `STRING`, so pushing string filters onto a `JSON`/`GEOGRAPHY` column causes BigQuery `400 INVALID_ARGUMENT`. | Explicitly exclude Mode B with explicit `schema=` for ambiguous storage types in `data-types.md` and `scan_table` docstrings; pass authoritative `bq_types` from `ExtractedSchema` on `schema=None` scans so Mode B remains type-safe there. |
| **Polars temporal `fill_null(strategy="zero"|"one")` behavior changes across `2.x`** | If Polars changes physical-integer fill behavior for temporal or decimal dtypes in a future `2.x` release, pushed filters could diverge from client-side Polars. | Pin differential integration tests (`scan_table().filter(...)` vs. `read_table().filter(...)`) across all supported types and `fill_null` strategies in `tests/system/test_read_bigquery_integration.py`. |
| **Silent column omission when Python warnings are suppressed** | Callers running with `warnings.simplefilter("ignore")` will not see `UnsupportedTypeWarning` when a column is omitted. | Emit structured `DEBUG` logs alongside `UnsupportedTypeWarning`, document `warnings.filterwarnings("error", category=UnsupportedTypeWarning)`, and fail with `ValueError` when zero supported columns remain. |
| **Hourly vs. daily ingestion-time partition pseudo-columns** | Requesting `_PARTITIONDATE` on an `HOUR`-partitioned table fails in Storage Read API with `INVALID_ARGUMENT`. | Gate `_PARTITIONDATE` on `DAY` partition granularity while always retaining `_PARTITIONTIME`, and cover both hourly and daily tables in unit and system tests. |

## Non-Functional Requirements

*   **Performance & Billing Cost**:
    *   Selecting the narrowest supported column (`BOOL` < `DATE` < 8-byte scalars < `NUMERIC` < variable/scalar extension < `ARRAY`/`STRUCT`) on empty projections prevents full-table or wide-column scans for `lf.select(pl.len())`.
    *   `resolve_and_lower_types` uses iterative BFS/reverse-BFS traversal (zero Python recursion) and bounds AST expansion (`MAX_DUPLICATED_OPERAND_NODES = 64`, `MAX_AST_NODES = 10_000`, `MAX_ROW_RESTRICTION_BYTES = 1_000_000`).
    *   `DEBUG` logs record table ID, `selected_fields`, compiled `row_restriction`, skipped columns, and type-driven pushdown fallbacks.
*   **Compatibility (Accepted Pre-1.0 Breaking Changes)**:
    *   Raising the dependency floor to `polars >= 2.0.0, < 3.0.0`.
    *   Changing `TIMESTAMP` and `_PARTITIONTIME` timezone from `"utc"` to `"UTC"`.
    *   Mapping `JSON`/`GEOGRAPHY` (and `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>`) to `pl.Extension(...)` by default.
    *   Changing single unsupported columns from `TypeError` to `UnsupportedTypeWarning` omission (and raising `BigQueryError` in Rust on `INTERVAL` instead of aborting).
    *   Making `schema` required on internal `predicate_to_row_restriction`.
*   **Security & Privacy**:
    *   `DEBUG` logs emit compiled SQL and column names (consistent with existing `parser/` debug logs) and never log OAuth tokens or row values beyond literal filter constants already present in the caller's Polars expression.
    *   Internal literal formatters enforce defensive validation: `BytesLiteral` hex-escapes every byte (`\x{b:02x}`), `DecimalLiteral` validates `^[+-]?\d+(?:\.\d+)?$`, and `TimeLiteral` validates integer bounds `0 <= micros < 86_400_000_000` (rejecting `bool`).

## Additional Resources

*   [Existing Row Predicate Pushdown Design](./polars_bigquery_scan_table_row_predicate_pushdown.md)
*   [BigQuery REST `Table` and `TableFieldSchema` Reference](https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema)
*   [BigQuery Storage Read API `TableReadOptions` Reference](https://cloud.google.com/bigquery/docs/reference/storage/rpc/google.cloud.bigquery.storage.v1#tablereadoptions)
*   Upstream Polars `MonthDayNano` issues: [pola-rs/polars#30920](https://github.com/pola-rs/polars/issues/30920) and [pola-rs/polars#30921](https://github.com/pola-rs/polars/issues/30921)

## Future Improvements

*   **Partial `STRUCT` subfield exclusion**: Project only supported subfields from a `STRUCT` containing an unsupported descendant and reconstruct pruned `STRUCT` columns client-side.
*   **User-facing `BYTES`, `TIME`, and `Decimal` literal and `IsIn` parsing**: Extend `core/compiler/parser/` to parse user-supplied `pl.Binary`, `pl.Time` (microsecond-aligned), and `pl.Decimal` scalar literals and `is_in` collections.
*   **Deferred BigQuery types**: Add full Arrow/Polars support for `BIGNUMERIC`, preserving narrow `(P, S)` Arrow metadata for parameterized `NUMERIC(P, S)` (currently scanned via widening to `Decimal(38, 9)`), picosecond `TIMESTAMP`, `INTERVAL` (once [pola-rs/polars#30920](https://github.com/pola-rs/polars/issues/30920) and [pola-rs/polars#30921](https://github.com/pola-rs/polars/issues/30921) are resolved), `RANGE<T>`, and unnamed/duplicate `STRUCT` subfields.
*   **`scan_query` integration**: Reuse `core/types.py`, `core/schema.py`, and the schema-aware compiler once query-backed lazy scanning is designed.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

The work is partitioned into **five incremental phases**. Each phase pairs implementation edits with corresponding tests and leaves the workspace buildable and green at its checkpoint.

---

### Phase 1: Rust-Layer `MonthDayNano` (`INTERVAL`) Guard

#### 1. `crates/arrow-bigquery/src/error.rs` (Modify)

*   **Type / Scope**: Rust Error Definitions
*   **Proposed Changes**:
    *   Add `UnsupportedType(String)` variant to `BigQueryError`.
    *   Implement `Display` as `"Unsupported BigQuery data type: {msg}"` and return `None` from `std::error::Error::source()` (so `py-arrow-bigquery/src/error.rs` maps it to Python `BigQueryError`).

#### 2. `crates/arrow-bigquery/src/decode.rs` (Modify)

*   **Type / Scope**: Rust Arrow IPC Schema Decoder
*   **Proposed Changes**:
    *   Add an internal helper `validate_supported_arrow_schema(schema: &ArrowSchema) -> Result<(), BigQueryError>` that recursively walks every field (descending through `ArrowDataType::List`, `LargeList`, `FixedSizeList`, `Struct`, `Map`, and `Extension`).
    *   On encountering `ArrowDataType::Interval(IntervalUnit::MonthDayNano)`, return `Err(BigQueryError::UnsupportedType(...))` naming the offending field path and citing `https://github.com/pola-rs/polars/issues/30920` and `https://github.com/pola-rs/polars/issues/30921`.
    *   Invoke `validate_supported_arrow_schema(&metadata.schema)?` inside `parse_arrow_schema(schema_bytes)` right after `read_stream_metadata(&mut cursor)?`.
    *   Add Rust unit tests in `crates/arrow-bigquery/src/decode.rs` covering top-level, `List`-nested, and `Struct`-nested `Interval(MonthDayNano)` schemas alongside supported schemas.

*   **Phase 1 Checkpoint**:
    *   `cargo test --workspace --exclude py-arrow-bigquery --exclude py-polars-expressions-testing --lib --bins --features testing`
    *   `uv run --project py-arrow-bigquery --group test pytest -n auto py-arrow-bigquery/tests/unit`

---

### Phase 2: Polars 2.0+ Floor, Public `dtypes`, Exceptions, and Canonical Type Layer

#### 3. `py-polars-bigquery/pyproject.toml` (Modify)

*   **Type / Scope**: Python Package Configuration
*   **Proposed Changes**: Raise `polars` dependency specifier from `polars >= 1.43.0, < 3.0.0` to `polars >= 2.0.0, < 3.0.0`.

#### 4. `uv.lock` (Modify)

*   **Type / Scope**: Workspace Dependency Lockfile
*   **Proposed Changes**: Re-lock workspace dependencies via `uv lock` and sync (`uv sync --project py-polars-bigquery --group dev`) so the locked Polars version satisfies `>= 2.0.0, < 3.0.0` (currently resolving to `2.1.x`).

#### 5. `py-polars-bigquery/polars_bigquery/core/types.py` (Create)

*   **Type / Scope**: Internal Type System Module
*   **Proposed Changes**: Centralize shared type definitions and helpers for `dtypes.py`, `core/schema.py`, `_read_bigquery.py`, and `core/compiler/`:
    *   `BQType(enum.Enum)`: closed 14-member enumeration of all supported BigQuery types (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`).
    *   `OPAQUE_BQ_TYPES: frozenset[BQType]`: `{BQType.JSON, BQType.GEOGRAPHY, BQType.ARRAY, BQType.STRUCT}`.
    *   Extension constants: `JSON_EXTENSION_NAME = "google:sqlType:json"`, `GEOGRAPHY_EXTENSION_NAME = "google:sqlType:geography"`, `GEOGRAPHY_EXTENSION_METADATA = '{"encoding": "WKT"}'`.
    *   `build_extension_dtype(name: str, storage: DataType, metadata: str | None = None) -> DataType`:
        1. Check `registered = pl.get_extension_type(name)`.
        2. If `isinstance(registered, type) and issubclass(registered, pl.BaseExtension)`: return `registered.ext_from_params(name, storage, metadata)` (Mode A).
        3. If `registered == "storage"`: return `storage` (Mode B).
        4. If `os.environ.get("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR") == "load_as_storage"`: return `storage` (Mode B).
        5. Otherwise: return `pl.Extension(name, storage, metadata)`.
    *   `polars_dtype_to_bq_type(dtype: DataType | DataTypeClass) -> BQType | None`:
        *   Handle uninstantiated `DataTypeClass` objects via `issubclass` dispatch (without calling `dtype()` on parameterized classes such as `pl.List`, `pl.Struct`, `pl.Array`, `pl.Enum`, `pl.Map`, `pl.Extension`, `pl.BaseExtension`): `pl.List` -> `ARRAY`, `pl.Struct` -> `STRUCT`, `pl.Datetime` -> `DATETIME`, `pl.Decimal` (unknown precision/scale) and parameterized non-BQ classes -> `None`; instantiate only non-parameterized classes (`pl.Boolean`, `pl.Int64`, etc.).
        *   For `pl.BaseExtension` instances: inspect `dtype.ext_name()`; for `JSON_EXTENSION_NAME` return `ARRAY` if `isinstance(dtype.ext_storage(), pl.List)` else `JSON`; for `GEOGRAPHY_EXTENSION_NAME` return `ARRAY` if `isinstance(dtype.ext_storage(), pl.List)` else `GEOGRAPHY`; return `None` for any other extension name.
        *   Map explicit integer classes (`Int8`, `Int16`, `Int32`, `Int64`, `Int128`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, excluding `UInt128`) -> `INT64`, and explicit float classes (`Float32`, `Float64`, excluding `Float16`) -> `FLOAT64` (do not use bare `dtype.is_integer()` / `dtype.is_float()`, which include `UInt128` and `Float16` in Polars 2.1.0).
        *   Map `pl.Decimal` -> `NUMERIC` only when `dtype.precision is not None and dtype.precision <= 38 and dtype.scale <= 9`; otherwise return `None`.
        *   Map `Boolean` -> `BOOL`, `String` -> `STRING`, `Binary` -> `BYTES`, `Date` -> `DATE`, `Datetime(time_zone=None)` -> `DATETIME`, `Datetime(time_zone!=None)` -> `TIMESTAMP`, `Time` -> `TIME`, `List` -> `ARRAY`, `Struct` -> `STRUCT`, and all other Polars dtypes -> `None`.
    *   `polars_datetime_time_unit(dtype: DataType | DataTypeClass) -> Literal["ms", "us", "ns"] | None`: returns `dtype.time_unit` for `pl.Datetime` instances (`"us"` for uninstantiated `pl.Datetime`), else `None`.
    *   `column_width_rank(dtype: DataType | DataTypeClass) -> int`: delegate to `polars_dtype_to_bq_type(dtype)` via `_BQ_TYPE_WIDTH_RANK.get(bq_type, 6)` where `BOOL` = 1, `DATE` = 2, `INT64`/`FLOAT64`/`TIME`/`DATETIME`/`TIMESTAMP` = 3, `NUMERIC` = 4, `STRING`/`BYTES`/`JSON`/`GEOGRAPHY` = 5, `ARRAY`/`STRUCT`/`None` = 6 (so `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` get rank 6 rather than 5).

#### 6. `py-polars-bigquery/polars_bigquery/dtypes.py` (Create)

*   **Type / Scope**: Public Python API Module
*   **Proposed Changes**: Define and export (`__all__ = ["JSON", "GEOGRAPHY"]`) scalar BigQuery extension dtypes with explicit type annotations and PEP 224 attribute docstrings for `mkdocstrings`/Griffe:
    *   `JSON: pl.DataType = pl.Extension(JSON_EXTENSION_NAME, pl.String())`
    *   `GEOGRAPHY: pl.DataType = pl.Extension(GEOGRAPHY_EXTENSION_NAME, pl.String(), GEOGRAPHY_EXTENSION_METADATA)`
    *   Document that these constants represent the default unregistered `pl.Extension` dtypes.

#### 7. `py-polars-bigquery/polars_bigquery/exceptions.py` (Modify)

*   **Type / Scope**: Public Exceptions & Warnings Module
*   **Proposed Changes**: Add `UnsupportedTypeWarning(UserWarning)` with a NumPy-style docstring describing its emission by `Client.scan_table` when unsupported BigQuery columns are omitted, and export it in `__all__`.

#### 8. `py-polars-bigquery/polars_bigquery/__init__.py` (Modify)

*   **Type / Scope**: Package Entrypoint
*   **Proposed Changes**: Import and export `dtypes` in `__all__` so `polars_bigquery.dtypes.JSON` and `polars_bigquery.dtypes.GEOGRAPHY` are available after `import polars_bigquery`.

#### 9. `py-polars-bigquery/tests/unit/core/test_types.py` (Create)

*   **Type / Scope**: Unit Tests
*   **Proposed Changes**:
    *   Verify `polars_bigquery.dtypes` (`JSON`, `GEOGRAPHY`) names, storage dtypes, and metadata.
    *   Test `build_extension_dtype()` across default `pl.Extension`, Mode A registered `BaseExtension` subclass, Mode B `as_storage=True` (with cleanup via `pl.unregister_extension_type` in `try/finally`), and Mode B `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` (via `monkeypatch`).
    *   Test `polars_dtype_to_bq_type()` across all supported and unsupported Polars dtypes, uninstantiated parameterized `DataTypeClass` objects (`pl.List`, `pl.Struct`, `pl.Array`, `pl.Extension`, `pl.Decimal`, `pl.Datetime`), `Decimal(precision=None, scale=2)`, `Decimal(38, 9)` vs. `Decimal(38, 10)`, `Float16`, `UInt128`, and outer-wrapped `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>`.
    *   Test `polars_datetime_time_unit()` and `column_width_rank()` (including `ARRAY<JSON>` -> rank 6 and non-BQ explicit-`schema=` dtypes -> rank 6).

*   **Phase 2 Checkpoint**:
    *   `uv sync --project py-polars-bigquery --group dev`
    *   `uv run --project py-polars-bigquery python -c "import polars; assert polars.__version__.startswith('2.')"`
    *   `uv run --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --resolution lowest --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --project py-polars-bigquery --group dev ruff check --no-fix py-polars-bigquery`

---

### Phase 3: Schema Extraction and Projection Safety (`core/schema.py`, `_read_bigquery.py`)

#### 10. `py-polars-bigquery/polars_bigquery/core/schema.py` (Modify)

*   **Type / Scope**: Internal Schema Extraction Module
*   **Proposed Changes**:
    *   Remove the `Duration` import (currently used at `schema.py:75` for `interval`, becoming unused once `interval` is omitted).
    *   Add frozen dataclasses:
        *   `SkippedColumn(name: str, reason: str, path: str)`
        *   `ExtractedSchema(polars_schema: pl.Schema, bq_types: dict[str, BQType], skipped_columns: tuple[SkippedColumn, ...])`
    *   Add internal `_UnsupportedBigQueryType(Exception)` carrying `(reason: str, path: str)` so any unsupported descendant inside `STRUCT` or `ARRAY` propagates to omit the enclosing top-level column.
    *   Define `_extract_data_type(field: dict, path: str) -> tuple[DataType, BQType]`:
        *   Extract `(field.get("mode") or "").casefold()` and `(field.get("type") or "").casefold()` null-safely.
        *   For `mode == "repeated"` and `type_ in ("json", "geography")`: return `(build_extension_dtype(..., pl.List(pl.String()), ...), BQType.ARRAY)`.
        *   For all other `mode == "repeated"` fields: recurse with `{**field, "mode": "NULLABLE"}` and same `path`, returning `(pl.List(inner_dtype), BQType.ARRAY)`.
        *   For `timestamp`: accept `precision = field.get("timestampPrecision")` only when `precision is None or (not isinstance(precision, bool) and precision in (6, "6"))`; return `(Datetime(time_unit="us", time_zone="UTC"), BQType.TIMESTAMP)`; otherwise raise `_UnsupportedBigQueryType(f"TIMESTAMP(precision={precision})", path)`.
        *   Map scalar `json` and `geography` via `build_extension_dtype(...)` returning `BQType.JSON` / `BQType.GEOGRAPHY`.
        *   Raise `_UnsupportedBigQueryType` for `bignumeric`/`bigdecimal` (`"BIGNUMERIC"`), `interval` (`"INTERVAL"`), `range` (`"RANGE"`), and unrecognized types (`f"unsupported type {type_ or '<missing>'}"`).
        *   For `record`/`struct`: read `subfields = field.get("fields") or []`. Reject empty `subfields` (`"empty STRUCT"`, `path`). Track seen subfield names; for subfield at index `i`, if `not (sub_name := subfield.get("name"))` raise `_UnsupportedBigQueryType("unnamed STRUCT subfield", f"{path}[{i}]")`; if `sub_name in seen` raise `_UnsupportedBigQueryType("duplicate STRUCT subfield", f"{path}.{sub_name}")`; recurse with `f"{path}.{sub_name}"` and return `(Struct(...), BQType.STRUCT)`.
    *   Add `extract_schema_metadata(table_metadata: dict) -> ExtractedSchema` without warning or exception side effects:
        *   Iterate `(table_metadata.get("schema") or {}).get("fields") or []`. If a top-level field has missing/empty `"name"`, record `SkippedColumn(name=f"<unnamed[{i}]>", reason="unnamed column", path=f"<unnamed[{i}]>")`. Otherwise call `_extract_data_type(field, name)` and record into `polars_schema`/`bq_types` or catch `_UnsupportedBigQueryType` and append `SkippedColumn(name=name, reason=exc.reason, path=exc.path)`.
        *   For ingestion-time partitioned tables (`(time_partitioning := table_metadata.get("timePartitioning")) is not None and time_partitioning.get("field") is None`):
            *   If `(time_partitioning.get("type") or "DAY").casefold() == "day"`: set `_PARTITIONDATE` -> `Date()` (`BQType.DATE`).
            *   Always set `_PARTITIONTIME` -> `Datetime(time_unit="us", time_zone="UTC")` (`BQType.TIMESTAMP`).
    *   Keep `extract_polars_schema(table_metadata: dict) -> pl.Schema` returning `extract_schema_metadata(table_metadata).polars_schema`.

#### 11. `py-polars-bigquery/polars_bigquery/_read_bigquery.py` (Modify — Phase 3 non-compiler updates)

*   **Type / Scope**: Public Client & Polars I/O Plugin Module
*   **Proposed Changes**:
    *   Add module logger `logger = logging.getLogger(__name__)`.
    *   In `Client.scan_table`:
        *   When `schema is None`: call `extracted = extract_schema_metadata(table_metadata)` and retain `bq_types = extracted.bq_types`.
            *   Format each skipped entry as `f"{c.name} ({c.reason})"` when `c.path == c.name` else `f"{c.name} ({c.reason} at {c.path})"`, capped at 5 entries with `f"... and {len(skipped) - 5} more"`.
            *   If `extracted.skipped_columns` is non-empty and `len(extracted.polars_schema) == 0`: raise `ValueError` (without emitting `UnsupportedTypeWarning`) listing the omitted columns and a `client.read_query()` workaround.
            *   Else if `extracted.skipped_columns` is non-empty: call `warnings.warn(..., UnsupportedTypeWarning, stacklevel=2)` directly inside `Client.scan_table` (not via a helper, preserving `stacklevel=2`), and log skipped columns at `DEBUG`.
        *   When `schema is not None`: normalize via `pl.Schema(schema)` and set `bq_types = None`.
    *   In `source_generator(with_columns, predicate, n_rows, batch_size)` (keeping it a regular function returning `Iterator[pl.DataFrame]` on every branch — never mixing `yield` with `return`):
        *   If `len(polars_schema) == 0`: `return iter([pl.DataFrame(schema=polars_schema)])` without calling `read_table()`.
        *   Compute `columns`, `predicate_cols`, and `selected_fields = list(dict.fromkeys([*columns, *predicate_cols]))`. If `not selected_fields`, choose the single column in `polars_schema` with minimum `column_width_rank(dtype)` (tie-breaking by schema order).
        *   Emit a `DEBUG` log with `table_ref`, `selected_fields`, and `row_restriction`.
        *   After `lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)`, apply `filter(predicate)` (if `predicate is not None`) and `limit(n_rows)` (if `n_rows is not None`).
        *   If `columns == []`: `return (batch.drop(batch.columns) for batch in lazyframe.collect_batches(chunk_size=batch_size))` (preserving row height via stable Polars API).
        *   Otherwise: `return lazyframe.select(columns).collect_batches(chunk_size=batch_size)`.

#### 12. `py-polars-bigquery/tests/unit/core/test_schema.py` (Modify)

*   **Type / Scope**: Unit Tests
*   **Proposed Changes**:
    *   Update existing assertions to `"UTC"`, `pl.Extension` for `JSON`/`GEOGRAPHY`, and omission semantics.
    *   Add tests for `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>` outer-extension wrapping (top-level and inside `STRUCT`), `ARRAY<STRUCT<j JSON, g GEOGRAPHY>>`, and `REPEATED` parameter preservation (`ARRAY<TIMESTAMP(12)>`).
    *   Add tests for `timestampPrecision` allowlist (`None`, `6`, `"6"`, `12`, `"12"`, `"9"`, `True`) and null-safe `schema`, `fields`, `mode`, `type`, `name` values.
    *   Add tests for unnamed, duplicate, and empty `STRUCT` omission, top-level unnamed fields, nested whole-column omission with exact `SkippedColumn` paths, and `ExtractedSchema.bq_types` preservation under Mode B (`as_storage=True` / `load_as_storage`).
    *   Add tests for ingestion-time partitioned tables across `DAY` (retains `_PARTITIONDATE` + `_PARTITIONTIME`), `HOUR` (retains only `_PARTITIONTIME`), `MONTH`/`YEAR` (retains only `_PARTITIONTIME`), and 0-column tables.

#### 13. `py-polars-bigquery/tests/unit/test_read_bigquery.py` (Modify — Phase 3 projection & warning tests)

*   **Type / Scope**: Unit Tests (Mocked & Non-Mocked Arrow C Stream)
*   **Proposed Changes**:
    *   Add tests for `UnsupportedTypeWarning` (message format for top-level vs. nested paths, 5-column cap + `... and N more`, `stacklevel`, `read_query()` hint).
    *   Add tests for `ValueError` on non-partitioned all-unsupported tables (verifying no `UnsupportedTypeWarning` is emitted), daily ingestion-time partitioned table retaining `_PARTITIONDATE`/`_PARTITIONTIME`, and hourly ingestion-time partitioned table retaining only `_PARTITIONTIME`.
    *   Add tests for empty-schema short-circuit (0-column table and explicit `schema={}` — verifying `read_table` is not called).
    *   Add tests for lowest-rank fallback `selected_fields` and height preservation on `with_columns=[]` (`lf.select(pl.len())`).
    *   Add non-mocked Arrow C stream unit tests passing `tsu:UTC` and `ARROW:extension:name`/`metadata` capsules through `pl.scan_arrow_c_stream` and `register_io_source`.

*   **Phase 3 Checkpoint**:
    *   `uv run --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --project py-polars-bigquery --group dev ruff check --no-fix py-polars-bigquery`

---

### Phase 4: 4-Stage Schema-Aware Compiler Pipeline and `_read_bigquery.py` Wiring

#### 14. `py-polars-bigquery/polars_bigquery/core/compiler/ir/boolean.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Import `typing` (or alias `TypingLiteral` to avoid colliding with `ir.base.Literal` imported at line 7) and add frozen dataclass `FillNullWithStrategy(UnaryExpr)` with `strategy: TypingLiteral["Zero", "One"]`.

#### 15. `py-polars-bigquery/polars_bigquery/core/compiler/ir/numeric.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Add internal frozen dataclass `DecimalLiteral(Literal)` with field `value: str`.

#### 16. `py-polars-bigquery/polars_bigquery/core/compiler/ir/string.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Add internal frozen dataclass `BytesLiteral(Literal)` with field `value: bytes`.

#### 17. `py-polars-bigquery/polars_bigquery/core/compiler/ir/temporal.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Add internal frozen dataclass `TimeLiteral(Literal)` with field `micros: int`.

#### 18. `py-polars-bigquery/polars_bigquery/core/compiler/ir/__init__.py` (Modify)

*   **Type / Scope**: Compiler IR Package Exports
*   **Proposed Changes**: Import and export `FillNullWithStrategy`, `BytesLiteral`, `DecimalLiteral`, and `TimeLiteral` in `__all__`.

#### 19. `py-polars-bigquery/polars_bigquery/core/compiler/parser/boolean.py` (Modify)

*   **Type / Scope**: Compiler Parser Module
*   **Proposed Changes**: Remove `_FILL_NULL_STRATEGY_VALUES`, `_construct_fill_null_with_strategy`, `IntLiteral`, and the newly unused `Mapping` import (preventing `ruff` `F401`); update `parse_fill_null_with_strategy` to validate `spec in ("Zero", "One")` and build `FillNullWithStrategy(expr=..., strategy=spec)`.

#### 20. `py-polars-bigquery/polars_bigquery/core/compiler/typing.py` (Create)

*   **Type / Scope**: Schema-Aware Compiler Pass
*   **Proposed Changes**: Implement `resolve_and_lower_types(root: Expr, schema: pl.Schema, bq_types: Mapping[str, BQType] | None = None) -> Expr` using an iterative two-pass traversal (top-down BFS indexing + bottom-up reverse-BFS evaluation) following the `@functools.singledispatch` pattern in `sql/base.py`, reusing `MAX_AST_NODES` from `parser/__init__.py`, and defining `MAX_DUPLICATED_OPERAND_NODES = 64`:
    *   Track per-node state `_TypedNode(node, bq_type, time_unit, is_known_non_nan, is_known_non_null, weight)`:
        *   Leaf weights: `1` for scalar leaves; `1 + len(node.values)` for `ListLiteral`. Initialize `remaining_budget = max(0, MAX_AST_NODES - total_input_weight)` and deduct `extra_weight` on node-expanding rewrites (`INT64` float predicates, `FillNullWithStrategy`, `FLOAT64` `NaN` lowering).
        *   Propagation rules:
            *   `Column`: `bq_type` from `bq_types` (fallback `polars_dtype_to_bq_type`), `time_unit` from `polars_datetime_time_unit`, `is_known_non_nan = (bq_type != BQType.FLOAT64)`, `is_known_non_null = False`. Missing column or `bq_type is None` -> bare `Unsupported()`.
            *   Non-`NullLiteral` scalar literals (including `DecimalLiteral`, `BytesLiteral`, `TimeLiteral`, `TimestampLiteral`): `is_known_non_nan = True` (except `FloatLiteral(NaN)` -> `False`), `is_known_non_null = True`; `TimestampLiteral` sets `time_unit` from `node.unit`.
            *   `NullLiteral`: `bq_type = None`, `is_known_non_nan = True`, `is_known_non_null = False`.
            *   `ListLiteral`: `bq_type = BQType.ARRAY`, `is_known_non_nan = True`, `is_known_non_null = True`.
            *   `DropNulls(e)`: preserves `e.bq_type`, `e.time_unit`, `e.is_known_non_nan`; sets `is_known_non_null = False` conservatively.
            *   `FillNull(l, r)`: `time_unit` if equal else `None`; `is_known_non_nan = l.is_known_non_nan and r.is_known_non_nan`; `is_known_non_null = l.is_known_non_null or r.is_known_non_null`.
            *   `Coalesce(ops)`: `time_unit` if all non-`NullLiteral` operands match else `None`; `is_known_non_nan = all(op.is_known_non_nan for op in ops)`; `is_known_non_null = any(op.is_known_non_null for op in ops)`.
            *   `Ternary(p, t, f)` (and defensive `When`): `time_unit` if all non-`NullLiteral` value branches match else `None`; `is_known_non_nan = all(b.is_known_non_nan for b in value_branches)`; `is_known_non_null = False` conservatively unless `p.is_known_non_null` and all value branches have `is_known_non_null`.
    *   Helper `_degrade(children)` preserving valid `drop_nulls` sibling hoisting: collect any child whose `node` is not bare `Unsupported()` and has a `DropNulls` descendant (or is `Unsupported(operands=...)` already holding valid `DropNulls` children); return `Unsupported(operands=tuple(kept))` if non-empty else bare `Unsupported()`. Apply `_degrade` to degraded operators and normalize pre-existing parser `Unsupported(operands=...)` nodes identically.
    *   `NullLiteral` vs. `Unsupported` vs. `ListLiteral` enforcement:
        *   `And`, `Or`, `Not`: accept `BOOL`, `Unsupported`, and `NullLiteral`; reject `ListLiteral` and any known non-`BOOL` `bq_type`.
        *   `IsNull`, `IsNotNull`, `DropNulls`: accept any supported `BQType` (including `OPAQUE_BQ_TYPES`), rejecting `NullLiteral`, `ListLiteral`, `Unsupported`, and `bq_type is None`.
        *   `Uppercase`, `Lowercase`, `Contains`, `StartsWith`, `EndsWith`: require strict `BQType.STRING` (rejecting `NullLiteral`, `ListLiteral`, `Unsupported`, `OPAQUE`).
        *   `IsNan`, `IsNotNan`, `IsInfinite`, `IsFinite`: keep for `FLOAT64`; lower `INT64` to `Ternary(IsNotNull(e), BoolLiteral(False|True), NullLiteral())` (budget-checked); degrade `NUMERIC` and all other types.
        *   `FillNullWithStrategy`: lower across `BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING` (`"Zero"`), `BYTES` (`"Zero"`), `DATE`, `DATETIME` (`"Zero"`, or `"One"` when `time_unit in ("ms", "us")`), `TIMESTAMP` (`"Zero"`, or `"One"` when `time_unit in ("ms", "us")`), and `TIME` (`"Zero"` only); degrade `"One"` on `STRING`/`BYTES`/`TIME`/`Datetime("ns")`/unknown `time_unit` and all `OPAQUE`/`None` operands.
        *   `FillNull` and `Coalesce`: reject `NullLiteral`, `ListLiteral`, `Unsupported`, and `OPAQUE` (including single-operand `Coalesce`); unify via `_unify_types` (`T + T`, `INT64 + FLOAT64 -> FLOAT64`, `INT64 + NUMERIC -> NUMERIC`).
        *   `Ternary` (and defensive `When`): reject `NullLiteral`/`ListLiteral`/non-`BOOL` predicates (`p` may be `BOOL` or `Unsupported`); reject `ListLiteral` or `OPAQUE` value branches and all-`NullLiteral` value branches; when any value branch is `Unsupported`, preserve `Ternary` with `bq_type = BQType.BOOL` only if all remaining non-`NullLiteral`/non-`Unsupported` value branches have `BQType.BOOL` (including zero remaining known branches), else degrade; otherwise unify non-`NullLiteral` value branches via `_unify_types`.
        *   `IsIn(left, right)`: require non-opaque scalar `left` (`not isinstance(left, (NullLiteral, ListLiteral, Unsupported))`) and `isinstance(right, ListLiteral)`; validate `_list_literal_element_type(right)` (non-empty, no `NullLiteral`/`ListLiteral`/`FloatLiteral(NaN)`, homogeneous non-opaque scalar `BQType == left.bq_type`).
        *   Comparisons (`Eq`, `NotEq`, `Gt`, `GtEq`, `Lt`, `LtEq`): reject `NullLiteral`, `ListLiteral`, `Unsupported`, and `OPAQUE`; unify via `_unify_types`; for `FLOAT64`, apply Case 0 fast path (`both is_known_non_nan`) and Cases A–D from Technical Plan Section 5 guarded by `MAX_DUPLICATED_OPERAND_NODES` and `remaining_budget`.
        *   Root check: reject `ListLiteral` or any known non-`BOOL` `bq_type` (degrading via `_degrade`).
        *   Default dispatch handler: fail closed via `_degrade(node.children())`.
    *   Log at `DEBUG` whenever a subtree degrades due to type incompatibility, missing column, or AST budget.

#### 21. `py-polars-bigquery/polars_bigquery/core/compiler/sql/string.py` (Modify)

*   **Type / Scope**: Compiler SQL Emitter
*   **Proposed Changes**: Register `@literal_ir_to_sql.register` for `BytesLiteral`, formatting every byte as lowercase hex escape (`b'...'` using `\x{b:02x}`, with empty bytes -> `b''`).

#### 22. `py-polars-bigquery/polars_bigquery/core/compiler/sql/temporal.py` (Modify)

*   **Type / Scope**: Compiler SQL Emitter
*   **Proposed Changes**: Register `@literal_ir_to_sql.register` for `TimeLiteral`, rejecting booleans and out-of-range integers (`not isinstance(micros, int) or isinstance(micros, bool) or not (0 <= micros < 86_400_000_000)`) with `ValueError`, and formatting as `TIME 'HH:MM:SS'` (or `TIME 'HH:MM:SS.ffffff'` when sub-second microseconds are non-zero).

#### 23. `py-polars-bigquery/polars_bigquery/core/compiler/sql/numeric.py` (Modify)

*   **Type / Scope**: Compiler SQL Emitter
*   **Proposed Changes**: Register `@literal_ir_to_sql.register` for `DecimalLiteral`, validating `node.value` against `re.fullmatch(r"[+-]?\d+(?:\.\d+)?", node.value)` (raising `ValueError` on `NaN`, `Inf`, scientific notation, or whitespace) and formatting as `NUMERIC '{node.value}'`.

#### 24. `py-polars-bigquery/polars_bigquery/core/compiler/__init__.py` (Modify)

*   **Type / Scope**: Compiler Pipeline Entrypoint
*   **Proposed Changes**:
    *   Export `typing` and `resolve_and_lower_types` in `__all__`.
    *   Define `MAX_ROW_RESTRICTION_BYTES = 1_000_000`.
    *   Update `_json_expr_to_row_restriction(expr_json, schema, bq_types=None)` to normalize `schema` via `pl.Schema(schema)`, run `json_to_ir -> resolve_and_lower_types -> rewrite_ir -> ir_to_sql`, and return `None` (with a `DEBUG` log) if `len(sql.encode("utf-8")) > MAX_ROW_RESTRICTION_BYTES`.
    *   Update `predicate_to_row_restriction(predicate: pl.Expr, *, schema: pl.Schema | Mapping[str, DataType | DataTypeClass], bq_types: Mapping[str, BQType] | None = None) -> str` to normalize `pl.Schema(schema)` before the `except (RecursionError, json.JSONDecodeError)` block.

#### 25. `py-polars-bigquery/polars_bigquery/_read_bigquery.py` (Modify — Phase 4 compiler callsite)

*   **Type / Scope**: Public Client & Polars I/O Plugin Module
*   **Proposed Changes**: Pass `schema=polars_schema, bq_types=bq_types` to `compiler.predicate_to_row_restriction(...)` inside `source_generator`.

#### 26. `py-polars-bigquery/tests/unit/core/test_predicates.py` (Modify)

*   **Type / Scope**: Unit Tests
*   **Proposed Changes**:
    *   Update existing `predicate_to_row_restriction` calls to pass `schema=`, update the 1-arg `_json_expr_to_row_restriction` monkeypatch stub at `test_predicates.py:496` to accept `(expr_json, schema, bq_types=None)`, and pass `schema={invalid_col: pl.Int64}` at `test_predicates.py:153-160`.
    *   Verify `drop_nulls_with_unsupported_comparison_sibling_pushes_down_is_not_null` (`test_predicates.py:3400`) remains `"(`a` IS NOT NULL)"` while invalid `DropNulls` (e.g., `DropNulls(Uppercase(int_col))`, `DropNulls(ListLiteral([1, 2]))`) produce `""`.
    *   Add parameterized tests for `FillNullWithStrategy` across all 14 `BQType`s × `"zero"`/`"one"` (including compound `"ms"`/`"us"`/`"ns"` datetimes, mixed time units, and unrepresentable `TIME`/`STRING`/`BYTES` `"one"`), plus `BytesLiteral`, `TimeLiteral`, and `DecimalLiteral` formatter validation errors.
    *   Add tests for `INT64` vs. `FLOAT64` vs. `NUMERIC` float predicates (positive, `~`, `.is_null()`, `.fill_null()`), integer bitwise `&`/`|`/`~` rejection, non-`BOOL` root rejection, `ListLiteral` isolation (`And`/`Or`/root/`IsNull`/`Eq`), `OPAQUE` allowlist (including single-operand `Coalesce` and `bq_types` override under Mode B `as_storage=True`), strict `STRING` operators, and the cross-type comparison/`Coalesce`/`Ternary`/`IsIn` matrix.
    *   Add tests for all 5 `FLOAT64` `NaN` comparison cases (Case 0 and Cases A–D across all 9 `{finite, NaN, NULL}` pairs, compound `is_known_non_nan`/`is_known_non_null` propagation), `ListLiteral` fanout weighting against `MAX_DUPLICATED_OPERAND_NODES`, `MAX_AST_NODES` iterative traversal without `RecursionError`, and `MAX_ROW_RESTRICTION_BYTES` 1 MB fallback.

#### 27. `py-polars-bigquery/tests/unit/test_read_bigquery.py` (Modify — Phase 4 compiler integration tests)

*   **Type / Scope**: Unit Tests
*   **Proposed Changes**: Add `scan_table` tests verifying that `bq_types` from `ExtractedSchema` is threaded into `predicate_to_row_restriction` on `schema=None` scans (so Mode B `as_storage=True` on a `JSON` column does not emit `STRING` SQL) and that explicit `schema=` derives compiler types from the supplied `pl.Schema`.

*   **Phase 4 Checkpoint**:
    *   `cargo test --workspace --exclude py-arrow-bigquery --exclude py-polars-expressions-testing --lib --bins --features testing`
    *   `uv run --project py-arrow-bigquery --group test pytest -n auto py-arrow-bigquery/tests/unit`
    *   `uv run --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --resolution lowest --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --project py-polars-bigquery --group test --with pytest-cov pytest --cov=polars_bigquery.core.types --cov=polars_bigquery.dtypes --cov=polars_bigquery.core.schema --cov=polars_bigquery.core.compiler.typing --cov-branch --cov-report=term-missing py-polars-bigquery/tests/unit`
    *   `uv run --project py-polars-bigquery --group dev ruff check --no-fix py-polars-bigquery`
    *   `uv run --project py-polars-bigquery --group dev ruff format --diff py-polars-bigquery`

---

### Phase 5: End-User Documentation and Live BigQuery Integration Tests

#### 28. `py-polars-bigquery/polars_bigquery/_read_bigquery.py` (Modify — docstrings)

*   **Type / Scope**: Public API Docstrings
*   **Proposed Changes**: Update `Client.scan_table` and `Client.read_table` docstrings: change `"utc"` -> `"UTC"`, document `UnsupportedTypeWarning` (`Warns`) and `ValueError` / `BigQueryError` (`Raises`), add explicit `schema=` examples using `polars_bigquery.dtypes` (noting Mode B exclusion for explicit `schema=`), document `_PARTITIONDATE` vs. `_PARTITIONTIME` granularity rules, explain `read_table()` vs. `scan_table()` behavior on unsupported types, and link to `data-types.md`.

#### 29. `docs/source/api/data-types.md` (Create)

*   **Type / Scope**: End-User Documentation
*   **Proposed Changes**: Publish the BigQuery-to-Polars Data Types guide covering:
    *   14-type mapping table (scalar and `REPEATED`, including `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` outer wrapping, `"UTC"`, `Decimal(38, 9)` for parameterized `NUMERIC(P, S)`, and `_PARTITIONDATE`/`_PARTITIONTIME` granularity rules).
    *   Working with `JSON`/`GEOGRAPHY` via `.ext.storage()` and `polars_bigquery.dtypes` in explicit `schema=`.
    *   Extension registration contract: Mode A (`BaseExtension` subclass), Mode B (`as_storage=True` / `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`) supported on `scan_table(schema=None)` and explicitly excluded with explicit `schema=` for ambiguous storage types.
    *   Unsupported BigQuery types (`BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, unnamed/duplicate/empty `STRUCT`), `UnsupportedTypeWarning`, `client.read_query()` SQL workarounds, and `read_table()` vs. `scan_table()` differences (including the Rust `BigQueryError` on `INTERVAL` citing Polars issues `#30920` and `#30921`).

#### 30. `mkdocs.yml` (Modify)

*   **Type / Scope**: Documentation Navigation Configuration
*   **Proposed Changes**: Add `Data types: api/data-types.md` under `nav -> API reference`.

#### 31. `docs/source/api/index.md` (Modify)

*   **Type / Scope**: API Reference Index
*   **Proposed Changes**: Add a link to `data-types.md` and register `::: polars_bigquery.dtypes` and `::: polars_bigquery.exceptions.UnsupportedTypeWarning`.

#### 32. `docs/source/api/predicate-pushdown.md` (Modify)

*   **Type / Scope**: Predicate Pushdown Documentation
*   **Proposed Changes**:
    *   Replace the existing inline schema mapping table with a link to `data-types.md`.
    *   Update `fill_null(strategy="zero"|"one")`, `FLOAT64` `NaN`, and `INT64` `is_nan`/`is_inf` documentation.
    *   Add a per-type pushdown support matrix for all 14 supported types.
    *   Add a "Common pitfalls that disable pushdown" section (`NUMERIC` vs. `FLOAT64`, cross-temporal mixes, naive vs. UTC `_PARTITIONTIME`, bitwise `&`/`|`/`~`, and `OPAQUE` types).

#### 33. `py-polars-bigquery/tests/system/test_read_bigquery_integration.py` (Modify)

*   **Type / Scope**: Live BigQuery Integration / Differential System Tests
*   **Proposed Changes**:
    *   Add a module-scoped pytest fixture provisioning an ephemeral dataset (`polars_bq_test_{uuid4().hex}`, location `"US"`, `defaultTableExpirationMs=3_600_000`) via BigQuery REST (`client._rest_client.session` + `polars_bigquery.core.bigquery_rest._get_headers(...)`) without adding `google-cloud-bigquery` as a dependency: use REST `POST .../datasets` (`datasets.insert`), `POST .../tables` (`tables.insert` for 0-column and ingestion-time partitioned tables), `client.read_query(...)` for DDL/DML population, and `DELETE .../datasets/{dataset}?deleteContents=true` in `finally:`.
    *   Add live integration tests covering:
        *   Mixed table containing supported types plus `BIGNUMERIC`, `INTERVAL`, `RANGE<DATE>`, `TIMESTAMP(12)`, and `STRUCT<ok INT64, bad INTERVAL>`: assert `UnsupportedTypeWarning`, omitted schema columns, exact row count from `lf.select(pl.len()).collect()`, and expected column values from `lf.collect()`.
        *   Calling `client.read_table(...)` on an `INTERVAL`-containing table: assert `BigQueryError` referencing upstream Polars issues `#30920`/`#30921` (no process abort).
        *   Daily and hourly ingestion-time partitioned tables whose only physical column is `INTERVAL`: assert daily retains `_PARTITIONDATE` + `_PARTITIONTIME`, hourly retains only `_PARTITIONTIME`, and partition row counts succeed on both without reading `INTERVAL`.
        *   0-column table: assert `scan_table().collect()` returns a 0-row, 0-column `DataFrame`.
        *   Schema parity (`scan_table().collect_schema() == read_table().schema`) on `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY<JSON>`, `ARRAY<GEOGRAPHY>`, and `STRUCT<j JSON, g GEOGRAPHY>`.
        *   Differential pushdown parity (`scan_table().filter(expr).collect()` vs. `read_table().filter(expr)`) across all 14 supported types × operators, including `sql/temporal.py`'s emitted `IFNULL` expressions (`DATE(TIMESTAMP_SECONDS(...))`, `DATETIME(TIMESTAMP_MICROS(...))`, `TIMESTAMP_MICROS(...)`), `FLOAT64` `NaN`/`±Inf`/`NULL` comparisons (all 6 ops, positive, negated, and compound), `INT64` float predicates (with `NULL` and `~`), `fill_null(strategy="zero"|"one")` across types, bitwise `INT64`, `INT64(2^53+1)` vs. `FLOAT64(2^53)`, `NUMERIC(2^53+1)` vs. `FLOAT64(2^53)`, `OPAQUE` `IS [NOT] NULL` / `.ext.storage()`, and Mode B `as_storage=True` on `scan_table(schema=None)`.

---

### Verification Plan

*   **Dependency Lock & Floor Check**:
    *   `uv sync --project py-polars-bigquery --group dev`
    *   `uv run --project py-polars-bigquery python -c "import polars; assert polars.__version__.startswith('2.')"`
*   **Unit Tests**:
    *   `cargo test --workspace --exclude py-arrow-bigquery --exclude py-polars-expressions-testing --lib --bins --features testing` (excludes PyO3 Python extension cdylib crates that link against `libpython` during `cargo test`, enabling `testing` for Rust test helpers)
    *   `uv run --project py-arrow-bigquery --group test pytest -n auto py-arrow-bigquery/tests/unit`
    *   `uv run --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --resolution lowest --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit` (matches CI lowest-resolution check on `polars==2.0.0`)
*   **Branch Coverage Gate**:
    *   `uv run --project py-polars-bigquery --group test --with pytest-cov pytest --cov=polars_bigquery.core.types --cov=polars_bigquery.dtypes --cov=polars_bigquery.core.schema --cov=polars_bigquery.core.compiler.typing --cov-branch --cov-report=term-missing py-polars-bigquery/tests/unit`
    *   Target: 100% branch coverage on `polars_bigquery.core.types`, `polars_bigquery.dtypes`, `polars_bigquery.core.schema`, and `polars_bigquery.core.compiler.typing`.
    *   Qualitative matrix requirements: all 14 `BQType`s × operators, all 9 `{finite, NaN, NULL}` `FLOAT64` comparison states, `MonthDayNano` Rust decoder tests, and non-mocked Arrow C stream FFI tests.
*   **Linting, Formatting & Docs Build**:
    *   `uv run --project py-polars-bigquery --group dev ruff check --no-fix py-polars-bigquery docs/source/`
    *   `uv run --project py-polars-bigquery --group dev ruff format --diff py-polars-bigquery docs/source/`
    *   `uv run --with-requirements docs/source/requirements.txt mkdocs build --strict`
*   **E2E / Live Integration Tests**:
    *   Prerequisites: Google Cloud Application Default Credentials (`gcloud auth application-default login` or service account credentials), `GOOGLE_CLOUD_PROJECT` set, and BigQuery IAM roles permitting ephemeral dataset/table creation, query jobs, and Storage Read sessions (`roles/bigquery.dataEditor`, `roles/bigquery.jobUser`, `roles/bigquery.readSessionUser`).
    *   `uv run --project py-polars-bigquery --group test pytest py-polars-bigquery/tests/system`

--------------------------------------------------------------------------------

*Drafted with EGM*
