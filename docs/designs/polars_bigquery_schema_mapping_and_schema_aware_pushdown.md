# Schema Mapping Fixes and Schema-Aware Predicate Pushdown for `polars-bigquery`

*   **Author:** @tswast
*   **Status:** Proposed
*   **Date:** 2026-11-19

--------------------------------------------------------------------------------

## Overview

`polars_bigquery.Client.scan_table` lazily scans BigQuery tables through the BigQuery Storage Read API with column projection and row predicate pushdown. Today, two coupled defects prevent `scan_table` from working reliably across BigQuery's supported type system on Polars 2.0+:

1.  **Schema extraction and projection defects**: `polars_bigquery.core.schema` produces Polars schemas that disagree with BigQuery's Arrow IPC output (lowercase `"utc"` on `TIMESTAMP`, plain `pl.String` instead of Polars 2.0+ `pl.Extension` on `JSON`, `GEOGRAPHY`, `ARRAY<JSON>`, and `ARRAY<GEOGRAPHY>`), drops type parameters during `REPEATED` recursion, crashes on tables containing unsupported or deferred columns (`BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, or unnamed/duplicate `STRUCT` subfields), and sends empty `selected_fields=[]` on zero-column projections (causing BigQuery to stream every column in the table, including unsupported `INTERVAL` columns that trigger upstream Polars/`polars-arrow` panics).
2.  **Schema-unaware predicate pushdown**: `polars_bigquery.core.compiler` compiles Polars filter expressions into BigQuery Storage Read API `row_restriction` SQL without knowing column data types. Valid Polars filters therefore either fail inside BigQuery with `400 INVALID_ARGUMENT` errors or silently drop rows because GoogleSQL evaluates `FLOAT64` `NaN` comparisons under IEEE-754 while Polars uses total ordering (and BigQuery compares `NUMERIC` vs `FLOAT64` exactly while Polars casts `Decimal` to lossy `Float64`).

This design fixes schema mapping, projection safety, and schema-aware predicate pushdown across all 14 currently supported BigQuery types on Polars 2.0+ (verified against Polars `2.1.0`), introduces a public `polars_bigquery.dtypes` module and `UnsupportedTypeWarning`, and cleanly omits unsupported or deferred columns so mixed tables remain scannable.

## The Problem

Users adopting `Client.scan_table` hit four classes of real-world failures even when querying standard BigQuery tables:

*   **Broken scans on tables with `JSON`, `GEOGRAPHY`, or `TIMESTAMP` columns**: BigQuery's Arrow IPC stream encodes `TIMESTAMP` with uppercase `"UTC"`, `JSON` with `ARROW:extension:name = "google:sqlType:json"`, and `GEOGRAPHY` with `ARROW:extension:name = "google:sqlType:geography"` and `ARROW:extension:metadata = '{"encoding": "WKT"}'` (and for `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>`, attaches the extension metadata to the outer `List` field rather than the inner `String` element). Polars 2.0+ loads unknown Arrow extension types as `pl.Extension(...)` by default. Because `core/schema.py` maps `JSON` and `GEOGRAPHY` to `pl.String` and `TIMESTAMP` to `time_zone="utc"`, `scan_table().collect()` fails Polars schema validation on `JSON`/`GEOGRAPHY` tables and disagrees with `read_table().schema` on `TIMESTAMP` tables.
*   **Unscannable mixed tables and process aborts**: Enterprise BigQuery tables can include a `BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, or irregular `STRUCT` column alongside ordinary scalar columns. Today `scan_table()` crashes at construction time (`Decimal(76, 38)` fails `register_io_source`, `RANGE<T>` raises `TypeError`) or crashes Python when an `INTERVAL` (`MonthDayNano`) column crosses Arrow FFI (`polars 2.1.0` panics with `PanicException` on schema import, and `polars-arrow 0.53.0` aborts with a non-unwinding `SIGABRT` on FFI export). Even when a caller only asks for row count (`lf.select(pl.len())`), passing `selected_fields=[]` instructs BigQuery Storage Read API to read every column in the table—causing full-table billing and pulling omitted `INTERVAL` columns.
*   **BigQuery SQL errors on valid Polars filters**: Because the predicate compiler ignores column types, valid Polars predicates crash `CreateReadSession` with GoogleSQL type errors—including `IS_NAN`/`IS_INF` on `INT64` columns, `fill_null(strategy="zero"|"one")` emitting `IFNULL(col, 0)` on `BOOL`/`STRING`/`BYTES`/temporal columns, integer bitwise `&`/`|`/`~` emitted as logical `AND`/`OR`/`NOT`, and comparisons or string functions pushed onto `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`, or mismatched cross-type operands (`DATE`/`DATETIME` vs `TIMESTAMP`).
*   **Silent data loss on `FLOAT64` `NaN` and `NUMERIC` vs `FLOAT64` comparisons**: Polars orders `NaN` greater than all other floats (`NaN == NaN`, `NaN > 5.0`, `NaN >= 5.0`, and `1.0 < NaN` all evaluate to `True`), whereas BigQuery evaluates IEEE-754 `NaN` comparisons as `FALSE`. Similarly, Polars `2.1.0` casts `Decimal(38, 9)` to lossy `Float64` when comparing against `Float64` (`Decimal('9007199254740993.0') == 9007199254740992.0` is `True`), whereas BigQuery compares `NUMERIC` against `FLOAT64` exactly (`FALSE`). Both violate the predicate pushdown superset invariant and silently drop matching rows before Polars can re-filter them.

The proposed solution aligns `polars-bigquery` with Polars 2.0+ (`pl.Extension` including `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` outer wrapping, and uppercase `"UTC"`), isolates unsupported/deferred types via whole-column omission with actionable warnings and safe projection, and adds an iterative pre-rewrite schema-aware typing and lowering stage to `core/compiler/` so pushed predicates are both valid GoogleSQL and semantically faithful to Polars.

## Goals

*   **Polars 2.0+ alignment**: Drop Polars 1.x support in `py-polars-bigquery/pyproject.toml` (`polars >= 2.0.0, < 3.0.0`) and expose Arrow-matching Polars 2.0+ schemas for all 14 supported BigQuery types (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`) plus ingestion-time pseudo-columns (`_PARTITIONDATE`, `_PARTITIONTIME`).
*   **First-class extension dtype ergonomics**: Publish a `polars_bigquery.dtypes` module exporting `JSON` and `GEOGRAPHY` (and register it in the API reference) alongside `UnsupportedTypeWarning` in `polars_bigquery.exceptions` so callers never hand-write Arrow extension metadata strings and can filter or promote skip warnings programmatically.
*   **Safe handling of unsupported and deferred columns**: Omit top-level columns whose type (or any nested descendant inside `STRUCT`/`ARRAY`) is unsupported (`BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, unnamed or duplicate `STRUCT` subfields, or unknown types) with a concise, actionable warning; preserve `_PARTITIONDATE`/`_PARTITIONTIME` on ingestion-time partitioned tables; fail early when a non-empty BigQuery table has zero supported columns remaining after pseudo-columns; keep genuinely 0-column tables valid by short-circuiting empty schemas without calling `CreateReadSession`.
*   **Safe, cost-conscious zero-column projections**: Guarantee that Storage Read API never receives `selected_fields=[]` on a non-empty schema, pick the smallest fixed-width supported column when Polars requests zero columns, and preserve batch row height without collapsing to zero rows.
*   **Zero GoogleSQL type errors and zero silent data loss in predicate pushdown**: Make `predicate_to_row_restriction` schema-aware across all 14 supported BigQuery types so every emitted `row_restriction` is valid GoogleSQL and returns a superset of the rows matched by Polars—including exact typed rewrites for representable `fill_null(strategy="zero"|"one")` types, null-safe 3-valued `INT64` `is_nan`/`is_inf` rewrites, Polars-aligned `FLOAT64` `NaN` comparisons (with AST-duplication bounds), bitwise integer rejection, `OPAQUE` null-check-only handling (`JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`), and strict cross-type rules (`INT64` vs `FLOAT64` and `INT64` vs `NUMERIC` allowed; `NUMERIC` vs `FLOAT64` and all cross-temporal mixes such as `DATE`/`DATETIME`/`TIMESTAMP`/`TIME` skipped) while preserving partial pushdown.
*   **Actionable explicit `schema=` diagnostics and observability**: Detect common explicit `schema=` mismatches (`pl.String` vs `JSON`/`GEOGRAPHY` extension types, lowercase `"utc"`, parameterized `Decimal`) with copy-pasteable fixes, and emit `DEBUG` logs for `selected_fields`, compiled `row_restriction`, and type-driven pushdown fallbacks.
*   **End-user data type and pushdown documentation**: Publish a dedicated BigQuery-to-Polars Data Types reference page (`docs/source/api/data-types.md`, wired into `mkdocs.yml`) documenting the complete supported type mapping (including `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>`, `TIMESTAMP` `"UTC"`, and ingestion-time pseudo-columns), skipped/unsupported types with `read_query()` workarounds, `.ext.storage()` usage, `polars_bigquery.dtypes` explicit `schema=` examples, and `read_table()` vs `scan_table()` differences—while updating `docs/source/api/predicate-pushdown.md` with the per-type operator matrix.
*   **End-to-end verification**: Validate schema mapping and pushdown parity using real Arrow C stream unit tests and live BigQuery Storage Read API differential tests.

## Non-Goals

*   **`scan_query` integration**: Query-backed lazy scanning and standard SQL `WHERE` pushdown are tracked separately.
*   **Full support for deferred BigQuery types**: Adding Arrow/Polars representations for `BIGNUMERIC`, parameterized `NUMERIC(P, S)` Arrow serialization changes, picosecond `TIMESTAMP`, `INTERVAL`, `RANGE<T>`, or unnamed/duplicate `STRUCT` subfields is out of scope.
*   **Partial `STRUCT` subfield exclusion**: Projecting only supported subfields out of a `STRUCT` containing an unsupported descendant (which requires client-side `STRUCT` reconstruction for Storage Read API compatibility) is deferred as future work.
*   **User-facing `BYTES`, `TIME`, and `Decimal` literal parsing**: Parsing user literals or `is_in` collections of `pl.Binary`, `pl.Time`, or `pl.Decimal` is deferred as future work; internal literal IR nodes are added only where needed for `fill_null(strategy="zero"|"one")` lowering.
*   **Third-party extension packages**: Integrating GeoPolars or custom JSON/WKT namespaces is out of scope beyond respecting extensions already registered via `pl.get_extension_type(...)`.
*   **Backward compatibility for legacy broken mappings**: Retaining lowercase `"utc"`, plain `pl.String` for `JSON`/`GEOGRAPHY`, or `TypeError` on a single unsupported column is not required.
*   **Upstream Polars / `polars-arrow` `MonthDayNano` crashes in `read_table()` / `read_query()`**: Resolving the upstream `polars` `2.1.0` `PanicException`/`SIGSEGV` and `polars-arrow 0.53.0` FFI export `SIGABRT` (tracked via upstream bug reports and a separate `crates/arrow-bigquery` guard) is out of scope; this design shields `scan_table(schema=None)` and documents the residual surface.

## Open Questions

None. All five technical questions were tested live against the BigQuery Storage Read API and Polars `2.1.0`:

*   **Resolved — `ARRAY`/`STRUCT` `IS [NOT] NULL` in `row_restriction`**: Live-tested on `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`. Storage Read API accepts `IS NULL` and `IS NOT NULL` on both `STRUCT` (`totals`) and `ARRAY<STRUCT>` (`hits`), rejects `hits = hits` (`Equality is not defined`), and evaluates `totals = totals` to `0` rows when subfields contain `NULL`. Confirms the null-check-only allowlist for `ARRAY` and `STRUCT`.
*   **Resolved — Pseudo-column-only `selected_fields=["_PARTITIONDATE"]`**: Live-tested on an ingestion-time partitioned table whose only physical column was `INTERVAL`. `read_table(selected_fields=["_PARTITIONDATE"])` and `selected_fields=["_PARTITIONTIME"]` both succeeded without transmitting the `INTERVAL` column, returning `pl.Date` / `pl.Datetime("us", "UTC")` and the exact row count (`height=2`). Confirms retaining `_PARTITIONDATE`/`_PARTITIONTIME` after physical omission.
*   **Resolved — Nested `ARRAY<JSON>` and `STRUCT<g GEOGRAPHY>` in Polars `2.1.0`**: Live-tested via `pl.DataFrame` and `pl.scan_arrow_c_stream`. BigQuery attaches `ARROW:extension:name` to the outer `List` field for `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>` (both top-level and inside a `STRUCT`), so Polars `2.1.0` produces `pl.Extension("google:sqlType:json", pl.List(pl.String))` and `pl.Extension("google:sqlType:geography", pl.List(pl.String), '{"encoding": "WKT"}')`—NOT `pl.List(pl.Extension(...))`—while `STRUCT<j JSON, g GEOGRAPHY>` and `ARRAY<STRUCT<j JSON, g GEOGRAPHY>>` keep `pl.Extension(..., pl.String)` on the struct fields.
*   **Resolved — GoogleSQL `NaN`, `INT64`/`NUMERIC` vs `FLOAT64`, temporal cross-types, and typed `IFNULL` literals**: Live-tested in Storage Read API `row_restriction` against Polars `2.1.0`:
    *   `CAST('nan' AS FLOAT64) != 5.0` evaluates to `TRUE` (matches Polars); `NaN != NaN` evaluates to `TRUE` in BigQuery vs `FALSE` in Polars (confirming `((a != b) AND NOT (IS_NAN(a) AND IS_NAN(b)))` for 3-valued exactness).
    *   `INT64(2^53+1) = FLOAT64(2^53)` is `TRUE` and `>` is `FALSE` in both BigQuery and Polars `2.1.0` (matching semantics).
    *   `NUMERIC(2^53+1) = FLOAT64(2^53)` is `FALSE` (`>` is `TRUE`) in BigQuery, whereas Polars `2.1.0` evaluates `Decimal == Float64` to `TRUE` (`>` to `FALSE`)—confirming `NUMERIC` vs `FLOAT64` pushdown MUST be skipped to prevent silent data loss. `NUMERIC` vs `INT64` is exact in both.
    *   `DATE = TIMESTAMP` and `DATETIME = TIMESTAMP` fail in BigQuery with `INVALID_ARGUMENT`; strict same-type temporal matching (`DATE` only with `DATE`, `DATETIME` only with `DATETIME`, `TIMESTAMP` only with `TIMESTAMP`, `TIME` only with `TIME`) is enforced across all temporal operators.
    *   All six typed `IFNULL` literals (`TIME '00:00:00'`, `b''`, `NUMERIC '0'`, `DATE '1970-01-01'`, `DATETIME '1970-01-01 00:00:00.000001'`, `TIMESTAMP_MICROS(1)`) succeeded in Storage Read API `row_restriction`.
*   **Resolved — Genuinely 0-column BigQuery table**: Live-tested via `tables.insert` with an empty schema. `tables.get` returns `{}` (no `schema` field), and Storage Read API `CreateReadSession` fails with `FAILED_PRECONDITION: Table or view (...) has empty schema, cannot create read session.` Short-circuiting empty schemas in `scan_table` cleanly returns a 0-row, 0-column `DataFrame` without failing or invoking `CreateReadSession`.

## Critical User Journeys (CUJs)

### CUJ 1: Scanning a Mixed Table Containing Deferred or Unsupported Columns

*   **Given**: A BigQuery table has supported columns (`id INT64`, `name STRING`) alongside unsupported or deferred columns (`amount BIGNUMERIC`, `dur INTERVAL`, `window RANGE<DATE>`, `ts_pico TIMESTAMP(12)`, and `meta STRUCT<ok INT64, bad INTERVAL>`).
*   **When**: A user calls `lf = client.scan_table(table)` and executes `lf.filter(pl.col("id") > 10).collect()` or `lf.select(pl.len()).collect()`.
*   **Then**: `scan_table()` emits a single `UnsupportedTypeWarning` listing the skipped top-level columns with their exact reasons and nested paths (capped with `"... and N more"` plus a `client.read_query()` workaround), exposes only `id` and `name` in `lf.collect_schema()`, requests only supported `selected_fields` from BigQuery (selecting `id` rather than `[]` for `pl.len()`), preserves exact row count on `pl.len()`, and completes without `InvalidOperationError`, `TypeError`, or upstream `MonthDayNano` crashes.

### CUJ 2: Scanning and Filtering Tables with `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, and `STRUCT`

*   **Given**: A BigQuery table contains `ts TIMESTAMP`, `payload JSON`, `geom GEOGRAPHY`, `json_list ARRAY<JSON>`, `tags ARRAY<STRING>`, and `info STRUCT<k STRING, v INT64, j JSON>` columns, and the caller runs on Polars 2.0+.
*   **When**: The user calls `lf = client.scan_table(table)` and runs `lf.collect()`, `lf.filter(pl.col("payload").is_not_null()).collect()`, and `lf.filter(pl.col("payload").ext.storage().str.starts_with("{")).collect()`.
*   **Then**: `lf.collect_schema()` matches `client.read_table(table).schema` (`pl.Datetime("us", "UTC")`, `polars_bigquery.dtypes.JSON`, `polars_bigquery.dtypes.GEOGRAPHY`, and `pl.Extension("google:sqlType:json", pl.List(pl.String))` for `ARRAY<JSON>`), `is_not_null()` pushes down to BigQuery `row_restriction`, `.ext.storage().str.starts_with("{")` falls back cleanly to Polars without sending invalid SQL to BigQuery, and all queries return exact results.

### CUJ 3: Schema-Aware Filtering with Typed `fill_null`, Float Checks, Bitwise Ops, and `FLOAT64` `NaN`

*   **Given**: A BigQuery table contains `flag BOOL`, `count INT64`, `score FLOAT64` (including `NaN`, `±Inf`, and `NULL`), `label STRING`, `t TIME`, and `price NUMERIC`.
*   **When**: The user applies filters including `pl.col("flag").fill_null(strategy="zero")`, `pl.col("label").fill_null(strategy="zero") == ""`, `pl.col("t").fill_null(strategy="one") > ...`, `~pl.col("count").is_nan()`, `(pl.col("count") & 4) == 4`, `pl.col("price") == 1.5`, and `pl.col("score") > 5.0`.
*   **Then**: Representable expressions compile to exact, Polars-aligned GoogleSQL (`IFNULL(flag, FALSE)`, `(IFNULL(label, '') = '')`, null-safe 3-valued `INT64` check, `((score > 5.0) OR IS_NAN(score))`), unrepresentable or divergent expressions (`TIME` `strategy="one"`, integer bitwise `&`, `NUMERIC == FLOAT64`) degrade cleanly to client-side filtering without breaking sibling `&` partial pushdown, zero `400 INVALID_ARGUMENT` errors occur, and collected rows match eager `read_table().filter(...)` identically.

### CUJ 4: Scanning with an Explicit User-Supplied `schema=`

*   **Given**: A user wants to skip the `tables.get` REST call and calls `client.scan_table(table, schema=...)` on a table with `TIMESTAMP`, `JSON`, and `GEOGRAPHY` columns.
*   **When**: The user supplies `schema={"ts": pl.Datetime("us", "UTC"), "j": polars_bigquery.dtypes.JSON, "g": polars_bigquery.dtypes.GEOGRAPHY}`—or mistakenly passes `schema={"j": pl.String}`.
*   **Then**: With `polars_bigquery.dtypes` and `"UTC"`, `scan_table` skips `tables.get`, pushes valid predicates using the supplied schema, and collects cleanly; with plain `pl.String` for a `JSON` column, collecting produces an actionable error directing the user to `polars_bigquery.dtypes.JSON`.

### CUJ 5: All-Unsupported Tables (Partitioned vs. Non-Partitioned) and 0-Column Tables

*   **Given**: Three tables: (A) a non-partitioned table with only an `INTERVAL` column, (B) an ingestion-time partitioned table whose only physical column is `INTERVAL`, and (C) a genuinely 0-column BigQuery table.
*   **When**: A user calls `client.scan_table(...)` on each table.
*   **Then**: Table (A) fails early at `scan_table()` with an error listing the omitted column, reason, and `read_query()` workaround (without emitting a redundant warning beforehand); Table (B) emits `UnsupportedTypeWarning` for the omitted physical column and retains `_PARTITIONDATE` and `_PARTITIONTIME` so partition row counts work via `selected_fields=["_PARTITIONDATE"]`; Table (C) returns an empty-schema `LazyFrame` and collects an empty `DataFrame` without invoking `CreateReadSession`.

## Technical Plan (Architecture)

The design splits into four cooperating layers sharing a single canonical BigQuery type model (`core/types.py`):

```text
                       +----------------------------+
                       |   Client.scan_table(...)   |
                       +-------------+--------------+
                                     |
                   schema is None    |    explicit schema=...
                   +-----------------+-----------------+
                   v                                   v
         +-------------------+               +-------------------+
         | BigQuery REST API |               | pl.Schema(schema) |
         |   (tables.get)    |               | (normalize dtypes)|
         +---------+---------+               +---------+---------+
                   |                                   |
                   v                                   |
    +--------------+----------------+                  |
    | core/schema.py                |                  |
    | extract_schema_metadata()     |<-- core/types.py |
    | - "UTC", {**field} REPEATED   |   (BQType enum,  |
    | - Extension(JSON / GEOGRAPHY) |    extension     |
    | - Extension(List(String)) for |    constructors, |
    |   ARRAY<JSON>, ARRAY<GEOG>    |    Polars->BQ    |
    | - Whole-col skip + paths      |    classifier)   |
    | - Retain _PARTITIONDATE/TIME  |                  |
    +--------------+----------------+                  |
                   |                                   |
      warn skipped / fail if non-empty                 |
      table has 0 supported columns                    |
                   +-----------------+-----------------+
                                     |
                                     v
                     +---------------+---------------+
                     | register_io_source(polars_    |
                     |   schema, source_generator)   |
                     +---------------+---------------+
                                     |  (on .collect() / .collect_batches())
                                     v
             +-----------------------+-----------------------+
             | empty polars_schema? -> yield empty DataFrame |
             +-----------------------+-----------------------+
                                     | (non-empty)
                                     v
      +------------------------------+------------------------------+
      | core/compiler/ (4-stage iterative pipeline)                 |
      | 1. json_to_ir(expr_json)            [schema-blind parser]   |
      | 2. resolve_and_lower_types(ir, bq)  [types, rewrites, NaN]  |
      | 3. rewrite_ir(ir)                   [boolean / De Morgan]   |
      | 4. ir_to_sql(ir)                    [mechanical formatter]  |
      +------------------------------+------------------------------+
                                     |
                                     v
      +------------------------------+------------------------------+
      | selected_fields safeguard (cheapest scalar if empty)        |
      | -> arrow_client.read_table(selected_fields, row_restriction)|
      | -> pl.scan_arrow_c_stream()                                 |
      | -> explicit schema= upfront collect_schema() validation     |
      | -> filter(predicate), limit(n_rows)                         |
      | -> if columns == []: yield DataFrame(height=batch.height)   |
      |    else: select(columns).collect_batches(chunk_size=...)    |
      +-------------------------------------------------------------+
```

### 1. Dependency Floor, Public `dtypes`, Warning Class, and Canonical Type Layer

*   **Polars 2.0+ dependency floor (`py-polars-bigquery/pyproject.toml`)**: Raise the dependency pin from `polars >= 1.43.0, < 3.0.0` to `polars >= 2.0.0, < 3.0.0` (verified on installed Polars `2.1.0`). This eliminates Polars 1.x `load_as_storage` vs. Polars 2.0 `load_as_extension` version skew.
*   **Public `polars_bigquery.dtypes` module (`polars_bigquery/dtypes.py`)**: Export first-class BigQuery extension dtype definitions and add `::: polars_bigquery.dtypes` to `docs/source/api/index.md` (using standard Polars `pl.List(...)` for ordinary arrays and direct `pl.Extension(..., pl.List(pl.String()), ...)` when explicit `schema=` declares `ARRAY<JSON>` or `ARRAY<GEOGRAPHY>`):
    *   `JSON`: `pl.Extension("google:sqlType:json", pl.String())`
    *   `GEOGRAPHY`: `pl.Extension("google:sqlType:geography", pl.String(), '{"encoding": "WKT"}')`
*   **Public warning subclass (`polars_bigquery/exceptions.py`)**: Export `UnsupportedTypeWarning(UserWarning)` and document it alongside `BigQueryError` in `docs/source/api/index.md` so users can filter or promote skip warnings (`warnings.filterwarnings("error", category=UnsupportedTypeWarning)`).
*   **Canonical internal type model (`polars_bigquery/core/types.py`)**: Single source of truth used by `dtypes.py`, `core/schema.py`, `_read_bigquery.py`, and `core/compiler/`:
    *   Closed `BQType(enum.Enum)` with all 14 supported BigQuery types (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`) plus `UNKNOWN`.
    *   Extension constants: `JSON_EXTENSION_NAME = "google:sqlType:json"`, `GEOGRAPHY_EXTENSION_NAME = "google:sqlType:geography"`, `GEOGRAPHY_EXTENSION_METADATA = '{"encoding": "WKT"}'`.
    *   `build_extension_dtype(name, storage, metadata=None)`: checks `pl.get_extension_type(name)` first. If a user registered a custom `BaseExtension` subclass (such as GeoPolars), instantiates it via `ext_cls.ext_from_params(name, storage, metadata)`; if registered with `as_storage=True` (`"storage"`), returns `storage`; otherwise returns `pl.Extension(name, storage, metadata)`.
    *   `polars_dtype_to_bq_type(dtype) -> BQType`: normalizes uninstantiated `DataTypeClass` objects (`pl.Int64`) and classifies any Polars dtype:
        *   `BaseExtension` -> checks `ext_name()` (`"google:sqlType:json"` -> `ARRAY` if `isinstance(ext_storage(), pl.List)` else `JSON`; `"google:sqlType:geography"` -> `ARRAY` if `isinstance(ext_storage(), pl.List)` else `GEOGRAPHY`; any other extension -> `UNKNOWN`).
        *   `Boolean` -> `BOOL`; signed/unsigned ints (`Int8`..`Int128`, `UInt8`..`UInt64`) -> `INT64`; `Float32`/`Float64` -> `FLOAT64`; `Decimal(precision <= 38, scale <= 9)` -> `NUMERIC` (other `Decimal` -> `UNKNOWN`); `String` -> `STRING`; `Binary` -> `BYTES`; `Date` -> `DATE`; `Datetime(time_zone=None)` -> `DATETIME`; `Datetime(time_zone!=None)` -> `TIMESTAMP` (accepting any tz string such as `"UTC"` or `"utc"` in the compiler); `Time` -> `TIME`; `List` -> `ARRAY`; `Struct` -> `STRUCT`; everything else (`Duration`, `Categorical`, `Array`, `Null`, `Object`) -> `UNKNOWN`.
    *   `polars_datetime_time_unit(dtype) -> Literal["ms", "us", "ns"] | None`: preserves the column's Polars time unit so `DATETIME`/`TIMESTAMP` `fill_null(strategy="one")` respects `"ms"` vs. `"us"` and skips `"ns"`.

### 2. Schema Extraction and Whole-Column Omission (`core/schema.py`)

*   **Exact BigQuery-to-Polars schema mapping** (published directly for end users in the new `docs/source/api/data-types.md` page; see Section 5 below):

| BigQuery Type (`REST`) | Mode | Polars `2.0+` DataType |
| --- | --- | --- |
| `BOOL` / `BOOLEAN` | scalar | `pl.Boolean()` |
| `INT64` / `INTEGER` | scalar | `pl.Int64()` |
| `FLOAT64` / `FLOAT` | scalar | `pl.Float64()` |
| `NUMERIC` / `DECIMAL` | scalar | `pl.Decimal(precision=38, scale=9)` |
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
| `RECORD` / `STRUCT` (all subfields named, unique, supported) | scalar | `pl.Struct([...])` |
| Any other supported type | `REPEATED` | `pl.List(_extract_data_type({**field, "mode": "NULLABLE"}))` |
| `_PARTITIONDATE` (ingestion-time partitioned) | pseudo | `pl.Date()` |
| `_PARTITIONTIME` (ingestion-time partitioned) | pseudo | `pl.Datetime(time_unit="us", time_zone="UTC")` |

*   **Metadata hardening and `REPEATED` handling**:
    *   Read string fields null-safely: `(field.get("mode") or "").casefold()`, `(field.get("type") or "").casefold()`.
    *   Check `JSON` and `GEOGRAPHY` with `mode == "repeated"` directly so the `Extension` wraps `pl.List(pl.String())` on the outside (matching the live Polars `2.1.0` Arrow layout both top-level and inside `STRUCT`s).
    *   For all other `REPEATED` fields, recurse with `{**field, "mode": "NULLABLE"}` so `timestampPrecision`, `precision`, `scale`, `rangeElementType`, and future parameters are never dropped.
    *   Allowlist `TIMESTAMP` precision: accept only `precision is None or (not isinstance(precision, bool) and precision in (6, "6"))`; reject `"12"`, `12`, or any other value as unsupported (`TIMESTAMP(precision=...)`).
    *   For `RECORD`/`STRUCT`: reject subfields with missing/empty `"name"` (`unnamed STRUCT subfield`), reject duplicate subfield names within the same struct (`duplicate STRUCT subfield name`), and reject empty structs with no `"fields"` (cannot construct valid named fields).
*   **Whole-column omission via structured return**:
    *   `_extract_data_type(field, path)` raises an internal `_UnsupportedBigQueryType(reason, path)` on any unsupported type (`BIGNUMERIC`/`BIGDECIMAL`, `INTERVAL`, `RANGE`, non-microsecond `TIMESTAMP`, unnamed/duplicate `STRUCT` subfields, unknown types). Because nested `STRUCT` and `REPEATED` calls do not catch `_UnsupportedBigQueryType`, any unsupported descendant automatically omits the **entire enclosing top-level column** while recording the exact nested path (for example, `meta.dur (INTERVAL)`).
    *   Introduce frozen dataclasses `SkippedColumn(name, reason, path)` and `ExtractedSchema(polars_schema, physical_columns, pseudo_columns, skipped_columns, total_table_fields)`.
    *   `extract_schema_metadata(table_metadata: dict) -> ExtractedSchema` stays pure (no side-effect warnings or exceptions): it iterates `table_metadata.get("schema", {}).get("fields", [])`, collects supported physical columns and `SkippedColumn` records, and appends `_PARTITIONDATE` and `_PARTITIONTIME` when `timePartitioning` exists with `field is None`.
    *   Keep `extract_polars_schema(table_metadata: dict) -> pl.Schema` as a thin convenience wrapper returning `extract_schema_metadata(table_metadata).polars_schema`.

### 3. Projection Safety, Height Preservation, and Explicit `schema=` Diagnostics (`_read_bigquery.py`)

*   **Early error vs. warning policy in `Client.scan_table`**:
    *   When `schema is None`: call `extracted = extract_schema_metadata(table_metadata)`.
        *   **Zero supported columns remaining (after pseudo-columns)**: if `extracted.total_table_fields > 0 and len(extracted.polars_schema) == 0`, raise `ValueError` immediately (no redundant warning beforehand) listing up to 5 skipped columns with their exact reasons/paths, `"... and N more"` if truncated, and advising `client.read_query()` with SQL `CAST` or column selection.
        *   **Partial omission (including ingestion-time partitioned tables that retain `_PARTITIONDATE`/`_PARTITIONTIME`)**: if `extracted.skipped_columns` is non-empty and `len(extracted.polars_schema) > 0`, emit `warnings.warn(..., UnsupportedTypeWarning, stacklevel=2)` with total skipped count, up to 5 `name (reason at path)` entries, `"... and N more"`, and the `client.read_query()` workaround.
    *   When `schema is not None`: normalize via `pl.Schema(schema)`. If any column (top-level or nested) declares `pl.Duration`, raise `ValueError` early explaining that BigQuery `INTERVAL` is not currently supported and passing `pl.Duration` would fail Arrow decoding.
*   **Empty-schema short-circuit (`len(polars_schema) == 0`)**:
    *   Covers genuinely 0-column BigQuery tables (live-verified: `CreateReadSession` fails with `FAILED_PRECONDITION`) and user-supplied `schema={}` (preventing `selected_fields=[]` from streaming an omitted `INTERVAL` column).
    *   Inside `source_generator`, immediately yield `pl.DataFrame(schema=polars_schema)` (0 rows, 0 columns) without calling `self._arrow_client.read_table(...)`.
*   **Lightweight non-empty `selected_fields` and zero-width height preservation**:
    *   Compute `selected_fields = list(dict.fromkeys([*columns, *predicate_cols]))`.
    *   If `not selected_fields` (possible when Polars passes `with_columns=[]` and `predicate=None` or a column-less predicate like `pl.lit(True)`): pick a single column from `polars_schema` with minimum estimated storage width using a deterministic rank: `Boolean` (1) < `Date` (2) < `Int64`/`Float64`/`Time`/`Datetime` (3) < `Decimal` (4) < `String`/`Binary`/`BaseExtension` (5) < `Struct`/`List` (6), breaking ties by schema order (and using `_PARTITIONDATE` when only pseudo-columns exist).
    *   After `lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)`, apply `predicate` (if not `None`) and `limit(n_rows)` (if not `None`).
    *   **Height preservation**: if `columns` is empty (`columns == []`), do **not** call `lazyframe.select([])` (live-verified on Polars `2.1.0` to return `[]` / 0 rows). Instead iterate `lazyframe.collect_batches(chunk_size=batch_size)` and yield `pl.DataFrame(height=batch.height)` for each batch so `lf.select(pl.len()).collect()` returns the exact row count.
*   **Explicit `schema=` upfront diagnostics**:
    *   When `schema is not None`, right after `lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)`, read `actual_schema = lazyframe.collect_schema()` (reads Arrow IPC stream header without consuming batches) and compare each requested column recursively against `polars_schema[col]`:
        *   Declared `pl.String` vs actual `BaseExtension("google:sqlType:json" | "google:sqlType:geography", pl.String)` -> raise `pl.exceptions.SchemaError` recommending `polars_bigquery.dtypes.JSON` or `polars_bigquery.dtypes.GEOGRAPHY`.
        *   Declared `pl.List(BaseExtension(...))` or `pl.List(pl.String)` vs actual `BaseExtension(..., pl.List(pl.String))` -> raise `pl.exceptions.SchemaError` showing the exact BigQuery outer-extension `pl.Extension("google:sqlType:json" | "google:sqlType:geography", pl.List(pl.String()), ...)` dtype.
        *   Declared `pl.Datetime(..., "utc")` vs actual `pl.Datetime("us", "UTC")` -> raise `pl.exceptions.SchemaError` recommending uppercase `time_zone="UTC"`.
        *   Declared `pl.Decimal(p, s)` (`(p, s) != (38, 9)`) vs actual `pl.Decimal(38, 9)` -> raise `pl.exceptions.SchemaError` explaining BigQuery Arrow always serializes `NUMERIC` as `pl.Decimal(38, 9)`.
    *   When `schema is not None` and `row_restriction != ""`, wrap `self._arrow_client.read_table(...)`: if `BigQueryError` is raised and `polars_schema` contains any `pl.String` column referenced by `predicate_cols`, chain a note suggesting `polars_bigquery.dtypes.JSON` / `polars_bigquery.dtypes.GEOGRAPHY` if a referenced column is actually BigQuery `JSON` or `GEOGRAPHY`.
*   **Observability**: Emit `DEBUG` logs via `logging.getLogger(__name__)` with table reference, `selected_fields`, compiled `row_restriction`, and skipped columns.

### 4. Schema-Aware Predicate Pushdown Compiler (`core/compiler/`)

The compiler transitions from a 3-stage schema-blind pipeline to a 4-stage schema-aware pipeline:

```text
Polars Expr JSON
   |
   v
1. json_to_ir(expr_json)               # Schema-blind parser (unchanged except FillNullWithStrategy)
   |
   v
2. resolve_and_lower_types(ir, schema) # NEW: iterative BFS + reverse-BFS typing, validation & lowering
   |
   v
3. rewrite_ir(ir)                      # Existing boolean rewriters (drop_nulls, when, literals, De Morgan)
   |
   v
4. ir_to_sql(ir)                       # Mechanical SQL emission (+ internal Bytes/Time/Decimal formatters)
```

#### Public Compiler Entrypoint Contract

*   Update `predicate_to_row_restriction(predicate: pl.Expr, *, schema: pl.Schema | Mapping[str, DataType | DataTypeClass]) -> str` so `schema` is a required keyword argument (single typed codepath; no permissive untyped legacy mode).
*   Normalize `schema` via `pl.Schema(schema)` and extract two lightweight column maps: `col_types: dict[str, BQType]` (via `polars_dtype_to_bq_type`) and `col_time_units: dict[str, str | None]` (via `polars_datetime_time_unit`).

#### New IR Nodes and SQL Formatters

*   `ir/boolean.py`: Add `FillNullWithStrategy(UnaryExpr)` with `strategy: Literal["Zero", "One"]`. Update `parser/boolean.py` (`parse_fill_null_with_strategy`) to produce `FillNullWithStrategy` instead of eagerly lowering to `FillNull(left, IntLiteral(0|1))`.
*   Internal-only literal IR nodes (produced exclusively by `resolve_and_lower_types` when lowering `FillNullWithStrategy`, not parsed from user JSON):
    *   `BytesLiteral(value: bytes)` -> `b''` (hex-escaped `b'...'` in `sql/`).
    *   `TimeLiteral(micros: int)` -> `TIME 'HH:MM:SS'` / `TIME 'HH:MM:SS.ffffff'` (for `0` -> `TIME '00:00:00'`).
    *   `DecimalLiteral(value: str)` -> `NUMERIC '0'` / `NUMERIC '1'`.

#### Iterative Pre-Rewrite Typing and Lowering Pass (`core/compiler/typing.py`)

Runs top-down BFS to index the tree (matching `rewrite_tree` and `_compile_ir_to_sql`, zero Python recursion), then evaluates bottom-up in reverse BFS order, tracking `(rewritten_node, inferred_type: BQType | None, is_known_non_nan: bool, node_count: int)` per node, plus a remaining node budget (`MAX_AST_NODES = 10_000`) to cap AST expansion.

1.  **Leaf inference**:
    *   `Column(name)` -> looks up `col_types.get(name)`. Missing column or `BQType.UNKNOWN` -> degrades to `Unsupported()` (`type=None`). `is_known_non_nan = (bq_type != BQType.FLOAT64)`.
    *   `BoolLiteral` -> `BOOL`; `IntLiteral` -> `INT64` (`non_nan=True`); `FloatLiteral(v)` -> `FLOAT64` (`non_nan = not math.isnan(v)`); `StringLiteral` -> `STRING`; `DateLiteral` -> `DATE`; `TimestampLiteral(tz=None)` -> `DATETIME`; `TimestampLiteral(tz!=None)` -> `TIMESTAMP`; `NullLiteral` -> `None` (wildcard); `ListLiteral` -> element type inferred from its homogeneous items (must all share one comparable scalar `BQType`, otherwise `Unsupported()`).
2.  **`BOOL` logical operators and partial-pushdown preservation**:
    *   `And`, `Or`, `Not`: **reject only when an operand has a known non-`BOOL` type** (catches bitwise integer `(flags & 4) == 4` and `~flags`). If an operand is `Unsupported()` (`type=None`), keep the `And`/`Or`/`Not` node intact with `inferred_type = BOOL` so `rewrite_de_morgan`, `rewrite_boolean_when`, and `emit_and_sql` preserve partial pushdown!
    *   `Root` validation: after bottom-up evaluation, if the root node has a known non-`BOOL` type (for example, bare `Column("int_col")`, `IntLiteral(1)`, non-bool `Coalesce`), return `Unsupported()`.
3.  **`OPAQUE` bucket (`JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`)**:
    *   `IsNull(e)` and `IsNotNull(e)`: valid for **any** known supported `BQType` (including `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT` — all live-verified in Storage Read API `row_restriction`), producing `BOOL`.
    *   `DropNulls(e)`: preserves `e`'s type when `e` has a known supported type (note: if an opaque `DropNulls(json_col)` is later consumed by a disallowed operator, that parent degrades to `Unsupported()`, and `rewrite_drop_nulls` won't hoist it).
    *   All other operators reject `OPAQUE` operands and degrade to `Unsupported()`.
4.  **String operators (`Uppercase`, `Lowercase`, `Contains`, `StartsWith`, `EndsWith`)**:
    *   Require every operand to have strict `BQType.STRING` (never `JSON`, `GEOGRAPHY`, `BYTES`, or `INT64`). `Uppercase`/`Lowercase` produce `STRING`; `Contains`/`StartsWith`/`EndsWith` produce `BOOL`.
5.  **Float predicates (`IsNan`, `IsNotNan`, `IsInfinite`, `IsFinite`)**:
    *   `FLOAT64` -> keep node as-is (`BOOL`).
    *   `INT64` -> lower to null-safe 3-valued `Ternary`:
        *   `IsNan(e)` / `IsInfinite(e)` -> `Ternary(IsNotNull(e), BoolLiteral(False), NullLiteral())`
        *   `IsNotNan(e)` / `IsFinite(e)` -> `Ternary(IsNotNull(e), BoolLiteral(True), NullLiteral())`
        *   In positive filter context, `rewrite_boolean_when` + `rewrite_boolean_literals` automatically fold these to `FALSE` / `(e IS NOT NULL)`; inside `Not`, `IsNull`, `FillNull`, or `==`, they emit `IF((e IS NOT NULL), FALSE|TRUE, NULL)` matching Polars' `[False, None]` / `[True, None]`.
    *   `NUMERIC` and all other types -> degrade to `Unsupported()`.
6.  **`FillNullWithStrategy(expr, strategy)` lowering**:
    *   Look up `expr`'s inferred `BQType` (and `time_unit` if `expr` is a `Column` of `DATETIME`/`TIMESTAMP`, defaulting to `"us"` for compound expressions):
        *   `BOOL`: `"Zero"` -> `BoolLiteral(False)`, `"One"` -> `BoolLiteral(True)`
        *   `INT64`: `"Zero"` -> `IntLiteral(0)`, `"One"` -> `IntLiteral(1)`
        *   `FLOAT64`: `"Zero"` -> `FloatLiteral(0.0)`, `"One"` -> `FloatLiteral(1.0)`
        *   `NUMERIC`: `"Zero"` -> `DecimalLiteral("0")`, `"One"` -> `DecimalLiteral("1")`
        *   `STRING`: `"Zero"` -> `StringLiteral("")`; `"One"` -> `Unsupported()`
        *   `BYTES`: `"Zero"` -> `BytesLiteral(b"")`; `"One"` -> `Unsupported()`
        *   `DATE`: `"Zero"` -> `DateLiteral(0)`, `"One"` -> `DateLiteral(1)`
        *   `DATETIME`: `"Zero"` -> `TimestampLiteral(0, MICROSECONDS, tz=None)`; `"One"` -> `TimestampLiteral(1, MILLISECONDS if unit == "ms" else MICROSECONDS, tz=None)` (if `unit == "ns"` -> `Unsupported()`)
        *   `TIMESTAMP`: `"Zero"` -> `TimestampLiteral(0, MICROSECONDS, tz="UTC")`; `"One"` -> `TimestampLiteral(1, MILLISECONDS if unit == "ms" else MICROSECONDS, tz="UTC")` (if `unit == "ns"` -> `Unsupported()`)
        *   `TIME`: `"Zero"` -> `TimeLiteral(0)`; `"One"` -> `Unsupported()` (Polars fills `1 ns`, unrepresentable in BigQuery microsecond `TIME`)
        *   `OPAQUE` / `None` -> `Unsupported()`
    *   Lowers to `FillNull(left=expr, right=typed_literal)` with result type equal to `expr`'s `BQType`.
7.  **Supertype unification for `FillNull`, `Coalesce`, `Ternary`, `When`, and `IsIn`**:
    *   Valid pairwise supertypes (live-verified against BigQuery + Polars `2.1.0`):
        *   `T + T -> T` for all 10 non-opaque scalar types (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`).
        *   `INT64 + FLOAT64 -> FLOAT64` (exact parity verified live for `2^53+1`).
        *   `INT64 + NUMERIC -> NUMERIC` (exact parity verified live).
        *   All other pairs — specifically **`NUMERIC + FLOAT64`** (live-verified divergence!), all cross-temporal pairs (`DATE` vs `DATETIME`, `DATE`/`DATETIME` vs `TIMESTAMP`, `TIME` vs non-`TIME`), `BYTES + STRING`, `BOOL + numeric`, and any `OPAQUE` type — return `None` -> degrade to `Unsupported()`.
    *   `Ternary(p, t, f)` and `When(p1, t1, ..., f)`: predicates must not have a known non-`BOOL` type; non-`NullLiteral` value branches unify under the supertype table above (preserving partial-pushdown `Ternary` when `t` or `f` is `Unsupported()` in boolean context, since `rewrite_boolean_when` splits it!).
    *   `IsIn(left, ListLiteral)`: strict same-type match (`left_type == element_type` for non-opaque scalar types, matching Polars `2.1.0`'s strict `is_in` typing; for `FLOAT64`, `parser/list_.py` already rejects `NaN` elements, so `col IN (finite_floats...)` is exact in both BigQuery and Polars).
8.  **Comparisons (`Eq`, `NotEq`, `Gt`, `GtEq`, `Lt`, `LtEq`) and Polars-Aligned `FLOAT64` `NaN` Lowering**:
    *   First unify `(left_type, right_type)` using the supertype table above; if incompatible or `OPAQUE`, degrade to `Unsupported()`.
    *   If promoted type is not `FLOAT64`, keep comparison node as-is (`inferred_type = BOOL`).
    *   If promoted type is `FLOAT64`, apply exact 3-valued total-order lowering (verified across all 9 `{finite/inf, NaN, NULL} × {finite/inf, NaN, NULL}` states):
        *   **Case A — One operand is `FloatLiteral(NaN)`** (shown for `a op NaN`; symmetric for `NaN op b`):
            *   `Eq` / `GtEq` -> `IsNan(a)` (if `a` can be NaN) else `Ternary(IsNotNull(a), BoolLiteral(False), NullLiteral())`
            *   `NotEq` / `Lt` -> `Not(IsNan(a))` (if `a` can be NaN) else `Ternary(IsNotNull(a), BoolLiteral(True), NullLiteral())`
            *   `LtEq` -> `Ternary(IsNotNull(a), BoolLiteral(True), NullLiteral())`
            *   `Gt` -> `Ternary(IsNotNull(a), BoolLiteral(False), NullLiteral())`
        *   **Case B — `b` is known non-`NaN` and non-`NULL` (finite/inf `FloatLiteral` or `IntLiteral`) and `a` can be `NaN`** (symmetric when `a` is literal):
            *   `Eq`, `NotEq`, `Lt`, `LtEq` -> unchanged (already exact; live-verified `NaN != 5.0` is `TRUE`).
            *   `Gt` -> `Or(Gt(a, b), IsNan(a))`
            *   `GtEq` -> `Or(GtEq(a, b), IsNan(a))`
        *   **Case C — `b` is known non-`NaN` but nullable (`INT64` expression) and `a` can be `NaN`** (symmetric when `a` is `INT64`):
            *   `Eq`, `NotEq`, `Lt`, `LtEq` -> unchanged.
            *   `Gt` -> `Or(Gt(a, b), And(IsNan(a), IsNotNull(b)))`
            *   `GtEq` -> `Or(GtEq(a, b), And(IsNan(a), IsNotNull(b)))`
        *   **Case D — Both `a` and `b` can be `NaN` (`FLOAT64` expressions)**:
            *   `Eq` -> `Or(Eq(a, b), And(IsNan(a), IsNan(b)))`
            *   `NotEq` -> `And(NotEq(a, b), Not(And(IsNan(a), IsNan(b))))`
            *   `Gt` -> `Or(Gt(a, b), And(IsNan(a), Not(IsNan(b))))`
            *   `GtEq` -> `Or(GtEq(a, b), And(IsNan(a), IsNotNull(b)))`
            *   `Lt` -> `Or(Lt(a, b), And(Not(IsNan(a)), IsNan(b)))`
            *   `LtEq` -> `Or(LtEq(a, b), And(IsNotNull(a), IsNan(b)))`
        *   **AST-duplication bound**: Before applying any Case B/C/D rewrite that duplicates an operand subtree `x`, check that `extra_nodes` fits within the remaining `MAX_AST_NODES` budget (and cap per-comparison operand duplication to `node_count(x) <= 64`, avoiding exponential growth on nested conditionals); if exceeded, degrade the comparison to `Unsupported()` and log at `DEBUG`.
9.  **`DEBUG` Logging**: Log whenever `resolve_and_lower_types` degrades a node due to type incompatibility, missing column, or AST-duplication budget.

### 5. End-User Documentation and API Reference

*   **New Data Types reference page (`docs/source/api/data-types.md`)**: Add a dedicated page to `mkdocs.yml` under `API reference` (cross-linked from `docs/source/api/index.md`, `docs/source/api/predicate-pushdown.md`, and `Client.scan_table` / `Client.read_table` docstrings) covering:
    *   The complete BigQuery-to-Polars type mapping table (all 14 supported types, scalar vs. `REPEATED` including `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` outer-extension wrapping, uppercase `"UTC"` on `TIMESTAMP`, `Decimal(38, 9)` for parameterized `NUMERIC(P, S)`, and ingestion-time `_PARTITIONDATE`/`_PARTITIONTIME` pseudo-columns).
    *   Working with `JSON` and `GEOGRAPHY` extension columns in Polars 2.0+ (copy-pasteable `.ext.storage()` examples for JSON/WKT access, `polars_bigquery.dtypes` usage in explicit `schema=`, and the `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` / `pl.register_extension_type(...)` contract).
    *   Unsupported and deferred BigQuery types (`BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, picosecond `TIMESTAMP`, unnamed/duplicate `STRUCT` subfields), whole-column omission with `UnsupportedTypeWarning`, `client.read_query()` SQL workarounds, and behavioral differences between `read_table()` and `scan_table()` (including the residual upstream `INTERVAL` crash in `read_table()`/`read_query()`).
*   **Updated API reference & predicate pushdown docs**:
    *   `docs/source/api/index.md`: add `::: polars_bigquery.dtypes` and `::: polars_bigquery.exceptions.UnsupportedTypeWarning`, plus a link to `data-types.md`.
    *   `docs/source/api/predicate-pushdown.md`: replace the stale embedded table-mapping section with a link to `data-types.md`, add a per-type operator pushdown matrix (including typed `fill_null(strategy="zero"|"one")`, `FLOAT64` `NaN` total-order parity, `OPAQUE` null-check-only types, and cross-type comparison rules), and add a "Common pitfalls that disable pushdown" callout (such as `NUMERIC` vs `FLOAT64`, cross-temporal mixes like `DATE`/`DATETIME`/`TIMESTAMP`, and naive vs. UTC datetimes on `_PARTITIONTIME`).

## Alternatives Considered

### Alternative A: Support Both Polars 1.x and 2.x Dynamically for `JSON` and `GEOGRAPHY`

*   **Pros**: Keeps existing Polars `1.43.x` installs working without raising the dependency floor in `pyproject.toml`.
*   **Cons**: Polars 1.x defaults to `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` (producing `pl.String`), while Polars 2.0+ defaults to `load_as_extension` (producing `pl.Extension`). Maintaining both paths requires version-gated schema construction, dual compiler classification rules, dual explicit-`schema=` guidance, and a split CI test matrix.
*   **Reason for Rejection**: Polars `2.1.0` is released and backward compatibility with Polars 1.x is not required. Raising the pin to `polars >= 2.0.0, < 3.0.0` leaves one consistent `pl.Extension` path aligned with `read_table()` and `read_query()`.

### Alternative B: Unwrap `JSON` and `GEOGRAPHY` to `pl.String` and Push `TO_JSON_STRING` / `ST_ASTEXT` (or `CAST(... AS STRING)`)

*   **Pros**: Lets users call `.str.*` directly on `JSON` and `GEOGRAPHY` columns without `.ext.storage()`, and avoids exposing Polars' `@unstable` `pl.Extension` dtype in `scan_table()` output.
*   **Cons**: Live tests against BigQuery Storage Read API showed `CAST(JSON AS STRING)` and `CAST(GEOGRAPHY AS STRING)` fail with `INVALID_ARGUMENT: Invalid cast`. Unwrapping `Extension` columns inside `source_generator` makes `scan_table()` schemas disagree with `read_table()` and `read_query()`, erases BigQuery type identity in explicit `schema=` calls, and risks silent row loss if `TO_JSON_STRING` or `ST_ASTEXT` formats any edge case differently from BigQuery's Arrow UTF-8 serializer.
*   **Reason for Rejection**: Preserving Polars 2.0+'s native `pl.Extension` types keeps `scan_table()` consistent with `read_table()`/`read_query()`, preserves BigQuery type identity inside `pl.Schema`, and eliminates SQL-vs-Arrow string formatting drift.

### Alternative C: Skip Pushdown for Non-Numeric `fill_null(strategy=...)` and `INT64` Float Predicates (Pure Fallback)

*   **Pros**: Smaller compiler footprint: avoids adding the `FillNullWithStrategy` IR node, internal `BytesLiteral`/`TimeLiteral`/`DecimalLiteral` formatters, and 3-valued `Ternary` rewrites for `INT64` `is_nan`/`is_inf`.
*   **Cons**: Leaves valid, representable filters (such as `pl.col("flag").fill_null(strategy="zero")`, `pl.col("label").fill_null(strategy="zero") == ""`, temporal `fill_null(strategy="zero")`, and `INT64` `is_nan`/`is_finite`) unpushed even though exact GoogleSQL equivalents exist and were verified live in Storage Read API `row_restriction`.
*   **Reason for Rejection**: Performing exact type-aware rewrites wherever GoogleSQL matches Polars semantics maximizes Storage Read API filtering while reserving fallback strictly for unrepresentable or divergent cases (`TIME` `"one"`, `STRING`/`BYTES` `"one"`, `Datetime("ns")` `"one"`, `NUMERIC` `is_nan`/`is_inf`, `NUMERIC` vs `FLOAT64`, and `OPAQUE` types).

### Alternative D: Fail Fast on Any Table Containing One Unsupported Column, or Strip Unsupported Nested `STRUCT` Subfields Now

*   **Pros**: Failing fast prevents silent schema differences between BigQuery and `lf.collect_schema()`; stripping only unsupported nested subfields would preserve supported sibling fields inside a `STRUCT`.
*   **Cons**: Failing on a single unsupported column makes entire mixed tables unscannable even when the user only selects `INT64`/`STRING` columns. Simply dropping a nested subfield from `polars_schema` breaks at runtime because BigQuery Storage Read API projects top-level columns in `selected_fields` and still transmits the omitted subfield (causing `SchemaError` or an upstream `INTERVAL` crash). Supporting partial `STRUCT` exclusion requires subfield projection and client-side `STRUCT` reconstruction.
*   **Reason for Rejection**: Omitting the enclosing top-level column with `UnsupportedTypeWarning` unblocks mixed tables now without risky client-side `STRUCT` reconstruction; partial `STRUCT` subfield exclusion is deferred under Future Improvements.

### Alternative E: Add Schema Checks Only Inside `core/compiler/sql/` Without a Pre-Rewrite Pass

*   **Pros**: Keeps the existing 3-stage compiler structure (`json_to_ir -> rewrite_ir -> ir_to_sql`) by passing `schema` into `_compile_ir_to_sql`.
*   **Cons**: Runs after `rewrite_drop_nulls` (which would hoist `IsNotNull` for type-invalid subtrees), after `rewrite_boolean_when` (which inspects `has_descendant(Unsupported)` for partial pushdown and wraps predicates in `COALESCE(..., FALSE)`), and after `rewrite_de_morgan` (which would rewrite bitwise integer `~(a | b)` as logical negation). It also cannot distinguish `fill_null(strategy="zero")` from `fill_null(0)` if `parser/boolean.py` still lowers strategies eagerly.
*   **Reason for Rejection**: Running an iterative `resolve_and_lower_types` pass between `json_to_ir` and `rewrite_ir` keeps `parser/` schema-blind, keeps `sql/` purely mechanical, and preserves all partial-pushdown and boolean-simplification rewrites.

### Alternative F: Allow Empty `polars_schema` to Call Storage Read API, or Reject Ingestion-Time Partitioned Tables Whose Physical Columns Are All Unsupported

*   **Pros**: Treating every empty `polars_schema` uniformly avoids special-casing 0-column tables; checking zero remaining columns before appending `_PARTITIONDATE`/`_PARTITIONTIME` avoids returning pseudo-column-only frames.
*   **Cons**: Live testing confirmed Storage Read API rejects `CreateReadSession` on genuinely 0-column tables (`FAILED_PRECONDITION`) and interprets `selected_fields=[]` on non-empty tables as "read all columns" (pulling omitted `INTERVAL` columns and crashing Python). Conversely, failing before appending `_PARTITIONDATE`/`_PARTITIONTIME` blocks partition row-count queries on ingestion-time partitioned tables whose physical columns are unsupported—even though live testing proved `selected_fields=["_PARTITIONDATE"]` works cleanly without reading omitted physical columns.
*   **Reason for Rejection**: Short-circuiting empty schemas protects 0-column tables and explicit `schema={}`, while checking for zero supported columns *after* appending `_PARTITIONDATE`/`_PARTITIONTIME` keeps partition-level scans working and fails early only when no scannable columns remain.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance & Billing Cost**:
    *   **Zero-column projection cost**: Choosing the narrowest fixed-width column (`Boolean` < `Date` < 8-byte scalars < `Decimal` < variable/nested) for empty `selected_fields` prevents scanning wide `STRING`/`JSON`/`STRUCT` columns (or the full table) on `lf.select(pl.len())`.
    *   **Bounded AST compilation time and SQL size**: `resolve_and_lower_types` uses strictly iterative BFS/reverse-BFS traversal (zero Python recursion) and caps `FLOAT64` `NaN` operand duplication (`node_count <= 64` per duplicated subtree and global `MAX_AST_NODES = 10_000`) so nested float comparisons cannot trigger exponential SQL expansion.
    *   **Observability**: `DEBUG` logs record table ID, `selected_fields`, compiled `row_restriction`, and type-driven pushdown fallbacks so engineers can diagnose unexpected full-table scans.
*   **Rollout & Compatibility**:
    *   **Breaking changes accepted pre-1.0**: Dropping Polars 1.x (`polars >= 2.0.0, < 3.0.0`), changing `TIMESTAMP`/`_PARTITIONTIME` from `"utc"` to `"UTC"`, mapping `JSON`/`GEOGRAPHY` (and `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>`) to `pl.Extension(...)`, switching single unsupported columns from `TypeError` to `UnsupportedTypeWarning` omission, and making `schema` required on internal `predicate_to_row_restriction`.
    *   **Residual crash boundary**: `Client.read_table()`, `Client.read_query()`, and explicit `scan_table(schema=...)` (unless passing `pl.Duration`, which is rejected early) remain exposed to upstream Polars/`polars-arrow` `MonthDayNano` crashes on `INTERVAL` columns until fixed upstream or guarded in `crates/arrow-bigquery`.
*   **Security & Privacy**:
    *   `DEBUG` logs emit compiled SQL and column names (consistent with existing `parser/` debug logs) and never log OAuth tokens or row values beyond literal filter constants already present in the user's Polars expression.
    *   Internal `BytesLiteral` SQL formatting must hex-escape every byte (`\xNN`) to prevent SQL injection if future callers ever parse user binary literals.

--------------------------------------------------------------------------------

### Additional Resources

*   Existing row predicate pushdown design: [`polars_bigquery_scan_table_row_predicate_pushdown.md`](./polars_bigquery_scan_table_row_predicate_pushdown.md)
*   BigQuery REST `TableFieldSchema` reference: <https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#TableFieldSchema>
*   BigQuery Storage Read API `TableReadOptions` reference: <https://cloud.google.com/bigquery/docs/reference/storage/rpc/google.cloud.bigquery.storage.v1#tablereadoptions>
*   Upstream Polars `2.1.0` & `polars-arrow 0.53.0` `MonthDayNano` crashes: `polars-core/src/datatypes/field.rs:326` `PanicException` on default import, `SIGSEGV` in `pl.DataFrame` under `POLARS_IMPORT_INTERVAL_AS_STRUCT=1`, and `polars-arrow 0.53.0` `ffi/bridge.rs:20:33` non-unwinding `SIGABRT` on Arrow C Stream export.

### Future Improvements

*   **Partial `STRUCT` subfield exclusion**: Project only supported subfields from a `STRUCT` containing an unsupported descendant and reconstruct pruned `STRUCT` columns client-side for BigQuery Storage Read API compatibility.
*   **User-facing `BYTES`, `TIME`, and `Decimal` literal and `IsIn` parsing**: Extend `core/compiler/parser/` to parse user-supplied `pl.Binary`, `pl.Time` (microsecond-aligned), and `pl.Decimal` scalar literals and `is_in` collections.
*   **Deferred BigQuery types**: Add full Arrow/Polars support for `BIGNUMERIC`, parameterized `NUMERIC(P, S)`, picosecond `TIMESTAMP`, `INTERVAL` (once upstream Polars/`polars-arrow` `MonthDayNano` issues are resolved), `RANGE<T>`, and unnamed/duplicate `STRUCT` subfields.
*   **Rust-layer `MonthDayNano` guard in `crates/arrow-bigquery`**: Detect `ArrowDataType::Interval(IntervalUnit::MonthDayNano)` in `decode.rs`/`stream.rs` and return a clean `BigQueryError` before Polars FFI export so `read_table()` and `read_query()` never abort Python.
*   **`scan_query` integration**: Reuse `core/types.py`, `core/schema.py`, and the schema-aware compiler once query-backed lazy scanning is designed.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Target Files to Create/Modify

#### 1. `py-polars-bigquery/pyproject.toml` (Modify)

*   **Type / Scope**: Python Package Configuration
*   **Proposed Changes**: Raise the `polars` dependency specifier from `polars >= 1.43.0, < 3.0.0` to `polars >= 2.0.0, < 3.0.0` so the package relies exclusively on Polars 2.0+ `pl.Extension` Arrow loading without Polars 1.x compatibility shims.

#### 2. `uv.lock` (Modify)

*   **Type / Scope**: Workspace Dependency Lockfile
*   **Proposed Changes**: Re-lock workspace dependencies via `uv lock` / `uv sync --project py-polars-bigquery --group dev` so the locked Polars version satisfies `>= 2.0.0, < 3.0.0` (Polars `2.1.0+`).

#### 3. `py-polars-bigquery/polars_bigquery/core/types.py` (Create)

*   **Type / Scope**: Internal Type System Module
*   **Proposed Changes**: Establish the single source of truth shared by `dtypes.py`, `core/schema.py`, `_read_bigquery.py`, and `core/compiler/`:
    *   `BQType(enum.Enum)`: closed enumeration of all 14 supported BigQuery types (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`) plus `UNKNOWN`.
    *   `OPAQUE_BQ_TYPES`: frozenset of `{BQType.JSON, BQType.GEOGRAPHY, BQType.ARRAY, BQType.STRUCT, BQType.UNKNOWN}`.
    *   Extension constants: `JSON_EXTENSION_NAME = "google:sqlType:json"`, `GEOGRAPHY_EXTENSION_NAME = "google:sqlType:geography"`, `GEOGRAPHY_EXTENSION_METADATA = '{"encoding": "WKT"}'`.
    *   `build_extension_dtype(name: str, storage: DataType, metadata: str | None = None) -> DataType`: resolves via `pl.get_extension_type(name)` (instantiating a registered `BaseExtension` subclass via `ext_from_params`, returning `storage` when registered as `"storage"`, or returning `pl.Extension(name, storage, metadata)` by default).
    *   `polars_dtype_to_bq_type(dtype: DataType | DataTypeClass) -> BQType`: normalizes `DataTypeClass` objects and maps every Polars dtype (including `BaseExtension` subclasses, outer-wrapped `List` extensions -> `ARRAY`, `Decimal(p <= 38, s <= 9)` -> `NUMERIC`, naive vs. tz-aware `Datetime`, and unknown dtypes -> `UNKNOWN`).
    *   `polars_datetime_time_unit(dtype: DataType | DataTypeClass) -> Literal["ms", "us", "ns"] | None`: extracts time unit from `Datetime` dtypes for `fill_null(strategy="one")` lowering.
    *   `column_width_rank(dtype: DataType) -> int`: returns a deterministic storage-width rank (`Boolean` < `Date` < `Int64`/`Float64`/`Time`/`Datetime` < `Decimal` < `String`/`Binary`/`BaseExtension` < `Struct`/`List`) used to select the cheapest fallback column when `selected_fields` would be empty.

#### 4. `py-polars-bigquery/polars_bigquery/dtypes.py` (Create)

*   **Type / Scope**: Public Python API Module
*   **Proposed Changes**: Define and export (`__all__ = ["JSON", "GEOGRAPHY"]`) with NumPy-style docstrings for `mkdocstrings`:
    *   `JSON`: canonical `pl.Extension("google:sqlType:json", pl.String())` dtype for BigQuery `JSON` columns.
    *   `GEOGRAPHY`: canonical `pl.Extension("google:sqlType:geography", pl.String(), '{"encoding": "WKT"}')` dtype for BigQuery `GEOGRAPHY` columns.

#### 5. `py-polars-bigquery/polars_bigquery/exceptions.py` (Modify)

*   **Type / Scope**: Public Exceptions & Warnings Module
*   **Proposed Changes**: Add `UnsupportedTypeWarning(UserWarning)` with a NumPy-style docstring explaining it is emitted by `Client.scan_table` when unsupported or deferred BigQuery columns are omitted, and export it in `__all__`.

#### 6. `py-polars-bigquery/polars_bigquery/__init__.py` (Modify)

*   **Type / Scope**: Package Entrypoint
*   **Proposed Changes**: Import and export `dtypes` in `__all__` so users can reference `polars_bigquery.dtypes.JSON` and `polars_bigquery.dtypes.GEOGRAPHY` immediately after `import polars_bigquery`.

#### 7. `py-polars-bigquery/polars_bigquery/core/schema.py` (Modify)

*   **Type / Scope**: Internal Schema Extraction Module
*   **Proposed Changes**:
    *   Remove the unused `Duration` import and add frozen dataclasses `SkippedColumn(name: str, reason: str, path: str)` and `ExtractedSchema(polars_schema: pl.Schema, physical_columns: tuple[str, ...], pseudo_columns: tuple[str, ...], skipped_columns: tuple[SkippedColumn, ...], total_table_fields: int)`.
    *   Add internal `_UnsupportedBigQueryType(Exception)` carrying `(reason, path)` so any unsupported descendant inside `STRUCT` or `ARRAY` bubbles up to omit the entire enclosing top-level column.
    *   In `_extract_data_type(field: dict, path: str) -> DataType`:
        *   Use null-safe extraction `(field.get("mode") or "").casefold()` and `(field.get("type") or "").casefold()`.
        *   Handle `mode == "repeated"` + `type_ in ("json", "geography")` explicitly by calling `build_extension_dtype(..., pl.List(pl.String()), ...)` so `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>` wrap `List(String)` on the outside.
        *   For all other `mode == "repeated"` fields, recurse with `{**field, "mode": "NULLABLE"}` to preserve all field parameters (`timestampPrecision`, `precision`, `scale`, `rangeElementType`, etc.).
        *   Map `timestamp` to `Datetime(time_unit="us", time_zone="UTC")` only when `timestampPrecision` is `None`, `6`, or `"6"` (excluding `bool`); raise `_UnsupportedBigQueryType` for picosecond or other precisions.
        *   Map scalar `json` and `geography` via `build_extension_dtype(...)`.
        *   Raise `_UnsupportedBigQueryType` for `bignumeric`/`bigdecimal`, `interval`, `range`, and any unrecognized type.
        *   For `record`/`struct`: reject missing/empty `fields`, reject any subfield with missing/empty `"name"` (unnamed subfield), reject duplicate subfield names within the same struct, and pass `f"{path}.{sub_name}"` into recursive calls.
    *   Add pure `extract_schema_metadata(table_metadata: dict) -> ExtractedSchema` (no warning/error side effects): processes `table_metadata.get("schema", {}).get("fields", [])`, records supported physical columns and `SkippedColumn`s, and appends `_PARTITIONDATE` (`pl.Date()`) and `_PARTITIONTIME` (`pl.Datetime(time_unit="us", time_zone="UTC")`) when `timePartitioning` exists with `field is None`.
    *   Keep `extract_polars_schema(table_metadata: dict) -> pl.Schema` as a thin wrapper returning `extract_schema_metadata(table_metadata).polars_schema`.

#### 8. `py-polars-bigquery/polars_bigquery/_read_bigquery.py` (Modify)

*   **Type / Scope**: Public Client & Polars I/O Plugin Module
*   **Proposed Changes**:
    *   Add module logger `logger = logging.getLogger(__name__)`.
    *   In `Client.scan_table`:
        *   When `schema is None`: call `extracted = extract_schema_metadata(table_metadata)`. If `extracted.total_table_fields > 0 and len(extracted.polars_schema) == 0`, raise `ValueError` immediately (no warning beforehand) listing up to 5 skipped columns with reasons/paths, `"... and N more"`, and a `client.read_query()` workaround. Otherwise, if `extracted.skipped_columns` is non-empty, emit `warnings.warn(..., UnsupportedTypeWarning, stacklevel=2)` with total count, up to 5 entries, `"... and N more"`, and the `client.read_query()` workaround.
        *   When `schema is not None`: normalize via `pl.Schema(schema)` and recursively validate that no field declares `pl.Duration` (raising `ValueError` early with a message explaining BigQuery `INTERVAL` is unsupported).
    *   In `source_generator`:
        *   If `len(polars_schema) == 0`: yield `pl.DataFrame(schema=polars_schema)` and return without calling `read_table()`.
        *   Compute `selected_fields = list(dict.fromkeys([*columns, *predicate_cols]))`. If `not selected_fields`, select the single column in `polars_schema` with the lowest `column_width_rank(dtype)` (tie-breaking by schema order).
        *   Compile predicates via `compiler.predicate_to_row_restriction(predicate=predicate, schema=polars_schema)`. Emit a `DEBUG` log with `table_ref`, `selected_fields`, and `row_restriction`.
        *   Wrap `self._arrow_client.read_table(...)`: when `schema is not None` and `row_restriction` is non-empty and `BigQueryError` is raised while `predicate_cols` references any `pl.String` column, raise a chained `BigQueryError` (or append a note via `add_note`) pointing to `polars_bigquery.dtypes.JSON` / `GEOGRAPHY`.
        *   After `lazyframe = pl.scan_arrow_c_stream(arrow_stream_exporter)`, when `schema is not None`, run an upfront helper `_validate_explicit_schema(expected_schema, lazyframe.collect_schema(), selected_fields)` recursively checking for `pl.String` vs `JSON`/`GEOGRAPHY`, `List(Extension)` / `List(String)` vs `Extension(List(String))`, `"utc"` vs `"UTC"`, and `Decimal != (38, 9)`, raising `pl.exceptions.SchemaError` with copy-pasteable fixes.
        *   Apply `filter(predicate)` and `limit(n_rows)`. If `columns == []`, return a generator yielding `pl.DataFrame(height=batch.height)` for each batch from `lazyframe.collect_batches(chunk_size=batch_size)` rather than calling `lazyframe.select([])`. Otherwise select `columns` and return `lazyframe.collect_batches(chunk_size=batch_size)`.
    *   Update `Client.scan_table` and `Client.read_table` docstrings: change `"utc"` -> `"UTC"`, document `UnsupportedTypeWarning` (`Warns` section) and `ValueError` (`Raises` section), add explicit `schema=` examples using `polars_bigquery.dtypes`, explain `read_table()` vs `scan_table()` differences on unsupported types, and link to `data-types.md`.

#### 9. `py-polars-bigquery/polars_bigquery/core/compiler/ir/boolean.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Add frozen dataclass `FillNullWithStrategy(UnaryExpr)` with field `strategy: Literal["Zero", "One"]` so `parser/boolean.py` preserves strategy intent without premature lowering to `IntLiteral`.

#### 10. `py-polars-bigquery/polars_bigquery/core/compiler/ir/numeric.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Add internal frozen dataclass `DecimalLiteral(Literal)` with field `value: str` (used by `typing.py` when lowering `NUMERIC` `FillNullWithStrategy`).

#### 11. `py-polars-bigquery/polars_bigquery/core/compiler/ir/string.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Add internal frozen dataclass `BytesLiteral(Literal)` with field `value: bytes` (used by `typing.py` when lowering `BYTES` `FillNullWithStrategy("Zero")`).

#### 12. `py-polars-bigquery/polars_bigquery/core/compiler/ir/temporal.py` (Modify)

*   **Type / Scope**: Compiler IR Module
*   **Proposed Changes**: Add internal frozen dataclass `TimeLiteral(Literal)` with field `micros: int` (used by `typing.py` when lowering `TIME` `FillNullWithStrategy("Zero")`).

#### 13. `py-polars-bigquery/polars_bigquery/core/compiler/ir/__init__.py` (Modify)

*   **Type / Scope**: Compiler IR Package Exports
*   **Proposed Changes**: Import and export `FillNullWithStrategy`, `BytesLiteral`, `DecimalLiteral`, and `TimeLiteral` in `__all__`.

#### 14. `py-polars-bigquery/polars_bigquery/core/compiler/parser/boolean.py` (Modify)

*   **Type / Scope**: Compiler Parser Module
*   **Proposed Changes**: Remove `_FILL_NULL_STRATEGY_VALUES` and `IntLiteral` import; update `parse_fill_null_with_strategy` to validate `spec in ("Zero", "One")` and construct `FillNullWithStrategy(expr=..., strategy=spec)` in a schema-blind manner.

#### 15. `py-polars-bigquery/polars_bigquery/core/compiler/typing.py` (Create)

*   **Type / Scope**: Schema-Aware Compiler Pass
*   **Proposed Changes**: Implement `resolve_and_lower_types(root: Expr, schema: pl.Schema) -> Expr` using a strictly iterative two-pass traversal (top-down BFS indexing + bottom-up reverse-BFS evaluation) with per-node metadata `(node, bq_type, is_known_non_nan, node_count)` and node-budget tracking (`MAX_AST_NODES = 10_000`, `MAX_DUPLICATED_OPERAND_NODES = 64`):
    *   Leaf inference for `Column` (degrading missing columns and `BQType.UNKNOWN` to `Unsupported()`), all scalar literals, `NullLiteral`, and `ListLiteral`.
    *   `BOOL` enforcement for `And`, `Or`, `Not`, `Ternary`/`When` predicates, and the root expression while preserving partial pushdown when an operand is `Unsupported()` (`bq_type is None`).
    *   `OPAQUE` allowlist (`JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`) permitting only `IsNull`, `IsNotNull`, and `DropNulls`.
    *   Strict `BQType.STRING` enforcement for `Uppercase`, `Lowercase`, `Contains`, `StartsWith`, and `EndsWith`.
    *   Float predicate lowering (`IsNan`, `IsNotNan`, `IsInfinite`, `IsFinite`): keep for `FLOAT64`; rewrite `INT64` to `Ternary(IsNotNull(e), BoolLiteral(False|True), NullLiteral())`; degrade `NUMERIC` and all other types to `Unsupported()`.
    *   Full exact `FillNullWithStrategy` lowering across `BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING` (`"Zero"`), `BYTES` (`"Zero"`), `DATE`, `DATETIME` (respecting `"ms"`/`"us"`, skipping `"ns"` `"One"`), `TIMESTAMP` (respecting `"ms"`/`"us"`, skipping `"ns"` `"One"`), and `TIME` (`"Zero"` only; skipping `"One"`).
    *   Supertype unification helper `_unify_types(t1, t2)` supporting `T + T`, `INT64 + FLOAT64 -> FLOAT64`, and `INT64 + NUMERIC -> NUMERIC`, while rejecting `NUMERIC + FLOAT64`, all cross-temporal pairs (`DATE` vs `DATETIME`, `DATE`/`DATETIME` vs `TIMESTAMP`, `TIME` vs non-`TIME`), `BYTES + STRING`, `BOOL + numeric`, and `OPAQUE`.
    *   Strict same-type validation for `IsIn` (non-opaque scalar types).
    *   Exact 9-state 3-valued `FLOAT64` `NaN` comparison lowering (Cases A, B, C, and D from Technical Plan Section 4) guarded by `MAX_DUPLICATED_OPERAND_NODES` and remaining `MAX_AST_NODES` budget.
    *   `DEBUG` logging whenever a subtree degrades to `Unsupported()`.

#### 16. `py-polars-bigquery/polars_bigquery/core/compiler/sql/string.py` (Modify)

*   **Type / Scope**: Compiler SQL Emitter
*   **Proposed Changes**: Register `@literal_ir_to_sql.register` for `BytesLiteral`, formatting every byte as hex escape (`b'...'`, with empty bytes -> `b''`).

#### 17. `py-polars-bigquery/polars_bigquery/core/compiler/sql/temporal.py` (Modify)

*   **Type / Scope**: Compiler SQL Emitter
*   **Proposed Changes**: Register `@literal_ir_to_sql.register` for `TimeLiteral`, validating bounds (`0 <= micros < 86_400_000_000`) and formatting as `TIME 'HH:MM:SS'` (or `TIME 'HH:MM:SS.ffffff'` when sub-second microseconds are non-zero).

#### 18. `py-polars-bigquery/polars_bigquery/core/compiler/sql/numeric.py` (Modify)

*   **Type / Scope**: Compiler SQL Emitter
*   **Proposed Changes**: Register `@literal_ir_to_sql.register` for `DecimalLiteral`, validating numeric syntax and formatting as `NUMERIC '{value}'`.

#### 19. `py-polars-bigquery/polars_bigquery/core/compiler/__init__.py` (Modify)

*   **Type / Scope**: Compiler Pipeline Entrypoint
*   **Proposed Changes**: Export `typing` and `resolve_and_lower_types` in `__all__`; update `_json_expr_to_row_restriction(expr_json, schema)` and `predicate_to_row_restriction(predicate: pl.Expr, *, schema: pl.Schema | Mapping[str, DataType | DataTypeClass]) -> str` to normalize `schema` via `pl.Schema(schema)` and run the 4-stage pipeline `json_to_ir -> resolve_and_lower_types -> rewrite_ir -> ir_to_sql`.

#### 20. `docs/source/api/data-types.md` (Create)

*   **Type / Scope**: End-User Documentation
*   **Proposed Changes**: Publish the complete BigQuery-to-Polars Data Types guide: supported 14-type mapping table (scalar and `REPEATED`, including `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>`, `"UTC"`, `Decimal(38, 9)`, and `_PARTITIONDATE`/`_PARTITIONTIME`), working with `JSON`/`GEOGRAPHY` via `.ext.storage()` and `polars_bigquery.dtypes` in explicit `schema=`, `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` and `pl.register_extension_type(...)` notes, skipped/unsupported types with `UnsupportedTypeWarning` and `client.read_query()` workarounds, and behavioral differences between `read_table()` and `scan_table()` (including the residual upstream `INTERVAL` crash).

#### 21. `mkdocs.yml` (Modify)

*   **Type / Scope**: Documentation Navigation Configuration
*   **Proposed Changes**: Add `Data types: api/data-types.md` under `nav -> API reference` so `mkdocs build --strict` builds and links the new page.

#### 22. `docs/source/api/index.md` (Modify)

*   **Type / Scope**: API Reference Index
*   **Proposed Changes**: Add a link to `data-types.md` and register `::: polars_bigquery.dtypes` and `::: polars_bigquery.exceptions.UnsupportedTypeWarning`.

#### 23. `docs/source/api/predicate-pushdown.md` (Modify)

*   **Type / Scope**: Predicate Pushdown Documentation
*   **Proposed Changes**: Replace the stale embedded BigQuery-to-Polars table schema mapping with a cross-link to `data-types.md`; update `fill_null(strategy="zero"|"one")`, `FLOAT64` `NaN`, and `INT64` `is_nan`/`is_inf` notes; add a per-type pushdown support matrix covering all 14 supported types; and add a "Common pitfalls that disable pushdown" section (`NUMERIC` vs `FLOAT64`, cross-temporal mixes like `DATE`/`DATETIME`/`TIMESTAMP`, naive vs. UTC `_PARTITIONTIME`, bitwise `&`/`|`/`~`, and `OPAQUE` types).

#### 24. `py-polars-bigquery/tests/unit/core/test_types.py` (Create)

*   **Type / Scope**: Unit Tests
*   **Proposed Changes**: Test `polars_bigquery.dtypes` (`JSON`, `GEOGRAPHY`), `build_extension_dtype()` (default `pl.Extension`, registered custom `BaseExtension` subclass, registered `as_storage=True` with proper cleanup via `unregister_extension_type`), `polars_dtype_to_bq_type()` across all supported and unsupported Polars dtypes (including `DataTypeClass`), `polars_datetime_time_unit()`, and `column_width_rank()`.

#### 25. `py-polars-bigquery/tests/unit/core/test_schema.py` (Modify)

*   **Type / Scope**: Unit Tests
*   **Proposed Changes**: Update existing assertions to `"UTC"`, `pl.Extension` for `JSON`/`GEOGRAPHY`, and omission semantics; add tests for `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` outer-extension wrapping (top-level and inside `STRUCT`), `ARRAY<STRUCT<j JSON, g GEOGRAPHY>>`, `REPEATED` parameter preservation (`ARRAY<TIMESTAMP(12)>`), `timestampPrecision` allowlist (`None`, `6`, `"6"`, `12`, `"12"`, `"9"`, `True`), null-safe `mode`/`type` (`None`), unnamed/duplicate/empty `STRUCT` omission, nested whole-column omission with exact paths, `ExtractedSchema` structure, ingestion-time partitioned tables with zero physical columns surviving, and genuinely 0-column tables.

#### 26. `py-polars-bigquery/tests/unit/core/test_predicates.py` (Modify)

*   **Type / Scope**: Unit Tests
*   **Proposed Changes**: Update existing `predicate_to_row_restriction` calls to supply `schema=`; add parameterized tests covering every rule in `typing.py`: `FillNullWithStrategy` across all 14 types × `"zero"`/`"one"` (including `"ms"`/`"us"`/`"ns"` datetimes and unrepresentable `TIME` `"one"`), `INT64` vs `FLOAT64` vs `NUMERIC` float predicates (positive, `~`, `.is_null()`, `.fill_null()`), bitwise `INT64` `&`/`|`/`~` rejection, non-`BOOL` root rejection, `OPAQUE` (`JSON`/`GEOGRAPHY`/`ARRAY`/`STRUCT`) allowlist, strict `STRING` operators, cross-type comparison/coalesce/`when`/`is_in` matrix, all 4 `FLOAT64` `NaN` comparison cases (evaluating generated expressions across all 9 `{finite, NaN, NULL}` pairs) + `MAX_DUPLICATED_OPERAND_NODES` budget cap, partial pushdown preservation (`And`, De Morgan, boolean `when`), missing columns, non-BQ Polars dtypes, and deep ASTs (`MAX_AST_NODES`) without `RecursionError`.

#### 27. `py-polars-bigquery/tests/unit/test_read_bigquery.py` (Modify)

*   **Type / Scope**: Unit Tests (Mocked & Real Arrow C Stream)
*   **Proposed Changes**: Update existing `scan_table` tests to `"UTC"` and `schema=`; add unit tests for `UnsupportedTypeWarning` (message format, 5-column cap + `... and N more`, `stacklevel`, `read_query()` hint), early `ValueError` on non-partitioned all-unsupported tables (without extra warning), ingestion-time partitioned table retaining `_PARTITIONDATE`/`_PARTITIONTIME`, empty-schema short-circuit (0-column table and explicit `schema={}`), early `ValueError` for explicit `schema=` containing `pl.Duration`, cheapest scalar fallback `selected_fields` and height preservation on `with_columns=[]` (`lf.select(pl.len())`), explicit `schema=` upfront `collect_schema()` diagnostics (`pl.String` vs `JSON`/`GEOGRAPHY`, `List(Extension)` vs `Extension(List)`, `"utc"`, `Decimal(10, 2)`), filtered `BigQueryError` hint on explicit `schema=`, and real Arrow C stream unit tests passing actual `tsu:UTC` and `ARROW:extension:name`/`metadata` capsules through `pl.scan_arrow_c_stream` and `register_io_source`.

#### 28. `py-polars-bigquery/tests/system/test_read_bigquery_integration.py` (Modify)

*   **Type / Scope**: Live BigQuery Integration / Differential System Tests
*   **Proposed Changes**: Add live BigQuery integration tests (cleaning up temporary auto-expiring tables in `finally:`) covering:
    *   Mixed table containing supported types plus `BIGNUMERIC`, `INTERVAL`, `RANGE<DATE>`, `TIMESTAMP(12)`, and `STRUCT<ok INT64, bad INTERVAL>`: verifies `UnsupportedTypeWarning`, schema omission, `lf.select(pl.len()).collect()` accuracy, and `lf.collect()` success.
    *   Ingestion-time partitioned table with only an `INTERVAL` column: verifies `_PARTITIONDATE`/`_PARTITIONTIME` partition row counts succeed without reading `INTERVAL`.
    *   Genuinely 0-column table: verifies `scan_table().collect()` returns a 0-row, 0-column `DataFrame`.
    *   Schema parity (`scan_table().collect_schema() == read_table().schema`) on `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY<JSON>`, `ARRAY<GEOGRAPHY>`, and `STRUCT<j JSON, g GEOGRAPHY>`.
    *   Differential pushdown parity (`scan_table().filter(expr).collect()` vs. `read_table().filter(expr)`) across all 14 supported types × operators, including `FLOAT64` `NaN`/`±Inf`/`NULL` comparisons (all 6 ops, positive and negated), `INT64` `is_nan`/`is_not_nan`/`is_infinite`/`is_finite` (with `NULL` and `~`), `fill_null(strategy="zero"|"one")` across types, bitwise `INT64`, `INT64(2^53+1)` vs `FLOAT64(2^53)`, `NUMERIC(2^53+1)` vs `FLOAT64(2^53)`, and `OPAQUE` `IS [NOT] NULL` / `.ext.storage()`.

### Verification Plan

*   **Dependency Lock & Build**:
    *   `uv sync --project py-polars-bigquery --group dev`
    *   Verify `uv run --project py-polars-bigquery python -c "import polars; assert polars.__version__.startswith('2.')"`
*   **Unit Tests**:
    *   `uv run --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --project py-arrow-bigquery --group test pytest -n auto py-arrow-bigquery/tests/unit`
    *   `cargo test --workspace --exclude py-arrow-bigquery --exclude py-polars-expressions-testing --lib --bins --features testing`
*   **Static Analysis, Linting & Docs Build**:
    *   `uv run --project py-polars-bigquery --group dev ruff check --no-fix py-polars-bigquery docs/source/`
    *   `uv run --project py-polars-bigquery --group dev ruff format --diff py-polars-bigquery docs/source/`
    *   `uv run --with-requirements docs/source/requirements.txt mkdocs build --strict`
*   **Test Coverage Requirement**:
    *   100% branch coverage of new `core/types.py`, `dtypes.py`, `core/schema.py` omission paths, and `core/compiler/typing.py` type/rewrite matrix (all 14 `BQType`s × operators, all 9 `{finite, NaN, NULL}` `FLOAT64` comparison combinations, and real Arrow C stream FFI tests).
*   **E2E / Integration Tests**:
    *   With `GOOGLE_CLOUD_PROJECT` set and ADC configured: `uv run --project py-polars-bigquery --group test pytest py-polars-bigquery/tests/system`

--------------------------------------------------------------------------------

*Drafted with EGM*
