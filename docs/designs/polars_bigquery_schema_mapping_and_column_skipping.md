# BigQuery Schema Mapping and Column Skipping for Polars 2.0+

*   **Author:** @tswast
*   **Status:** Proposed
*   **Created:** 2026-11-24
*   **Last updated:** 2026-11-24
*   **Related design docs:** [Row Predicate Pushdown](./polars_bigquery_scan_table_row_predicate_pushdown.md), [`Client.scan_query()`](./polars_bigquery_scan_query.md)

--------------------------------------------------------------------------------

## Overview

Before reading rows over the BigQuery Storage Read API, `Client.scan_table()` must supply Polars with the full output schema. Unlike eager `Client.read_table()` (which infers the schema from BigQuery's Arrow IPC stream), `Client.scan_table()` builds a `pl.Schema` from the JSON response returned by the BigQuery `tables.get` REST API (unless the caller passes an explicit `schema=`). Similarly, `Client.scan_query()` discovers its initial schema from the JSON `bigquerySchema` payload returned by a `jobs.query` dry run.

In Polars 2.0+, Arrow C Stream validation fails `.collect()` when the declared schema differs from the Arrow IPC stream returned at execution time. Note that all baseline statements in `The Problem` describe the `main` branch prior to this change (commit `32fdc1f`). This design updates the schema-mapping layer in `polars_bigquery`:

1.  Aligning extracted Polars dtypes with BigQuery's Arrow IPC output (uppercase `"UTC"` timezone, Arrow extension types for `JSON` and `GEOGRAPHY`, and outer-wrapped `pl.List(pl.String())` extensions for `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>`).
2.  Distinguishing table-scan extraction (which skips unsupported top-level columns with `UnsupportedTypeWarning` so mixed tables remain readable, and raises `UnsupportedTypeError` if zero supported columns or pseudo-columns remain) from query-scan extraction (which rejects unsupported columns immediately with `TypeError` so verbatim `SELECT` outputs are never omitted).
3.  Preserving REST field attributes across `REPEATED` (`ARRAY`) recursion and validating `timestampPrecision`.
4.  Restricting the `_PARTITIONDATE` pseudo-column to daily (`DAY`) ingestion-time partitioned tables with at least one physical column.
5.  Short-circuiting discovered zero-column tables (including schemaless ingestion-time partitioned tables) without opening a Storage Read API session, while rejecting explicit empty `schema={}` (`ValueError`, matching `Client.scan_query()`).
6.  Consolidating zero-column fallback column ranking inside `ReadApiExecutor` onto the existing `core/types.py::column_width_rank` utility (`1`–`6` across the 14 supported `BQType` values, `7` for unsupported/unknown types), choosing `_PARTITIONDATE` (`DATE`, rank `2`) over `_PARTITIONTIME` (`TIMESTAMP`, rank `3`) on daily tables when only pseudo-columns remain.

## The Problem

This repository spans three layers: `crates/arrow-bigquery` (Rust Storage Read API and `jobs.query` Arrow IPC client), `py-arrow-bigquery` (PyO3 bindings exposing Arrow C Stream export), and `py-polars-bigquery` (Python Polars connector). Prior PRs added [`UnsupportedTypeError` (`#116`)](https://github.com/pola-rs/polars-bigquery-client/pull/116) (a `BigQueryError` subclass guarding against Arrow C Data Interface FFI failures inside `arrow-bigquery`), introduced `BQType`, `build_extension_dtype`, `column_width_rank`, and scalar constants `polars_bigquery.dtypes.JSON` and `GEOGRAPHY` ([`#121`](https://github.com/pola-rs/polars-bigquery-client/pull/121)), and stripped `google:sqlType:datetime` extension metadata during Arrow decoding so `DATETIME` decodes as `pl.Datetime("us")` ([`#123`](https://github.com/pola-rs/polars-bigquery-client/pull/123)).

Prior to this change (`main` at commit `32fdc1f`), `py-polars-bigquery/polars_bigquery/core/schema.py` fails on BigQuery tables under Polars 2.0+ for five reasons:

1.  **`TIMESTAMP` timezone casing mismatch.** BigQuery's Arrow IPC stream marks `TIMESTAMP` columns and `_PARTITIONTIME` with uppercase `"UTC"`. The schema extractor emitted lowercase `time_zone="utc"`, causing `scan_table().collect_schema()` to disagree with `read_table().schema`.
2.  **`JSON` and `GEOGRAPHY` extension mismatch.** BigQuery tags `JSON` with Arrow extension name `google:sqlType:json` (`metadata=None`) and `GEOGRAPHY` with `google:sqlType:geography` and Well-Known Text (WKT) metadata `'{"encoding": "WKT"}'`. For `ARRAY<JSON>` and `ARRAY<GEOGRAPHY>`, BigQuery places the extension metadata on the outer Arrow `List` field rather than the inner `String` element. Polars 2.0+ loads unregistered Arrow extension types as `pl.Extension(...)` by default (`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_extension`). Mapping `JSON` and `GEOGRAPHY` to plain `pl.String` causes collecting `scan_table()` or `scan_query()` to fail Polars schema validation (`SchemaError`).
3.  **Unscannable mixed tables and stale `INTERVAL`/`BIGNUMERIC` mappings.** A single unsupported column fails the entire `scan_table()` call instead of allowing users to read the remaining supported columns, and `scan_query()` maps unsupported types inaccurately:
    *   `BIGNUMERIC` / `BIGDECIMAL` maps to `pl.Decimal(76, 38)`, exceeding Polars' maximum `Decimal` precision of `38` (raising `pl.exceptions.InvalidOperationError` when building a `LazyFrame`).
    *   `INTERVAL` previously mapped to `pl.Duration("us")`, but requires Arrow `Interval(MonthDayNano)`, which `arrow-bigquery` rejects with `UnsupportedTypeError` to avoid FFI failure.
    *   `RANGE<T>` has no Polars representation.
    *   Non-microsecond `TIMESTAMP` (`timestampPrecision` other than `6`, `"6"`, or unset — coarser or finer) does not match the connector's fixed microsecond `pl.Datetime("us", "UTC")` schema.
    *   Unrecognized future BigQuery data types or field modes would fail the entire scan unless omitted individually.
4.  **Dropped REST field attributes and inaccurate partition pseudo-columns.**
    *   When recursing into `mode="REPEATED"` (`ARRAY`), the extractor rebuilt a synthetic dict with only `name`, `type`, `fields`, and `mode`. Rebuilding the dictionary dropped `timestampPrecision` and other attributes, letting `ARRAY<TIMESTAMP>` with non-microsecond precision pass through unvalidated as `pl.List(pl.Datetime("us", "UTC"))`.
    *   The extractor unconditionally attached `_PARTITIONDATE` to every ingestion-time partitioned table (even when empty or non-daily). BigQuery exposes `_PARTITIONDATE` only on `DAY`-partitioned tables; `HOUR`, `MONTH`, and `YEAR` tables expose only `_PARTITIONTIME`.
5.  **Zero-column tables failing session creation.** BigQuery's `CreateReadSession` RPC rejects empty zero-column tables — **including** schemaless ingestion-time partitioned tables (`{"timePartitioning": {"type": "DAY"}}` with no `schema.fields`) — with `FAILED_PRECONDITION`. Previously, `scan_table()` raised `BigQueryError` or passed non-empty pseudo-columns into `CreateReadSession`.

## Goals

*   **Schema parity with Polars 2.0+ Arrow C Streams:** For supported BigQuery table and query columns matching the 14 `BQType` values (`BOOL`, `INT64`, `FLOAT64`, `NUMERIC`, `STRING`, `BYTES`, `DATE`, `DATETIME`, `TIME`, `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`), ensure `scan_table().collect_schema()` and `scan_query().collect_schema()` match BigQuery's Arrow IPC stream — subject to three intentional differences: (1) ingestion-time partition pseudo-columns are exposed by `scan_table()` but omitted by `read_table()`, (2) unsupported columns are skipped with a warning by `scan_table()` while `read_table()` raises `UnsupportedTypeError`, and (3) parameterized `NUMERIC(P, S)` on stored tables returns `Decimal(P, S)` in `read_table()` and is deferred under [issue #128](https://github.com/pola-rs/polars-bigquery-client/issues/128).
*   **Skip unsupported columns on mixed tables (`scan_table()`):** Omit top-level columns whose own type, mode, or nested subfield is unsupported (`BIGNUMERIC`, `INTERVAL`, `RANGE`, non-microsecond `TIMESTAMP`, or unrecognized future BigQuery types/modes), emitting one `UnsupportedTypeWarning` listing up to five omitted columns with dotted field paths (`col.sub.leaf`).
*   **Fail fast on unusable table schemas (`scan_table()`):** Raise `UnsupportedTypeError` without emitting a warning when a table with one or more defined columns has zero supported columns (and zero retained partition pseudo-columns) remaining after skipping.
*   **Forward-compatible REST parsing without defensive bloat:** Keep malformed top-level metadata envelope checks and the recursion depth cap (`64`, matching `crates/arrow-bigquery/src/decode.rs`) raising `BigQueryError`, without adding closed allowlists for `timePartitioning.type` or redundant type guards for payloads BigQuery does not emit. Gate `_PARTITIONDATE` on `timePartitioning.type` being `"DAY"` (or omitted/`""`/`None`), and retain `_PARTITIONTIME` for any other granularity without rejecting unknown future granularities.
*   **Strict extraction for queries (`scan_query()`):** Reject unsupported types/modes eagerly with `TypeError` (naming the top-level column) so verbatim SQL outputs are never omitted. Reject exact duplicate top-level columns and `STRUCT` subfields with `ValueError`, while still accepting empty `STRUCT([])`, single unnamed subfield `""`, and case-differing subfield names (`a` vs. `A`).
*   **Shared mapping tables and rank helper:** Share declarative BigQuery-to-Polars type-mapping tables and a single recursive dispatcher across table-scan and query-scan modes. Reuse `polars_bigquery.core.types.column_width_rank` inside `ReadApiExecutor` instead of maintaining a parallel width-rank table.
*   **Fail loudly on custom extension hooks:** Invoke extension dtype resolution directly in both `scan_table()` and `scan_query()` without catching exceptions from user-registered `pl.BaseExtension.ext_from_params(...)` hooks, so errors propagate with their original exception type and traceback.
*   **Preserve column attributes and accurate partition metadata:** Preserve extra REST keys across `REPEATED` recursion, validate `timestampPrecision`, prevent pseudo-columns from overwriting existing (including skipped) columns, skip appending pseudo-columns when `schema.fields` is empty, and expose `_PARTITIONDATE` only on daily ingestion-time partitioned tables.
*   **Zero-column table short-circuit (`schema=None`):** For discovered tables with zero defined physical columns (unpartitioned or ingestion-time partitioned), return a `(0, 0)` `pl.DataFrame` without invoking `CreateReadSession`, while enforcing `Client.close()` checks at collection time. Reject explicit empty `schema={}` eagerly with `ValueError` (matching `Client.scan_query()`).
*   **Consistent zero-column fallback ranking:** Select the narrowest physical column first using `column_width_rank`, breaking ties by schema order. When only ingestion-time pseudo-columns remain, apply the same width-then-order selector so `_PARTITIONDATE` (`DATE`, rank `2`) is selected before `_PARTITIONTIME` (`TIMESTAMP`, rank `3`) on daily tables.

## Non-Goals

*   **Plan-node `BQType` propagation (`TableScan.bq_types`, `QueryScan.bq_types`, `ExtractedSchema.bq_types`):** Zero-column fallback ranking yields identical ranks with or without `bq_types` for all 14 `BQType` values (`STRING`/`JSON`/`GEOGRAPHY` share rank `5`; `List(String)`/`ARRAY` share rank `6`). Defer adding `bq_types` propagation to `TableScan`/`QueryScan` until the schema-aware predicate pushdown PR (`core/compiler/`) actually consumes it.
*   **Public `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>` `dtypes` constants:** Defer exporting array extension constants in `polars_bigquery.dtypes` (YAGNI); document explicit outer-wrapped `pl.Extension(...)` syntax in `scan_table()` docstrings and [predicate-pushdown.md](../../docs/source/api/predicate-pushdown.md).
*   **Zero-column batch height workaround in Polars 2.0–2.1:** Keeping row height exact under `drop(pl.all()).collect_batches()` in `core/executors/base.py` and capping `max_stream_count=1` on small `n_rows` limits belong to a separate PR.
*   **Full support for deferred BigQuery types:** Arrow/Polars decoding for `BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, non-microsecond `TIMESTAMP`, and parameterized `NUMERIC(P, S)` precision preservation ([issue #128](https://github.com/pola-rs/polars-bigquery-client/issues/128)) are out of scope.
*   **Partial `STRUCT` pruning:** Extracting only supported subfields of a `STRUCT` containing an unsupported nested field is deferred; the entire enclosing top-level column is omitted.
*   **Dedicated API data-types page (`docs/source/api/data-types.md`):** Adding the new data-types reference page and updating `docs/source/api/index.md` are deferred to a separate docs PR. (Existing `docs/source/api/predicate-pushdown.md` *is* updated in this change so its mapping table stays accurate.)
*   **Backward compatibility for prior behavior:** Lowercase `"utc"`, plain `pl.String` for unregistered `JSON`/`GEOGRAPHY`, `pl.Duration("us")` for `INTERVAL`, `TypeError` for a single unsupported column in a mixed table, and unconditional `_PARTITIONDATE` on non-daily ingestion-time tables are not preserved.

## Open Questions

None. All decisions from multi-stage review have been resolved:

*   All-unsupported non-partitioned `scan_table()` raises **`UnsupportedTypeError`** (subclass of `BigQueryError`), aligning unsupported-type handling with `read_table()` and `read_query()`.
*   Explicit empty `schema={}` raises **`ValueError`** (matching `scan_query()`); `(0, 0)` short-circuiting applies only to discovered zero-column tables (`schema=None`).
*   `ReadApiExecutor` reuses **`core/types.py::column_width_rank`** (with unknown/unsupported dtypes and nested `BIGNUMERIC` moved to rank `7`, behind `ARRAY`/`STRUCT` at `6`).
*   `bq_types` propagation is **omitted** from `ExtractedSchema`, `TableScan`, and `QueryScan` until `core/compiler/` consumes it.
*   Public `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>` constants are **omitted** from `polars_bigquery.dtypes`.
*   Custom `BaseExtension.ext_from_params` exceptions and non-`DataType` return values propagate directly in both `scan_table()` and `scan_query()` (verified offline in Polars `2.1.0`: `pl.Schema` / `pl.Field` fail fast with `TypeError: cannot parse input of type '<type>' into Polars data type`).
*   Pseudo-column fallback applies the same `(rank, schema_index)` rule as physical columns, selecting `_PARTITIONDATE` over `_PARTITIONTIME` on daily tables.

## Critical User Journeys (CUJs)

### CUJ 1: Scanning a table with `TIMESTAMP`, `JSON`, `GEOGRAPHY`, and array variants

*   **Given:** A BigQuery table containing `TIMESTAMP`, `JSON`, `GEOGRAPHY`, `ARRAY<JSON>`, and `ARRAY<GEOGRAPHY>` columns and default Polars 2.0+ extension settings (`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` unset).
*   **When:** A user runs `lf = client.scan_table("project.dataset.tbl")` and collects with `lf.collect()`.
*   **Then:** `lf.collect_schema()` reports `pl.Datetime("us", "UTC")` for `TIMESTAMP`, `polars_bigquery.dtypes.JSON` / `GEOGRAPHY` for scalar extensions, and outer-wrapped `pl.Extension(..., pl.List(pl.String()))` for `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>`. Collecting the Arrow C Stream succeeds without `SchemaError`, and the resulting `DataFrame` schema matches `read_table().schema`.

### CUJ 2: Scanning a mixed table containing an unsupported column

*   **Given:** A BigQuery table with supported column `id INT64`, top-level `dur INTERVAL`, and nested `meta STRUCT<ok INT64, bad BIGNUMERIC>`.
*   **When:** A user runs `client.scan_table("project.dataset.mixed")`.
*   **Then:** The client omits `dur` and `meta` from the returned `LazyFrame` schema (retaining `id`), and emits one `UnsupportedTypeWarning` (at the caller's stack frame) naming up to five skipped columns with nested paths (`dur (INTERVAL)`, `meta (BIGNUMERIC at meta.bad)`), directing the user to cast or exclude those columns in SQL using `client.scan_query()` or `client.read_query()`.

### CUJ 3: Scanning a table with zero supported physical columns

*   **Given:** A BigQuery table with one or more defined columns whose physical columns are all unsupported (for example, `INTERVAL` and `BIGNUMERIC`).
*   **When:** A user runs `client.scan_table("project.dataset.all_bad")`.
*   **Then:**
    *   If the table is **not** ingestion-time partitioned, `scan_table()` raises `UnsupportedTypeError` listing the skipped columns and does **not** emit `UnsupportedTypeWarning`.
    *   If the table **is** ingestion-time partitioned, `scan_table()` emits `UnsupportedTypeWarning` and returns a `LazyFrame` exposing only the valid pseudo-columns (`_PARTITIONDATE` and `_PARTITIONTIME` for `DAY`, or `_PARTITIONTIME` only for `HOUR`, `MONTH`, and `YEAR`).

### CUJ 4: Scanning a discovered zero-column table

*   **Given:** A BigQuery table whose REST metadata defines zero physical columns (`schema` absent, `None`, or `fields: []` — whether unpartitioned or ingestion-time partitioned).
*   **When:** A user builds `lf = client.scan_table("project.dataset.empty")` and calls `lf.collect()` (before or after `client.close()`).
*   **Then:** Before `client.close()`, `lf.collect()` returns a `(0, 0)` `pl.DataFrame` without issuing a `CreateReadSession` call. After `client.close()`, `lf.collect()` raises `RuntimeError`. Passing an explicit empty `schema={}` raises `ValueError` eagerly (matching `client.scan_query()`).

### CUJ 5: Providing an explicit `schema=` for `JSON`, `GEOGRAPHY`, or `TIMESTAMP`

*   **Given:** A user wants to skip the `tables.get` REST call and supplies an explicit non-empty `schema=` matching their active Polars Arrow extension mode (`dtypes.JSON` / `dtypes.GEOGRAPHY` by default, their registered `BaseExtension` subclass, or `pl.String()` when loaded as storage). For `ARRAY<JSON>`, callers pass the outer-wrapped `pl.Extension("google:sqlType:json", pl.List(pl.String()))` (not `pl.List(dtypes.JSON)`).
*   **When:** The user collects `lf.collect()`.
*   **Then:** Matching extension schemas succeed through Arrow C Stream import without calling `tables.get`. Supplying plain `pl.String()` for an unregistered `JSON` column under default mode fails with Polars' `SchemaError` naming `google:sqlType:json`.

### CUJ 6: Running `scan_query()` against a query selecting an unsupported type

*   **Given:** A SQL query whose dry-run schema contains `BIGNUMERIC`, `INTERVAL`, `RANGE`, non-microsecond `TIMESTAMP`, or an unrecognized type/mode (including inside `ARRAY` or `STRUCT`).
*   **When:** A user runs `client.scan_query(sql)`.
*   **Then:** `scan_query()` fails immediately with `TypeError` naming the top-level column, without executing the query or silently dropping columns.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### High-Level Data Flow

```text
Client.scan_table(table, schema=None)
  |-- schema is not None (explicit)
  |     -> pl.Schema(schema), require non-empty (empty -> ValueError)
  |     -> TableScan -> ReadApiExecutor -> Arrow C Stream import
  |
  +-- schema is None (REST discovery)
        -> BigQuery REST tables.get (JSON metadata)
        -> extract_schema_metadata(table_metadata)
             |-- per field: recursive dispatcher (strict=False)
             |-- if fields non-empty & ingestion-time partitioned:
             |     append _PARTITIONDATE (DAY/default only) & _PARTITIONTIME
             v
           ExtractedSchema(polars_schema, skipped_columns)
             |-- skipped_columns & len == 0 -> raise UnsupportedTypeError
             |-- skipped_columns & len > 0  -> warn UnsupportedTypeWarning
             |-- no skipped & len == 0      -> yield (0, 0) DataFrame
             |                                 (no CreateReadSession RPC)
             +-- len > 0 -> TableScan -> ReadApiExecutor
                              -> 0-col fallback: column_width_rank(dtype)
                              -> arrow_client.read_table(selected_fields=...)

Client.scan_query(query, schema=None)
  |-- schema is not None (explicit)
  |     -> pl.Schema(schema), require non-empty (empty -> ValueError)
  |     -> QueryScan -> QueryExecutor
  |
  +-- schema is None (dry run)
        -> jobs.query(dryRun=True) -> JSON bigquerySchema.fields
        -> extract_polars_schema_from_fields(fields) [strict=True]
        -> QueryScan -> QueryExecutor
```

### Schema Mapping Contract

| BigQuery REST `type` (casefolded) | REST `mode` | Polars `DataType` | Dispatch path |
| --- | --- | --- | --- |
| `bool`, `boolean` | non-`REPEATED` | `pl.Boolean()` | scalar table |
| `bytes` | non-`REPEATED` | `pl.Binary()` | scalar table |
| `date` | non-`REPEATED` | `pl.Date()` | scalar table |
| `datetime` | non-`REPEATED` | `pl.Datetime(time_unit="us")` | scalar table |
| `float`, `float64` | non-`REPEATED` | `pl.Float64()` | scalar table |
| `integer`, `int64` | non-`REPEATED` | `pl.Int64()` | scalar table |
| `numeric`, `decimal` | non-`REPEATED` | `pl.Decimal(precision=38, scale=9)` | scalar table |
| `string` | non-`REPEATED` | `pl.String()` | scalar table |
| `time` | non-`REPEATED` | `pl.Time()` | scalar table |
| `json` | non-`REPEATED` | `build_extension_dtype("google:sqlType:json", pl.String(), None)` | extension table |
| `json` | `REPEATED` | `build_extension_dtype("google:sqlType:json", pl.List(pl.String()), None)` | extension table |
| `geography` | non-`REPEATED` | `build_extension_dtype("google:sqlType:geography", pl.String(), '{"encoding": "WKT"}')` | extension table |
| `geography` | `REPEATED` | `build_extension_dtype("google:sqlType:geography", pl.List(pl.String()), '{"encoding": "WKT"}')` | extension table |
| `timestamp` (`timestampPrecision` in `None`, `6`, `"6"`) | non-`REPEATED` | `pl.Datetime(time_unit="us", time_zone="UTC")` | timestamp branch |
| `record`, `struct` | non-`REPEATED` | `pl.Struct([...])` | unified struct helper |
| Any other supported type | `REPEATED` | `pl.List(inner_dtype)` | recursive branch |
| `bignumeric`, `bigdecimal` | any | Unsupported (`BIGNUMERIC`) | unsupported table |
| `interval` | any | Unsupported (`INTERVAL`) | unsupported table |
| `range` | any | Unsupported (`RANGE`) | unsupported table |
| `timestamp` (`timestampPrecision` not in `None`, `6`, `"6"`) | any | Unsupported (`TIMESTAMP(precision=...)`) | timestamp branch |
| Unrecognized `mode` (not in `""`, `nullable`, `required`, `repeated`) | any | Unsupported (`unsupported mode ...`) | mode check |
| Any unrecognized `type` | any | Unsupported (`unsupported type ...`) | fallback branch |

*Notes:* **`non-REPEATED`** means `mode` is `"NULLABLE"`, `"REQUIRED"`, `""`, `None`, or omitted. Scalar lookups use callables so every column receives a fresh `pl.DataType` instance. When recursing for `mode="REPEATED"`, field attributes (`timestampPrecision`, `fields`) are preserved so `ARRAY<TIMESTAMP>` with non-microsecond precision is rejected rather than misclassified.

### Extension Dtype Resolution Across Polars Configurations

`polars_bigquery.core.types.build_extension_dtype(name, storage, metadata)` resolves `JSON` (`google:sqlType:json`, `metadata=None`) and `GEOGRAPHY` (`google:sqlType:geography`, `metadata='{"encoding": "WKT"}'`) according to the active Polars configuration:

| Configuration | Scalar `JSON` in `pl.Schema` | `ARRAY<JSON>` in `pl.Schema` |
| --- | --- | --- |
| Default (`load_as_extension`) | `pl.Extension("google:sqlType:json", pl.String(), None)` | `pl.Extension("google:sqlType:json", pl.List(pl.String()), None)` |
| Registered `pl.BaseExtension` subclass | `MyExt("google:sqlType:json", pl.String(), None)` | `MyExt("google:sqlType:json", pl.List(pl.String()), None)` |
| `register_extension_type(..., as_storage=True)` or `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage` | `pl.String()` | `pl.List(pl.String())` |

*(Rows for `GEOGRAPHY` follow the same rules with metadata `'{"encoding": "WKT"}'`. Custom `BaseExtension` subclasses registered for these extension names receive **both** `pl.String()` for scalars and `pl.List(pl.String())` for `ARRAY<...>` inside `ext_from_params`.)*

**Purity / env-var latch caveat:** Schema extraction is pure given a deterministic injected `extension_dtype_builder`. The default `build_extension_dtype` reads Polars' global extension registry and compares `os.environ` against its module-import latch of `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR`, logging a warning if mutated after import. Set `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` before importing `polars` or `polars_bigquery` (or prefer `pl.register_extension_type(..., as_storage=True)`).

### Why `scan_query()` Is Strict While `scan_table()` Skips Columns

1.  **Server projection limitation:** `scan_table()` passes an explicit `selected_fields=[...]` allowlist to the BigQuery Storage Read API, so omitted columns are never encoded or streamed over the wire. By contrast, `scan_query()` executes verbatim SQL by default (`maintain_order=True`) and falls back to verbatim SQL when GoogleSQL pipe rewrites (`|> SELECT` / `|> WHERE` / `|> LIMIT`) exceed BigQuery's ~1 MB query-size limit under `maintain_order=False` (see [`polars_bigquery_scan_query.md`](./polars_bigquery_scan_query.md)). Omitting a column from `pl.Schema` without removing it from SQL would still stream that column and crash at `.collect()` time with `UnsupportedTypeError` or Polars `SchemaError`.
2.  **Caller control:** With `scan_table()`, a caller often reads a shared table they do not own and wants the supported columns. With `scan_query()`, the caller authored the `SELECT` list directly and can drop or cast columns in SQL.

Both modes share the same unified `STRUCT` helper. Exact duplicate top-level columns (in `scan_query()`) and exact duplicate `STRUCT` subfield names raise `ValueError`. Non-duplicate empty `STRUCT([])`, single unnamed subfield (`""`), and case-differing names (`a` vs. `A`) are accepted.

### Ingestion-Time Partitioning Pseudo-Columns

For ingestion-time partitioned tables (`timePartitioning` present with absent or `""` `field`) that have **at least one physical column in `schema.fields`**:

*   Append `_PARTITIONDATE` (`pl.Date()`) only when `timePartitioning.type` is `"DAY"` (case-insensitive), `None`, `""`, or omitted — never when already present in the table schema (including skipped columns, matched case-insensitively).
*   Append `_PARTITIONTIME` (`pl.Datetime("us", "UTC")`) for any granularity (unless already present, including skipped columns). No closed allowlist rejects unknown future `timePartitioning.type` values (`WEEK`, `MINUTE`, etc.).
*   When `schema.fields` is empty or absent (a schemaless zero-column partitioned table), **do not** append pseudo-columns so the table short-circuits cleanly to `(0, 0)`.

### Exception & Warning Contract

| Failure Condition | `scan_table()` | `scan_query()` | `read_table()` / `read_query()` |
| --- | --- | --- | --- |
| Mixed table with >=1 supported column (or partition pseudo-column) and >=1 unsupported column | Emits `UnsupportedTypeWarning` (lists up to 5 entries); omits unsupported column(s) | `TypeError` naming the top-level column | `UnsupportedTypeError` (`BigQueryError` subclass) |
| All columns unsupported (non-partitioned table) | **`UnsupportedTypeError`** (no warning emitted) | `TypeError` | `UnsupportedTypeError` |
| Explicit empty `schema={}` | `ValueError` | `ValueError` | — |
| Explicit `schema=` containing `Decimal(precision > 38)` | `TypeError` | `TypeError` | — |
| Exact duplicate column / `STRUCT` subfield names | `ValueError` (`STRUCT`) | `ValueError` | — |
| Malformed outer REST metadata envelope or nesting depth > `64` | `BigQueryError` | `BigQueryError` | `BigQueryError` |
| User-registered `BaseExtension.ext_from_params(...)` error / non-`DataType` return | Propagates original exception (e.g., `TypeError`) | Propagates original exception | Propagates original exception |

### Zero-Column Fallback Column Selection (`ReadApiExecutor`)

When Polars pushes down a zero-column projection (`selected_fields=[]` would instruct the Storage Read API to stream *every* column, including skipped unsupported ones), `ReadApiExecutor` chooses a single narrowest column via **`core/types.py::column_width_rank`**:

*   `1`: `BOOL`
*   `2`: `DATE` (`_PARTITIONDATE`)
*   `3`: `INT64`, `FLOAT64`, `TIME`, `DATETIME`, `TIMESTAMP` (`_PARTITIONTIME`)
*   `4`: `NUMERIC` (`Decimal` within precision 38, scale 9)
*   `5`: `STRING`, `BYTES`, `JSON`, `GEOGRAPHY`
*   `6`: `ARRAY` (`List`), `STRUCT`
*   `7`: unknown/unsupported dtypes (`Duration`, `Float16`, `BIGNUMERIC` / nested `BIGNUMERIC`, unparameterized `Decimal(precision=None)`, `Null`, `Unknown`, `Object`)

Custom non-BigQuery `pl.BaseExtension` types unwrap `ext_storage()` before ranking. Physical columns always take priority over ingestion-time pseudo-columns so explicit schemas and mixed tables read a concrete column first. When only pseudo-columns remain (all physical columns were omitted as unsupported), the same `(rank, schema_index)` rule picks `_PARTITIONDATE` (`DATE`, rank `2`) ahead of `_PARTITIONTIME` (`TIMESTAMP`, rank `3`) on daily tables, and `_PARTITIONTIME` on non-daily tables.

### Verified Live BigQuery Wire-Format Probes (2026-11-24)

Verified live against project `bigframes-dev` using `polars 2.1.0`:

| Probe | API | Observed Wire / Schema Output | Result |
| --- | --- | --- | --- |
| `JSON`, `GEOGRAPHY`, `ARRAY<JSON>`, `ARRAY<GEOGRAPHY>`, `STRUCT<...>`, `ARRAY<STRUCT<...>>` | `jobs.query` (`read_query` / `scan_query`) & Storage Read API (`read_table` / `scan_table`) | Identical extension tags on both paths: `google:sqlType:json` (`meta=None`), `google:sqlType:geography` (`meta='{"encoding": "WKT"}'`), outer `List(String)` wrapping for `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>` including inside `STRUCT`. | Both collected (`(1, 6)` and `(1, 7)`) |
| `DAY` table with single column `(dur INTERVAL)` | Storage Read API (`selected_fields`) | `["_PARTITIONDATE"]` → `(2, 1) Date`; `["_PARTITIONTIME"]` → `(2, 1) Datetime('us', 'UTC')`; `scan_table().collect()` → `(2, 2)` + `UnsupportedTypeWarning`. | Confirmed CUJ 3 |
| `HOUR` table with single column `(dur INTERVAL)` | Storage Read API (`selected_fields`) | `["_PARTITIONTIME"]` → `(1, 1) Datetime('us', 'UTC')`; `["_PARTITIONDATE"]` → `INVALID_ARGUMENT` (`Name _PARTITIONDATE not found`). | Confirmed `DAY`-only gate |
| Empty 0-column `DAY` table (`{"timePartitioning": {"type": "DAY"}}`, no `schema`) | Storage Read API | `CreateReadSession` **failed** with `FAILED_PRECONDITION` (`empty schema`) even when selecting `["_PARTITIONDATE"]` or `["_PARTITIONTIME"]`. | Confirmed skipping pseudo-columns on empty `schema.fields` |
| Empty 0-column unpartitioned table (`{}`) | `scan_table()` | Short-circuits without `CreateReadSession` to `(0, 0) pl.DataFrame`. | Matches CUJ 4 |
| Parameterized `NUMERIC(10, 2)` table column | `tables.get` + Storage Read API vs `jobs.query` | `read_table` emits **`Decimal(10, 2)`** (`scan_table()` fails `SchemaError` against `Decimal(38, 9)`); `read_query` / dry-run emits **`Decimal(38, 9)`**. | Deferred to [issue #128](https://github.com/pola-rs/polars-bigquery-client/issues/128) |
| `SELECT STRUCT(1)`, `STRUCT(1, 2)`, `STRUCT(1 AS a, 'x' AS A)` | `jobs.query` dry run + Arrow IPC | BigQuery auto-synthesizes `_field_1` / `_field_2` (and renames case-duplicate `A` to `_field_2`) in both `bigquerySchema` JSON and Arrow IPC. | Collected (`(1, 1)`) |

### Backward-Incompatible Behavioral Changes

| Case | Previous behavior (`32fdc1f`) | New behavior |
| --- | --- | --- |
| `TIMESTAMP` / `_PARTITIONTIME` timezone (`scan_table`, `scan_query`) | `pl.Datetime("us", "utc")` | `pl.Datetime("us", "UTC")` |
| `JSON` / `GEOGRAPHY` default dtype (`scan_table`, `scan_query`) | `pl.String()` | `pl.Extension("google:sqlType:json", pl.String())` / `pl.Extension("google:sqlType:geography", pl.String(), '{"encoding": "WKT"}')` |
| `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>` default dtype (`scan_table`, `scan_query`) | `pl.List(pl.String())` | `pl.Extension("google:sqlType:json", pl.List(pl.String()))` / `pl.Extension("google:sqlType:geography", pl.List(pl.String()), '{"encoding": "WKT"}')` |
| Mixed table with an unsupported column (`scan_table`) | Fails with `TypeError` or Polars `InvalidOperationError` (`Decimal(76, 38)`) | Skips enclosing top-level column with `UnsupportedTypeWarning` |
| Non-partitioned table with *only* unsupported columns (`scan_table`) | Raises `TypeError` / `InvalidOperationError` | Raises `UnsupportedTypeError` (no warning emitted) |
| Ingestion-time partitioned table with *only* unsupported physical columns (`scan_table`) | Raises `TypeError` / `InvalidOperationError` | Emits `UnsupportedTypeWarning`; retains `_PARTITIONDATE`/`_PARTITIONTIME` (`DAY`) or `_PARTITIONTIME` (non-`DAY`) |
| Non-`DAY` (`HOUR`/`MONTH`/`YEAR`) ingestion-time table (`scan_table`) | Exposed `_PARTITIONDATE` and `_PARTITIONTIME` | Exposes `_PARTITIONTIME` only |
| Discovered 0-column table (`scan_table`, `schema=None`) | Raises `BigQueryError` (or `FAILED_PRECONDITION`) | Short-circuits to `(0, 0)` `pl.DataFrame` without `CreateReadSession` |
| `INTERVAL` / `RANGE` / `BIGNUMERIC` / non-microsecond `TIMESTAMP` (`scan_query`) | `INTERVAL` → `pl.Duration("us")`; `BIGNUMERIC` → `pl.Decimal(76, 38)` | Raises `TypeError` naming the top-level column |

--------------------------------------------------------------------------------

## Alternatives Considered

### Alternative A: Support Both Polars 1.x and 2.x Dynamically for `JSON` and `GEOGRAPHY`

*   **Pros:** Keeps Polars 1.x installations working without requiring `polars >= 2.0.0`.
*   **Cons:** Requires version-gated schema logic, separate documentation, and a split CI matrix.
*   **Reason for Rejection:** Polars 2.x is the target runtime (`polars >= 2.0.0, < 3.0.0` in `py-polars-bigquery/pyproject.toml`).

### Alternative B: Unwrap `JSON` and `GEOGRAPHY` to Plain `pl.String` Inside `scan_table()`

*   **Pros:** Avoids exposing Polars' unstable `pl.Extension` dtype by default and lets callers invoke `.str.*` directly without `.ext.storage()`.
*   **Cons:** Makes `scan_table().collect_schema()` disagree with `read_table().schema` and `read_query().schema`, erases BigQuery type identity, and causes invalid SQL filter pushdown on `JSON`/`GEOGRAPHY`.
*   **Reason for Rejection:** Keeping Polars 2.0+ native Arrow extension dtypes preserves parity across `read_table()`, `read_query()`, `scan_table()`, and `scan_query()`.

### Alternative C: Fail Fast on Any Unsupported Column in `scan_table()`, or Strip Nested `STRUCT` Subfields

*   **Fail fast (`Pros`/`Cons`):** Keeps `collect_schema()` equal to table DDL, but blocks scanning an entire table when one unused `INTERVAL`, `BIGNUMERIC`, `RANGE`, or non-microsecond `TIMESTAMP` column exists. Callers wanting fail-fast behavior can promote `UnsupportedTypeWarning` via `warnings.filterwarnings("error", category=UnsupportedTypeWarning)`.
*   **Strip nested `STRUCT` subfields (`Pros`/`Cons`):** Keeps sibling `STRUCT` subfields readable, but the Storage Read API projects by top-level column name, requiring nested projection and client-side `STRUCT` reconstruction.
*   **Reason for Rejection:** Omitting the enclosing top-level column with `UnsupportedTypeWarning` unblocks mixed tables without implementing nested projection or client-side reconstruction.

### Alternative D: Skip Unsupported Columns with a Warning in `scan_query()`

*   **Pros:** Uniform warning behavior across `scan_table()` and `scan_query()`.
*   **Cons:** Under default `maintain_order=True` (and on oversized pipe-rewrite fallback under `maintain_order=False`), `scan_query()` runs verbatim SQL without `|> SELECT`. BigQuery still encodes and streams the unsupported column, crashing at `.collect()` with `UnsupportedTypeError` or Polars `SchemaError`.
*   **Reason for Rejection:** Fail-fast `TypeError` inside `scan_query()` stops queries before execution and billing and avoids dropping columns the user explicitly requested.

### Alternative E: Catch Custom `BaseExtension.ext_from_params` Errors and Convert Them into `SkippedColumn` Entries

*   **Pros:** A scalar-only custom `pl.BaseExtension` subclass whose `ext_from_params` rejects `pl.List(pl.String())` would skip `ARRAY<JSON>` columns during `scan_table()`.
*   **Cons:** Catching broad exceptions masks real defects inside third-party extension hooks as a dropped-column warning and diverges from `read_table()` and `scan_query()`, which raise immediately.
*   **Reason for Rejection:** Propagating exceptions directly surfaces defects uniformly across `read_table()`, `scan_table()`, and `scan_query()`.

### Alternative F: Short-Circuit Explicit `schema={}` to `(0, 0)` in `scan_table()`

*   **Pros:** Uniform handling between explicit `schema={}` and discovered zero-column tables (`schema=None`).
*   **Cons:** Silently yields `0` rows on non-empty tables (`lf.select(pl.len()).collect()` → `0`), skips table-existence and IAM permission checks, masks caller bugs, and contradicts `scan_query(..., schema={})` (`ValueError`, specified in `polars_bigquery_scan_query.md`).
*   **Reason for Rejection:** Rejecting explicit `schema={}` with `ValueError` keeps `scan_table()` and `scan_query()` consistent; only metadata-discovered zero-column tables short-circuit.

### Alternative G: Pre-Flight Schema Interception on Explicit `schema=` Mismatches

*   **Pros:** Could customize Polars' error when a caller supplies `pl.String()` for a `JSON` column.
*   **Cons:** Requires adding a new `__arrow_c_schema__` export to `arrow_bigquery` and maintaining a duplicate Python schema comparator ahead of Polars' Rust importer.
*   **Reason for Rejection:** Polars' native `SchemaError` already reports both expected and actual dtypes, including the extension name (`google:sqlType:json` / `google:sqlType:geography`).

*Smaller design choices kept declarative rather than parallel:* One recursive dispatcher with lookup tables (rather than duplicated `if` chains for table vs. query modes), unified `core/types.py::column_width_rank` (rather than a separate rank map in `read_api.py`), and width-first ranking of pseudo-columns (`_PARTITIONDATE` over `_PARTITIONTIME` on daily tables).

--------------------------------------------------------------------------------

## Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| `polars-bigquery` wheel installed with older `arrow-bigquery == 0.3.2` (missing `#116` and `#123`) | Every `DATETIME` scan fails with `SchemaError` (`google:sqlType:datetime`) | Release `arrow-bigquery` first and bump `arrow-bigquery >= <next-version>` in `py-polars-bigquery/pyproject.toml` (rollback reverses both versions together). |
| Interim filter pushdown under `load_as_storage` before `core/compiler/` PR lands | String ops pushed on `JSON`/`GEOGRAPHY` columns fail in BigQuery (`INVALID_ARGUMENT`) | Document behavior and scope `core/compiler/` schema awareness as the follow-up PR. |
| Parameterized `NUMERIC(P, S)` columns on `scan_table()` | `read_table` returns `Decimal(P, S)` while `scan_table()` declares `Decimal(38, 9)` (`SchemaError`) | Documented exclusion; tracked in [issue #128](https://github.com/pola-rs/polars-bigquery-client/issues/128). Workaround: pass explicit `schema=` with `pl.Decimal(P, S)`. |
| Time-of-check/time-of-use drift between scan construction and `.collect()` | Table DDL changes or mid-process `pl.register_extension_type()` mutations raise `SchemaError` | Set `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` before importing `polars` and avoid mutating the global extension registry between `scan_table()` and `.collect()`. |
| Future BigQuery `timePartitioning.type` lacking `_PARTITIONTIME` | Calling `scan_table()` includes `_PARTITIONTIME` in schema | Only emitted when `timePartitioning.field` is empty; no closed allowlist breaks reads of physical columns. |
| Scalar-only custom `BaseExtension` on tables containing `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>` | `scan_table()` raises immediately instead of skipping `ARRAY<...>` | Document that registered `BaseExtension` subclasses for `JSON`/`GEOGRAPHY` must handle both `pl.String()` and `pl.List(pl.String())`. |

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Release dependency order:** `arrow-bigquery` must release with `#116` (`UnsupportedTypeError`) and `#123` (`google:sqlType:datetime` stripping) before releasing `polars-bigquery`. `py-polars-bigquery/pyproject.toml` pins the floor alongside the release note already tracked on lines 15–17.
*   **Privacy & logging:** `UnsupportedTypeWarning` messages and `DEBUG` logs record table IDs, column names, dotted schema paths, and BigQuery type names only (`timestampPrecision` uses `repr` formatting so empty/control-character values render safely). No row data or predicate literal values are logged. Warning summaries cap output at 5 entries (`"... and N more"`). Note that Python's `warnings` module writes to `stderr` by default and suppresses duplicate alerts at the same call site.
*   **Network & cost impact:**
    *   Short-circuiting discovered zero-column tables (including schemaless ingestion-time partitioned tables) avoids a failing `CreateReadSession` RPC.
    *   Zero-column fallback requests a single narrowest column (`column_width_rank`) rather than `selected_fields=[]`, preventing full-table scans over the Storage Read API. (Uncompressed Arrow width: `_PARTITIONDATE` is 4 bytes vs. `_PARTITIONTIME` 8 bytes; billing/compressed wire impact is negligible since both are per-partition metadata columns.)
*   **Rollout & backward compatibility:** Callers needing strict failure on mixed tables can set `warnings.filterwarnings("error", category=UnsupportedTypeWarning)` and catch `(UnsupportedTypeWarning, UnsupportedTypeError)`. See [Backward-Incompatible Behavioral Changes](#backward-incompatible-behavioral-changes).

--------------------------------------------------------------------------------

## Future Improvements

*   **Parameterized `NUMERIC(P, S)` support:** Tracked in [issue #128](https://github.com/pola-rs/polars-bigquery-client/issues/128) (`tables.get` includes `"precision"` and `"scale"` strings for `NUMERIC(P, S)` columns).
*   **Partial `STRUCT` subfield exclusion:** Support omitting only unsupported nested subfields of a `STRUCT` and reconstructing remaining subfields client-side once nested field projection is wired in `arrow-bigquery`.
*   **Schema-aware predicate pushdown:** Introduce BigQuery type propagation on plan nodes and consume it inside `core/compiler/` to reject invalid GoogleSQL expressions on `JSON`/`GEOGRAPHY` under `load_as_storage`, bitwise operators, `NaN`/`Inf`, and cross-temporal comparisons.
*   **Dedicated API data-types documentation:** Publish `docs/source/api/data-types.md` and register `UnsupportedTypeWarning`, `UnsupportedTypeError`, and `polars_bigquery.dtypes` inside `docs/source/api/index.md`.
*   **Zero-column row-height preservation (Polars 2.0–2.1):** Avoid batch collapse during empty zero-column projections in `core/executors/base.py`.
*   **Non-destructive `__arrow_c_schema__` on `arrow_bigquery.ArrowStreamExporter`:** Expose Arrow PyCapsule schema export without consuming the single-use stream for ecosystem interop (`pyarrow`, `arro3`, DuckDB).

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Each entry lists the required changes for a target file (Markdown links resolve relative to `docs/designs/`). All source and test updates below are relative to the pre-change baseline (`main` at commit `32fdc1f`).

### Target Files to Modify

#### 1. [`py-polars-bigquery/polars_bigquery/core/types.py`](../../py-polars-bigquery/polars_bigquery/core/types.py) (Modify)

*   **Type / Scope:** Canonical BigQuery type & width-rank helper module.
*   **Rationale:** Avoids duplicating a second 14-type width-rank lookup table in `read_api.py`.
*   **Required Changes:**
    *   Update `column_width_rank(dtype, *, bq_type=None) -> int` so unknown/unsupported dtypes (such as `pl.Duration`, `pl.Float16`, `pl.Decimal(76, 38)`, unparameterized `pl.Decimal` with `precision=None`, `pl.Null`, `pl.Unknown`, `pl.Object`) return rank `7` (strictly behind `ARRAY`/`STRUCT` at `6`).
    *   For custom non-BigQuery `pl.BaseExtension` types (`resolved is None`), unwrap `dtype.ext_storage()` and recurse.

#### 2. [`py-polars-bigquery/polars_bigquery/core/schema.py`](../../py-polars-bigquery/polars_bigquery/core/schema.py) (Modify)

*   **Type / Scope:** Schema-extraction module.
*   **Rationale:** Centralizes BigQuery-to-Polars schema mapping across `scan_table()` (skip/warn on unsupported types) and `scan_query()` (strict extraction) using one set of lookup tables and one unified `STRUCT` helper.
*   **Required Changes:**
    *   Define data structures and type alias:
        *   `SkippedColumn(name: str, reason: str, path: str)` (frozen dataclass).
        *   `ExtractedSchema(polars_schema: pl.Schema, skipped_columns: tuple[SkippedColumn, ...])` (frozen dataclass).
        *   `_UnsupportedBigQueryType(reason: str, path: str)` (internal `Exception` subclass).
        *   `_ExtensionTypeSpec(extension_name: str, metadata: str | None)` (internal frozen, slotted dataclass).
        *   `ExtensionDtypeBuilder = Callable[[str, pl.DataType, str | None], pl.DataType]`.
    *   Define module-level lookup tables (reusing `BQ_NUMERIC_MAX_PRECISION` and `BQ_NUMERIC_MAX_SCALE` from `polars_bigquery.core.types`):
        *   `_SCALAR_TYPE_MAP`: map casefolded type names and aliases (`bool`, `boolean`, `bytes`, `date`, `datetime`, `float`, `float64`, `integer`, `int64`, `numeric`, `decimal`, `string`, `time`) to zero-arg callables returning fresh `pl.DataType` instances.
        *   `_EXTENSION_TYPE_MAP`: map `json` and `geography` to `_ExtensionTypeSpec` instances (`google:sqlType:json`, `metadata=None`; `google:sqlType:geography`, `metadata='{"encoding": "WKT"}'`).
        *   `_UNSUPPORTED_TYPE_MAP`: map `bignumeric`/`bigdecimal`, `interval`, and `range` to `(skip_reason, strict_message_template)` tuples.
    *   Define recursive `_extract_field_dtype(field, *, path, strict, top_col_name="", depth=0, extension_dtype_builder=build_extension_dtype) -> pl.DataType`:
        *   Raise `BigQueryError` when `depth > MAX_SCHEMA_NESTING_DEPTH` (`64`).
        *   Validate casefolded `mode` against `_VALID_FIELD_MODES = frozenset({"", "nullable", "required", "repeated"})` (skip in `scan_table()`, `TypeError` in `scan_query()`).
        *   Handle `mode == "repeated"` first: for extension types, invoke `extension_dtype_builder(spec.extension_name, pl.List(pl.String()), spec.metadata)` directly (no `try/except`); otherwise recurse on `{**field, "mode": "NULLABLE"}` to preserve `timestampPrecision` and nested attributes.
        *   Map `timestamp` to `pl.Datetime(time_unit="us", time_zone="UTC")`; validate `timestampPrecision` via `_is_supported_timestamp_precision(precision)` (`None`, `int` `6` excluding `bool`, or `"6"`). Format non-microsecond errors with `{precision!r}`.
        *   Dispatch `record`/`struct` to one unified `_extract_struct_dtype` helper supporting empty `STRUCT([])`, single unnamed subfield `""`, and case-differing subfield names while raising `ValueError` on exact duplicate subfield names.
        *   For unrecognized `type`, raise `_UnsupportedBigQueryType` (`strict=False`) or `TypeError` naming `top_col_name` (`strict=True`).
    *   In `extract_schema_metadata(table_metadata, *, extension_dtype_builder=build_extension_dtype) -> ExtractedSchema`:
        *   Validate outer envelope (`table_metadata`, `schema`, `schema.fields`, `timePartitioning`), raising `BigQueryError` directly on malformed non-mapping/non-sequence envelopes (allowing `None`).
        *   Record each column's casefolded `name` in `seen_names` before evaluating its type so a skipped column prevents pseudo-column shadowing.
        *   Gate ingestion-time partition pseudo-columns on **non-empty `fields`** (`if fields and time_partitioning is not None:`) so schemaless zero-column partitioned tables return `pl.Schema({})`. Append `_PARTITIONDATE` (`pl.Date()`) only when `type` is `None`, `""`, or `"day"` (case-insensitive), and `_PARTITIONTIME` (`pl.Datetime(time_unit="us", time_zone="UTC")`) for any granularity without rejecting unrecognized future `timePartitioning.type` values. Skip either pseudo-column if its casefolded name is already in `seen_names`.
    *   Retain `extract_polars_schema(table_metadata, *, extension_dtype_builder=build_extension_dtype, require_non_empty=False) -> pl.Schema` delegating to `extract_schema_metadata()`.
    *   Update `has_bignumeric_dtype(dtype)` to guard `current.precision is not None and current.precision > BQ_NUMERIC_MAX_PRECISION` and unwrap `pl.BaseExtension` (`current.ext_storage()`).

#### 3. [`py-polars-bigquery/polars_bigquery/exceptions.py`](../../py-polars-bigquery/polars_bigquery/exceptions.py) (Modify)

*   **Type / Scope:** Public exceptions and warnings module.
*   **Rationale:** Exposes `UnsupportedTypeWarning` so callers can capture, ignore, or promote skipped-column warnings with Python's `warnings` module.
*   **Required Changes:**
    *   Define `UnsupportedTypeWarning(UserWarning)` with a docstring listing omitted categories (`BIGNUMERIC`, `INTERVAL`, `RANGE<T>`, non-microsecond `TIMESTAMP`, unrecognized future types/modes) and a `warnings.filterwarnings("error", category=UnsupportedTypeWarning)` example.
    *   Export `"UnsupportedTypeWarning"` in `__all__` alongside `"BigQueryError"` and `"UnsupportedTypeError"`.

#### 4. [`py-polars-bigquery/polars_bigquery/core/executors/read_api.py`](../../py-polars-bigquery/polars_bigquery/core/executors/read_api.py) (Modify)

*   **Type / Scope:** Storage Read API plan compiler and executor.
*   **Rationale:** Avoids passing `selected_fields=[]` (which streams every column, including skipped unsupported ones) using a single width-rank function shared with `core/types.py`.
*   **Required Changes:**
    *   Replace the duplicated dtype ranking ladder with `_zero_column_rank(dtype: pl.DataType) -> int`, returning `7` when `has_bignumeric_dtype(dtype)` is true and `column_width_rank(dtype)` otherwise.
    *   In `_select_zero_column_fallback(schema: pl.Schema) -> str`, select `min(candidates, key=lambda item: (_zero_column_rank(item[2]), item[0]))` where `candidates` prefers physical columns over ingestion-time pseudo-columns. When only pseudo-columns remain, `_PARTITIONDATE` (`DATE`, rank `2`) is chosen over `_PARTITIONTIME` (`TIMESTAMP`, rank `3`).

#### 5. [`py-polars-bigquery/polars_bigquery/_read_bigquery.py`](../../py-polars-bigquery/polars_bigquery/_read_bigquery.py) (Modify)

*   **Type / Scope:** Public `Client` implementation and docstrings.
*   **Rationale:** Coordinates warnings, all-unsupported `UnsupportedTypeError`, empty-schema short-circuiting for discovered zero-column tables, and accurate API docstrings.
*   **Required Changes:**
    *   Add `_MAX_SKIPPED_COLUMNS_DISPLAY = 5` and `_format_skipped_columns(skipped_columns)` formatting entries as `"col (reason)"` or `"col (reason at path)"`, appending `"... and N more"` when truncated.
    *   In `Client.scan_table()`:
        *   When `schema is None`, call `extract_schema_metadata(table_metadata)` directly (no `except (TypeError, ValueError)` wrapper so custom `BaseExtension` exceptions propagate with their original type and traceback).
        *   If `extracted.skipped_columns` is non-empty and `len(polars_schema) == 0`, raise `UnsupportedTypeError` with the skipped-column summary and a hint pointing to `client.scan_query()` / `client.read_query()` (no warning emitted).
        *   If `extracted.skipped_columns` is non-empty and `len(polars_schema) > 0`, log at `DEBUG` level and emit `warnings.warn(..., UnsupportedTypeWarning, stacklevel=2)`.
        *   For explicit `schema=`, call `_normalize_schema(schema, require_non_empty=True)` (raising `ValueError` on empty `schema={}`, matching `Client.scan_query()`).
        *   When `len(polars_schema) == 0` (discovered zero-column table), leave `source = None` and `executor = None`; inside `source_generator`, check `closed_event.is_set()` first, then short-circuit to `iter([pl.DataFrame(schema=polars_schema)])` without calling `arrow_client.read_table()`.
    *   Update docstrings on `Client.read_table()`, `Client.scan_table()`, and `Client.scan_query()` (uppercase `"UTC"`, explicit `schema=` guidance across default / registered / `load_as_storage` modes, outer-wrapped `ARRAY<JSON>` spelling warning, `UnsupportedTypeWarning` / `UnsupportedTypeError`, `DAY`-gated `_PARTITIONDATE`, full list of `scan_query()` `TypeError` types).

#### 6. [`docs/source/api/predicate-pushdown.md`](../../docs/source/api/predicate-pushdown.md) (Modify)

*   **Type / Scope:** Published MkDocs reference page linked by `Client.scan_table()`.
*   **Rationale:** Keeps the published BigQuery-to-Polars schema mapping table accurate with updated `TIMESTAMP` casing, `JSON`/`GEOGRAPHY` extensions, outer-wrapped arrays, skipped types, and `DAY`-only `_PARTITIONDATE`.
*   **Required Changes:** Update lines 97–124 (`### BigQuery-to-Polars Schema Mapping`).

#### 7. Unit & System Tests (Modify)

*   [`py-polars-bigquery/tests/unit/core/test_types.py`](../../py-polars-bigquery/tests/unit/core/test_types.py): Verify `UnsupportedTypeWarning` public export, rank `7` for unknown/unsupported dtypes and unparameterized `pl.Decimal` (`precision=None` mutated per Polars 2.x normalization), and `pl.BaseExtension` storage unwrapping.
*   [`py-polars-bigquery/tests/unit/core/test_schema.py`](../../py-polars-bigquery/tests/unit/core/test_schema.py): Cover all 14 `BQType` mappings, `ARRAY<JSON>`/`ARRAY<GEOGRAPHY>`, 15-level nested `STRUCT`s, storage-mode loading (`load_as_storage` and `as_storage=True`), `timestampPrecision` (`6`, `"6"`, `12`, `"12"`, `"9"`, `True`, `6.0`, `""`, and inside `REPEATED`), ingestion-time partitioning granularities (`DAY`, `"day"`, `""`, `None`, omitted, `HOUR`, `MONTH`, `YEAR`, unknown `"WEEK"` retaining `_PARTITIONTIME`), schemaless zero-column partitioned tables returning empty `pl.Schema({})`, `_PARTITIONTIME` skip-shadowing prevention, direct exception propagation from `extension_dtype_builder` / custom `BaseExtension.ext_from_params`, strict `extract_polars_schema_from_fields` (empty `STRUCT`, single unnamed `""`, case-differing nested `STRUCT`, unknown `type` column name, unknown `mode`), and `has_bignumeric_dtype` (`precision=None` and `pl.BaseExtension` unwrapping).
*   [`py-polars-bigquery/tests/unit/core/test_read_api_executor.py`](../../py-polars-bigquery/tests/unit/core/test_read_api_executor.py): Update zero-column fallback tier test to real BigQuery width tiers (`1`–`7`), uppercase `"UTC"`, custom `pl.BaseExtension` unwrapping, `_PARTITIONDATE` winning over `_PARTITIONTIME` in both declaration orders, and nested `BIGNUMERIC` inside `pl.BaseExtension`.
*   [`py-polars-bigquery/tests/unit/test_read_bigquery.py`](../../py-polars-bigquery/tests/unit/test_read_bigquery.py): Update warning formatting/stacklevel/5-entry cap tests, all-unsupported `UnsupportedTypeError` without warning, `DAY`/`HOUR` pseudo-column retention (including direct `io_source([], None, None, None)` check), discovered empty table short-circuit (including schemaless `DAY`/`HOUR` partitioned tables), explicit `schema={}` rejecting with `ValueError`, client-close guards for empty scans, in-memory Arrow C Stream round-trips via `_FakeArrowStreamExporter`, custom `BaseExtension` `TypeError` propagation (with `try/finally: pl.unregister_extension_type`), and malformed envelope `BigQueryError` assertions.
*   [`py-polars-bigquery/tests/system/test_read_bigquery_integration.py`](../../py-polars-bigquery/tests/system/test_read_bigquery_integration.py): Live tests confirming `scan_query` / `scan_table` extension parity (`JSON`, `GEOGRAPHY`, `ARRAY<JSON>`, `ARRAY<GEOGRAPHY>`, `STRUCT`), BigQuery synthetic `_field_1`/`_field_2` naming, `DAY`/`HOUR` `INTERVAL`-only ingestion-time tables, non-partitioned all-unsupported `UnsupportedTypeError`, and empty 0-column `DAY` table short-circuit.

### Verification Plan

*   **Rust / Build:** None; this design requires no changes to `crates/arrow-bigquery` or `py-arrow-bigquery`.
*   **Python Unit Tests:**
    *   `uv run --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit`
    *   `uv run --project py-arrow-bigquery --group test pytest -n auto py-arrow-bigquery/tests/unit`
*   **Static Analysis & Formatting:**
    *   `uv run --project py-polars-bigquery --group dev ruff check --no-fix py-polars-bigquery/`
    *   `uv run --project py-polars-bigquery --group dev ruff format --diff py-polars-bigquery/`
*   **Documentation Build (`strict: true`):**
    *   `uv run --with-requirements docs/source/requirements.txt mkdocs build --strict`
*   **Polars CI Matrix ([`.github/workflows/test-py-polars-unit.yml`](../../.github/workflows/test-py-polars-unit.yml)):** Verify unit tests pass with both `--isolated --resolution lowest` (`polars 2.0.0`, Python 3.11) and `--isolated --resolution highest --upgrade` (latest `polars 2.x`), passing `--isolated` locally to avoid mutating `uv.lock`.
*   **Live Integration Tests:** With `GOOGLE_CLOUD_PROJECT` configured:
    `uv run --project py-polars-bigquery --group test pytest py-polars-bigquery/tests/system -k "extension or partition_skipping"`

--------------------------------------------------------------------------------

*Drafted with the Elephant-Goldfish Model (EGM)*
