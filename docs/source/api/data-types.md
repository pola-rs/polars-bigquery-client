# Data Types

`polars_bigquery` targets **Polars 2.0+** and aligns the schemas produced by [`Client.scan_table`][polars_bigquery.Client.scan_table] (derived from BigQuery REST metadata) with the Arrow IPC streams returned by [`Client.read_table`][polars_bigquery.Client.read_table] and [`Client.read_query`][polars_bigquery.Client.read_query].

## BigQuery-to-Polars Type Mapping

All 14 supported BigQuery types map as follows:

| BigQuery Type | Mode | Polars `2.0+` DataType | Notes |
| --- | --- | --- | --- |
| `BOOL` / `BOOLEAN` | scalar | `pl.Boolean()` | — |
| `INT64` / `INTEGER` | scalar | `pl.Int64()` | — |
| `FLOAT64` / `FLOAT` | scalar | `pl.Float64()` | — |
| `NUMERIC` / `DECIMAL` | scalar | `pl.Decimal(precision=38, scale=9)` | Parameterized `NUMERIC(P, S)` columns are also read as `Decimal(38, 9)` matching BigQuery's Arrow output. |
| `STRING` | scalar | `pl.String()` | — |
| `BYTES` | scalar | `pl.Binary()` | — |
| `DATE` | scalar | `pl.Date()` | — |
| `DATETIME` | scalar | `pl.Datetime(time_unit="us")` | Timezone-naive (`time_zone=None`). BigQuery attaches the `google:sqlType:datetime` Arrow extension name to microsecond `Timestamp(us, None)` payloads; `arrow-bigquery` strips this wrapper during schema decoding so `.dt.*`, `.is_in()`, and `.fill_null()` work natively. |
| `TIME` | scalar | `pl.Time()` | Microsecond precision from BigQuery; Polars stores nanoseconds. |
| `TIMESTAMP` | scalar | `pl.Datetime(time_unit="us", time_zone="UTC")` | Uppercase `"UTC"` matches BigQuery's Arrow IPC schema. Microsecond precision only (`timestampPrecision` omitted or `6`). |
| `JSON` | scalar | `polars_bigquery.dtypes.JSON` (`pl.Extension("google:sqlType:json", pl.String())`) | Default unregistered Polars 2.0+ extension type; see [Working with `JSON` and `GEOGRAPHY`](#working-with-json-and-geography). |
| `JSON` | `REPEATED` (`ARRAY<JSON>`) | `pl.Extension("google:sqlType:json", pl.List(pl.String()))` | BigQuery attaches extension metadata to the outer `List` field rather than the inner `String` element. |
| `GEOGRAPHY` | scalar | `polars_bigquery.dtypes.GEOGRAPHY` (`pl.Extension("google:sqlType:geography", pl.String(), '{"encoding": "WKT"}')`) | Default unregistered Polars 2.0+ extension type (Well-Known Text). |
| `GEOGRAPHY` | `REPEATED` (`ARRAY<GEOGRAPHY>`) | `pl.Extension("google:sqlType:geography", pl.List(pl.String()), '{"encoding": "WKT"}')` | Outer `List` field carries the Arrow extension metadata. |
| `STRUCT` / `RECORD` | scalar | `pl.Struct([...])` | Must be non-empty with named, unique, supported subfields. Nested `STRUCT<j JSON, g GEOGRAPHY>` and `ARRAY<STRUCT<j JSON, g GEOGRAPHY>>` keep scalar extension dtypes on the struct fields; `STRUCT<j ARRAY<JSON>>` uses the outer-wrapped list extension. |
| Any other supported type | `REPEATED` (`ARRAY<T>`) | `pl.List(<inner>)` | Preserves field parameters (such as `timestampPrecision`) during recursion. |

### Ingestion-Time Partition Pseudo-Columns

For **ingestion-time partitioned tables** (`timePartitioning` present without a `field`), [`Client.scan_table`][polars_bigquery.Client.scan_table] automatically appends partition pseudo-columns when `schema` is not supplied:

| Pseudo-Column | Polars DataType | Availability |
| --- | --- | --- |
| `_PARTITIONDATE` | `pl.Date()` | Only on `DAY`-granularity ingestion-time partitioned tables (including tables where `timePartitioning.type` is omitted). |
| `_PARTITIONTIME` | `pl.Datetime(time_unit="us", time_zone="UTC")` | All ingestion-time partitioned tables (`HOUR`, `DAY`, `MONTH`, `YEAR`). |

When filtering on `_PARTITIONTIME`, pass a **timezone-aware** `datetime` (e.g. `datetime(2024, 1, 1, tzinfo=timezone.utc)`) so the filter matches `TIMESTAMP` rather than timezone-naive `DATETIME`.

---

## Working with `JSON` and `GEOGRAPHY`

BigQuery encodes `JSON` and `GEOGRAPHY` over Arrow IPC using Arrow extension metadata (`google:sqlType:json` and `google:sqlType:geography`). In Polars 2.0+, unregistered Arrow extension types are loaded as `pl.Extension(...)`.

### Accessing the Underlying String Storage

Use Polars' `.ext.storage()` accessor to operate on the underlying JSON string or WKT geometry string in Polars expressions:

```python
import polars as pl
import polars_bigquery

with polars_bigquery.Client(quota_project_id="my-project") as client:
    lf = client.scan_table("my-project.my_dataset.events")

    # Null checks push down to BigQuery Storage Read API:
    non_null = lf.filter(pl.col("payload").is_not_null()).collect()

    # String operations run in Polars via .ext.storage() (without sending invalid SQL):
    objs = lf.filter(pl.col("payload").ext.storage().str.starts_with("{")).collect()
```

### Supplying an Explicit `schema=`

When you pass an explicit `schema=` to [`Client.scan_table`][polars_bigquery.Client.scan_table] (skipping the BigQuery `tables.get` REST call), use [`polars_bigquery.dtypes.JSON`][polars_bigquery.dtypes.JSON] and [`polars_bigquery.dtypes.GEOGRAPHY`][polars_bigquery.dtypes.GEOGRAPHY] for scalar columns and outer-wrapped `pl.Extension(..., pl.List(pl.String()), ...)` for `ARRAY<JSON>` / `ARRAY<GEOGRAPHY>`:

```python
import polars as pl
import polars_bigquery

lf = client.scan_table(
    "my-project.my_dataset.events",
    schema={
        "ts": pl.Datetime("us", "UTC"),
        "payload": polars_bigquery.dtypes.JSON,
        "geom": polars_bigquery.dtypes.GEOGRAPHY,
        "payload_list": pl.Extension("google:sqlType:json", pl.List(pl.String())),
        "geom_list": pl.Extension(
            "google:sqlType:geography",
            pl.List(pl.String()),
            '{"encoding": "WKT"}',
        ),
    },
)
```

Passing plain `pl.String` for a `JSON` or `GEOGRAPHY` column under the default extension configuration raises Polars' `SchemaError` during collection because the Arrow stream emits `pl.Extension(...)`.

### Extension Registration Modes

`polars_bigquery` honors Polars' runtime extension registry (`polars.register_extension_type`) and environment variable (`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR`):

1. **Default (`load_as_extension`)**: Unregistered `JSON` and `GEOGRAPHY` columns map to `pl.Extension(...)` (matching [`polars_bigquery.dtypes.JSON`][polars_bigquery.dtypes.JSON] and [`polars_bigquery.dtypes.GEOGRAPHY`][polars_bigquery.dtypes.GEOGRAPHY]).
2. **Mode A — Custom `pl.BaseExtension` subclass**: If you register a subclass with `pl.register_extension_type("google:sqlType:geography", MyGeoExt)`, `scan_table(schema=None)` instantiates `MyGeoExt.ext_from_params(...)` to match Arrow decoding, and explicit `schema=` calls can pass instances of `MyGeoExt`.
3. **Mode B — Storage mode (`as_storage=True` or `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`)**:
    * **Supported on `scan_table(schema=None)`**: `scan_table` exposes `pl.String()` / `pl.List(pl.String())` in `lf.collect_schema()` while passing the authoritative BigQuery column types from `tables.get` into the predicate compiler so string operations on `JSON`/`GEOGRAPHY` are never pushed as invalid GoogleSQL.
    * **Process-lifetime latch for `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR`**: In Polars 2.x, the Arrow C Stream importer reads `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` **once per process** on first extension import (`pl.Config.reload_env_vars()` does not refresh it). Set the variable *before* importing `polars` / reading Arrow data, or prefer `pl.register_extension_type(..., as_storage=True)` for runtime control.
    * **Unsupported with explicit `schema=` for ambiguous storage types**: Without `tables.get` metadata, `pl.String` and `pl.List(pl.String)` cannot be distinguished from BigQuery `STRING` and `ARRAY<STRING>`, so pushing string filters onto a `JSON` or `GEOGRAPHY` column would fail in BigQuery with `400 INVALID_ARGUMENT`. When supplying an explicit `schema=`, use `polars_bigquery.dtypes` (`pl.Extension`) or a registered `BaseExtension` subclass instead.

---

## Unsupported BigQuery Types and Workarounds

The following BigQuery types are currently unsupported in `polars_bigquery`:

| BigQuery Type | Reason | `client.read_query()` SQL Workaround |
| --- | --- | --- |
| `BIGNUMERIC` / `BIGDECIMAL` | 76-digit precision exceeds Polars' 38-digit `Decimal` limit. | `CAST(col AS STRING)` or `CAST(col AS FLOAT64)` |
| `INTERVAL` | BigQuery encodes `INTERVAL` as Arrow `Interval(MonthDayNano)`, which is unsupported in Polars Arrow FFI (see [pola-rs/polars#30920](https://github.com/pola-rs/polars/issues/30920) and [pola-rs/polars#30921](https://github.com/pola-rs/polars/issues/30921)). | `CAST(col AS STRING)` |
| `RANGE<T>` | No native Polars `Range` data type. | `STRUCT(RANGE_START(col) AS start_val, RANGE_END(col) AS end_val)` |
| Picosecond `TIMESTAMP` (`timestampPrecision != 6`) | Polars `Datetime` supports up to nanosecond precision only. | `CAST(col AS STRING)` or truncate to microseconds |
| Empty, unnamed, or duplicate-field `STRUCT` | Polars `pl.Struct` requires non-empty, named, unique fields. | Select and alias individual subfields in SQL |

### `scan_table()` vs. `read_table()` Behavior

* **`Client.scan_table(table)` (`schema=None`)**:
    * **Mixed tables**: Any top-level column whose type (or any nested descendant inside `STRUCT`/`ARRAY`) is unsupported is omitted from the `LazyFrame` schema, and an [`UnsupportedTypeWarning`][polars_bigquery.exceptions.UnsupportedTypeWarning] is emitted listing up to 5 omitted columns (with nested paths when applicable) and a `client.read_query()` hint. Supported columns and row counts (`lf.select(pl.len())`) scan normally without transmitting omitted columns.
    * **Strict mode**: Promote omission warnings to exceptions with `warnings.filterwarnings("error", category=UnsupportedTypeWarning)`.
    * **Tables with zero remaining supported columns**: If a non-empty table has no supported columns left (after retaining valid ingestion-time partition pseudo-columns `_PARTITIONTIME` / `_PARTITIONDATE`), `scan_table()` raises `ValueError` immediately without emitting `UnsupportedTypeWarning`.
    * **0-column tables**: Tables with an empty BigQuery schema return an empty-schema `LazyFrame` and collect a 0-row, 0-column `DataFrame` without invoking `CreateReadSession`.
* **`Client.read_table(table)` and `Client.read_query(query)`**:
    * Stream all returned Arrow columns directly without omitting unsupported columns.
    * Any Arrow schema containing unsupported Arrow types that cannot be decoded or exported over Arrow C Data Interface FFI—including BigQuery `INTERVAL` (`Interval(MonthDayNano)`) and `BIGNUMERIC` (`Decimal256`, top-level or nested)—is rejected immediately in Rust with [`UnsupportedTypeError`][polars_bigquery.exceptions.UnsupportedTypeError] (a subclass of [`BigQueryError`][polars_bigquery.exceptions.BigQueryError]) rather than aborting the Python process. For `INTERVAL`, the message cites upstream Polars issues [#30920](https://github.com/pola-rs/polars/issues/30920) and [#30921](https://github.com/pola-rs/polars/issues/30921).
    * Use `client.read_query("SELECT * EXCEPT (bad_col), CAST(bad_col AS STRING) AS bad_col FROM ...")` to read tables that contain unsupported columns.
