# Row Predicate Pushdown

When scanning a table with [`Client.scan_table`][polars_bigquery.Client.scan_table], Polars passes `.filter()` predicates to `polars_bigquery`, which compiles supported Polars expressions into a BigQuery Storage Read API `row_restriction` SQL filter. [`Client.scan_query`][polars_bigquery.Client.scan_query] (with `maintain_order=False`) uses the same expression compiler to append GoogleSQL pipe operators (`|> WHERE`, `|> SELECT`, and `|> LIMIT`) directly to the user query.

!!! note "Safe fallback and exact filtering"
    `polars_bigquery` always applies the original `.filter()` predicate in Polars after streaming batches from BigQuery. If an expression (or a branch of a conjunctive `&` filter) uses an unsupported operation or literal, `polars_bigquery` safely falls back to evaluating the unsupported portion in memory without data loss.

## Supported Polars Expressions

### Column and Literal Primitives

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `pl.col("name")` | `` `name` `` | Supports standard and [flexible BigQuery column names](#column-naming-rules-and-limits), plus `_PARTITIONDATE` and `_PARTITIONTIME` on ingestion-time partitioned tables (in `Client.scan_table` only). |
| `pl.lit(value)` | SQL literal | See [Supported Polars Data Types](#supported-polars-data-types) for literal formatting and bounds. |

### Comparison Operators

All comparison operators require both operands to be exact, supported expressions and neither operand to be a bare `Null` literal (use `.is_null()` or `.is_not_null()` for `NULL` checks).

| Polars Expression | Operator / Method | BigQuery SQL Translation |
| --- | --- | --- |
| `a == b` | `Expr.eq` | `(`a` = `b`)` |
| `a != b` | `Expr.ne` | `(`a` != `b`)` |
| `a > b` | `Expr.gt` | `(`a` > `b`)` |
| `a >= b` | `Expr.ge` | `(`a` >= `b`)` |
| `a < b` | `Expr.lt` | `(`a` < `b`)` |
| `a <= b` | `Expr.le` | `(`a` <= `b`)` |

### Boolean and Logical Operators

| Polars Expression | Operator / Method | BigQuery SQL Translation | Notes |
| --- | --- | --- | --- |
| `a & b` | `And` (`&`) | `(`a` AND `b`)` | **Supports partial pushdown**: if one branch cannot be pushed down, the supported branch is still pushed down to BigQuery. |
| `pl.all_horizontal(a, b, ...)` | `all_horizontal` | `((`a` AND `b`) ...)` | Folded into left-associative `AND` expressions; supports partial pushdown. |
| `a \| b` | `Or` (`\|`) | `(`a` OR `b`)` | Pushed down when both branches are supported and exact. |
| `~a` | `Expr.not_` (`~`) | `(NOT `a`)` | Pushed down when the inner expression is exact. Negated disjunctions `~(a \| b)` are automatically rewritten via De Morgan's Law into `(~a) & (~b)` so supported branches can still be pushed down. |
| `pl.when(p).then(t).otherwise(f)` | `when` / `then` / `otherwise` | Boolean conditionals (`AND` / `OR` / `NOT COALESCE(..., FALSE)`), `IF(`p`, `t`, `f`)`, or `CASE WHEN `p1` THEN `t1` WHEN `p2` THEN `t2` ... [ELSE `f`] END` | When used as a boolean filter (or inside `&` / `\|` or `when` predicates and boolean branches), falsy branches (`False` or `NULL`) and partial pushdown expressions are rewritten into null-safe boolean conditionals (`(`p` AND `t`)` or `((NOT COALESCE(`p`, FALSE)) AND `f`)`), **supporting partial pushdown**. Supported, exact `when().then().otherwise()` expressions compile to `IF(...)` or `CASE WHEN ... END`. Supports multi-predicate `pl.when(p1, p2, ...)`. Requires no `NULL` predicates and at least one non-`NULL` result branch. |

### Null Checks and Handling

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `expr.is_null()` | `(`expr` IS NULL)` | — |
| `expr.is_not_null()` | `(`expr` IS NOT NULL)` | — |
| `expr.drop_nulls()` | `... AND (`expr` IS NOT NULL)` | Replaces `expr.drop_nulls()` with `expr` and conjoins `(`expr` IS NOT NULL)` to the top-level row restriction when `expr` is supported, exact, and non-`Null`. |
| `pl.coalesce(a, b, ...)` | `COALESCE(`a`, `b`, ...)` | Pushed down when all operands are supported, exact, and non-`Null`. |
| `expr.fill_null(value)` | `IFNULL(`expr`, `value`)` | Pushed down when both `expr` and `value` are supported, exact, and non-`Null`. |
| `expr.fill_null(strategy="zero" \| "one")` | `IFNULL(`expr`, 0)` or `IFNULL(`expr`, 1)` | Analytical and aggregate strategies (`"forward"`, `"backward"`, `"min"`, `"max"`, `"mean"`) are not pushed down. |

### Numeric Predicates

| Polars Expression | BigQuery SQL Translation |
| --- | --- |
| `expr.is_nan()` | `IS_NAN(`expr`)` |
| `expr.is_not_nan()` | `(NOT IS_NAN(`expr`))` |
| `expr.is_infinite()` | `IS_INF(`expr`)` |
| `expr.is_finite()` | `(NOT IS_INF(`expr`) AND NOT IS_NAN(`expr`))` |

### String Expressions (`Expr.str`)

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `expr.str.starts_with(prefix)` | `STARTS_WITH(`expr`, `prefix`)` | `prefix` can be a string literal or column expression. |
| `expr.str.ends_with(suffix)` | `ENDS_WITH(`expr`, `suffix`)` | `suffix` can be a string literal or column expression. |
| `expr.str.contains(pattern, literal=False, strict=True)` | `REGEXP_CONTAINS(`expr`, `pattern`)` | Default regex mode (`literal=False`, `strict=True`). `strict=False` with `literal=False` is not pushed down. |
| `expr.str.contains(pattern, literal=True)` | `(STRPOS(`expr`, `pattern`) > 0)` | Literal substring match. |
| `expr.str.to_uppercase()` | `UPPER(`expr`)` | Can be composed with comparisons or other string predicates. |
| `expr.str.to_lowercase()` | `LOWER(`expr`)` | Can be composed with comparisons or other string predicates. |

### Set Membership (`Expr.is_in`)

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `expr.is_in(values, nulls_equal=False)` | `(`expr` IN (v1, v2, ...))` | `values` must be a non-empty homogeneous Python `list` or `pl.Series` of up to 10,000 non-null, non-`NaN` scalar literals. `nulls_equal=True` is not pushed down. |

---

## Supported Polars Data Types

### Literal and Predicate Data Types

The following Polars data types are supported as scalar literals in comparisons and as elements of `list` / `pl.Series` collections in `Expr.is_in`:

| Polars Data Type | Python Type | BigQuery SQL Representation | Constraints & Bounds |
| --- | --- | --- | --- |
| `pl.Boolean` | `bool` | `TRUE`, `FALSE` | — |
| `pl.Int8`, `pl.Int16`, `pl.Int32`, `pl.Int64`, `pl.Int128` | `int` | `INT64` integer literal | Value must fit within signed 64-bit integer bounds (`-9,223,372,036,854,775,808` to `9,223,372,036,854,775,807`). |
| `pl.UInt8`, `pl.UInt16`, `pl.UInt32`, `pl.UInt64` | `int` | `INT64` integer literal | Value must fit within signed 64-bit integer bounds (`0` to `9,223,372,036,854,775,807`). |
| `pl.Float32`, `pl.Float64` | `float` | `FLOAT64` literal | Scalar comparisons push down **finite** floats only (`float("nan")`, `float("inf")`, and `float("-inf")` serialize as JSON `null` in Polars and fall back to in-memory filtering). `is_in` over float `Series` pushes down finite floats and `±Inf` (`CAST('inf' AS FLOAT64)` / `CAST('-inf' AS FLOAT64)`) while rejecting `NaN`. For `NaN` or `±Inf` checks, use `.is_nan()`, `.is_not_nan()`, `.is_infinite()`, or `.is_finite()`. |
| `pl.String` | `str` | Single-quoted `'...'` | Special characters (`\`, `'`, `\n`, `\r`, `\t`, `\0`) are automatically escaped. |
| `pl.Date` | `datetime.date` | `DATE(TIMESTAMP_SECONDS(days * 86400))` | Must be within BigQuery `DATE` bounds (`0001-01-01` to `9999-12-31`). |
| `pl.Datetime(time_unit, time_zone=...)` (timezone-aware) | `datetime.datetime` (with `tzinfo`) | `TIMESTAMP_MICROS(...)` or `TIMESTAMP_MILLIS(...)` | Compiles to a BigQuery `TIMESTAMP` literal. Supports `"ms"`, `"us"`, and `"ns"` (when divisible by `1,000` for exact microsecond precision) within `0001-01-01T00:00:00Z` to `9999-12-31T23:59:59.999999Z`. |
| `pl.Datetime(time_unit, time_zone=None)` (timezone-naive) | `datetime.datetime` (without `tzinfo`) | `DATETIME(TIMESTAMP_MICROS(...))` or `DATETIME(TIMESTAMP_MILLIS(...))` | Compiles to a BigQuery `DATETIME` literal. Supports `"ms"`, `"us"`, and `"ns"` (when divisible by `1,000` for exact microsecond precision) within `0001-01-01T00:00:00` to `9999-12-31T23:59:59.999999`. |
| `pl.List` / `pl.Series` | `list` / `pl.Series` | `(v1, v2, ...)` | Supported as the right-hand side of `Expr.is_in()`. Must be non-empty, homogeneous, free of `Null` and `NaN` values, and contain at most 10,000 elements (and at most 1 MiB when serialized as an Arrow IPC `Series`). In `Datetime` lists, all elements must be consistently timezone-aware or timezone-naive. |

### BigQuery-to-Polars Schema Mapping

When [`Client.scan_table`][polars_bigquery.Client.scan_table] inspects a BigQuery table's REST metadata, or when [`Client.scan_query`][polars_bigquery.Client.scan_query] runs a dry-run query for schema discovery (and when passing an explicit `schema` to either method), BigQuery column types correspond to the following Polars data types:

| BigQuery Type | Polars Data Type |
| --- | --- |
| `BOOL` / `BOOLEAN` | `pl.Boolean` |
| `INT64` / `INTEGER` | `pl.Int64` |
| `FLOAT64` / `FLOAT` | `pl.Float64` |
| `NUMERIC` / `DECIMAL` | `pl.Decimal(precision=38, scale=9)` |
| `BIGNUMERIC` / `BIGDECIMAL` | `pl.Decimal(precision=76, scale=38)` |
| `STRING` | `pl.String` |
| `BYTES` | `pl.Binary` |
| `DATE` | `pl.Date` |
| `DATETIME` | `pl.Datetime(time_unit="us", time_zone=None)` |
| `TIME` | `pl.Time` |
| `TIMESTAMP` | `pl.Datetime(time_unit="us", time_zone="utc")` |
| `INTERVAL` | `pl.Duration(time_unit="us")` |
| `JSON` | `pl.String` |
| `GEOGRAPHY` | `pl.String` |
| `STRUCT` / `RECORD` | `pl.Struct(...)` |
| `ARRAY` (`mode="REPEATED"`) | `pl.List(...)` |

For **ingestion-time partitioned tables**, [`Client.scan_table`][polars_bigquery.Client.scan_table] (only — not `scan_query`) also exposes two pseudo-columns in the `LazyFrame` schema that can be used in `.filter()` predicates for partition pruning:

- `_PARTITIONDATE`: `pl.Date`
- `_PARTITIONTIME`: `pl.Datetime(time_unit="us", time_zone="utc")`

---

## `scan_query` Pipe Pushdown Notes (`maintain_order=False`)

With the default `maintain_order=True`, [`Client.scan_query`][polars_bigquery.Client.scan_query] executes your SQL query verbatim without pushing down filters, projections, or limits. When `maintain_order=False`, `polars_bigquery` appends GoogleSQL pipe operators (`|> WHERE`, `|> SELECT`, `|> LIMIT`) to the right-trimmed query (preserving leading newlines and indentation so BigQuery `[line:col]` error coordinates stay unshifted). Keep these caveats in mind:

| Query Pattern / Limitation | Observed Behavior | Guidance / Workaround |
| --- | --- | --- |
| Explicit (including partial) `schema` | Unlike dry-run discovery (`schema=None`), which omits `\|> SELECT` when all discovered columns are needed, passing an explicit `schema` always emits `\|> SELECT` for the needed declared columns so **partial** schemas prune unlisted query columns server-side. | Supply a full or partial `schema` only when all referenced columns have explicit aliases; keep `maintain_order=True` for verbatim execution. |
| Anonymous / unaliased columns (`f0_`, `f1_`) | Dry-run all-column scans omit `\|> SELECT`, so `f0_` flows through unfiltered or filtered/sliced scans. Explicit `\|> SELECT` (including with explicit `schema={"f0_": ...}`) or `\|> WHERE` on `f0_` fails with `Unrecognized name: f0_`. | Alias expressions explicitly with `AS`, use `schema=None` for all-column scans, or use `maintain_order=True`. |
| Duplicate or case-colliding SQL aliases (`SELECT 1 AS a, 2 AS a` or `2 AS A`) | BigQuery auto-suffixes duplicates in the dry-run schema to `a`/`a_1` and `a`/`A_1`. `\|> LIMIT n` succeeds, ``\|> WHERE (`a` = 1)`` fails (`Column name a is ambiguous`), and selecting `a_1` / `A_1` fails (`Unrecognized name`). | Give every output column a unique, case-insensitive `AS` alias, or use `maintain_order=True`. |
| Trailing comments after `;` (`SELECT 1; -- note`, `; # note`, `; /* note */`) and multiple `;;` | Only a single bare trailing `;` is stripped before appending `\|>`. A comment *after* `;` (or `;;`) leaves `;` before `\n\|>` and fails at `.collect()`. *(Trailing comments without `;` succeed.)* | Omit trailing `;`, move the comment before `;`, or pass at most one trailing `;`. |
| Multi-statement scripts | `CREATE TEMP FUNCTION ...; SELECT ...` works with pipe pushdown. Other scripts are blocked eagerly during dry run (`statementType == "SCRIPT"`), and fail if appended with `\|>` under explicit `schema`. | Use a single `SELECT` statement, [`Client.read_query`][polars_bigquery.Client.read_query], or `maintain_order=True`. |
| Trailing `ORDER BY` in SQL | Appending `\|>` parses validly with `ORDER BY` (and `ORDER BY ... LIMIT ...`), but row order is **not** preserved under `maintain_order=False`. | Keep the default `maintain_order=True` when ordering must be preserved. |
| `lf.filter(p).limit(n)` vs. `lf.limit(n)` | `lf.limit(n)` pushes down `\|> LIMIT n`. For `lf.filter(p).limit(n)`, Polars keeps the slice above the scan node, so BigQuery executes `\|> WHERE ...` **without** SQL `LIMIT` (billing the full filtered query and holding the anonymous table cap and initial wait). | Add `LIMIT` inside the SQL query or use `lf.limit(n)` when a server-side cap is needed. |
| `\|> LIMIT` billing | `\|> LIMIT n` caps transferred rows and anonymous output size, **not** scanned bytes (`totalBytesProcessed`). | Push partition/cluster filters via `lf.filter(...)` or explicit SQL `WHERE`. |
| Query cache keys | Each distinct pushed `\|> WHERE` / `\|> SELECT` / `\|> LIMIT` combination changes the submitted SQL string and creates a separate BigQuery cache entry. Projected columns are emitted in canonical schema declaration order so column reordering reuses the same cache entry. | Keep `maintain_order=True` for verbatim exact-string query cache reuse, or materialize to a destination table. |
| Unresolved SQL length (> 1 MB) | Queries approaching BigQuery's ~1 MB unresolved query limit degrade automatically in tiers (pruning the largest top-level `AND` conjuncts first, then dropping `\|> WHERE` while retaining `\|> SELECT` and pre-filter `\|> LIMIT`, then falling back to verbatim SQL) and log one `WARNING`. | Avoid oversized inline `is_in(...)` lists or join against a temporary table. |

---

## Column Naming Rules and Limits

- **Column identifiers**: Standard and flexible BigQuery column names are quoted with backticks (`` `col` ``). Allowed characters include Unicode letters (`\p{L}`), numbers (`\p{N}`), marks (`\p{M}`), underscores (`\p{Pc}`), hyphens/dashes (`\p{Pd}`), spaces, and the flexible special characters `&`, `%`, `=`, `+`, `:`, `'`, `<`, `>`, `#`, and `|`. Column names containing characters disallowed by BigQuery (such as `` ` ``, `\`, `.`, `$`, or `!`) raise a `ValueError`.
- **Expression tree size**: Predicate ASTs up to 10,000 nodes (`MAX_AST_NODES = 10_000`) are compiled; subtrees exceeding this budget gracefully degrade to client-side filtering while preserving shallower conjunctive (`&`) filters.
