# Row Predicate Pushdown

When scanning a table with [`Client.scan_table`][polars_bigquery.Client.scan_table], Polars passes `.filter()` predicates to `polars_bigquery`, which compiles supported Polars expressions into a BigQuery Storage Read API `row_restriction` SQL filter using the table's schema (and authoritative BigQuery column types when `schema=None`).

!!! note "Safe fallback and exact filtering"
    `polars_bigquery` always applies the original `.filter()` predicate in Polars after streaming batches from BigQuery. If an expression (or a branch of a conjunctive `&` filter) uses an unsupported operation, incompatible operand types, or an unrepresentable literal, `polars_bigquery` safely falls back to evaluating the unsupported portion in memory without data loss.

For the BigQuery-to-Polars table schema mapping (including `TIMESTAMP` `"UTC"`, `JSON`/`GEOGRAPHY` extension types, ingestion-time partition pseudo-columns, and unsupported types), see [Data types](data-types.md).

## Supported Polars Expressions

### Column and Literal Primitives

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `pl.col("name")` | `` `name` `` | Supports standard and [flexible BigQuery column names](#column-naming-rules-and-limits), as well as `_PARTITIONDATE` (`DAY` partitions) and `_PARTITIONTIME` on ingestion-time partitioned tables. |
| `pl.lit(value)` | SQL literal | See [Supported Literal Data Types](#supported-literal-data-types) for literal formatting and bounds. |

### Comparison Operators

All comparison operators require both operands to be exact, supported, non-opaque scalar expressions with compatible types and neither operand to be a bare `Null` literal (use `.is_null()` or `.is_not_null()` for `NULL` checks).

| Polars Expression | Operator / Method | BigQuery SQL Translation | Notes |
| --- | --- | --- | --- |
| `a == b` | `Expr.eq` | `(`a` = `b`)` | For `FLOAT64` operands that may contain `NaN`, lowered to preserve Polars' total float order (`NaN == NaN` is `True`, `NaN > x` is `True`) under 3-valued logic (`NULL` stays `NULL`). |
| `a != b` | `Expr.ne` | `(`a` != `b`)` | `FLOAT64` column-to-column comparisons guard `NaN != NaN` -> `FALSE` while preserving `NULL`. |
| `a > b` | `Expr.gt` | `(`a` > `b`)` | `FLOAT64` comparisons include `IS_NAN(...)` guards (subject to AST duplication and size caps). |
| `a >= b` | `Expr.ge` | `(`a` >= `b`)` | `FLOAT64` comparisons include `IS_NAN(...)` / `IS NOT NULL` guards. |
| `a < b` | `Expr.lt` | `(`a` < `b`)` | Symmetric `FLOAT64` `NaN` total-order guards. |
| `a <= b` | `Expr.le` | `(`a` <= `b`)` | Symmetric `FLOAT64` `NaN` total-order guards. |

### Boolean and Logical Operators

Logical `&`, `|`, and `~` require `BOOL` operands (integer bitwise `&`, `|`, and `~` are not pushed down).

| Polars Expression | Operator / Method | BigQuery SQL Translation | Notes |
| --- | --- | --- | --- |
| `a & b` | `And` (`&`) | `(`a` AND `b`)` | **Supports partial pushdown**: if one branch cannot be pushed down, the supported branch is still pushed down to BigQuery. |
| `pl.all_horizontal(a, b, ...)` | `all_horizontal` | `((`a` AND `b`) ...)` | Folded into left-associative `AND` expressions; supports partial pushdown. |
| `a \| b` | `Or` (`\|`) | `(`a` OR `b`)` | Pushed down when both branches are supported `BOOL` expressions and exact. |
| `~a` | `Expr.not_` (`~`) | `(NOT `a`)` | Pushed down when the inner `BOOL` expression is exact. Negated disjunctions `~(a \| b)` are automatically rewritten via De Morgan's Law into `(~a) & (~b)` so supported branches can still be pushed down. |
| `pl.when(p).then(t).otherwise(f)` | `when` / `then` / `otherwise` | Boolean conditionals (`AND` / `OR` / `NOT COALESCE(..., FALSE)`), `IF(`p`, `t`, `f`)`, or `CASE WHEN `p1` THEN `t1` WHEN `p2` THEN `t2` ... [ELSE `f`] END` | Requires `BOOL` predicates (never `NULL`), non-opaque compatible value branches, and at least one non-`NULL` result branch. Supports partial pushdown in boolean filter context and multi-predicate `pl.when(p1, p2, ...)`. |

### Null Checks and Handling

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `expr.is_null()` | `(`expr` IS NULL)` | Supported on all 14 BigQuery types, including `JSON`, `GEOGRAPHY`, `ARRAY`, and `STRUCT`. |
| `expr.is_not_null()` | `(`expr` IS NOT NULL)` | Supported on all 14 BigQuery types, including `JSON`, `GEOGRAPHY`, `ARRAY`, and `STRUCT`. |
| `expr.drop_nulls()` | *(not pushed down)* | Evaluates in Polars. `drop_nulls()` shrinks Series length and broadcasts length-1 results back across null rows, so hoisting `IS NOT NULL` cannot guarantee a row-level superset without risking silent row loss or masking Polars `ShapeError`. Use `.is_not_null()` for pushdown. |
| `pl.coalesce(a, b, ...)` | `COALESCE(`a`, `b`, ...)` | Pushed down when all operands are exact, non-`Null`, non-opaque, and type-compatible. |
| `expr.fill_null(value)` | `IFNULL(`expr`, `value`)` | Pushed down when both `expr` and `value` are exact, non-`Null`, non-opaque, and type-compatible. |
| `expr.fill_null(strategy="zero" \| "one")` | Type-specific `IFNULL(`expr`, <zero/one>)` | Lowers to the Polars-equivalent BigQuery literal for each type (`FALSE`/`TRUE` for `BOOL`, `0`/`1` for `INT64`, `0.0`/`1.0` for `FLOAT64`, `NUMERIC '0'`/`'1'` for `NUMERIC`, `''` for `STRING` `"zero"`, `b''` for `BYTES` `"zero"`, `DATE`/`DATETIME`/`TIMESTAMP` epoch + 0 or 1 unit for `"ms"`/`"us"`, `TIME '00:00:00'` for `TIME` `"zero"`). Unrepresentable cases (`"one"` on `STRING`/`BYTES`/`TIME`/`Datetime("ns")`, opaque types, and analytical strategies `"forward"`, `"backward"`, `"min"`, `"max"`, `"mean"`) fall back to Polars. |

### Numeric Predicates

| Polars Expression | `FLOAT64` SQL Translation | `INT64` SQL Translation | Other Types |
| --- | --- | --- | --- |
| `expr.is_nan()` | `IS_NAN(`expr`)` | `FALSE` (positive filter) / `IF((`expr` IS NOT NULL), FALSE, NULL)` (3-valued) | Not pushed down |
| `expr.is_not_nan()` | `(NOT IS_NAN(`expr`))` | `(`expr` IS NOT NULL)` (positive filter) / `IF((`expr` IS NOT NULL), TRUE, NULL)` (3-valued) | Not pushed down |
| `expr.is_infinite()` | `IS_INF(`expr`)` | `FALSE` (positive filter) / `IF((`expr` IS NOT NULL), FALSE, NULL)` (3-valued) | Not pushed down |
| `expr.is_finite()` | `(NOT IS_INF(`expr`) AND NOT IS_NAN(`expr`))` | `(`expr` IS NOT NULL)` (positive filter) / `IF((`expr` IS NOT NULL), TRUE, NULL)` (3-valued) | Not pushed down |

### String Expressions (`Expr.str`)

All `Expr.str` operators require strict `STRING` operands (never `JSON`, `GEOGRAPHY`, `BYTES`, or numeric columns).

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `expr.str.starts_with(prefix)` | `STARTS_WITH(`expr`, `prefix`)` | `prefix` must be a `STRING` literal or column expression. |
| `expr.str.ends_with(suffix)` | `ENDS_WITH(`expr`, `suffix`)` | `suffix` must be a `STRING` literal or column expression. |
| `expr.str.contains(pattern, literal=False, strict=True)` | `REGEXP_CONTAINS(`expr`, `pattern`)` | Default regex mode (`literal=False`, `strict=True`). `strict=False` with `literal=False` is not pushed down. |
| `expr.str.contains(pattern, literal=True)` | `(STRPOS(`expr`, `pattern`) > 0)` | Literal substring match. |
| `expr.str.to_uppercase()` | `UPPER(`expr`)` | Can be composed with comparisons or other string predicates. |
| `expr.str.to_lowercase()` | `LOWER(`expr`)` | Can be composed with comparisons or other string predicates. |

### Set Membership (`Expr.is_in`)

| Polars Expression | BigQuery SQL Translation | Notes |
| --- | --- | --- |
| `expr.is_in(values, nulls_equal=False)` | `(`expr` IN (v1, v2, ...))` | Requires a strict same-type match between a non-opaque scalar `expr` and a non-empty homogeneous Python `list` or `pl.Series` of up to 10,000 non-null, non-`NaN` scalar literals. `nulls_equal=True` is not pushed down. |

---

## Per-Type Pushdown Support Matrix

| BigQuery Type | `IS [NOT] NULL` | Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) | `is_in` | `fill_null(val)` / `coalesce` / `when` | `fill_null(strategy=...)` | `is_nan` / `is_inf` / `is_finite` | `str.*` | Logical `&` / `\|` / `~` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `BOOL` | Yes | Same type | `BOOL` | Same type | `"zero"`, `"one"` | No | No | Yes |
| `INT64` | Yes | `INT64`, `FLOAT64`, `NUMERIC` | `INT64` | `INT64`, `FLOAT64`, `NUMERIC` | `"zero"`, `"one"` | Yes (null-safe 3VL) | No | No (bitwise) |
| `FLOAT64` | Yes | `FLOAT64`, `INT64` (NaN-safe) | `FLOAT64` | `FLOAT64`, `INT64` | `"zero"`, `"one"` | Yes | No | No |
| `NUMERIC` | Yes | `NUMERIC`, `INT64` | No\* | `NUMERIC`, `INT64` | `"zero"`, `"one"` | No | No | No |
| `STRING` | Yes | Same type | `STRING` | Same type | `"zero"` only | No | Yes | No |
| `BYTES` | Yes | Same type | No\* | Same type | `"zero"` only | No | No | No |
| `DATE` | Yes | Same type | `DATE` | Same type | `"zero"`, `"one"` | No | No | No |
| `DATETIME` | Yes | Same type | `DATETIME` | Same type | `"zero"`, `"one"` (`"ms"`/`"us"`) | No | No | No |
| `TIME` | Yes | Same type | No\* | Same type | `"zero"` only | No | No | No |
| `TIMESTAMP` | Yes | Same type | `TIMESTAMP` | Same type | `"zero"`, `"one"` (`"ms"`/`"us"`) | No | No | No |
| `JSON` | Yes | No | No | No | No | No | No | No |
| `GEOGRAPHY` | Yes | No | No | No | No | No | No | No |
| `ARRAY` | Yes | No | No | No | No | No | No | No |
| `STRUCT` | Yes | No | No | No | No | No | No | No |

\* User-facing `pl.Decimal`, `pl.Binary`, and `pl.Time` literal / `is_in` parsing is not yet supported; column-to-column comparisons and `fill_null(strategy=...)` are supported as shown above.

---

## Common Pitfalls That Disable Pushdown

1. **Comparing `NUMERIC` (`pl.Decimal(38, 9)`) to `FLOAT64` (`float`)**: Polars casts `Decimal` to lossy `Float64` (`Decimal('9007199254740993.0') == 9007199254740992.0` is `True`), whereas BigQuery compares `NUMERIC` and `FLOAT64` exactly (`FALSE`). Pushdown is skipped to prevent silent row loss; compare against an integer (`INT64`) when possible.
2. **Mixing temporal types (`DATE`, `DATETIME`, `TIMESTAMP`, `TIME`)**: BigQuery rejects cross-temporal comparisons such as `DATE = TIMESTAMP` or `DATETIME = TIMESTAMP`. Always compare each temporal column against a matching literal type.
3. **Comparing `_PARTITIONTIME` (`TIMESTAMP`) to a timezone-naive `datetime`**: A naive `datetime(2024, 1, 1)` compiles as BigQuery `DATETIME`, which does not match `_PARTITIONTIME` (`TIMESTAMP`). Pass a timezone-aware datetime (`datetime(2024, 1, 1, tzinfo=timezone.utc)`) for partition pruning pushdown.
4. **Integer bitwise `&`, `|`, `~`**: Polars overloads `&`, `|`, and `~` for both boolean logic and integer bitwise operations. On `INT64` columns these expressions are not pushed down (any sibling boolean `&` filter is still pushed down).
5. **Filtering `OPAQUE` columns (`JSON`, `GEOGRAPHY`, `ARRAY`, `STRUCT`)**: Only `.is_null()` and `.is_not_null()` push down on opaque columns. Operations such as `.ext.storage().str.starts_with(...)`, `.drop_nulls()`, or equality comparisons execute safely in Polars.

---

## Supported Literal Data Types

The following Polars data types are supported as user-supplied scalar literals in comparisons and as elements of `list` / `pl.Series` collections in `Expr.is_in`:

| Polars Data Type | Python Type | BigQuery SQL Representation | Constraints & Bounds |
| --- | --- | --- | --- |
| `pl.Boolean` | `bool` | `TRUE`, `FALSE` | — |
| `pl.Int8`, `pl.Int16`, `pl.Int32`, `pl.Int64`, `pl.Int128` | `int` | `INT64` integer literal | Value must fit within signed 64-bit integer bounds (`-9,223,372,036,854,775,808` to `9,223,372,036,854,775,807`). |
| `pl.UInt8`, `pl.UInt16`, `pl.UInt32`, `pl.UInt64` | `int` | `INT64` integer literal | Value must fit within signed 64-bit integer bounds (`0` to `9,223,372,036,854,775,807`). (`pl.UInt128` is excluded.) |
| `pl.Float32`, `pl.Float64` | `float` | `FLOAT64` literal | Finite floats are pushed down in comparisons and `is_in` (`Float32` literals are rounded through IEEE-754 binary32 so `f32 -> f64` widening matches Polars). (`pl.Float16` is excluded.) For `NaN` or `±Inf` checks, use `.is_nan()`, `.is_not_nan()`, `.is_infinite()`, or `.is_finite()`. |
| `pl.String` | `str` | Single-quoted `'...'` | Special characters (`\`, `'`, `\n`, `\r`, `\t`, `\0`) are automatically escaped. |
| `pl.Date` | `datetime.date` | `DATE(TIMESTAMP_SECONDS(days * 86400))` | Must be within BigQuery `DATE` bounds (`0001-01-01` to `9999-12-31`). |
| `pl.Datetime(time_unit, time_zone=...)` (timezone-aware) | `datetime.datetime` (with `tzinfo`) | `TIMESTAMP_MICROS(...)` or `TIMESTAMP_MILLIS(...)` | Compiles to a BigQuery `TIMESTAMP` literal. Supports `"ms"`, `"us"`, and `"ns"` (when divisible by `1,000` for exact microsecond precision) within `0001-01-01T00:00:00Z` to `9999-12-31T23:59:59.999999Z`. |
| `pl.Datetime(time_unit, time_zone=None)` (timezone-naive) | `datetime.datetime` (without `tzinfo`) | `DATETIME(TIMESTAMP_MICROS(...))` or `DATETIME(TIMESTAMP_MILLIS(...))` | Compiles to a BigQuery `DATETIME` literal. Supports `"ms"`, `"us"`, and `"ns"` (when divisible by `1,000` for exact microsecond precision) within `0001-01-01T00:00:00` to `9999-12-31T23:59:59.999999`. |
| `pl.List` / `pl.Series` | `list` / `pl.Series` | `(v1, v2, ...)` | Supported as the right-hand side of `Expr.is_in()`. Must be non-empty, homogeneous, free of `Null` and `NaN` values, and contain at most 10,000 elements (and at most 1 MiB when serialized as an Arrow IPC `Series`). In `Datetime` lists, all elements must be consistently timezone-aware or timezone-naive. |

---

## Column Naming Rules and Limits

- **Column identifiers**: Standard and flexible BigQuery column names are quoted with backticks (`` `col` ``). Allowed characters include Unicode letters (`\p{L}`), numbers (`\p{N}`), marks (`\p{M}`), underscores (`\p{Pc}`), hyphens/dashes (`\p{Pd}`), spaces, and the flexible special characters `&`, `%`, `=`, `+`, `:`, `'`, `<`, `>`, `#`, and `|`. Column names containing characters disallowed by BigQuery (such as `` ` ``, `\`, `.`, `$`, or `!`) raise a `ValueError`.
- **Expression tree and SQL size limits**: Predicate ASTs up to 10,000 nodes (`MAX_AST_NODES = 10_000`) are compiled, duplicated operands in `FLOAT64` `NaN` rewrites are capped at 64 nodes (`MAX_DUPLICATED_OPERAND_NODES = 64`), and compiled `row_restriction` SQL exceeding 1,000,000 UTF-8 bytes (`MAX_ROW_RESTRICTION_BYTES = 1_000_000`) falls back to client-side filtering.
