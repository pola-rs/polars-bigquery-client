# Polars-BigQuery scan_table API coverage
*Extending Polars-BigQuery to support row predicate pushdown*

*   **Author:** @tswast
*   **Status:** Implementation in progress
*   **Created:** 2026-09-16
*   **Last updated:** 2026-09-25


## Overview

This document is a sub-project of the overall go/polars-bigquery-design . The goal of this document is to expand on the work done in [Milestone 6: LazyFrame predicate pushdown](?tab=t.0#bookmark=id.cqjwmia3pssa) to support more Polars expressions and eventually even BigQuery APIs that aren't yet supported in Polars.


## Background

The polars-bigquery package aims to provide the following APIs, with the focus of this tab on **scan_table**:



*   Client.read_table / Client.read_query
    *   Downloads data immediately to a Polars DataFrame.
*   **Client.scan_table** (method under consideration for this design)
    *   Returns a LazyFrame, allowing Polars expressions to be built on top of a BigQuery table and operations to push down to BigQuery. Currently only column and row filters can be pushed down by a Polars I/O plugin.
*   Client.write_table
    *   Takes a DataFrame / LazyFrame and writes it to a BigQuery table using the BQ Storage Write API.


### Polars LazyFrame expressions

Polars itself provides an [I/O plugin interface](https://docs.pola.rs/user-guide/plugins/io_plugins/), which we use to implement the scan_table API. Most relevant to this design, an I/O plugin takes the following argument: `predicate: pl.Expr | None`.

The polars.Expr API does not currently (2026-09-16) provide a stable, documented API for walking the tree of these expressions. Instead, we serialize to JSON and walk through the serialized expression.


```python
predicate_json_file = io.BytesIO()
predicate.meta.serialize(predicate_json_file, format="json")
predicate_json_file.seek(0)
predicate_json = json.load(predicate_json_file)
```


This produces JSON such as the following:


```python
import io
import json
import polars as pl

predicate_json_file = io.BytesIO()
predicate = (
  pl.col("name").str.starts_with("T")
  & (pl.col("number") > 10)
  & (pl.col("year") == 2000)
)
predicate.meta.serialize(predicate_json_file, format="json")
predicate_json_file.getvalue()
predicate_json_file.seek(0)
predicate_json = json.load(predicate_json_file)
print(json.dumps(predicate_json, indent=2))
```



```json
{
  "BinaryExpr": {
    "left": {
      "BinaryExpr": {
        "left": {
          "Function": {
            "input": [
              {
                "Column": "name"
              },
              {
                "Literal": {
                  "Scalar": {
                    "String": "T"
                  }
                }
              }
            ],
            "function": {
              "StringExpr": "StartsWith"
            }
          }
        },
        "op": "And",
        "right": {
          "BinaryExpr": {
            "left": {
              "Column": "number"
            },
            "op": "Gt",
            "right": {
              "Literal": {
                "Dyn": {
                  "Int": 10
                }
              }
            }
          }
        }
      }
    },
    "op": "And",
    "right": {
      "BinaryExpr": {
        "left": {
          "Column": "year"
        },
        "op": "Eq",
        "right": {
          "Literal": {
            "Dyn": {
              "Int": 2000
            }
          }
        }
      }
    }
  }
}

```



### Polars-BigQuery compilation

We compile the LazyFrame expression predicates described above to a BigQuery Storage Read API [row_restriction](https://docs.cloud.google.com/bigquery/docs/reference/storage/rpc/google.cloud.bigquery.storage.v1#google.cloud.bigquery.storage.v1.ReadSession.TableReadOptions.FIELDS.string.google.cloud.bigquery.storage.v1.ReadSession.TableReadOptions.row_restriction).


    _SQL text filtering statement, similar to a WHERE clause in a query. Aggregates are not supported._


    _Examples: "int_field > 5" "date_field = CAST('2014-9-27' as DATE)" "nullable_field is not NULL" "st_equals(geo_field, st_geofromtext("POINT(2, 2)"))" "numeric_field BETWEEN 1.0 AND 5.0"_


    _Restricted to a maximum length for 1 MB._


```python
from polars_bigquery.core import predicates

predicates.predicate_to_row_restriction((
  pl.col("name").str.starts_with("T")
  & (pl.col("number") > 10)
  & (pl.col("year") == 2000)
))
# ((STARTS_WITH(`name`, 'T') AND (`number` > 10)) AND (`year` = 2000))
```



So that we can support Polars expressions that don't yet have an implementation in Polars-BigQuery, we do as much as we can with the BQ Storage Read API but then also filter locally for any rows we missed. This partial expression support is powered by the observation that that omitting one side of an AND clause will give us a superset of the desired rows:


```python
if "BinaryExpr" in expr_json:
    ...

    if polars_op == "And":
        # With 'And', filtering by just one of the two children will still
        # give a superset of the filtered rows. The rest of the filters can
        # be applied by Polars instead of BigQuery.
        if left is None:
            return right
        if right is None:
            return left

        return f"({left} AND {right})"

    # The rest of these operators need both left and right to be converted
    # correctly for correctness.
    if left is None or right is None:
        return None
```



## Design

This design covers pushdown of row predicates and column selection as currently supported by the Polars I/O plugin interface. I have hopes of supporting aggregate and analytic function pushdown in the future (see https://github.com/pola-rs/polars/issues/23727), but since that requires changes to Polars itself, that will be a separate design.


### Task prioritization

The potential scope of this is huge, covering potentially all scalar functions in Polars, commonly used Polars extension crates, and BigQuery scalar functions. This design prioritizes implementing expressions that would be commonly desired for filter pushdown in BigQuery, especially those that would decrease the number of bytes scanned. This includes:



1. Boolean operations and [conditional expressions](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/conditional_expressions)
    1. case/when/if b/567584320
    2. coalesce b/567589406
    3. fill_null b/567595178
    4. IS NULL / IS NOT NULL b/567595288
2. Operations on data types that are compatible with [BigQuery partitioning](https://docs.cloud.google.com/bigquery/docs/partitioned-tables#types_of_partitioning), including:
    5. INTEGER
    6. DATE
    7. TIMESTAMP
    8. DATETIME
3. Operations on data types that are compatible with [BigQuery clustering](https://docs.cloud.google.com/bigquery/docs/clustered-tables#cluster_column_types), including:
    9. BIGNUMERIC
    10. BOOL
    11. DATE
    12. DATETIME
    13. GEOGRAPHY
    14. INT64
    15. NUMERIC
    16. RANGE
    17. STRING
    18. TIMESTAMP

Translating these requires multiple steps:



1. Identify the equivalent to the BigQuery expression as a Polars expression.
    1. Watch out for semantic differences. Even if a function is named identically, it won't always behave the same in Polars versus BigQuery.
2. Determine how that Polars expression serializes to JSON. Test this across multiple Polars versions, since this serialization isn't guaranteed to be stable.


### Client-side architecture

There are several client-side components involved:



*   Python polars-bigquery, responsible for
    *   providing the Polars I/O plugin
    *   compiling Polars expressions into the subset of SQL supported by the BQ Storage Read API's row_predicate field
*   Python arrow-bigquery, responsible for
    *   providing a thin layer over the Rust client
    *   returning PyCapsule Arrow streams that can then be converted to Polars (or pandas) DataFrames.
*   Rust arrow-bigquery, responsible for
    *   reading from tables with the BQ Storage Read API to Arrow's C ABI for Arrow streams
    *   eventually (out of scope for this tab):
        *   fast query path execution
        *   writes from Arrow C ABI streams to BigQuery tables


### Compiler architecture

There are several components to the polars-bigquery row_predicate compiler:



*   Parsing Polars Expr into IR
*   Intermediate Representation (IR)
*   Compiling IR to a subset of SQL


#### Parsing Polars Expressions

There is no API exposed in Python for fully walking a Polars expression tree. Instead, we serialize to JSON.

We dispatch from JSON keys using JSON Path routes to various parsing functions.

We walk the tree breadth-first to avoid Python's stack size limitations.


#### Intermediate Representation (IR)

The Intermediate Representation (IR) is a tree of frozen dataclass objects for the layer in between parsing and code generation. It represents Polars semantics, though in the case of most scalar operations, there is little difference between Polars and BigQuery.

Importantly, the IR doesn't need to represent the full Polars expression tree. Anything we can't yet parse gets turned into an "Unsupported" node.


#### SQL code generation

An important trick regarding Unsupported. It's OK to return more rows than required and then filter client-side:


*   a AND b where one of a/b is Unsupported can return just one of the branches and be less selective than either combined.
