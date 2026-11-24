"""Exceptions and warnings raised by `polars_bigquery`."""

from __future__ import annotations

from arrow_bigquery.exceptions import BigQueryError, UnsupportedTypeError


class UnsupportedTypeWarning(UserWarning):
    """Warning emitted when `Client.scan_table` omits unsupported BigQuery columns.

    BigQuery tables may contain columns whose types (or nested subfield types)
    are not supported by `polars_bigquery`, such as `BIGNUMERIC`, `INTERVAL`,
    `RANGE<T>`, non-microsecond `TIMESTAMP`, or unrecognized future BigQuery
    types or field modes. When `Client.scan_table` inspects table metadata and
    at least one supported column (or ingestion-time partition pseudo-column)
    remains, the unsupported top-level columns are omitted from the `LazyFrame`
    schema and this warning is emitted.

    You can filter or turn this warning into an error with the standard
    `warnings` module:

    ```python
    import warnings
    from polars_bigquery.exceptions import UnsupportedTypeWarning

    warnings.filterwarnings("error", category=UnsupportedTypeWarning)
    ```
    """


__all__ = ["BigQueryError", "UnsupportedTypeError", "UnsupportedTypeWarning"]
