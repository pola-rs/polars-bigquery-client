"""Exceptions raised by `polars_bigquery`."""


class BigQueryError(Exception):
    """Error raised when a BigQuery REST API request or query job fails or times out."""
