import datetime
from typing import Any

class BigQueryError(RuntimeError):
    """Error raised when a BigQuery API request, stream, or query job fails or times out."""

class UnsupportedTypeError(BigQueryError):
    """Error raised when a BigQuery schema contains a data type that cannot be decoded or exported over FFI."""

class ArrowStreamExporter:
    """Bridge exporting BigQuery Arrow record batches via the Arrow C Stream interface."""

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any: ...

class BigQueryTableId:
    """Immutable BigQuery table identifier (`project_id.dataset_id.table_id`)."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
    ) -> None: ...
    @staticmethod
    def from_string(s: str) -> BigQueryTableId: ...
    @staticmethod
    def from_str(s: str) -> BigQueryTableId: ...
    @staticmethod
    def parse(table_id: Any) -> BigQueryTableId: ...
    @property
    def project_id(self) -> str: ...
    @property
    def dataset_id(self) -> str: ...
    @property
    def table_id(self) -> str: ...
    @property
    def project(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Client:
    """Native Rust BigQuery client keeping connections and OAuth2 tokens cached."""

    def __init__(
        self,
        *,
        quota_project_id: str,
        credentials_provider: Any = None,
        user_agent: str | None = None,
        bq_endpoint: str | None = None,
        bqstorage_endpoint: str | None = None,
        preserve_datetime_extension: bool = False,
        validate_ffi_types: bool = True,
    ) -> None: ...
    def read_table(
        self,
        table: BigQueryTableId,
        *,
        arrow_buffer_compression: str = "lz4frame",
        maintain_order: bool = False,
        max_stream_count: int | None = None,
        row_restriction: str = "",
        sample_percentage: float | None = None,
        selected_fields: list[str] | None = None,
        snapshot_time: datetime.datetime | None = None,
    ) -> ArrowStreamExporter: ...
    def read_query(
        self,
        query: str,
        *,
        arrow_buffer_compression: str = "lz4frame",
        dry_run: bool = False,
        maintain_order: bool = True,
        use_query_cache: bool | None = None,
    ) -> ArrowStreamExporter: ...

class DropFlag:
    def is_set(self) -> bool: ...

def _create_test_exporter() -> ArrowStreamExporter: ...
def _test_create_exporter_with_drop_flag() -> tuple[ArrowStreamExporter, DropFlag]: ...
