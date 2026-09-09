"""Utilies for parsing BigQuery table IDs"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

_TABLE_REFEREENCE_PATTERN = re.compile(
    # In the past, organizations could prefix their project IDs with a domain
    # name. Such projects still exist, especially at Google.
    r"^(?P<legacy_project_domain>[^:]+:)?"
    r"(?P<project>[^.]+)\."
    # Match dataset or catalog + namespace.
    #
    # Namespace could be arbitrarily deeply nested in Iceberg/BigLake. Support
    # this without catastrophic backtracking by moving the trailing "." to the
    # table group.
    r"(?P<inner_parts>.*)"
    # Table names can't contain ".", as that's used as the separator.
    r"\.(?P<table>[^.]+)$"
)


@dataclasses.dataclass(frozen=True)
class BigQueryTableId:
    project_id: str
    dataset_id: str
    table_id: str


def parse_table_id(table_id: Any) -> BigQueryTableId:
    """Turn a string or BigQuery table object into a BigQueryTableId.

    Raises:
        ValueError: If the table ID is invalid.
        TypeError: If the table ID is not a string or BigQuery table object.
    """
    if (
        hasattr(table_id, "project")
        and hasattr(table_id, "dataset_id")
        and hasattr(table_id, "table_id")
    ):
        return BigQueryTableId(
            project_id=table_id.project,
            dataset_id=table_id.dataset_id,
            table_id=table_id.table_id,
        )

    if not isinstance(table_id, str):
        raise TypeError(f"Expected table_id to be a string, got {type(table_id)}")

    if "`" in table_id:
        raise ValueError(f"Invalid table ID: {table_id}")

    regex_match = _TABLE_REFEREENCE_PATTERN.match(table_id)
    if not regex_match:
        raise ValueError(f"Invalid table ID: {table_id}")

    inner_parts = regex_match.group("inner_parts").split(".")
    if any(part == "" for part in inner_parts):
        raise ValueError(f"Invalid table ID: {table_id}")

    return BigQueryTableId(
        project_id=regex_match.group("project"),
        dataset_id=".".join(inner_parts),
        table_id=regex_match.group("table"),
    )
