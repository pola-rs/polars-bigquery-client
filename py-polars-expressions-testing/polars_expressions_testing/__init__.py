from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

__all__ = ["unsupported"]


def unsupported(expr: IntoExpr) -> pl.Expr:
    """No-op Polars expression plugin for testing unsupported expressions."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="unsupported",
        args=expr,
        is_elementwise=True,
    )
