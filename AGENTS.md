# Agent Guidelines

## Python Environment and Package Management

- Always use `uv` for running Python commands, managing dependencies, and executing test suites.
- Do not use global `python`, `pip`, or `pytest` directly. Always prefix with `uv run` (e.g. `uv run pytest`).
- To run unit tests:
  - For `py-arrow-bigquery`: `uv run pytest py-arrow-bigquery`
  - For `py-polars-bigquery`: `uv run pytest py-polars-bigquery`
- For Rust testing: `cargo test`
