# Tasks

Development and documentation tasks for `polars-bigquery-client`.

Requires Rust (install with `rustup`), [`uv`](https://docs.astral.sh/uv/), and
[`mask`](https://github.com/jacobdeichert/mask).

For integration and benchmark tests, authenticate with Google Cloud
(`gcloud auth application-default login`) and set the `GOOGLE_CLOUD_PROJECT`
environment variable.

## build

> Build all sub-projects for development.

```bash
set -euo pipefail
$MASK rust-arrow-bigquery build
$MASK py-arrow-bigquery build
$MASK py-polars-bigquery build
$MASK docs build
```

## rust-arrow-bigquery

> Commands for the `arrow-bigquery` Rust crate (`crates/arrow-bigquery`).

### build

> Compile the Rust workspace and unit test binaries (`crates/arrow-bigquery`).

```bash
set -euo pipefail
cargo test --workspace --exclude py-arrow-bigquery --lib --bins --all-features --no-run
```

### test

> Run tests for `crates/arrow-bigquery`.

#### unit

> Run Rust unit tests (`crates/arrow-bigquery`), matching `.github/workflows/test-rust-unit.yml`.

```bash
set -euo pipefail
cargo test --workspace --exclude py-arrow-bigquery --lib --bins --all-features
```

#### integration

> Run Rust BigQuery integration tests (`crates/arrow-bigquery/tests/integration_test.rs`). Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to run BigQuery integration tests}"
cargo test --package arrow-bigquery --test integration_test --all-features
```

## py-arrow-bigquery

> Commands for the `arrow-bigquery` Python package (`py-arrow-bigquery`).

### build

> Build the `py-arrow-bigquery` Rust extension and install editable dependencies via `uv`.

**OPTIONS**
* resolution
    * flags: -r --resolution
    * type: string
    * desc: Dependency resolution strategy for uv (highest, lowest, lowest-direct)
* upgrade
    * flags: -U --upgrade
    * desc: Allow package upgrades, ignoring pinned versions in uv.lock

```bash
set -euo pipefail
uv_args=()
if [ -n "${resolution:-}" ]; then
  uv_args+=(--resolution "$resolution")
fi
if [ "${upgrade:-}" = "true" ]; then
  uv_args+=(--upgrade)
fi
uv sync "${uv_args[@]}" --project py-arrow-bigquery --group dev
```

### test

> Run tests for `py-arrow-bigquery`.

#### unit

> Run `py-arrow-bigquery` unit tests, matching `.github/workflows/test-py-arrow-unit.yml`.

**OPTIONS**
* resolution
    * flags: -r --resolution
    * type: string
    * desc: Dependency resolution strategy for uv (highest, lowest, lowest-direct)
* upgrade
    * flags: -U --upgrade
    * desc: Allow package upgrades, ignoring pinned versions in uv.lock

```bash
set -euo pipefail
uv_args=()
if [ -n "${resolution:-}" ]; then
  uv_args+=(--resolution "$resolution")
fi
if [ "${upgrade:-}" = "true" ]; then
  uv_args+=(--upgrade)
fi
uv run "${uv_args[@]}" --project py-arrow-bigquery --group test pytest -n auto py-arrow-bigquery/tests/unit
```

#### integration

> Run `py-arrow-bigquery` integration tests (if present).

```bash
set -euo pipefail
echo "No standalone integration tests in py-arrow-bigquery; covered by rust-arrow-bigquery and py-polars-bigquery."
```

## py-polars-bigquery

> Commands for the `polars-bigquery` Python package (`py-polars-bigquery`).

### build

> Sync dependencies and install `py-polars-bigquery` (and workspace member `arrow-bigquery`) in editable mode via `uv`.

**OPTIONS**
* resolution
    * flags: -r --resolution
    * type: string
    * desc: Dependency resolution strategy for uv (highest, lowest, lowest-direct)
* upgrade
    * flags: -U --upgrade
    * desc: Allow package upgrades, ignoring pinned versions in uv.lock

```bash
set -euo pipefail
uv_args=()
if [ -n "${resolution:-}" ]; then
  uv_args+=(--resolution "$resolution")
fi
if [ "${upgrade:-}" = "true" ]; then
  uv_args+=(--upgrade)
fi
uv sync "${uv_args[@]}" --project py-polars-bigquery --group dev
```

### test

> Run tests for `py-polars-bigquery`.

#### unit

> Run `py-polars-bigquery` unit tests, matching `.github/workflows/test-py-polars-unit.yml`.

**OPTIONS**
* resolution
    * flags: -r --resolution
    * type: string
    * desc: Dependency resolution strategy for uv (highest, lowest, lowest-direct)
* upgrade
    * flags: -U --upgrade
    * desc: Allow package upgrades, ignoring pinned versions in uv.lock

```bash
set -euo pipefail
uv_args=()
if [ -n "${resolution:-}" ]; then
  uv_args+=(--resolution "$resolution")
fi
if [ "${upgrade:-}" = "true" ]; then
  uv_args+=(--upgrade)
fi
uv run "${uv_args[@]}" --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit
```

#### integration

> Run `py-polars-bigquery` system/integration tests (`py-polars-bigquery/tests/system`), matching `.github/workflows/test-py-polars-integration.yml`. Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to run BigQuery integration tests}"
uv run --project py-polars-bigquery --group test pytest py-polars-bigquery/tests/system
```

#### benchmark

> Run `py-polars-bigquery` local benchmarks (`py-polars-bigquery/tests/benchmark`) with `pytest-benchmark`. Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to run BigQuery benchmark tests}"
uv run --project py-polars-bigquery --group test pytest --benchmark-only py-polars-bigquery/tests/benchmark
```

## docs

> Commands relating to the documentation (`mkdocs.yml`).

### build

> Build the static HTML documentation into `site/` with strict validation enabled.

```bash
set -euo pipefail
uv run --with-requirements docs/source/requirements.txt mkdocs build --strict
```

### serve

> Start a local live-reloading MkDocs development server (default: http://127.0.0.1:8000).

**OPTIONS**
* addr
    * flags: -a --addr
    * type: string
    * desc: IP address and port to serve documentation on (default: 127.0.0.1:8000)

```bash
set -euo pipefail
uv run --with-requirements docs/source/requirements.txt mkdocs serve --dev-addr "${addr:-127.0.0.1:8000}"
```

## lint

> Run ruff lint and formatting checks against the repository and `docs/source/`.

```bash
set -euo pipefail
uv run --project py-polars-bigquery --group dev ruff check --no-fix docs/source/
uv run --project py-polars-bigquery --group dev ruff format --diff docs/source/
```

## test

> Run project test suites across Rust and Python workspaces.

### unit

> Run all unit tests (`rust-arrow-bigquery`, `py-arrow-bigquery`, and `py-polars-bigquery`).

**OPTIONS**
* resolution
    * flags: -r --resolution
    * type: string
    * desc: Dependency resolution strategy for uv (highest, lowest, lowest-direct)
* upgrade
    * flags: -U --upgrade
    * desc: Allow package upgrades, ignoring pinned versions in uv.lock

```bash
set -euo pipefail
mask_args=()
if [ -n "${resolution:-}" ]; then
  mask_args+=(--resolution "$resolution")
fi
if [ "${upgrade:-}" = "true" ]; then
  mask_args+=(--upgrade)
fi
$MASK rust-arrow-bigquery test unit
$MASK py-arrow-bigquery test unit "${mask_args[@]}"
$MASK py-polars-bigquery test unit "${mask_args[@]}"
```

### integration

> Run all BigQuery integration tests (`rust-arrow-bigquery`, `py-arrow-bigquery`, and `py-polars-bigquery`). Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
$MASK rust-arrow-bigquery test integration
$MASK py-arrow-bigquery test integration
$MASK py-polars-bigquery test integration
```

### benchmark

> Run all BigQuery benchmark tests locally (`py-polars-bigquery`). Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
$MASK py-polars-bigquery test benchmark
```
