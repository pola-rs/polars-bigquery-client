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

```powershell
$ErrorActionPreference = "Stop"
& $env:MASK rust-arrow-bigquery build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $env:MASK py-arrow-bigquery build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $env:MASK py-polars-bigquery build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $env:MASK docs build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## rust-arrow-bigquery

> Commands for the `arrow-bigquery` Rust crate (`crates/arrow-bigquery`).

### build

> Compile the Rust workspace and unit test binaries (`crates/arrow-bigquery`).

```bash
set -euo pipefail
cargo test --workspace --exclude py-arrow-bigquery --lib --bins --all-features --no-run
```

```powershell
$ErrorActionPreference = "Stop"
cargo test --workspace --exclude py-arrow-bigquery --lib --bins --all-features --no-run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

### test

> Run tests for `crates/arrow-bigquery`.

#### unit

> Run Rust unit tests (`crates/arrow-bigquery`), matching `.github/workflows/test-rust-unit.yml`.

```bash
set -euo pipefail
cargo test --workspace --exclude py-arrow-bigquery --lib --bins --all-features
```

```powershell
$ErrorActionPreference = "Stop"
cargo test --workspace --exclude py-arrow-bigquery --lib --bins --all-features
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

#### integration

> Run Rust BigQuery integration tests (`crates/arrow-bigquery/tests/integration_test.rs`). Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to run BigQuery integration tests}"
cargo test --package arrow-bigquery --test integration_test --all-features
```

```powershell
$ErrorActionPreference = "Stop"
if (-not $env:GOOGLE_CLOUD_PROJECT) {
  Write-Error "Set GOOGLE_CLOUD_PROJECT to run BigQuery integration tests"
  exit 1
}
cargo test --package arrow-bigquery --test integration_test --all-features
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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

```powershell
$ErrorActionPreference = "Stop"
$uvArgs = @()
if ($env:resolution) {
  $uvArgs += @("--resolution", $env:resolution)
}
if ($env:upgrade -eq "true") {
  $uvArgs += "--upgrade"
}
uv sync @uvArgs --project py-arrow-bigquery --group dev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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

```powershell
$ErrorActionPreference = "Stop"
$uvArgs = @()
if ($env:resolution) {
  $uvArgs += @("--resolution", $env:resolution)
}
if ($env:upgrade -eq "true") {
  $uvArgs += "--upgrade"
}
uv run @uvArgs --project py-arrow-bigquery --group test pytest -n auto py-arrow-bigquery/tests/unit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

#### integration

> Run `py-arrow-bigquery` integration tests (if present).

```bash
set -euo pipefail
echo "No standalone integration tests in py-arrow-bigquery; covered by rust-arrow-bigquery and py-polars-bigquery."
```

```powershell
Write-Host "No standalone integration tests in py-arrow-bigquery; covered by rust-arrow-bigquery and py-polars-bigquery."
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

```powershell
$ErrorActionPreference = "Stop"
$uvArgs = @()
if ($env:resolution) {
  $uvArgs += @("--resolution", $env:resolution)
}
if ($env:upgrade -eq "true") {
  $uvArgs += "--upgrade"
}
uv sync @uvArgs --project py-polars-bigquery --group dev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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

```powershell
$ErrorActionPreference = "Stop"
$uvArgs = @()
if ($env:resolution) {
  $uvArgs += @("--resolution", $env:resolution)
}
if ($env:upgrade -eq "true") {
  $uvArgs += "--upgrade"
}
uv run @uvArgs --project py-polars-bigquery --group test pytest -n auto py-polars-bigquery/tests/unit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

#### integration

> Run `py-polars-bigquery` system/integration tests (`py-polars-bigquery/tests/system`), matching `.github/workflows/test-py-polars-integration.yml`. Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to run BigQuery integration tests}"
uv run --project py-polars-bigquery --group test pytest py-polars-bigquery/tests/system
```

```powershell
$ErrorActionPreference = "Stop"
if (-not $env:GOOGLE_CLOUD_PROJECT) {
  Write-Error "Set GOOGLE_CLOUD_PROJECT to run BigQuery integration tests"
  exit 1
}
uv run --project py-polars-bigquery --group test pytest py-polars-bigquery/tests/system
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

#### benchmark

> Run `py-polars-bigquery` local benchmarks (`py-polars-bigquery/tests/benchmark`) with `pytest-benchmark`. Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to run BigQuery benchmark tests}"
uv run --project py-polars-bigquery --group test pytest --benchmark-only py-polars-bigquery/tests/benchmark
```

```powershell
$ErrorActionPreference = "Stop"
if (-not $env:GOOGLE_CLOUD_PROJECT) {
  Write-Error "Set GOOGLE_CLOUD_PROJECT to run BigQuery benchmark tests"
  exit 1
}
uv run --project py-polars-bigquery --group test pytest --benchmark-only py-polars-bigquery/tests/benchmark
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## docs

> Commands relating to the documentation (`mkdocs.yml`).

### build

> Build the static HTML documentation into `site/` with strict validation enabled.

```bash
set -euo pipefail
uv run --with-requirements docs/source/requirements.txt mkdocs build --strict
```

```powershell
$ErrorActionPreference = "Stop"
uv run --with-requirements docs/source/requirements.txt mkdocs build --strict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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

```powershell
$ErrorActionPreference = "Stop"
$devAddr = if ($env:addr) { $env:addr } else { "127.0.0.1:8000" }
uv run --with-requirements docs/source/requirements.txt mkdocs serve --dev-addr $devAddr
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## lint

> Run ruff lint and formatting checks against the repository and `docs/source/`.

```bash
set -euo pipefail
uv run --project py-polars-bigquery --group dev ruff check --no-fix docs/source/
uv run --project py-polars-bigquery --group dev ruff format --diff docs/source/
```

```powershell
$ErrorActionPreference = "Stop"
uv run --project py-polars-bigquery --group dev ruff check --no-fix docs/source/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run --project py-polars-bigquery --group dev ruff format --diff docs/source/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
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

```powershell
$ErrorActionPreference = "Stop"
$maskArgs = @()
if ($env:resolution) {
  $maskArgs += @("--resolution", $env:resolution)
}
if ($env:upgrade -eq "true") {
  $maskArgs += "--upgrade"
}
& $env:MASK rust-arrow-bigquery test unit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $env:MASK py-arrow-bigquery test unit @maskArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $env:MASK py-polars-bigquery test unit @maskArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

### integration

> Run all BigQuery integration tests (`rust-arrow-bigquery`, `py-arrow-bigquery`, and `py-polars-bigquery`). Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
$MASK rust-arrow-bigquery test integration
$MASK py-arrow-bigquery test integration
$MASK py-polars-bigquery test integration
```

```powershell
$ErrorActionPreference = "Stop"
& $env:MASK rust-arrow-bigquery test integration
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $env:MASK py-arrow-bigquery test integration
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $env:MASK py-polars-bigquery test integration
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

### benchmark

> Run all BigQuery benchmark tests locally (`py-polars-bigquery`). Requires `GOOGLE_CLOUD_PROJECT`.

```bash
set -euo pipefail
$MASK py-polars-bigquery test benchmark
```

```powershell
$ErrorActionPreference = "Stop"
& $env:MASK py-polars-bigquery test benchmark
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```
