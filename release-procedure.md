# Releasing polars-bigquery

- Update the CHANGELOG.
- Update the version in `py-arrow-bigquery/pyproject.toml`.
- Update the version in `py-arrow-bigquery/arrow_bigquery/core/version.py`.
- Update the version in `py-polars-bigquery/pyproject.toml`, **and** update its `"arrow-bigquery >= <version>"` minimum dependency pin to match the released `arrow-bigquery` version so users always get matching Rust schema decoding / FFI safety behavior.
- Update the version in `py-polars-bigquery/polars_bigquery/core/version.py`.
- Update uv lock files. `uv lock`.
- Send PR.
- Merge.
- Trigger Build Wheels workflow.
- Download the wheels.
- Test locally with a downloaded built wheel.
- Tag and publish a GitHub Release (triggers the `Release` workflow to publish `arrow-bigquery` and `polars-bigquery` to PyPI).
- Share on social media.
