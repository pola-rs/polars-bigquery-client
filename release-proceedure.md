# Releasing polars-bigquery

- Update the CHANGELOG.
- Update the version in `py-polars-bigquery/pyproject.toml`.
- Update the version in `py-polars-bigquery/polars_bigquery/core/version.py`.
- Send PR.
- Merge.
- Trigger Build Wheels workflow.
- Download the wheels.
- Test locally with a downloaded built wheel.
- Upload and release with `twine`.
- Tag release on GitHub.
- Share on social media.
