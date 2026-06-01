# How to contribute

To contribute to this project, you agree to follow the [Polars Code of Conduct](https://github.com/pola-rs/.github/blob/master/CODE_OF_CONDUCT.md).

## Setting up a development environment

Install `uv`.

Authenticate with Google Cloud using `gcloud auth application-default login`.

## Running the test suite

Set the `GOOGLE_CLOUD_PROJECT` environment variable.

Run all tests:

```
LD_LIBRARY_PATH="$HOME/.pyenv/versions/3.14.3/lib" \
  source .venv/bin/activate \
  && cargo test
```
