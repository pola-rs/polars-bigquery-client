FROM rust:bookworm

# Install uv and uvx from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install cmake (needed for aws-lc-sys) and mask CLI (statically-linked musl binary)
ARG MASK_VERSION=0.11.7
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/jacobdeichert/mask/releases/download/mask/${MASK_VERSION}/mask-${MASK_VERSION}-x86_64-unknown-linux-musl.zip" -o /tmp/mask.zip \
    && unzip -j /tmp/mask.zip "mask-${MASK_VERSION}-x86_64-unknown-linux-musl/mask" -d /usr/local/bin \
    && chmod +x /usr/local/bin/mask \
    && rm /tmp/mask.zip

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/opt/cache/target \
    UV_CACHE_DIR=/opt/cache/uv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin \
    UV_PYTHON=3.14 \
    RUSTFLAGS="-C debuginfo=0" \
    RUST_BACKTRACE=1 \
    PYTHONUTF8=1

# Install Python 3.14 via uv
RUN uv python install 3.14 --default

# Install Rust toolchain specified by rust-toolchain.toml
WORKDIR /tmp/warmup
COPY rust-toolchain.toml ./
RUN rustup show

# Pre-warm Cargo and uv caches using project manifests and lockfiles
COPY Cargo.toml Cargo.lock pyproject.toml uv.lock ./
COPY crates/arrow-bigquery/Cargo.toml crates/arrow-bigquery/Cargo.toml
COPY py-arrow-bigquery/Cargo.toml py-arrow-bigquery/pyproject.toml py-arrow-bigquery/
COPY py-polars-bigquery/pyproject.toml py-polars-bigquery/pyproject.toml

RUN mkdir -p crates/arrow-bigquery/src crates/arrow-bigquery/tests \
             py-arrow-bigquery/src py-arrow-bigquery/arrow_bigquery \
             py-polars-bigquery/polars_bigquery \
    && touch crates/arrow-bigquery/src/lib.rs \
             crates/arrow-bigquery/tests/integration_test.rs \
             py-arrow-bigquery/src/lib.rs \
             py-arrow-bigquery/arrow_bigquery/__init__.py \
             py-polars-bigquery/polars_bigquery/__init__.py \
    && cargo test --package arrow-bigquery --test integration_test --all-features --no-run \
    && cargo build --release --package py-arrow-bigquery --features extension-module \
    && uv sync --project py-polars-bigquery --group dev --group test --no-install-workspace \
    && cargo clean -p arrow-bigquery -p py-arrow-bigquery \
    && cargo clean --release -p arrow-bigquery -p py-arrow-bigquery \
    && rm -rf /tmp/warmup

ENV UV_LINK_MODE=copy
WORKDIR /workspace
