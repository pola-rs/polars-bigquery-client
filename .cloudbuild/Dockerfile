FROM rust:bookworm AS base

# Install uv and uvx from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install cmake (needed for aws-lc-sys), cargo-chef, and mask CLI (statically-linked musl binaries)
ARG CARGO_CHEF_VERSION=0.1.78
ARG MASK_VERSION=0.11.7
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    unzip \
    xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/LukeMathWalker/cargo-chef/releases/download/v${CARGO_CHEF_VERSION}/cargo-chef-x86_64-unknown-linux-musl.tar.xz" \
       | tar -xJ --strip-components=1 -C /usr/local/bin cargo-chef-x86_64-unknown-linux-musl/cargo-chef \
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

FROM base AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM base
# Pre-warm Cargo and uv caches using cargo-chef recipe and Python manifests/lockfiles
COPY --from=planner /tmp/warmup/recipe.json recipe.json
COPY pyproject.toml uv.lock ./
COPY py-arrow-bigquery/pyproject.toml py-arrow-bigquery/pyproject.toml
COPY py-polars-bigquery/pyproject.toml py-polars-bigquery/pyproject.toml

RUN cargo chef cook --package arrow-bigquery --tests --benches --all-features --recipe-path recipe.json \
    && cargo chef cook --release --package py-arrow-bigquery --features extension-module --recipe-path recipe.json \
    && uv sync --project py-polars-bigquery --group dev --group test --no-install-workspace \
    && rm -rf /tmp/warmup

ENV UV_LINK_MODE=copy
WORKDIR /workspace
