ARG UV_VERSION=0.12.19
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM rust:bookworm AS base

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Install pinned uv and uvx from the official image
COPY --from=uv /uv /uvx /usr/local/bin/

# Install cmake (needed for aws-lc-sys), cargo-chef, and mask CLI (statically-linked musl binaries)
ARG CARGO_CHEF_VERSION=0.1.78
ARG CARGO_CHEF_SHA256=aca691abfbfbbe00d482e0ed2249eec3091b65e96a0ce92947fb5b254d48b16d
ARG MASK_VERSION=0.11.7
ARG MASK_SHA256=5c9fed48ecd6a9cbbf7332d67258930c0b2fcc18689850ce617b899c0eeae0c9
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    unzip \
    xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && curl --retry 5 --retry-delay 2 --retry-all-errors -fsSL \
       "https://github.com/LukeMathWalker/cargo-chef/releases/download/v${CARGO_CHEF_VERSION}/cargo-chef-x86_64-unknown-linux-musl.tar.xz" \
       -o /tmp/cargo-chef.tar.xz \
    && echo "${CARGO_CHEF_SHA256}  /tmp/cargo-chef.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/cargo-chef.tar.xz --strip-components=1 -C /usr/local/bin cargo-chef-x86_64-unknown-linux-musl/cargo-chef \
    && chmod +x /usr/local/bin/cargo-chef \
    && rm /tmp/cargo-chef.tar.xz \
    && curl --retry 5 --retry-delay 2 --retry-all-errors -fsSL \
       "https://github.com/jacobdeichert/mask/releases/download/mask/${MASK_VERSION}/mask-${MASK_VERSION}-x86_64-unknown-linux-musl.zip" \
       -o /tmp/mask.zip \
    && echo "${MASK_SHA256}  /tmp/mask.zip" | sha256sum -c - \
    && unzip -j /tmp/mask.zip "mask-${MASK_VERSION}-x86_64-unknown-linux-musl/mask" -d /usr/local/bin \
    && chmod +x /usr/local/bin/mask \
    && rm /tmp/mask.zip

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/opt/cache/target \
    CARGO_NET_RETRY=5 \
    RUSTUP_MAX_RETRIES=5 \
    UV_CACHE_DIR=/opt/cache/uv \
    UV_HTTP_RETRIES=5 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin \
    UV_PYTHON=3.14 \
    RUSTFLAGS="-C debuginfo=0" \
    RUST_BACKTRACE=1 \
    PYTHONUTF8=1

# Install Python 3.14 via uv
RUN uv python install 3.14 --default

# Install Rust toolchain specified by rust-toolchain.toml and remove unused pre-installed toolchains
WORKDIR /tmp/warmup
COPY rust-toolchain.toml ./
RUN rustup set profile minimal \
    && rustup show \
    && ACTIVE_TOOLCHAIN="$(rustup show active-toolchain | awk '{print $1}')" \
    && rustup default "${ACTIVE_TOOLCHAIN}" \
    && for tc in $(rustup toolchain list | awk '{print $1}'); do \
         if [ "${tc}" != "${ACTIVE_TOOLCHAIN}" ]; then rustup toolchain uninstall "${tc}"; fi; \
       done

FROM base AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM base
# 1. Pre-warm Cargo cache in an isolated layer so Python lockfile updates never trigger Rust rebuilds
COPY --from=planner /tmp/warmup/recipe.json recipe.json
RUN cargo chef cook --locked --package arrow-bigquery --tests --all-features --recipe-path recipe.json \
    && cargo chef cook --locked --release --package py-arrow-bigquery --features extension-module --recipe-path recipe.json \
    && cargo clean -p arrow-bigquery -p py-arrow-bigquery \
    && cargo clean --release -p arrow-bigquery -p py-arrow-bigquery

# 2. Pre-warm uv cache across all workspace packages and dependency groups (including maturin/patchelf)
COPY pyproject.toml uv.lock ./
COPY py-arrow-bigquery/pyproject.toml py-arrow-bigquery/pyproject.toml
COPY py-polars-bigquery/pyproject.toml py-polars-bigquery/pyproject.toml
RUN uv sync --locked --all-packages --all-groups --no-install-workspace \
    && rm -rf /tmp/warmup

ENV UV_LINK_MODE=copy
WORKDIR /workspace
