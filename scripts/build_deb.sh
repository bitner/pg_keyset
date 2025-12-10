#!/bin/bash
set -e

PG_VERSION=$1
if [ -z "$PG_VERSION" ]; then
    echo "Usage: $0 <pg_version>"
    exit 1
fi

echo "Building for PostgreSQL $PG_VERSION..."

# Install dependencies
apt-get update
apt-get install -y \
    curl \
    build-essential \
    libclang-dev \
    pkg-config \
    libssl-dev \
    git \
    postgresql-server-dev-${PG_VERSION}

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install cargo-pgrx
cargo install cargo-pgrx --version 0.16.1 --locked

# Initialize pgrx with the system postgres
cargo pgrx init --pg${PG_VERSION}=/usr/bin/pg_config

# Package
# This creates artifacts in target/release/pg_keyset-pg${PG_VERSION}/
cargo pgrx package --pg-config /usr/bin/pg_config

# Prepare deb directory
DEB_DIR="deb_dist"
rm -rf $DEB_DIR
mkdir -p $DEB_DIR/DEBIAN
mkdir -p $DEB_DIR/usr/lib/postgresql/${PG_VERSION}/lib
mkdir -p $DEB_DIR/usr/share/postgresql/${PG_VERSION}/extension

# Copy artifacts
# We find the files in the package directory.
# Note: cargo-pgrx package output structure depends on the pg_config paths.
# It usually mirrors the absolute path.
# So if pg_config says /usr/lib/postgresql/15/lib, it creates target/release/pg_keyset-pg15/usr/lib/postgresql/15/lib/pg_keyset.so

PKG_DIR="target/release/pg_keyset-pg${PG_VERSION}"

echo "Copying files from $PKG_DIR..."
# Copy shared object
find $PKG_DIR -name "*.so" -exec cp {} $DEB_DIR/usr/lib/postgresql/${PG_VERSION}/lib/ \;
# Copy control and sql files
find $PKG_DIR -name "*.control" -exec cp {} $DEB_DIR/usr/share/postgresql/${PG_VERSION}/extension/ \;
find $PKG_DIR -name "*.sql" -exec cp {} $DEB_DIR/usr/share/postgresql/${PG_VERSION}/extension/ \;

# Create control file
# Get version from Cargo.toml
VERSION=$(grep '^version =' Cargo.toml | head -n1 | cut -d '"' -f 2)

cat <<EOF > $DEB_DIR/DEBIAN/control
Package: pg-keyset
Version: $VERSION
Section: database
Priority: optional
Architecture: amd64
Depends: postgresql-${PG_VERSION}
Maintainer: David Bitner <bitner@dbspatial.com>
Description: A PostgreSQL extension for keyset pagination.
