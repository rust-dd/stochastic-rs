#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
artifact=stochastic-rs-stochastic/src/noise/fgn/metal/artifact
if [[ "${1:-}" == "--check" ]]; then
  shasum -a 256 -c "$artifact/sources.sha256"
  exit
fi
log=$(mktemp)
trap 'rm -f "$log"' EXIT
cargo metal build -p stochastic-rs-stochastic --math-mode fast 2>&1 | tee "$log"
directory=$(sed -nE 's/^(Built|Cached) Metal artifact (.*)$/\2/p' "$log")
test -d "$directory"
mkdir -p "$artifact"
cp "$directory/manifest.json" "$directory/kernels.metallib" "$artifact/"
shasum -a 256 Cargo.toml Cargo.lock stochastic-rs-metal-kernels/Cargo.toml stochastic-rs-metal-kernels/src/*.rs > "$artifact/sources.sha256"
