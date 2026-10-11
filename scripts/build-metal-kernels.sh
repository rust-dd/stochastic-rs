#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
artifact=stochastic-rs-stochastic/src/device/metal/artifact
python3 scripts/generate-euler-kernels.py --check
source_hashes() {
  shasum -a 256 Cargo.toml Cargo.lock scripts/build-metal-kernels.sh \
    scripts/generate-euler-kernels.py stochastic-rs-metal-kernels/Cargo.toml \
    stochastic-rs-stochastic/Cargo.toml
  find stochastic-rs-metal-kernels/src stochastic-rs-stochastic/src/euler/families/definitions \
    -type f -name '*.rs' -print | LC_ALL=C sort | while IFS= read -r source; do
    shasum -a 256 "$source"
  done
  shasum -a 256 "$artifact/bindings.rs" "$artifact/kernels.metallib" "$artifact/manifest.json"
}

hashes=$(mktemp)
log=$(mktemp)
trap 'rm -f "$hashes" "$log"' EXIT
if [[ "${1:-}" == "--check" ]]; then
  source_hashes > "$hashes"
  if ! diff -u "$artifact/sources.sha256" "$hashes"; then
    echo 'Metal kernel sources or artifacts changed; run scripts/build-metal-kernels.sh.' >&2
    exit 1
  fi
  exit
fi
cargo metal build -p stochastic-rs-stochastic --math-mode fast 2>&1 | tee "$log"
directory=$(sed -nE 's/^(Built|Cached) Metal artifact (.*)$/\2/p' "$log")
test -d "$directory"
mkdir -p "$artifact"
cp "$directory/manifest.json" "$directory/kernels.metallib" "$directory/bindings.rs" "$artifact/"
source_hashes > "$artifact/sources.sha256"
