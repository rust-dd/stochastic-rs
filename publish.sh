#!/usr/bin/env bash
# Publish all stochastic-rs sub-crates to crates.io in topological order,
# then publish the umbrella crate.
#
# Usage:
#   ./publish.sh                 # publish for real (runs gate: fmt + clippy + tests)
#   ./publish.sh --dry-run       # one `cargo publish --workspace --dry-run`, no upload (still runs gate)
#   ./publish.sh --allow-dirty   # publish with uncommitted changes
#   ./publish.sh --skip-gate     # bypass fmt/clippy/test gate (NOT recommended)
#
# Notes:
# - `stochastic-rs-py` is intentionally skipped (publish = false in its
#   Cargo.toml — it is a maturin-built cdylib for PyPI, not a crates.io
#   crate); release via `.github/workflows/pypi.yml`.
# - `--dry-run` is one `cargo publish --workspace --dry-run`, verified against a local registry of
#   the siblings; a per-crate dry run would stop at the second crate after a bump.
# - Crates already published at the current local version are skipped, so the
#   script is safe to re-run if a publish fails partway through.
# - Modern cargo (>=1.66) waits for the registry index to sync after each
#   publish, so no manual sleep is needed.

set -euo pipefail

DRY_RUN=()
ALLOW_DIRTY=()
SKIP_GATE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)     DRY_RUN=(--dry-run) ;;
    --allow-dirty) ALLOW_DIRTY=(--allow-dirty) ;;
    --skip-gate)   SKIP_GATE=1 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

# Pre-publish gate: fmt, clippy and the workspace tests without the py cdylib; `--skip-gate` is
# for a re-run after a transient crates.io failure on a known-good commit.
if [[ $SKIP_GATE -eq 0 ]]; then
  echo "==> pre-publish gate: cargo fmt --check"
  cargo fmt --all -- --check

  echo "==> pre-publish gate: cargo clippy --workspace --no-default-features"
  cargo clippy --workspace --no-default-features --all-targets --locked -- -D warnings

  echo "==> pre-publish gate: cargo clippy --workspace --all-features"
  cargo clippy --workspace --all-features --all-targets --locked -- -D warnings

  echo "==> pre-publish gate: cargo test --workspace --exclude stochastic-rs-py"
  # No default features, to keep the gate tractable; CI covers the `ai` and `viz` combinations
  # before tagging.
  cargo test --workspace --exclude stochastic-rs-py --no-default-features --locked
else
  echo "==> WARNING: --skip-gate is set; skipping clippy + workspace test"
fi

# (crate, extra_flags) — order matters: each crate must come AFTER its deps.
PUBLISH_ORDER=(
  "stochastic-rs-core"            # no internal deps
  "stochastic-rs-distributions"   # core
  "stochastic-rs-stochastic"      # core, distributions
  "stochastic-rs-stats"           # core, distributions, stochastic
  "stochastic-rs-copulas"         # core, distributions
  "stochastic-rs-quant"           # core, distributions, stochastic, stats
  "stochastic-rs-ai"              # core, distributions, quant
  "stochastic-rs"                 # umbrella — last
)

local_version() {
  # A member's resolved version: the sub-crates inherit `version.workspace = true`, which a grep of
  # their Cargo.toml cannot see, so cargo metadata resolves it.
  local crate="$1"
  cargo metadata --no-deps --format-version 1 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(next((p['version'] for p in d['packages'] if p['name']=='$crate'), ''))" \
    2>/dev/null
}

registry_has_version() {
  # "yes" if $crate@$version is already on the crates.io index, else "no"; crates.io refuses a
  # request without a User-Agent, which would read as unpublished.
  local crate="$1" version="$2"
  curl -fsS -H "User-Agent: stochastic-rs publish.sh (dancixx@gmail.com)" \
    "https://crates.io/api/v1/crates/$crate/versions" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if '$version' in [v['num'] for v in d.get('versions',[])] else 'no')" \
    2>/dev/null || echo "no"
}

publish_one() {
  local spec="$1"
  # spec may be "crate-name" or "crate-name --no-verify"
  local crate="${spec%% *}"
  local extra=()
  if [[ "$spec" == *" "* ]]; then
    read -r -a extra <<< "${spec#* }"
  fi

  local lv published
  lv=$(local_version "$crate")
  published=$(registry_has_version "$crate" "$lv")

  echo
  echo "==> $crate (local=$lv, on crates.io=$published)"

  if [[ "$published" == "yes" ]]; then
    echo "    already published at $lv — skipping"
    return 0
  fi

  set -x
  cargo publish -p "$crate" --locked \
    ${extra[@]+"${extra[@]}"} \
    ${ALLOW_DIRTY[@]+"${ALLOW_DIRTY[@]}"}
  set +x
}

if [[ ${#DRY_RUN[@]} -gt 0 ]]; then
  set -x
  cargo publish --workspace --dry-run --locked ${ALLOW_DIRTY[@]+"${ALLOW_DIRTY[@]}"}
  set +x
  exit 0
fi

for spec in "${PUBLISH_ORDER[@]}"; do
  publish_one "$spec"
done

echo
echo "All crates processed."
