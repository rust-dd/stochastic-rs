#!/usr/bin/env bash
# Publish all stochastic-rs sub-crates to crates.io in topological order,
# then publish the umbrella crate.
#
# Usage:
#   ./publish.sh                 # publish for real
#   ./publish.sh --dry-run       # validate without uploading
#   ./publish.sh --allow-dirty   # publish with uncommitted changes
#
# Notes:
# - `stochastic-rs-py` is intentionally skipped (publish = false in its
#   Cargo.toml — it is a placeholder cdylib).
# - Crates already published at the current local version are skipped, so the
#   script is safe to re-run if a publish fails partway through.
# - Modern cargo (>=1.66) waits for the registry index to sync after each
#   publish, so no manual sleep is needed.

set -euo pipefail

DRY_RUN=()
ALLOW_DIRTY=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)     DRY_RUN=(--dry-run) ;;
    --allow-dirty) ALLOW_DIRTY=(--allow-dirty) ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

# (crate, extra_flags) — order matters: each crate must come AFTER its deps.
PUBLISH_ORDER=(
  "stochastic-rs-core"            # no internal deps
  "stochastic-rs-distributions"   # core
  "stochastic-rs-stochastic"      # core, distributions
  "stochastic-rs-stats"           # core, distributions, stochastic
  "stochastic-rs-copulas"         # core, distributions
  "stochastic-rs-quant"           # core, distributions, stochastic, stats, copulas
  "stochastic-rs-ai"              # core, distributions, stochastic
  "stochastic-rs-viz"             # core, distributions, stochastic
  "stochastic-rs"                 # umbrella — last
)

local_version() {
  # extract the [package] version from a crate's Cargo.toml
  local crate="$1"
  local manifest
  if [[ "$crate" == "stochastic-rs" ]]; then
    manifest="Cargo.toml"
  else
    manifest="$crate/Cargo.toml"
  fi
  awk '/^\[package\]/{p=1; next} /^\[/{p=0} p && /^version[[:space:]]*=/{gsub(/"/,"",$3); print $3; exit}' "$manifest"
}

registry_version() {
  # latest version on crates.io, "-" if not published
  local crate="$1"
  curl -fsS "https://crates.io/api/v1/crates/$crate" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('crate',{}).get('newest_version','-'))" \
    2>/dev/null || echo "-"
}

publish_one() {
  local spec="$1"
  # spec may be "crate-name" or "crate-name --no-verify"
  local crate="${spec%% *}"
  local extra=()
  if [[ "$spec" == *" "* ]]; then
    read -r -a extra <<< "${spec#* }"
  fi

  local lv rv
  lv=$(local_version "$crate")
  rv=$(registry_version "$crate")

  echo
  echo "==> $crate (local=$lv, registry=$rv)"

  if [[ ${#DRY_RUN[@]} -eq 0 && "$lv" == "$rv" ]]; then
    echo "    already published at $lv — skipping"
    return 0
  fi

  set -x
  cargo publish -p "$crate" \
    ${extra[@]+"${extra[@]}"} \
    ${DRY_RUN[@]+"${DRY_RUN[@]}"} \
    ${ALLOW_DIRTY[@]+"${ALLOW_DIRTY[@]}"}
  set +x
}

for spec in "${PUBLISH_ORDER[@]}"; do
  publish_one "$spec"
done

echo
echo "All crates processed."
