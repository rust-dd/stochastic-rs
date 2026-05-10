---
name: release-checklist
description: Step-by-step release workflow for stochastic-rs (rc → stable → patch). Invoke when bumping versions, cutting a tag, publishing to crates.io, or shipping wheels to PyPI.
---

# Release checklist — stochastic-rs

This SKILL formalises the release workflow used to ship rc.0 → rc.1 →
rc.2 → stable. Follow it whenever the user asks to "cut a release", "bump
to vX.Y.Z", "tag stable", "publish wheels", or similar.

The workflow has **8 stages**. Stages 1-5 are local and reversible; stages
6-8 publish to public registries and are not. Stop and confirm with the
user before stage 6.

## Stage 1 — pre-flight checks

Run from the workspace root (`/Users/danixx/Desktop/stochastic-rs`):

```bash
# 1.1 Working tree clean?
git status --porcelain     # must be empty (no uncommitted, no untracked)
git log -1 --oneline       # capture the HEAD commit for the changelog

# 1.2 Tests + clippy + features matrix
cargo test --workspace --exclude stochastic-rs-py --features openblas --no-fail-fast
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace --all-features                       # catches the §4.1 feature-flag traps
cargo check --workspace --no-default-features                # baseline build still works
```

If any of those fail, **stop**. Do not bump versions on a broken HEAD —
the rc.1 → rc.2 cycle exists exactly because we caught issues post-bump
and had to revert.

The `stochastic-rs-py` crate is excluded from `cargo test` because it is
a `cdylib` requiring a Python extension-module link (handled by maturin,
not cargo). It is built and tested in stage 7.

## Stage 2 — version bumps (9 crates)

The workspace has 9 crates that all share a version number. Bump them
together:

```
stochastic-rs                  (umbrella, Cargo.toml workspace.package + dependencies)
stochastic-rs-core
stochastic-rs-distributions
stochastic-rs-stochastic
stochastic-rs-copulas
stochastic-rs-stats
stochastic-rs-quant
stochastic-rs-ai
stochastic-rs-viz
stochastic-rs-py
```

**One-pass approach** (preferred): edit `Cargo.toml` `[workspace.package]`
`version = "X.Y.Z"`, then in `[workspace.dependencies]` update every
`stochastic-rs-*` line's `version` field. Sub-crates inherit the
workspace-package version via `version.workspace = true` already, so this
is the only file that needs editing for version itself.

Cross-checks after editing:

```bash
# 2.1 No leftover old-version literals
grep -r '"X-1.Y.Z"' Cargo.toml stochastic-rs-*/Cargo.toml || echo "clean"

# 2.2 Workspace re-resolves
cargo metadata --no-deps --format-version=1 | jq '.packages[] | {name, version}' | grep stochastic-rs
```

## Stage 3 — CHANGELOG + V1_TO_V2.md updates

`CHANGELOG.md` follows Keep-a-Changelog format with one section per
release, dated. Pattern:

```
## [vX.Y.Z] — YYYY-MM-DD

<one-paragraph summary>

### <Section: bug fix / feature / breaking change / etc.>

- One-line description, file:line reference where helpful.
```

If this is a stable release (vX.Y.0), also update
`docs/V1_TO_V2.md` (or `Vn-1_TO_Vn.md` going forward) with the
**accumulated** breaking changes since the previous stable. The rc.1 →
rc.2 → stable cycle taught us to keep this updated *as breaking changes
land*, not at the very end.

## Stage 4 — local publish dry run

```bash
# 4.1 Dry-run publish.sh: this runs `cargo publish --dry-run` on each
# sub-crate in dependency order (core → distributions → stochastic →
# copulas → stats → quant → ai → viz → umbrella).
./publish.sh --dry-run
```

If a dry-run fails, fix and restart from stage 1. Common causes:
- A new `path = "../foo"` dependency was added without a matching
  `version = "X.Y.Z"` (crates.io rejects path-only deps).
- A `[features]` block accidentally references a stripped-by-publish
  member.

## Stage 5 — sign and push the tag

```bash
git add -A
git commit -m "release vX.Y.Z"
git tag -s "vX.Y.Z" -m "stochastic-rs vX.Y.Z"
git push origin main
git push origin "vX.Y.Z"
```

If GPG signing isn't configured, drop `-s` and use the unsigned `-a`
form. Confirm the tag rendered correctly on GitHub before stage 6.

## Stage 6 — publish to crates.io

**This stage is irreversible.** Only run it after the user confirms.

```bash
./publish.sh                  # without --dry-run; same dependency order
```

The script publishes one sub-crate at a time and waits ~30 s between
each so crates.io has time to index dependencies. If a sub-crate fails
mid-flight, fix the cause, bump the patch version of *only the
remaining* sub-crates (not the ones already published), re-tag, and
resume.

## Stage 7 — build & publish PyPI wheels

```bash
cd stochastic-rs-py
maturin build --release --strip --features python   # local platform wheel
```

For multi-platform wheels (Linux x86_64, macOS x86_64 + aarch64, Windows
x86_64), use the `.github/workflows/release.yml` GitHub Action triggered
by the `vX.Y.Z` tag push. Do not publish wheels manually unless you are
specifically testing a single-platform fix.

**macOS Intel (x86_64) note:** the action matrix must include
`macos-13` (Intel) explicitly — `macos-14` and `macos-15` runners are
all aarch64 and produce ARM-only wheels. Without the macos-13 leg,
Intel-Mac users get an installation error on `pip install
stochastic-rs`.

After wheels build, publish to TestPyPI first:

```bash
twine upload --repository testpypi target/wheels/*.whl
pip install -i https://test.pypi.org/simple/ stochastic-rs==X.Y.Z
python -c "import stochastic_rs; print(stochastic_rs.__version__)"
```

If the smoke install works:

```bash
twine upload target/wheels/*.whl     # production PyPI
```

## Stage 8 — post-release housekeeping

```bash
# 8.1 Bump main to next-dev version (open the next rc / patch cycle)
# (manual edit Cargo.toml workspace.package.version → "X.Y.Z+1-dev" or "X.Y+1.0-dev")

# 8.2 Verify docs.rs picked up the build (auto-triggered on crates.io publish)
open https://docs.rs/stochastic-rs/X.Y.Z/

# 8.3 Update any open issue / PR references to the new version
gh issue list --state open --label "v$(MAJOR.MINOR)"

# 8.4 Update CLAUDE.md / per-crate CLAUDE.md notes if the surface changed.
```

## Anti-patterns (do not do)

- **Do not** publish a single sub-crate "to test" without running the
  workspace-wide test/clippy gate. Sub-crates depend on each other; a
  partially-published version will leave users with a broken graph.
- **Do not** force-push `main` after a tag is published. The tag points
  at a specific commit; rewriting the branch breaks reproducibility.
- **Do not** skip `cargo check --all-features`. The audit §4.1 trap
  (a feature-gated symbol whose dispatch type didn't compile) escaped
  rc.0 because nothing in the test suite forced the all-features path.
- **Do not** bump `stochastic-rs-py` ahead of the workspace. PyPI
  versions must match crates.io versions exactly so users tracking one
  ecosystem can predict the other.

## Related SKILLs

- `feature-flag-management` — for ensuring the all-features build stays
  clean (mandatory pre-bump check).
- `python-bindings` — for the per-class registration steps that need to
  be in place before stage 7 can succeed.
- `bench-writing` — for keeping the rc.X benchmark baseline current
  between cycles.

## Reference: rc.1 → rc.2 → stable cycle (2026-05)

The complete sequence executed in this codebase was:

1. rc.0 (2026-05-07): initial RC; failed on §4.1 feature-flag trap.
2. rc.1 (2026-05-10 morning): closed 6 P0 + 13 P1 + 23 P2 from
   `docs/V2_RELEASE_AUDIT_2026-05-07.md`; passed publish.
3. rc.2 (2026-05-10 evening): closed 11 P0 + 3 P1 from the deep quant
   audit (`docs/QUANT_AUDIT_2026-05-10_*.md`) + §1.6 stable-cut
   residuals (clippy clean, Fukasawa seed pinned, bench baseline,
   Phase A Python coverage gaps).
4. stable v2.0.0: ~3-7 day soak; then re-run this checklist.

Each stage was triggered by the user; do not autonomously start a
release without explicit confirmation.
