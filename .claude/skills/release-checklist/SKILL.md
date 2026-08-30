---
name: release-checklist
description: Step-by-step release workflow for stochastic-rs (rc → stable → patch). Invoke when bumping versions, cutting a tag, publishing to crates.io, or shipping wheels to PyPI.
---

# Release checklist — stochastic-rs

This SKILL formalises the release workflow used to ship rc.0 → rc.1 →
rc.2 → stable. Follow it whenever the user asks to "cut a release", "bump
to vX.Y.Z", "tag stable", "publish wheels", or similar.

The workflow has **8 stages**. Stages 1-5 are local and reversible;
stages 5.5-8 publish to public registries and are not. Stop and confirm
with the user before stage 5.5.

**Note the real ordering.** Stage 7 (PyPI wheels) is not a step you
run — it is a GitHub Action that fires on `release: published`, i.e. the
moment stage 5.5 creates the release. So the irreversible sequence is
5.5 (GitHub release → Zenodo DOI *and* PyPI wheels) then 6 (crates.io),
not 6 then 7. Confirm before 5.5, not before 6.

## Stage 1 — pre-flight checks

Run from the workspace root (`/Users/danixx/Desktop/stochastic-rs`):

```bash
# 1.1 Working tree clean?
git status --porcelain     # must be empty (no uncommitted, no untracked)
git log -1 --oneline       # capture the HEAD commit for the release notes

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
stochastic-rs-py
```

**One-pass approach** (preferred): edit `Cargo.toml` `[workspace.package]`
`version = "X.Y.Z"`, then in `[workspace.dependencies]` update every
`stochastic-rs-*` line's `version` field. Sub-crates inherit the
workspace-package version via `version.workspace = true` already, so this
is the only file that needs editing for version itself.

Cross-checks after editing:

**One gotcha before you commit anything.** `.claude/` is gitignored, but
`.agents/skills` is a symlink to `.claude/skills`, and the umbrella
`cargo publish` walks it — untracked files under there abort the publish
with a dirty-tree error that `git status` will not show you. `git add -f`
any new or modified skill files before stage 4.

```bash
# 2.1 No leftover old-version literals
grep -r '"X-1.Y.Z"' Cargo.toml stochastic-rs-*/Cargo.toml || echo "clean"

# 2.2 Workspace re-resolves
cargo metadata --no-deps --format-version=1 | jq '.packages[] | {name, version}' | grep stochastic-rs
```

## Stage 3 — `MIGRATION.md` and `CITATION.cff`

**There is no `CHANGELOG.md` in this repo, for any release, and there
never has been.** Do not create one as part of a release; do not cite
one. The release-notes surface is:

- **`MIGRATION.md`** (repo root) — the breaking-changes record, written
  *as changes land*, not at cut time. Entries accumulate under
  `## Unreleased`; at release, retitle that section to the version and
  open a fresh `## Unreleased`. Each entry is a `###` heading naming the
  crate and the change, a before/after code block, and prose on what
  replaces what. Read the top of the file for the shape.
- **`git log`** — the per-change sequence.
- **`docs/V*_UPDATE.md`** (e.g. `docs/V2_3_0_UPDATE.md`,
  `docs/V2_4_0_UPDATE.md`) — occasional release-scope planning
  documents, written when a release has a theme worth narrating. Not
  required, and not written for every release.
- **The GitHub release body** — the human-facing notes, composed at
  stage 5.5. That is where a Keep-a-Changelog-style summary goes if you
  want one.

There is no `docs/V1_TO_V2.md`; `MIGRATION.md` absorbed that role.

Bump `CITATION.cff` alongside the crate version — `version` and
`date-released`. Leave the two `identifiers:` DOIs alone for now; the
version DOI is only known after stage 5.5.

## Stage 4 — local publish dry run

```bash
# 4.1 Dry-run publish.sh: runs `cargo publish --dry-run` on each
# sub-crate in dependency order (core → distributions → stochastic →
# copulas → stats → quant → ai → umbrella).
./publish.sh --dry-run
```

`publish.sh` runs its **own** fmt + clippy + test gate before
publishing anything, so stage 1 and this stage overlap deliberately.
Its other flags: `--allow-dirty` (publish with uncommitted changes) and
`--skip-gate` (bypass the fmt/clippy/test gate — not recommended). It
skips `stochastic-rs-py` entirely (`publish = false`; it is a
maturin-built cdylib for PyPI, not a crates.io crate), and it skips any
crate already published at the current local version, so it is safe to
re-run after a partial failure.

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

## Stage 5.5 — GitHub release and Zenodo DOI

`gh release create vX.Y.Z` is what mints the DOI: the repo carries a Zenodo
webhook on `release` events, so every release is archived automatically.
Zenodo never backfills — a release created before the webhook existed has no
DOI and cannot get one retroactively.

`.zenodo.json` at the repo root controls the record's title, abstract,
keywords and ORCID. Without it Zenodo names the record
`rust-dd/stochastic-rs: vX.Y.Z` with an empty description, so keep it in the
tree at tag time.

```bash
gh release create "vX.Y.Z" --title "vX.Y.Z" --notes-file notes.md

# The record appears within ~30 s. Two DOIs come back:
curl -s 'https://zenodo.org/api/records?q=title:%22stochastic-rs%22&all_versions=true' \
  | python3 -c "import sys,json; [print(h.get('conceptdoi'), h.get('doi'), h['metadata'].get('version')) for h in json.load(sys.stdin)['hits']['hits']]"
```

- **Concept DOI** (`10.5281/zenodo.21553307`) always resolves to the newest
  version. The README badge uses it, so it never needs touching.
- **Version DOI** changes every release. Update the second `identifiers:`
  entry in `CITATION.cff` with it, then commit.

## Stage 6 — publish to crates.io

**This stage is irreversible.** Only run it after the user confirms.

```bash
./publish.sh                  # without --dry-run; same dependency order
```

The script publishes one sub-crate at a time. There is **no manual
sleep** between crates — cargo ≥ 1.66 waits for the registry index to
sync after each publish on its own. If a sub-crate fails mid-flight,
fix the cause and simply re-run `./publish.sh`: already-published
versions are detected and skipped, so a resume needs no version
surgery.

## Stage 7 — build & publish PyPI wheels

**This stage is automatic.** `.github/workflows/pypi.yml` (there is no
`release.yml`) is triggered by `release: published` — that is, by
stage 5.5's `gh release create`, **not** by the tag push — and also by
`workflow_dispatch` for a manual re-run. It builds the Linux, macOS,
Windows and sdist legs and then publishes them itself:

```yaml
- uses: PyO3/maturin-action@v1
  with:
    command: upload
    args: --non-interactive --skip-existing dist/*
```

So there is no `twine`, no TestPyPI hop, and nothing to run locally. The
build args differ per leg — Linux and macOS pass
`--features openblas` (macOS adds `--auditwheel skip`), Windows passes
neither (it links `openblas-static` via its own config). Note that the
`stochastic-rs-py` crate has no `python` feature: it forces
`pyo3/extension-module` unconditionally, which is exactly why
`cargo test --workspace` needs `--exclude stochastic-rs-py`.

A local wheel for debugging one platform:

```bash
cd stochastic-rs-py
maturin build --release --strip --features openblas
```

**Platform coverage is settled — do not re-add Intel macOS.** The macOS
matrix has a single `macos-14` runner (aarch64) and that is deliberate:
Intel-Mac wheels were dropped at v3.0.0-beta.2 (user decision,
2026-08-28). GitHub has retired the `macos-13` runner, so adding that
leg back does not fail — it **queues forever with no runner assigned**
(observed: 14 hours), and because `publish` needs every build job, one
stuck leg blocks the whole PyPI release. If Intel support is ever
revisited, it needs a cross-compilation lane, not a `macos-13` entry.

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
- `bench-writing` — note there is no tracked bench baseline and no
  bench CI job; benchmarking is not a gate in this checklist.

## Reference: the v2.0.0 cycle (2026-05)

The shape that produced v2.0.0 was rc.0 → rc.1 → rc.2 → stable, each rc
closing a numbered audit's findings, with a multi-day soak before the
stable cut. The audit documents those rcs closed are **no longer in the
tree** — do not follow links to `docs/V2_RELEASE_AUDIT_*.md` or
`docs/QUANT_AUDIT_*.md`; they were removed. `docs/` today holds the
wave ledgers (`A1*_WAVE_LEDGER.md`, `A2_WAVE_LEDGER.md`,
`DETERMINISTIC_PARALLELISM_LEDGER.md`, …) and
`docs/VALIDATION_COVERAGE_AUDIT_2026_08_12.md`.

The workspace is on `3.0.0-beta.1` (root `Cargo.toml`,
`[workspace.package] version`), so the next cut is a v3 beta / rc, not
another v2 patch. Check the actual version before assuming.

Each stage is triggered by the user; do not autonomously start a
release without explicit confirmation. Note the standing preference: an
explicit "cut X.Y.Z" authorises the whole tag → push → `gh release`
flow, so do not re-confirm at every step.
