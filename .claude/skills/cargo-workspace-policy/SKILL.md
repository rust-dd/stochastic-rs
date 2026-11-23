---
name: cargo-workspace-policy
description: Use when adding a Rust dependency or a new sub-crate to a workspace, or when editing any `Cargo.toml` — every dep lives once in `[workspace.dependencies]`, every consumer inherits via `{ workspace = true }`, `[workspace.members]` and every `[dependencies]` block are alphabetically sorted, no categorical separator comments between entries; in stochastic-rs any root `Cargo.toml` edit also makes the committed Metal artifact stale.
---

# Cargo workspace policy — stochastic-rs

## The rule

1. Every dependency, external or internal, is declared once in the root `[workspace.dependencies]`.
2. A consumer crate inherits it with `{ workspace = true }`, plus extra `features` only; no `version` or `path` at consumer level, no `package` rename there.
3. `[workspace.members]` and every `[dependencies]` / `[dev-dependencies]` / `[build-dependencies]` block are sorted alphabetically by key.
4. No categorical separator comments between entries; the order is the structure.
5. The `stochastic-rs-*` entries pin `version = "=X.Y.Z"` exactly (a caret on an rc also matches the next rc).

A new sub-crate: add it to `[workspace.members]` and `[workspace.dependencies]` in alphabetical position, and to `publish.sh`'s `PUBLISH_ORDER` after its dependencies.

## Any root `Cargo.toml` edit makes the Metal artifact stale

`stochastic-rs-stochastic/metal/src/device/metal.rs/bindings-fast/inputs.json` records a hash of the root manifest's bytes, so the committed Metal artifact goes stale on every edit of the root `Cargo.toml`: a version bump, a dependency bump, a new `[workspace.dependencies]` line, an `include` or `exclude` change, a comment. A stale artifact fails every macOS `--features metal` build: the all-features clippy, the py `metal,accelerate` check, `METAL_OXIDE_ARTIFACTS=locked` and CI's `check_macos_backends`.

- A lane agent does not regenerate it; the controller does, as its own `chore:` commit after the merge, with the procedure in `.superpowers/sdd/rc5-lanes/metal-oxide-compiler.md`.
- Report each Metal command that fails for this reason alone, and prove it by passing with BASE's copy of the root `Cargo.toml` swapped in.
- The other inputs are the `stochastic-rs-metal-kernels` crate, the family definitions, `euler/philox.rs` and metal-oxide's `Cargo.lock` entries.
- Prefer one root-manifest edit per change set, so the artifact is regenerated once.

## Checklist

- [ ] The dependency sits in `[workspace.dependencies]`, sorted.
- [ ] The consumer entry is `{ workspace = true }`, sorted.
- [ ] No `# category` comments.
- [ ] `Cargo.lock` is updated in the same commit (CI passes `--locked`).
- [ ] If the root `Cargo.toml` changed: the Metal artifact is stale, tell the controller.
- [ ] `cargo check --workspace --no-default-features` and `cargo fmt --all -- --check` pass.
