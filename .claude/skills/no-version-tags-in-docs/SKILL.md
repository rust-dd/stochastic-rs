---
name: no-version-tags-in-docs
description: Use when writing or editing Rust `///` or `//!` doc comments — never tag sections with `## v2.3.0 design choice — …` or `## v2.4 deferred — …`. Version history goes in `CHANGELOG.md`, git log, or `docs/V*_UPDATE.md`, never in source doc comments.
---

# No Version-Tagged Sections in Source Doc Comments

## The Rule

Doc comments describe **what the code does**, not which release introduced or will introduce a feature.

**Never** emit headers like:

```rust
//! ## v2.3.0 design choice — XYZ
//! ## v2.4 deferred — ABC
//!
//! In v2.3.0 we ship only the closed-form path; the iterative refinement
//! lands in v2.4 alongside the structure-selection algorithm.
```

The reader hitting `cargo doc` doesn't have the version landscape. They have the code in front of them, and they need to know what it does, what the math is, and what its preconditions are. Versions move on; the prose stays.

## Where versioning DOES belong

- `CHANGELOG.md` — human-curated release notes.
- `git log` / commit messages — sequence of changes.
- `docs/V*_UPDATE.md` (or equivalent planning docs) — release-scope documents that LIVE outside `cargo doc`.

## How to express "this is a partial implementation"

When a future improvement is genuinely worth recording near the code:

- `// TODO:` with a short rationale.
- A doc paragraph that describes **what is NOT supported and why**, without naming the release number:

```rust
//! Pure-jump leverage (ρ ≠ 0 between W and Z) is not yet implemented;
//! the BNS-2001b shifted-Brownian construction needs careful
//! pre-allocation of two correlated noise streams.
```

This stays accurate forever — when the feature lands, the paragraph gets deleted, not retroactively re-numbered.

## Why

Source-embedded version tags cause three failure modes:

1. **They lie over time.** A "v2.4 deferred" tag stays in the source after v2.4 ships, telling future readers the feature is missing when in fact it is right there.
2. **They duplicate the changelog.** Two places to update for every release; one of them will drift.
3. **They imply scope discussions matter to library consumers.** They don't. Consumers want to know if the API works today.
