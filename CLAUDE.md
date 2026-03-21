# CLAUDE.md — stochastic-rs

Rust library for quantitative finance: stochastic process simulation, pricing, statistics, copulas, distributions, and AI-based volatility models. Published on crates.io as `stochastic-rs`.

## Build & test

```bash
cargo build                    # build
cargo test                     # run tests
cargo bench                    # run benchmarks
cargo build --features cuda    # build with CUDA support
cargo build --features python  # build with Python bindings (pyo3)
```

## Clippy usage

Always run `cargo clippy` to adopt the latest compiler recommendations.

## Key traits

- `FloatExt` — core float trait bound (extends `num_traits::Float + Send + Sync + ...`)
- `ProcessExt<T>` — stochastic process simulation

## Skills

Development rules and conventions are in `.claude/skills/dev-rules/SKILL.md`.
