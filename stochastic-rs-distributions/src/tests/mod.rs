//! Crate-level visual / benchmark tests. All `#[ignore]`'d — they generate
//! HTML plots or run 5–10 M sample throughput benchmarks, not normal CI checks.

mod bench_continuous_a;
mod bench_continuous_b;
mod bench_summary;
mod distribution_plot;
mod simd_ops;
