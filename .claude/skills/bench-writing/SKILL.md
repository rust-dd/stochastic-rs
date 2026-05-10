---
name: bench-writing
description: Conventions for writing criterion benchmarks in stochastic-rs. Group naming, parameter sweep, [[bench]] required-features gating, no-println / no-dead-helper rules. Invoke when adding a new bench or refactoring an existing one.
---

# Bench writing — stochastic-rs

Benchmarks live under `benches/` (umbrella) and use the
[`criterion`](https://github.com/bheisler/criterion.rs) harness with
`harness = false` per `[[bench]]` entry in `Cargo.toml`. The §6.1
audit trap was a benchmark file with three dead helper functions and a
`println!("starting...")` that shipped to crates.io as dev-deps but was
never run; this SKILL prevents that drift.

For the baseline / regression workflow (`--save-baseline rc2` /
`--baseline rc2`), see `docs/BENCH_BASELINE.md`. This SKILL is about
*writing* the bench, not maintaining the baseline.

## 1. The skeleton

```rust
// benches/foo.rs

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_foo(c: &mut Criterion) {
    let mut group = c.benchmark_group("foo");
    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let process = stochastic_rs::stochastic::diffusion::gbm::Gbm::seeded(
                        0.05, 0.2, n, None, None, /* seed */ 42,
                    );
                    let _ = process.sample();   // black_box-equivalent for IndexOp
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_foo);
criterion_main!(benches);
```

Then add to `Cargo.toml` (root):

```toml
[[bench]]
name = "foo"
harness = false
```

If the bench requires a feature (e.g. `cuda`, `openblas`):

```toml
[[bench]]
name = "fgn_cuda_native"
harness = false
required-features = ["cuda"]
```

The `required-features` gate prevents the bench from being silently
skipped under `cargo bench --workspace` without the feature; criterion
emits a "skipped" line so the user notices.

## 2. Group naming convention

Each bench file declares **one** `criterion_group!` at the bottom, but
internally uses `c.benchmark_group("name")` to scope the runs. Naming:

- `foo` — single function under bench (matches the file name).
- `foo/parameter_sweep` — when there's a single sweep dim (n_paths).
- `foo/method/parameter` — when comparing methods (e.g. `fgn/cpu/n_2048`,
  `fgn/cuda/n_2048`).

The slash separators end up in the criterion output / HTML report;
keep them informative but consistent (no `_` next to `/`, no spaces).

## 3. Throughput hints

For samplers, `Throughput::Elements(n)` lets criterion report
"elements per second" (= sample throughput). For pricers,
`Throughput::Bytes(...)` is irrelevant; just leave it off. For
end-to-end calibration, omit throughput.

## 4. Parameter sweep

Sweep parameters by `bench_with_input` rather than by writing five
near-identical functions:

```rust
for &(h, n) in &[(0.3, 1024), (0.5, 1024), (0.7, 1024), (0.5, 8192)] {
    let id = BenchmarkId::from_parameter(format!("h={h}_n={n}"));
    group.bench_with_input(id, &(h, n), |b, &(h, n)| {
        b.iter(|| sample_fbm(h, n));
    });
}
```

Criterion will emit one line per (h, n) pair in its output. The
`BenchmarkId::from_parameter(...)` also feeds the comparison-vs-baseline
machinery — keep the parameter format stable across releases or you
lose the diff.

## 5. Anti-patterns: what kills benches

- **`println!`**: criterion runs the bench function many times
  per measurement; a single `println!` floods stderr + skews the
  measurement. The §6.1 trap left a `println!("starting...")` that
  was harmless during dev but invisible after merge.
- **Dead helper functions**: `fn helper_v1` / `fn helper_v2` that
  remained from rapid iteration. Delete them before commit. Criterion
  doesn't run them, so the compiler doesn't catch when their
  internals drift.
- **Hidden allocations in the hot loop**: `vec![0.0; n]` per
  iteration leaks N * iter_count allocations into the measurement.
  Pre-allocate outside `b.iter`:

  ```rust
  let mut buf = vec![0.0; n];
  group.bench_with_input(id, &n, |b, _| {
      b.iter(|| {
          buf.fill(0.0);
          // use buf
      });
  });
  ```

- **No `cargo bench --workspace` gate**: if the bench compiles only
  under a feature, the `[[bench]] required-features` is the gate.
  Forgetting it produces "compilation error" rows in the bench report.

## 6. Hot-path benches vs end-to-end

Two flavours:

- **Hot-path** (`benches/fgn_*.rs`, `benches/distributions.rs`): the
  inner loop of a sampler / kernel. These run *fast* (microseconds);
  criterion's default 100-sample setting is right.
- **End-to-end** (`benches/option.rs`, `benches/instruments.rs`): a
  full pricing / calibration call. These run *slow* (milliseconds);
  reduce sample count to keep wall-time reasonable:

  ```rust
  group.sample_size(10).measurement_time(std::time::Duration::from_secs(5));
  ```

## 7. Mandatory `cargo build --benches` check

Before commit, verify:

```bash
cargo build --benches --workspace                                 # default features
cargo build --benches --workspace --features cuda                 # if applicable
cargo build --benches --workspace --features openblas             # if applicable
```

If any leg fails, the bench has drifted from the lib's API. Fix
before commit; the §6.1 trap was exactly a bench that hadn't compiled
in 6 months because nobody ran `cargo build --benches`.

## 8. Reference benches

- `benches/distributions.rs` — sweep over distribution × sample-count.
- `benches/fgn_fbm.rs` — comparison sweep CPU vs GPU (gated).
- `benches/option.rs` — end-to-end Heston / Bates / Merton with
  reduced sample count.
- `benches/risk.rs` — VaR / ES estimators on synthetic samples.
- `benches/dist_multicore.rs` — `sample_par` parallelism vs serial.

## 9. Updating `docs/BENCH_BASELINE.md`

When you add a new `[[bench]]`, append it to the rc.2 baseline list in
`docs/BENCH_BASELINE.md` so the next release run captures it. The
release-checklist SKILL references that doc; out-of-date baseline
lists silently miss new benches in the regression check.

## Anti-patterns

- **Do not** `println!` inside `b.iter`.
- **Do not** leave dead helpers / unused fns in the bench file.
- **Do not** allocate inside the hot loop.
- **Do not** add a feature-gated bench without `[[bench]]
  required-features`.
- **Do not** ship a bench that doesn't compile under
  `cargo build --benches --workspace`.

## Related SKILLs

- `release-checklist` — cargo bench is a release gate.
- `add-gpu-sampler` — the natural source of CUDA-only benches.
- `feature-flag-management` — `required-features` propagation.
- `integration-test-writing` — same pinned-seed mandate; bench-time
  drift is the test-suite's parallel.
