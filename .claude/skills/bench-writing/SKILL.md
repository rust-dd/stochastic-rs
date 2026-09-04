---
name: bench-writing
description: Conventions for writing criterion benchmarks in stochastic-rs. Group naming, parameter sweep, [[bench]] required-features gating, no-println / no-dead-helper rules. Invoke when adding a new bench or refactoring an existing one.
---

# Bench writing — stochastic-rs

Benchmarks live under `benches/` at the **workspace root only** — no
sub-crate has a `benches/` directory — and use the
[`criterion`](https://github.com/bheisler/criterion.rs) harness with
`harness = false` per `[[bench]]` entry in the root `Cargo.toml`. The
§6.1 audit trap was a benchmark file with three dead helper functions
and a `println!("starting...")` that shipped to crates.io as dev-deps
but was never run; this SKILL prevents that drift.

Baselines are criterion's own (`cargo bench -- --save-baseline <name>`,
then `--baseline <name>` to compare). There is **no tracked baseline
document** in this repo and **no bench job in CI**
(`.github/workflows/rust.yml` has none), so a regression is only caught
by whoever runs `cargo bench` locally. Do not cite a
`docs/BENCH_BASELINE.md`; it does not exist.

## 1. The skeleton

```rust
// benches/foo.rs

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::traits::ProcessExt;
use stochastic_rs_core::simd_rng::Deterministic;

fn bench_foo(c: &mut Criterion) {
    let mut group = c.benchmark_group("foo");
    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        // Build the process ONCE, outside b.iter — construction is not
        // what you are measuring, and re-seeding per iteration hides the
        // sampler's own per-call cost.
        let process = Gbm::<f64, _>::new(0.05, 0.2, n, None, None, Deterministic::new(42));
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, _| {
                b.iter(|| criterion::black_box(process.sample()));
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

If the bench requires a feature, gate it. Use the **real** feature
names from the root `Cargo.toml` — there is no bare `cuda` feature:

```toml
[[bench]]
name = "fgn_cuda"
harness = false
required-features = ["cuda"]          # cudarc + cuFFT
```

The gated benches in tree today, with their exact feature sets:

| Bench | `required-features` |
|---|---|
| `fgn_cubecl` | `["cubecl-wgpu"]` |
| `fgn_cuda` | `["cuda"]` |
| `fgn_cuda_compare` | `["cuda", "cubecl-cuda"]` |
| `fgn_all_backends` | `["cubecl-wgpu", "metal", "accelerate"]` |
| `fgn_accelerate` | `["accelerate"]` |
| `fgn_metal` | `["metal"]` |
| `hotpath_profile` | `["hotpath"]` |
| `dual_stream_compare` | `["dual-stream-rng"]` |

Without the gate, cargo tries to compile the bench regardless and you
get a compilation error rather than a skip.

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

- **No feature gate**: if the bench compiles only under a feature, the
  `[[bench]] required-features` entry is the gate. Forgetting it means
  `cargo bench -p stochastic-rs` tries to compile it unconditionally and
  fails the whole run.

## 6. Hot-path benches vs end-to-end

Two flavours:

- **Hot-path** (`benches/fgn_*.rs`, `benches/distributions/`): the
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
cargo build --benches -p stochastic-rs                              # default features
cargo build --benches -p stochastic-rs --features cuda       # if applicable
```

Use `-p stochastic-rs`, **not** `--workspace`: every bench lives in the
umbrella, and `--workspace` drags in `stochastic-rs-py`, which forces
`pyo3/extension-module` unconditionally and fails to link outside a
maturin build (same reason `cargo test --workspace` needs
`--exclude stochastic-rs-py` — see `CLAUDE.md`).

If any leg fails, the bench has drifted from the lib's API. Fix
before commit; the §6.1 trap was exactly a bench that hadn't compiled
in 6 months because nobody ran `cargo build --benches`. Nothing in CI
does this for you.

## 8. Reference benches

`benches/` holds 32 `.rs` files plus one `distributions/` **directory**,
matched one-to-one by 33 `[[bench]]` entries. Check whether your target
is a file or a directory before editing.

- `benches/distributions/` — sweep over distribution × sample-count.
- `benches/fgn_fbm.rs` — ungated CPU fGn / fBm sweep.
- `benches/fgn_all_backends.rs` — the actual cross-backend comparison;
  gated on `["cubecl-wgpu", "metal", "accelerate"]`.
- `benches/option.rs` — end-to-end pricing with reduced sample count.
- `benches/risk.rs` — VaR / ES estimators on synthetic samples.
- `benches/dist_multicore.rs` — `sample_par` parallelism vs serial.
- `benches/sampler_compare.rs`, `benches/hotpath_profile.rs` — the
  sampler-v3 refactor's own measurement harnesses.

## 9. Registering a new bench

A new bench needs **two** edits, both in the root `Cargo.toml`: the
file under `benches/`, and a `[[bench]]` entry with `harness = false`
(plus `required-features` if gated). Cargo does not auto-discover with
`harness = false`, so a missing entry means the bench never runs and
never compiles — which is exactly the §6.1 drift.

There is no baseline document to append to and no CI job to update.

## Anti-patterns

- **Do not** `println!` inside `b.iter`.
- **Do not** leave dead helpers / unused fns in the bench file.
- **Do not** allocate inside the hot loop.
- **Do not** add a feature-gated bench without `[[bench]]
  required-features`.
- **Do not** ship a bench that doesn't compile under
  `cargo build --benches -p stochastic-rs`.

## Related SKILLs

- `release-checklist` — note that `cargo bench` is **not** one of its
  gates today; benchmarking is a local, manual step.
- `add-gpu-sampler` — the natural source of CUDA-only benches.
- `feature-flag-management` — `required-features` propagation.
- `integration-test-writing` — same pinned-seed mandate; bench-time
  drift is the test-suite's parallel.
