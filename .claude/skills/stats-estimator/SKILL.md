---
name: stats-estimator
description: How to add a statistical estimator to stochastic-rs-stats. Covers ArrayView1<T> input shape, *Result struct conventions, parametric vs bootstrap p-values, linear-algebra helpers, paper-citation requirements, and reference-comparison tests.
---

# Stats estimator — stochastic-rs-stats

A "stats estimator" in `stochastic-rs-stats` consumes a 1-D series
(`ArrayView1<T>`), optionally a second covariate / regressor, and
returns a typed `XxxResult` struct with named fields plus a converged /
p-value indicator.

This SKILL covers MLE, MoM, Hurst estimators, stationarity tests,
fractal-dimension estimators, and bootstrap variants. The pattern is
uniform: the user receives a data type they can introspect (rather
than a tuple of unnamed `f64`s) and the test reports its own
diagnostic without hidden state.

## 1. Function signature

```rust
// stochastic-rs-stats/src/foo_estimator.rs

use ndarray::ArrayView1;
use crate::traits::FloatExt;

pub fn estimate<T: FloatExt>(
    samples: ArrayView1<T>,
    /* extra parameters: m_window, lag, alpha, etc. */
) -> FooResult { /* ... */ }
```

Three rules:

- **Input is `ArrayView1<T>`**, never `&[T]`. ndarray slices interop
  with the rest of the stats / quant pipeline; raw slices are ergonomic
  but the boundary cost shows up in benchmarks.
- **Generic over `T: FloatExt`** so f32 and f64 callers share code.
  The estimator typically internally lifts to f64 for accumulation
  (`.to_f64().unwrap()`) and returns f64 in the `*Result` struct.
- **Return a typed struct**, never a tuple. Estimators evolve (a future
  version might add `iterations`, `pvalue`, `confidence_interval`); a
  named struct can grow a field without breaking existing call sites.

## 2. The `*Result` struct

```rust
#[derive(Debug, Clone)]
pub struct FooResult {
    /// Point estimate.
    pub estimate: f64,
    /// Optional p-value (parametric / asymptotic).
    pub pvalue: Option<f64>,
    /// 95% confidence interval; None when bootstrap not requested.
    pub ci_95: Option<(f64, f64)>,
    /// Number of optimiser iterations (None for closed-form estimators).
    pub iterations: Option<usize>,
    /// Whether the optimiser converged (true for closed-form).
    pub converged: bool,
    /// Optional bootstrap p-value (separate from the parametric one
    /// because the asymptotic form may not apply on small samples).
    pub bootstrap_pvalue: Option<f64>,
}
```

`Debug + Clone` is mandatory. Calibrators and downstream pipelines
clone results into pipelines; `Debug` is what dumps to logs.

## 3. Parametric vs bootstrap p-values

Two distinct fields, deliberately not collapsed:

- `pvalue` — analytic / asymptotic. E.g. ADF unit-root test:
  Mackinnon (1996) regression coefficients give p-value as a function
  of sample size + the test statistic.
- `bootstrap_pvalue` — non-parametric resampling. Slower (typical: 1000
  bootstrap replicates) but valid on any DGP.

Estimators with an exact / asymptotic formula populate `pvalue` only.
Bootstrap-only estimators populate `bootstrap_pvalue` only. Estimators
where both make sense populate both, and document the difference in
the struct doc.

## 4. Linear algebra

Dense linear algebra (least squares, LU solve/inverse, SPD Cholesky,
eigenvalues) is ungated: it runs on the pure-Rust `faer` through the
crate-private helpers in `stochastic-rs-stats/src/linalg.rs`
(`lstsq` / `solve` / `inverse` / `spd_cholesky_lower` / `eigenvalues`).
Call those instead of touching `faer` directly — they own the
ndarray↔faer conversions and the finite-solution singularity probe.
There is no feature flag to gate on.

## 5. Paper citation header

The source file's `//!` header **must** name the paper:

```rust
//! # Foo estimator
//!
//! Implements the <Name> (<Year>) estimator for <quantity>:
//!
//! \[LaTeX block of the canonical formula\]
//!
//! Reference: <Author, Year>, "<Title>", *<Journal>* <vol>(<num>), <pages>.
```

For Hurst estimators specifically, also cite the Mandelbrot &
Van Ness (1968) prior so users orienting themselves can map between
estimators.

## 6. Reference-comparison test

Mandatory: at least one test that compares to a manually-computed
reference (R, scipy, Mathematica, or a published paper Table). Pinned
seed, named tolerance:

```rust
#[cfg(test)]
mod tests {
    use ndarray::ArrayView1;
    use stochastic_rs_core::simd_rng::Deterministic;
    use stochastic_rs_stochastic::process::fbm::Fbm;
    use stochastic_rs_stochastic::traits::ProcessExt;

    /// Reference: <Author, Year>, Table 3 row 2 — H = 0.7 +/- 0.02.
    #[test]
    fn hurst_recovery_matches_paper_table3() {
        // Seed the *process*, not an external Rng. `dev-rules` §7a:
        // `StdRng` / `rand_distr` belong to `benches/` only, and a
        // `Simd*` distribution ignores any `Rng` handed to `fill_slice`
        // — the seed must reach the constructor.
        let series = Fbm::<f64, _>::new(0.7, 5_000, None, Deterministic::new(42)).sample();
        let result = estimate_hurst(series.view());
        assert!(
            (result.estimate - 0.7).abs() < 0.02,
            "H = {}, expected 0.7 +/- 0.02",
            result.estimate
        );
    }
}
```

Pin the seed through the workspace's own constructor. See
`integration-test-writing` §1.1 for the trap this avoids — an external
`StdRng` handed to a `Simd*` distribution is silently discarded, so the
test looks seeded and is not.

The `feedback_test_batching` memory entry says "when adding many tests
write all first, then run cargo test once". For estimators that take
seconds per test (large Monte Carlo runs), batch the tests and watch
for parallel-execution flakes (the rc.2 Fukasawa fix taught us this).

## 7. Python wrapper

Expose as `#[pyfunction]` (preferred for stateless estimators) or
`#[pyclass]` (when the estimator carries state — e.g. a fitted model
that supports `predict`):

Wrappers live in **`stochastic-rs-stats/src/python/`** — the stats
crate's own module, not the quant crate's. It is a directory split by
topic (`hurst.rs`, `mle.rs`, `normality.rs`, `realized.rs`,
`stationarity`-adjacent `misc.rs`, …) with a `mod.rs` re-exporting the
`Py*` types, and `stochastic-rs-py/src/lib.rs` imports them as
`stochastic_rs_stats::python::PyXxx`.

```rust
// stochastic-rs-stats/src/python/foo.rs

#[pyfunction]
#[pyo3(signature = (samples))]
pub fn estimate_foo<'py>(samples: numpy::PyReadonlyArray1<'py, f64>) -> PyResult<PyFooResult> {
    let result = stochastic_rs_stats::foo::estimate(samples.as_array());
    Ok(PyFooResult { inner: result })
}

#[pyclass(name = "FooResult", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyFooResult {
    pub inner: stochastic_rs_stats::foo::FooResult,
}

#[pymethods]
impl PyFooResult {
    #[getter] fn estimate(&self) -> f64 { self.inner.estimate }
    #[getter] fn pvalue(&self) -> Option<f64> { self.inner.pvalue }
    // ... one getter per field ...
}
```

Then register both in `stochastic-rs-py/src/lib.rs`.

## 8. Anti-patterns

- **Do not** return a tuple `(f64, f64, bool)`. Always return a typed
  struct.
- **Do not** bypass `src/linalg.rs` with hand-rolled matrix inverses
  or direct `faer` calls — the helpers carry the singularity probe and
  the conversion conventions.
- **Do not** roll your own ADF / KPSS regression. Shared helpers live
  in `stochastic-rs-stats/src/stationarity/common.rs`.
- **Do not** seed a test with `StdRng` / `rand_distr`. `dev-rules` §7a
  reserves those for `benches/`; seed the workspace's own process or
  `Simd*` distribution at its constructor.
- **Do not** depend on `statrs` for distribution math — write closed
  forms via `stochastic_rs_distributions::DistributionExt`. See
  `feedback_no_statrs_distributions` memory entry.
- **Do not** combine parametric and bootstrap p-values into a single
  field. They have different validity domains; users need both.

## 9. Reference impls

- `hurst::whittle::estimate` (`hurst/whittle.rs`) — the Fukasawa
  rough-vol Hurst estimator (L-BFGS-B + Paxson + Eq. 16 corrections),
  returning `FukasawaResult`; paper-Table 1 validation test. Note the
  path: there is no `fukasawa_hurst.rs`; it lives under `hurst/` and is
  named for the *method*, not the author. `estimate_from_prices` and
  `estimate_from_prices_generic<T>` are the convenience entry points.
- `stationarity::adf::adf_test(y, cfg)` — Augmented Dickey-Fuller with
  Mackinnon p-values. Note the second argument: this crate's
  stationarity tests take a `*Config` struct, not loose parameters
  (`AdfConfig`, `KpssConfig`, …). Follow that shape.
- `stationarity::kpss::kpss_test(y, cfg)` — KPSS with bandwidth-aware
  long-run variance.
- `hurst/` — nine sibling estimators (`dfa`, `gph`, `rs`, `variations`,
  `wavelet`, `whittle`, …) sharing one result convention. The best place
  to see the pattern repeated.
- `mle/` — MLE family (`density.rs`, `fit.rs`, `process_impls/`).

## Related SKILLs

- `add-diffusion-process` — when validating an estimator on simulated
  process samples.
- `python-bindings` — for the `PyFooResult` / `estimate_foo` wrappers.
- `integration-test-writing` — for reference-comparison test
  conventions.
