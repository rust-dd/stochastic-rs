---
name: stats-estimator
description: How to add a statistical estimator to stochastic-rs-stats. Covers ArrayView1<T> input shape, *Result struct conventions, parametric vs bootstrap p-values, openblas gating, paper-citation requirements, and reference-comparison tests.
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

## 4. openblas-gating

Estimators that need LAPACK (linear regression, SVD, eigendecomposition)
gate behind `#[cfg(feature = "openblas")]`:

```rust
#[cfg(feature = "openblas")]
pub fn estimate_lapack<T: FloatExt>(samples: ArrayView1<T>) -> FooResult {
    use ndarray_linalg::SVD;
    // ...
}
```

The non-openblas baseline must always compile (and be the default
build). If the openblas-only path is **strictly** more accurate,
provide a closed-form fallback that's slower but always available.
Don't make the user discover at runtime that they need a feature flag.

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
    /// Reference: Hurst & Co. (2024), Table 3 row 2 — H = 0.7 ± 0.02.
    #[test]
    fn hurst_recovery_matches_paper_table3() {
        let mut rng = StdRng::seed_from_u64(42);
        let series = simulate_fbm(0.7, 5_000, &mut rng);
        let result = estimate_hurst(series.view());
        assert!(
            (result.estimate - 0.7).abs() < 0.02,
            "H = {}, expected 0.7 ± 0.02",
            result.estimate
        );
    }
}
```

The `feedback_test_batching` memory entry says "when adding many tests
write all first, then run cargo test once". For estimators that take
seconds per test (large Monte Carlo runs), batch the tests and watch
for parallel-execution flakes (the rc.2 Fukasawa fix taught us this).

## 7. Python wrapper

Expose as `#[pyfunction]` (preferred for stateless estimators) or
`#[pyclass]` (when the estimator carries state — e.g. a fitted model
that supports `predict`):

```rust
// stochastic-rs-quant/src/python.rs (yes, the Python wrappers for
// stats estimators live in the quant crate's python.rs because that's
// where stochastic-rs-py registers them; alternative is direct
// registration from -stats but the existing pattern is via -quant).

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
- **Do not** silently fall through to a slow path when openblas isn't
  available. Either gate explicitly with `#[cfg(feature = "openblas")]`
  or provide a documented closed-form fallback.
- **Do not** roll your own ADF / KPSS regression. If you need linear
  regression, use the helpers in `stochastic-rs-stats::stationarity`.
- **Do not** depend on `statrs` for distribution math — write closed
  forms via `stochastic_rs_distributions::DistributionExt`. See
  `feedback_no_statrs_distributions` memory entry.
- **Do not** combine parametric and bootstrap p-values into a single
  field. They have different validity domains; users need both.

## 9. Reference impls

- `fukasawa_hurst::estimate` (`fukasawa_hurst.rs`) — rough-vol Hurst
  via L-BFGS-B + Paxson + Eq. 16 corrections; paper-Table 1 validation
  test.
- `stationarity::adf::adf_test` (`stationarity/adf.rs`) — Augmented
  Dickey-Fuller with Mackinnon p-values + test-statistic struct.
- `stationarity::kpss::kpss_test` — KPSS with bandwidth-aware long-run
  variance estimator.
- `mle::*` — MLE family (gamma / lognormal / NIG / Heston) with
  closed-form score / Fisher info where available.

## Related SKILLs

- `add-diffusion-process` — when validating an estimator on simulated
  process samples.
- `python-bindings` — for the `PyFooResult` / `estimate_foo` wrappers.
- `integration-test-writing` — for reference-comparison test
  conventions.
