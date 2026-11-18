---
name: stats-estimator
description: How to add an estimator or a hypothesis test to stochastic-rs-stats. Covers fit_<model> naming, ArrayView1<T> input, *Fit / *Test / *Result structs, Result<_, StatsError> against panics, CriticalValues and rejects_null, the shared linalg helpers, the module-doc shape, reference tests and the Python wrapper.
---

# Stats estimator — stochastic-rs-stats

An estimator or test in `stochastic-rs-stats` reads a 1-D series (`ArrayView1<T>`), sometimes a
second series or a design matrix, and answers a typed struct inside a `Result`. Read
`dev-rules` §14 (the error policy) beside this file; where they differ, `dev-rules` wins.

## 1. Function signature

```rust
// stochastic-rs-stats/src/foo.rs

use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::validate;
use crate::traits::FloatExt;

/// Foo's estimate of `theta`; `Err` below 20 points or on a non-finite one; panics unless
/// `0 < dt < ∞`.
pub fn fit_foo<T: FloatExt>(series: ArrayView1<T>, dt: f64) -> Result<FooFit, StatsError> {
  assert!(
    dt > 0.0 && dt.is_finite(),
    "dt must satisfy `0.0 < dt < ∞`, got dt = {dt:?}"
  );
  let x = as_slice_or_copy(series);
  validate(&x, 20)?;
  // ...
}
```

- **The name** is `fit_<model>` for an estimate (`fit_ou_mle`, `fit_garch`, `fit_gpd`) and
  `<name>_test` for a hypothesis test (`adf_test`, `kpss_test`). An estimator with several
  methods is a struct that owns its configuration and implements a trait (`HurstEstimator`,
  `FractalDimEstimator`), with one `estimate` entry point.
- **The input** is `ArrayView1<T>`, never `&[T]` or an owned array. A strided or reversed view is
  accepted: take `as_slice_or_copy(view)` where a slice is needed, and sum in index order
  (`mean_in_order`) so that a view and its copy round alike.
- **Generic over `T: FloatExt`** where the arithmetic allows it; the regression-based tests take
  `f64`. Lift to `f64` for accumulation.
- **The return type is `Result<_, StatsError>`.** A broken precondition of the call (a
  configuration value out of range, `dt <= 0`, a `Trend::None` a test cannot take) panics with
  ``<arg> must satisfy `<pred>`, got <arg> = {value:?}``. Anything the data decide is an `Err`:
  `InsufficientData`, `InvalidObservation`, `DimensionMismatch`, `DegenerateData` (a constant
  series, an exact fit), `Singular`, `Numerical { stage }`. Nothing answers `0.0` or a floor as a
  stand-in.
- **A pure realized measure** that answers one `T` answers `T::nan()` below its minimum length.
- **Validate first**, with the helpers in `input.rs` (`validate`, `validate_domain`,
  `validate_rows`, `reject_constant`, `assert_alpha`), in the order length, finiteness,
  degeneracy.

## 2. The result struct

`*Fit` for an estimate, `*Test` for a test, `*Result` for anything else (`EwmaResult`); the
sample size is `nobs`. All derive `Debug + Clone`.

```rust
/// Foo's estimate; `std_err` is `None` where no closed form exists.
#[derive(Debug, Clone)]
pub struct FooFit {
  pub theta: f64,
  pub std_err: Option<f64>,
  pub nobs: usize,
  pub converged: bool,
}

/// Foo test (null: a unit root); the table holds the 1 %, 5 % and 10 % levels.
#[derive(Debug, Clone)]
pub struct FooTest {
  pub statistic: f64,
  /// `None`: no p-value table is shipped, and the decision is taken against `critical_values`.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  pub rejects_null: bool,
  pub used_lags: usize,
  pub nobs: usize,
}

impl crate::traits::HypothesisTest for FooTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}
```

- Every test leads with the four shared fields, `statistic`, `p_value: Option<f64>`,
  `critical_values: Option<CriticalValues>` and `rejects_null`, and implements `HypothesisTest`.
  `tests/test_results.rs` checks each test against the trait; add yours.
- A capability the method does not have is `None`; `Some(NaN)` means "computed, undefined here".
  A test that cannot decide off its table has `rejects_null: Option<bool>`.
- `CriticalValues::new([(level, value), ..])` holds the table and `value_at(alpha)` reads the
  largest tabulated level at or below `alpha`, so no decision runs at a size above `alpha`.
- A test's knobs travel in a `*Config` struct with a `Default` (`AdfConfig`, `KpssConfig`), `alpha`
  among them.

## 3. p-values

One `p_value` field, whatever produced it: a closed form, a table bracket (Andrews–Ploberger
answers the smallest tabulated level the statistic reaches) or a parametric bootstrap
(Leybourne–McCabe, Shapiro–Francia). The struct doc says which, and a bootstrap takes its
replication count and its seed in the config, so that the answer is reproducible.

## 4. Linear algebra

Least squares, solves, inverses, the SPD Cholesky factor and symmetric eigen-decompositions come
from `stochastic_rs_distributions::linalg` (`lstsq`, `solve`, `inverse`, `spd_cholesky_lower`,
`symmetric_eigen`). The stats crate has no `faer` dependency of its own.

A regression inside a test goes through `stationarity::common` (`ols`, `fit_adf`,
`regress_on_deterministics`, `newey_west_long_run_variance`): `ols` refuses a design past its
condition bound as `Singular("OLS design")`, and `reject_exact_fit` refuses an exact fit as
`DegenerateData("zero residual variance")`. Enter through `prepared(y, min_n)`, which validates,
refuses a constant series and scales by a power of two, and centre the series
(`centred`) before any regression with a constant, so that a level far above the variation does
not round it away.

## 5. Module doc

One prose line without TeX (rustdoc shows it raw in the module list), then the display formula,
then at most one more prose line, then one `Reference:` line per source. Every comment line stays
within 100 columns, and a `Reference:` line keeps its locator (`§`, `eq.`, `Table`, volume and
pages) and its DOI or arXiv id; shorten the title or the journal name to make it fit.

```rust
//! Foo (1999) estimator of the tail index from the `k` largest observations.
//!
//! $$
//! \hat\xi_k = \frac1k\sum_{i=1}^{k}\log X_{(i)} - \log X_{(k+1)}
//! $$
//!
//! The standard error is `ξ̂_k / √k`, the limit's.
//!
//! Reference: Foo & Bar (1999), Annals of Statistics 3(5), 1163-1174, DOI: 10.1214/aos/1176343247
```

An item doc is at most two lines. It states what the name cannot: a unit, a convention, which
inputs are the `Err`, and the panic as `panics unless <predicate>`, with every bound the call
checks, the ones a constructor it calls checks included.

## 6. Reference test

At least one test compares the answer with an independent reference (R, scipy, statsmodels,
`arch`, mpmath, a table of the paper) on a pinned seed, with a stated tolerance. Its doc names the
reference and the call that produced it.

```rust
#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stochastic::process::fbm::Fbm;
  use stochastic_rs_stochastic::traits::ProcessExt;

  /// Foo & Bar (1999), Table 3 row 2: `H = 0.7` within `0.02` on 5000 points.
  #[test]
  fn the_estimate_matches_table_3() {
    let series = Fbm::<f64, _>::new(0.7, 5_000, None, Deterministic::new(42)).sample();
    let fit = fit_foo(series.view(), 1.0).unwrap();
    assert!((fit.theta - 0.7).abs() < 0.02, "H = {}", fit.theta);
  }
}
```

- The seed goes to the workspace's own constructor; a `StdRng` handed to a `Simd*` distribution
  is discarded (`integration-test-writing` §1.1).
- A statistical assertion asks the best of three pinned seeds, as a correct test rejects a true
  null at rate `alpha` and the SIMD stream differs by platform.
- Pin the degenerate rows in `tests/degenerate_inputs.rs` (empty, a NaN at an index, a constant
  series) and the refusal of an exact fit; pin scale and level invariance where the statistic has
  them (`tests/stationarity_rescaling.rs`).

## 7. Python wrapper

Wrappers live in `stochastic-rs-stats/src/python/`, split by topic, and `stochastic-rs-py`
registers them. A `StatsError` becomes a `ValueError` through `crate::python::stats_error`, and a
precondition panic through `value_error_on_panic`.

```rust
// stochastic-rs-stats/src/python/foo.rs

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use super::stats_error;

#[pyclass(name = "FooTest", unsendable)]
pub struct PyFooTest {
  inner: crate::foo::FooTest,
}

#[pymethods]
impl PyFooTest {
  #[new]
  #[pyo3(signature = (sample, alpha=0.05))]
  fn new<'py>(sample: PyReadonlyArray1<'py, f64>, alpha: f64) -> PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      let cfg = crate::foo::FooConfig { alpha };
      let inner = crate::foo::foo_test(sample.as_array(), cfg).map_err(stats_error)?;
      Ok(Self { inner })
    })?
  }

  #[getter]
  fn statistic(&self) -> f64 {
    self.inner.statistic
  }
}
```

See `python-bindings` for registration and the pytest rows.

## 8. Anti-patterns

- A tuple return, a `0.0` or `1e-30` stand-in, an `unwrap` on data.
- A panic inside a fit on something the data decide, or an `Err` for a configuration value the
  caller wrote.
- A second regression helper beside `stationarity::common`, or a direct `faer` call.
- A statistic computed at the series' level where a constant is in the regression: centre first.
- `statrs` for distribution math: use `stochastic_rs_distributions::DistributionExt`.
- A test seeded through `StdRng` or `rand_distr` (`dev-rules` §7a keeps those for `benches/`).
- A doc block over two lines, a `# Panics` heading, TeX in a module doc's first line, a
  `Reference:` line without its locator and DOI.

## 9. Reference implementations

- `stationarity::adf::adf_test(y, cfg)`, `kpss::kpss_test`, `phillips_perron::phillips_perron_test`:
  the `*Config` and `*Test` shape, `prepared`, the exact-fit rule and the critical-value table.
- `hurst/`: five estimators behind `HurstEstimator`, each a struct with its configuration;
  `Whittle::fit` returns a `FukasawaFit` and is not a `HurstEstimator`.
- `qmle::fit_ou_mle`, `qmle::fit_diffusion_qmle`, `gmm_cir::fit_cir_gmm`: fits that scale the path
  by a power of two and answer in the data's units.
- `evt::fit_gpd`, `garch::fit_garch`, `distfit::fit_johnson_su`: maximum likelihood with
  inverse-information standard errors and a scipy or `arch` reference test.

## Related skills

- `dev-rules` — §9 comments, §14 the error policy.
- `python-bindings` — registration and tests of the wrappers.
- `integration-test-writing` — seeding and the multi-seed rule.
