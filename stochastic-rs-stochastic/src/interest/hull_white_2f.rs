//! # Hull White 2f
//!
//! $$
//! dr_t=\bigl[\theta(t)+u_t-a\,r_t\bigr]dt+\sigma_1\,dW_t^1,\qquad
//! du_t=-b\,u_t\,dt+\sigma_2\,dW_t^2
//! $$
//!
//! State-space (Brigo–Mercurio §4.2-style) two-factor Hull-White: `r_t`
//! (returned as `x` below) directly carries the short rate — the
//! calibration function `θ(t)` and the auxiliary factor `u_t` both feed
//! straight into `r`'s own drift. Output is `[r, u]`; unlike the
//! equivalent G2++ parametrization, `r` is never formed by summing two
//! independent zero-drift factors — `u_t` is that second factor, but it
//! enters `r`'s drift directly rather than being added to it externally.
//!
//! References:
//! - Hull J. & White A. (1994) — *Numerical Procedures for Implementing
//!   Term Structure Models II: Two-Factor Models*, Journal of
//!   Derivatives 2(2), 37–48, DOI: 10.3905/jod.1994.407908.
//! - Brigo D. & Mercurio F. (2006) — *Interest Rate Models — Theory and
//!   Practice*, 2nd ed., Springer Finance,
//!   DOI: 10.1007/978-3-540-34604-3 — source of the §4.2-style
//!   state-space parametrization above.
//!
use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct HullWhite2F<T: FloatExt, S: SeedExt = Unseeded> {
  /// Time-dependent drift target $\theta(t)$, fitted to the initial term
  /// structure — the additive role `HullWhite::theta` plays for the
  /// single-factor model.
  pub theta: Fn1D<T>,
  /// Mean-reversion speed $a$ of the primary state variable.
  pub a: T,
  /// Diffusion/noise scale for factor 1.
  pub sigma1: T,
  /// Diffusion/noise scale for factor 2.
  pub sigma2: T,
  /// Instantaneous correlation ρ between the two driving Brownian motions
  /// `dW1`/`dW2`.
  pub rho: T,
  /// Mean-reversion speed b of the auxiliary factor u_t (multiplies
  /// `-u_{t-1}` in u's own drift) — not a diffusion term despite the
  /// field's former doc; `sigma1`/`sigma2` are the diffusion terms.
  pub b: T,
  /// Initial short rate r₀ (returned as `x[0]`).
  pub x0: Option<T>,
  /// Simulation horizon [0, t] shared by both factors (defaults to 1 when
  /// omitted).
  pub t: Option<T>,
  /// Number of points sampled along each of the `r`/`u` paths.
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  cgns: Cgns<T>,
}

impl<T: FloatExt, S: SeedExt> HullWhite2F<T, S> {
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    a: T,
    sigma1: T,
    sigma2: T,
    rho: T,
    b: T,
    x0: Option<T>,
    t: Option<T>,
    n: usize,
    seed: S,
  ) -> Self {
    Self {
      theta: theta.into(),
      a,
      sigma1,
      sigma2,
      rho,
      b,
      x0,
      t,
      n,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for HullWhite2F<T, S> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HullWhite2FSampler<'s, T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> HullWhite2FSampler<'_, T, S> {
    HullWhite2FSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      a: self.a,
      sigma1: self.sigma1,
      sigma2: self.sigma2,
      b: self.b,
      theta: &self.theta,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`HullWhite2F`] sampling state. Borrows the process for its
/// time-dependent drift `theta(t)` and owns the correlated-Gaussian generator
/// plus the seed source so a Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
pub struct HullWhite2FSampler<'a, T: FloatExt, S: SeedExt> {
  n: usize,
  x0: T,
  a: T,
  sigma1: T,
  sigma2: T,
  b: T,
  theta: &'a Fn1D<T>,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> HullWhite2FSampler<'_, T, S> {
  fn fill_paths(&mut self, x: &mut [T], u: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    x[0] = self.x0;
    u[0] = T::zero();

    for i in 1..self.n {
      x[i] = x[i - 1]
        + (self.theta.call(T::from_usize_(i) * dt) + u[i - 1] - self.a * x[i - 1]) * dt
        + self.sigma1 * cgn1[i - 1];

      u[i] = u[i - 1] - self.b * u[i - 1] * dt + self.sigma2 * cgn2[i - 1];
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for HullWhite2FSampler<'_, T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x_arr, u_arr] = out;
    let x = x_arr
      .as_slice_mut()
      .expect("HullWhite2F output must be contiguous");
    let u = u_arr
      .as_slice_mut()
      .expect("HullWhite2F output must be contiguous");
    self.fill_paths(x, u);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut x = Array1::<T>::zeros(self.n);
    let mut u = Array1::<T>::zeros(self.n);
    self.fill_paths(
      x.as_slice_mut().expect("contiguous"),
      u.as_slice_mut().expect("contiguous"),
    );
    [x, u]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHullWhite2F {
  inner: Option<HullWhite2F<f64>>,
  seeded: Option<HullWhite2F<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHullWhite2F {
  // Python-visible parameter names stay `k`/`theta` (pre-existing public
  // API surface); `k=` forwards into `HullWhite2F::new`'s drift-function
  // `theta` parameter and `theta=` forwards into its mean-reversion-speed
  // `a` parameter, so the Python signature is unaffected but its keyword
  // names map onto differently-named Rust parameters.
  #[new]
  #[pyo3(signature = (k, theta, sigma1, sigma2, rho, b, n, x0=None, t=None, seed=None))]
  fn new(
    k: pyo3::Py<pyo3::PyAny>,
    theta: f64,
    sigma1: f64,
    sigma2: f64,
    rho: f64,
    b: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(HullWhite2F::new(
          Fn1D::Py(k),
          theta,
          sigma1,
          sigma2,
          rho,
          b,
          x0,
          t,
          n,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(HullWhite2F::new(
          Fn1D::Py(k),
          theta,
          sigma1,
          sigma2,
          rho,
          b,
          x0,
          t,
          n,
          Unseeded,
        )),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn const_theta(_t: f64) -> f64 {
    0.5
  }

  #[test]
  fn sample_returns_two_paths() {
    let hw2 = HullWhite2F::<f64>::new(
      const_theta as fn(f64) -> f64,
      0.04,
      0.01,
      0.005,
      -0.3,
      0.4,
      Some(0.04),
      Some(1.0),
      64,
      Unseeded,
    );
    let [x, u] = hw2.sample();
    assert_eq!(x.len(), 64);
    assert_eq!(u.len(), 64);
  }

  /// Guards the field-vs-doc contradiction fixed in A1-b: `theta` is the
  /// additive time-dependent target θ(t) and `a` is the multiplicative
  /// mean-reversion speed — matches this module's own SDE and the sibling
  /// `HullWhite::theta`/`alpha` split. Also pins the pre-rename recursion
  /// (with diffusion zeroed out) as a behavior-regression guard.
  #[test]
  fn hw2f_a_and_theta_zero_diffusion_matches_deterministic_euler() {
    let a = 0.6_f64;
    let b = 0.3_f64;
    let n = 33;
    let t = 1.0_f64;
    let x0 = 0.02_f64;

    let hw2 = HullWhite2F::<f64>::new(
      const_theta as fn(f64) -> f64,
      a,
      0.0,
      0.0,
      -0.3,
      b,
      Some(x0),
      Some(t),
      n,
      Unseeded,
    );
    assert_eq!(hw2.a, a);
    assert_eq!(hw2.theta.call(0.25), const_theta(0.25));

    let [x, u] = hw2.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected_x = x0;
    let mut expected_u = 0.0_f64;
    for i in 1..n {
      let next_x = expected_x + (const_theta(i as f64 * dt) + expected_u - a * expected_x) * dt;
      let next_u = expected_u - b * expected_u * dt;
      expected_x = next_x;
      expected_u = next_u;
      assert!((x[i] - expected_x).abs() < 1e-12, "x mismatch at {i}");
      assert!((u[i] - expected_u).abs() < 1e-12, "u mismatch at {i}");
    }
  }
}
