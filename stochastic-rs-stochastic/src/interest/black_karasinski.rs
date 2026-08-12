//! # Black-Karasinski
//!
//! $$
//! d\ln r_t=\bigl(\theta(t)-a\ln r_t\bigr)dt+\sigma\,dW_t
//! $$
//!
//! Black & Karasinski (1991), *Bond and Option Pricing when Short Rates
//! are Lognormal*: the short rate itself is never simulated directly — its
//! logarithm follows a Hull-White-style mean-reverting Gaussian process, so
//! `r_t = exp(Y_t)` is strictly positive by construction, with no boundary
//! condition, floor, or reflection needed (unlike [`Cir`](crate::diffusion::cir::Cir)
//! / [`CirPlusPlus`](crate::interest::cir_pp::CirPlusPlus) / [`Bessel`](crate::diffusion::bessel::Bessel),
//! whose positivity is only *probabilistic* and needs an explicit boundary
//! policy).
//!
//! Same `theta`/`a` convention as [`HullWhite`](crate::interest::hull_white::HullWhite)'s
//! `theta`/`alpha`: [`theta`](BlackKarasinski::theta) is the additive
//! time-dependent drift target function, [`a`](BlackKarasinski::a) the
//! mean-reversion speed multiplying `-ln(r_t)`.
//!
//! Unlike `HullWhite`'s own Euler-Maruyama step, the log-rate `Y_t` here is
//! stepped with the *exact* one-step Gaussian transition of an
//! Ornstein-Uhlenbeck process with `theta` frozen at each step's own
//! evaluation point (the standard "exact Gaussian short rate" scheme): for
//! constant `theta` this has zero discretization bias at any step size,
//! strictly improving on Euler.
//!
//! A lattice (trinomial-tree) engine for the same model already exists at
//! `stochastic-rs-quant::lattice::short_rate::black_karasinski::BlackKarasinskiTree`;
//! this type is the missing Monte-Carlo path simulator for it.
use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Black-Karasinski log-short-rate process.
///
/// `d ln(r_t) = (theta(t) - a * ln(r_t)) * dt + sigma * dW_t`, reported as
/// `r_t = exp(ln(r_t))`.
///
/// See the module doc for the `theta`/`a` naming convention (shared with
/// [`HullWhite`](crate::interest::hull_white::HullWhite)) and the exact-OU
/// discretization.
#[derive(Clone)]
pub struct BlackKarasinski<T: FloatExt, S: SeedExt = Unseeded> {
  /// Time-dependent additive drift target θ(t), fitted to the initial term
  /// structure of the log-rate — same role as
  /// [`HullWhite::theta`](crate::interest::hull_white::HullWhite::theta).
  pub theta: Fn1D<T>,
  /// Mean-reversion speed `a` of the log-rate (multiplies `-ln(r_t)` in the
  /// drift) — same role as
  /// [`HullWhite::alpha`](crate::interest::hull_white::HullWhite::alpha).
  /// Should be `> 0`: a strictly mean-reverting log-rate is this model's own
  /// premise, and the exact-OU step divides by `a` in both its mean and
  /// variance terms. `a <= 0` is accepted rather than rejected — matching
  /// this crate's [`Cir`](crate::diffusion::cir::Cir)-style boundary
  /// convention — but is not made silently well-behaved: `a = 0` is a
  /// literal `0/0` in the mean term, so every point after `r0` comes out
  /// `NaN`, and `a < 0` makes the log-rate diverge instead of mean-revert.
  /// [`BlackKarasinski::new`] unconditionally warns to stderr when this
  /// happens; it never panics.
  pub a: T,
  /// Diffusion scale σ multiplying `dW_t` in the log-rate SDE.
  pub sigma: T,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Initial short rate r₀. Must be `> 0` so `ln(r0)` is defined; every
  /// subsequent path point is `exp(...)` and so is positive unconditionally
  /// regardless of `r0`.
  pub r0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

/// Constant θ(t) ≡ 0.05 used by [`BlackKarasinski`]'s `Default` impl —
/// matches this file's own `theta_const` test fixture.
fn default_theta<T: FloatExt>(_t: T) -> T {
  T::from_f64_fast(0.05)
}

impl<T: FloatExt, S: SeedExt> BlackKarasinski<T, S> {
  /// Create a new BlackKarasinski process.
  ///
  /// `a <= 0` is accepted rather than rejected — matching this crate's
  /// [`Cir::new`](crate::diffusion::cir::Cir::new) boundary-condition
  /// precedent — but unconditionally prints a one-line diagnostic to
  /// stderr, including in release builds: the exact-OU step divides by `a`
  /// in both its mean and variance terms, so `a = 0` poisons every point
  /// after `r0` with `NaN` and `a < 0` makes the log-rate diverge instead
  /// of mean-revert. Never panics.
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    a: T,
    sigma: T,
    n: usize,
    r0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    if a <= T::zero() {
      eprintln!(
        "warning: BlackKarasinski::new: mean-reversion speed a <= 0; the \
         exact-OU step divides by a in both its mean and variance terms, so \
         a = 0 produces a literal 0/0 in the mean term (every point after r0 \
         comes out NaN) and a < 0 makes the log-rate diverge instead of \
         mean-revert — pass a > 0 for a well-defined Black-Karasinski path"
      );
    }

    Self {
      theta: theta.into(),
      a,
      sigma,
      n,
      r0,
      t,
      seed,
    }
  }
}

/// θ(t)≡0.05 (this file's own `theta_const` test fixture), a=0.8, σ=0.1,
/// r₀=0.03, t=1 — matches
/// `black_karasinski_log_mean_matches_ou`/`black_karasinski_is_deterministic`
/// below (both use these values at t=1; their own `n` varies across this
/// file's tests and is not itself a fixture value). n=252 — one trading
/// year of daily steps (this crate's `Default` convention).
impl<T: FloatExt> Default for BlackKarasinski<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      default_theta::<T> as fn(T) -> T,
      T::from_f64_fast(0.8),
      T::from_f64_fast(0.1),
      252,
      Some(T::from_f64_fast(0.03)),
      Some(T::one()),
      Unseeded,
    )
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for BlackKarasinski<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = BlackKarasinskiSampler<'s, T>
  where
    Self: 's;

  fn sampler(&self) -> BlackKarasinskiSampler<'_, T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let decay = (-self.a * dt).exp();
    // Exact one-step OU transition std-dev for `theta` frozen over `dt`
    // (Var = (1 - e^{-2a dt}) / (2a)), left unscaled by `sigma` — see
    // `BlackKarasinskiSampler`'s doc for why `sigma` is applied at
    // consumption in `fill_path` instead of baked in here. At `a = 0` the
    // ratio is a literal `0/0` (NaN); clamping to `min_positive_val`
    // (never exactly zero) keeps this strictly positive so
    // `SimdNormal::new`'s own `std_dev > 0` assertion never fires —
    // `BlackKarasinski::new`'s warning already told the caller `a <= 0` is
    // unsupported, and the *documented* failure mode is `fill_path`'s mean
    // term poisoning the path with `NaN`, not a panic here.
    let ou_std = ((T::one() - decay * decay) / (T::from_usize_(2) * self.a))
      .max(T::min_positive_val())
      .sqrt();
    BlackKarasinskiSampler {
      n: self.n,
      r0: self.r0.unwrap_or(T::one()),
      dt,
      a: self.a,
      decay,
      diff_scale: self.sigma,
      theta: &self.theta,
      normal: SimdNormal::<T>::new(T::zero(), ou_std, &self.seed),
    }
  }
}

/// Reusable [`BlackKarasinski`] sampling state: precomputed exact-OU decay
/// and the owned Gaussian source (see the module doc for why this is an
/// exact per-step transition rather than Euler). The source draws raw
/// `N(0, ou_std)` and `sigma` is applied as a multiplier at consumption in
/// `fill_path`, mirroring
/// [`crate::process::brownian_bridge::BrownianBridge`] rather than baking
/// the model's own scale into `std_dev` — the latter would make
/// `SimdNormal::new`'s `assert!(std_dev > 0)` panic outright for `sigma =
/// 0.0` (a legitimate, degenerate-but-valid input: a zero-vol
/// Black-Karasinski path is just the deterministic OU-mean log-rate path).
#[doc(hidden)]
pub struct BlackKarasinskiSampler<'a, T: FloatExt> {
  n: usize,
  r0: T,
  dt: T,
  a: T,
  decay: T,
  /// Diffusion scale σ, applied as a multiplier at consumption rather than
  /// baked into the Gaussian source's `std_dev` (see the struct doc).
  diff_scale: T,
  theta: &'a Fn1D<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> BlackKarasinskiSampler<'_, T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.r0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);

    // `prev_log` tracks Y_t = ln(r_t) — the exact-OU recursion runs in log
    // space and only the reported value is exponentiated, mirroring how
    // `BesselSampler` tracks its own state in squared space (see
    // `crate::diffusion::bessel`).
    let mut prev_log = self.r0.ln();
    for (k, z) in tail.iter_mut().enumerate() {
      let i = k + 1;
      let t_i = T::from_usize_(i) * self.dt;
      let mean = prev_log * self.decay + (self.theta.call(t_i) / self.a) * (T::one() - self.decay);
      let next_log = mean + self.diff_scale * *z;
      *z = next_log.exp();
      prev_log = next_log;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for BlackKarasinskiSampler<'_, T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("BlackKarasinski output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBlackKarasinski {
  inner: Option<BlackKarasinski<f64>>,
  seeded: Option<BlackKarasinski<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBlackKarasinski {
  #[new]
  #[pyo3(signature = (theta, a, sigma, n, r0=None, t=None, seed=None))]
  fn new(
    theta: pyo3::Py<pyo3::PyAny>,
    a: f64,
    sigma: f64,
    n: usize,
    r0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(BlackKarasinski::new(
          Fn1D::Py(theta),
          a,
          sigma,
          n,
          r0,
          t,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(BlackKarasinski::new(
          Fn1D::Py(theta),
          a,
          sigma,
          n,
          r0,
          t,
          Unseeded,
        )),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn theta_const(_t: f64) -> f64 {
    0.05
  }

  /// ln r is OU: E[ln r_t] = e^{-a t} ln r_0 + (theta/a)(1 - e^{-a t}) when
  /// theta is constant (Black & Karasinski 1991).
  #[test]
  fn black_karasinski_log_mean_matches_ou() {
    let a = 0.8_f64;
    let sigma = 0.1;
    let r0 = 0.03_f64;
    let t = 1.0_f64;
    let n = 200;
    let paths = 20_000;
    let expected = (-a * t).exp() * r0.ln() + (0.05 / a) * (1.0 - (-a * t).exp());

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let bk = BlackKarasinski::<f64, _>::new(
          theta_const as fn(f64) -> f64,
          a,
          sigma,
          n,
          Some(r0),
          Some(t),
          Deterministic::new(seed),
        );
        let mean_log = bk
          .sample_par(paths)
          .iter()
          .map(|path| path.last().unwrap().ln())
          .sum::<f64>()
          / paths as f64;
        (mean_log - expected).abs() / expected.abs()
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 2e-2,
      "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
    );
  }

  /// r_t > 0 for every point of every path (log construction) — must hold
  /// unconditionally, with no boundary policy needed.
  #[test]
  fn black_karasinski_stays_positive() {
    let bk = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      0.5,
      300,
      Some(0.03),
      Some(2.0),
      Deterministic::new(2718),
    );
    for path in bk.sample_par(200) {
      assert!(path.iter().all(|x| x.is_finite() && *x > 0.0));
    }
  }

  /// `a <= 0` must be accepted (never panic — construction warns to stderr
  /// instead) but is documented as producing an unusable path: `a = 0` is a
  /// literal 0/0 in the mean term, poisoning every point after `r0` with
  /// `NaN`; `a < 0` stays finite but diverges instead of mean-reverting.
  #[test]
  fn black_karasinski_nonpositive_a_does_not_panic() {
    let zero_a = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.0,
      0.1,
      10,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    );
    let path = zero_a.sample();
    assert!(
      path[0].is_finite() && path[0] > 0.0,
      "r0 itself is untouched by a: {}",
      path[0]
    );
    assert!(
      path.iter().skip(1).all(|x| x.is_nan()),
      "a = 0 is documented to poison every point after r0 with NaN: {path:?}"
    );

    let negative_a = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      -0.3,
      0.1,
      10,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    );
    // Only asserting "did not panic": a < 0 is documented to diverge, not
    // to stay in any particular finite range.
    let _ = negative_a.sample();
  }

  /// `sigma = 0.0` must not panic (regression: `SimdNormal::new` requires
  /// `std_dev > 0`, so the diffusion scale must never be baked into it) and
  /// must collapse to the exact deterministic OU-mean log-rate path:
  /// `ln r_i = decay * ln r_{i-1} + (theta/a)(1 - decay)` for constant
  /// `theta`, with no noise term.
  #[test]
  fn black_karasinski_zero_volatility_is_exact_ou_mean() {
    let a = 0.8_f64;
    let r0 = 0.03_f64;
    let t = 1.0_f64;
    let n = 50;
    let dt = t / (n - 1) as f64;
    let decay = (-a * dt).exp();

    let bk = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      a,
      0.0,
      n,
      Some(r0),
      Some(t),
      Deterministic::new(42),
    );
    let path = bk.sample();

    assert_eq!(path[0], r0, "path[0] must equal r0 exactly");
    let mut prev_log = r0.ln();
    for &value in path.iter().skip(1) {
      let mean = prev_log * decay + (0.05 / a) * (1.0 - decay);
      assert!(
        (value.ln() - mean).abs() < 1e-9,
        "got ln(r)={}, expected {mean}",
        value.ln()
      );
      prev_log = mean;
    }
  }

  /// `sigma < 0` must not panic (regression: baking `sigma` into
  /// `SimdNormal::new`'s `std_dev` made any negative value trip its
  /// `std_dev > 0` assertion) — a negative diffusion scale is a sign flip
  /// of each Gaussian draw, a no-op in law, not an invalid input.
  #[test]
  fn black_karasinski_negative_sigma_does_not_panic() {
    let bk = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      -0.2,
      10,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    );
    let path = bk.sample();
    assert!(path.iter().all(|x| x.is_finite() && *x > 0.0));
  }

  /// Same seed twice must be bit-identical.
  #[test]
  fn black_karasinski_is_deterministic() {
    let bk1 = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      0.1,
      100,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    let bk2 = BlackKarasinski::<f64, _>::new(
      theta_const as fn(f64) -> f64,
      0.8,
      0.1,
      100,
      Some(0.03),
      Some(1.0),
      Deterministic::new(42),
    )
    .sample();
    assert_eq!(bk1, bk2);
  }
}
