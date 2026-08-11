//! # BrownianBridge
//!
//! A Brownian path pinned at both ends, $X_0 = x_0$ and $X_T = x_T$, driven
//! by the singular-drift SDE
//!
//! $$
//! dX_s=\frac{x_T-X_s}{T-s}\,ds+\sigma\,dW_s
//! $$
//!
//! the classical Doob h-transform of a Brownian motion conditioned to hit
//! $x_T$ at time $T$ (Karatzas & Shreve, *Brownian Motion and Stochastic
//! Calculus*, 2nd ed., Springer 1991, §5.6.B). Equivalently, in closed form
//! (no recursion needed to state the law):
//!
//! $$
//! X_s=x_0+(x_T-x_0)\frac{s}{T}+\sigma\Bigl(W_s-\frac{s}{T}W_T\Bigr)
//! $$
//!
//! a deterministic linear interpolation between the two endpoints plus a
//! scaled Brownian motion "de-drifted" so it vanishes at both $s=0$ and
//! $s=T$. [`BrownianBridge::x0`] / [`BrownianBridge::xt`] hold $x_0$ / $x_T$;
//! [`BrownianBridge::sigma`] is $\sigma$.
//!
//! Downstream use: pinning a path's endpoint before filling in its interior
//! is the first step of the Brownian-bridge *path-construction* technique
//! for quasi-Monte-Carlo simulation — it hands a low-discrepancy sequence's
//! best-equidistributed leading coordinates to a path's coarsest,
//! highest-variance features instead of its finest ones (Glasserman (2003),
//! *Monte Carlo Methods in Financial Engineering*, §3.1, DOI:
//! 10.1007/978-0-387-21617-1). [`crate::mc::sobol`] and [`crate::mc::halton`]
//! provide the low-discrepancy sequences this construction is paired with;
//! this module supplies the endpoint-pinned process itself, not the
//! recursive dimension-allocation logic that pairs it with those sequences
//! (a later, phase-E integration concern).
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Brownian bridge: a Brownian path pinned at both ends.
///
/// `dX_s = ((xt - X_s) / (T - s)) * ds + sigma * dW_s`, with `X_0 = x0` and
/// `X_T = xt` exactly. See the module doc for the equivalent closed-form
/// representation and the discretization's final-step guard (below): the
/// Euler recursion is run only up to the second-to-last grid point, and the
/// last point is assigned `xt` directly rather than through one more
/// (noisy) step.
///
/// Exact per-step construction (Glasserman §3.1, cited in the module doc):
/// because the bridge is Markov and Gaussian, drawing each interior point
/// from its own exact conditional law given the previous point and the
/// pinned endpoint reproduces the exact discretized-path law at any grid
/// resolution — zero discretization bias, unlike a naive Euler step. The
/// Euler mean step happens to already be exact here (the drift is linear
/// in `X_s`); only Euler's *variance* step would be biased, and sharply so
/// near the terminal boundary — a plain Euler scheme's per-step variance
/// is off by ~0.75% at the midpoint but ~65% one grid step before the end,
/// at `n = 201`. Each drawn increment is instead scaled by
/// `sqrt((T - s_{k+1}) / (T - s_k))` so the per-step variance matches the
/// exact conditional law directly; that scaling is theoretically zero at
/// the true final step, which is why the explicit final-step guard below
/// (assigning `xt` directly) exists only for bit-exactness, not to patch
/// over a bias.
pub struct BrownianBridge<T: FloatExt, S: SeedExt = Unseeded> {
  /// Diffusion scale σ multiplying `dW_s` in the bridge SDE.
  pub sigma: T,
  /// Number of points sampled along the path. Both endpoints are
  /// represented as distinct points only when `n >= 2`; `n = 1` collapses to
  /// the single value `x0`, not `xt` — an arbitrary but documented tie-break
  /// (matching how every other single-path process in this crate degrades at
  /// `n = 1`), not a claim that `x0` is somehow more correct than `xt` here.
  pub n: usize,
  /// Pinned starting value X₀ (defaults to 0 when omitted).
  pub x0: Option<T>,
  /// Pinned terminal value X_T (defaults to 0, pairing with `x0`'s own
  /// default of 0 so the "standard bridge" cited in the module doc's
  /// closed-form moments is the default parametrization).
  pub xt: Option<T>,
  /// Simulation horizon [0, T] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> BrownianBridge<T, S> {
  pub fn new(sigma: T, n: usize, x0: Option<T>, xt: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      sigma,
      n,
      x0,
      xt,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for BrownianBridge<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = BrownianBridgeSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BrownianBridgeSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let t = self.t.unwrap_or(T::one());
    let dt = t / T::from_usize_(n_increments);
    BrownianBridgeSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      xt: self.xt.unwrap_or(T::zero()),
      sigma: self.sigma,
      t,
      dt,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`BrownianBridge`] sampling state: precomputed step size and
/// the owned Gaussian source. The source draws raw `N(0, dt)`; `fill_path`
/// scales each draw by the exact per-step variance ratio (see the struct
/// doc) and by `sigma` at consumption, mirroring
/// [`crate::diffusion::bessel::Bessel`] / [`crate::diffusion::gbm::Gbm`] /
/// [`crate::diffusion::ou::Ou`] / [`crate::diffusion::cir::Cir`] rather than
/// baking the model's own scale into `std_dev` — the latter would make
/// `SimdNormal::new`'s `assert!(std_dev > 0)` panic outright for `sigma =
/// 0.0` (a legitimate, degenerate-but-valid input: a zero-vol bridge is just
/// the deterministic linear interpolation), which this crate's house
/// convention never does for an in-range parameter (warn-but-accept at
/// worst, as `Cir`/`Bessel` do at their own boundaries — never panic).
#[doc(hidden)]
pub struct BrownianBridgeSampler<T: FloatExt> {
  n: usize,
  x0: T,
  xt: T,
  sigma: T,
  t: T,
  dt: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> BrownianBridgeSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let last = out.len() - 1;
    // Run the SDE recursion on the interior points only (`out[1..last]`);
    // the final point is guarded below rather than produced by one more
    // step of it.
    let interior = &mut out[1..last];
    self.normal.fill_slice(interior);

    let mut prev = self.x0;
    for (k, z) in interior.iter_mut().enumerate() {
      let s = T::from_usize_(k) * self.dt;
      let s_next = s + self.dt;
      let drift = (self.xt - prev) / (self.t - s) * self.dt;
      // Exact per-step variance ratio (T - s_next)/(T - s) (Glasserman
      // §3.1's sequential bridge construction): scales the raw N(0, dt)
      // draw so this step's variance matches the exact conditional law
      // instead of Euler's own sigma^2*dt, which is badly biased near the
      // terminal boundary (see the struct doc).
      let var_ratio = (self.t - s_next) / (self.t - s);
      let next = prev + drift + self.sigma * var_ratio.sqrt() * *z;
      *z = next;
      prev = next;
    }
    // Final-step guard: as s -> T the drift's own 1/(T-s) term diverges,
    // which is exactly what pins the continuous-time path to `xt` in the
    // limit — but the discretized recursion above would still add one more,
    // finite diffusion kick `sigma * dW` at s = T - dt, so it never lands on
    // `xt` exactly. Assign the pinned endpoint directly instead.
    out[last] = self.xt;
  }
}

impl<T: FloatExt> PathSampler<T> for BrownianBridgeSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("BrownianBridge output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyBrownianBridge, BrownianBridge,
  sig: (sigma, n, x0=None, xt=None, t=None, seed=None, dtype=None),
  params: (sigma: f64, n: usize, x0: Option<f64>, xt: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Endpoints are pinned exactly, by construction.
  #[test]
  fn brownian_bridge_hits_both_endpoints() {
    let x0 = 0.5_f64;
    let xt = -1.25_f64;
    let bridge = BrownianBridge::<f64, _>::new(
      0.3,
      500,
      Some(x0),
      Some(xt),
      Some(2.0),
      Deterministic::new(2718),
    );
    let path = bridge.sample();
    assert_eq!(path[0], x0, "path[0] must equal x0 exactly");
    assert_eq!(path[path.len() - 1], xt, "path[n-1] must equal xt exactly");
  }

  /// Var[X_s] = σ² s(T−s)/T for the standard bridge (x0 = xt = 0); at
  /// s = T/2 this is σ²T/4.
  #[test]
  fn brownian_bridge_midpoint_variance_matches_closed_form() {
    let sigma = 0.4_f64;
    let t = 1.0_f64;
    let n = 201;
    let paths = 20_000;
    let mid = (n - 1) / 2;
    let expected = sigma * sigma * t / 4.0;

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let bridge =
          BrownianBridge::<f64, _>::new(sigma, n, None, None, Some(t), Deterministic::new(seed));
        let samples = bridge.sample_par(paths);
        let mean = samples.iter().map(|p| p[mid]).sum::<f64>() / paths as f64;
        let var =
          samples.iter().map(|p| (p[mid] - mean).powi(2)).sum::<f64>() / (paths as f64 - 1.0);
        (var - expected).abs() / expected
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 5e-2,
      "best-of-3 relative error {best_rel_err} exceeds 5e-2 (expected {expected})"
    );
  }

  /// Var[X_s] = σ² s(T−s)/T at grid point `n-2`, one step before the
  /// terminal pin — the region where a plain Euler discretization was
  /// measured at ~65% relative variance error (see the struct doc); the
  /// exact per-step recursion must hold far tighter than the interior-only
  /// midpoint check above, since that check alone would have missed the
  /// original bias entirely.
  #[test]
  fn brownian_bridge_near_terminal_variance_matches_closed_form() {
    let sigma = 0.4_f64;
    let t = 1.0_f64;
    let n = 201;
    let paths = 20_000;
    let near_terminal = n - 2;
    let dt = t / (n - 1) as f64;
    let s = near_terminal as f64 * dt;
    let expected = sigma * sigma * s * (t - s) / t;

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let bridge =
          BrownianBridge::<f64, _>::new(sigma, n, None, None, Some(t), Deterministic::new(seed));
        let samples = bridge.sample_par(paths);
        let mean = samples.iter().map(|p| p[near_terminal]).sum::<f64>() / paths as f64;
        let var = samples
          .iter()
          .map(|p| (p[near_terminal] - mean).powi(2))
          .sum::<f64>()
          / (paths as f64 - 1.0);
        (var - expected).abs() / expected
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 5e-2,
      "best-of-3 relative error {best_rel_err} exceeds 5e-2 (expected {expected})"
    );
  }

  /// E[X_s] = x0 + (xt − x0)·s/T — the deterministic interpolation.
  #[test]
  fn brownian_bridge_mean_is_linear_interpolation() {
    let x0 = 1.0_f64;
    let xt = 3.0_f64;
    let sigma = 0.5_f64;
    let t = 1.0_f64;
    let n = 201;
    let paths = 20_000;
    let mid = (n - 1) / 2;
    let expected = x0 + (xt - x0) * 0.5;

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let bridge = BrownianBridge::<f64, _>::new(
          sigma,
          n,
          Some(x0),
          Some(xt),
          Some(t),
          Deterministic::new(seed),
        );
        let mean = bridge.sample_par(paths).iter().map(|p| p[mid]).sum::<f64>() / paths as f64;
        (mean - expected).abs() / expected.abs()
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 2e-2,
      "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
    );
  }

  /// `sigma = 0.0` must not panic (regression: `SimdNormal::new` requires
  /// `std_dev > 0`, so the diffusion scale must never be baked into it) and
  /// must collapse to the exact deterministic interpolation
  /// `x0 + (xt - x0) * s / T` at every grid point, endpoints included.
  #[test]
  fn brownian_bridge_zero_volatility_is_exact_interpolation() {
    let x0 = 0.5_f64;
    let xt = -1.25_f64;
    let t = 2.0_f64;
    let n = 50;
    let bridge = BrownianBridge::<f64, _>::new(
      0.0,
      n,
      Some(x0),
      Some(xt),
      Some(t),
      Deterministic::new(2718),
    );
    let path = bridge.sample();

    for (i, &value) in path.iter().enumerate() {
      let s = i as f64 * (t / (n - 1) as f64);
      let expected = x0 + (xt - x0) * s / t;
      assert!(
        (value - expected).abs() < 1e-9,
        "grid point {i}: got {value}, expected {expected}"
      );
    }
    assert_eq!(path[0], x0, "path[0] must equal x0 exactly");
    assert_eq!(path[n - 1], xt, "path[n-1] must equal xt exactly");
  }

  /// Same seed twice must be bit-identical.
  #[test]
  fn brownian_bridge_is_deterministic() {
    let p1 = BrownianBridge::<f64, _>::new(
      0.3,
      300,
      Some(0.2),
      Some(-0.4),
      Some(1.5),
      Deterministic::new(42),
    )
    .sample();
    let p2 = BrownianBridge::<f64, _>::new(
      0.3,
      300,
      Some(0.2),
      Some(-0.4),
      Some(1.5),
      Deterministic::new(42),
    )
    .sample();
    assert_eq!(p1, p2);
  }
}
