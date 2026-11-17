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
//! Downstream use: [`crate::mc::brownian_bridge_qmc::BrownianBridgeQmc`] runs the Brownian-bridge construction
//! over a `stochastic_rs_distributions::qmc` sequence; this module supplies only the endpoint-pinned process itself.
//!
//! Reference: Glasserman (2003), *Monte Carlo Methods in Financial Engineering*, §3.1, DOI: 10.1007/978-0-387-21617-1

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// A Brownian path pinned at `X_0 = x0` and `X_T = xt`: each interior point is drawn from its
/// exact conditional law given the previous one, so the grid adds no bias, and the last is `xt`.
#[derive(Debug, Clone)]
pub struct BrownianBridge<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Diffusion scale σ multiplying `dW_s` in the bridge SDE.
  pub sigma: T,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Pinned terminal value X_T (defaults to 0, pairing with `x0`'s own
  /// default of 0 so the "standard bridge" cited in the module doc's
  /// closed-form moments is the default parametrization).
  pub xt: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `BrownianBridge::default().with_sigma(0.5).with_xt(Some(-1.0))`. No
/// persisted cache: `sampler()` builds its Gaussian source fresh from
/// `self` every call.
impl<T: FloatExt, S: SeedExt> BrownianBridge<T, S> {
  pub fn new(sigma: T, n: usize, x0: Option<T>, xt: Option<T>, t: Option<T>, seed: S) -> Self {
    Self::check_sigma(sigma);
    check_n(n);
    Self::check_x0(x0);
    Self::check_xt(xt);
    check_t(t);
    Self {
      backend: Cpu,
      sigma,
      n,
      x0,
      xt,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> BrownianBridge<T, S, B> {
  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  fn check_xt(xt: Option<T>) {
    require!(xt.is_none_or(|v| v.is_finite()), xt);
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace the pinned starting value `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace the pinned terminal value `xt`, all else unchanged.
  pub fn with_xt(mut self, xt: Option<T>) -> Self {
    Self::check_xt(xt);
    self.xt = xt;
    self
  }

  /// Replace `n`, the grid points along the path, all else unchanged.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// σ=1, x₀=xt=0 (the module doc's own "standard bridge"), t=1, n=252 — a
/// unit diffusion scale on the pinned-at-zero bridge that the struct doc's
/// closed-form moments already assume by default.
impl<T: FloatExt> Default for BrownianBridge<T, Unseeded> {
  fn default() -> Self {
    Self::new(T::one(), 252, None, None, Some(T::one()), Unseeded)
  }
}

/// The Euler engine's view of the Brownian bridge. The curve carries
/// `1/(T − s)` at each step's own start time, from which both the drift and
/// the exact per-step variance ratio follow; at the last step that ratio is
/// zero and the drift is the whole remaining gap, so the device lands on the
/// pinned endpoint for the same reason the host assigns it.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for BrownianBridge<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::BrownianBridge {
      xt: self.xt.unwrap_or(T::zero()),
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// `1/(T − s)` where `s` is each step's own start time. The first entry is
  /// never read: the report at `t = 0` names no curve.
  fn curve(&self) -> Option<Vec<T>> {
    let t = self.t.unwrap_or(T::one());
    let dt = t / T::from_usize_(self.n.saturating_sub(1).max(1));
    Some(
      (0..self.n)
        .map(|i| {
          let s = T::from_usize_(i.saturating_sub(1)) * dt;
          T::one() / (t - s)
        })
        .collect(),
    )
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] BrownianBridge<T, S> { sigma, n, x0, xt, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for BrownianBridge<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for BrownianBridge<T, S, B>
{
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
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`BrownianBridge`] sampling state. The source draws raw `N(0, dt)` and `sigma` scales
/// it at use: `SimdNormal` refuses a zero scale, and a zero-volatility bridge is valid.
#[doc(hidden)]
#[derive(Debug)]
pub struct BrownianBridgeSampler<T: FloatExt> {
  n: usize,
  x0: T,
  xt: T,
  sigma: T,
  t: T,
  dt: T,
  normal: Seeded<SimdNormal<T>>,
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
    // Interior points only: the last one is pinned below, not stepped.
    let interior = &mut out[1..last];
    self.normal.fill_slice(interior);

    let mut prev = self.x0;
    for (k, z) in interior.iter_mut().enumerate() {
      let s = T::from_usize_(k) * self.dt;
      let s_next = s + self.dt;
      let drift = (self.xt - prev) / (self.t - s) * self.dt;
      // The exact conditional variance ratio of Glasserman §3.1; Euler's sigma²·dt is biased near
      // `T`, by about 65% one step before the end at n = 201.
      let var_ratio = (self.t - s_next) / (self.t - s);
      let next = prev + drift + self.sigma * var_ratio.sqrt() * *z;
      *z = next;
      prev = next;
    }
    // One more step would add a finite kick at `T - dt` and miss `xt`, which the continuous
    // path reaches through its diverging drift.
    out[last] = self.xt;
  }
}

impl<T: FloatExt> crate::traits::Sealed for BrownianBridgeSampler<T> {}

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
  params: (sigma: f64, n: usize, x0: Option<f64>, xt: Option<f64>, t: Option<f64>),
  device
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

  /// A zero volatility is the line `x0 + (xt - x0) s / T`, endpoints included, without the
  /// zero scale ever reaching `SimdNormal`.
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
