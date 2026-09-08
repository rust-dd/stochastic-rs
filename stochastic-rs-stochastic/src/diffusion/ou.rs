//! # Ou
//!
//! $$
//! dX_t=\theta(\mu-X_t)\,dt+\sigma\,dW_t
//! $$
//!
//! The symbols are the field names: `theta` is the reversion speed, `mu` the
//! level the path reverts to, `sigma` the noise scale. A general `mu` is the
//! standard form of the process — the zero-mean `dX = −θX dt + σ dW` is the
//! `mu = 0` case of it, not a different equation.
//!
//! ## Same SDE as [`Vasicek`]
//!
//! [`Vasicek`] is this equation under short-rate vocabulary: its `a` is this
//! `theta` (speed), its `b` is this `mu` (level), its `r` is this `X`. The
//! two are not parallel implementations — `Vasicek` holds an `Ou`, samples
//! through [`OuSampler`] and reports this process's Euler family, so there
//! is one recursion and one device kernel behind both names. What differs is
//! the application: reach for `Ou` when the state is a spread, a log-price
//! or a mean-reverting signal, and for `Vasicek` when it is a short rate and
//! the bond, yield-curve and calibration machinery in `stochastic-rs-quant`
//! applies to it.
//!
//! Reference: Uhlenbeck G. E., Ornstein L. S. (1930) — *On the Theory of
//! the Brownian Motion*, Physical Review 36(5), 823–841,
//! DOI: 10.1103/PhysRev.36.823.
//!
//! [`Vasicek`]: crate::interest::vasicek::Vasicek

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::euler::EulerBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Ou<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed (θ in the SDE `dX = θ(μ − X) dt + σ dW`). Controls
  /// how fast `X` is pulled back toward [`mu`](Self::mu). The same quantity
  /// [`Vasicek::theta`](crate::interest::vasicek::Vasicek::theta) calls `a`.
  pub theta: T,
  /// Long-run mean level (μ in the SDE). The value `X` reverts to as
  /// `t → ∞`. The same quantity
  /// [`Vasicek::mu`](crate::interest::vasicek::Vasicek::mu) calls `b`.
  pub mu: T,
  /// Diffusion scale σ multiplying `dW_t` (σ in the SDE).
  pub sigma: T,
  /// Number of points sampled along the OU path.
  pub n: usize,
  /// Initial value X₀ of the OU path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Ou::default().with_theta(3.0).with_sigma(0.3)`. No persisted cache:
/// `sampler()` builds its Gaussian stream fresh from `self` every call.
impl<T: FloatExt, S: SeedExt> Ou<T, S> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
      backend: Cpu,
    }
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// θ=2.0, μ=0.0, σ=0.2, x₀=0 — a textbook Ou parameterization. t=1, n=252
/// — one trading year of daily steps (this crate's `Default` convention).
impl<T: FloatExt> Default for Ou<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(2.0),
      T::zero(),
      T::from_f64_fast(0.2),
      252,
      Some(T::zero()),
      Some(T::one()),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Ou<T, S> { theta, mu, sigma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> ProcessExt<T> for Ou<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = OuSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> OuSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    OuSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      mu: self.mu,
      drift_scale: self.theta * dt,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }

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

  fn sample_reduce(&self, m: usize, reduce: crate::euler::Reduce) -> Vec<T> {
    crate::euler::EulerBackend::try_euler_reduce(&self.backend, self, m, reduce)
      .unwrap_or_else(crate::device::device_panic)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`Ou`] sampling state: precomputed mean-reversion scales and the
/// owned Gaussian source. Also the whole of
/// [`Vasicek`](crate::interest::vasicek::Vasicek)'s sampler, which is an
/// alias for this type — the mean-reverting recursion is written once.
#[doc(hidden)]
pub struct OuSampler<T: FloatExt> {
  n: usize,
  x0: T,
  mu: T,
  drift_scale: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> OuSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let next = prev + self.drift_scale * (self.mu - prev) + self.diff_scale * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for OuSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Ou output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyOu, Ou,
  sig: (theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
