//! # Vg
//!
//! $$
//! X_t=\theta G_t+\sigma W_{G_t},\quad G_t\sim\Gamma(\nu^{-1}t,\nu)
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Vg<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift-in-subordinated-time θ (matches the module header's own θ,
  /// multiplying the gamma-subordinator increment `G_t`) — a skewness
  /// parameter, not a mean-reversion level; VG is a pure Lévy process
  /// with no mean reversion, despite the field's name.
  pub mu: T,
  /// Diffusion scale σ multiplying `√(G_t) W_{G_t}`, the
  /// Brownian-subordinated component riding on top of the gamma
  /// time-change.
  pub sigma: T,
  /// Variance rate ν of the gamma subordinator `G_t ~ Gamma(t/ν, ν)` —
  /// controls kurtosis/tail thickness of the VG increments, not a
  /// vol-of-vol in the Heston sense (there is no separate volatility
  /// process here).
  pub nu: T,
  /// Number of points sampled along the VG path.
  pub n: usize,
  /// Initial value X₀ of the VG path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Vg::default().with_nu(0.25).with_sigma(0.3)`. No persisted cache:
/// `sampler()` builds its gamma subordinator and Gaussian source fresh
/// from `self` every call.
impl<T: FloatExt, S: SeedExt> Vg<T, S> {
  pub fn new(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    Self {
      backend: Cpu,
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Vg<T, S, B> {
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

  /// Replace `nu`, all else unchanged.
  pub fn with_nu(mut self, nu: T) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    self.nu = nu;
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

/// μ=0.0, σ=0.2, ν=0.15, x₀=0, t=1, n=252 — a textbook Vg parameterization.
impl<T: FloatExt> Default for Vg<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::zero(),
      T::from_f64_fast(0.2),
      T::from_f64_fast(0.15),
      252,
      Some(T::zero()),
      Some(T::one()),
      Unseeded,
    )
  }
}

impl<T: FloatExt, S: SeedExt, B> Vg<T, S, B> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

/// The Euler engine's view of the variance gamma process.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Vg<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::VarianceGamma {
      mu: self.mu,
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

  fn time_step(&self) -> T {
    self.dt()
  }

  /// The gamma clock: shape `dt/ν` at scale `ν`.
  fn gamma_draws(&self) -> Option<crate::euler::GammaDraws<T>> {
    let dt = self.dt();
    Some(crate::euler::GammaDraws {
      first: (dt / self.nu, self.nu, T::zero()),
      second: None,
    })
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

backend_switch!([T: FloatExt, S: SeedExt] Vg<T, S> { mu, sigma, nu, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Vg<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = VgSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VgSampler<T> {
    // Gamma subordinator and standard-normal source derived from `self.seed`
    // in the same order as the legacy `sample()`, so the first fill matches
    // bit-for-bit; both owned sources advance on reuse for independent paths.
    let dt = self.dt();
    VgSampler {
      n: self.n,
      mu: self.mu,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      gamma: SimdGamma::<T>::new(dt / self.nu, self.nu, &self.seed),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }

  /// Through the Euler engine: on a device the increment is drawn in the
  /// kernel, on the host devices it is this process's own sampler, chunked
  /// exactly as `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
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

/// Reusable [`Vg`] sampling state: owns the gamma subordinator and the
/// Gaussian source so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
pub struct VgSampler<T: FloatExt> {
  n: usize,
  mu: T,
  sigma: T,
  x0: T,
  gamma: SimdGamma<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> VgSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut gammas = Array1::<T>::zeros(out.len() - 1);
    self.gamma.fill_slice(gammas.as_slice_mut().unwrap());
    let mut z = Array1::<T>::zeros(out.len() - 1);
    self.normal.fill_slice(z.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * z[i - 1];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for VgSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Vg output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = Vg::new(0.1_f64, 0.2, 0.3, 1, Some(2.5), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 2.5);
  }
}

py_process_1d!(PyVg, Vg,
  sig: (mu, sigma, nu, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
