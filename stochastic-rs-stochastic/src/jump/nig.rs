//! # Nig
//!
//! $$
//! X_t=\mu t+\beta I_t+W_{I_t},\quad I_t\sim\mathrm{Ig}(\delta t,\gamma)
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Nig<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Skewness-in-subordinated-time β (matches the module header's own β,
  /// multiplying the IG-subordinator increment `I_t`). NIG is a pure Lévy
  /// process with **no mean reversion** — despite the field's name, this
  /// is not a target level.
  pub theta: T,
  /// Diffusion scale σ multiplying `√(I_t) dW_t`, the Brownian-subordinated
  /// component riding on top of the IG time-change.
  pub sigma: T,
  /// Variance-rate of the inverse-Gaussian subordinator (its shape
  /// parameter is `dt²/kappa`) — controls the IG time-change's dispersion,
  /// akin to `nu` in [`Vg`](super::vg::Vg). Despite the letter, this is
  /// **not** a mean-reversion speed; NIG has no mean reversion.
  pub kappa: T,
  /// Number of points sampled along the NIG path.
  pub n: usize,
  /// Initial value X₀ of the NIG path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Nig<T, S> {
  pub fn new(theta: T, sigma: T, kappa: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(kappa > T::zero(), "kappa must be positive");
    Self {
      backend: Cpu,
      theta,
      sigma,
      kappa,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Nig<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B> Nig<T, S, B> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

/// The Euler engine's view of the normal inverse-Gaussian process: the
/// inverse-Gaussian clock and the Brownian shock it scales, in one step.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Nig<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let two = T::from_f64_fast(2.0);
    let four = T::from_f64_fast(4.0);
    let dt = self.dt();
    let shape = dt * dt / self.kappa;
    crate::euler::EulerSpec::NormalInverseGaussian {
      theta: self.theta,
      sigma: self.sigma,
      mu_ig: dt,
      two_lam: two * shape,
      four_mu_lam: four * dt * shape,
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

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Nig<T, S> { theta, sigma, kappa, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Nig<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = NigSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> NigSampler<T> {
    // For Nig: G_dt ~ Ig(mean=dt, shape=dt^2/kappa). The IG subordinator and
    // the standard-normal source are derived from `self.seed` in the same
    // order as the legacy `sample()`, so the first fill reproduces it
    // bit-for-bit; both owned sources advance on reuse for independent paths.
    let dt = self.dt();
    let shape = dt * dt / self.kappa;
    NigSampler {
      n: self.n,
      theta: self.theta,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      ig_dist: SimdInverseGauss::<T>::new(dt, shape, &self.seed),
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

/// Reusable [`Nig`] sampling state: owns the inverse-Gaussian subordinator and
/// the Gaussian source so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
pub struct NigSampler<T: FloatExt> {
  n: usize,
  theta: T,
  sigma: T,
  x0: T,
  ig_dist: SimdInverseGauss<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> NigSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut ig = Array1::<T>::zeros(out.len() - 1);
    self.ig_dist.fill_slice(ig.as_slice_mut().unwrap());
    let mut z = Array1::<T>::zeros(out.len() - 1);
    self.normal.fill_slice(z.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * z[i - 1]
    }
  }
}

impl<T: FloatExt> PathSampler<T> for NigSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Nig output must be contiguous"));
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
    let p = Nig::new(0.1_f64, 0.2, 0.3, 1, Some(4.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 4.0);
  }
}

py_process_1d!(PyNig, Nig,
  sig: (theta, sigma, kappa, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, sigma: f64, kappa: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
