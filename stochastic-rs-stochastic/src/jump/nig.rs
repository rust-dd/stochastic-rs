//! Normal inverse Gaussian process: a drifted Brownian motion on an inverse-Gaussian clock.
//!
//! $$
//! X_t=x_0+\theta I_t+\sigma W_{I_t},\quad \mathbb{E}(I_t)=t,\quad \mathrm{Var}(I_t)=\kappa t
//! $$
//!
//! For `σ > 0`, the NIG Lévy process at `μ = 0`: `β = θ/σ²`, `δ = σ/√κ`, `α = √(β² + 1/(κσ²))`.
//!
//! Reference: Barndorff-Nielsen & Shephard (2001), Theory Probab. Math. Statist. 65, 1–19

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Nig<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift θ per unit of the clock `I_t`, which skews the law; NIG has no mean reversion.
  pub theta: T,
  /// Diffusion scale σ multiplying `W_{I_t}`, the Brownian motion run on the IG clock.
  pub sigma: T,
  /// Variance rate of the inverse-Gaussian subordinator, whose shape is `dt²/kappa`, akin to
  /// `nu` in [`Vg`](super::vg::Vg); not a mean-reversion speed, which NIG does not have.
  pub kappa: T,
  /// Number of points sampled along the NIG path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Nig<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, `kappa > 0`, and every parameter
  /// is finite.
  pub fn new(theta: T, sigma: T, kappa: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      kappa > T::zero(),
      "kappa must satisfy `kappa > T::zero()`, got kappa = {kappa:?}"
    );
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

impl<T: FloatExt, S: SeedExt, B> Nig<T, S, B> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
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

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Nig<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Nig<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = NigSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> NigSampler<T> {
    // `G_dt ~ IG(mean = dt, shape = dt²/kappa)`. The IG subordinator, then the normal source are
    // derived from `self.seed`, an order the pinned streams depend on; reuse advances both.
    let dt = self.dt();
    let shape = dt * dt / self.kappa;
    NigSampler {
      n: self.n,
      theta: self.theta,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      ig_dist: SimdInverseGauss::<T>::new(dt, shape).seeded(&self.seed),
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device draws the increment in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
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

/// Reusable [`Nig`] sampling state: owns the inverse-Gaussian subordinator and
/// the Gaussian source so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct NigSampler<T: FloatExt> {
  n: usize,
  theta: T,
  sigma: T,
  x0: T,
  ig_dist: Seeded<SimdInverseGauss<T>>,
  normal: Seeded<SimdNormal<T>>,
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

impl<T: FloatExt> crate::traits::Sealed for NigSampler<T> {}

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

py_process_1d!(PyNig, Nig,
  sig: (theta, sigma, kappa, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, sigma: f64, kappa: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
