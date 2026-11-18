//! Volterra Heston variance, rough Heston under the Riemann–Liouville kernel, by full truncation.
//!
//! $$
//! V_t = V_0 + \int_0^t K(t-s)\,\kappa(\theta - V_s)\,ds
//!           + \int_0^t K(t-s)\,\nu\sqrt{V_s}\,dW_s
//! $$
//!
//! Truncated at $V^+ = \max(V, 0)$ with a floored output: nonnegative always, and biased in law.
//!
//! Reference: Abi Jaber & El Euch, arXiv:1803.00477, the Volterra Heston model
//! Reference: Lord, Koekkoek & van Dijk (2010), Quantitative Finance 10(2), full truncation
//! Reference: Abi Jaber, Bayer & Breneis (2024), arXiv:2412.17526, the exact state space
//! Reference: Abi Jaber (2024), arXiv:2412.11264, the integrated classical square-root process
//! Reference: Abi Jaber & Attal (2025), arXiv:2504.19885, iVi: weak convergence, not exactness
//! Reference: Friesen, Gerhold & Wiedermann (2026), arXiv:2606.07290, rough kernels reach zero

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::noise::gn::Gn;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::kernel::VolterraKernel;
use crate::volterra::lift::VolterraLift;

/// Volterra square-root process with a nonnegativity-preserving scheme.
#[derive(Clone, Debug)]
pub struct VolterraSquareRoot<T: FloatExt, K, S: SeedExt = Unseeded, B = Cpu>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Kernel $K$ — the Riemann–Liouville kernel gives rough Heston's variance.
  pub kernel: K,
  /// Mean-reversion speed $\kappa > 0$.
  pub kappa: T,
  /// Long-run level $\theta \ge 0$.
  pub theta: T,
  /// Volatility of volatility $\nu > 0$.
  pub nu: T,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Initial variance $V_0 \ge 0$.
  pub v0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The Markov lift of `kernel` at the grid spacing, which the host steps and a device replays
  /// node by node. The grid setters rebuild it; a direct write to `n` or `t` leaves it stale.
  pub(crate) lift: VolterraLift<T, K>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, K, S: SeedExt> VolterraSquareRoot<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Panics unless `kappa > 0`, `nu > 0`, `theta >= 0`, `v0 >= 0`, `n >= 2`, a given `t` is
  /// positive and finite, and every parameter is finite.
  #[must_use]
  pub fn new(kernel: K, kappa: T, theta: T, nu: T, n: usize, v0: T, t: Option<T>, seed: S) -> Self {
    Self::check_kappa(kappa);
    Self::check_theta(theta);
    Self::check_nu(nu);
    Self::check_v0(v0);
    check_t(t);
    check_n(n);
    let lift = VolterraLift::new(kernel.clone(), grid_spacing(n, t));
    Self {
      backend: Cpu,
      kernel,
      kappa,
      theta,
      nu,
      n,
      v0,
      t,
      seed,
      lift,
    }
  }
}

/// The grid spacing of `n` points over `[0, t]`, `t` defaulting to one.
fn grid_spacing<T: FloatExt>(n: usize, t: Option<T>) -> T {
  t.unwrap_or(T::one()) / T::from_usize_(n - 1)
}

impl<T: FloatExt, K, S: SeedExt, B> VolterraSquareRoot<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  fn check_kappa(kappa: T) {
    require!(kappa.is_finite(), kappa);
    require!(kappa > T::zero(), kappa);
  }

  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
    require!(theta >= T::zero(), theta);
  }

  fn check_nu(nu: T) {
    require!(nu.is_finite(), nu);
    require!(nu > T::zero(), nu);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
    require!(v0 >= T::zero(), v0);
  }

  /// Replace `kappa`; panics unless `kappa > 0` and `kappa` is finite.
  #[must_use]
  pub fn with_kappa(mut self, kappa: T) -> Self {
    Self::check_kappa(kappa);
    self.kappa = kappa;
    self
  }

  /// Replace `theta`; panics unless `theta >= 0` and `theta` is finite.
  #[must_use]
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  /// Replace `nu`; panics unless `nu > 0` and `nu` is finite.
  #[must_use]
  pub fn with_nu(mut self, nu: T) -> Self {
    Self::check_nu(nu);
    self.nu = nu;
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
  #[must_use]
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.lift = VolterraLift::new(self.kernel.clone(), grid_spacing(n, self.t));
    self
  }

  /// Replace the initial variance $V_0$; panics unless `v0 >= 0` and `v0` is finite.
  #[must_use]
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `t`; panics unless `t`, when given, is positive and finite.
  #[must_use]
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.lift = VolterraLift::new(self.kernel.clone(), grid_spacing(self.n, t));
    self
  }

  /// Replace the seed strategy.
  #[must_use]
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt, K, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for VolterraSquareRoot<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::VolterraSquareRoot {
      kappa: self.kappa,
      theta: self.theta,
      nu: self.nu,
    }
  }

  fn initial_value(&self) -> T {
    self.v0
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The lift's nodes, decays and weights, and the boundary terms of its
  /// first step, exactly as the host's stepper holds them.
  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, T>> {
    Some(crate::euler::LiftSpec {
      decay: self.lift.exp_neg_x_dt.as_slice().expect("contiguous"),
      weight: self.lift.we.as_slice().expect("contiguous"),
      drift_scale: self.lift.one_minus_e_over_x.as_slice().expect("contiguous"),
      drift_boundary: self.lift.drift_boundary,
      diffusion_boundary: self.lift.diffusion_boundary,
      x0: self.v0,
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

backend_switch!([T: FloatExt, K, S: SeedExt] VolterraSquareRoot<T, K, S> { kernel, kappa, theta, nu, n, v0, t, seed, lift } via euler where  K: VolterraKernel<T> + Send + Sync);

impl<T: FloatExt, K, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for VolterraSquareRoot<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
}

impl<T: FloatExt, K, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for VolterraSquareRoot<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = VolterraSquareRootSampler<T, K, S>
  where
    Self: 's;

  fn sampler(&self) -> VolterraSquareRootSampler<T, K, S> {
    VolterraSquareRootSampler {
      n: self.n,
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      nu: self.nu,
      lift: self.lift.clone(),
      gn: Gn::<T, S> {
        backend: Cpu,
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
    }
  }

  /// Through the Euler engine: a device replays the lift in its kernel, the host runs this
  /// process's own sampler, chunked as [`ProcessExt`] chunks.
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

/// Reusable [`VolterraSquareRoot`] sampling state.
#[doc(hidden)]
#[derive(Debug)]
pub struct VolterraSquareRootSampler<T: FloatExt, K, S: SeedExt>
where
  K: VolterraKernel<T> + Send + Sync,
{
  n: usize,
  v0: T,
  kappa: T,
  theta: T,
  nu: T,
  lift: VolterraLift<T, K>,
  gn: Gn<T, S>,
}

impl<T: FloatExt, K, S: SeedExt> VolterraSquareRootSampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let dw = self.gn.sample();
    let (kappa, theta, nu) = (self.kappa, self.theta, self.nu);
    let path = self.lift.simulate(
      self.v0,
      |_, v| kappa * (theta - truncate(v)),
      |_, v| nu * truncate(v).sqrt(),
      dw.as_slice().expect("dw must be contiguous"),
    );
    out.copy_from_slice(path.as_slice().expect("lift path must be contiguous"));
    for v in out.iter_mut() {
      *v = truncate(*v);
    }
  }
}

/// $x^+ = \max(x, 0)$ as an explicit comparison: `T::max` returns the non-NaN operand and would
/// rewrite a NaN state to `0`, where here it falls through to the output.
#[inline]
fn truncate<T: FloatExt>(x: T) -> T {
  if x > T::zero() {
    x
  } else if x < T::zero() {
    T::zero()
  } else {
    x
  }
}

impl<T: FloatExt, K, S: SeedExt> crate::traits::Sealed for VolterraSquareRootSampler<T, K, S> where
  K: VolterraKernel<T> + Send + Sync
{
}

impl<T: FloatExt, K, S: SeedExt> PathSampler<T> for VolterraSquareRootSampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("VolterraSquareRoot output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests;
