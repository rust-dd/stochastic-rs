//! $$
//! X_t = X_0 + \int_0^t K(t-s)\,b(s,X_s)\,ds + \int_0^t K(t-s)\,\sigma(s,X_s)\,dW_s
//! $$
//!
//! [`VolterraLift`] as a [`ProcessExt`]: any [`VolterraKernel`] with time-and-state coefficients
//! at $O(nN')$. The cited rates are the literature's for this scheme class, not measured here.
//!
//! Reference: Abi Jaber & El Euch (2018), arXiv:1801.10359
//! Reference: Li, Huang & Hu (2020), arXiv:2004.04916, rate $\min\{1-\alpha,\,\tfrac12-\beta\}$
//! Reference: Liu, Hu & Gao (2024), arXiv:2412.11126, $n^{-2H}$ is optimal for Milstein

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::noise::gn::Gn;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::Fn2D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::kernel::VolterraKernel;
use crate::volterra::lift::VolterraLift;

/// General stochastic Volterra equation by Markovian lift at $O(nN')$, over any
/// [`VolterraKernel`]. No `Default`: a kernel has no canonical value.
#[derive(Clone, Debug)]
pub struct VolterraSde<T: FloatExt, K, S: SeedExt = Unseeded, B = Cpu>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Kernel $K$: exact where one exists (an `ExponentialKernel`), otherwise its $N'$-term
  /// exponential-sum fit.
  pub kernel: K,
  /// Drift coefficient $b(s, X_s)$.
  pub drift: Fn2D<T>,
  /// Diffusion coefficient $\sigma(s, X_s)$.
  pub diffusion: Fn2D<T>,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
  /// The Markov lift of `kernel` at this grid's step, the tables a device launch binds; rebuilt
  /// whenever the kernel, the step count or the horizon changes.
  pub(crate) lift: VolterraLift<T, K>,
}

impl<T: FloatExt, K, S: SeedExt> VolterraSde<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// A solver for the given kernel, coefficients and grid; panics unless `n >= 2`, a given `x0`
  /// is finite and a given `t` is positive and finite.
  #[must_use]
  pub fn new(
    kernel: K,
    drift: impl Into<Fn2D<T>>,
    diffusion: impl Into<Fn2D<T>>,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_x0(x0);
    check_t(t);
    check_n(n);
    let lift = VolterraLift::new(
      kernel.clone(),
      t.unwrap_or(T::one()) / T::from_usize_(n - 1),
    );
    Self {
      backend: Cpu,
      kernel,
      drift: drift.into(),
      diffusion: diffusion.into(),
      n,
      x0,
      t,
      seed,
      lift,
    }
  }
}

impl<T: FloatExt, K, S: SeedExt, B> VolterraSde<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `kernel` and rebuild its Markov lift at this grid's step.
  pub fn with_kernel(mut self, kernel: K) -> Self {
    self.kernel = kernel;
    self.lift = VolterraLift::new(self.kernel.clone(), self.step_size());
    self
  }

  /// The grid's step, `t / (n − 1)`.
  pub fn step_size(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }

  /// The time the step starting at each grid point sees, `(i − 1) Δt` at point `i`: the launch's
  /// first curve, where the coefficients are evaluated, as the host does at the left point.
  fn step_time_curve(&self) -> Vec<T> {
    let dt = self.step_size();
    (0..self.n)
      .map(|i| T::from_usize_(i.saturating_sub(1)) * dt)
      .collect()
  }

  /// Replace `drift`, all else unchanged.
  pub fn with_drift(mut self, drift: impl Into<Fn2D<T>>) -> Self {
    self.drift = drift.into();
    self
  }

  /// Replace `diffusion`, all else unchanged.
  pub fn with_diffusion(mut self, diffusion: impl Into<Fn2D<T>>) -> Self {
    self.diffusion = diffusion.into();
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.lift = VolterraLift::new(self.kernel.clone(), self.step_size());
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace the simulation horizon `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.lift = VolterraLift::new(self.kernel.clone(), self.step_size());
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

backend_switch!([T: FloatExt, K, S: SeedExt] VolterraSde<T, K, S> { kernel, drift, diffusion, n, x0, t, seed, lift } via euler where  K: VolterraKernel<T> + Send + Sync);

impl<T: FloatExt, K, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for VolterraSde<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// [`ProcessExt::sample`] keeps a closure drift or diffusion, which the family cannot carry,
  /// on the host, so a launch that asks here bypassed that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    assert!(
      self.device_ready(),
      "VolterraSde: a launch carries a drift and a diffusion written as expressions; sample \
       through `ProcessExt`, which keeps closures on the host"
    );
    crate::euler::EulerSpec::VolterraProgram
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

  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![self.step_time_curve()])
  }

  /// The kernel's Markov lift: the same per-node constants and boundary
  /// terms the host sampler steps with, and the start the lift adds back.
  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, T>> {
    let lift = &self.lift;
    Some(crate::euler::LiftSpec {
      decay: lift.exp_neg_x_dt.as_slice().expect("contiguous"),
      weight: lift.we.as_slice().expect("contiguous"),
      drift_scale: lift.one_minus_e_over_x.as_slice().expect("contiguous"),
      drift_boundary: lift.drift_boundary,
      diffusion_boundary: lift.diffusion_boundary,
      x0: self.x0.unwrap_or(T::zero()),
    })
  }

  /// The drift's and the diffusion's programs, read by the family as `pv`
  /// and `pv2`.
  fn program_spec(&self) -> Option<crate::euler::ProgramSpec<'_>> {
    Some(crate::euler::ProgramSpec {
      first: self.drift.program()?,
      second: Some(self.diffusion.program()?),
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

impl<T: FloatExt, K, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for VolterraSde<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
}

impl<T: FloatExt, K, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for VolterraSde<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = VolterraSdeSampler<T, K, S>
  where
    Self: 's;

  /// Builds a [`VolterraLift`] from `self.kernel`, an $O(N')$ precompute, and a [`Gn`] whose seed
  /// is derived, not cloned, from `self.seed`, as [`ProcessExt`] requires of implementors.
  fn sampler(&self) -> VolterraSdeSampler<T, K, S> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    VolterraSdeSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      drift: self.drift.clone(),
      diffusion: self.diffusion.clone(),
      lift: VolterraLift::new(self.kernel.clone(), dt),
      gn: Gn::<T, S> {
        backend: Cpu,
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
    }
  }

  /// Through the Euler engine when the drift and the diffusion are expressions; a closure stays
  /// on the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs: a drift and a diffusion written as [`Expr`](crate::traits::Expr)s and a
  /// kernel fit of at most [`LIFT_SLOTS`](crate::euler::LIFT_SLOTS) nodes; closures stay on host.
  fn device_fallback(&self) -> Option<&'static str> {
    let ready = {
      self.drift.program().is_some()
        && self.diffusion.program().is_some()
        && self.kernel.degree() <= crate::euler::LIFT_SLOTS
    };
    (!ready).then_some(
      "a coefficient that is a closure rather than an Expr, or a kernel degree past the lift",
    )
  }
}

/// Reusable [`VolterraSde`] sampling state: a [`VolterraLift`] built once per chunk and the
/// Gaussian source. It clones `drift` and `diffusion` to stay lifetime-free for `Volterra`.
#[doc(hidden)]
#[derive(Debug)]
pub struct VolterraSdeSampler<T: FloatExt, K, S: SeedExt>
where
  K: VolterraKernel<T> + Send + Sync,
{
  n: usize,
  x0: T,
  drift: Fn2D<T>,
  diffusion: Fn2D<T>,
  lift: VolterraLift<T, K>,
  gn: Gn<T, S>,
}

impl<T: FloatExt, K, S: SeedExt> VolterraSdeSampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let dw = self.gn.sample();
    let drift = &self.drift;
    let diffusion = &self.diffusion;
    let path = self.lift.simulate(
      self.x0,
      |t, x| drift.call(t, x),
      |t, x| diffusion.call(t, x),
      dw.as_slice().expect("dw must be contiguous"),
    );
    out.copy_from_slice(path.as_slice().expect("lift path must be contiguous"));
  }
}

impl<T: FloatExt, K, S: SeedExt> crate::traits::Sealed for VolterraSdeSampler<T, K, S> where
  K: VolterraKernel<T> + Send + Sync
{
}

impl<T: FloatExt, K, S: SeedExt> PathSampler<T> for VolterraSdeSampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("VolterraSde output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::VolterraSde;
  use crate::traits::ProcessExt;
  use crate::volterra::kernel::ExponentialKernel;

  fn zero_2d(_t: f64, _x: f64) -> f64 {
    0.0
  }

  fn one_2d(_t: f64, _x: f64) -> f64 {
    1.0
  }

  fn mean_reverting_drift(_t: f64, x: f64) -> f64 {
    0.3 * (0.5 - x)
  }

  fn const_diffusion(_t: f64, _x: f64) -> f64 {
    0.2
  }

  #[test]
  #[should_panic(expected = "n must satisfy `n >= 2`, got n = 1")]
  fn rejects_too_short_grid() {
    let kernel = ExponentialKernel::new(0.5_f64, 1.0_f64);
    let _ = VolterraSde::new(
      kernel,
      zero_2d as fn(f64, f64) -> f64,
      one_2d as fn(f64, f64) -> f64,
      1,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
  }

  #[test]
  fn starts_at_x0_and_is_finite() {
    let kernel = ExponentialKernel::new(0.7_f64, 1.0_f64);
    let sde = VolterraSde::new(
      kernel,
      mean_reverting_drift as fn(f64, f64) -> f64,
      const_diffusion as fn(f64, f64) -> f64,
      64,
      Some(0.1),
      Some(1.0),
      Deterministic::new(7),
    );
    let path = sde.sample();
    assert_eq!(path.len(), 64);
    assert_eq!(path[0], 0.1);
    assert!(path.iter().all(|v| v.is_finite()));
  }

  #[test]
  fn same_seed_reproduces_bit_for_bit() {
    let build = || {
      VolterraSde::new(
        ExponentialKernel::new(0.7_f64, 1.0_f64),
        mean_reverting_drift as fn(f64, f64) -> f64,
        const_diffusion as fn(f64, f64) -> f64,
        32,
        Some(0.0),
        Some(1.0),
        Deterministic::new(99),
      )
    };
    assert_eq!(build().sample(), build().sample());
  }
}
