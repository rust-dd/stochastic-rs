//! # General stochastic Volterra equation
//!
//! $$
//! X_t = X_0 + \int_0^t K(t-s)\,b(s,X_s)\,ds + \int_0^t K(t-s)\,\sigma(s,X_s)\,dW_s
//! $$
//!
//! [`VolterraSde`] is [`VolterraLift`] promoted to a first-class
//! [`ProcessExt`]: any [`VolterraKernel`] paired with time-and-state
//! dependent drift/diffusion coefficients, solved at $O(nN')$ by the
//! Markov-lift stepper instead of the $O(n^2)$ direct convolution
//! ([`reference_path`](super::reference::reference_path) is that direct
//! discretisation, kept as this engine's cross-implementation oracle).
//!
//! **On convergence, no more than the literature supports.** This stepper's
//! explicit, non-anticipating drift/diffusion evaluation puts it in the
//! $\theta$-Euler–Maruyama class Li, Huang & Hu (arXiv:2004.04916, 2020)
//! analyse for weakly singular kernels ($K(t)=t^{H-1/2}/\Gamma(H+1/2)$ or
//! similar): for that class the strong rate is $\min\{1-\alpha,\,
//! \tfrac12-\beta\}$, not the usual $\tfrac12$, and for the Milstein scheme
//! (which this crate does not implement here) $n^{-2H}$ is provably optimal
//! (Liu, Hu & Gao, arXiv:2412.11126, 2024). Neither result is re-derived or
//! independently verified for *this* exact implementation (no rate-sweep
//! test lives in this file); they are cited as the relevant literature, not
//! claimed as a measured property of this code. A second, independent error
//! source — the kernel's own exponential-sum fit ($N'$ nodes approximating
//! $K$, see [`VolterraKernel::weights`]) — is outside what either rate
//! covers and is bounded separately per kernel (e.g. the 5e-3 relative
//! bound [`crate::volterra::kernel`]'s tests pin for
//! [`RlKernel`](crate::rough::kernel::RlKernel)/[`GammaKernel`](super::kernel::GammaKernel)).
//!
//! # References
//! - Abi Jaber E., El Euch O. *Multi-factor approximation of rough
//!   volatility models*, arXiv:1801.10359 (2018).
//! - Li M., Huang C., Hu Y. *Numerical methods for stochastic Volterra
//!   integral equations with weakly singular kernels*, arXiv:2004.04916
//!   (2020).

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::noise::gn::Gn;
use crate::rough::markov_lift::RoughSimd;
use crate::traits::FloatExt;
use crate::traits::Fn2D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::kernel::VolterraKernel;
use crate::volterra::lift::VolterraLift;

/// General stochastic Volterra equation, solved by Markovian lift at
/// $O(nN')$: $X_t = X_0 + \int_0^t K(t-s)\,b(s,X_s)\,ds + \int_0^t
/// K(t-s)\,\sigma(s,X_s)\,dW_s$.
///
/// Generic over any [`VolterraKernel`] implementor, so a single type covers
/// every kernel family this crate ships ([`ExponentialKernel`](super::kernel::ExponentialKernel),
/// [`GammaKernel`](super::kernel::GammaKernel), [`SumOfExponentials`](super::kernel::SumOfExponentials),
/// [`RlKernel`](crate::rough::kernel::RlKernel)) plus any externally fitted
/// one supplied through [`SumOfExponentials`](super::kernel::SumOfExponentials).
/// [`rough::MarkovLift`](crate::rough::markov_lift::MarkovLift) is the
/// historical, `RlKernel`-specialised, $(x)$-only-coefficient special case
/// of the same machinery.
///
/// No blanket `Default`: unlike a process whose fields all have an obvious
/// canonical value, `K` has none in general (an `ExponentialKernel` needs a
/// decay rate, an `RlKernel` needs a Hurst exponent and quadrature degree,
/// …) — the same reason [`CompoundPoisson`](crate::process::cpoisson::CompoundPoisson)`<T,
/// D, S>` has no `Default` for its own extra generic `D`.
pub struct VolterraSde<T: FloatExt, K, S: SeedExt = Unseeded, B = Cpu>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Kernel $K$ — exact where representable exactly (e.g. [`ExponentialKernel`](super::kernel::ExponentialKernel)),
  /// otherwise its $N'$-term exponential-sum fit.
  pub kernel: K,
  /// Drift coefficient $b(s, X_s)$.
  pub drift: Fn2D<T>,
  /// Diffusion coefficient $\sigma(s, X_s)$.
  pub diffusion: Fn2D<T>,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Initial value $X_0$ (defaults to $0$ when omitted).
  pub x0: Option<T>,
  /// Simulation horizon $[0, t]$ for the path (defaults to $1$ when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
  /// The Markov lift of `kernel` at this grid's step, the tables a device
  /// launch binds; rebuilt whenever the kernel, the step count or the
  /// horizon changes.
  pub(crate) lift: VolterraLift<T, K>,
}

impl<T: FloatExt, K, S: SeedExt> Clone for VolterraSde<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
  S: Clone,
{
  fn clone(&self) -> Self {
    Self {
      backend: Cpu,
      kernel: self.kernel.clone(),
      drift: self.drift.clone(),
      diffusion: self.diffusion.clone(),
      n: self.n,
      x0: self.x0,
      t: self.t,
      seed: self.seed.clone(),
      lift: VolterraLift::new(self.kernel.clone(), self.step_size()),
    }
  }
}

impl<T: FloatExt, K, S: SeedExt> VolterraSde<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Build a stochastic Volterra equation solver for the given kernel,
  /// coefficients, and grid.
  ///
  /// # Panics
  /// - if `n < 2`
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
    assert!(n >= 2, "n must be at least 2");
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
  /// Replace `kernel`, all else unchanged.
  pub fn with_kernel(mut self, kernel: K) -> Self {
    self.kernel = kernel;
    self.lift = VolterraLift::new(self.kernel.clone(), self.step_size());
    self
  }

  /// The grid's step, `t / (n − 1)`.
  pub fn step_size(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }

  /// The time the step starting at each grid point sees, `(i − 1) Δt` at
  /// point `i`: the launch's first curve, the `t` the coefficients are
  /// evaluated at, exactly as the host evaluates them at the left point.
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

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.lift = VolterraLift::new(self.kernel.clone(), self.step_size());
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
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

backend_switch!([T: FloatExt + RoughSimd, K, S: SeedExt] VolterraSde<T, K, S> { kernel, drift, diffusion, n, x0, t, seed, lift } via euler where  K: VolterraKernel<T> + Send + Sync);

impl<T: FloatExt + RoughSimd, K, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for VolterraSde<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// A configuration the family cannot carry — a closure for the drift or
  /// the diffusion — never reaches a launch: [`ProcessExt::sample`] keeps it
  /// on the host, so asking here is a caller bypassing that guard.
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

impl<T: FloatExt + RoughSimd, K, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for VolterraSde<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = VolterraSdeSampler<T, K, S>
  where
    Self: 's;

  /// Builds `dt` from `n`/`t` and a fresh [`VolterraLift`] from `self.kernel`
  /// (cheap — an $O(N')$ precompute over an already-fitted kernel, not the
  /// quadrature that built the kernel itself) plus a [`Gn`] source whose
  /// seed is *derived*, not cloned, from `self.seed` — see [`ProcessExt`]'s
  /// "Reproducibility requirement on implementors".
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

  /// Through the Euler engine when the drift and the diffusion are
  /// expressions; a closure keeps the process on the host, chunked exactly
  /// as [`ProcessExt`] chunks.
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

  /// What a device runs here: a drift and a diffusion written
  /// as [`Expr`](crate::traits::Expr)s, which the kernel interprets at every
  /// step, and a kernel whose exponential fit has at most
  /// [`LIFT_SLOTS`](crate::euler::LIFT_SLOTS) nodes. Closures keep the
  /// process on the host.
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

/// Reusable [`VolterraSde`] sampling state: owns a freshly built
/// [`VolterraLift`] (so a Monte-Carlo loop pays the $O(N')$ boundary-weight
/// precompute once per chunk, not once per path) and the Gaussian-increment
/// source. Cloning `drift`/`diffusion` here (rather than borrowing, as
/// [`HullWhiteSampler`](crate::interest::hull_white::HullWhiteSampler) does
/// for its own [`Fn1D`](crate::traits::Fn1D)) keeps this type lifetime-free,
/// which is what lets [`Volterra`](crate::process::volterra::Volterra)
/// embed it directly when delegating to this engine.
#[doc(hidden)]
pub struct VolterraSdeSampler<T: FloatExt + RoughSimd, K, S: SeedExt>
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

impl<T: FloatExt + RoughSimd, K, S: SeedExt> VolterraSdeSampler<T, K, S>
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

impl<T: FloatExt + RoughSimd, K, S: SeedExt> PathSampler<T> for VolterraSdeSampler<T, K, S>
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
  #[should_panic(expected = "n must be at least 2")]
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
