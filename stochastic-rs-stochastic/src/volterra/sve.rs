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
pub struct VolterraSde<T: FloatExt, K, S: SeedExt = Unseeded>
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
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

impl<T: FloatExt, K, S: SeedExt> Clone for VolterraSde<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
  S: Clone,
{
  fn clone(&self) -> Self {
    Self {
      kernel: self.kernel.clone(),
      drift: self.drift.clone(),
      diffusion: self.diffusion.clone(),
      n: self.n,
      x0: self.x0,
      t: self.t,
      seed: self.seed.clone(),
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
    Self {
      kernel,
      drift: drift.into(),
      diffusion: diffusion.into(),
      n,
      x0,
      t,
      seed,
    }
  }

  /// Replace `kernel`, all else unchanged.
  pub fn with_kernel(mut self, kernel: K) -> Self {
    self.kernel = kernel;
    self
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
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt + RoughSimd, K, S: SeedExt> ProcessExt<T> for VolterraSde<T, K, S>
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
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
    }
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
