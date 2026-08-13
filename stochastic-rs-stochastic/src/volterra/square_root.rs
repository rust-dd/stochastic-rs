//! # Volterra square-root process
//!
//! $$
//! V_t = V_0 + \int_0^t K(t-s)\,\kappa(\theta - V_s)\,ds
//!           + \int_0^t K(t-s)\,\nu\sqrt{V_s}\,dW_s
//! $$
//!
//! The variance leg of the Volterra Heston model (Abi Jaber & El Euch,
//! *Markovian structure of the Volterra Heston model*, arXiv:1803.00477), and
//! the rough Heston model when $K$ is the Riemann–Liouville kernel.
//!
//! ## Why this type exists rather than a plain [`VolterraSde`](super::sve::VolterraSde)
//!
//! Two reasons. First, `VolterraSde`'s coefficients are
//! [`Fn2D`](crate::traits::Fn2D)`::Native`, a bare `fn` pointer, so it cannot
//! carry $\kappa$, $\theta$ and $\nu$ — a closure over the parameters is
//! required. Second, and more importantly, $\sqrt{V}$ is not Lipschitz at the
//! origin and a naive discretisation of the lifted system drives $V$ negative
//! whenever the Feller condition is violated, at which point the diffusion
//! coefficient is undefined.
//!
//! ## The scheme, and exactly what it guarantees
//!
//! This uses **full truncation** — Lord, Koekkoek & van Dijk (2010),
//! *A comparison of biased simulation schemes for stochastic volatility
//! models*, Quantitative Finance 10(2) — evaluating both coefficients at
//! $V^+ = \max(V, 0)$, plus a floor on the reported path.
//!
//! What that buys: the output is nonnegative by construction, always, at any
//! parameters. What it costs: the scheme is **biased**. Full truncation is the
//! least-biased member of the fix-it-by-truncating family, not an exact-law
//! method, and this type does not claim otherwise.
//!
//! ## The exact-law alternatives, and why they are not here
//!
//! Two routes in the literature preserve the law rather than merely the sign,
//! and both are worth having:
//!
//! - **Abi Jaber, Bayer & Breneis (2024)**, *State spaces of multifactor
//!   approximations of nonnegative Volterra processes*, arXiv:2412.17526 —
//!   the state space of the multifactor approximation is an explicit linear
//!   transformation of the nonnegative orthant, so the invariant region can be
//!   enforced exactly instead of by clipping.
//! - **Abi Jaber (2024)**, arXiv:2412.11264 — simulate the **integrated**
//!   square-root process first; nonnegativity then comes for free, and the
//!   Inverse-Gaussian limit is exact in two regimes with a single time step.
//!   That exactness result is for the **classical** square-root process.
//! - **Abi Jaber & Attal (2025)**, *iVi*, arXiv:2504.19885 — the Volterra
//!   generalisation of the same idea, which proves **weak convergence** with
//!   few time steps rather than exactness. The distinction matters: citing the
//!   two together would credit iVi with a stronger result than it claims.
//!
//! Both hinge on explicit constructions from those papers, and implementing
//! either from a paraphrase would produce something that resembles the method
//! without being it. They are deferred deliberately, not overlooked.
//!
//! ## Hitting zero is not necessarily a bug
//!
//! Friesen, Gerhold & Wiedermann (2026), *Boundary behaviour of the Volterra
//! square-root process*, arXiv:2606.07290, give a **time-dependent** Feller
//! condition, and show that for rough kernels the boundary genuinely **can**
//! be attained. A path touching zero is therefore a property of the model at
//! those parameters, not evidence of a broken scheme.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::noise::gn::Gn;
use crate::rough::markov_lift::RoughSimd;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::kernel::VolterraKernel;
use crate::volterra::lift::VolterraLift;

/// Volterra square-root process with a nonnegativity-preserving scheme.
pub struct VolterraSquareRoot<T: FloatExt, K, S: SeedExt = Unseeded>
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
  /// Initial variance $V_0 \ge 0$ (defaults to $\theta$ when omitted, the
  /// stationary choice, rather than to zero — starting a square-root process
  /// exactly at its absorbing-looking boundary is almost never what a caller
  /// means).
  pub v0: Option<T>,
  /// Simulation horizon $[0, t]$ (defaults to $1$ when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

impl<T: FloatExt, K, S: SeedExt> Clone for VolterraSquareRoot<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
  S: Clone,
{
  /// Snapshot semantics, matching every other process in this crate: the
  /// clone resumes the same stream rather than advancing it, so bumping one
  /// parameter on a clone isolates that parameter under common random numbers.
  fn clone(&self) -> Self {
    Self {
      kernel: self.kernel.clone(),
      kappa: self.kappa,
      theta: self.theta,
      nu: self.nu,
      n: self.n,
      v0: self.v0,
      t: self.t,
      seed: self.seed.clone(),
    }
  }
}

impl<T: FloatExt, K, S: SeedExt> VolterraSquareRoot<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// # Panics
  /// - if `n < 2`
  /// - if `kappa`, `nu` are not strictly positive, or `theta` is negative
  /// - if `v0` is given and negative
  #[must_use]
  pub fn new(
    kernel: K,
    kappa: T,
    theta: T,
    nu: T,
    n: usize,
    v0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(kappa > T::zero(), "kappa must be strictly positive");
    assert!(nu > T::zero(), "nu must be strictly positive");
    assert!(theta >= T::zero(), "theta must be non-negative");
    if let Some(v) = v0 {
      assert!(v >= T::zero(), "v0 must be non-negative");
    }
    Self {
      kernel,
      kappa,
      theta,
      nu,
      n,
      v0,
      t,
      seed,
    }
  }

  /// Replace `kappa`, all else unchanged.
  #[must_use]
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert!(kappa > T::zero(), "kappa must be strictly positive");
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  #[must_use]
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(theta >= T::zero(), "theta must be non-negative");
    self.theta = theta;
    self
  }

  /// Replace `nu`, all else unchanged.
  #[must_use]
  pub fn with_nu(mut self, nu: T) -> Self {
    assert!(nu > T::zero(), "nu must be strictly positive");
    self.nu = nu;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  #[must_use]
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self
  }

  /// Replace the initial variance $V_0$, all else unchanged.
  #[must_use]
  pub fn with_v0(mut self, v0: T) -> Self {
    assert!(v0 >= T::zero(), "v0 must be non-negative");
    self.v0 = Some(v0);
    self
  }

  /// Replace the horizon, all else unchanged.
  #[must_use]
  pub fn with_horizon(mut self, t: T) -> Self {
    self.t = Some(t);
    self
  }

  /// Replace the seed strategy, all else unchanged.
  #[must_use]
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt + RoughSimd, K, S: SeedExt> ProcessExt<T> for VolterraSquareRoot<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = VolterraSquareRootSampler<T, K, S>
  where
    Self: 's;

  fn sampler(&self) -> VolterraSquareRootSampler<T, K, S> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    VolterraSquareRootSampler {
      n: self.n,
      v0: self.v0.unwrap_or(self.theta),
      kappa: self.kappa,
      theta: self.theta,
      nu: self.nu,
      lift: VolterraLift::new(self.kernel.clone(), dt),
      gn: Gn::<T, S> {
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
    }
  }
}

/// Reusable [`VolterraSquareRoot`] sampling state.
#[doc(hidden)]
pub struct VolterraSquareRootSampler<T: FloatExt + RoughSimd, K, S: SeedExt>
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

impl<T: FloatExt + RoughSimd, K, S: SeedExt> VolterraSquareRootSampler<T, K, S>
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

/// $x^+ = \max(x, 0)$, written as an explicit comparison rather than
/// `T::max`.
///
/// The difference matters: `f64::max` returns the non-NaN operand, so a NaN
/// state would be silently rewritten to `0` and the failure that produced it
/// would vanish. Here a NaN falls through unchanged and surfaces in the
/// output, where a caller can see it. This crate has twice shipped bugs that
/// `f64::max` hid in exactly this way.
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

impl<T: FloatExt + RoughSimd, K, S: SeedExt> PathSampler<T> for VolterraSquareRootSampler<T, K, S>
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
#[path = "square_root_tests.rs"]
mod tests;
