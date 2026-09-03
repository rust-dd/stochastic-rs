//! # Gaussian polynomial volatility
//!
//! $$
//! X_t = \int_0^t K(t-s)\,dW_s, \qquad \sigma_t = p(X_t) = \sum_{k=0}^{d} c_k X_t^k
//! $$
//!
//! Volatility as a polynomial of a *Gaussian Volterra process* — a stochastic
//! convolution of a kernel against a Brownian motion. The family is the one
//! behind the strongest published results on the joint SPX/VIX calibration
//! problem, long considered the hardest fitting exercise in volatility
//! modelling:
//!
//! - **Abi Jaber, Illand & Li (2022)**, *Joint SPX-VIX calibration with
//!   Gaussian polynomial volatility models*, arXiv:2212.08297.
//! - **Abi Jaber, Illand & Li (2022)**, *The quintic Ornstein-Uhlenbeck
//!   volatility model that jointly calibrates SPX & VIX smiles*,
//!   arXiv:2212.10917 — the degree-five case over a single fast-mean-reverting
//!   OU process.
//!
//! ## Why this is cheap to have here
//!
//! Setting $b\equiv 0$ and $\sigma\equiv 1$ in
//! [`VolterraSde`](super::sve::VolterraSde) already produces the Gaussian
//! Volterra process $X$, and the lift makes it $O(n N')$ rather than
//! $O(n^2)$. Everything this type adds on top is a polynomial evaluated
//! pointwise, so the whole family costs one Horner loop over an existing
//! primitive.
//!
//! Choosing [`ExponentialKernel`](super::kernel::ExponentialKernel) makes $X$
//! an Ornstein–Uhlenbeck process **exactly** — the exponential kernel is
//! represented by a single mode with no approximation error — which is the
//! quintic model's own setting.
//!
//! ## Scope
//!
//! This type is the **volatility** process $\sigma_t$. The price leg
//! $dS_t = \sigma_t S_t\,dW^S_t$ with $d\langle W, W^S\rangle = \rho\,dt$ is
//! not included; a correlated two-dimensional output is a separate piece of
//! work, and stating that plainly is better than shipping half of it under a
//! name that implies the whole model.

use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::gn::Gn;
use crate::rough::markov_lift::RoughSimd;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::kernel::VolterraKernel;
use crate::volterra::lift::VolterraLift;

/// Volatility as a polynomial of a Gaussian Volterra process.
pub struct GaussianPolynomialVolatility<T: FloatExt, K, S: SeedExt = Unseeded, B = Cpu>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Kernel $K$ of the driving Gaussian Volterra process.
  pub kernel: K,
  /// Polynomial coefficients $c_0,\dots,c_d$ in ascending order, so
  /// `coefficients[k]` multiplies $X^k$.
  pub coefficients: Array1<T>,
  /// Number of points sampled along the path.
  pub n: usize,
  /// Simulation horizon $[0, t]$ (defaults to $1$ when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

impl<T: FloatExt, K, S: SeedExt> Clone for GaussianPolynomialVolatility<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
  S: Clone,
{
  /// Snapshot semantics, matching every other process in this crate.
  fn clone(&self) -> Self {
    Self {
      backend: PhantomData,
      kernel: self.kernel.clone(),
      coefficients: self.coefficients.clone(),
      n: self.n,
      t: self.t,
      seed: self.seed.clone(),
    }
  }
}

impl<T: FloatExt, K, S: SeedExt> GaussianPolynomialVolatility<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// # Panics
  /// - if `n < 2`
  /// - if `coefficients` is empty (a polynomial needs at least a constant term)
  #[must_use]
  pub fn new(kernel: K, coefficients: Array1<T>, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(
      !coefficients.is_empty(),
      "coefficients must contain at least a constant term"
    );
    Self {
      backend: PhantomData,
      kernel,
      coefficients,
      n,
      t,
      seed,
    }
  }

  /// The quintic parameterisation of arXiv:2212.10917.
  ///
  /// The model's polynomial is **sparse**, not a general quintic: the paper
  /// fixes the quadratic and quartic terms at zero, so
  ///
  /// $$ p(x) = \alpha_0 + \alpha_1 x + \alpha_3 x^3 + \alpha_5 x^5 . $$
  ///
  /// This constructor therefore takes those four coefficients and fills
  /// degrees 2 and 4 with zero, rather than accepting six free values —
  /// passing a dense six-coefficient polynomial would produce a strictly
  /// larger family than the one the citation names. (The paper's "six
  /// parameters" are $\{\rho, H, \alpha_0, \alpha_1, \alpha_3, \alpha_5\}$,
  /// counting the correlation and Hurst exponent, neither of which lives on
  /// this type — see the scope note in the module docs.)
  ///
  /// The calibrated instance in the paper's Figure 1 is
  /// $(\alpha_0, \alpha_1, \alpha_3, \alpha_5) = (0.5907, 1, 0.2893, 0.0549)$
  /// at $\rho = -0.6843$, $H = -0.0358$.
  ///
  /// Use [`new`](Self::new) directly for an unrestricted degree-five
  /// polynomial; it is a different model and this crate does not claim the
  /// paper's results for it.
  ///
  /// # Panics
  /// - under the same conditions as [`new`](Self::new)
  #[must_use]
  pub fn quintic(
    kernel: K,
    alpha0: T,
    alpha1: T,
    alpha3: T,
    alpha5: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    let coefficients = Array1::from_vec(vec![alpha0, alpha1, T::zero(), alpha3, T::zero(), alpha5]);
    Self::new(kernel, coefficients, n, t, seed)
  }
}

impl<T: FloatExt, K, S: SeedExt, B> GaussianPolynomialVolatility<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  /// Replace the polynomial, all else unchanged.
  ///
  /// # Panics
  /// - if `coefficients` is empty
  #[must_use]
  pub fn with_coefficients(mut self, coefficients: Array1<T>) -> Self {
    assert!(
      !coefficients.is_empty(),
      "coefficients must contain at least a constant term"
    );
    self.coefficients = coefficients;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  ///
  /// # Panics
  /// - if `n < 2`
  #[must_use]
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
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

  /// Evaluate $p(x)$ by Horner's rule.
  ///
  /// Horner rather than a naive power sum because the quintic case raises $x$
  /// to the fifth, and a fast-mean-reverting driver with large vol-of-vol —
  /// the regime arXiv:2212.10917 calibrates in — makes $|x|$ large enough for
  /// the cancellation to matter.
  #[must_use]
  pub fn evaluate_polynomial(&self, x: T) -> T {
    let mut acc = T::zero();
    for c in self.coefficients.iter().rev() {
      acc = acc * x + *c;
    }
    acc
  }
}

backend_switch!([T: FloatExt + RoughSimd, K, S: SeedExt] GaussianPolynomialVolatility<T, K, S> { kernel, coefficients, n, t, seed } via host where  K: VolterraKernel<T> + Send + Sync);

impl<T: FloatExt + RoughSimd, K, S: SeedExt, B: HostBackend> ProcessExt<T>
  for GaussianPolynomialVolatility<T, K, S, B>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = GaussianPolynomialVolatilitySampler<T, K, S>
  where
    Self: 's;

  fn sampler(&self) -> GaussianPolynomialVolatilitySampler<T, K, S> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    GaussianPolynomialVolatilitySampler {
      n: self.n,
      coefficients: self.coefficients.clone(),
      lift: VolterraLift::new(self.kernel.clone(), dt),
      gn: Gn::<T, S> {
        backend: PhantomData,
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
    }
  }
}

/// Reusable [`GaussianPolynomialVolatility`] sampling state.
#[doc(hidden)]
pub struct GaussianPolynomialVolatilitySampler<T: FloatExt + RoughSimd, K, S: SeedExt>
where
  K: VolterraKernel<T> + Send + Sync,
{
  n: usize,
  coefficients: Array1<T>,
  lift: VolterraLift<T, K>,
  gn: Gn<T, S>,
}

impl<T: FloatExt + RoughSimd, K, S: SeedExt> GaussianPolynomialVolatilitySampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let dw = self.gn.sample();
    let path = self.lift.simulate(
      T::zero(),
      |_, _| T::zero(),
      |_, _| T::one(),
      dw.as_slice().expect("dw must be contiguous"),
    );
    for (o, x) in out
      .iter_mut()
      .zip(path.as_slice().expect("lift path must be contiguous"))
    {
      let mut acc = T::zero();
      for c in self.coefficients.iter().rev() {
        acc = acc * *x + *c;
      }
      *o = acc;
    }
  }
}

impl<T: FloatExt + RoughSimd, K, S: SeedExt> PathSampler<T>
  for GaussianPolynomialVolatilitySampler<T, K, S>
where
  K: VolterraKernel<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("GaussianPolynomialVolatility output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
#[path = "gaussian_polynomial_tests.rs"]
mod tests;
