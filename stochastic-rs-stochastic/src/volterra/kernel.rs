//! # Exponential-sum kernels for stochastic Volterra equations
//!
//! [`VolterraKernel`] is the shared interface for a kernel $K$ that admits
//! (exactly, or by a fitted approximation) an exponential-sum representation
//!
//! $$
//! K(t) \approx \sum_{l=1}^{N'} w_l\, e^{-x_l t},
//! $$
//!
//! plus the two closed-form pieces a Markov-lift stepper needs to advance
//! one time step $\delta t$: the kernel value $K(\delta t)$ itself
//! ([`VolterraKernel::evaluate`]) and the drift boundary weight
//! $\int_0^{\delta t} K(u)\,du$ ([`VolterraKernel::integral_from_zero`]).
//! Neither is recoverable from the exponential-sum history alone, since the
//! increment over the *current* step has no elapsed history to draw on yet.
//!
//! Three implementors:
//! - [`ExponentialKernel`] — a single exponential, exact at $N'=1$.
//! - [`GammaKernel`] — the exponentially damped (tempered) fractional
//!   kernel, built by shifting [`crate::rough::kernel::RlKernel`]'s nodes.
//! - [`SumOfExponentials`] — an explicitly supplied fit: the escape hatch
//!   for kernels calibrated outside this crate.
use ndarray::Array1;
use stochastic_rs_distributions::special::gamma_p;

use crate::rough::kernel::RlKernel;
use crate::traits::FloatExt;

/// Exponential-sum representation of a Volterra kernel:
/// $K(t) \approx \sum_l w_l\, e^{-x_l t}$.
///
/// Implementors supply nodes and weights for that sum — fitted, or exact
/// when the kernel itself is a finite sum of exponentials — plus the exact
/// kernel value and its integral from the origin.
pub trait VolterraKernel<T: FloatExt>: Clone {
  /// Quadrature nodes $x_l$.
  fn nodes(&self) -> &Array1<T>;
  /// Scaled weights $w_l$, already absorbing any normalising constant.
  ///
  /// **Invariant, true for every implementor**: $\sum_l w_l\, e^{-x_l t}
  /// \approx K(t)$ using *these* $w_l$ (`weights()[l]`) and $x_l$
  /// (`nodes()[l]`), where $K$ is exactly what
  /// [`evaluate`](Self::evaluate) returns; and
  /// [`integral_from_zero`](Self::integral_from_zero) is $\int_0^{dt}
  /// K(u)\,du$ for that same $K$. A kernel-generic caller **must** build
  /// per-mode products from these trait methods alone and **must not**
  /// apply any further normalising factor on top (e.g. the
  /// Riemann–Liouville kernel's $1/\Gamma(H+1/2)$) — it is already folded
  /// into `weights`, `evaluate`, and `integral_from_zero` here. The
  /// existing [`MarkovLift`](crate::rough::markov_lift::MarkovLift)
  /// deliberately does the opposite: it reads
  /// [`RlKernel`]'s *inherent*, un-normalised `weights`/`evaluate`,
  /// applying that factor itself,
  /// once, outside the sum. That split belongs to `MarkovLift`'s own
  /// hand-written loop and must not be copied into a kernel-generic
  /// stepper built on this trait.
  fn weights(&self) -> &Array1<T>;
  /// Number of exponential factors $N'$.
  fn degree(&self) -> usize {
    self.nodes().len()
  }
  /// Exact kernel value $K(t)$, $t > 0$. Used by the reference path, by the
  /// diffusion boundary weight, and by the fit diagnostics.
  fn evaluate(&self, t: T) -> T;
  /// Drift boundary weight $\int_0^{\delta t} K(u)\,du$, in closed form
  /// where available. This is what generalises
  /// [`MarkovLift`](crate::rough::markov_lift::MarkovLift)'s
  /// $\delta t^{H+1/2} / \Gamma(H+3/2)$.
  fn integral_from_zero(&self, dt: T) -> T;
}

/// Pure exponential kernel $K(t) = c\,e^{-\beta t}$.
///
/// Representable exactly by a single exponential, so [`VolterraKernel`]'s
/// $N'=1$ exponential sum reproduces `evaluate` with no fitting error.
#[derive(Debug, Clone)]
pub struct ExponentialKernel<T: FloatExt> {
  /// Decay rate $\beta > 0$.
  pub beta: T,
  /// Amplitude $c$.
  pub c: T,
  /// Single-node representation `[beta]`, materialised so
  /// [`VolterraKernel::nodes`] can return a borrow.
  nodes: Array1<T>,
  /// Single-weight representation `[c]`.
  weights: Array1<T>,
}

impl<T: FloatExt> ExponentialKernel<T> {
  /// Construct $K(t) = c\,e^{-\beta t}$.
  ///
  /// # Panics
  /// - if $\beta \le 0$
  #[must_use]
  pub fn new(beta: T, c: T) -> Self {
    assert!(
      beta > T::zero(),
      "ExponentialKernel requires beta > 0, got {beta:?}"
    );
    Self {
      beta,
      c,
      nodes: Array1::from_vec(vec![beta]),
      weights: Array1::from_vec(vec![c]),
    }
  }
}

impl<T: FloatExt> VolterraKernel<T> for ExponentialKernel<T> {
  fn nodes(&self) -> &Array1<T> {
    &self.nodes
  }

  fn weights(&self) -> &Array1<T> {
    &self.weights
  }

  fn evaluate(&self, t: T) -> T {
    self.c * (-self.beta * t).exp()
  }

  /// $\int_0^{dt} c\,e^{-\beta u}\,du = c\,(1-e^{-\beta\,dt})/\beta$.
  fn integral_from_zero(&self, dt: T) -> T {
    self.c * (T::one() - (-self.beta * dt).exp()) / self.beta
  }
}

/// Exponentially damped (tempered) fractional kernel
/// $K(t) = t^{H-1/2}\,e^{-\lambda t}/\Gamma(H+1/2)$.
///
/// Its exponential sum is [`RlKernel`]'s own sum for the un-damped
/// power-law kernel $t^{H-1/2}$, with every node shifted by $\lambda$:
/// multiplying $\sum_l w_l e^{-x_l t}$ by $e^{-\lambda t}$ gives
/// $\sum_l w_l e^{-(x_l+\lambda)t}$ exactly, so tempering introduces no
/// additional fitting error beyond the RL sum's own.
#[derive(Debug, Clone)]
pub struct GammaKernel<T: FloatExt> {
  /// Hurst exponent $H \in (0, 1/2)$.
  pub hurst: T,
  /// Tempering (damping) rate $\lambda > 0$.
  pub lambda: T,
  /// RL nodes $x_l$ shifted by $\lambda$.
  nodes: Array1<T>,
  /// RL weights normalised by $\Gamma(H+1/2)$.
  weights: Array1<T>,
  /// Cached $\Gamma(H+1/2)$, reused by [`GammaKernel::evaluate`].
  gamma_h_half: T,
}

impl<T: FloatExt> GammaKernel<T> {
  /// Construct the tempered kernel for Hurst $H$, tempering rate $\lambda$,
  /// using $N'$ Laguerre nodes (see [`RlKernel::new`]).
  ///
  /// `degree` is **not** defaulted or capped here: it funnels straight
  /// into [`RlKernel::new`]'s own quadrature, so
  /// [`RlKernel::MAX_STABLE_DEGREE`] applies to it transitively — a
  /// `degree` past that ceiling panics there, not silently.
  ///
  /// # Panics
  /// - if $H \notin (0, 1/2)$, `degree == 0`, or `degree` exceeds
  ///   [`RlKernel::MAX_STABLE_DEGREE`] (all raised inside the
  ///   [`RlKernel::new`] call below)
  /// - if $\lambda \le 0$ (checked ahead of that call)
  #[must_use]
  pub fn new(hurst: T, lambda: T, degree: usize) -> Self {
    assert!(
      lambda > T::zero(),
      "GammaKernel requires lambda > 0 (use RlKernel for the undamped case), got {lambda:?}"
    );
    let rl = RlKernel::<T>::new(hurst, degree);
    let mut nodes = Array1::<T>::zeros(degree);
    let mut weights = Array1::<T>::zeros(degree);
    for l in 0..degree {
      nodes[l] = rl.nodes[l] + lambda;
      weights[l] = rl.weights[l] / rl.gamma_h_half;
    }
    Self {
      hurst,
      lambda,
      nodes,
      weights,
      gamma_h_half: rl.gamma_h_half,
    }
  }
}

impl<T: FloatExt> VolterraKernel<T> for GammaKernel<T> {
  fn nodes(&self) -> &Array1<T> {
    &self.nodes
  }

  fn weights(&self) -> &Array1<T> {
    &self.weights
  }

  fn evaluate(&self, t: T) -> T {
    t.powf(self.hurst - T::from_f64_fast(0.5)) * (-self.lambda * t).exp() / self.gamma_h_half
  }

  /// $\int_0^{dt} K(u)\,du = P(H{+}1/2,\ \lambda\,dt) / \lambda^{H+1/2}$,
  /// via the regularised lower incomplete gamma $P$ — the substitution
  /// $v=\lambda u$ turns $\int_0^{dt} u^{H-1/2}e^{-\lambda u}du$ into
  /// $\lambda^{-(H+1/2)}\gamma(H{+}1/2,\lambda\,dt)$, and dividing by
  /// $\Gamma(H+1/2)$ gives $P = \gamma/\Gamma$.
  fn integral_from_zero(&self, dt: T) -> T {
    let a = self
      .hurst
      .to_f64()
      .expect("Hurst must be convertible to f64")
      + 0.5;
    let lambda_f64 = self
      .lambda
      .to_f64()
      .expect("lambda must be convertible to f64");
    let dt_f64 = dt.to_f64().expect("dt must be convertible to f64");
    T::from_f64_fast(gamma_p(a, lambda_f64 * dt_f64) / lambda_f64.powf(a))
  }
}

/// Explicitly supplied exponential-sum kernel $K(t) = \sum_l w_l e^{-x_l t}$.
///
/// The escape hatch for a kernel fitted outside this crate — e.g. by an
/// external calibration routine — with no other closed form to check
/// against: `evaluate` and `integral_from_zero` are computed directly from
/// `nodes`/`weights`, term by term, so this type introduces no fitting
/// error of its own beyond whatever the caller's `nodes`/`weights` already
/// carry.
#[derive(Debug, Clone)]
pub struct SumOfExponentials<T: FloatExt> {
  /// Quadrature nodes $x_l$.
  pub nodes: Array1<T>,
  /// Quadrature weights $w_l$.
  pub weights: Array1<T>,
}

impl<T: FloatExt> SumOfExponentials<T> {
  /// This is the trust boundary for externally calibrated `nodes`/`weights`
  /// — nothing here checks that the caller's fit is *good*, only that it is
  /// well-formed enough not to fail silently.
  ///
  /// # Panics
  /// - if `nodes` and `weights` have different lengths, or either is empty
  /// - if any node is not strictly positive (a zero node makes
  ///   `integral_from_zero`'s `(1-e^0)/x_l` divide `0.0/0.0` to `NaN` with
  ///   no panic; a negative node yields a growing, non-decaying "kernel"
  ///   with no diagnostic at all)
  #[must_use]
  pub fn new(nodes: Array1<T>, weights: Array1<T>) -> Self {
    assert_eq!(
      nodes.len(),
      weights.len(),
      "nodes and weights must have the same length"
    );
    assert!(
      !nodes.is_empty(),
      "SumOfExponentials requires at least one term"
    );
    assert!(
      nodes.iter().all(|&x| x > T::zero()),
      "SumOfExponentials requires all nodes to be strictly positive"
    );
    Self { nodes, weights }
  }
}

impl<T: FloatExt> VolterraKernel<T> for SumOfExponentials<T> {
  fn nodes(&self) -> &Array1<T> {
    &self.nodes
  }

  fn weights(&self) -> &Array1<T> {
    &self.weights
  }

  fn evaluate(&self, t: T) -> T {
    self
      .nodes
      .iter()
      .zip(self.weights.iter())
      .map(|(&x, &w)| w * (-x * t).exp())
      .sum()
  }

  /// $\int_0^{dt} \sum_l w_l e^{-x_l u}\,du = \sum_l w_l(1-e^{-x_l\,dt})/x_l$,
  /// term by term.
  fn integral_from_zero(&self, dt: T) -> T {
    self
      .nodes
      .iter()
      .zip(self.weights.iter())
      .map(|(&x, &w)| w * (T::one() - (-x * dt).exp()) / x)
      .sum()
  }
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
