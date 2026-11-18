//! $$
//! K(t) \approx \sum_{l=1}^{N'} w_l\, e^{-x_l t},
//! $$
//!
//! [`VolterraKernel`]: an exponential-sum representation plus the two closed forms a lift needs
//! for the current step, which has no history yet, $K(\delta t)$ and $\int_0^{\delta t} K(u)\,du$.
use ndarray::Array1;
use stochastic_rs_distributions::special::gamma_p;

use crate::precondition::assert_entries;
use crate::rough::kernel::RlKernel;
use crate::traits::FloatExt;

/// Exponential-sum representation of a Volterra kernel, $K(t) \approx \sum_l w_l\, e^{-x_l t}$,
/// fitted or exact, with the exact kernel value and its integral from the origin.
pub trait VolterraKernel<T: FloatExt>: Clone {
  /// Quadrature nodes $x_l$.
  fn nodes(&self) -> &Array1<T>;
  /// Normalised weights: $\sum_l w_l e^{-x_l t} \approx K(t)$ for `evaluate`'s $K$, with nothing
  /// to renormalise; `RlKernel::scaled_weights` and `exp_sum` are an un-normalised pair outside it.
  fn weights(&self) -> &Array1<T>;
  /// Number of exponential factors $N'$.
  fn degree(&self) -> usize {
    self.nodes().len()
  }

  /// Exact kernel value $K(t)$, $t > 0$. Used by the reference path, by the
  /// diffusion boundary weight, and by the fit diagnostics.
  fn evaluate(&self, t: T) -> T;
  /// Drift boundary weight $\int_0^{\delta t} K(u)\,du$, in closed form where available; it
  /// generalises [`MarkovLift`](crate::rough::MarkovLift)'s $\delta t^{H+1/2} / \Gamma(H+3/2)$.
  fn integral_from_zero(&self, dt: T) -> T;
}

/// Pure exponential kernel $K(t) = c\,e^{-\beta t}$: one exponential, so the $N'=1$ sum
/// reproduces `evaluate` with no fitting error.
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
  /// $K(t) = c\,e^{-\beta t}$; panics unless both are finite and $\beta > 0$.
  #[must_use]
  pub fn new(beta: T, c: T) -> Self {
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    assert!(
      c.is_finite(),
      "c must satisfy `c.is_finite()`, got c = {c:?}"
    );
    assert!(
      beta > T::zero(),
      "beta must satisfy `beta > T::zero()`, got beta = {beta:?}"
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

/// Tempered fractional kernel $K(t) = t^{H-1/2}\,e^{-\lambda t}/\Gamma(H+1/2)$: [`RlKernel`]'s sum
/// with every node shifted by $\lambda$, which is exact, so tempering adds no fitting error.
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
  /// The kernel for Hurst $H$ tempered at rate $\lambda$, on [`RlKernel::new`]'s nodes; panics
  /// unless $\lambda > 0$ is finite, and where [`RlKernel::new`] panics.
  #[must_use]
  pub fn new(hurst: T, lambda: T, degree: usize) -> Self {
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      lambda > T::zero(),
      "lambda must satisfy `lambda > T::zero()`, got lambda = {lambda:?}"
    );
    let rl = RlKernel::<T>::new(hurst, degree);
    let mut nodes = Array1::<T>::zeros(degree);
    let mut weights = Array1::<T>::zeros(degree);
    for l in 0..degree {
      nodes[l] = rl.nodes[l] + lambda;
      weights[l] = rl.scaled_weights[l] / rl.gamma_h_half;
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

  /// $\int_0^{dt} K(u)\,du = P(H{+}1/2,\ \lambda\,dt) / \lambda^{H+1/2}$ with $P = \gamma/\Gamma$
  /// the regularised lower incomplete gamma, by the substitution $v=\lambda u$.
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

/// Explicitly supplied exponential-sum kernel $K(t) = \sum_l w_l e^{-x_l t}$, for a fit made
/// outside this crate: `evaluate` and `integral_from_zero` are the sum itself, term by term.
#[derive(Debug, Clone)]
pub struct SumOfExponentials<T: FloatExt> {
  /// Quadrature nodes $x_l$.
  pub nodes: Array1<T>,
  /// Quadrature weights $w_l$.
  pub weights: Array1<T>,
}

impl<T: FloatExt> SumOfExponentials<T> {
  /// The trust boundary for calibrated nodes and weights: panics unless all are finite, the two are
  /// equally long and not empty, and every node is positive, so the kernel decays.
  #[must_use]
  pub fn new(nodes: Array1<T>, weights: Array1<T>) -> Self {
    assert_entries("nodes", "nodes[i].is_finite()", &nodes, |v| v.is_finite());
    assert_entries("weights", "weights[i].is_finite()", &weights, |v| {
      v.is_finite()
    });
    assert!(
      nodes.len() == weights.len(),
      "nodes.len() must satisfy `nodes.len() == weights.len()`, got nodes.len() = {:?}, weights.len() = {:?}",
      nodes.len(),
      weights.len()
    );
    assert!(
      !nodes.is_empty(),
      "nodes must satisfy `!nodes.is_empty()`, got nodes = []"
    );
    assert_entries("nodes", "nodes[i] > T::zero()", &nodes, |x| x > T::zero());
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
mod tests;
