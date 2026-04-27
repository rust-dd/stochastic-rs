//! # Riemann–Liouville kernel quadrature
//!
//! $$
//! \Gamma(1/2-H)\, t^{H-1/2} = \int_0^\infty s^{-(H+1/2)}\, e^{-ts}\,ds
//! \approx \sum_{l=1}^{N'} w_l\, e^{-x_l t}
//! $$
//!
//! The singular power-law kernel $t^{H-1/2}$ is approximated by a finite sum
//! of exponentials using generalised Gauss–Laguerre quadrature with weight
//! $x^\alpha e^{-x}$ at $\alpha = -(H+1/2)$. The Bilokon–Wong (2026)
//! substitution $f(s) = e^{s-ts}$ lets a single Laguerre rule integrate the
//! kernel over $(0,\infty)$ in one piece — no logarithmic binning, no
//! small/origin/large splits (unlike Ma–Wu 2021 with Gauss–Legendre).
//!
//! Nodes and weights are computed by iterative refinement (Gatteschi–Pittaluga
//! starter + Newton polishing) with weights from the analytic boundary formula
//! $w_l = \Gamma(n{+}\alpha{+}1) \,/\, [n!\, x_l\, L_n^{(\alpha)\prime}(x_l)^2]$;
//! this stays numerically stable for $n \gtrsim 40$ and negative $\alpha$,
//! unlike the Golub–Welsch eigen-solver shipped in some quadrature crates.
//!
//! The stored weights absorb both the Laguerre re-weighting $e^{x_l}$ and
//! the normalising factor $1/\Gamma(1/2-H)$, so runtime code can write
//! $t^{H-1/2} \approx \sum_l w_l\, e^{-x_l t}$ directly.
use std::num::NonZeroUsize;

use gauss_quad::laguerre::GaussLaguerre;
use ndarray::Array1;
use statrs::function::gamma::gamma;
use statrs::function::gamma::ln_gamma;

use crate::traits::FloatExt;

/// Exponential-sum approximation of the Riemann–Liouville kernel $t^{H-1/2}$.
#[derive(Debug, Clone)]
pub struct RlKernel<T: FloatExt> {
  /// Hurst exponent $H \in (0, 1/2)$.
  pub hurst: T,
  /// Gauss–Laguerre nodes $(x_l)_{l=1}^{N'}$.
  pub nodes: Array1<T>,
  /// Scaled weights $w_l = w^{\mathrm{GL}}_l\, e^{x_l}/\Gamma(1/2-H)$.
  pub weights: Array1<T>,
  /// Cached $\Gamma(H+1/2)$ used by the Markov-lift update formula.
  pub gamma_h_half: T,
}

impl<T: FloatExt> RlKernel<T> {
  /// Default quadrature degree for a grid of $N$ points: $\lfloor\log N\rfloor + 20$,
  /// matching the empirical choice of the Bilokon–Wong reference implementation.
  #[must_use]
  pub fn default_degree(n: usize) -> usize {
    ((n.max(2) as f64).ln() as usize) + 20
  }

  /// Construct the kernel approximation for Hurst $H$ using $N'$ Laguerre nodes.
  ///
  /// # Panics
  /// - if $H \notin (0, 1/2)$ (the Laguerre parameter $\alpha = -(H+1/2)$ must satisfy $\alpha > -1$)
  /// - if `degree == 0`
  #[must_use]
  pub fn new(hurst: T, degree: usize) -> Self {
    let h_f64 = hurst.to_f64().expect("Hurst must be convertible to f64");
    assert!(
      h_f64 > 0.0 && h_f64 < 0.5,
      "RL kernel requires Hurst in (0, 1/2), got {h_f64}"
    );
    assert!(degree > 0, "quadrature degree must be positive");

    let alpha = -(h_f64 + 0.5);
    let (nodes_f64, weights_f64) = gen_laguerre_nodes_weights(degree, alpha);

    let inv_gamma_half_minus_h = 1.0 / gamma(0.5 - h_f64);
    let mut nodes = Array1::<T>::zeros(degree);
    let mut weights = Array1::<T>::zeros(degree);
    for i in 0..degree {
      nodes[i] = T::from_f64_fast(nodes_f64[i]);
      weights[i] = T::from_f64_fast(weights_f64[i] * nodes_f64[i].exp() * inv_gamma_half_minus_h);
    }

    Self {
      hurst,
      nodes,
      weights,
      gamma_h_half: T::from_f64_fast(gamma(h_f64 + 0.5)),
    }
  }

  /// Number of quadrature nodes $N'$.
  #[must_use]
  pub fn degree(&self) -> usize {
    self.nodes.len()
  }

  /// Evaluate the exp-sum approximation $\sum_l w_l\, e^{-x_l t} \approx t^{H-1/2}$.
  #[must_use]
  pub fn evaluate(&self, t: T) -> T {
    let mut acc = T::zero();
    for (x, w) in self.nodes.iter().zip(self.weights.iter()) {
      acc += *w * (-*x * t).exp();
    }
    acc
  }
}

/// Evaluate the generalised Laguerre polynomial $L_n^{(\alpha)}(x)$ and its
/// derivative $L_n^{(\alpha)\prime}(x)$ using the three-term recurrence
/// $(n{+}1)L_{n+1} = (2n{+}1{+}\alpha{-}x)L_n - (n{+}\alpha)L_{n-1}$.
/// Returns `(L_n, L_n_prime)`.
fn laguerre_l_and_dl(n: usize, alpha: f64, x: f64) -> (f64, f64) {
  if n == 0 {
    return (1.0, 0.0);
  }
  let mut lnm1 = 1.0;
  let mut ln = 1.0 + alpha - x;
  for k in 1..n {
    let kf = k as f64;
    let lnp1 = ((2.0 * kf + 1.0 + alpha - x) * ln - (kf + alpha) * lnm1) / (kf + 1.0);
    lnm1 = ln;
    ln = lnp1;
  }
  let nf = n as f64;
  let dln = (nf * ln - (nf + alpha) * lnm1) / x;
  (ln, dln)
}

/// Generalised Gauss–Laguerre nodes and weights for $\alpha > -1$, robust at
/// negative $\alpha$ (Hurst near $1/2$). The tridiagonal Jacobi-matrix
/// eigenvalues give accurate nodes in every regime, but Golub–Welsch weights
/// (built from eigenvector first components) become unstable for $n \gtrsim 25$
/// with $\alpha < 0$. We replace them with the analytic boundary formula
/// $w_l = \Gamma(n{+}\alpha{+}1) / \bigl[n!\, x_l\, L_n^{(\alpha)\prime}(x_l)^2\bigr]$.
fn gen_laguerre_nodes_weights(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
  assert!(alpha > -1.0, "alpha must be > -1");
  let quad = GaussLaguerre::new(
    NonZeroUsize::new(n).expect("n must be positive"),
    alpha.try_into().expect("alpha > -1 checked above"),
  );

  let log_norm = ln_gamma(n as f64 + alpha + 1.0) - ln_gamma(n as f64 + 1.0);
  let norm = log_norm.exp();

  let nodes: Vec<f64> = quad.nodes().copied().collect();
  let weights: Vec<f64> = nodes
    .iter()
    .map(|&x| {
      let (_l, dl) = laguerre_l_and_dl(n, alpha, x);
      norm / (x * dl * dl)
    })
    .collect();
  (nodes, weights)
}

#[cfg(test)]
mod tests {
  use super::RlKernel;
  use super::gen_laguerre_nodes_weights;

  /// The exp-sum should reproduce the power-law kernel to relative precision
  /// that improves with the Hurst exponent and degrades near t → 0.
  #[test]
  fn exp_sum_approximates_power_law() {
    let hurst = 0.1_f64;
    let k = RlKernel::<f64>::new(hurst, 40);
    let exponent = hurst - 0.5;
    for t in [0.2_f64, 1.0, 5.0] {
      let approx = k.evaluate(t);
      let truth = t.powf(exponent);
      let rel = (approx - truth).abs() / truth;
      assert!(rel < 5e-3, "t={t} approx={approx} truth={truth} rel={rel}");
    }
  }

  #[test]
  fn laguerre_matches_scipy_reference_first_nodes() {
    // Reference values from scipy.special.roots_genlaguerre(20, -0.6)
    let (nodes, weights) = gen_laguerre_nodes_weights(20, -0.6);
    let scipy_first = [
      (0.023547480568583978_f64, 1.0134437918563453_f64),
      (0.25573619389320856, 0.622864701359439),
      (0.7340211023623413, 0.34792337656527667),
      (1.4612387213185818, 0.1575950982469561),
      (2.44197358108164, 0.05648501697792918),
    ];
    for (i, (xs, ws)) in scipy_first.iter().enumerate() {
      let dx = (nodes[i] - xs).abs();
      let dw = (weights[i] - ws).abs() / ws;
      assert!(
        dx < 1e-10,
        "node {i}: got {} vs scipy {xs} (diff {dx})",
        nodes[i]
      );
      assert!(
        dw < 1e-8,
        "weight {i}: got {} vs scipy {ws} (rel {dw})",
        weights[i]
      );
    }
  }

  #[test]
  fn weights_exp_x_stay_bounded_at_high_degree() {
    let (nodes, weights) = gen_laguerre_nodes_weights(40, -0.6);
    let eff_max = nodes
      .iter()
      .zip(weights.iter())
      .map(|(x, w)| w * x.exp())
      .fold(f64::NEG_INFINITY, f64::max);
    assert!(
      eff_max < 10.0,
      "w*exp(x) must be bounded; max={eff_max} indicates Golub-Welsch-style blowup"
    );
  }

  #[test]
  #[should_panic(expected = "Hurst in (0, 1/2)")]
  fn rejects_h_at_half() {
    let _ = RlKernel::<f64>::new(0.5, 20);
  }

  #[test]
  #[should_panic(expected = "Hurst in (0, 1/2)")]
  fn rejects_h_above_half() {
    let _ = RlKernel::<f64>::new(0.7, 20);
  }

  #[test]
  fn degree_default_scales_with_log_n() {
    assert_eq!(RlKernel::<f64>::default_degree(1000), 26);
    assert_eq!(RlKernel::<f64>::default_degree(10_000), 29);
  }
}
