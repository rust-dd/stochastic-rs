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
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_distributions::special::ln_gamma;

use crate::traits::FloatExt;
use crate::volterra::VolterraKernel;

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
  /// `weights` normalised by $\Gamma(H+1/2)$: $w_l/\Gamma(H+1/2)$. Backs
  /// the [`VolterraKernel`] impl below, so that its `weights()` and
  /// `evaluate()` describe the same kernel $K(t) = t^{H-1/2}/\Gamma(H+1/2)$
  /// used by the SDE in the [`rough`](crate::rough) module docs.
  ///
  /// **Do not divide by `gamma_h_half` again when consuming this field (or
  /// the trait's `weights()`/`evaluate()`/`integral_from_zero()`) — the
  /// $1/\Gamma(H+1/2)$ factor is already folded in here.** Doing so would
  /// not panic; it would silently produce a kernel too small by that
  /// factor. `weights` itself stays un-normalised for a *different*,
  /// non-trait call path:
  /// [`MarkovLift`](crate::rough::markov_lift::MarkovLift) reads the
  /// **inherent** (un-normalised) `weights`/`evaluate` directly and
  /// applies $1/\Gamma(H+1/2)$ itself, once, outside the per-mode sum, as
  /// an optimisation. A kernel-generic stepper driven through
  /// [`VolterraKernel`] instead must not carry that split over — see the
  /// invariant stated on [`VolterraKernel::weights`].
  pub normalized_weights: Array1<T>,
}

impl<T: FloatExt> RlKernel<T> {
  /// Quadrature degrees at or below this are confirmed numerically stable.
  /// Above it, the underlying `gen_laguerre_nodes_weights` measurably
  /// starts producing non-finite weights — independently confirmed finite
  /// through 175, non-finite somewhere in 176–189, for every Hurst tested
  /// — and it does so silently, with no panic: `weights`/
  /// `normalized_weights` just turn `NaN`, and every Markov-lift path
  /// built from them follows. [`RlKernel::new`] enforces this as a hard
  /// ceiling rather than handing back a NaN-poisoned kernel.
  pub const MAX_STABLE_DEGREE: usize = 175;

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
  /// - if `degree` exceeds [`Self::MAX_STABLE_DEGREE`]
  #[must_use]
  pub fn new(hurst: T, degree: usize) -> Self {
    let h_f64 = hurst.to_f64().expect("Hurst must be convertible to f64");
    assert!(
      h_f64 > 0.0 && h_f64 < 0.5,
      "RL kernel requires Hurst in (0, 1/2), got {h_f64}"
    );
    assert!(degree > 0, "quadrature degree must be positive");
    let max_degree = Self::MAX_STABLE_DEGREE;
    assert!(
      degree <= max_degree,
      "quadrature degree must be <= {max_degree} (see RlKernel::MAX_STABLE_DEGREE's docs — the underlying quadrature measurably produces non-finite weights above this), got {degree}"
    );

    let alpha = -(h_f64 + 0.5);
    let (nodes_f64, weights_f64) = gen_laguerre_nodes_weights(degree, alpha);

    let inv_gamma_half_minus_h = 1.0 / gamma(0.5 - h_f64);
    let inv_gamma_h_half = 1.0 / gamma(h_f64 + 0.5);
    let mut nodes = Array1::<T>::zeros(degree);
    let mut weights = Array1::<T>::zeros(degree);
    let mut normalized_weights = Array1::<T>::zeros(degree);
    for i in 0..degree {
      nodes[i] = T::from_f64_fast(nodes_f64[i]);
      let w = weights_f64[i] * nodes_f64[i].exp() * inv_gamma_half_minus_h;
      weights[i] = T::from_f64_fast(w);
      normalized_weights[i] = T::from_f64_fast(w * inv_gamma_h_half);
    }

    Self {
      hurst,
      nodes,
      weights,
      gamma_h_half: T::from_f64_fast(gamma(h_f64 + 0.5)),
      normalized_weights,
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

impl<T: FloatExt> VolterraKernel<T> for RlKernel<T> {
  fn nodes(&self) -> &Array1<T> {
    &self.nodes
  }

  // Deliberately not `&self.weights`: this trait method's contract is the
  // *normalised* kernel (see `normalized_weights`'s field doc), so returning
  // the un-normalised `weights` field here would be the actual bug.
  #[allow(clippy::misnamed_getters)]
  fn weights(&self) -> &Array1<T> {
    &self.normalized_weights
  }

  /// Exact closed form $K(t) = t^{H-1/2}/\Gamma(H+1/2)$ — the kernel that
  /// the `rough` module's SDE actually uses (see the
  /// [module docs](crate::rough)) — computed directly, **not** through the
  /// exponential sum. The inherent [`RlKernel::evaluate`] is itself only an
  /// approximation of $t^{H-1/2}$ (see its own docs), so routing the exact
  /// reference value through it would make `evaluate` inherit the fit's
  /// approximation error instead of supplying the ground truth that
  /// [`nodes`](VolterraKernel::nodes)/[`weights`](VolterraKernel::weights)
  /// are fitted against.
  fn evaluate(&self, t: T) -> T {
    t.powf(self.hurst - T::from_f64_fast(0.5)) / self.gamma_h_half
  }

  /// $\int_0^{dt} K(u)\,du = \delta t^{H+1/2}/\Gamma(H+3/2)$, the closed
  /// form obtained by integrating $u^{H-1/2}/\Gamma(H+1/2)$ term-by-term —
  /// this is the quantity [`MarkovLift`](crate::rough::markov_lift::MarkovLift)
  /// hard-codes as `dt_pow_h_plus_half / gamma_h_plus_three_half`.
  fn integral_from_zero(&self, dt: T) -> T {
    let h_f64 = self
      .hurst
      .to_f64()
      .expect("Hurst must be convertible to f64");
    dt.powf(self.hurst + T::from_f64_fast(0.5)) / T::from_f64_fast(gamma(h_f64 + 1.5))
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
  use super::gamma;
  use super::gen_laguerre_nodes_weights;
  use crate::volterra::VolterraKernel;

  /// `VolterraKernel::evaluate` must be the *exact* closed-form kernel
  /// $t^{H-1/2}/\Gamma(H+1/2)$, computed independently of both the exp-sum
  /// fit and of `RlKernel`'s own cached `gamma_h_half` — it is the ground
  /// truth the fit is judged against, not a rescaling of the (approximate)
  /// inherent `evaluate`.
  #[test]
  fn volterra_kernel_evaluate_matches_independent_closed_form() {
    let hurst = 0.3_f64;
    let k = RlKernel::<f64>::new(hurst, 40);
    for &t in &[0.05_f64, 0.5, 2.0] {
      let expected = t.powf(hurst - 0.5) / gamma(hurst + 0.5);
      let via_trait = VolterraKernel::evaluate(&k, t);
      let rel = (via_trait - expected).abs() / expected.abs();
      assert!(
        rel < 1e-12,
        "t={t}: via_trait={via_trait} expected={expected} rel={rel}"
      );
    }
  }

  /// Mirrors `volterra::kernel::tests::integral_from_zero_matches_numerical_quadrature`
  /// for the RL kernel specifically: `integral_from_zero` must equal the
  /// midpoint-rule quadrature of the *normalised* `VolterraKernel::evaluate`.
  /// Getting the normalisation wrong (e.g. forgetting to divide by
  /// `gamma_h_half`) would miss this by a factor of `Γ(H+1/2)` — comfortably
  /// outside the 1e-4 tolerance for every Hurst below.
  ///
  /// Uses a finer grid than the brief's own `dt=0.01, n=200_000` recipe:
  /// at `hurst=0.1` the integrand `u^{-0.4}` is singular enough that
  /// `n=200_000` midpoints alone carry ~1.4e-4 discretisation error even
  /// against the *exact* power law (independently confirmed with no
  /// `RlKernel` involved), which would swamp the 1e-4 tolerance before this
  /// test ever got to check normalisation. `n=2_000_000` brings that down
  /// to ~3.6e-5, leaving headroom to actually test what this test is for.
  #[test]
  fn volterra_kernel_integral_from_zero_matches_quadrature() {
    let dt = 0.01_f64;
    let n = 2_000_000;
    let h = dt / n as f64;
    for hurst in [0.1_f64, 0.3, 0.45] {
      let k = RlKernel::<f64>::new(hurst, 40);
      let mut acc = 0.0;
      for i in 0..n {
        acc += VolterraKernel::evaluate(&k, (i as f64 + 0.5) * h);
      }
      acc *= h;
      let closed = VolterraKernel::integral_from_zero(&k, dt);
      let rel = (acc - closed).abs() / closed.abs().max(1e-300);
      assert!(
        rel < 1e-4,
        "hurst={hurst}: quadrature={acc} closed={closed} rel={rel}"
      );
    }
  }

  /// `VolterraKernel::weights`/`nodes` must reproduce the *exact*
  /// `VolterraKernel::evaluate` via the exponential sum, within the same
  /// 5e-3 tolerance the brief's own kernel tests use. `degree=150` is
  /// chosen empirically: `degree=40` (the existing `exp_sum_approximates_power_law`
  /// test's degree, tuned for `t >= 0.2`) misses by ~5x at `t=1e-2`, and
  /// `degree=200` hits a pre-existing instability in the underlying
  /// Laguerre quadrature (`normalized_weights` turns non-finite) — a
  /// latent issue in `gen_laguerre_nodes_weights` at high degree, outside
  /// this trait's scope, so `degree=150` stays comfortably clear of both
  /// edges.
  #[test]
  fn volterra_kernel_exponential_sum_matches_evaluate() {
    let k = RlKernel::<f64>::new(0.3, 150);
    for &t in &[1e-2, 0.1, 0.5, 1.0] {
      let nodes = VolterraKernel::nodes(&k);
      let weights = VolterraKernel::weights(&k);
      let approx: f64 = (0..VolterraKernel::degree(&k))
        .map(|l| weights[l] * (-nodes[l] * t).exp())
        .sum();
      let truth = VolterraKernel::evaluate(&k, t);
      let rel = (approx - truth).abs() / truth.abs();
      assert!(rel < 5e-3, "t={t}: approx={approx} truth={truth} rel={rel}");
    }
  }

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

  /// The ceiling exists because degrees above it measure as producing
  /// non-finite weights (see `MAX_STABLE_DEGREE`'s docs) with no panic of
  /// their own — this locks in that a caller gets a hard error instead.
  #[test]
  #[should_panic(expected = "quadrature degree must be <=")]
  fn rejects_degree_above_stability_ceiling() {
    let _ = RlKernel::<f64>::new(0.3, RlKernel::<f64>::MAX_STABLE_DEGREE + 1);
  }

  #[test]
  fn degree_default_scales_with_log_n() {
    assert_eq!(RlKernel::<f64>::default_degree(1000), 26);
    assert_eq!(RlKernel::<f64>::default_degree(10_000), 29);
  }
}
