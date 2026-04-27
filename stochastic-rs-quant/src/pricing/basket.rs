//! # Basket
//!
//! Basket option payoff: $\max(\phi(B - K), 0)$ where
//! $B = \sum_i w_i S_{i,T}$ is a weighted average of $n$ assets.
//!
//! - **Geometric** (closed-form, Kemna–Vorst style): the weighted geometric
//!   mean of correlated log-normals is itself log-normal; price via BSM with
//!   adjusted volatility and forward.
//! - **Arithmetic via Levy (1992) moment matching**: match the first two
//!   moments of $B$ to a log-normal and use BSM. Standard market practice
//!   for pricing index options.
//! - **Monte Carlo**: general benchmark, supports arbitrary correlation.
//!
//! Source:
//! - Levy, E. (1992), "Pricing European average rate currency options", J. Int. Money & Finance 11
//! - Turnbull, S. & Wakeman, L. (1991), "A quick algorithm for pricing European average options"
//! - Hu, D., Sayit, H. & Viens, F. (2023), "Pricing basket options with the first three moments
//!   of the basket: log-normal models and beyond", arXiv:2302.08041
//!
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
#[cfg(feature = "openblas")]
use ndarray_linalg::Cholesky;
#[cfg(feature = "openblas")]
use ndarray_linalg::UPLO;
#[cfg(feature = "openblas")]
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::OptionType;
#[cfg(feature = "openblas")]
use crate::traits::FloatExt;

/// Geometric basket call/put on $n$ correlated Gbm assets. Uses the fact
/// that the weighted geometric mean is log-normal under Gbm.
///
/// $$
/// G_T = \prod_{i=1}^n S_{i,T}^{w_i},\qquad
/// \ln G_T \sim \mathcal N\!\big(\mu_G T, \sigma_G^2 T\big)
/// $$
///
/// where the weights satisfy $\sum_i w_i = 1$.
#[derive(Debug, Clone)]
pub struct GeometricBasketPricer {
  /// Spot prices $S_{i,0}$.
  pub s: Array1<f64>,
  /// Basket weights $w_i$ (must sum to one).
  pub weights: Array1<f64>,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Dividend yields.
  pub q: Array1<f64>,
  /// Correlation matrix $\rho_{ij}$ ($n \times n$, symmetric, ones on
  /// diagonal).
  pub rho: Array2<f64>,
  /// Strike.
  pub k: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Time to maturity.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl GeometricBasketPricer {
  pub fn price(&self) -> f64 {
    let n_assets = self.s.len();
    assert_eq!(self.weights.len(), n_assets);
    assert_eq!(self.sigma.len(), n_assets);
    assert_eq!(self.q.len(), n_assets);
    assert_eq!(self.rho.shape(), [n_assets, n_assets]);

    // Geometric basket vol: sigma_G^2 = sum_i sum_j w_i w_j rho_{ij} sigma_i sigma_j
    let mut sigma_g_sq = 0.0;
    for i in 0..n_assets {
      for j in 0..n_assets {
        sigma_g_sq +=
          self.weights[i] * self.weights[j] * self.rho[[i, j]] * self.sigma[i] * self.sigma[j];
      }
    }
    let sigma_g = sigma_g_sq.max(0.0).sqrt();

    // Drift of log G under risk neutral (with continuous dividends per asset)
    // mu_G = sum_i w_i [r - q_i - 0.5 sigma_i^2] + 0.5 sigma_G^2
    let mut mu_g = 0.5 * sigma_g_sq;
    for i in 0..n_assets {
      mu_g += self.weights[i] * (self.r - self.q[i] - 0.5 * self.sigma[i] * self.sigma[i]);
    }

    // log of geometric forward
    let mut log_g0 = 0.0;
    for i in 0..n_assets {
      log_g0 += self.weights[i] * self.s[i].ln();
    }
    let g_fwd = (log_g0 + mu_g * self.t).exp();
    let disc = (-self.r * self.t).exp();

    let n = Normal::new(0.0, 1.0).unwrap();
    let sqrt_t = self.t.sqrt();
    let d1 = ((g_fwd / self.k).ln() + 0.5 * sigma_g_sq * self.t) / (sigma_g * sqrt_t);
    let d2 = d1 - sigma_g * sqrt_t;
    match self.option_type {
      OptionType::Call => disc * (g_fwd * n.cdf(d1) - self.k * n.cdf(d2)),
      OptionType::Put => disc * (self.k * n.cdf(-d2) - g_fwd * n.cdf(-d1)),
    }
  }
}

/// Arithmetic basket option priced via Levy (1992) two-moment matching.
/// The arithmetic basket is approximated by a log-normal whose first two
/// moments match those of $B = \sum_i w_i S_{i,T}$ under the risk-neutral
/// measure.
///
/// $$
/// E[B] = \sum_i w_i S_{i,0} e^{(r-q_i)T},\qquad
/// E[B^2] = \sum_{i,j} w_i w_j S_i S_j e^{((r-q_i) + (r-q_j) + \rho_{ij}\sigma_i\sigma_j)T}
/// $$
#[derive(Debug, Clone)]
pub struct ArithmeticBasketLevyPricer {
  /// Spot prices.
  pub s: Array1<f64>,
  /// Weights (need not sum to one).
  pub weights: Array1<f64>,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Dividend yields.
  pub q: Array1<f64>,
  /// Correlation matrix.
  pub rho: Array2<f64>,
  /// Strike.
  pub k: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Time to maturity.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl ArithmeticBasketLevyPricer {
  /// Price using Levy (1992) two-moment lognormal approximation.
  pub fn price(&self) -> f64 {
    let m1 = first_moment(
      self.s.view(),
      self.weights.view(),
      self.q.view(),
      self.r,
      self.t,
    );
    let m2 = second_moment(
      self.s.view(),
      self.weights.view(),
      self.sigma.view(),
      self.q.view(),
      self.rho.view(),
      self.r,
      self.t,
    );
    let var = (m2 / (m1 * m1)).ln().max(1e-14);
    let sigma_eff = (var / self.t).sqrt();
    let n = Normal::new(0.0, 1.0).unwrap();
    let sqrt_t = self.t.sqrt();
    let d1 = ((m1 / self.k).ln() + 0.5 * var) / (sigma_eff * sqrt_t);
    let d2 = d1 - sigma_eff * sqrt_t;
    let disc = (-self.r * self.t).exp();
    match self.option_type {
      OptionType::Call => disc * (m1 * n.cdf(d1) - self.k * n.cdf(d2)),
      OptionType::Put => disc * (self.k * n.cdf(-d2) - m1 * n.cdf(-d1)),
    }
  }
}

fn first_moment(s: ArrayView1<f64>, w: ArrayView1<f64>, q: ArrayView1<f64>, r: f64, t: f64) -> f64 {
  let mut m = 0.0;
  for i in 0..s.len() {
    m += w[i] * s[i] * ((r - q[i]) * t).exp();
  }
  m
}

fn second_moment(
  s: ArrayView1<f64>,
  w: ArrayView1<f64>,
  sigma: ArrayView1<f64>,
  q: ArrayView1<f64>,
  rho: ArrayView2<f64>,
  r: f64,
  t: f64,
) -> f64 {
  let n = s.len();
  let mut m = 0.0;
  for i in 0..n {
    for j in 0..n {
      let exponent = ((r - q[i]) + (r - q[j]) + rho[[i, j]] * sigma[i] * sigma[j]) * t;
      m += w[i] * w[j] * s[i] * s[j] * exponent.exp();
    }
  }
  m
}

/// Monte Carlo basket option pricer. Supports arithmetic and geometric
/// payoffs. Uses `ndarray_linalg::Cholesky` for the correlation factor and
/// is therefore gated behind the `openblas` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasketAverageType {
  Arithmetic,
  Geometric,
}

#[cfg(feature = "openblas")]
#[derive(Debug, Clone)]
pub struct McBasketPricer {
  /// Spot prices.
  pub s: Array1<f64>,
  /// Weights.
  pub weights: Array1<f64>,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Dividend yields.
  pub q: Array1<f64>,
  /// Correlation matrix.
  pub rho: Array2<f64>,
  /// Strike.
  pub k: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Time to maturity.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
  /// Average type.
  pub avg_type: BasketAverageType,
  /// Number of MC paths.
  pub n_paths: usize,
}

#[cfg(feature = "openblas")]
impl McBasketPricer {
  pub fn price(&self) -> f64 {
    let n_assets = self.s.len();
    let l: Array2<f64> = self
      .rho
      .cholesky(UPLO::Lower)
      .expect("correlation matrix must be positive definite");
    let drifts: Vec<f64> = (0..n_assets)
      .map(|i| (self.r - self.q[i] - 0.5 * self.sigma[i] * self.sigma[i]) * self.t)
      .collect();
    let vols: Vec<f64> = (0..n_assets)
      .map(|i| self.sigma[i] * self.t.sqrt())
      .collect();
    let phi = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    let n_paths = self.n_paths;

    // Generate one big block of standard normals using the project's
    // SIMD ziggurat path, then map paths in parallel.
    let mut all_z = vec![0.0_f64; n_paths * n_assets];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let sum: f64 = (0..n_paths)
      .into_par_iter()
      .map(|p| {
        let z = &all_z[p * n_assets..(p + 1) * n_assets];
        let mut zc = vec![0.0_f64; n_assets];
        for i in 0..n_assets {
          let mut acc = 0.0;
          for j in 0..=i {
            acc += l[[i, j]] * z[j];
          }
          zc[i] = acc;
        }
        let s_t: Vec<f64> = (0..n_assets)
          .map(|i| self.s[i] * (drifts[i] + vols[i] * zc[i]).exp())
          .collect();
        let basket = match self.avg_type {
          BasketAverageType::Arithmetic => {
            (0..n_assets).map(|i| self.weights[i] * s_t[i]).sum::<f64>()
          }
          BasketAverageType::Geometric => {
            let mut log_g = 0.0;
            for i in 0..n_assets {
              log_g += self.weights[i] * s_t[i].ln();
            }
            log_g.exp()
          }
        };
        (phi * (basket - self.k)).max(0.0)
      })
      .sum();

    (-self.r * self.t).exp() * sum / n_paths as f64
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn iid_basket(
    n: usize,
    sigma: f64,
    rho: f64,
  ) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array2<f64>,
  ) {
    let s = Array1::from_elem(n, 100.0);
    let w = Array1::from_elem(n, 1.0 / n as f64);
    let sig = Array1::from_elem(n, sigma);
    let q = Array1::from_elem(n, 0.0);
    let mut rho_m = Array2::<f64>::from_elem((n, n), rho);
    for i in 0..n {
      rho_m[[i, i]] = 1.0;
    }
    (s, w, sig, q, rho_m)
  }

  /// Geometric basket on a single asset must equal a vanilla BSM call.
  #[test]
  fn geometric_basket_n1_matches_bsm() {
    let p = GeometricBasketPricer {
      s: array![100.0],
      weights: array![1.0],
      sigma: array![0.2],
      q: array![0.0],
      rho: array![[1.0]],
      k: 100.0,
      r: 0.05,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let price = p.price();
    let bsm = 10.4506;
    assert!((price - bsm).abs() < 0.005, "geo n=1: {price}");
  }

  /// Geometric basket with perfectly correlated identical assets equals
  /// a single-asset vanilla.
  #[test]
  fn geometric_basket_perfect_corr_equals_single() {
    let (s, w, sig, q, rho) = iid_basket(5, 0.20, 1.0);
    let p = GeometricBasketPricer {
      s,
      weights: w,
      sigma: sig,
      q,
      rho,
      k: 100.0,
      r: 0.05,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let price = p.price();
    let bsm = 10.4506;
    assert!((price - bsm).abs() < 0.01, "geo perf-corr: {price}");
  }

  /// Geometric basket should be cheaper than arithmetic (Jensen).
  #[test]
  fn geometric_below_arithmetic() {
    let (s, w, sig, q, rho) = iid_basket(4, 0.30, 0.5);
    let geo = GeometricBasketPricer {
      s: s.clone(),
      weights: w.clone(),
      sigma: sig.clone(),
      q: q.clone(),
      rho: rho.clone(),
      k: 100.0,
      r: 0.04,
      t: 1.0,
      option_type: OptionType::Call,
    }
    .price();
    let ari = ArithmeticBasketLevyPricer {
      s,
      weights: w,
      sigma: sig,
      q,
      rho,
      k: 100.0,
      r: 0.04,
      t: 1.0,
      option_type: OptionType::Call,
    }
    .price();
    assert!(geo < ari, "geo={geo} should be < ari={ari}");
  }

  /// Levy and MC should agree within ~3% for a 4-asset arithmetic basket.
  #[cfg(feature = "openblas")]
  #[test]
  fn levy_vs_mc_arithmetic() {
    let (s, w, sig, q, rho) = iid_basket(4, 0.25, 0.4);
    let levy = ArithmeticBasketLevyPricer {
      s: s.clone(),
      weights: w.clone(),
      sigma: sig.clone(),
      q: q.clone(),
      rho: rho.clone(),
      k: 100.0,
      r: 0.05,
      t: 1.0,
      option_type: OptionType::Call,
    }
    .price();
    let mc = McBasketPricer {
      s,
      weights: w,
      sigma: sig,
      q,
      rho,
      k: 100.0,
      r: 0.05,
      t: 1.0,
      option_type: OptionType::Call,
      avg_type: BasketAverageType::Arithmetic,
      n_paths: 100_000,
    }
    .price();
    let rel = (levy - mc).abs() / mc;
    assert!(rel < 0.03, "levy={levy}, mc={mc}, rel={rel}");
  }

  /// MC geometric vs analytical geometric basket (should match closely).
  #[cfg(feature = "openblas")]
  #[test]
  fn mc_geometric_matches_closed_form() {
    let (s, w, sig, q, rho) = iid_basket(3, 0.25, 0.5);
    let cf = GeometricBasketPricer {
      s: s.clone(),
      weights: w.clone(),
      sigma: sig.clone(),
      q: q.clone(),
      rho: rho.clone(),
      k: 100.0,
      r: 0.05,
      t: 1.0,
      option_type: OptionType::Call,
    }
    .price();
    let mc = McBasketPricer {
      s,
      weights: w,
      sigma: sig,
      q,
      rho,
      k: 100.0,
      r: 0.05,
      t: 1.0,
      option_type: OptionType::Call,
      avg_type: BasketAverageType::Geometric,
      n_paths: 200_000,
    }
    .price();
    let rel = (cf - mc).abs() / cf;
    assert!(rel < 0.02, "cf={cf}, mc={mc}");
  }

  /// Arithmetic basket put-call parity: $C - P = e^{-rT}(F - K)$ where
  /// $F = E[B]$.
  #[test]
  fn arithmetic_basket_parity() {
    let (s, w, sig, q, rho) = iid_basket(3, 0.25, 0.3);
    let r = 0.04;
    let t = 1.0;
    let k = 95.0;
    let c = ArithmeticBasketLevyPricer {
      s: s.clone(),
      weights: w.clone(),
      sigma: sig.clone(),
      q: q.clone(),
      rho: rho.clone(),
      k,
      r,
      t,
      option_type: OptionType::Call,
    }
    .price();
    let p = ArithmeticBasketLevyPricer {
      s: s.clone(),
      weights: w.clone(),
      sigma: sig.clone(),
      q: q.clone(),
      rho: rho.clone(),
      k,
      r,
      t,
      option_type: OptionType::Put,
    }
    .price();
    let f = first_moment(s.view(), w.view(), q.view(), r, t);
    let lhs = c - p;
    let rhs = (-r * t).exp() * (f - k);
    assert!((lhs - rhs).abs() < 0.01, "lhs={lhs}, rhs={rhs}");
  }
}
