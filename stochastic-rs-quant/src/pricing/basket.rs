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
use stochastic_rs_distributions::special::norm_cdf;

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
///
/// The struct holds **model and contract state only** — the per-asset
/// volatilities and their correlation, plus the basket weights. The spot
/// vector, the strike, the rate, the dividend-yield vector and the maturity
/// are the pricing *query* and travel as arguments, so one instance prices a
/// whole strike/maturity grid.
///
/// The weights are the field where the split is not obvious, and they land
/// on the struct for two independent reasons. They are a **contract** term:
/// $\prod_i S_i^{w_i}$ is what the option is written on, fixed by the term
/// sheet and not by the market, exactly like the digitals' cash payout. They
/// are also inseparable from the model, because the effective volatility
/// $\sigma_G^2=\sum_{i,j} w_i w_j \rho_{ij}\sigma_i\sigma_j$ is a function of
/// the weights, the volatilities and the correlation and of nothing else — a
/// query-side weight vector would leave the basket with no volatility of its
/// own.
///
/// The dividend yields go the other way. A yield is a market quote, which
/// the crate already says in
/// [`ModelPricer`](crate::traits::ModelPricer)'s
/// `price_call(s, k, r, q, tau)`; a vector of them is the same quantity per
/// asset, so it travels with the query.
///
/// $\sigma_G$ could be cached at construction, since no query enters it, and
/// deliberately is not: the drift $\mu_G$ and the geometric forward both
/// carry the query's own rate, yields and spots, so caching one of the three
/// would put a struct field next to two that can never be one.
///
/// An $n$-asset payoff carries no
/// [`ModelPricer`](crate::traits::ModelPricer), whose
/// `price_call(s, k, r, q, tau)` has a single underlying — this is the
/// multi-asset "convention, no trait" family, see
/// [`KirkSpreadPricer`](crate::pricing::kirk::KirkSpreadPricer).
///
/// ```
/// use ndarray::array;
/// use stochastic_rs_quant::pricing::basket::GeometricBasketPricer;
///
/// let model = GeometricBasketPricer::new(
///   array![0.5, 0.5],
///   array![0.20, 0.30],
///   array![[1.0, 0.4], [0.4, 1.0]],
/// );
/// let s = array![100.0, 100.0];
/// let q = array![0.0, 0.0];
/// let itm = model.price_call(s.view(), 90.0, 0.05, q.view(), 1.0);
/// let otm = model.price_call(s.view(), 110.0, 0.05, q.view(), 1.0);
/// assert!(itm > otm);
/// ```
#[derive(Debug, Clone)]
pub struct GeometricBasketPricer {
  /// Basket weights $w_i$ (must sum to one) — a term of the contract, not a
  /// market quote.
  pub weights: Array1<f64>,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Correlation matrix $\rho_{ij}$ ($n \times n$, symmetric, ones on
  /// diagonal).
  pub rho: Array2<f64>,
}

impl GeometricBasketPricer {
  /// Builds the pricer from the basket weights, the per-asset volatilities
  /// and their correlation matrix.
  pub fn new(weights: Array1<f64>, sigma: Array1<f64>, rho: Array2<f64>) -> Self {
    Self {
      weights,
      sigma,
      rho,
    }
  }

  /// Price either leg at one query point.
  ///
  /// # Panics
  /// If the query's spot or yield vector disagrees with the length the
  /// weights, volatilities and correlation matrix fix between them.
  pub fn price_option(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    let n_assets = s.len();
    assert_eq!(self.weights.len(), n_assets);
    assert_eq!(self.sigma.len(), n_assets);
    assert_eq!(q.len(), n_assets);
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
      mu_g += self.weights[i] * (r - q[i] - 0.5 * self.sigma[i] * self.sigma[i]);
    }

    // log of geometric forward
    let mut log_g0 = 0.0;
    for i in 0..n_assets {
      log_g0 += self.weights[i] * s[i].ln();
    }
    let g_fwd = (log_g0 + mu_g * tau).exp();
    let disc = (-r * tau).exp();

    let sqrt_t = tau.sqrt();
    let d1 = ((g_fwd / k).ln() + 0.5 * sigma_g_sq * tau) / (sigma_g * sqrt_t);
    let d2 = d1 - sigma_g * sqrt_t;
    match option_type {
      OptionType::Call => disc * (g_fwd * norm_cdf(d1) - k * norm_cdf(d2)),
      OptionType::Put => disc * (k * norm_cdf(-d2) - g_fwd * norm_cdf(-d1)),
    }
  }

  /// Price the geometric basket call at one query point.
  pub fn price_call(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> f64 {
    self.price_option(s, k, r, q, tau, OptionType::Call)
  }

  /// Price the geometric basket put at one query point.
  pub fn price_put(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> f64 {
    self.price_option(s, k, r, q, tau, OptionType::Put)
  }
}

/// Arithmetic basket option priced via Levy (1992) two-moment matching.
/// The arithmetic basket is approximated by a log-normal whose first two
/// moments match those of $B = \sum_i w_i S_{i,T}$ under the risk-neutral
/// measure.
///
/// $$
/// E\left[B\right] = \sum_i w_i S_{i,0} e^{(r-q_i)T},\qquad
/// E\left[B^2\right] = \sum_{i,j} w_i w_j S_i S_j e^{((r-q_i) + (r-q_j) + \rho_{ij}\sigma_i\sigma_j)T}
/// $$
///
/// The struct holds **model and contract state only** — the per-asset
/// volatilities and their correlation, plus the basket weights. The spot
/// vector, the strike, the rate, the dividend-yield vector and the maturity
/// are the pricing *query* and travel as arguments, so one instance prices a
/// whole strike/maturity grid.
///
/// The weights split the same way as
/// [`GeometricBasketPricer`]'s and for the same reason. Here the case is if
/// anything plainer: both matched moments carry the query's own rate, yields
/// and spots, so no part of this approximation can be cached at construction
/// even in principle.
///
/// ```
/// use ndarray::array;
/// use stochastic_rs_quant::pricing::basket::ArithmeticBasketLevyPricer;
///
/// let model = ArithmeticBasketLevyPricer::new(
///   array![0.5, 0.5],
///   array![0.20, 0.30],
///   array![[1.0, 0.4], [0.4, 1.0]],
/// );
/// let s = array![100.0, 100.0];
/// let q = array![0.0, 0.0];
/// let itm = model.price_call(s.view(), 90.0, 0.05, q.view(), 1.0);
/// let otm = model.price_call(s.view(), 110.0, 0.05, q.view(), 1.0);
/// assert!(itm > otm);
/// ```
#[derive(Debug, Clone)]
pub struct ArithmeticBasketLevyPricer {
  /// Weights (need not sum to one) — a term of the contract, not a market
  /// quote.
  pub weights: Array1<f64>,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Correlation matrix.
  pub rho: Array2<f64>,
}

impl ArithmeticBasketLevyPricer {
  /// Builds the pricer from the basket weights, the per-asset volatilities
  /// and their correlation matrix.
  pub fn new(weights: Array1<f64>, sigma: Array1<f64>, rho: Array2<f64>) -> Self {
    Self {
      weights,
      sigma,
      rho,
    }
  }

  /// Price either leg at one query point using Levy (1992) two-moment
  /// lognormal approximation.
  pub fn price_option(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    let m1 = first_moment(s, self.weights.view(), q, r, tau);
    let m2 = second_moment(
      s,
      self.weights.view(),
      self.sigma.view(),
      q,
      self.rho.view(),
      r,
      tau,
    );
    let var = (m2 / (m1 * m1)).ln().max(1e-14);
    let sigma_eff = (var / tau).sqrt();
    let sqrt_t = tau.sqrt();
    let d1 = ((m1 / k).ln() + 0.5 * var) / (sigma_eff * sqrt_t);
    let d2 = d1 - sigma_eff * sqrt_t;
    let disc = (-r * tau).exp();
    match option_type {
      OptionType::Call => disc * (m1 * norm_cdf(d1) - k * norm_cdf(d2)),
      OptionType::Put => disc * (k * norm_cdf(-d2) - m1 * norm_cdf(-d1)),
    }
  }

  /// Price the arithmetic basket call at one query point.
  pub fn price_call(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> f64 {
    self.price_option(s, k, r, q, tau, OptionType::Call)
  }

  /// Price the arithmetic basket put at one query point.
  pub fn price_put(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> f64 {
    self.price_option(s, k, r, q, tau, OptionType::Put)
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
  /// Time to maturity in years.
  pub tau: f64,
  /// Option type.
  pub option_type: OptionType,
  /// Average type.
  pub avg_type: BasketAverageType,
  /// Number of MC paths.
  pub n_paths: usize,
}

#[cfg(feature = "openblas")]
impl McBasketPricer {
  /// Falliable variant of [`Self::price`] that surfaces invalid inputs
  /// (non-SPD correlation, dimension mismatch) as an `Err` instead of a panic.
  pub fn try_price(&self) -> anyhow::Result<f64> {
    let n_assets = self.s.len();
    if self.rho.shape() != [n_assets, n_assets] {
      anyhow::bail!(
        "rho shape {:?} does not match n_assets={n_assets}",
        self.rho.shape()
      );
    }
    if self.weights.len() != n_assets || self.sigma.len() != n_assets || self.q.len() != n_assets {
      anyhow::bail!(
        "weights/sigma/q lengths must match n_assets={n_assets} (got {}/{}/{})",
        self.weights.len(),
        self.sigma.len(),
        self.q.len()
      );
    }
    let _ = self
      .rho
      .cholesky(UPLO::Lower)
      .map_err(|e| anyhow::anyhow!("correlation matrix is not positive definite: {e}"))?;
    Ok(self.price())
  }

  pub fn price(&self) -> f64 {
    let n_assets = self.s.len();
    let l: Array2<f64> = self.rho.cholesky(UPLO::Lower).expect(
      "correlation matrix must be positive definite — call try_price() to handle this gracefully",
    );
    let drifts: Vec<f64> = (0..n_assets)
      .map(|i| (self.r - self.q[i] - 0.5 * self.sigma[i] * self.sigma[i]) * self.tau)
      .collect();
    let vols: Vec<f64> = (0..n_assets)
      .map(|i| self.sigma[i] * self.tau.sqrt())
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

    (-self.r * self.tau).exp() * sum / n_paths as f64
  }
}

#[cfg(test)]
#[path = "basket_tests.rs"]
mod tests;
