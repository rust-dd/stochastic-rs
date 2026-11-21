//! Basket options on `B = Σ wᵢSᵢ`: the geometric one in closed form, the arithmetic one by Levy's
//! two-moment lognormal, and either average by Monte Carlo under any correlation.
//!
//! Reference: Levy (1992), J. Int. Money Finance 11
//! Reference: Turnbull & Wakeman (1991), "A quick algorithm for pricing European average options"
//! Reference: Hu et al. (2023), arXiv:2302.08041
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::not_positive;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::rainbow::assert_volatilities;

mod monte_carlo;

pub use monte_carlo::BasketAverageType;
pub use monte_carlo::McBasketPricer;

/// One weight and one volatility per asset, a square `rho`, and each entry in its range.
#[track_caller]
fn assert_basket_model(weights: &Array1<f64>, sigma: &Array1<f64>, rho: &Array2<f64>) {
  assert_basket_shape(weights, sigma, rho);
  assert_volatilities(sigma);
  for ((i, j), &v) in rho.indexed_iter() {
    assert!(
      (-1.0..=1.0).contains(&v),
      "rho[{i}][{j}] must satisfy `(-1.0..=1.0).contains(&rho[{i}][{j}])`, got rho[{i}][{j}] = {v:?}"
    );
  }
}

/// `weights` and `rho` agree with `sigma` on the asset count.
fn assert_basket_shape(weights: &Array1<f64>, sigma: &Array1<f64>, rho: &Array2<f64>) {
  let (n, shape) = (sigma.len(), rho.dim());
  assert!(
    weights.len() == n,
    "weights must satisfy `weights.len() == sigma.len()`, got weights.len() = {:?}",
    weights.len()
  );
  assert!(
    shape == (n, n),
    "rho must satisfy `rho.dim() == (sigma.len(), sigma.len())`, got rho.dim() = {shape:?}"
  );
}

/// A query's spot and yield vectors hold one entry per asset; the model's `pub` fields may have
/// been written since `new`, so its shape is read again.
fn assert_basket_query(
  (weights, sigma, rho): (&Array1<f64>, &Array1<f64>, &Array2<f64>),
  s: ArrayView1<'_, f64>,
  q: ArrayView1<'_, f64>,
) {
  assert_basket_shape(weights, sigma, rho);
  let (n, spots, yields) = (sigma.len(), s.len(), q.len());
  assert!(
    spots == n,
    "s must satisfy `s.len() == sigma.len()`, got s.len() = {spots:?}"
  );
  assert!(
    yields == n,
    "q must satisfy `q.len() == sigma.len()`, got q.len() = {yields:?}"
  );
}

/// Geometric basket on `n` correlated Gbm assets, whose weighted geometric mean is lognormal; the
/// weights are contract terms that also fix `σ_G`, and `n` assets take no `ModelPricer`.
///
/// $$
/// G_T = \prod_{i=1}^n S_{i,T}^{w_i},\qquad
/// \ln G_T \sim \mathcal N\!\big(\mu_G T, \sigma_G^2 T\big)
/// $$
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
  /// Basket weights `wᵢ`, a term of the contract; summing to one by convention, not enforced.
  pub weights: Array1<f64>,
  /// Annualised Black–Scholes volatility of each asset.
  pub sigma: Array1<f64>,
  /// Correlation matrix $\rho_{ij}$ ($n \times n$, symmetric, ones on
  /// diagonal).
  pub rho: Array2<f64>,
}

impl GeometricBasketPricer {
  /// Panics unless `weights`, `sigma` and `rho` agree on the asset count, `sigma` is finite and
  /// `≥ 0` and `rho` in `[-1, 1]`; weight sum, symmetry and PSD-ness are deliberately unchecked.
  pub fn new(weights: Array1<f64>, sigma: Array1<f64>, rho: Array2<f64>) -> Self {
    assert_basket_model(&weights, &sigma, &rho);
    Self {
      weights,
      sigma,
      rho,
    }
  }

  /// Either leg: the payoff at `tau = 0`; NaN past expiry, for a spot or strike not positive before
  /// it, or at zero volatility with the forward at the strike; panics on a mis-sized `s` or `q`.
  pub fn price_option(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    let n_assets = self.sigma.len();
    assert_basket_query((&self.weights, &self.sigma, &self.rho), s, q);
    let spot = || {
      (0..n_assets)
        .map(|i| self.weights[i] * s[i].ln())
        .sum::<f64>()
        .exp()
    };
    if let Some(value) = expired_basket(tau, r, q, || vanilla_payoff(option_type, spot(), k)) {
      return value;
    }
    if not_positive(&[k]) || !s.iter().all(|&spot| spot > 0.0) {
      return f64::NAN;
    }

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

/// Arithmetic basket by Levy (1992): the lognormal matching `B = Σ wᵢS_{i,T}`'s first two
/// risk-neutral moments, both of which carry the query's rate, yields and spots.
///
/// $$
/// E\left[B\right] = \sum_i w_i S_{i,0} e^{(r-q_i)T},\qquad
/// E\left[B^2\right] = \sum_{i,j} w_i w_j S_i S_j e^{((r-q_i) + (r-q_j) + \rho_{ij}\sigma_i\sigma_j)T}
/// $$
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
  /// Annualised Black–Scholes volatility of each asset.
  pub sigma: Array1<f64>,
  /// Correlation matrix.
  pub rho: Array2<f64>,
}

impl ArithmeticBasketLevyPricer {
  /// Panics unless `weights`, `sigma` and `rho` agree on the asset count, `sigma` is finite and
  /// `≥ 0` and `rho` in `[-1, 1]`; weight sum, symmetry and PSD-ness are deliberately unchecked.
  pub fn new(weights: Array1<f64>, sigma: Array1<f64>, rho: Array2<f64>) -> Self {
    assert_basket_model(&weights, &sigma, &rho);
    Self {
      weights,
      sigma,
      rho,
    }
  }

  /// Levy's (1992) two-moment lognormal: the payoff at `tau = 0`; NaN past expiry, for a spot or
  /// strike not positive, or where the moments define no lognormal; panics on a mis-sized `s`, `q`.
  pub fn price_option(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    assert_basket_query((&self.weights, &self.sigma, &self.rho), s, q);
    let spot = || (0..s.len()).map(|i| self.weights[i] * s[i]).sum::<f64>();
    if let Some(value) = expired_basket(tau, r, q, || vanilla_payoff(option_type, spot(), k)) {
      return value;
    }
    if not_positive(&[k]) || !s.iter().all(|&spot| spot > 0.0) {
      return f64::NAN;
    }
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
    // Floor the variance at `1e-14` (zero variance at `m2 == m1²` would divide by zero) but keep a
    // NaN, which `f64::max` would turn into a zero-volatility intrinsic price.
    let log_ratio = (m2 / (m1 * m1)).ln();
    let var = if log_ratio.is_nan() {
      log_ratio
    } else {
      log_ratio.max(1e-14)
    };
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

/// [`expired`] over the rate and every asset's dividend yield.
fn expired_basket(
  tau: f64,
  r: f64,
  q: ArrayView1<'_, f64>,
  payoff: impl FnOnce() -> f64,
) -> Option<f64> {
  if tau > 0.0 && tau < f64::INFINITY {
    return None;
  }
  let rates = q.iter().copied().chain([r]).collect::<Vec<f64>>();
  expired(tau, &rates, payoff)
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

#[cfg(test)]
#[path = "basket_tests.rs"]
mod tests;
