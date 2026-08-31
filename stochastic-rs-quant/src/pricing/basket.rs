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
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;

mod monte_carlo;

pub use monte_carlo::BasketAverageType;
pub use monte_carlo::McBasketPricer;

/// Model-internal shape agreement for a basket pricer's three parameters.
///
/// The `who` prefix keeps the two callers' panic messages distinct, so a
/// `should_panic` anchored on one cannot be satisfied by the other firing.
fn assert_basket_shape(who: &str, weights: &Array1<f64>, sigma: &Array1<f64>, rho: &Array2<f64>) {
  let n = sigma.len();
  assert!(
    weights.len() == n && rho.shape() == [n, n],
    "{who}::new: weights, sigma and rho must agree on the asset count \
     (got weights {}, sigma {n}, rho {:?})",
    weights.len(),
    rho.shape()
  );
}

/// Range checks for a basket pricer's volatilities and correlation matrix.
fn assert_basket_parameters(who: &str, sigma: &Array1<f64>, rho: &Array2<f64>) {
  for (i, &v) in sigma.iter().enumerate() {
    assert!(
      v >= 0.0,
      "{who}::new: sigma[{i}] must be a non-negative volatility (got {v})"
    );
  }
  for ((i, j), &v) in rho.indexed_iter() {
    assert!(
      (-1.0..=1.0).contains(&v),
      "{who}::new: rho[{i}][{j}] must be in [-1, 1] (got {v})"
    );
  }
}

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
  /// Basket weights $w_i$, conventionally summing to one — a term of the
  /// contract, not a market quote. The sum is not enforced; see
  /// [`new`](Self::new).
  pub weights: Array1<f64>,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Correlation matrix $\rho_{ij}$ ($n \times n$, symmetric, ones on
  /// diagonal).
  pub rho: Array2<f64>,
}

impl GeometricBasketPricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `weights`, `sigma` and `rho` disagree on the asset count
  /// - if any `sigma[i]` is negative or `NaN` — not a volatility
  /// - if any `rho[[i, j]]` is outside `[-1, 1]` or `NaN` — not a
  ///   correlation. The range test covers the diagonal, so a correlation
  ///   matrix carrying variances instead of correlations is rejected by
  ///   the same check.
  ///
  /// **Not** checked, deliberately:
  ///
  /// - the **weight sum**. $\prod_i S_i^{w_i}$ and $\sum_i w_i S_i$ are
  ///   both well defined for any weight vector, and a long/short basket
  ///   (`w = [-1, 2]`, which prices at `29.44` / `22.81`) is a real
  ///   product rather than an invalid one. Summing to one is the
  ///   Kemna-Vorst normalisation convention, not a domain constraint.
  /// - **symmetry** of `rho`. An exact-equality test would reject a
  ///   correlation matrix a covariance estimator produced symmetric only
  ///   to round-off, and any tolerance would be invented rather than
  ///   measured. The residual is recorded rather than hidden: an
  ///   asymmetric `rho` is silently symmetrised by the geometric basket —
  ///   `[[1, 0.4], [0.9, 1]]` prices *bit-identically* to
  ///   `[[1, 0.65], [0.65, 1]]`, because $\sigma_G^2$ sums over both
  ///   $(i,j)$ and $(j,i)$ — and is **not** symmetrised by the Levy
  ///   basket, whose second moment exponentiates each entry separately
  ///   (`11.530486` against the symmetrised `11.525904`).
  /// - **positive semi-definiteness** of `rho`, which needs a Cholesky
  ///   which these two pricers do not need.
  ///
  /// Measured against a healthy `10.224832`: `sigma = [-0.20, 0.30]`
  /// prices at `6.946344`, an off-diagonal `rho` of `5` at `23.016764`,
  /// and a `rho` of `-5` at **`0.0`** — the `sigma_g_sq.max(0.0)` floor
  /// swallowing a negative basket variance, the same `f64::max` trap
  /// `pricing::fourier::pricer`'s `floor_price` names. A correlation matrix with `3` on the
  /// diagonal prices at `16.837128`.
  ///
  /// The length check is model-internal only. The three `assert_eq!`s in
  /// [`price_option`](Self::price_option) compare the *query*'s asset
  /// count against the model's and **stay**: the fields are `pub`, so this
  /// constructor is a front door and not a wall.
  pub fn new(weights: Array1<f64>, sigma: Array1<f64>, rho: Array2<f64>) -> Self {
    assert_basket_shape("GeometricBasketPricer", &weights, &sigma, &rho);
    assert_basket_parameters("GeometricBasketPricer", &sigma, &rho);
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
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `weights`, `sigma` and `rho` disagree on the asset count
  /// - if any `sigma[i]` is negative or `NaN` — not a volatility
  /// - if any `rho[[i, j]]` is outside `[-1, 1]` or `NaN` — not a
  ///   correlation. The range test covers the diagonal, so a correlation
  ///   matrix carrying variances instead of correlations is rejected by
  ///   the same check.
  ///
  /// **Not** checked, deliberately:
  ///
  /// - the **weight sum**. $\prod_i S_i^{w_i}$ and $\sum_i w_i S_i$ are
  ///   both well defined for any weight vector, and a long/short basket
  ///   (`w = [-1, 2]`, which prices at `29.44` / `22.81`) is a real
  ///   product rather than an invalid one. Summing to one is the
  ///   Kemna-Vorst normalisation convention, not a domain constraint.
  /// - **symmetry** of `rho`. An exact-equality test would reject a
  ///   correlation matrix a covariance estimator produced symmetric only
  ///   to round-off, and any tolerance would be invented rather than
  ///   measured. The residual is recorded rather than hidden: an
  ///   asymmetric `rho` is silently symmetrised by the geometric basket —
  ///   `[[1, 0.4], [0.9, 1]]` prices *bit-identically* to
  ///   `[[1, 0.65], [0.65, 1]]`, because $\sigma_G^2$ sums over both
  ///   $(i,j)$ and $(j,i)$ — and is **not** symmetrised by the Levy
  ///   basket, whose second moment exponentiates each entry separately
  ///   (`11.530486` against the symmetrised `11.525904`).
  /// - **positive semi-definiteness** of `rho`, which needs a Cholesky
  ///   which these two pricers do not need.
  ///
  /// The length check earns its place here rather than at the accessor:
  /// unlike its geometric sibling this pricer has **no** dimension
  /// assertion at all, and its moment loops run over the *query*'s asset
  /// count while indexing the model's vectors. A `sigma` one entry too
  /// long returned `10.894912090686852` — bit-identical to the healthy
  /// price, with the surplus volatility silently ignored; a `weights` one
  /// entry too long returned `0.483024`, and a 3x3 `rho` against a 2-asset
  /// model `9.783632`. None of the three announced anything.
  ///
  /// Measured against a healthy `10.894912`: `sigma = [-0.20, 0.30]`
  /// prices at `8.487146`, an off-diagonal `rho` of `5` at `19.358891`,
  /// a diagonal of `3` at `15.697454`, and a `rho` of `-5` at
  /// **`4.877058`** — the basket's zero-volatility intrinsic, the same
  /// number a `NaN` `sigma` produced before the floor was split.
  pub fn new(weights: Array1<f64>, sigma: Array1<f64>, rho: Array2<f64>) -> Self {
    assert_basket_shape("ArithmeticBasketLevyPricer", &weights, &sigma, &rho);
    assert_basket_parameters("ArithmeticBasketLevyPricer", &sigma, &rho);
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
    // A floor and a poison check are different operations, and `f64::max`
    // runs them together: `f64::NAN.max(1e-14)` is `1e-14`. The floor
    // itself is right — a basket whose two matched moments satisfy
    // `m2 == m1²` has zero variance, and `1e-14` keeps `sigma_eff` out of
    // a division by zero at that degenerate limit. A `NaN` log-ratio has
    // no variance to floor: it means a model parameter or a market input
    // was undefined, and squashing it to `1e-14` gave `sigma_eff ~ 1e-7`
    // and priced the basket at its zero-volatility intrinsic. Same split
    // as `pricing::fourier::pricer`'s `floor_price`.
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
