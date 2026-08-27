//! # Rainbow
//!
//! Multi-asset options on the maximum or minimum of $n$ correlated assets.
//!
//! Two-asset closed forms (Stulz 1982):
//! $$
//! C_{\max}(S_1,S_2,K) = C_1 + C_2 - C_{\min}(S_1,S_2,K)
//! $$
//! $$
//! C_{\min}(S_1,S_2,K) = S_1 e^{-q_1T}M(\gamma_1,d;\rho_1)
//!   + S_2 e^{-q_2T}M(\gamma_2,d-\sigma\sqrt T;\rho_2)
//!   - K e^{-rT}M(\gamma_1-\sigma_1\sqrt T,\gamma_2-\sigma_2\sqrt T;\rho)
//! $$
//!
//! Source:
//! - Stulz, R. M. (1982), "Options on the minimum or the maximum of two risky assets",
//!   J. Financial Economics 10
//! - Johnson, H. (1987), "Options on the maximum or the minimum of several assets",
//!   J. Financial & Quantitative Analysis 22
//! - Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", 2nd ed., Ch. 5
//!
#[cfg(feature = "openblas")]
use ndarray::Array1;
#[cfg(feature = "openblas")]
use ndarray::Array2;
#[cfg(feature = "openblas")]
use ndarray::ArrayView1;
#[cfg(feature = "openblas")]
use ndarray_linalg::Cholesky;
#[cfg(feature = "openblas")]
use ndarray_linalg::UPLO;
use owens_t::biv_norm;
#[cfg(feature = "openblas")]
use rayon::prelude::*;

#[cfg(feature = "openblas")]
use crate::traits::FloatExt;

/// Type of multi-asset rainbow payoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainbowPayoff {
  /// $\max(\max(S_i) - K, 0)$
  CallOnMax,
  /// $\max(\min(S_i) - K, 0)$
  CallOnMin,
  /// $\max(K - \max(S_i), 0)$
  PutOnMax,
  /// $\max(K - \min(S_i), 0)$
  PutOnMin,
}

/// `a.max(b)` with a `NaN` operand **propagated** instead of discarded.
///
/// `f64::max` returns the non-`NaN` operand, which is right for an
/// ordering question and wrong for every use in this file. Folded over an
/// asset vector it silently *drops* an undefined leg, so an $n$-asset
/// best-of prices as an $(n-1)$-asset best-of; applied as a payoff floor
/// it turns an undefined payoff into a plausible zero. Same split as
/// `pricing::fourier::pricer`'s `floor_price`, which closes the identical
/// trap on the Fourier path.
#[inline]
fn nan_max(a: f64, b: f64) -> f64 {
  if a.is_nan() || b.is_nan() {
    f64::NAN
  } else {
    a.max(b)
  }
}

/// `a.min(b)` with a `NaN` operand propagated — see `nan_max`.
#[inline]
fn nan_min(a: f64, b: f64) -> f64 {
  if a.is_nan() || b.is_nan() {
    f64::NAN
  } else {
    a.min(b)
  }
}

impl RainbowPayoff {
  /// The contract's payoff on one simulated terminal-price vector.
  ///
  /// A non-finite leg poisons the payoff rather than dropping out of it.
  /// Both reductions and the floor go through `nan_max` / `nan_min`:
  /// with the plain `f64::max`/`f64::min` a `NaN` asset was discarded by
  /// the fold *and* the surviving `(max_p - k).max(0.0)` floor would have
  /// discarded it again, so `CallOnMax` on `[120, NaN, 90]` at `K = 100`
  /// returned `20.0` — exactly the two-asset answer, with nothing marking
  /// the third asset as missing.
  ///
  /// An **empty** `prices` slice still yields `0.0` through the untouched
  /// `±inf` fold seeds. That is a different question — a basket with no
  /// assets — and it is not reachable from either Monte Carlo caller,
  /// whose asset count comes from the query's own spot vector.
  pub fn evaluate(&self, prices: &[f64], k: f64) -> f64 {
    let max_p = prices.iter().copied().fold(f64::NEG_INFINITY, nan_max);
    let min_p = prices.iter().copied().fold(f64::INFINITY, nan_min);
    match self {
      RainbowPayoff::CallOnMax => nan_max(max_p - k, 0.0),
      RainbowPayoff::CallOnMin => nan_max(min_p - k, 0.0),
      RainbowPayoff::PutOnMax => nan_max(k - max_p, 0.0),
      RainbowPayoff::PutOnMin => nan_max(k - min_p, 0.0),
    }
  }
}

/// Stulz (1982) closed-form pricer for a two-asset max/min option.
///
/// The struct holds **model and contract state only** — the two
/// volatilities, their correlation, and which of the four best-of/worst-of
/// payoffs this contract pays. The two spots, the strike, the rate, the two
/// dividend yields and the maturity are the pricing *query* and travel as
/// arguments to [`price`](Self::price), so one instance prices a whole
/// strike/maturity grid.
///
/// [`RainbowPayoff`] stays on the struct for the same reason the digitals'
/// cash payout does: it names *which contract* this is, not how the market
/// is quoted. It carries a call/put axis of its own, which is the one place
/// this family departs from the crate's `price_call`/`price_put`
/// convention — separating that axis out would mean breaking a second
/// public enum, so the whole payoff travels together.
///
/// Nothing derived from a query is cached. The combined spread volatility
/// $\sqrt{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}$ is model-only, and
/// is still recomputed per call; $d$, $y_1$ and $y_2$ could not be cached
/// in any case, since each carries the query's own rates and maturity.
///
/// A two-asset payoff carries no [`ModelPricer`](crate::traits::ModelPricer),
/// whose `price_call(s, k, r, q, tau)` has one underlying. This is the
/// multi-asset "convention, no trait" family — see
/// [`KirkSpreadPricer`](crate::pricing::kirk::KirkSpreadPricer).
///
/// ```
/// use stochastic_rs_quant::pricing::rainbow::RainbowPayoff;
/// use stochastic_rs_quant::pricing::rainbow::StulzRainbowPricer;
///
/// let worst_of = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.20, 0.30, 0.5);
/// let best_of = StulzRainbowPricer::new(RainbowPayoff::CallOnMax, 0.20, 0.30, 0.5);
/// let cmin = worst_of.price(100.0, 105.0, 100.0, 0.05, 0.0, 0.0, 1.0);
/// let cmax = best_of.price(100.0, 105.0, 100.0, 0.05, 0.0, 0.0, 1.0);
/// assert!(cmax > cmin, "the best of two assets is worth more than the worst");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StulzRainbowPricer {
  /// Payoff type — a term of the contract, not a market quote, so it stays
  /// on the struct next to the model rather than travelling with the query.
  pub payoff: RainbowPayoff,
  /// Volatility 1.
  pub sigma1: f64,
  /// Volatility 2.
  pub sigma2: f64,
  /// Correlation.
  pub rho: f64,
}

impl StulzRainbowPricer {
  /// Builds the pricer from the payoff this contract pays, the two
  /// volatilities and their correlation.
  pub const fn new(payoff: RainbowPayoff, sigma1: f64, sigma2: f64, rho: f64) -> Self {
    Self {
      payoff,
      sigma1,
      sigma2,
      rho,
    }
  }

  /// Price the contract at one query point.
  pub fn price(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    match self.payoff {
      RainbowPayoff::CallOnMin => self.call_on_min(s1, s2, k, r, q1, q2, tau),
      RainbowPayoff::CallOnMax => self.call_on_max(s1, s2, k, r, q1, q2, tau),
      RainbowPayoff::PutOnMin => self.put_on_min(s1, s2, k, r, q1, q2, tau),
      RainbowPayoff::PutOnMax => self.put_on_max(s1, s2, k, r, q1, q2, tau),
    }
  }

  fn call_on_min(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    let v1 = self.sigma1;
    let v2 = self.sigma2;
    let rho = self.rho;
    let t = tau;
    let sqrt_t = t.sqrt();

    // Combined spread vol
    let sigma_sq = v1 * v1 + v2 * v2 - 2.0 * rho * v1 * v2;
    let sigma = sigma_sq.max(1e-14).sqrt();
    let rho_1 = (v1 - rho * v2) / sigma;
    let rho_2 = (v2 - rho * v1) / sigma;

    let y1 = ((s1 / k).ln() + (r - q1 + 0.5 * v1 * v1) * t) / (v1 * sqrt_t);
    let y2 = ((s2 / k).ln() + (r - q2 + 0.5 * v2 * v2) * t) / (v2 * sqrt_t);
    let d = ((s1 / s2).ln() + (q2 - q1 + 0.5 * sigma_sq) * t) / (sigma * sqrt_t);

    let bvn = |a: f64, b: f64, c: f64| -> f64 { biv_norm(-a, -b, c) };

    s1 * (-q1 * t).exp() * bvn(y1, -d, -rho_1)
      + s2 * (-q2 * t).exp() * bvn(y2, d - sigma * sqrt_t, -rho_2)
      - k * (-r * t).exp() * bvn(y1 - v1 * sqrt_t, y2 - v2 * sqrt_t, rho)
  }

  fn call_on_max(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    use crate::pricing::bsm::BSMCoc;
    use crate::pricing::bsm::BSMPricer;
    use crate::traits::ModelPricer;

    // Stulz identity: max(max(S1,S2) - K, 0) = call(S1, K) + call(S2, K)
    // - call_on_min(S1, S2, K)
    let c1 = BSMPricer::new(self.sigma1, BSMCoc::Merton1973).price_call(s1, k, r, q1, tau);
    let c2 = BSMPricer::new(self.sigma2, BSMCoc::Merton1973).price_call(s2, k, r, q2, tau);
    c1 + c2 - self.call_on_min(s1, s2, k, r, q1, q2, tau)
  }

  fn put_on_min(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    // Stulz identity: max(K - min(S1,S2), 0) = K * e^{-rT} - min_call(0)
    //   = call_on_min(S1, S2, K) - F_min where F_min = E[min(S1,S2)] discounted
    // Easier: put-call parity for min options:
    // put_on_min - call_on_min = K * e^{-rT} - E[min(S1, S2)] * e^{-rT}
    // E[min(S1, S2)] = S1 e^{(r-q1)T} + S2 e^{(r-q2)T} - E[max(S1, S2)]
    // and E[max(S1, S2)] - E[min(S1, S2)] is the Margrabe expected payoff.
    let call_min = self.call_on_min(s1, s2, k, r, q1, q2, tau);
    // Use put-call-min parity: P_min = C_min + K e^{-rT} - F_min where
    // F_min = S1 e^{-q1 T} + S2 e^{-q2 T} - F_max and the difference
    // F_max - F_min equals the Margrabe expected payoff $E[(S_1-S_2)^+] +
    // E[(S_2-S_1)^+]$. Use Margrabe to compute it.
    use crate::pricing::spread::MargrabePricer;
    let m12 = MargrabePricer::new(self.sigma1, self.sigma2, self.rho).price(s1, s2, q1, q2, tau);
    let m21 = MargrabePricer::new(self.sigma2, self.sigma1, self.rho).price(s2, s1, q2, q1, tau);
    // F_max + F_min = s1 e^{-q1T} + s2 e^{-q2T}, F_max - F_min = m12 + m21,
    // so F_min = (s1 e^{-q1T} + s2 e^{-q2T} - (m12 + m21)) / 2.
    let f_min = 0.5 * (s1 * (-q1 * tau).exp() + s2 * (-q2 * tau).exp() - (m12 + m21));
    call_min + k * (-r * tau).exp() - f_min
  }

  fn put_on_max(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    let call_max = self.call_on_max(s1, s2, k, r, q1, q2, tau);
    use crate::pricing::spread::MargrabePricer;
    let m12 = MargrabePricer::new(self.sigma1, self.sigma2, self.rho).price(s1, s2, q1, q2, tau);
    let m21 = MargrabePricer::new(self.sigma2, self.sigma1, self.rho).price(s2, s1, q2, q1, tau);
    let f_max = 0.5 * (s1 * (-q1 * tau).exp() + s2 * (-q2 * tau).exp() + (m12 + m21));
    call_max + k * (-r * tau).exp() - f_max
  }
}

/// Monte-Carlo rainbow pricer for arbitrary $n$ assets. Gated behind the
/// `openblas` feature because it relies on `ndarray_linalg::Cholesky` for
/// the correlation factor.
///
/// The struct holds **model, contract and method state only** — the vector
/// of volatilities, the correlation matrix, which best-of/worst-of payoff
/// this contract pays, and the Monte Carlo path count. The spot vector, the
/// strike, the rate, the dividend-yield vector and the maturity are the
/// pricing *query* and travel as arguments to [`price`](Self::price).
///
/// The volatilities and the correlation matrix fix how many assets the
/// model has, and the query's spot and yield vectors have to agree with it.
/// That check stays in [`try_price`](Self::try_price) rather than moving to
/// [`new`](Self::new), alongside the positive-definiteness test: `try_price`
/// is the only advertised way to surface either as an `Err`, and a
/// constructor that panicked on them first would leave it nothing to report.
#[cfg(feature = "openblas")]
#[derive(Debug, Clone)]
pub struct McRainbowPricer {
  /// Payoff type — a term of the contract, not a market quote.
  pub payoff: RainbowPayoff,
  /// Volatilities.
  pub sigma: Array1<f64>,
  /// Correlation matrix.
  pub rho: Array2<f64>,
  /// Number of MC paths.
  pub n_paths: usize,
}

#[cfg(feature = "openblas")]
impl McRainbowPricer {
  /// Builds the pricer from the payoff this contract pays, the per-asset
  /// volatilities and their correlation matrix, plus the Monte Carlo path
  /// count every price off this instance uses.
  pub fn new(payoff: RainbowPayoff, sigma: Array1<f64>, rho: Array2<f64>, n_paths: usize) -> Self {
    Self {
      payoff,
      sigma,
      rho,
      n_paths,
    }
  }

  /// Falliable variant of [`Self::price`] that surfaces invalid inputs
  /// (non-SPD correlation, dimension mismatch) as an `Err` instead of a panic.
  pub fn try_price(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> anyhow::Result<f64> {
    let n_assets = s.len();
    if self.rho.shape() != [n_assets, n_assets] {
      anyhow::bail!(
        "rho shape {:?} does not match n_assets={n_assets}",
        self.rho.shape()
      );
    }
    if self.sigma.len() != n_assets || q.len() != n_assets {
      anyhow::bail!(
        "sigma/q lengths must match n_assets={n_assets} (got {}/{})",
        self.sigma.len(),
        q.len()
      );
    }
    let _ = self
      .rho
      .cholesky(UPLO::Lower)
      .map_err(|e| anyhow::anyhow!("correlation matrix is not positive definite: {e}"))?;
    Ok(self.price(s, k, r, q, tau))
  }

  /// Price the contract at one query point.
  ///
  /// # Panics
  /// If the correlation matrix is not positive definite — call
  /// [`try_price`](Self::try_price) to handle that gracefully.
  pub fn price(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> f64 {
    let n_assets = s.len();
    let l: Array2<f64> = self.rho.cholesky(UPLO::Lower).expect(
      "correlation matrix must be positive definite — call try_price() to handle this gracefully",
    );
    let drifts: Vec<f64> = (0..n_assets)
      .map(|i| (r - q[i] - 0.5 * self.sigma[i] * self.sigma[i]) * tau)
      .collect();
    let vols: Vec<f64> = (0..n_assets).map(|i| self.sigma[i] * tau.sqrt()).collect();
    let n_paths = self.n_paths;

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
          .map(|i| s[i] * (drifts[i] + vols[i] * zc[i]).exp())
          .collect();
        self.payoff.evaluate(&s_t, k)
      })
      .sum();

    (-r * tau).exp() * sum / n_paths as f64
  }
}

#[cfg(test)]
mod tests {
  #[cfg(feature = "openblas")]
  use ndarray::array;

  use super::*;

  /// Cross-arch tolerance: these goldens come from `biv_norm` and
  /// `norm_cdf`, whose last bit is a hostage to FMA contraction and libm
  /// differences between the aarch64-darwin dev machine and CI's ubuntu
  /// x86_64.
  const TOL: f64 = 1e-12;

  /// Values captured from the bundled-market-data `StulzRainbowPricer`
  /// **before** the model/query reshape. The reshape is an API change only,
  /// so these must not move. All four payoffs are pinned, because
  /// `PutOnMin` and `PutOnMax` route through `MargrabePricer`, which the
  /// same wave reshaped one commit earlier.
  #[test]
  fn stulz_matches_pre_refactor_goldens() {
    let expected = [
      (RainbowPayoff::CallOnMin, 6.572032430799396),
      (RainbowPayoff::CallOnMax, 21.3836021143453),
      (RainbowPayoff::PutOnMin, 10.164180157272469),
      (RainbowPayoff::PutOnMax, 3.0373392880150476),
    ];
    for (payoff, want) in expected {
      let got = StulzRainbowPricer::new(payoff, 0.20, 0.30, 0.5)
        .price(100.0, 105.0, 100.0, 0.05, 0.0, 0.0, 1.0);
      assert!((got - want).abs() < TOL, "{payoff:?} {got}");
    }

    // Asymmetric: distinct spots, strike, both dividend yields, a negative
    // correlation and a non-unit maturity, so a query field left behind on
    // the struct could not survive by coinciding with a default.
    let asymmetric = [
      (RainbowPayoff::CallOnMin, 4.089461008811323),
      (RainbowPayoff::CallOnMax, 39.80815123244825),
      (RainbowPayoff::PutOnMin, 18.459568337156256),
      (RainbowPayoff::PutOnMax, 0.4223483563151831),
    ];
    for (payoff, want) in asymmetric {
      let got = StulzRainbowPricer::new(payoff, 0.33, 0.19, -0.4)
        .price(88.0, 121.0, 95.0, 0.037, 0.021, 0.013, 1.75);
      assert!((got - want).abs() < TOL, "{payoff:?} {got}");
    }
  }

  /// One model instance prices a whole strike grid — the point of the split.
  #[test]
  fn stulz_one_model_prices_a_strike_grid() {
    let model = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.25, 0.30, 0.4);
    let prices = [90.0, 100.0, 110.0].map(|k| model.price(100.0, 100.0, k, 0.05, 0.0, 0.0, 1.0));
    assert!(
      prices[0] > prices[1] && prices[1] > prices[2],
      "worst-of calls must decay in the strike: {prices:?}"
    );
  }

  /// The maturity is a query argument, so one instance covers a term
  /// structure. A `tau` cached at construction would return the same number
  /// three times.
  #[test]
  fn stulz_one_model_prices_a_maturity_grid() {
    let model = StulzRainbowPricer::new(RainbowPayoff::CallOnMax, 0.25, 0.30, 0.4);
    let prices = [0.25, 1.0, 4.0].map(|tau| model.price(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, tau));
    assert!(
      prices[0] < prices[1] && prices[1] < prices[2],
      "best-of calls must rise in tau: {prices:?}"
    );
  }

  /// Stulz: $C_{\min} + C_{\max} = C_1 + C_2$ (vanilla call sum).
  #[test]
  fn stulz_min_max_decomposition() {
    use crate::pricing::bsm::BSMCoc;
    use crate::pricing::bsm::BSMPricer;
    use crate::traits::ModelPricer;

    let s1 = 100.0;
    let s2 = 105.0;
    let k = 100.0;
    let v1 = 0.20;
    let v2 = 0.30;
    let rho = 0.5;
    let r = 0.05;
    let q1 = 0.0;
    let q2 = 0.0;
    let tau = 1.0;
    let cmin = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, v1, v2, rho)
      .price(s1, s2, k, r, q1, q2, tau);
    let cmax = StulzRainbowPricer::new(RainbowPayoff::CallOnMax, v1, v2, rho)
      .price(s1, s2, k, r, q1, q2, tau);
    let c1 = BSMPricer::new(v1, BSMCoc::Merton1973).price_call(s1, k, r, q1, tau);
    let c2 = BSMPricer::new(v2, BSMCoc::Merton1973).price_call(s2, k, r, q2, tau);
    let lhs = cmin + cmax;
    let rhs = c1 + c2;
    assert!((lhs - rhs).abs() < 0.01, "lhs={lhs}, rhs={rhs}");
  }

  /// Stulz call-on-min should match Monte Carlo within 2%.
  #[cfg(feature = "openblas")]
  #[test]
  fn stulz_min_matches_mc() {
    let stulz = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.25, 0.30, 0.4)
      .price(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 1.0);
    let s = array![100.0, 100.0];
    let q = array![0.0, 0.0];
    let mc = McRainbowPricer::new(
      RainbowPayoff::CallOnMin,
      array![0.25, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
      200_000,
    )
    .price(s.view(), 100.0, 0.05, q.view(), 1.0);
    let rel = (stulz - mc).abs() / stulz.max(1e-10);
    assert!(rel < 0.03, "stulz={stulz}, mc={mc}, rel={rel}");
  }

  /// CallOnMax >= each individual vanilla call (always have at least one
  /// asset path in the money).
  #[test]
  fn call_on_max_dominates_vanilla() {
    use crate::pricing::bsm::BSMCoc;
    use crate::pricing::bsm::BSMPricer;
    use crate::traits::ModelPricer;

    let s1 = 100.0;
    let s2 = 100.0;
    let v1 = 0.25;
    let v2 = 0.25;
    let rho = 0.0;
    let cmax = StulzRainbowPricer::new(RainbowPayoff::CallOnMax, v1, v2, rho)
      .price(s1, s2, 100.0, 0.05, 0.0, 0.0, 1.0);
    let c1 = BSMPricer::new(v1, BSMCoc::Merton1973).price_call(s1, 100.0, 0.05, 0.0, 1.0);
    assert!(cmax > c1, "cmax={cmax} should be > c1={c1}");
  }

  /// 5-asset MC rainbow CallOnMax should be greater than CallOnMin.
  #[cfg(feature = "openblas")]
  #[test]
  fn mc_call_on_max_above_min() {
    let n = 5;
    let s = Array1::from_elem(n, 100.0);
    let sig = Array1::from_elem(n, 0.25);
    let q = Array1::from_elem(n, 0.0);
    let mut rho = Array2::<f64>::from_elem((n, n), 0.3);
    for i in 0..n {
      rho[[i, i]] = 1.0;
    }
    let mc_max = McRainbowPricer::new(RainbowPayoff::CallOnMax, sig.clone(), rho.clone(), 50_000)
      .price(s.view(), 100.0, 0.05, q.view(), 1.0);
    let mc_min = McRainbowPricer::new(RainbowPayoff::CallOnMin, sig, rho, 50_000).price(
      s.view(),
      100.0,
      0.05,
      q.view(),
      1.0,
    );
    assert!(mc_max > mc_min);
  }

  /// One Monte Carlo model instance prices a whole strike grid. The
  /// strikes are far enough apart that the ordering survives the sampling
  /// error of independent simulations.
  #[cfg(feature = "openblas")]
  #[test]
  fn mc_rainbow_one_model_prices_a_strike_grid() {
    let s = array![100.0, 100.0];
    let q = array![0.0, 0.0];
    let model = McRainbowPricer::new(
      RainbowPayoff::CallOnMax,
      array![0.25, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
      50_000,
    );
    let prices = [80.0, 100.0, 130.0].map(|k| model.price(s.view(), k, 0.05, q.view(), 1.0));
    assert!(
      prices[0] > prices[1] && prices[1] > prices[2],
      "best-of calls must decay in the strike: {prices:?}"
    );
  }

  /// The model fixes how many assets there are; a query that disagrees is
  /// reported by `try_price` as an `Err`, not a panic. Pinned because that
  /// is the reason the check did not move to the constructor.
  #[cfg(feature = "openblas")]
  #[test]
  fn mc_rainbow_try_price_reports_a_query_dimension_mismatch() {
    let model = McRainbowPricer::new(
      RainbowPayoff::CallOnMin,
      array![0.25, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
      1_000,
    );
    let s = array![100.0, 100.0, 100.0];
    let q = array![0.0, 0.0, 0.0];
    let err = model
      .try_price(s.view(), 100.0, 0.05, q.view(), 1.0)
      .expect_err("a three-asset query against a two-asset model is not priceable");
    assert!(
      err.to_string().contains("does not match n_assets=3"),
      "{err}"
    );
  }

  /// A correlation matrix that is symmetric but not positive definite is
  /// also an `Err` rather than a panic — the other half of what keeps the
  /// constructor unguarded.
  #[cfg(feature = "openblas")]
  #[test]
  fn mc_rainbow_try_price_reports_a_non_spd_correlation() {
    let model = McRainbowPricer::new(
      RainbowPayoff::CallOnMin,
      array![0.25, 0.30],
      array![[1.0, 2.0], [2.0, 1.0]],
      1_000,
    );
    let s = array![100.0, 100.0];
    let q = array![0.0, 0.0];
    let err = model
      .try_price(s.view(), 100.0, 0.05, q.view(), 1.0)
      .expect_err("rho = 2 is not a correlation");
    assert!(err.to_string().contains("not positive definite"), "{err}");
  }

  /// A `NaN` leg used to be **dropped**, so an $n$-asset best-of priced as
  /// an $(n-1)$-asset best-of.
  ///
  /// The identity with the two-asset answer is what makes it a silent
  /// defect rather than a visible one: `CallOnMax` on `[120, NaN, 90]` at
  /// `K = 100` returned `20.0`, bit-for-bit the value of the same contract
  /// written on `[120, 90]`. Nothing in the number marks the third asset
  /// as missing.
  ///
  /// All four payoffs are pinned. Two of them (`CallOnMin`, `PutOnMax`)
  /// returned `0.0` instead, through the *second* copy of the same trap —
  /// the surviving `(min_p - k).max(0.0)` floor — so a fix to the fold
  /// alone would have left them laundering.
  #[test]
  fn a_nan_leg_poisons_the_rainbow_payoff_instead_of_dropping_out() {
    let legs = [120.0, f64::NAN, 90.0];
    for payoff in [
      RainbowPayoff::CallOnMax,
      RainbowPayoff::CallOnMin,
      RainbowPayoff::PutOnMax,
      RainbowPayoff::PutOnMin,
    ] {
      let got = payoff.evaluate(&legs, 100.0);
      assert!(got.is_nan(), "{payoff:?} on a NaN leg must not pay: {got}");
      // Every payoff is also poisoned by an undefined strike.
      let by_strike = payoff.evaluate(&[120.0, 90.0], f64::NAN);
      assert!(
        by_strike.is_nan(),
        "{payoff:?} at a NaN strike must not pay: {by_strike}"
      );
    }

    // The two-asset value the three-asset contract used to impersonate.
    assert_eq!(
      RainbowPayoff::CallOnMax.evaluate(&[120.0, 90.0], 100.0),
      20.0
    );
    // And the floor is still a floor for a real, out-of-the-money basket.
    assert_eq!(
      RainbowPayoff::CallOnMax.evaluate(&[80.0, 90.0], 100.0),
      0.0,
      "a worthless best-of still pays zero"
    );
    assert_eq!(
      RainbowPayoff::PutOnMin.evaluate(&[120.0, 90.0], 100.0),
      10.0,
      "the surviving payoffs must be unchanged"
    );
  }

  /// Stulz put-on-min via parity should be positive.
  #[test]
  fn stulz_put_on_min_positive() {
    let p = StulzRainbowPricer::new(RainbowPayoff::PutOnMin, 0.25, 0.20, 0.3);
    let price = p.price(100.0, 105.0, 100.0, 0.05, 0.0, 0.0, 0.5);
    assert!(price > 0.0, "put_on_min={price}");
  }
}
