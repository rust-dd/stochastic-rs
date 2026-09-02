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
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use owens_t::biv_norm;
use rayon::prelude::*;

use crate::mc::McEstimate;
use crate::pricing::mc_stats::std_err_from_sums;
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
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `sigma1` or `sigma2` is negative or `NaN` — not a volatility
  /// - if `rho` is outside `[-1, 1]` or `NaN` — not a correlation
  ///
  /// The volatilities are the ones that return a wrong *number*: at
  /// `sigma2 = -0.30` a `CallOnMax` prices at **-11.38**, a negative call
  /// — case 1 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails) — while the
  /// `CallOnMin` leg of the same model comes back a plausible `14.20`
  /// against the correct `6.57`.
  ///
  /// The correlation is guarded for a different reason, and it is worth
  /// keeping: an out-of-range `rho` does *not* return a number here, it
  /// trips an assertion inside the third-party `owens_t::biv_norm`, whose
  /// message is the bare offending float (`13000000.000000002`) and names
  /// neither the parameter, the pricer, nor the crate. The same `rho`
  /// reaches [`MargrabePricer`](crate::pricing::spread::MargrabePricer)
  /// through [`price`](Self::price)'s `PutOnMin`/`PutOnMax` legs, where it
  /// *is* a silent wrong number, so leaving it unchecked here would leave
  /// one payoff of four announcing a bad correlation and three not.
  ///
  /// Admissible and still accepted: perfect correlation either way, and a
  /// zero-volatility leg.
  ///
  /// No longer `const fn`. What made that safe was measured rather than
  /// assumed: **zero** `const` or `static` items of this type exist in the
  /// workspace, against 25 `StulzRainbowPricer::new` call sites.
  pub fn new(payoff: RainbowPayoff, sigma1: f64, sigma2: f64, rho: f64) -> Self {
    assert!(
      sigma1 >= 0.0,
      "StulzRainbowPricer::new: sigma1 must be a non-negative volatility (got {sigma1})"
    );
    assert!(
      sigma2 >= 0.0,
      "StulzRainbowPricer::new: sigma2 must be a non-negative volatility (got {sigma2})"
    );
    assert!(
      (-1.0..=1.0).contains(&rho),
      "StulzRainbowPricer::new: rho must be in [-1, 1] (got {rho})"
    );
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
/// It relies on a Cholesky factorization for
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

impl McRainbowPricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if any `sigma[i]` is negative or `NaN` — not a volatility
  /// - if `n_paths` is `0`
  ///
  /// `rho` is **deliberately unchecked here**, and so are the dimensions.
  /// [`try_price`](Self::try_price) is the only advertised way to surface
  /// a non-SPD correlation or a query/model dimension mismatch as an
  /// `Err`, and a constructor that panicked on them first would leave it
  /// nothing to report — `mc_rainbow_try_price_reports_a_non_spd_correlation`
  /// pins that with `rho = [[1, 2], [2, 1]]`, an entry a range check would
  /// have rejected at construction. The only element-wise correlation
  /// guard that would not pre-empt that contract is a unit-diagonal test,
  /// and a correlation check that inspects the diagonal but not the
  /// entries is precisely the asymmetry this validation exists to avoid.
  /// The array siblings that have no `try_price` —
  /// [`GeometricBasketPricer`](crate::pricing::basket::GeometricBasketPricer)
  /// and
  /// [`ArithmeticBasketLevyPricer`](crate::pricing::basket::ArithmeticBasketLevyPricer)
  /// — do get the range check, for that reason and no other.
  ///
  /// `n_paths == 0` does not close a wrong number either; it is refused
  /// where it is supplied, matching the other Monte Carlo pricers.
  pub fn new(payoff: RainbowPayoff, sigma: Array1<f64>, rho: Array2<f64>, n_paths: usize) -> Self {
    for (i, &v) in sigma.iter().enumerate() {
      assert!(
        v >= 0.0,
        "McRainbowPricer::new: sigma[{i}] must be a non-negative volatility (got {v})"
      );
    }
    assert!(
      n_paths >= 1,
      "McRainbowPricer::new: n_paths must be at least 1 (got {n_paths})"
    );
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
  ) -> anyhow::Result<McEstimate<f64>> {
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
    if !crate::linalg::is_spd_t(&self.rho) {
      anyhow::bail!("correlation matrix is not positive definite");
    }
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
  ) -> McEstimate<f64> {
    let n_assets = s.len();
    let l: Array2<f64> = crate::linalg::spd_cholesky_lower(&self.rho).expect(
      "correlation matrix must be positive definite — call try_price() to handle this gracefully",
    );
    let drifts: Vec<f64> = (0..n_assets)
      .map(|i| (r - q[i] - 0.5 * self.sigma[i] * self.sigma[i]) * tau)
      .collect();
    let vols: Vec<f64> = (0..n_assets).map(|i| self.sigma[i] * tau.sqrt()).collect();
    let n_paths = self.n_paths;

    let mut all_z = vec![0.0_f64; n_paths * n_assets];
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let payoff_of = |p: usize| -> f64 {
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
    };
    let sum: f64 = (0..n_paths).into_par_iter().map(&payoff_of).sum();
    let sum_sq: f64 = (0..n_paths)
      .into_par_iter()
      .map(|p| {
        let y = payoff_of(p);
        y * y
      })
      .sum();

    let discount = (-r * tau).exp();
    McEstimate {
      mean: discount * sum / n_paths as f64,
      std_err: discount * std_err_from_sums(sum, sum_sq, n_paths),
      n_samples: n_paths,
    }
  }
}

#[cfg(test)]
#[path = "rainbow_tests.rs"]
mod tests;
