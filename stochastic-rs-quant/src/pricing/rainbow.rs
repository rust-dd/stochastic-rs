//! Options on the maximum or minimum of correlated assets: Stulz's two-asset closed forms, and
//! Monte Carlo for any number of assets.
//!
//! $$
//! C_{\max}(S_1,S_2,K) = C_1 + C_2 - C_{\min}(S_1,S_2,K)
//! $$
//! $$
//! C_{\min}(S_1,S_2,K) = S_1 e^{-q_1T}M(\gamma_1,d;\rho_1) +
//!   S_2 e^{-q_2T}M(\gamma_2,d-\sigma\sqrt T;\rho_2) -
//!   K e^{-rT}M(\gamma_1-\sigma_1\sqrt T,\gamma_2-\sigma_2\sqrt T;\rho)
//! $$
//!
//! Reference: Stulz (1982), J. Financ. Econ. 10
//! Reference: Johnson (1987), J. Financ. Quant. Anal. 22
//! Reference: Haug (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., Ch. 5

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use owens_t::biv_norm;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::linalg::spd_cholesky_lower;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::QuantError;
use crate::error::check_grid_shape;
use crate::pricing::digital::expired;
use crate::pricing::mc_stats::off_domain;
use crate::pricing::mc_stats::par_estimate;
use crate::pricing::mc_stats::pathless;
use crate::pricing::spread::assert_two_asset_model;
use crate::traits::RealExt;

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

impl RainbowPayoff {
  /// The payoff on one terminal-price vector, NaN if a leg is (`f64::max` would price `CallOnMax`
  /// on `[120, NaN, 90]` at `20.0`); an empty slice gives `0.0`, which no Monte Carlo call reaches.
  pub fn evaluate(&self, prices: &[f64], k: f64) -> f64 {
    let max_p = f64::max_or_nan_fold(prices.iter().copied());
    let min_p = f64::min_or_nan_fold(prices.iter().copied());
    match self {
      RainbowPayoff::CallOnMax => (max_p - k).max_or_nan(0.0),
      RainbowPayoff::CallOnMin => (min_p - k).max_or_nan(0.0),
      RainbowPayoff::PutOnMax => (k - max_p).max_or_nan(0.0),
      RainbowPayoff::PutOnMin => (k - min_p).max_or_nan(0.0),
    }
  }
}

/// Stulz (1982) closed form for a two-asset best-of or worst-of option; the payoff is the contract
/// and its call or put side, as two assets fit no [`ModelPricer`](crate::traits::ModelPricer).
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
  /// Panics unless `sigma1, sigma2 ≥ 0` and `rho ∈ [-1, 1]`: a negative vol prices negative calls,
  /// and a bad `rho` would trip an anonymous `owens_t::biv_norm` assert or skew the Margrabe legs.
  pub fn new(payoff: RainbowPayoff, sigma1: f64, sigma2: f64, rho: f64) -> Self {
    assert_two_asset_model(sigma1, sigma2, rho);
    Self {
      payoff,
      sigma1,
      sigma2,
      rho,
    }
  }

  /// The contract at one query point: its payoff at `tau = 0`, NaN past expiry.
  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
  pub fn price(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    if let Some(value) = expired(tau, &[r, q1, q2], || self.payoff.evaluate(&[s1, s2], k)) {
      return value;
    }
    match self.payoff {
      RainbowPayoff::CallOnMin => self.call_on_min(s1, s2, k, r, q1, q2, tau),
      RainbowPayoff::CallOnMax => self.call_on_max(s1, s2, k, r, q1, q2, tau),
      RainbowPayoff::PutOnMin => self.put_on_min(s1, s2, k, r, q1, q2, tau),
      RainbowPayoff::PutOnMax => self.put_on_max(s1, s2, k, r, q1, q2, tau),
    }
  }

  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
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

  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
  fn call_on_max(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    use crate::pricing::bsm::BsmCoc;
    use crate::pricing::bsm::BsmPricer;
    use crate::traits::ModelPricer;

    // Stulz identity: max(max(S1,S2) - K, 0) = call(S1, K) + call(S2, K)
    // - call_on_min(S1, S2, K)
    let c1 = BsmPricer::new(self.sigma1, BsmCoc::Merton1973).price_call(s1, k, r, q1, tau);
    let c2 = BsmPricer::new(self.sigma2, BsmCoc::Merton1973).price_call(s2, k, r, q2, tau);
    c1 + c2 - self.call_on_min(s1, s2, k, r, q1, q2, tau)
  }

  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
  fn put_on_min(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    // Min-option parity, `P_min = C_min + Ke^{−rT} − F_min`, `F_min` the discounted `E[min]`.
    let call_min = self.call_on_min(s1, s2, k, r, q1, q2, tau);
    // `F_max − F_min` is the Margrabe expected payoff `E[(S₁ − S₂)⁺] + E[(S₂ − S₁)⁺]`.
    use crate::pricing::spread::MargrabePricer;
    let m12 = MargrabePricer::new(self.sigma1, self.sigma2, self.rho).price(s1, s2, q1, q2, tau);
    let m21 = MargrabePricer::new(self.sigma2, self.sigma1, self.rho).price(s2, s1, q2, q1, tau);
    // F_max + F_min = s1 e^{-q1T} + s2 e^{-q2T}, F_max - F_min = m12 + m21,
    // so F_min = (s1 e^{-q1T} + s2 e^{-q2T} - (m12 + m21)) / 2.
    let f_min = 0.5 * (s1 * (-q1 * tau).exp() + s2 * (-q2 * tau).exp() - (m12 + m21));
    call_min + k * (-r * tau).exp() - f_min
  }

  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
  fn put_on_max(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    let call_max = self.call_on_max(s1, s2, k, r, q1, q2, tau);
    use crate::pricing::spread::MargrabePricer;
    let m12 = MargrabePricer::new(self.sigma1, self.sigma2, self.rho).price(s1, s2, q1, q2, tau);
    let m21 = MargrabePricer::new(self.sigma2, self.sigma1, self.rho).price(s2, s1, q2, q1, tau);
    let f_max = 0.5 * (s1 * (-q1 * tau).exp() + s2 * (-q2 * tau).exp() + (m12 + m21));
    call_max + k * (-r * tau).exp() - f_max
  }
}

/// One volatility per asset, each finite and `>= 0`.
#[track_caller]
pub(crate) fn assert_volatilities(sigma: &Array1<f64>) {
  for (i, &v) in sigma.iter().enumerate() {
    assert!(
      v.is_finite() && v >= 0.0,
      "sigma[{i}] must satisfy `sigma[{i}].is_finite() && sigma[{i}] >= 0.0`, got sigma[{i}] = {v:?}"
    );
  }
}

/// Monte Carlo rainbow pricer on `n` correlated Gbm assets through a Cholesky factor; the shape and
/// positive-definiteness checks stay in [`try_price`](Self::try_price), its only way to an `Err`.
#[derive(Debug, Clone)]
pub struct McRainbowPricer<S: SeedExt = Unseeded> {
  /// Payoff type — a term of the contract, not a market quote.
  pub payoff: RainbowPayoff,
  /// Annualised Black–Scholes volatility of each asset.
  sigma: Array1<f64>,
  /// Correlation matrix.
  pub rho: Array2<f64>,
  /// Number of MC paths.
  n_paths: usize,
  /// Each call draws from a clone, so a `Deterministic` seed replays.
  pub seed: S,
}

impl McRainbowPricer {
  /// Panics unless every `sigma[i]` is finite and `>= 0` and `n_paths >= 1`; `rho` and the
  /// dimensions are data, which [`try_price`](Self::try_price) reports as an `Err`.
  pub fn new(payoff: RainbowPayoff, sigma: Array1<f64>, rho: Array2<f64>, n_paths: usize) -> Self {
    assert_volatilities(&sigma);
    assert!(
      n_paths >= 1,
      "n_paths must satisfy `n_paths >= 1`, got n_paths = {n_paths:?}"
    );
    Self {
      payoff,
      sigma,
      rho,
      n_paths,
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> McRainbowPricer<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> McRainbowPricer<S2> {
    McRainbowPricer {
      payoff: self.payoff,
      sigma: self.sigma,
      rho: self.rho,
      n_paths: self.n_paths,
      seed,
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
  ) -> Result<McEstimate<f64>, QuantError> {
    let n_assets = s.len();
    check_grid_shape(
      ["rho rows, one per asset", "rho columns, one per asset"],
      (n_assets, n_assets),
      self.rho.dim(),
    )?;
    for (what, got) in [
      ("sigma, one per asset", self.sigma.len()),
      ("q, one per asset", q.len()),
    ] {
      if got != n_assets {
        return Err(QuantError::DimensionMismatch {
          what,
          expected: n_assets,
          got,
        });
      }
    }
    if !crate::linalg::is_spd_t(self.rho.view()) {
      return Err(QuantError::NotPositiveDefinite {
        what: "the correlation matrix rho",
      });
    }
    Ok(self.price(s, k, r, q, tau))
  }

  /// The contract at one query point: NaN for a NaN query, `tau < 0` or, before expiry, a spot or
  /// strike not positive; panics on a `rho` not positive definite, which `try_price` reports.
  pub fn price(
    &self,
    s: ArrayView1<'_, f64>,
    k: f64,
    r: f64,
    q: ArrayView1<'_, f64>,
    tau: f64,
  ) -> McEstimate<f64> {
    let positive = s.iter().copied().chain([k]).collect::<Vec<f64>>();
    let others = q.iter().copied().chain([r]).collect::<Vec<f64>>();
    if off_domain(&positive, &others, tau) {
      return pathless(f64::NAN);
    }
    let n_assets = s.len();
    let l: Array2<f64> = spd_cholesky_lower(self.rho.view()).expect(
      "correlation matrix must be positive definite — call try_price() to handle this gracefully",
    );
    let drifts: Vec<f64> = (0..n_assets)
      .map(|i| (r - q[i] - 0.5 * self.sigma[i] * self.sigma[i]) * tau)
      .collect();
    let vols: Vec<f64> = (0..n_assets).map(|i| self.sigma[i] * tau.sqrt()).collect();
    let n_paths = self.n_paths;

    let mut all_z = vec![0.0_f64; n_paths * n_assets];
    let seed = self.seed.clone();
    SimdNormal::<f64>::new(0.0, 1.0)
      .seeded(&seed)
      .fill_slice(&mut all_z);

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
    par_estimate(n_paths, (-r * tau).exp(), payoff_of)
  }
}

#[cfg(test)]
#[path = "rainbow_tests.rs"]
mod tests;
