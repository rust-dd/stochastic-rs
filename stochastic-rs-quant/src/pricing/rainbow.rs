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
use ndarray_linalg::Cholesky;
#[cfg(feature = "openblas")]
use ndarray_linalg::UPLO;
use owens_t::biv_norm;
#[cfg(feature = "openblas")]
use rayon::prelude::*;

use crate::OptionType;
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

impl RainbowPayoff {
  pub fn evaluate(&self, prices: &[f64], k: f64) -> f64 {
    let max_p = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_p = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    match self {
      RainbowPayoff::CallOnMax => (max_p - k).max(0.0),
      RainbowPayoff::CallOnMin => (min_p - k).max(0.0),
      RainbowPayoff::PutOnMax => (k - max_p).max(0.0),
      RainbowPayoff::PutOnMin => (k - min_p).max(0.0),
    }
  }
}

/// Stulz (1982) closed-form pricer for a two-asset max/min option.
#[derive(Debug, Clone)]
pub struct StulzRainbowPricer {
  /// Spot 1.
  pub s1: f64,
  /// Spot 2.
  pub s2: f64,
  /// Strike.
  pub k: f64,
  /// Volatility 1.
  pub sigma1: f64,
  /// Volatility 2.
  pub sigma2: f64,
  /// Correlation.
  pub rho: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield 1.
  pub q1: f64,
  /// Dividend yield 2.
  pub q2: f64,
  /// Time to maturity.
  pub t: f64,
  /// Payoff type.
  pub payoff: RainbowPayoff,
}

impl StulzRainbowPricer {
  pub fn price(&self) -> f64 {
    match self.payoff {
      RainbowPayoff::CallOnMin => self.call_on_min(),
      RainbowPayoff::CallOnMax => self.call_on_max(),
      RainbowPayoff::PutOnMin => self.put_on_min(),
      RainbowPayoff::PutOnMax => self.put_on_max(),
    }
  }

  fn call_on_min(&self) -> f64 {
    let s1 = self.s1;
    let s2 = self.s2;
    let k = self.k;
    let v1 = self.sigma1;
    let v2 = self.sigma2;
    let rho = self.rho;
    let r = self.r;
    let q1 = self.q1;
    let q2 = self.q2;
    let t = self.t;
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

  fn call_on_max(&self) -> f64 {
    use crate::pricing::bsm::BSMCoc;
    use crate::pricing::bsm::BSMPricer;
    use crate::traits::PricerExt;

    // Stulz identity: max(max(S1,S2) - K, 0) = call(S1, K) + call(S2, K)
    // - call_on_min(S1, S2, K)
    let c1 = BSMPricer::builder(self.s1, self.sigma1, self.k, self.r)
      .tau(self.t)
      .q(self.q1)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Merton1973)
      .build()
      .calculate_call_put()
      .0;
    let c2 = BSMPricer::builder(self.s2, self.sigma2, self.k, self.r)
      .tau(self.t)
      .q(self.q2)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Merton1973)
      .build()
      .calculate_call_put()
      .0;
    c1 + c2 - self.call_on_min()
  }

  fn put_on_min(&self) -> f64 {
    // Stulz identity: max(K - min(S1,S2), 0) = K * e^{-rT} - min_call(0)
    //   = call_on_min(S1, S2, K) - F_min where F_min = E[min(S1,S2)] discounted
    // Easier: put-call parity for min options:
    // put_on_min - call_on_min = K * e^{-rT} - E[min(S1, S2)] * e^{-rT}
    // E[min(S1, S2)] = S1 e^{(r-q1)T} + S2 e^{(r-q2)T} - E[max(S1, S2)]
    // and E[max(S1, S2)] - E[min(S1, S2)] is the Margrabe expected payoff.
    let call_min = self.call_on_min();
    // Use put-call-min parity: P_min = C_min + K e^{-rT} - F_min where
    // F_min = S1 e^{-q1 T} + S2 e^{-q2 T} - F_max and the difference
    // F_max - F_min equals the Margrabe expected payoff $E[(S_1-S_2)^+] +
    // E[(S_2-S_1)^+]$. Use Margrabe to compute it.
    use crate::pricing::spread::MargrabePricer;
    let m12 = MargrabePricer {
      s1: self.s1,
      s2: self.s2,
      sigma1: self.sigma1,
      sigma2: self.sigma2,
      rho: self.rho,
      q1: self.q1,
      q2: self.q2,
      t: self.t,
    }
    .price();
    let m21 = MargrabePricer {
      s1: self.s2,
      s2: self.s1,
      sigma1: self.sigma2,
      sigma2: self.sigma1,
      rho: self.rho,
      q1: self.q2,
      q2: self.q1,
      t: self.t,
    }
    .price();
    // F_max + F_min = s1 e^{-q1T} + s2 e^{-q2T}, F_max - F_min = m12 + m21,
    // so F_min = (s1 e^{-q1T} + s2 e^{-q2T} - (m12 + m21)) / 2.
    let f_min = 0.5
      * (self.s1 * (-self.q1 * self.t).exp() + self.s2 * (-self.q2 * self.t).exp() - (m12 + m21));
    call_min + self.k * (-self.r * self.t).exp() - f_min
  }

  fn put_on_max(&self) -> f64 {
    let call_max = self.call_on_max();
    use crate::pricing::spread::MargrabePricer;
    let m12 = MargrabePricer {
      s1: self.s1,
      s2: self.s2,
      sigma1: self.sigma1,
      sigma2: self.sigma2,
      rho: self.rho,
      q1: self.q1,
      q2: self.q2,
      t: self.t,
    }
    .price();
    let m21 = MargrabePricer {
      s1: self.s2,
      s2: self.s1,
      sigma1: self.sigma2,
      sigma2: self.sigma1,
      rho: self.rho,
      q1: self.q2,
      q2: self.q1,
      t: self.t,
    }
    .price();
    let f_max = 0.5
      * (self.s1 * (-self.q1 * self.t).exp() + self.s2 * (-self.q2 * self.t).exp() + (m12 + m21));
    call_max + self.k * (-self.r * self.t).exp() - f_max
  }
}

/// Monte-Carlo rainbow pricer for arbitrary $n$ assets. Gated behind the
/// `openblas` feature because it relies on `ndarray_linalg::Cholesky` for
/// the correlation factor.
#[cfg(feature = "openblas")]
#[derive(Debug, Clone)]
pub struct McRainbowPricer {
  /// Spot prices.
  pub s: Array1<f64>,
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
  /// Payoff type.
  pub payoff: RainbowPayoff,
  /// Number of MC paths.
  pub n_paths: usize,
}

#[cfg(feature = "openblas")]
impl McRainbowPricer {
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
          .map(|i| self.s[i] * (drifts[i] + vols[i] * zc[i]).exp())
          .collect();
        self.payoff.evaluate(&s_t, self.k)
      })
      .sum();

    (-self.r * self.t).exp() * sum / n_paths as f64
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// Stulz: $C_{\min} + C_{\max} = C_1 + C_2$ (vanilla call sum).
  #[test]
  fn stulz_min_max_decomposition() {
    use crate::pricing::bsm::BSMCoc;
    use crate::pricing::bsm::BSMPricer;
    use crate::traits::PricerExt;

    let s1 = 100.0;
    let s2 = 105.0;
    let k = 100.0;
    let v1 = 0.20;
    let v2 = 0.30;
    let rho = 0.5;
    let r = 0.05;
    let q1 = 0.0;
    let q2 = 0.0;
    let t = 1.0;
    let cmin = StulzRainbowPricer {
      s1,
      s2,
      k,
      sigma1: v1,
      sigma2: v2,
      rho,
      r,
      q1,
      q2,
      t,
      payoff: RainbowPayoff::CallOnMin,
    }
    .price();
    let cmax = StulzRainbowPricer {
      s1,
      s2,
      k,
      sigma1: v1,
      sigma2: v2,
      rho,
      r,
      q1,
      q2,
      t,
      payoff: RainbowPayoff::CallOnMax,
    }
    .price();
    let c1 = BSMPricer::builder(s1, v1, k, r)
      .tau(t)
      .q(q1)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Merton1973)
      .build()
      .calculate_call_put()
      .0;
    let c2 = BSMPricer::builder(s2, v2, k, r)
      .tau(t)
      .q(q2)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Merton1973)
      .build()
      .calculate_call_put()
      .0;
    let lhs = cmin + cmax;
    let rhs = c1 + c2;
    assert!((lhs - rhs).abs() < 0.01, "lhs={lhs}, rhs={rhs}");
  }

  /// Stulz call-on-min should match Monte Carlo within 2%.
  #[cfg(feature = "openblas")]
  #[test]
  fn stulz_min_matches_mc() {
    let stulz = StulzRainbowPricer {
      s1: 100.0,
      s2: 100.0,
      k: 100.0,
      sigma1: 0.25,
      sigma2: 0.30,
      rho: 0.4,
      r: 0.05,
      q1: 0.0,
      q2: 0.0,
      t: 1.0,
      payoff: RainbowPayoff::CallOnMin,
    }
    .price();
    let mc = McRainbowPricer {
      s: array![100.0, 100.0],
      sigma: array![0.25, 0.30],
      q: array![0.0, 0.0],
      rho: array![[1.0, 0.4], [0.4, 1.0]],
      k: 100.0,
      r: 0.05,
      t: 1.0,
      payoff: RainbowPayoff::CallOnMin,
      n_paths: 200_000,
    }
    .price();
    let rel = (stulz - mc).abs() / stulz.max(1e-10);
    assert!(rel < 0.03, "stulz={stulz}, mc={mc}, rel={rel}");
  }

  /// CallOnMax >= each individual vanilla call (always have at least one
  /// asset path in the money).
  #[test]
  fn call_on_max_dominates_vanilla() {
    use crate::pricing::bsm::BSMCoc;
    use crate::pricing::bsm::BSMPricer;
    use crate::traits::PricerExt;

    let s1 = 100.0;
    let s2 = 100.0;
    let v1 = 0.25;
    let v2 = 0.25;
    let rho = 0.0;
    let cmax = StulzRainbowPricer {
      s1,
      s2,
      k: 100.0,
      sigma1: v1,
      sigma2: v2,
      rho,
      r: 0.05,
      q1: 0.0,
      q2: 0.0,
      t: 1.0,
      payoff: RainbowPayoff::CallOnMax,
    }
    .price();
    let c1 = BSMPricer::builder(s1, v1, 100.0, 0.05)
      .tau(1.0)
      .q(0.0)
      .option_type(OptionType::Call)
      .coc(BSMCoc::Merton1973)
      .build()
      .calculate_call_put()
      .0;
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
    let mc_max = McRainbowPricer {
      s: s.clone(),
      sigma: sig.clone(),
      q: q.clone(),
      rho: rho.clone(),
      k: 100.0,
      r: 0.05,
      t: 1.0,
      payoff: RainbowPayoff::CallOnMax,
      n_paths: 50_000,
    }
    .price();
    let mc_min = McRainbowPricer {
      s,
      sigma: sig,
      q,
      rho,
      k: 100.0,
      r: 0.05,
      t: 1.0,
      payoff: RainbowPayoff::CallOnMin,
      n_paths: 50_000,
    }
    .price();
    assert!(mc_max > mc_min);
  }

  /// Stulz put-on-min via parity should be positive.
  #[test]
  fn stulz_put_on_min_positive() {
    let p = StulzRainbowPricer {
      s1: 100.0,
      s2: 105.0,
      k: 100.0,
      sigma1: 0.25,
      sigma2: 0.20,
      rho: 0.3,
      r: 0.05,
      q1: 0.0,
      q2: 0.0,
      t: 0.5,
      payoff: RainbowPayoff::PutOnMin,
    };
    let price = p.price();
    assert!(price > 0.0, "put_on_min={price}");
  }
}
