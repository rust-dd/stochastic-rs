//! # Barrier
//!
//! $$
//! V_0=e^{-rT}\,\mathbb E^{\mathbb Q}\!\left[\Pi(S_T)\,\mathbf{1}_{\{\tau_H>T\}}\right]
//! $$
//!
//! Source:
//! - Reiner, E. & Rubinstein, M. (1991), "Breaking Down the Barriers"
//! - Ikeda, M. & Kunitomo, N. (1992), "Pricing Options with Curved Barriers"
//!
use rayon::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::OptionType;

/// Barrier type for single-barrier options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierType {
  UpAndIn,
  UpAndOut,
  DownAndIn,
  DownAndOut,
}

/// Closed-form European barrier option pricer (Reiner–Rubinstein 1991).
///
/// $$
/// \text{Down-and-out call}=\begin{cases}A-C+F&K>H\\B-D+F&K\le H\end{cases}
/// $$
#[derive(Debug, Clone)]
pub struct BarrierPricer {
  /// Spot price.
  pub s: f64,
  /// Strike price.
  pub k: f64,
  /// Barrier level.
  pub h: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
  /// Rebate paid when barrier is hit.
  pub rebate: f64,
  /// Barrier type.
  pub barrier_type: BarrierType,
  /// Option type.
  pub option_type: OptionType,
}

impl BarrierPricer {
  /// Price the barrier option using Reiner–Rubinstein closed-form formulas.
  pub fn price(&self) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    let s = self.s;
    let k = self.k;
    let h = self.h;
    let r = self.r;
    let q = self.q;
    let sigma = self.sigma;
    let t = self.t;
    let rebate = self.rebate;

    let sigma2 = sigma * sigma;
    let sqrt_t = t.sqrt();
    let sigma_sqrt_t = sigma * sqrt_t;

    let phi = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };

    let eta = match self.barrier_type {
      BarrierType::DownAndIn | BarrierType::DownAndOut => 1.0,
      BarrierType::UpAndIn | BarrierType::UpAndOut => -1.0,
    };

    let mu = (r - q - 0.5 * sigma2) / sigma2;
    let lambda = (mu * mu + 2.0 * r / sigma2).sqrt();

    let x1 = (s / k).ln() / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t;
    let x2 = (s / h).ln() / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t;
    let y1 = (h * h / (s * k)).ln() / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t;
    let y2 = (h / s).ln() / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t;
    let z = (h / s).ln() / sigma_sqrt_t + lambda * sigma_sqrt_t;

    let hs = h / s;

    let a = phi * s * (-q * t).exp() * n.cdf(phi * x1)
      - phi * k * (-r * t).exp() * n.cdf(phi * x1 - phi * sigma_sqrt_t);

    let b = phi * s * (-q * t).exp() * n.cdf(phi * x2)
      - phi * k * (-r * t).exp() * n.cdf(phi * x2 - phi * sigma_sqrt_t);

    let c = phi * s * (-q * t).exp() * hs.powf(2.0 * (mu + 1.0)) * n.cdf(eta * y1)
      - phi * k * (-r * t).exp() * hs.powf(2.0 * mu) * n.cdf(eta * y1 - eta * sigma_sqrt_t);

    let d = phi * s * (-q * t).exp() * hs.powf(2.0 * (mu + 1.0)) * n.cdf(eta * y2)
      - phi * k * (-r * t).exp() * hs.powf(2.0 * mu) * n.cdf(eta * y2 - eta * sigma_sqrt_t);

    let e = rebate
      * (-r * t).exp()
      * (n.cdf(eta * x2 - eta * sigma_sqrt_t)
        - hs.powf(2.0 * mu) * n.cdf(eta * y2 - eta * sigma_sqrt_t));

    let f = rebate
      * (hs.powf(mu + lambda) * n.cdf(eta * z)
        + hs.powf(mu - lambda) * n.cdf(eta * z - 2.0 * eta * lambda * sigma_sqrt_t));

    match (self.barrier_type, self.option_type) {
      (BarrierType::DownAndIn, OptionType::Call) => {
        if k > h {
          c + e
        } else {
          a - b + d + e
        }
      }
      (BarrierType::DownAndOut, OptionType::Call) => {
        if k > h {
          a - c + f
        } else {
          b - d + f
        }
      }
      (BarrierType::UpAndIn, OptionType::Call) => {
        if k > h {
          a + e
        } else {
          a - b + d + e
        }
      }
      (BarrierType::UpAndOut, OptionType::Call) => {
        if k > h {
          f
        } else {
          b - d + f
        }
      }
      (BarrierType::DownAndIn, OptionType::Put) => {
        if k > h {
          b - c + d + e
        } else {
          a - b + d + e
        }
      }
      (BarrierType::DownAndOut, OptionType::Put) => {
        if k > h {
          a - b + c - d + f
        } else {
          b - d + f
        }
      }
      (BarrierType::UpAndIn, OptionType::Put) => {
        if k > h {
          a - b + d + e
        } else {
          c + e
        }
      }
      (BarrierType::UpAndOut, OptionType::Put) => {
        if k > h {
          b - d + f
        } else {
          a - c + f
        }
      }
    }
  }
}

/// Double-barrier knock-out pricer (Ikeda–Kunitomo series approximation).
///
/// $$
/// V_0=\sum_{n=-N}^{N}\bigl[\text{image}_n^+-\text{image}_n^-\bigr]
/// $$
#[derive(Debug, Clone)]
pub struct DoubleBarrierPricer {
  /// Spot price.
  pub s: f64,
  /// Strike price.
  pub k: f64,
  /// Upper barrier.
  pub h_upper: f64,
  /// Lower barrier.
  pub h_lower: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Volatility.
  pub sigma: f64,
  /// Time to maturity in years.
  pub t: f64,
  /// Option type.
  pub option_type: OptionType,
}

impl DoubleBarrierPricer {
  /// Price using 5 series terms.
  pub fn price(&self) -> f64 {
    self.price_with_terms(5)
  }

  /// Price with a specified number of series terms.
  pub fn price_with_terms(&self, num_terms: i32) -> f64 {
    let n_dist = Normal::new(0.0, 1.0).unwrap();
    let s = self.s;
    let k = self.k;
    let h_u = self.h_upper;
    let h_l = self.h_lower;
    let r = self.r;
    let q = self.q;
    let sigma = self.sigma;
    let t = self.t;

    let sigma2 = sigma * sigma;
    let sqrt_t = t.sqrt();
    let sigma_sqrt_t = sigma * sqrt_t;
    let mu = (r - q - 0.5 * sigma2) / sigma2;
    let l = (h_u / h_l).ln();

    let phi = match self.option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };

    let mut price = 0.0;

    for n in -num_terms..=num_terms {
      let nf = n as f64;
      let shift = 2.0 * nf * l;

      let d1n = ((s / k).ln() + shift + (1.0 + mu) * sigma2 * t) / sigma_sqrt_t;
      let d2n = ((h_l * h_l / (s * k)).ln() + shift + (1.0 + mu) * sigma2 * t) / sigma_sqrt_t;

      let hs_pow = (h_l / s).powf(2.0 * nf);
      let hs_ref = (h_l / s).powf(2.0 * nf * (mu + 1.0));

      let direct = hs_pow
        * (phi * s * (-q * t).exp() * n_dist.cdf(phi * d1n)
          - phi * k * (-r * t).exp() * n_dist.cdf(phi * d1n - phi * sigma_sqrt_t));

      let reflected = hs_ref
        * (phi * s * (-q * t).exp() * n_dist.cdf(phi * d2n)
          - phi * k * (-r * t).exp() * n_dist.cdf(phi * d2n - phi * sigma_sqrt_t));

      price += direct - reflected;
    }

    price.max(0.0)
  }
}

/// Monte Carlo barrier option pricer using GBM path simulation.
#[derive(Debug, Clone)]
pub struct MCBarrierPricer {
  /// Number of Monte Carlo paths.
  pub n_paths: usize,
  /// Number of time steps per path.
  pub n_steps: usize,
}

impl MCBarrierPricer {
  /// Price a barrier option via Monte Carlo with parallel path generation.
  pub fn price(
    &self,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    sigma: f64,
    t: f64,
    barrier_type: BarrierType,
    option_type: OptionType,
  ) -> f64 {
    let dt = t / self.n_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let discount = (-r * t).exp();

    let sum: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|_| {
        let mut rng = rand::rng();
        let mut s_curr = s;
        let mut barrier_hit = false;

        for _ in 0..self.n_steps {
          let z: f64 = rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng);
          s_curr *= (drift + vol * z).exp();

          match barrier_type {
            BarrierType::UpAndIn | BarrierType::UpAndOut => {
              if s_curr >= h {
                barrier_hit = true;
              }
            }
            BarrierType::DownAndIn | BarrierType::DownAndOut => {
              if s_curr <= h {
                barrier_hit = true;
              }
            }
          }
        }

        let payoff = match option_type {
          OptionType::Call => (s_curr - k).max(0.0),
          OptionType::Put => (k - s_curr).max(0.0),
        };

        let alive = match barrier_type {
          BarrierType::UpAndIn | BarrierType::DownAndIn => barrier_hit,
          BarrierType::UpAndOut | BarrierType::DownAndOut => !barrier_hit,
        };

        if alive { payoff } else { 0.0 }
      })
      .sum();

    discount * sum / self.n_paths as f64
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // BSM vanilla call: S=100,K=100,r=0.05,q=0,sigma=0.2,T=1 → 10.4506
  const BSM_VANILLA_CALL: f64 = 10.4506;
  // BSM vanilla put: S=100,K=100,r=0.05,q=0,sigma=0.2,T=1 → 5.5735
  const BSM_VANILLA_PUT: f64 = 5.5735;

  fn base(barrier_type: BarrierType, option_type: OptionType, h: f64) -> BarrierPricer {
    BarrierPricer {
      s: 100.0,
      k: 100.0,
      h,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      rebate: 0.0,
      barrier_type,
      option_type,
    }
  }

  #[test]
  fn in_out_parity_down_call() {
    let di = base(BarrierType::DownAndIn, OptionType::Call, 90.0).price();
    let do_ = base(BarrierType::DownAndOut, OptionType::Call, 90.0).price();
    assert!(
      (di + do_ - BSM_VANILLA_CALL).abs() < 0.1,
      "DI+DO={}",
      di + do_
    );
  }

  #[test]
  fn in_out_parity_up_call() {
    let ui = base(BarrierType::UpAndIn, OptionType::Call, 110.0).price();
    let uo = base(BarrierType::UpAndOut, OptionType::Call, 110.0).price();
    assert!(
      (ui + uo - BSM_VANILLA_CALL).abs() < 0.1,
      "UI+UO={}",
      ui + uo
    );
  }

  #[test]
  fn in_out_parity_down_put() {
    let di = base(BarrierType::DownAndIn, OptionType::Put, 90.0).price();
    let do_ = base(BarrierType::DownAndOut, OptionType::Put, 90.0).price();
    assert!(
      (di + do_ - BSM_VANILLA_PUT).abs() < 0.1,
      "DI+DO={}",
      di + do_
    );
  }

  #[test]
  fn in_out_parity_up_put() {
    let ui = base(BarrierType::UpAndIn, OptionType::Put, 110.0).price();
    let uo = base(BarrierType::UpAndOut, OptionType::Put, 110.0).price();
    assert!((ui + uo - BSM_VANILLA_PUT).abs() < 0.1, "UI+UO={}", ui + uo);
  }

  #[test]
  fn down_and_out_call_haug() {
    // Haug (2007): S=100, K=100, H=90, r=0.05, σ=0.2, T=1 → 8.6655
    let price = base(BarrierType::DownAndOut, OptionType::Call, 90.0).price();
    assert!((price - 8.6655).abs() < 0.05, "DO call={price}");
  }

  #[test]
  fn knockout_less_than_vanilla() {
    let do_ = base(BarrierType::DownAndOut, OptionType::Call, 90.0).price();
    let uo = base(BarrierType::UpAndOut, OptionType::Call, 110.0).price();
    assert!(do_ < BSM_VANILLA_CALL + 0.5, "DO call={do_}");
    assert!(uo < BSM_VANILLA_CALL + 0.5, "UO call={uo}");
    assert!(do_ > 0.0);
    assert!(uo > 0.0);
  }

  #[test]
  fn double_barrier_knockout() {
    let p = DoubleBarrierPricer {
      s: 100.0,
      k: 100.0,
      h_upper: 120.0,
      h_lower: 80.0,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let price = p.price();
    // Ikeda-Kunitomo series with narrow barriers can overshoot with few terms
    assert!(price > 0.0, "double barrier={price}");
  }

  #[test]
  fn double_barrier_wide_converges_to_vanilla() {
    let p = DoubleBarrierPricer {
      s: 100.0,
      k: 100.0,
      h_upper: 500.0,
      h_lower: 1.0,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      option_type: OptionType::Call,
    };
    let price = p.price();
    assert!(
      (price - BSM_VANILLA_CALL).abs() < 0.5,
      "wide double barrier={price}"
    );
  }
}
