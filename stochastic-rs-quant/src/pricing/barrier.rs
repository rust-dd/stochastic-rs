//! Barrier options: Reiner–Rubinstein for one barrier, Ikeda–Kunitomo for two, and Monte Carlo.
//!
//! $$
//! V_0=e^{-rT}\,\mathbb E^{\mathbb Q}\!\left[\Pi(S_T)\,\mathbf{1}_{\{\tau_H>T\}}\right]
//! $$
//!
//! Reference: Reiner, E. & Rubinstein, M. (1991), "Breaking Down the Barriers", *Risk* 4(8), 28–35
//! Reference: Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", ISBN 0071389970
//! Reference: Vector Risk (2017), "Single Barrier Option", v8.0.7905, p. 2: Haug's §4.17.1, p. 152
//! Reference: Ikeda, M. & Kunitomo, N. (1992), "Pricing Options with Curved Boundaries"
//!
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ln_norm_cdf;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::OptionType;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::exotic::BarrierOption;
use crate::pricing::exotic::DoubleBarrierOption;
use crate::pricing::exotic::assert_sigma;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::ModelPricer;

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarrierPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl BarrierPricer {
  /// Panics unless `sigma` is finite and `>= 0`; at `0` an unbreached barrier is NaN before expiry.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// Reiner–Rubinstein price; a spot on or past the barrier has knocked out to the rebate or in to
  /// the vanilla, `tau = 0` pays out, and a negative or NaN `tau` or a NaN barrier is NaN.
  pub fn price(&self, option: &BarrierOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let BarrierOption {
      strike: k,
      barrier: h,
      rebate,
      barrier_type,
      option_type,
    } = *option;
    if !(s > 0.0 && tau >= 0.0) || h.is_nan() || r.is_nan() || q.is_nan() {
      return f64::NAN;
    }
    let knock_in = matches!(barrier_type, BarrierType::DownAndIn | BarrierType::UpAndIn);
    let breached = match barrier_type {
      BarrierType::DownAndIn | BarrierType::DownAndOut => s <= h,
      BarrierType::UpAndIn | BarrierType::UpAndOut => s >= h,
    };
    match (breached, knock_in) {
      (true, false) => return rebate,
      (true, true) => {
        let vanilla = BsmPricer::new(self.sigma, BsmCoc::Merton1973);
        return vanilla.price_option(s, k, r, q, tau, option_type);
      }
      (false, false) if tau == 0.0 => return vanilla_payoff(option_type, s, k),
      (false, true) if tau == 0.0 => return rebate,
      _ => {}
    }
    let sigma = self.sigma;

    let sigma2 = sigma * sigma;
    let sqrt_t = tau.sqrt();
    let sigma_sqrt_t = sigma * sqrt_t;

    let phi = match option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };

    let eta = match barrier_type {
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

    let a = phi * s * (-q * tau).exp() * norm_cdf(phi * x1)
      - phi * k * (-r * tau).exp() * norm_cdf(phi * x1 - phi * sigma_sqrt_t);

    let b = phi * s * (-q * tau).exp() * norm_cdf(phi * x2)
      - phi * k * (-r * tau).exp() * norm_cdf(phi * x2 - phi * sigma_sqrt_t);

    let c = phi * s * (-q * tau).exp() * hs.powf(2.0 * (mu + 1.0)) * norm_cdf(eta * y1)
      - phi * k * (-r * tau).exp() * hs.powf(2.0 * mu) * norm_cdf(eta * y1 - eta * sigma_sqrt_t);

    let d = phi * s * (-q * tau).exp() * hs.powf(2.0 * (mu + 1.0)) * norm_cdf(eta * y2)
      - phi * k * (-r * tau).exp() * hs.powf(2.0 * mu) * norm_cdf(eta * y2 - eta * sigma_sqrt_t);

    let e = rebate
      * (-r * tau).exp()
      * (norm_cdf(eta * x2 - eta * sigma_sqrt_t)
        - hs.powf(2.0 * mu) * norm_cdf(eta * y2 - eta * sigma_sqrt_t));

    let f = rebate
      * (hs.powf(mu + lambda) * norm_cdf(eta * z)
        + hs.powf(mu - lambda) * norm_cdf(eta * z - 2.0 * eta * lambda * sigma_sqrt_t));

    match (barrier_type, option_type) {
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
          b - c + d + e
        }
      }
      (BarrierType::UpAndOut, OptionType::Call) => {
        if k > h {
          f
        } else {
          a - b + c - d + f
        }
      }
      (BarrierType::DownAndIn, OptionType::Put) => {
        if k > h {
          b - c + d + e
        } else {
          a + e
        }
      }
      (BarrierType::DownAndOut, OptionType::Put) => {
        if k > h {
          a - b + c - d + f
        } else {
          f
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

/// Double knock-out on flat barriers: Kunitomo–Ikeda's series as Haug (2007, p. 156) gives it, the
/// strike bounded to the corridor by `α` and `β` (Vector Risk 2017); a put flips both legs.
///
/// $$
/// c=Se^{-qT}\sum_{n}\Bigl[\bigl(\tfrac{U^n}{L^n}\bigr)^{\mu_1}\bigl(N(d_1)-N(d_2)\bigr)
/// -\bigl(\tfrac{L^{n+1}}{U^nS}\bigr)^{\mu_1}\bigl(N(d_3)-N(d_4)\bigr)\Bigr]
/// -Ke^{-rT}\sum_{n}\Bigl[\bigl(\tfrac{U^n}{L^n}\bigr)^{\mu_1-2}\bigl(N(d_1')-N(d_2')\bigr)
/// -\bigl(\tfrac{L^{n+1}}{U^nS}\bigr)^{\mu_1-2}\bigl(N(d_3')-N(d_4')\bigr)\Bigr],
/// $$
/// $$
/// d_{1,2}=\frac{\ln\frac{SU^{2n}}{\{\alpha,\beta\}L^{2n}}+(b+\frac{\sigma^2}{2})T}
/// {\sigma\sqrt T},\quad
/// d_{3,4}=\frac{\ln\frac{L^{2n+2}}{\{\alpha,\beta\}SU^{2n}}+(b+\frac{\sigma^2}{2})T}
/// {\sigma\sqrt T},\quad
/// d'=d-\sigma\sqrt T,\quad \mu_1=\frac{2b}{\sigma^2}+1,\quad
/// (\alpha,\beta)=\begin{cases}(\max(K,L),\,U)&\text{call}\\(L,\,\min(K,U))&\text{put}\end{cases}
/// $$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleBarrierPricer {
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
}

impl DoubleBarrierPricer {
  /// Panics unless `sigma` is finite and `>= 0`; at `0` the deterministic path's payoff, NaN where
  /// that path ends on a bound.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// Price at `(s, r, q, tau)`: 0 on or outside the corridor, the payoff at `tau = 0`, and NaN for
  /// a negative or NaN `tau`, a NaN bound or rate, a spot `<= 0`, or a `lower <= 0` before expiry.
  pub fn price(&self, option: &DoubleBarrierOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let DoubleBarrierOption {
      strike: k,
      lower: l,
      upper: u,
      option_type,
    } = *option;
    if !(s > 0.0 && tau >= 0.0) || l.is_nan() || u.is_nan() || r.is_nan() || q.is_nan() {
      return f64::NAN;
    }
    if tau > 0.0 && l <= 0.0 {
      return f64::NAN;
    }
    if s <= l || s >= u {
      return 0.0;
    }
    if tau == 0.0 {
      return vanilla_payoff(option_type, s, k);
    }
    let discounted = |s_now: f64| vanilla_payoff(option_type, s_now, k * (-r * tau).exp());
    if self.sigma == 0.0 {
      let forward = s * ((r - q) * tau).exp();
      return if forward == l || forward == u {
        f64::NAN
      } else if forward < l || forward > u {
        0.0
      } else {
        discounted(s * (-q * tau).exp())
      };
    }
    let (alpha, beta) = match option_type {
      OptionType::Call => (k.max_or_nan(l), u),
      OptionType::Put => (l, k.min_or_nan(u)),
    };
    if alpha.is_nan() || beta.is_nan() {
      return f64::NAN;
    }
    if alpha >= beta {
      return 0.0;
    }
    self.image_series(option_type, (s, k, l, u), (alpha, beta), r, q, tau)
  }

  /// Haug's series with each power joined to its normal difference in logs, where neither can
  /// overflow, out to `n = ±N` for `N` below.
  fn image_series(
    &self,
    option_type: OptionType,
    (s, k, l, u): (f64, f64, f64, f64),
    (alpha, beta): (f64, f64),
    r: f64,
    q: f64,
    tau: f64,
  ) -> f64 {
    let sigma = self.sigma;
    let sd = sigma * tau.sqrt();
    let b = r - q;
    let mu1 = 2.0 * b / (sigma * sigma) + 1.0;
    let drift = (b + 0.5 * sigma * sigma) * tau / sd;
    let (ln_s, ln_l, ln_u) = (s.ln(), l.ln(), u.ln());
    let (ln_alpha, ln_beta) = (alpha.ln(), beta.ln());
    let width = ln_u - ln_l;
    let z = sd / width;
    if z > IMAGE_SPREAD_CUTOFF {
      return 0.0;
    }
    let n_max = (1.0 + 0.5 * IMAGE_DECAY * z * z).sqrt().ceil() as i32;
    let (mut asset, mut cash) = (0.0, 0.0);
    for n in -n_max..=n_max {
      let n = f64::from(n);
      let shift = 2.0 * n * width;
      let mirror = 2.0 * (n + 1.0) * ln_l - 2.0 * n * ln_u - ln_s;
      let d1 = (ln_s - ln_alpha + shift) / sd + drift;
      let d2 = (ln_s - ln_beta + shift) / sd + drift;
      let d3 = (mirror - ln_alpha) / sd + drift;
      let d4 = (mirror - ln_beta) / sd + drift;
      let direct = n * width;
      let reflected = (n + 1.0) * ln_l - n * ln_u - ln_s;
      asset += weighted(mu1 * direct, d1, d2) - weighted(mu1 * reflected, d3, d4);
      cash += weighted((mu1 - 2.0) * direct, d1 - sd, d2 - sd)
        - weighted((mu1 - 2.0) * reflected, d3 - sd, d4 - sd);
    }
    let (spot, strike) = (s * (-q * tau).exp(), k * (-r * tau).exp());
    let price = match option_type {
      OptionType::Call => spot * asset - strike * cash,
      OptionType::Put => strike * cash - spot * asset,
    };
    price.max_or_nan(0.0)
  }
}

/// `2(N + 1)(N − 1)/z² ≥ IMAGE_DECAY` leaves out only images below `e^{-40}` of the leading ones.
const IMAGE_DECAY: f64 = 40.0;

/// Past `z = σ√T / ln(U/L) = 27` the survival probability is below `2e^{-π²z²/2} < e^{-3590}`.
const IMAGE_SPREAD_CUTOFF: f64 = 27.0;

/// `e^{ln_coef} (N(x) − N(y))` for `x ≥ y`, joined in logs.
fn weighted(ln_coef: f64, x: f64, y: f64) -> f64 {
  if x == y {
    return 0.0;
  }
  (ln_coef + ln_norm_diff(x, y)).exp()
}

/// `ln(N(x) − N(y))` for `x ≥ y`, read from the tail where both terms are small.
fn ln_norm_diff(x: f64, y: f64) -> f64 {
  let (upper, lower) = if y >= 0.0 {
    (ln_norm_cdf(-y), ln_norm_cdf(-x))
  } else if x <= 0.0 {
    (ln_norm_cdf(x), ln_norm_cdf(y))
  } else {
    return (norm_cdf(x) - norm_cdf(y)).ln();
  };
  let gap = lower - upper;
  upper
    + if gap > -std::f64::consts::LN_2 {
      (-gap.exp_m1()).ln()
    } else {
      (-gap.exp()).ln_1p()
    }
}

/// Monte Carlo barrier option pricer using Gbm path simulation.
#[derive(Debug, Clone)]
pub struct McBarrierPricer {
  /// Number of Monte Carlo paths.
  pub n_paths: usize,
  /// Number of time steps per path.
  pub n_steps: usize,
}

impl McBarrierPricer {
  /// Price a barrier option via Monte Carlo with parallel path generation.
  ///
  /// The base seed comes from a fresh [`Unseeded`] draw, so repeated calls
  /// give independent estimates. Use [`Self::price_seeded`] when the estimate
  /// has to be reproducible.
  #[allow(clippy::too_many_arguments)]
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
  ) -> McEstimate<f64> {
    let base = Unseeded.next_seed();
    self.price_from(s, k, h, r, sigma, t, barrier_type, option_type, base)
  }

  /// Reproducible counterpart of [`Self::price`]: the same `seed` always
  /// yields the same estimate, independently of how rayon schedules the
  /// paths, because each path derives its own stream from `seed` and its
  /// index.
  #[allow(clippy::too_many_arguments)]
  pub fn price_seeded(
    &self,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    sigma: f64,
    t: f64,
    barrier_type: BarrierType,
    option_type: OptionType,
    seed: u64,
  ) -> McEstimate<f64> {
    self.price_from(s, k, h, r, sigma, t, barrier_type, option_type, seed)
  }

  #[allow(clippy::too_many_arguments)]
  fn price_from(
    &self,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    sigma: f64,
    t: f64,
    barrier_type: BarrierType,
    option_type: OptionType,
    base_seed: u64,
  ) -> McEstimate<f64> {
    let dt = t / self.n_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let discount = (-r * t).exp();

    let payoff_of = |path: usize| -> f64 {
      {
        // Golden-ratio stride keeps consecutive path seeds far apart in the
        // splitmix state space, so neighbouring paths do not share a stream.
        let seed = base_seed.wrapping_add((path as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut normals = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
        let mut z_buf = vec![0.0_f64; self.n_steps];
        normals.fill_slice(&mut z_buf);

        let mut s_curr = s;
        let mut barrier_hit = false;

        for &z in z_buf.iter() {
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
      }
    };
    let sum: f64 = (0..self.n_paths).into_par_iter().map(&payoff_of).sum();
    let sum_sq: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|path| {
        let y = payoff_of(path);
        y * y
      })
      .sum();

    McEstimate {
      mean: discount * sum / self.n_paths as f64,
      std_err: discount * std_err_from_sums(sum, sum_sq, self.n_paths),
      n_samples: self.n_paths,
    }
  }
}

#[cfg(test)]
mod tests;
