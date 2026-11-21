//! Barrier options: Reiner–Rubinstein for one barrier, Kunitomo–Ikeda for two, and Monte Carlo.
//!
//! $$
//! V_0=e^{-rT}\,\mathbb E^{\mathbb Q}\!\left[\Pi(S_T)\,\mathbf{1}_{\{\tau_H>T\}}\right]
//! $$
//!
//! Reference: Reiner, E. & Rubinstein, M. (1991), "Breaking Down the Barriers", *Risk* 4(8), 28–35
//! Reference: Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", ISBN 0071389970
//! Reference: Vector Risk (2017), "Single Barrier Option", v8.0.7905, p. 2: Haug's §4.17.1, p. 152
//! Reference: Kunitomo & Ikeda (1992), Math. Finance 2(4), DOI 10.1111/j.1467-9965.1992.tb00033.x
//! Reference: Vector Risk (2017), "Double Barrier Option", v8.0.7970: Haug (1998), §2.10.2, p. 72
//!
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::mills_ratio;
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
  /// the vanilla, `tau = 0` pays out, and NaN for a negative or NaN `tau` or a NaN barrier or spot.
  pub fn price(&self, option: &BarrierOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let BarrierOption {
      strike: k,
      barrier: h,
      rebate,
      barrier_type,
      option_type,
    } = *option;
    if s.is_nan() || tau.is_nan() || tau < 0.0 || h.is_nan() || r.is_nan() || q.is_nan() {
      return f64::NAN;
    }
    if tau > 0.0 && s <= 0.0 {
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
  /// a negative or NaN `tau`, a NaN bound or rate, or a spot or `lower` not positive before expiry.
  pub fn price(&self, option: &DoubleBarrierOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let DoubleBarrierOption {
      strike: k,
      lower: l,
      upper: u,
      option_type,
    } = *option;
    if s.is_nan()
      || tau.is_nan()
      || tau < 0.0
      || l.is_nan()
      || u.is_nan()
      || r.is_nan()
      || q.is_nan()
    {
      return f64::NAN;
    }
    if tau > 0.0 && !(s > 0.0 && l > 0.0) {
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

  /// The series, each power folded into its Gaussian so that no image cancels as `σ → 0`, out to
  /// `n = ±N` for `N` below.
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
    let corridor = Corridor::new(s, (l, u), (alpha, beta));
    let z = sd / corridor.width;
    if z > IMAGE_SPREAD_CUTOFF {
      return 0.0;
    }
    let n_max = (1.0 + 0.5 * IMAGE_DECAY * z * z).sqrt().ceil() as i32;
    let drift = (b + 0.5 * sigma * sigma) * tau / sd;
    let mu1 = 2.0 * b / (sigma * sigma) + 1.0;
    let [asset_leg, cash_leg] = [(drift, mu1), (drift - sd, mu1 - 2.0)].map(|(drift, mu)| Leg {
      drift,
      mu,
      ln_pdf: corridor
        .ends
        .map(|end| ln_norm_pdf(end.spot_over_end / sd + drift)),
    });
    let (mut asset, mut cash) = (0.0, 0.0);
    for n in -n_max..=n_max {
      let [direct, reflected] = corridor.images(n, sd);
      asset += asset_leg.weighted(&direct, sd) - asset_leg.weighted(&reflected, sd);
      cash += cash_leg.weighted(&direct, sd) - cash_leg.weighted(&reflected, sd);
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

/// The logs of the ratios between the spot, the corridor `[L, U]` and each end `a` of the strike
/// range, all `≥ 0` but `ln(S/a)`.
struct Corridor {
  width: f64,
  spot_over_lower: f64,
  upper_over_spot: f64,
  ends: [End; 2],
}

#[derive(Clone, Copy)]
struct End {
  spot_over_end: f64,
  end_over_lower: f64,
  upper_over_end: f64,
}

impl Corridor {
  fn new(s: f64, (l, u): (f64, f64), (alpha, beta): (f64, f64)) -> Self {
    Self {
      width: (u / l).ln(),
      spot_over_lower: (s / l).ln(),
      upper_over_spot: (u / s).ln(),
      ends: [alpha, beta].map(|a| End {
        spot_over_end: (s / a).ln(),
        end_over_lower: (a / l).ln(),
        upper_over_end: (u / a).ln(),
      }),
    }
  }

  /// The direct and the reflected image `n`, each `c` and `c + ln(S/a)` summed from terms of one
  /// sign, so that `E ≤ 0` keeps its digits and is exactly 0 where it vanishes.
  fn images(&self, n: i32, sd: f64) -> [Image; 2] {
    let (w, p, p_up) = (self.width, self.spot_over_lower, self.upper_over_spot);
    let (nw, outer) = (f64::from(n) * w, f64::from(n.abs() - 1) * w);
    let reflected_c = if n >= 0 { -(p + nw) } else { p_up + outer };
    let direct = self.ends.map(|end| {
      let c_plus_log = match n {
        0 => 0.0,
        1.. => outer + p + end.upper_over_end,
        _ => -(outer + p_up + end.end_over_lower),
      };
      (end.spot_over_end + 2.0 * nw, c_plus_log)
    });
    let reflected = self.ends.map(|end| {
      let c_plus_log = if n >= 0 {
        -(end.end_over_lower + nw)
      } else {
        end.upper_over_end + outer
      };
      (2.0 * reflected_c + end.spot_over_end, c_plus_log)
    });
    [
      Image::new(nw, direct, sd),
      Image::new(reflected_c, reflected, sd),
    ]
  }
}

/// One image: its power `e^{μc}` and, per end, its argument's numerator `c″` and its `E`.
struct Image {
  c: f64,
  ends: [(f64, f64); 2],
}

impl Image {
  fn new(c: f64, ends: [(f64, f64); 2], sd: f64) -> Self {
    let c_over_sd = c / sd;
    let ends = ends.map(|(numerator, g)| (numerator, -2.0 * c_over_sd * (g / sd)));
    Self { c, ends }
  }
}

/// The asset leg (`D`, `μ₁`) or the cash leg (`D − σ√T`, `μ₁ − 2`), with `ln φ` at each end's
/// argument for `n = 0`, which every folded image shares.
struct Leg {
  drift: f64,
  mu: f64,
  ln_pdf: [f64; 2],
}

impl Leg {
  /// `e^{μc}(N(x_α) − N(x_β))` for `x = c″/σ√T + D`; in a tail each `e^{μc}φ(x)` is folded, and
  /// Mills' ratio carries the rest:
  /// $$
  /// e^{\mu c}\,\varphi\Bigl(\frac{c''}{\sigma\sqrt T}+D\Bigr)
  /// =\varphi\Bigl(\frac{\ln(S/a)}{\sigma\sqrt T}+D\Bigr)\,e^{E},\qquad
  /// E=-\frac{2c\,\bigl(c+\ln(S/a)\bigr)}{\sigma^2T}\le0
  /// $$
  fn weighted(&self, image: &Image, sd: f64) -> f64 {
    let [x, y] = image.ends.map(|(numerator, _)| numerator / sd + self.drift);
    if x == y {
      return 0.0;
    }
    let folded = |end: usize, t: f64| self.ln_pdf[end] + image.ends[end].1 + mills_ratio(t).ln();
    let (near, far) = if y >= 0.0 {
      (folded(1, y), folded(0, x))
    } else if x <= 0.0 {
      (folded(0, -x), folded(1, -y))
    } else {
      let power = if image.c == 0.0 {
        1.0
      } else {
        (self.mu * image.c).exp()
      };
      return power * (norm_cdf(x) - norm_cdf(y));
    };
    if near == f64::NEG_INFINITY {
      return 0.0;
    }
    near.exp() * -(far - near).exp_m1()
  }
}

fn ln_norm_pdf(x: f64) -> f64 {
  -0.5 * x * x - 0.5 * std::f64::consts::TAU.ln()
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
