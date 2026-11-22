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
//! Reference: Vector Risk (2017), "Double Barrier Option", v8.0.7970, p. 2: Haug (1998), p. 72
//! Reference: QuantLib 1.43, `analyticdoublebarrierengine.cpp`; its header: Haug (2007), p. 156
//!
use stochastic_rs_distributions::RealExt;
use stochastic_rs_distributions::special::mills_ratio;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::digital::not_positive;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::exotic::BarrierOption;
use crate::pricing::exotic::DoubleBarrierOption;
use crate::pricing::exotic::assert_sigma;
use crate::traits::ModelPricer;

mod monte_carlo;

pub use monte_carlo::McBarrierPricer;

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
  /// Panics unless `sigma` is finite and `>= 0`; at `0` the path `S e^{(r−q)t}` is priced.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// On or past the barrier its rebate (any strike) or knocked-in vanilla, at `tau = 0` its payoff;
  /// NaN for a NaN query, a barrier or `tau` not finite, `tau < 0`, or `s, k ≤ 0` before expiry.
  pub fn price(&self, option: &BarrierOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    let BarrierOption {
      strike: k,
      barrier: h,
      rebate,
      barrier_type,
      option_type,
    } = *option;
    if s.is_nan()
      || !(0.0..f64::INFINITY).contains(&tau)
      || !h.is_finite()
      || r.is_nan()
      || q.is_nan()
    {
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
      (false, false) if tau == 0.0 => return vanilla_payoff(option_type, s, k),
      (false, true) if tau == 0.0 => return rebate,
      _ if tau > 0.0 && not_positive(&[k]) => return f64::NAN,
      (true, true) if self.sigma == 0.0 => return deterministic(option, s, r, q, tau),
      (true, true) => {
        let vanilla = BsmPricer::new(self.sigma, BsmCoc::Merton1973);
        return vanilla.price_option(s, k, r, q, tau, option_type);
      }
      _ => {}
    }
    let sigma = self.sigma;
    if sigma == 0.0 {
      // An unbreached option reads its rebate, as the formula's `e` and `f` do, off the path too.
      return if rebate.is_nan() {
        f64::NAN
      } else {
        deterministic(option, s, r, q, tau)
      };
    }

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

/// The path `S e^{(r−q)t}`, which meets `H` at `t* = ln(H/S)/(r − q)` or never: a knock-out pays
/// its rebate at `t*`, a knock-in its payoff once in, at the start or later, else its rebate.
fn deterministic(option: &BarrierOption, s: f64, r: f64, q: f64, tau: f64) -> f64 {
  let BarrierOption {
    strike: k,
    barrier: h,
    rebate,
    barrier_type,
    option_type,
  } = *option;
  let terminal = s * ((r - q) * tau).exp();
  let hit = match barrier_type {
    BarrierType::DownAndIn | BarrierType::DownAndOut => s.min(terminal) <= h,
    BarrierType::UpAndIn | BarrierType::UpAndOut => s.max(terminal) >= h,
  };
  let knock_in = matches!(barrier_type, BarrierType::DownAndIn | BarrierType::UpAndIn);
  let discount = (-r * tau).exp();
  match (knock_in, hit) {
    (false, true) => rebate * (-r * (h / s).ln() / (r - q)).exp(),
    (true, false) => rebate * discount,
    _ => discount * vanilla_payoff(option_type, terminal, k),
  }
}

/// Double knock-out on flat barriers: Kunitomo–Ikeda's series as QuantLib's engine codes it, the
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

#[cfg(test)]
mod tests;
