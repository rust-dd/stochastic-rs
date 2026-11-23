//! What a tempered stable shot-noise series drops past its `J`-th arrival, a Lévy process of
//! jumps below `ε = (Γ_J · rate)^{-1/α}`, as the drift and variance of the Brownian motion for it.
//!
//! $$
//! \nu_s^\varepsilon((x,\varepsilon))=C_s\left(x^{-\alpha}-\varepsilon^{-\alpha}\right)
//! x^{\alpha}\int_x^\infty q_s(r)\,r^{-\alpha-1}dr,\quad 0<x<\varepsilon,\quad
//! q_s(r)=P(Z>\lambda_s r)
//! $$
//!
//! $$
//! V(\varepsilon)=\sum_s C_s\left[\int_0^\varepsilon q_s r^{1-\alpha}dr
//! -\frac{2}{(\alpha+2)\varepsilon^{\alpha}}\int_0^\varepsilon q_s r\,dr
//! +\frac{\alpha\varepsilon^{2}}{\alpha+2}\int_\varepsilon^\infty q_s r^{-\alpha-1}dr\right]
//! $$
//!
//! $$
//! \mu(\varepsilon)=-\sum_s s\,C_s\left[\int_0^\varepsilon(1-q_s)r^{-\alpha}dr
//! -\frac{1}{(\alpha+1)\varepsilon^{\alpha}}\int_0^\varepsilon(1-q_s)dr
//! +\frac{\alpha\varepsilon}{\alpha+1}\int_\varepsilon^\infty(1-q_s)r^{-\alpha-1}dr\right]
//! -\frac{(C_+-C_-)\,\varepsilon^{1-\alpha}}{\alpha-1}
//! $$
//!
//! Reference: Asmussen & Rosiński (2001), J. Appl. Prob. 38(2), 482–493, DOI 10.1239/jap/996986757
//! Reference: Rosiński (2004), Thm 5.4, preprint of SPA 117(6), mediatum.ub.tum.de/doc/1072618

use stochastic_rs_distributions::special::expm1_over;
use stochastic_rs_distributions::special::gamma_li;
use stochastic_rs_distributions::special::gamma_ui;
use stochastic_rs_distributions::special::ln_gamma;

use crate::euler::EulerSpec;
use crate::traits::FloatExt;
use crate::traits::RealExt;

/// Terms of the power series on `(0, 1]`: `1/k!` is below `1e-47` by then.
const SERIES_TERMS: usize = 40;

/// A term this far below its sum no longer moves it.
const SERIES_FLOOR: f64 = 1e-17;

/// Past this exponent `e^{−x}` is zero in `f64`, so a larger cut changes nothing.
const TEMPERING_FLOOR: f64 = 745.0;

/// Eight-point Gauss–Legendre nodes and weights on `[−1, 1]`, the positive half.
const LEGENDRE_8: [(f64, f64); 4] = [
  (0.183_434_642_495_649_8, 0.362_683_783_378_362),
  (0.525_532_409_916_329, 0.313_706_645_877_887_3),
  (0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
  (0.960_289_856_497_536_3, 0.101_228_536_290_376_3),
];

/// Panels of the average over `Γ_J`, each under one standard deviation wide.
const ARRIVAL_PANELS: usize = 24;

/// `Σ_{k ≥ 1} (−1)^k (1 − x^{c+k}) / (k! (c + k))` on `0 < x ≤ 1`: `−∫_x^1 (1 − e^{−t}) t^{c−1} dt`.
/// A zero `c + k` is the limit `−ln x`, not a pole.
fn unit_series(c: f64, x: f64) -> f64 {
  let ln_x = x.ln();
  let mut coefficient = 1.0;
  let mut sum = 0.0;
  for k in 1..SERIES_TERMS {
    let k = k as f64;
    coefficient /= -k;
    let term = coefficient * (-ln_x * expm1_over((c + k) * ln_x));
    sum += term;
    if k > 1.0 && term.abs() <= SERIES_FLOOR * sum.abs() {
      break;
    }
  }
  sum
}

/// `Γ(c, x)` for any real `c` at `x ≥ 1`, by the modified Lentz continued fraction.
fn upper_gamma_fraction(c: f64, x: f64) -> f64 {
  const TINY: f64 = 1e-300;
  let mut b = x + 1.0 - c;
  let mut ratio = 1.0 / TINY;
  let mut d = 1.0 / b;
  let mut h = d;
  for i in 1..=1_000 {
    let i = i as f64;
    let an = -i * (i - c);
    b += 2.0;
    d = an * d + b;
    if d.abs() < TINY {
      d = TINY;
    }
    ratio = b + an / ratio;
    if ratio.abs() < TINY {
      ratio = TINY;
    }
    d = 1.0 / d;
    let step = d * ratio;
    h *= step;
    if (step - 1.0).abs() < 1e-16 {
      break;
    }
  }
  (c * x.ln() - x).exp() * h
}

/// `Γ(c, x)` on `x ≥ 1`, a positive shape through the library's own.
fn upper_gamma_tail(c: f64, x: f64) -> f64 {
  if c > 0.0 {
    gamma_ui(c, x)
  } else {
    upper_gamma_fraction(c, x)
  }
}

/// `L(c, x) = ∫_0^x (1 − e^{−t}) t^{c−1} dt`, `c > −1`.
fn tempered_lower(c: f64, x: f64) -> f64 {
  let below_one = |x: f64| {
    let mut power = x.powf(c + 1.0);
    let mut coefficient = 1.0;
    let mut sum = 0.0;
    for k in 1..SERIES_TERMS {
      let k = k as f64;
      let term = coefficient * power / (c + k);
      sum += term;
      if term.abs() <= SERIES_FLOOR * sum.abs() {
        break;
      }
      power *= x;
      coefficient /= -(k + 1.0);
    }
    sum
  };
  if x <= 1.0 {
    below_one(x)
  } else {
    let ln_x = x.ln();
    below_one(1.0) + ln_x * expm1_over(c * ln_x)
      - (upper_gamma_tail(c, 1.0) - upper_gamma_tail(c, x))
  }
}

/// `Γ(c, x)` and `M(c, x) = ∫_x^∞ (1 − e^{−t}) t^{c−1} dt` for `c < 0`, from one series or one
/// fraction; `at_one` is `Γ(c, 1)`.
fn upper_pair(c: f64, x: f64, at_one: f64) -> (f64, f64) {
  if x >= 1.0 {
    let upper = upper_gamma_fraction(c, x);
    (upper, -x.powf(c) / c - upper)
  } else {
    let ln_x = x.ln();
    let series = unit_series(c, x);
    (
      at_one - ln_x * expm1_over(c * ln_x) + series,
      -1.0 / c - at_one - series,
    )
  }
}

/// One side's scale `C_s`, sign, tempering rate `θ = λ/e_scale` and the powers of `θ` that turn
/// the integrals in `t = (θr)^p` back into `r`.
#[derive(Debug, Clone, Copy)]
struct Side {
  sign: f64,
  weight: f64,
  scale: f64,
  theta: f64,
  second_moment: f64,
  first_moment: f64,
  tail: f64,
  compensated: f64,
  length: f64,
}

/// The remainder of one series: `rate = α/(T‖σ‖)` scales the arrivals, `power = 1/e_pow` is the
/// tempering's exponent, `p = 1` exponential and `p = 2` Gaussian.
#[derive(Debug, Clone)]
pub(crate) struct SeriesRemainder {
  alpha: f64,
  pub(crate) rate: f64,
  horizon: f64,
  power: f64,
  sides: [Side; 2],
  tail_at_one: f64,
  cut_cap: f64,
}

impl SeriesRemainder {
  /// From the series' device declaration, so the host and the launch read one set of constants.
  pub(crate) fn of<T: FloatExt>(spec: &EulerSpec<T>, alpha: f64, t_max: f64) -> Self {
    let EulerSpec::TemperedStableSeries {
      rate,
      e_scale,
      e_pow,
      w_plus,
      lambda_plus,
      lambda_minus,
      ..
    } = *spec
    else {
      unreachable!("a tempered stable series declares the series family");
    };
    let f = |v: T| v.to_f64().unwrap();
    let (rate, w_plus, e_scale, e_pow) = (f(rate), f(w_plus), f(e_scale), f(e_pow));
    let (lambda_plus, lambda_minus) = (f(lambda_plus), f(lambda_minus));
    let power = 1.0 / e_pow;
    let mass = alpha / (rate * t_max);
    let side = |sign: f64, weight: f64, lambda: f64| {
      let theta = lambda / e_scale;
      Side {
        sign,
        weight,
        scale: weight * mass,
        theta,
        second_moment: theta.powf(alpha - 2.0) / power,
        first_moment: 1.0 / (theta * theta * power),
        tail: theta.powf(alpha) / power,
        compensated: theta.powf(alpha - 1.0) / power,
        length: 1.0 / (theta * power),
      }
    };
    let theta_min = lambda_plus.min(lambda_minus) / e_scale;
    Self {
      alpha,
      rate,
      horizon: t_max,
      power,
      sides: [
        side(1.0, w_plus, lambda_plus),
        side(-1.0, 1.0 - w_plus, lambda_minus),
      ],
      tail_at_one: upper_gamma_fraction(-alpha / power, 1.0),
      cut_cap: TEMPERING_FLOOR.powf(e_pow) / theta_min,
    }
  }

  /// The drift and variance per unit time of the terms past the arrival `Γ_J = arrival`.
  pub(crate) fn rates(&self, arrival: f64) -> (f64, f64) {
    let (drift, variance, _) = self.moments(arrival);
    (drift, variance)
  }

  /// [`rates`](Self::rates) and `E[H(Γ)]` at `Γ = arrival`, the mean of the term arriving there.
  fn moments(&self, arrival: f64) -> (f64, f64, f64) {
    let (alpha, p) = (self.alpha, self.power);
    let inv_cut_alpha = arrival * self.rate;
    let cut = inv_cut_alpha.powf(-1.0 / alpha);
    // Past the cap the tempering is zero and only the explicit powers of the cut still move.
    let saturated = cut > self.cut_cap;
    let limit = cut.min(self.cut_cap);
    let (mut drift, mut variance, mut term) = (0.0, 0.0, 0.0);
    for side in &self.sides {
      let x = (side.theta * limit).powf(p);
      if x == 0.0 {
        return (0.0, 0.0, 0.0);
      }
      let i1 = side.second_moment * gamma_li((2.0 - alpha) / p, x);
      let i2 = side.first_moment * gamma_li(2.0 / p, x);
      let (upper, tempered) = upper_pair(-alpha / p, x, self.tail_at_one);
      let (i3, k3) = (side.tail * upper, side.tail * tempered);
      let j1 = side.compensated * tempered_lower((1.0 - alpha) / p, x);
      let j0 = side.length * tempered_lower(1.0 / p, x);
      variance += side.scale
        * (i1 - 2.0 * inv_cut_alpha * i2 / (alpha + 2.0)
          + alpha * limit * limit * i3 / (alpha + 2.0));
      let (compensated, capped) = if saturated {
        let beyond = if alpha == 1.0 {
          0.0
        } else {
          limit.powf(1.0 - alpha) / (1.0 - alpha)
        };
        (
          j1 - inv_cut_alpha * (j0 - limit) / (alpha + 1.0) - beyond,
          -alpha * (j0 - limit) / (alpha + 1.0),
        )
      } else {
        (
          j1 - inv_cut_alpha * j0 / (alpha + 1.0) + alpha * cut * k3 / (alpha + 1.0),
          cut - alpha * (j0 + cut * k3 / inv_cut_alpha) / (alpha + 1.0),
        )
      };
      drift -= side.sign * side.scale * compensated;
      term += side.sign * side.weight * capped;
    }
    // The centring of unequal sides; at α = 1 it is logarithmic, which `Kobol::new` refuses.
    let imbalance = self.sides[0].scale - self.sides[1].scale;
    if imbalance != 0.0 && alpha != 1.0 && !saturated {
      drift -= imbalance * cut * inv_cut_alpha / (alpha - 1.0);
    }
    (drift, variance.max_or_nan(0.0), term)
  }

  /// `spec` with the launch's remainder: its drift added to `b_t`, its deviation as `rem_sd`.
  pub(crate) fn launch<T: FloatExt>(&self, spec: EulerSpec<T>, terms: usize) -> EulerSpec<T> {
    let (drift, variance) = self.expected_rates(terms);
    match spec {
      EulerSpec::TemperedStableSeries {
        b_t,
        rate,
        inv_alpha,
        e_scale,
        e_pow,
        w_plus,
        lambda_plus,
        lambda_minus,
        ..
      } => EulerSpec::TemperedStableSeries {
        b_t: b_t + T::from_f64_fast(drift),
        rate,
        inv_alpha,
        e_scale,
        e_pow,
        w_plus,
        lambda_plus,
        lambda_minus,
        rem_sd: T::from_f64_fast(variance.sqrt()),
      },
      other => other,
    }
  }

  /// The constants a device launch takes: the drift averaged over `Γ_J ~ Gamma(J, 1)`, and the
  /// variance that, with the kept terms' own, is the law's at the horizon.
  pub(crate) fn expected_rates(&self, terms: usize) -> (f64, f64) {
    let (whole_drift, whole_variance) = self.rates(0.0);
    if terms == 0 {
      return (whole_drift, whole_variance);
    }
    let (j, t) = (terms as f64, self.horizon);
    let spread = 10.0 * j.sqrt();
    let (lo, hi) = ((j - spread).max(0.0), j + spread + 30.0);
    let half = (hi - lo) / (2 * ARRIVAL_PANELS) as f64;
    let ln_norm = ln_gamma(j);
    let mut sums = [0.0; 6];
    for panel in 0..ARRIVAL_PANELS {
      let mid = lo + (2 * panel + 1) as f64 * half;
      for (node, weight) in LEGENDRE_8 {
        for arrival in [mid - half * node, mid + half * node] {
          let density = ((j - 1.0) * arrival.ln() - arrival - ln_norm).exp() * weight;
          let (drift, variance, term) = self.moments(arrival);
          // Given `Γ_J` the earlier arrivals are uniform below it: `J − 1` draws of this mean.
          let each = t * (whole_drift - drift) / arrival;
          let kept = (j - 1.0) * each + term;
          let within = (j - 1.0) * each * each + term * term;
          for (sum, value) in sums
            .iter_mut()
            .zip([1.0, drift, variance, kept, kept * kept, within])
          {
            *sum += density * value;
          }
        }
      }
    }
    let [_, drift, variance, kept, kept_square, within] = sums.map(|sum| sum / sums[0]);
    let between = kept_square - kept * kept;
    (drift, (variance + (within - between) / t).max_or_nan(0.0))
  }
}

#[cfg(test)]
mod tests;
