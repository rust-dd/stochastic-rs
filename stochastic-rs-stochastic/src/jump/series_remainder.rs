//! What a tempered stable shot-noise series drops past its `J`-th arrival, a Lévy process of
//! jumps below `ε = (Γ_J · rate)^{-1/α}`, as the drift and variance of the Brownian motion replacing it.
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
//! Reference: Asmussen & Rosiński (2001), J. Appl. Probab. 38(2), 482–493, DOI 10.1239/jap/996986757
//! Reference: Rosiński (2007), Thm 5.4, SPA 117(6), 677–707, DOI 10.1016/j.spa.2006.10.003

use stochastic_rs_distributions::special::expm1_over;
use stochastic_rs_distributions::special::gamma_li;
use stochastic_rs_distributions::special::gamma_ui;
use stochastic_rs_distributions::special::ln_gamma;

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

/// `Σ_{k ≥ from} (−1)^k (1 − x^{c+k}) / (k! (c + k))` on `0 < x ≤ 1`: `∫_x^1 e^{−t} t^{c−1} dt`
/// from `from = 0`. A zero `c + k` is the limit `−ln x`, not a pole.
fn unit_series(c: f64, x: f64, from: usize) -> f64 {
  let ln_x = x.ln();
  let mut coefficient = 1.0;
  let mut sum = 0.0;
  for k in 0..SERIES_TERMS {
    if k > 0 {
      coefficient /= -(k as f64);
    }
    if k >= from {
      let term = coefficient * (-ln_x * expm1_over((c + k as f64) * ln_x));
      sum += term;
      if k > from && term.abs() <= SERIES_FLOOR * sum.abs() {
        break;
      }
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

/// `Γ(c, x)` for `c ≤ 0`, `x > 0`; `at_one` is `Γ(c, 1)`.
fn upper_gamma(c: f64, x: f64, at_one: f64) -> f64 {
  if x >= 1.0 {
    upper_gamma_fraction(c, x)
  } else {
    at_one + unit_series(c, x, 0)
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

/// `M(c, x) = ∫_x^∞ (1 − e^{−t}) t^{c−1} dt`, `c < 0`; `at_one` is `Γ(c, 1)`.
fn tempered_upper(c: f64, x: f64, at_one: f64) -> f64 {
  if x >= 1.0 {
    -x.powf(c) / c - upper_gamma_fraction(c, x)
  } else {
    -1.0 / c - at_one - unit_series(c, x, 1)
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
  /// From the series' own constants, the ones its device declaration takes.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    alpha: f64,
    rate: f64,
    t_max: f64,
    w_plus: f64,
    lambda_plus: f64,
    lambda_minus: f64,
    e_scale: f64,
    e_pow: f64,
  ) -> Self {
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
      let tail_shape = -alpha / p;
      let i1 = side.second_moment * gamma_li((2.0 - alpha) / p, x);
      let i2 = side.first_moment * gamma_li(2.0 / p, x);
      let i3 = side.tail * upper_gamma(tail_shape, x, self.tail_at_one);
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
        let k3 = side.tail * tempered_upper(tail_shape, x, self.tail_at_one);
        (
          j1 - inv_cut_alpha * j0 / (alpha + 1.0) + alpha * cut * k3 / (alpha + 1.0),
          cut - alpha * (j0 + cut * k3 / inv_cut_alpha) / (alpha + 1.0),
        )
      };
      drift -= side.sign * side.scale * compensated;
      term += side.sign * side.weight * capped;
    }
    // The centring of unequal sides; at α = 1 it is logarithmic and has no finite continuation.
    let imbalance = self.sides[0].scale - self.sides[1].scale;
    if imbalance != 0.0 && alpha != 1.0 && !saturated {
      drift -= imbalance * cut * inv_cut_alpha / (alpha - 1.0);
    }
    (drift, variance.max(0.0), term)
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
    (drift, (variance + (within - between) / t).max(0.0))
  }
}

#[cfg(test)]
mod tests;
