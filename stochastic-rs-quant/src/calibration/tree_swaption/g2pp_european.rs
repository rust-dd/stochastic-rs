//! Brigo & Mercurio's European swaption under G2++, both factors from zero, a constant shift `φ`.
//!
//! $$
//! \mathrm{ES}=\omega P(0,T)\int_{-\infty}^{\infty}
//! \frac{e^{-\frac12\left(\frac{x-\mu_x}{\sigma_x}\right)^2}}{\sigma_x\sqrt{2\pi}}
//! \Bigl[\Phi(-\omega h_1(x))-\sum_{i=1}^{n}\lambda_i(x)\,e^{\kappa_i(x)}\,\Phi(-\omega h_2(x))\Bigr]dx,
//! \qquad\sum_{i=1}^{n}c_i\,A(T,t_i)\,e^{-B(a,T,t_i)\,x-B(b,T,t_i)\,\bar y(x)}=1
//! $$
//!
//! Reference: Brigo & Mercurio (2006), 2nd ed., §4.2.4, p. 158, as restated in QuantLib's `G2`
//! (`ql/models/shortrate/twofactormodels/g2.cpp`) and by Acar & Natcheva-Acar (2009), Thm 1.1–1.3

use std::cell::Cell;
use std::sync::OnceLock;

use gauss_quad::GaussLegendre;
use stochastic_rs_distributions::brent;
use stochastic_rs_distributions::special::norm_cdf;

use super::G2ppParams;
use crate::instruments::option::types::SwaptionDirection;
use crate::traits::RealExt;

/// Standard deviations of `x(T)` integrated each side, past the receiver's exponential growth.
const NORMAL_RANGE: f64 = 9.0;

/// Intervals searched for a sign change of `h₁`, where the exercise boundary crosses `E[y | x]`.
const ROOT_SAMPLES: usize = 12;

/// `Φ` moves less than `1e-15` past this many units of `h₁` from the kink.
const TRANSITION: f64 = 8.0;

/// A panel is accepted when its halves agree with it within this, on a unit bond.
const TOLERANCE: f64 = 1e-13;

const MAX_DEPTH: u32 = 30;

fn gauss_legendre_10() -> &'static GaussLegendre {
  static RULE: OnceLock<GaussLegendre> = OnceLock::new();
  RULE.get_or_init(|| GaussLegendre::new(10.try_into().expect("10 > 0")))
}

/// `B(z, t, t + τ) = (1 − e^{−zτ})/z`.
fn loading(z: f64, tau: f64) -> f64 {
  -(-z * tau).exp_m1() / z
}

/// `∫₀¹ s² f(us) f(vs) ds`, `f(x) = (1 − e^{−x})/x`, as QuantLib's `G2::V` evaluates it: the closed
/// form `(1 − f(u) − f(v) + f(u + v))/(uv)` cancels to `O(uv)`, so small arguments take series.
fn variance_kernel(u: f64, v: f64) -> f64 {
  let (u, v) = if u < v { (v, u) } else { (u, v) };
  if u + v <= 2.0 {
    // Σ_{k≥2} (−1)^k P_k/(k + 1)!, P_k = ((u + v)^k − u^k − v^k)/(uv).
    let s = u + v;
    let (mut p, mut u_power, mut v_power) = (2.0, u, v);
    let (mut factorial, mut sign, mut sum) = (6.0, 1.0, 0.0_f64);
    for k in 2..60 {
      let term = sign * p / factorial;
      sum += term;
      if term.abs() < 0.01 * f64::EPSILON * sum.abs() {
        break;
      }
      p = s * p + u_power + v_power;
      u_power *= u;
      v_power *= v;
      factorial *= (k + 2) as f64;
      sign = -sign;
    }
    return sum;
  }
  let f = |x: f64| -(-x).exp_m1() / x;
  if v >= 0.5 {
    return (1.0 - f(u) - f(v) + f(u + v)) / (u * v);
  }
  // `1 − f(v) = v·ω(v)`, `ω(v) = Σ_{k≥0} (−v)^k/(k + 2)!`.
  let (mut omega, mut term) = (0.0_f64, 0.5);
  for k in 2..40 {
    omega += term;
    term *= -v / (k + 1) as f64;
    if term.abs() < 0.01 * f64::EPSILON * omega.abs() {
      break;
    }
  }
  let difference = (u * (-u).exp() * (-v).exp_m1() - v * (-u).exp_m1()) / (u * (u + v));
  (v * omega - difference) / (u * v)
}

/// `V(t, t + τ)`, the variance of `∫(x + y)` over `τ`.
fn integrated_variance(p: &G2ppParams, tau: f64) -> f64 {
  let (u, v) = (p.mean_reversion_x * tau, p.mean_reversion_y * tau);
  let (sigma, eta) = (p.sigma_x, p.sigma_y);
  let cross = 2.0 * p.rho * sigma * eta * variance_kernel(u, v);
  tau.powi(3) * (sigma * sigma * variance_kernel(u, u) + eta * eta * variance_kernel(v, v) + cross)
}

/// The model's own bond, `P(0, t) = exp(−φt + V(0, t)/2)`.
fn bond(p: &G2ppParams, t: f64) -> f64 {
  (-p.phi * t + 0.5 * integrated_variance(p, t)).exp()
}

/// `μ_x, μ_y, σ_x, σ_y, ρ_xy`: the law of `(x(T), y(T))` under the `T`-forward measure.
struct ForwardMoments {
  mu_x: f64,
  mu_y: f64,
  sigma_x: f64,
  sigma_y: f64,
  rho_xy: f64,
}

impl ForwardMoments {
  fn new(p: &G2ppParams, t: f64) -> Self {
    let (a, b) = (p.mean_reversion_x, p.mean_reversion_y);
    let (sigma, eta, rho) = (p.sigma_x, p.sigma_y, p.rho);
    let decay = |k: f64| -(-k * t).exp_m1();
    let sigma_x = sigma * (decay(2.0 * a) / (2.0 * a)).sqrt();
    let sigma_y = eta * (decay(2.0 * b) / (2.0 * b)).sqrt();
    let cross = rho * sigma * eta;
    Self {
      mu_x: -((sigma * sigma / (a * a) + cross / (a * b)) * decay(a)
        - 0.5 * sigma * sigma / (a * a) * decay(2.0 * a)
        - cross / (b * (a + b)) * decay(a + b)),
      mu_y: -((eta * eta / (b * b) + cross / (a * b)) * decay(b)
        - 0.5 * eta * eta / (b * b) * decay(2.0 * b)
        - cross / (a * (a + b)) * decay(a + b)),
      sigma_x,
      sigma_y,
      rho_xy: cross * decay(a + b) / ((a + b) * sigma_x * sigma_y),
    }
  }
}

/// A swaption per unit notional on a swap paying `strike·accrual` at each of `coupon_times`;
/// `strike ≥ 0`, both reversions and vols positive and `|ρ| < 1`, as the calibrator's trials are.
pub(super) fn g2pp_european_swaption(
  p: &G2ppParams,
  direction: SwaptionDirection,
  strike: f64,
  expiry: f64,
  coupon_times: &[f64],
  accrual: f64,
) -> f64 {
  match coupon_times {
    [payment] => one_period(p, direction, strike, expiry, *payment, accrual),
    _ => integral(p, direction, strike, expiry, coupon_times, accrual),
  }
}

/// One payment: `(1 + Kτ)` bond puts (payer) or calls struck at `1/(1 + Kτ)`, in closed form.
fn one_period(
  p: &G2ppParams,
  direction: SwaptionDirection,
  strike: f64,
  expiry: f64,
  payment: f64,
  accrual: f64,
) -> f64 {
  let m = ForwardMoments::new(p, expiry);
  let tau = payment - expiry;
  let (b_x, b_y) = (
    loading(p.mean_reversion_x, tau),
    loading(p.mean_reversion_y, tau),
  );
  let covariance = m.rho_xy * m.sigma_x * m.sigma_y;
  let variance = b_x * b_x * m.sigma_x * m.sigma_x
    + b_y * b_y * m.sigma_y * m.sigma_y
    + 2.0 * b_x * b_y * covariance;
  let sigma_p = variance.sqrt();
  let (to_expiry, to_payment) = (bond(p, expiry), bond(p, payment));
  let gross = 1.0 + strike * accrual;
  let bond_strike = 1.0 / gross;
  let h = (to_payment / (bond_strike * to_expiry)).ln() / sigma_p + 0.5 * sigma_p;
  let option = match direction {
    SwaptionDirection::Payer => {
      bond_strike * to_expiry * norm_cdf(sigma_p - h) - to_payment * norm_cdf(-h)
    }
    SwaptionDirection::Receiver => {
      to_payment * norm_cdf(h) - bond_strike * to_expiry * norm_cdf(h - sigma_p)
    }
  };
  gross * option
}

/// `c_i A(T, t_i)` and the two loadings of one payment.
struct Leg {
  ln_weight: f64,
  weight: f64,
  b_x: f64,
  b_y: f64,
}

fn integral(
  p: &G2ppParams,
  direction: SwaptionDirection,
  strike: f64,
  expiry: f64,
  coupon_times: &[f64],
  accrual: f64,
) -> f64 {
  let omega = match direction {
    SwaptionDirection::Payer => 1.0,
    SwaptionDirection::Receiver => -1.0,
  };
  let m = ForwardMoments::new(p, expiry);
  let to_expiry = bond(p, expiry);
  let (v_expiry, last) = (integrated_variance(p, expiry), coupon_times.len() - 1);
  let legs = coupon_times
    .iter()
    .enumerate()
    .map(|(i, &t)| {
      let c = strike * accrual + if i == last { 1.0 } else { 0.0 };
      let v = integrated_variance(p, t - expiry) - integrated_variance(p, t) + v_expiry;
      let weight = c * bond(p, t) / to_expiry * (0.5 * v).exp();
      Leg {
        ln_weight: weight.ln(),
        weight,
        b_x: loading(p.mean_reversion_x, t - expiry),
        b_y: loading(p.mean_reversion_y, t - expiry),
      }
    })
    .collect::<Vec<_>>();
  let txy = (1.0 - m.rho_xy * m.rho_xy).sqrt();
  let boundary = Cell::new(0.0);
  let h1 = |z: f64| {
    let (y_bar, slope) = exercise_boundary(&legs, m.mu_x + m.sigma_x * z, boundary.get());
    boundary.set(y_bar);
    let h = (y_bar - m.mu_y) / (m.sigma_y * txy) - m.rho_xy * z / txy;
    (
      h,
      (slope * m.sigma_x - m.rho_xy * m.sigma_y) / (m.sigma_y * txy),
    )
  };
  let integrand = |z: f64| {
    let x = m.mu_x + m.sigma_x * z;
    let h_1 = h1(z).0;
    let mut bracket = norm_cdf(-omega * h_1);
    for leg in &legs {
      let lambda = leg.weight * (-leg.b_x * x).exp();
      let h_2 = h_1 + leg.b_y * m.sigma_y * txy;
      let kappa = -leg.b_y
        * (m.mu_y - 0.5 * (1.0 - m.rho_xy * m.rho_xy) * m.sigma_y * m.sigma_y * leg.b_y
          + m.rho_xy * m.sigma_y * (x - m.mu_x) / m.sigma_x);
      bracket -= lambda * kappa.exp() * norm_cdf(-omega * h_2);
    }
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt() * bracket
  };
  let growth = legs.iter().map(|l| l.b_x).fold(0.0, f64::max_or_nan) * m.sigma_x
    + legs.iter().map(|l| l.b_y).fold(0.0, f64::max_or_nan) * m.sigma_y;
  let range = NORMAL_RANGE + growth;
  let edges = breakpoints(range, &h1);
  let total = edges
    .windows(2)
    .map(|w| {
      adaptive(
        &integrand,
        w[0],
        w[1],
        gauss_legendre(&integrand, w[0], w[1]),
        MAX_DEPTH,
      )
    })
    .sum::<f64>();
  omega * to_expiry * total
}

/// `ȳ(x)` and `ȳ'(x)`: Newton on `ln Σ c_i A_i e^{−B_x x − B_y y} = 0`, convex, decreasing in `y`.
fn exercise_boundary(legs: &[Leg], x: f64, start: f64) -> (f64, f64) {
  let mut y = start;
  let (mut slope_x, mut slope_y) = (0.0, 1.0);
  for _ in 0..64 {
    let peak = f64::max_or_nan_fold(legs.iter().map(|l| l.ln_weight - l.b_x * x - l.b_y * y));
    let (mut total, mut sum_x, mut sum_y) = (0.0, 0.0, 0.0);
    for leg in legs {
      let w = (leg.ln_weight - leg.b_x * x - leg.b_y * y - peak).exp();
      total += w;
      sum_x += w * leg.b_x;
      sum_y += w * leg.b_y;
    }
    (slope_x, slope_y) = (sum_x, sum_y);
    let step = (peak + total.ln()) * total / sum_y;
    y += step;
    if !step.is_finite() || step.abs() <= 1e-14 * (1.0 + y.abs()) {
      break;
    }
  }
  (y, -slope_x / slope_y)
}

/// `[−range, 0, range]` and, at each root of `h₁`, the root and the transition's two ends.
fn breakpoints(range: f64, h1: &impl Fn(f64) -> (f64, f64)) -> Vec<f64> {
  let mut edges = vec![-range, 0.0, range];
  let step = 2.0 * range / ROOT_SAMPLES as f64;
  let mut previous = (-range, h1(-range).0);
  for k in 1..=ROOT_SAMPLES {
    let z = -range + step * k as f64;
    let h = h1(z).0;
    if (previous.1 < 0.0) != (h < 0.0)
      && let Some(root) = brent::zero(previous.0, z, 1e-12, |s| h1(s).0)
    {
      let width = TRANSITION / h1(root).1.abs();
      edges.extend(
        [root, root - width, root + width]
          .iter()
          .filter(|e| e.abs() < range),
      );
    }
    previous = (z, h);
  }
  edges.sort_by(f64::total_cmp);
  edges.dedup();
  edges
}

fn gauss_legendre(f: &impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
  let rule = gauss_legendre_10();
  let (mid, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
  half
    * rule
      .nodes()
      .zip(rule.weights())
      .map(|(node, weight)| weight * f(mid + half * node))
      .sum::<f64>()
}

fn adaptive(f: &impl Fn(f64) -> f64, lo: f64, hi: f64, whole: f64, depth: u32) -> f64 {
  let mid = 0.5 * (lo + hi);
  let (left, right) = (gauss_legendre(f, lo, mid), gauss_legendre(f, mid, hi));
  let gap = (left + right - whole).abs();
  if gap <= TOLERANCE || gap.is_nan() || depth == 0 {
    return left + right;
  }
  adaptive(f, lo, mid, left, depth - 1) + adaptive(f, mid, hi, right, depth - 1)
}

#[cfg(test)]
mod tests;
