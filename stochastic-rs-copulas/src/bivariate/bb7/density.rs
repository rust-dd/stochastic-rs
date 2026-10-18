use stochastic_rs_distributions::RealExt;

use super::Bb7;
use super::ln_phi;
use crate::bivariate::conditional::expm1_over;
use crate::bivariate::conditional::ln_1p_over;
use crate::bivariate::conditional::ln_add_exp;
use crate::bivariate::conditional::ln_one_minus_exp;
use crate::bivariate::conditional::ln_one_minus_pow_ratio;
use crate::bivariate::conditional::ln_ratio_complement;
use crate::bivariate::conditional::softplus;

/// One coordinate at `θ`: `ln(1 − u)`, `ln α = θ ln(1 − u)`, `w = −ln(1 − α)` and `κ = ln w − ln α`, the limit `α/2`
/// where `α` is below rounding, so `ln w` survives the underflow of `w`.
struct Leg {
  ln_bar: f64,
  ln_alpha: f64,
  w: f64,
  kappa: f64,
}

impl Leg {
  fn new(theta: f64, ln_bar: f64) -> Self {
    let ln_alpha = theta * ln_bar;
    let (w, ln_w) = Bb7::w_of(theta, ln_bar, ln_alpha);
    let kappa = if ln_alpha < -36.7 {
      0.5 * ln_alpha.exp()
    } else {
      ln_w - ln_alpha
    };
    Self {
      ln_bar,
      ln_alpha,
      w,
      kappa,
    }
  }
}

/// `ln(1 − T)` and its pieces: `R = ln(1 − T) − θt`, `t ≥ b` the two of `ln(1 − u)`, `ln(1 − v)`, `b − t`, `Δ = |w_u − w_v|`,
/// the smaller `w`, `ℓ = ln(1 + e^{−δΔ}(1 − e^{−δw_min}))` and `ℓ/δ`, so `ln s = δw_max + ℓ`.
struct Tail {
  ln_1mt: f64,
  r: f64,
  top: f64,
  gap: f64,
  spread: f64,
  w_min: f64,
  ell: f64,
  ell_over: f64,
}

/// BB7's log-density at one `(θ, δ)`, with what no point changes formed once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bb7Density {
  theta: f64,
  delta: f64,
  ln_delta: f64,
  /// `ln(1 + θδ)` and `ln(θ − 1)`, the logarithms of the last factor's two terms.
  ln_1p_theta_delta: f64,
  ln_theta_m1: f64,
}

impl Bb7Density {
  pub(crate) fn new(theta: f64, delta: f64) -> Self {
    let ln_delta = delta.ln();
    Self {
      theta,
      delta,
      ln_delta,
      ln_1p_theta_delta: softplus(theta.ln() + ln_delta),
      ln_theta_m1: (theta - 1.0).ln(),
    }
  }

  /// `ln c` on the open square: the module's `c` term by term while its terms are finite and under `L`, `7Lε ≤ 1e-12 ·
  /// max(1, |ln c|)`, each then within a few `ε` of `L`, and `θ ln(1 − u)`, `s − 1`, `−ln T` are normal; else `recentred`.
  pub(crate) fn ln_c(&self, u: f64, v: f64) -> f64 {
    let (theta, delta) = (self.theta, self.delta);
    let (bar_u, bar_v) = ((-u).ln_1p(), (-v).ln_1p());
    let (ln_a, ln_b) = (
      ln_one_minus_exp(theta * bar_u),
      ln_one_minus_exp(theta * bar_v),
    );
    let ln_s = ((-delta * ln_a).exp_m1() + (-delta * ln_b).exp_m1()).ln_1p();
    let ln_t = -ln_s / delta;
    let ln_1mt = ln_one_minus_exp(ln_t);
    let terms = [
      (1.0 / theta - 2.0) * ln_1mt,
      -(1.0 / delta + 2.0) * ln_s,
      -(delta + 1.0) * (ln_a + ln_b),
      (theta - 1.0) * (bar_u + bar_v),
      ln_add_exp(self.ln_1p_theta_delta + ln_1mt, self.ln_theta_m1),
    ];
    let sum = terms.iter().sum::<f64>();
    let size = terms.iter().fold(0.0_f64, |m, t| m.max_or_nan(t.abs()));
    let normal = -theta * bar_u.max(bar_v) >= f64::MIN_POSITIVE
      && ln_s >= f64::MIN_POSITIVE
      && -ln_t >= f64::MIN_POSITIVE;
    if normal && size.is_finite() && 7.0 * f64::EPSILON * size <= 1e-12 * sum.abs().max(1.0) {
      return sum;
    }
    self.recentred(u, v, bar_u, bar_v)
  }

  /// `ln c` re-centred twice: `(θ − 1)(b − t) − t + (1/θ − 2)R` for `(θ − 1)(t + b) + (1/θ − 2) ln(1 − T)` and
  /// `−δΔ + w_min − (1/δ + 2)ℓ` for `−(1/δ + 2) ln s + (δ + 1)(w_u + w_v)`, the last factor `(θδ + 1)(1 − T) + θ − 1`.
  fn recentred(&self, u: f64, v: f64, bar_u: f64, bar_v: f64) -> f64 {
    let (theta, delta) = (self.theta, self.delta);
    let t = self.tail(u, v, bar_u, bar_v);
    (theta - 1.0) * t.gap - t.top + (1.0 / theta - 2.0) * t.r - delta * t.spread + t.w_min
      - t.ell_over
      - 2.0 * t.ell
      + ln_add_exp(self.ln_1p_theta_delta + t.ln_1mt, self.ln_theta_m1)
  }

  /// `T = s^{−1/δ}`, `s = 1 + F(α) + F(β)`, `F = expm1(δw)`: while `δw_max < 1`, `ln(L/δ)`, `L = ln s`, from `ln(S/δ) − θt`
  /// of the `κ`s, which holds as `α` underflows; past it `L/δ = w_max + ℓ/δ`, `ℓ/δ` from `expm1(t)/t`, free of `ln δ`.
  fn tail(&self, u: f64, v: f64, bar_u: f64, bar_v: f64) -> Tail {
    let (theta, delta) = (self.theta, self.delta);
    // `w` grows as `u` falls, so the order of `u, v` is that of `w` where both powers underflow and `Δ` rounds to 0.
    let (hi, lo) = if u <= v {
      (Leg::new(theta, bar_u), Leg::new(theta, bar_v))
    } else {
      (Leg::new(theta, bar_v), Leg::new(theta, bar_u))
    };
    let gap = -ln_ratio_complement(u, v).abs();
    let spread = ln_one_minus_pow_ratio(theta, bar_v, bar_u).abs();
    let z_lo = delta * lo.w;
    let (ell, ell_over) = if z_lo < 1.0 {
      let t_over = (-delta * spread).exp() * lo.w * expm1_over(-z_lo);
      let t = delta * t_over;
      let ell = t.ln_1p();
      (ell, t_over * if t == 0.0 { 1.0 } else { ell / t })
    } else {
      let ell = ((-delta * spread).exp() * -(-z_lo).exp_m1()).ln_1p();
      (ell, ell / delta)
    };
    let z_hi = delta * hi.w;
    // `ln(L/δ)` and `ln(L/δ) − θt`.
    let (ln_ld, rel) = if z_hi < 1.0 {
      let rel = ln_add_exp(
        hi.kappa + expm1_over(z_hi).ln(),
        theta * gap + lo.kappa + expm1_over(z_lo).ln(),
      );
      let rel = rel + ln_1p_over((hi.ln_alpha + rel + self.ln_delta).exp()).ln();
      (hi.ln_alpha + rel, rel)
    } else {
      let ln_ld = (hi.w + ell_over).ln();
      (ln_ld, ln_ld - hi.ln_alpha)
    };
    // `ln(1 − e^{−z})` directly once `z = L/δ ≥ 1`, where `ln z + ln φ(z)` would round its small value away.
    let ld = ln_ld.exp();
    let (ln_1mt, r) = if ld >= 1.0 {
      let ln_1mt = ln_one_minus_exp(-ld);
      (ln_1mt, ln_1mt - hi.ln_alpha)
    } else {
      let phi = ln_phi(ld, ln_ld);
      (ln_ld + phi, rel + phi)
    };
    Tail {
      ln_1mt,
      r,
      top: hi.ln_bar,
      gap,
      spread,
      w_min: lo.w,
      ell,
      ell_over,
    }
  }

  /// `C = 1 − (1 − T)^{1/θ} = −expm1(ln(1 − T)/θ)` on the open square, `ln(1 − T)` from `tail`.
  pub(crate) fn cdf(&self, u: f64, v: f64) -> f64 {
    let t = self.tail(u, v, (-u).ln_1p(), (-v).ln_1p());
    -(t.ln_1mt / self.theta).exp_m1()
  }
}
