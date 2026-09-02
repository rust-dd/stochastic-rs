//! Exposure of a payer interest-rate swap under Hull–White, produced with
//! the crate's own stack: the short rate is simulated as `r_t = x_t + α(t)`
//! with `x` the zero-mean Ornstein–Uhlenbeck factor (`dx = −a x dt + σ dW`)
//! and `α(t) = f(0, t) + σ²(1 − e^{−at})² / (2a²)` the deterministic shift
//! that reproduces the initial curve (Brigo & Mercurio 2006, eq. 3.36), and
//! the swap is revalued on its own payment dates from the Hull–White
//! zero-coupon bond formula, `V = N[1 − P(t, T_n) − K Σ_{T_i > t} δ_i P(t, T_i)]`.
//!
//! Reference: Brigo, D. & Mercurio, F. (2006), *Interest Rate Models —
//! Theory and Practice*, 2nd ed., Springer, §3.3; Gregory (2015), Ch. 7.

use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::interest::hull_white::HullWhite as HullWhiteProcess;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::ExposureProfile;
use crate::bonds::HullWhite;
use crate::curves::DiscountCurve;
use crate::traits::ShortRatePricer;

fn zero_drift(_t: f64) -> f64 {
  0.0
}

/// Payer swap exposure under a Hull–White short rate calibrated to `curve`.
#[derive(Clone, Debug)]
pub struct HullWhiteSwapExposure {
  /// Mean reversion `a`.
  pub mean_reversion: f64,
  /// Short-rate volatility `σ`.
  pub sigma: f64,
  /// Notional.
  pub notional: f64,
  /// Fixed rate `K`.
  pub fixed_rate: f64,
  /// Payment times of the fixed leg (increasing, the last one the maturity).
  pub payment_times: Vec<f64>,
  /// Fixed-leg accrual fraction per period.
  pub accrual: f64,
  /// Simulation steps per year of the short rate.
  pub steps_per_year: usize,
}

impl HullWhiteSwapExposure {
  pub fn new(
    mean_reversion: f64,
    sigma: f64,
    notional: f64,
    fixed_rate: f64,
    payment_times: Vec<f64>,
    accrual: f64,
  ) -> Self {
    assert!(
      mean_reversion > 0.0 && sigma >= 0.0,
      "Hull-White needs a > 0 and σ ≥ 0"
    );
    assert!(
      !payment_times.is_empty()
        && payment_times.windows(2).all(|w| w[0] < w[1])
        && payment_times[0] > 0.0,
      "payment times must be positive and increasing"
    );
    Self {
      mean_reversion,
      sigma,
      notional,
      fixed_rate,
      payment_times,
      accrual,
      steps_per_year: 52,
    }
  }

  /// Simulation resolution of the short rate.
  pub fn with_steps_per_year(mut self, steps_per_year: usize) -> Self {
    assert!(steps_per_year >= 1, "at least one step per year");
    self.steps_per_year = steps_per_year;
    self
  }

  /// Swap value at `t` given the short rate `r_t`, on the initial `curve`
  /// (whose knots should start no later than the first exposure date).
  pub fn value_at(&self, curve: &DiscountCurve<f64>, t: f64, r_t: f64) -> f64 {
    let remaining: Vec<f64> = self
      .payment_times
      .iter()
      .copied()
      .filter(|&ti| ti > t + 1e-12)
      .collect();
    let Some(&maturity) = remaining.last() else {
      return 0.0;
    };
    // At inception the bonds are the curve itself; later dates use the
    // Hull–White reconstruction from the simulated short rate. (The curve is
    // flat below its first knot, so P(0, ·) must not go through the formula's
    // P(0, t) normalisation at t = 0.)
    let bond = |tau: f64| {
      if t <= 0.0 {
        curve.discount_factor(tau)
      } else {
        HullWhite::from_curve(curve, self.mean_reversion, self.sigma, t, tau)
          .zero_coupon_price(r_t, tau)
      }
    };
    let annuity: f64 = remaining
      .iter()
      .map(|&ti| self.accrual * bond(ti - t))
      .sum();
    self.notional * (1.0 - bond(maturity - t) - self.fixed_rate * annuity)
  }

  /// Par fixed rate of the swap at inception on `curve`.
  pub fn par_rate(&self, curve: &DiscountCurve<f64>) -> f64 {
    let annuity: f64 = self
      .payment_times
      .iter()
      .map(|&ti| self.accrual * curve.discount_factor(ti))
      .sum();
    (1.0 - curve.discount_factor(*self.payment_times.last().expect("non-empty"))) / annuity
  }

  /// Mark-to-market matrix (paths × payment dates) from `paths` simulated
  /// short-rate paths seeded by `seed`.
  pub fn mtm_matrix<S: SeedExt>(
    &self,
    curve: &DiscountCurve<f64>,
    paths: usize,
    seed: S,
  ) -> Array2<f64> {
    let maturity = *self.payment_times.last().expect("non-empty");
    let steps = ((maturity * self.steps_per_year as f64).round() as usize).max(1);
    let dt = maturity / steps as f64;
    let process = HullWhiteProcess::new(
      zero_drift as fn(f64) -> f64,
      self.mean_reversion,
      self.sigma,
      steps + 1,
      Some(0.0),
      Some(maturity),
      seed,
    );
    let factor_paths = process.sample_par(paths);
    let a = self.mean_reversion;
    let shift = |t: f64| {
      let f0 = instantaneous_forward(curve, t);
      f0 + self.sigma * self.sigma * (1.0 - (-a * t).exp()).powi(2) / (2.0 * a * a)
    };
    let mut mtm = Array2::<f64>::zeros((paths, self.payment_times.len()));
    for (p, path) in factor_paths.iter().enumerate() {
      for (c, &t) in self.payment_times.iter().enumerate() {
        let idx = ((t / dt).round() as usize).min(steps);
        let r_t = path[idx] + shift(t);
        mtm[(p, c)] = self.value_at(curve, t, r_t);
      }
    }
    mtm
  }

  /// Exposure profile on the payment dates.
  pub fn profile<S: SeedExt>(
    &self,
    curve: &DiscountCurve<f64>,
    paths: usize,
    quantile: f64,
    seed: S,
  ) -> ExposureProfile {
    ExposureProfile::from_mtm(
      &self.mtm_matrix(curve, paths, seed),
      self.payment_times.clone(),
      quantile,
    )
  }
}

/// `f(0, t) = −∂_t ln P(0, t)` by a centred difference on the curve.
fn instantaneous_forward(curve: &DiscountCurve<f64>, t: f64) -> f64 {
  let h = 1e-4_f64.max(1e-4 * t);
  let lo = (t - h).max(0.0);
  let hi = t + h;
  -(curve.discount_factor(hi).ln() - curve.discount_factor(lo).ln()) / (hi - lo)
}
