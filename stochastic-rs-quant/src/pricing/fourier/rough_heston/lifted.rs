//! The lifted Heston model's characteristic function: `n` factors on one Brownian motion.
//!
//! $$
//! (\psi^i)' = -x_i\psi^i + F\Bigl(z, \sum_j c_j\psi^j\Bigr),\qquad
//! \ln\phi = \int_0^T F\Bigl(z, \sum_i c_i\psi^i(s)\Bigr)\,g_0(T-s)\,ds,\qquad
//! g_0(t) = V_0 + \kappa\theta\sum_i c_i\,\frac{1-e^{-x_it}}{x_i}
//! $$
//!
//! Reference: Abi Jaber (2019), Quant. Finance 19(12), 1995–2013, DOI 10.1080/14697688.2019.1615113
//! Reference: Abi Jaber & El Euch (2019), SIAM J. Financ. Math. 10(2), 309–349, DOI 10.1137/18M1170236

use ndarray::Array1;
use stochastic_rs_distributions::special::gamma;

use super::LogChf;
use super::Riccati;
use super::RoughHestonParams;
use super::scalar::RiccatiScalar;

/// Past this a solve is NaN: about `ν|ξ|t = 10⁵` for twenty factors, beyond any price's reach.
const MAX_STEPS: usize = 1 << 22;

/// The factors' weights `cᵢ` and mean reversions `xᵢ`, and the step count (C.2) starts from.
#[derive(Clone, Debug)]
pub(super) struct LiftedCf {
  params: RoughHestonParams,
  riccati: Riccati,
  pub(super) weights: Array1<f64>,
  pub(super) reversions: Array1<f64>,
  steps: usize,
}

impl LiftedCf {
  /// Abi Jaber's (3.3): the fractional kernel's measure on the geometric partition `r^{i−n/2}`.
  pub(super) fn new(params: RoughHestonParams, factors: usize, ratio: f64, steps: usize) -> Self {
    let alpha = params.alpha();
    let half = 0.5 * factors as f64;
    let step = ratio.powf(1.0 - alpha) - 1.0;
    let weight =
      step * ratio.powf((alpha - 1.0) * (1.0 + half)) / (gamma(alpha) * gamma(2.0 - alpha));
    let reversion = (1.0 - alpha) / (2.0 - alpha) * (ratio.powf(2.0 - alpha) - 1.0) / step;
    let index = |i: usize| (i + 1) as f64;
    let weights = (0..factors).map(|i| weight * ratio.powf((1.0 - alpha) * index(i)));
    let reversions = (0..factors).map(|i| reversion * ratio.powf(index(i) - 1.0 - half));
    Self::from_kernel(params, weights.collect(), reversions.collect(), steps)
  }

  /// Any kernel `Σcᵢe^{−xᵢt}`, `cᵢ, xᵢ ≥ 0`; `params.hurst` is then unused.
  pub(super) fn from_kernel(
    params: RoughHestonParams,
    weights: Array1<f64>,
    reversions: Array1<f64>,
    steps: usize,
  ) -> Self {
    Self {
      params,
      riccati: Riccati::of(&params),
      weights,
      reversions,
      steps,
    }
  }

  /// `Δ·Σcᵢ/(2 + xᵢΔ)`: times the linearised rate `μ` it is 1 where (C.2)'s eigenvalue is −1.
  pub(super) fn load(&self, dt: f64) -> f64 {
    let factors = self.weights.iter().zip(&self.reversions);
    dt * factors.map(|(c, x)| c / (2.0 + x * dt)).sum::<f64>()
  }

  /// The configured count doubled until `μ·load ≤ ½`; `None` for a NaN rate or past the cap.
  pub(super) fn steps_for(&self, t: f64, mu: f64) -> Option<usize> {
    let mut steps = self.steps;
    while steps <= MAX_STEPS {
      if mu * self.load(t / steps as f64) <= 0.5 {
        return Some(steps);
      }
      if mu.is_nan() {
        return None;
      }
      steps *= 2;
    }
    None
  }

  /// One step of (C.2): implicit in each mean reversion, explicit in `forcing = F(z, Σcψ)`.
  pub(super) fn advance<S: RiccatiScalar>(&self, psi: &mut [S], dt: f64, forcing: S) {
    let step = forcing * dt;
    for (value, x) in psi.iter_mut().zip(&self.reversions) {
      *value = (*value + step) * (1.0 + x * dt).recip();
    }
  }

  /// `g₀` of (2.6) on the grid, each `(1 − e^{−xt})/x` by its own recurrence.
  fn input_curve(&self, dt: f64, steps: usize) -> Vec<f64> {
    let decay = self.reversions.mapv(|x| (-x * dt).exp());
    let gain = self
      .reversions
      .mapv(|x| if x > 0.0 { -(-x * dt).exp_m1() / x } else { dt });
    let level = self.params.kappa * self.params.theta;
    let mut integral = Array1::<f64>::zeros(self.weights.len());
    let mut curve = Vec::with_capacity(steps + 1);
    curve.push(self.params.v0);
    for _ in 0..steps {
      integral = &integral * &decay + &gain;
      curve.push(self.params.v0 + level * self.weights.dot(&integral));
    }
    curve
  }
}

impl LogChf for LiftedCf {
  fn log_chf<S: RiccatiScalar>(&self, t: f64, z: S) -> S {
    let moment = Riccati::moment(z);
    let square = self.riccati.nu * self.riccati.nu;
    let reversion = S::real(self.riccati.kappa) - z * (self.riccati.rho * self.riccati.nu);
    let rate = (reversion * reversion - moment * (2.0 * square))
      .sqrt()
      .norm();
    let Some(steps) = self.steps_for(t, rate) else {
      return S::real(f64::NAN);
    };
    let dt = t / steps as f64;
    let curve = self.input_curve(dt, steps);
    let mut psi = vec![S::ZERO; self.weights.len()];
    let mut sum = S::ZERO;
    for k in 0..=steps {
      let factors = psi.iter().zip(&self.weights);
      let total = factors.fold(S::ZERO, |sum, (&value, &c)| sum + value * c);
      let forcing = self.riccati.rhs(moment, z, total);
      let end = if k == 0 || k == steps { 0.5 } else { 1.0 };
      sum = sum + forcing * (end * curve[steps - k]);
      if k < steps {
        self.advance(&mut psi, dt, forcing);
      }
    }
    sum * dt
  }
}
