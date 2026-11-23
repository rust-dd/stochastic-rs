//! The lifted Heston model's characteristic function: `n` factors on one Brownian motion.
//!
//! $$
//! (\psi^i)' = -x_i\psi^i + F\Bigl(z, \sum_j c_j\psi^j\Bigr),\qquad
//! \ln\phi = \int_0^T F\Bigl(z, \sum_i c_i\psi^i(s)\Bigr)\,g_0(T-s)\,ds,\qquad
//! g_0(t) = V_0 + \kappa\theta\sum_i c_i\,\frac{1-e^{-x_it}}{x_i}
//! $$
//!
//! The Riccati system steps by Euler, implicit in the mean reversions as his (C.2) and in `F` too.
//!
//! Reference: Abi Jaber (2019), Quant. Finance 19(12), 1995–2013, DOI 10.1080/14697688.2019.1615113
//! Reference: Abi Jaber & El Euch (2019), SIAM J. Financ. Math. 10, 309–349, DOI 10.1137/18M1170236

use ndarray::Array1;
use stochastic_rs_distributions::special::gamma;

use super::LogChf;
use super::Riccati;
use super::RoughHestonParams;
use super::scalar::RiccatiScalar;

/// When the step reads `F(z, Σcψ)`: `Solved` at its end, by the quadratic it then closes on.
#[derive(Clone, Copy, Debug)]
pub(super) enum Forcing {
  Solved,
  /// At its start, Abi Jaber's (C.2): stable only while `Δ·ν|A|·Σcᵢ/(2 + xᵢΔ) < 1`.
  #[cfg(test)]
  Explicit,
}

/// The factors' weights `cᵢ` and mean reversions `xᵢ`, and the step count.
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
  #[cfg(test)]
  pub(super) fn load(&self, dt: f64) -> f64 {
    let factors = self.weights.iter().zip(&self.reversions);
    dt * factors.map(|(c, x)| c / (2.0 + x * dt)).sum::<f64>()
  }

  /// `ψⁱ ← (ψⁱ + Δ·forcing)/(1 + xᵢΔ)`: (C.2)'s step, implicit in each mean reversion.
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

impl LiftedCf {
  /// (2.12) by the trapezoid on the grid of (2.10)'s Euler steps.
  pub(super) fn solve<S: RiccatiScalar>(&self, t: f64, z: S, forcing: Forcing) -> S {
    let moment = Riccati::moment(z);
    let steps = self.steps;
    let dt = t / steps as f64;
    let curve = self.input_curve(dt, steps);
    let damped = &self.weights / &self.reversions.mapv(|x| 1.0 + x * dt);
    let total = |psi: &[S], weights: &Array1<f64>| {
      let factors = psi.iter().zip(weights);
      factors.fold(S::ZERO, |sum, (&value, &c)| sum + value * c)
    };
    let mut psi = vec![S::ZERO; self.weights.len()];
    let mut force = moment;
    let mut sum = force * (0.5 * curve[steps]);
    for k in 1..=steps {
      if matches!(forcing, Forcing::Solved) {
        let known = total(&psi, &damped);
        let next = self.riccati.implicit(moment, z, damped.sum() * dt, known);
        force = self.riccati.rhs(moment, z, next);
      }
      self.advance(&mut psi, dt, force);
      #[cfg(test)]
      if matches!(forcing, Forcing::Explicit) {
        force = self.riccati.rhs(moment, z, total(&psi, &self.weights));
      }
      let end = if k == steps { 0.5 } else { 1.0 };
      sum = sum + force * (end * curve[steps - k]);
    }
    sum * dt
  }
}

impl LogChf for LiftedCf {
  fn log_chf<S: RiccatiScalar>(&self, t: f64, z: S) -> S {
    self.solve(t, z, Forcing::Solved)
  }
}
