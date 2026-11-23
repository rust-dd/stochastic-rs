//! The fractional Adams scheme for the rough Heston Riccati equation, its corrector solved exactly.
//!
//! $$
//! \hat h_{k+1} = \sum_{0\le j\le k} a_{j,k+1}F(z,\hat h_j) + a_{k+1,k+1}F(z,\hat h_{k+1}),\qquad
//! a_{j,k+1} = \tfrac{\Delta^\alpha}{\Gamma(\alpha+2)}
//! \bigl((k-j+2)^{\alpha+1} + (k-j)^{\alpha+1} - 2(k-j+1)^{\alpha+1}\bigr)
//! $$
//!
//! Reference: El Euch & Rosenbaum (2019), Math. Finance 29(1), 3–38, (26)–(27), DOI 10.1111/mafi.12173
//! Reference: Diethelm et al. (2002), Nonlinear Dyn. 29, 3–22, DOI 10.1023/A:1016592219341
//! Reference: Diethelm et al. (2004), Numer. Algorithms 36, 31–52, DOI 10.1023/B:NUMA.0000027736.85078.be

use ndarray::Array1;
use stochastic_rs_distributions::special::gamma;

use super::Riccati;
use super::scalar::RiccatiScalar;

/// How (26)'s implicit term is met: `Solved` takes its quadratic's root, stable at any step.
#[derive(Clone, Copy, Debug)]
pub(super) enum Corrector {
  Solved,
  /// The predictor put into (26) once: explicit, so stable only while `ν|A|Δ^α ≲ 1.3`.
  #[cfg(test)]
  Predicted,
}

/// (27)'s weights without `Δ^α/Γ(α+2)`, and the same rule at order `1 − α` for `I^{1−α}h(t_N)`.
#[derive(Clone, Debug)]
pub(super) struct Adams {
  alpha: f64,
  steps: usize,
  /// `a_{0,k+1}`'s factor, by `k`.
  pub(super) first: Array1<f64>,
  /// `a_{j,k+1}`'s factor, by `k − j`.
  pub(super) second_difference: Array1<f64>,
  fractional: Array1<f64>,
}

fn first_weight(k: usize, order: f64) -> f64 {
  let k = k as f64;
  k.powf(order + 1.0) - (k - order) * (k + 1.0).powf(order)
}

fn second_difference(m: usize, order: f64) -> f64 {
  let m = m as f64;
  (m + 2.0).powf(order + 1.0) + m.powf(order + 1.0) - 2.0 * (m + 1.0).powf(order + 1.0)
}

impl Adams {
  /// `alpha` in `(½, 1]`, 1 being the classical equation; `steps ≥ 1`.
  pub(super) fn new(alpha: f64, steps: usize) -> Self {
    let rest = 1.0 - alpha;
    let last = steps - 1;
    let fractional = (0..=steps).map(|j| match j {
      0 => first_weight(last, rest),
      j if j == steps => 1.0,
      j => second_difference(last - j, rest),
    });
    Self {
      alpha,
      steps,
      first: (0..steps).map(|k| first_weight(k, alpha)).collect(),
      second_difference: (0..steps).map(|m| second_difference(m, alpha)).collect(),
      fractional: fractional.collect(),
    }
  }

  /// `ĥ` on the grid `t_k = kt/N`, from `ĥ₀ = 0`.
  pub(super) fn path<S: RiccatiScalar>(
    &self,
    t: f64,
    z: S,
    riccati: &Riccati,
    corrector: Corrector,
  ) -> Vec<S> {
    let dt = t / self.steps as f64;
    let c = dt.powf(self.alpha) / gamma(self.alpha + 2.0);
    let moment = Riccati::moment(z);
    let square = 0.5 * riccati.nu * riccati.nu * c;
    let linear = (z * (riccati.rho * riccati.nu) - S::real(riccati.kappa)) * c - S::real(1.0);
    let mut h = Vec::with_capacity(self.steps + 1);
    let mut f = Vec::with_capacity(self.steps + 1);
    h.push(S::ZERO);
    f.push(moment);
    for k in 0..self.steps {
      let history = (1..=k).fold(f[0] * self.first[k], |sum, j| {
        sum + f[j] * self.second_difference[k - j]
      });
      let next = match corrector {
        Corrector::Solved => {
          // The root whose slope is `−√disc`: the explicit value as `c → 0`, `r₋/ν` as `c → ∞`.
          let constant = (history + moment) * c;
          let root = (linear * linear - constant * (4.0 * square)).sqrt();
          if linear.re() <= 0.0 {
            constant * 2.0 / (root - linear)
          } else {
            (-linear - root) / S::real(2.0 * square)
          }
        }
        #[cfg(test)]
        Corrector::Predicted => {
          let rectangle =
            |m: usize| ((m + 1) as f64).powf(self.alpha) - (m as f64).powf(self.alpha);
          let predictor = (0..=k).fold(S::ZERO, |sum, j| sum + f[j] * rectangle(k - j));
          let predicted = predictor * (c * (self.alpha + 1.0));
          (history + riccati.rhs(moment, z, predicted)) * c
        }
      };
      h.push(next);
      f.push(riccati.rhs(moment, z, next));
    }
    h
  }

  /// `(I¹h, I^{1−α}h)` at `t` of a grid function: the trapezoid, and (27)'s rule at `1 − α`.
  pub(super) fn integrals<S: RiccatiScalar>(&self, t: f64, h: &[S]) -> (S, S) {
    let dt = t / self.steps as f64;
    let rest = 1.0 - self.alpha;
    let ends = (h[0] + h[self.steps]) * 0.5;
    let trapezoid = h.iter().fold(-ends, |sum, &value| sum + value) * dt;
    let weighted = h.iter().zip(&self.fractional);
    let fractional = weighted.fold(S::ZERO, |sum, (&value, &weight)| sum + value * weight);
    (trapezoid, fractional * (dt.powf(rest) / gamma(rest + 2.0)))
  }
}
