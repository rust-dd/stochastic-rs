//! The Heston PDE with correlation by in 't Hout & Foulon's ADI: sinh meshes (§2.2), the Douglas,
//! Craig–Sneyd, MCS (the default, at `θ = ⅓`) and HV steps with Rannacher damping (§2.4–§2.6).
//!
//! $$
//! u_t = \tfrac12 s^2 v\,u_{ss} + \rho\sigma s v\,u_{sv} + \tfrac12\sigma^2 v\,u_{vv} + (r_d - r_f)\,s\,u_s + \kappa(\eta - v)\,u_v - r_d u
//! $$
//!
//! Reference: in 't Hout & Foulon (2010), Int. J. Numer. Anal. Model. 7, 303–320, arXiv:0811.3427

mod grid;
mod operators;
mod schemes;

pub use schemes::AdiScheme;

use self::grid::origin_centred_mesh;
use self::grid::strike_centred_mesh;
use self::operators::HestonCoefficients;
use self::operators::Operators;
use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::heston::HestonParams;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Upper `v` boundary, the paper's `V = 5`; the mesh reaches at least twice the initial variance.
const VARIANCE_CAP: f64 = 5.0;
/// Upper `s` boundary as a multiple of the strike, the paper's `S = 8K`; at least twice the spot.
const SPOT_CAP_STRIKES: f64 = 8.0;

/// Heston ADI finite-difference pricer; the query's `r` and `q` are the paper's `r_d` and `r_f`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonAdiPricer {
  /// `theta` is the paper's long-run variance `η`.
  params: HestonParams,
  m1: usize,
  m2: usize,
  steps: usize,
  /// Splitting scheme.
  pub scheme: AdiScheme,
  /// Scheme parameter `θ`; `None` takes the scheme's recommended value.
  pub adi_theta: Option<f64>,
  /// Rannacher start-up damping.
  pub damping: bool,
  barrier: Option<f64>,
}

impl HestonAdiPricer {
  /// In 't Hout & Foulon's mesh (`m1 = 100`, `m2 = 50`, 50 damped MCS steps at `θ = ⅓`); panics
  /// unless `params` is valid with `kappa, theta > 0`. Their Feller assumption is not imposed.
  pub fn new(params: HestonParams) -> Self {
    let params = params.checked();
    let HestonParams { kappa, theta, .. } = params;
    assert!(
      kappa > 0.0,
      "kappa must satisfy `kappa > 0.0`, got kappa = {kappa:?}"
    );
    assert!(
      theta > 0.0,
      "theta must satisfy `theta > 0.0`, got theta = {theta:?}"
    );
    Self {
      params,
      m1: 100,
      m2: 50,
      steps: 50,
      scheme: AdiScheme::ModifiedCraigSneyd,
      adi_theta: None,
      damping: true,
      barrier: None,
    }
  }

  pub fn params(&self) -> HestonParams {
    self.params
  }

  /// Mesh intervals in `s` and in `v`, and time steps.
  pub fn grid(&self) -> (usize, usize, usize) {
    (self.m1, self.m2, self.steps)
  }

  /// The down-and-out barrier on the spot, if any.
  pub fn barrier(&self) -> Option<f64> {
    self.barrier
  }

  /// Mesh intervals and time steps; panics unless `m1 >= 4`, `m2 >= 4` and `steps >= 1`.
  pub fn with_grid(mut self, m1: usize, m2: usize, steps: usize) -> Self {
    assert!(m1 >= 4, "m1 must satisfy `m1 >= 4`, got m1 = {m1}");
    assert!(m2 >= 4, "m2 must satisfy `m2 >= 4`, got m2 = {m2}");
    assert!(
      steps >= 1,
      "steps must satisfy `steps >= 1`, got steps = {steps}"
    );
    self.m1 = m1;
    self.m2 = m2;
    self.steps = steps;
    self
  }

  /// Splitting scheme, with its recommended `θ` unless overridden.
  pub fn with_scheme(mut self, scheme: AdiScheme) -> Self {
    self.scheme = scheme;
    self
  }

  /// Explicit scheme parameter `θ`.
  pub fn with_adi_theta(mut self, theta: f64) -> Self {
    self.adi_theta = Some(theta);
    self
  }

  /// Switches the Rannacher damping on or off.
  pub fn with_damping(mut self, damping: bool) -> Self {
    self.damping = damping;
    self
  }

  /// Down-and-out barrier `B ∈ (0, K)`: the lower `s` boundary moves to `B` with `u(B, v, t) = 0`,
  /// and `u(s, V, t) = (s − B) e^{−r_f t}` (§2.6); panics unless `B` is finite and positive.
  pub fn with_barrier(mut self, barrier: f64) -> Self {
    assert!(
      barrier.is_finite() && barrier > 0.0,
      "barrier must satisfy `barrier.is_finite() && barrier > 0.0`, got barrier = {barrier:?}"
    );
    self.barrier = Some(barrier);
    self
  }

  /// Effective scheme parameter.
  pub fn scheme_theta(&self) -> f64 {
    self
      .adi_theta
      .unwrap_or_else(|| self.scheme.default_theta())
  }

  /// Solves the PDE for the call payoff and reads the price at `(s, v₀)`; at `tau = 0` the
  /// payoff, zero at or below a barrier, and NaN past expiry.
  fn solve_call(&self, s: f64, k: f64, r_d: f64, r_f: f64, tau: f64) -> f64 {
    let p = self.params;
    let alive = self.barrier.is_none_or(|barrier| s > barrier);
    let call = || vanilla_payoff(OptionType::Call, if alive || s.is_nan() { s } else { k }, k);
    if let Some(value) = expired(tau, &[r_d, r_f], call) {
      return value;
    }
    if !(s > 0.0 && k > 0.0 && tau > 0.0 && p.v0 >= 0.0)
      || [s, k, tau, r_d, r_f, p.v0].iter().any(|v| !v.is_finite())
    {
      return f64::NAN;
    }
    let lower = self.barrier.unwrap_or(0.0);
    if lower >= k || s <= lower {
      return if s <= lower { 0.0 } else { f64::NAN };
    }
    let spot_cap = (SPOT_CAP_STRIKES * k).max(2.0 * s);
    let variance_cap = VARIANCE_CAP.max(2.0 * p.v0);
    if !spot_cap.is_finite() || !variance_cap.is_finite() {
      return f64::NAN;
    }
    let s_mesh = strike_centred_mesh(lower, spot_cap, k, self.m1);
    let v_mesh = origin_centred_mesh(variance_cap, self.m2);
    let ops = Operators::new(
      s_mesh,
      v_mesh,
      HestonCoefficients {
        kappa: p.kappa,
        eta: p.theta,
        sigma: p.sigma,
        rho: p.rho,
        r_d,
        r_f,
      },
    );
    let mut u0 = vec![0.0; ops.len()];
    for j in 0..ops.m2 {
      for i in 1..=ops.m1 {
        u0[j * ops.m1 + (i - 1)] = (ops.s[i] - k).max(0.0);
      }
    }
    let u = schemes::march(
      &ops,
      self.scheme,
      self.scheme_theta(),
      self.damping,
      self.steps,
      tau,
      u0,
    );
    ops.interpolate(&u, s, p.v0, (-r_f * tau).exp())
  }
}

impl ModelPricer for HestonAdiPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.solve_call(s, k, r, q, tau)
  }

  /// European put-call parity, exact at carry `r − q`; NaN with a barrier, as a down-and-out put
  /// has no parity with the down-and-out call.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.barrier.is_some() {
      return f64::NAN;
    }
    self.price_call(s, k, r, q, tau) - s * (-q * tau).exp() + k * (-r * tau).exp()
  }
}

/// A European vanilla call on the default forward without a barrier; with one the instance prices a
/// down-and-out call, so its forward is NaN.
impl VanillaEuropeanCall for HestonAdiPricer {
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.barrier.is_some() {
      f64::NAN
    } else {
      s * ((r - q) * tau).exp()
    }
  }
}

#[cfg(test)]
mod tests;
