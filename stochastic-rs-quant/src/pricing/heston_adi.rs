//! # Heston ADI
//!
//! $$
//! u_t = \tfrac12 s^2 v\,u_{ss} + \rho\sigma s v\,u_{sv} + \tfrac12\sigma^2 v\,u_{vv} + (r_d - r_f)\,s\,u_s + \kappa(\eta - v)\,u_v - r_d u
//! $$
//!
//! Finite-difference solution of the Heston PDE with correlation by the
//! Alternating Direction Implicit schemes of in 't Hout & Foulon (2008):
//! sinh-stretched meshes clustering at the strike and at `v = 0` (§2.2,
//! `S = 8K`, `V = 5`, `c = K/5`, `d = V/500`), second-order central
//! stencils with upwinding of `u_v` where the `v`-flow points outward, the
//! call boundary conditions `u(0, v) = 0`, `u_s(S, v) = e^{−r_f t}`,
//! `u(s, V) = s e^{−r_f t}`, and the Douglas, Craig–Sneyd, Modified
//! Craig–Sneyd and Hundsdorfer–Verwer time steppers of §2.4 with the
//! Rannacher damping of §2.5. The default is the paper's recommendation:
//! MCS at `θ = ⅓` with damping. A down-and-out barrier moves the lower
//! `s` boundary to the barrier (§2.6).
//!
//! The struct holds model and method state; the query `(s, k, r, q, τ)`
//! travels as arguments with `r = r_d` and `q = r_f`, and the price is read
//! off the grid by bilinear interpolation at `(s, v₀)`.
//!
//! Reference: in 't Hout, K. J. & Foulon, S. (2010), *ADI finite difference
//! schemes for option pricing in the Heston model with correlation*,
//! International Journal of Numerical Analysis and Modeling 7(2), 303–320;
//! arXiv:0811.3427.

mod grid;
mod operators;
mod schemes;

pub use schemes::AdiScheme;

use self::grid::origin_centred_mesh;
use self::grid::strike_centred_mesh;
use self::operators::HestonCoefficients;
use self::operators::Operators;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Upper `v` boundary of the computational domain (`V = 5` in the paper).
const VARIANCE_CAP: f64 = 5.0;
/// Upper `s` boundary as a multiple of the strike (`S = 8K`).
const SPOT_CAP_STRIKES: f64 = 8.0;

/// Heston ADI finite-difference pricer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonAdiPricer {
  /// Initial variance `v₀`.
  pub v0: f64,
  /// Mean reversion `κ`.
  pub kappa: f64,
  /// Long-run variance (`η` in the paper).
  pub theta: f64,
  /// Volatility of variance `σ`.
  pub sigma: f64,
  /// Correlation `ρ`.
  pub rho: f64,
  /// Mesh intervals in `s`.
  pub m1: usize,
  /// Mesh intervals in `v`.
  pub m2: usize,
  /// Time steps.
  pub steps: usize,
  /// Splitting scheme.
  pub scheme: AdiScheme,
  /// Scheme parameter `θ`; `None` takes the scheme's recommended value.
  pub adi_theta: Option<f64>,
  /// Rannacher start-up damping.
  pub damping: bool,
  /// Down-and-out barrier on the spot, if any.
  pub barrier: Option<f64>,
}

impl HestonAdiPricer {
  /// Pricer with the paper's default numerics: `m1 = 100`, `m2 = 50`,
  /// 50 time steps, MCS at `θ = ⅓` with damping.
  pub fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Self {
    assert!(
      v0 >= 0.0 && kappa > 0.0 && theta > 0.0 && sigma > 0.0,
      "Heston parameters must be positive"
    );
    assert!(rho.abs() <= 1.0, "rho must lie in [-1, 1]");
    Self {
      v0,
      kappa,
      theta,
      sigma,
      rho,
      m1: 100,
      m2: 50,
      steps: 50,
      scheme: AdiScheme::ModifiedCraigSneyd,
      adi_theta: None,
      damping: true,
      barrier: None,
    }
  }

  /// Mesh intervals and time steps.
  pub fn with_grid(mut self, m1: usize, m2: usize, steps: usize) -> Self {
    assert!(
      m1 >= 4 && m2 >= 4 && steps >= 1,
      "grid needs at least four intervals per direction and one step"
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

  /// Down-and-out barrier `B ∈ (0, K)`: the lower `s` boundary moves to `B`
  /// with `u(B, v, t) = 0` and `u(s, V, t) = (s − B) e^{−r_f t}` (§2.6).
  pub fn with_barrier(mut self, barrier: f64) -> Self {
    assert!(barrier > 0.0, "barrier must be positive");
    self.barrier = Some(barrier);
    self
  }

  /// Effective scheme parameter.
  pub fn scheme_theta(&self) -> f64 {
    self
      .adi_theta
      .unwrap_or_else(|| self.scheme.default_theta())
  }

  /// Solves the PDE for the call payoff and reads the price at `(s, v₀)`.
  fn solve_call(&self, s: f64, k: f64, r_d: f64, r_f: f64, tau: f64) -> f64 {
    if !(s > 0.0 && k > 0.0 && tau > 0.0) || !s.is_finite() || !k.is_finite() {
      return f64::NAN;
    }
    let lower = self.barrier.unwrap_or(0.0);
    if lower >= k || s <= lower {
      return if s <= lower { 0.0 } else { f64::NAN };
    }
    let s_mesh = strike_centred_mesh(lower, SPOT_CAP_STRIKES * k, k, self.m1);
    let v_mesh = origin_centred_mesh(VARIANCE_CAP, self.m2);
    let ops = Operators::new(
      s_mesh,
      v_mesh,
      HestonCoefficients {
        kappa: self.kappa,
        eta: self.theta,
        sigma: self.sigma,
        rho: self.rho,
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
    ops.interpolate(&u, s, self.v0, (-r_f * tau).exp())
  }
}

impl ModelPricer for HestonAdiPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.solve_call(s, k, r, q, tau)
  }

  /// Vanilla puts come from European put-call parity, which is exact here
  /// (carry `r − q`, European exercise); a down-and-out put has no parity
  /// relation with the down-and-out call and returns `NaN`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.barrier.is_some() {
      return f64::NAN;
    }
    self.price_call(s, k, r, q, tau) - s * (-q * tau).exp() + k * (-r * tau).exp()
  }
}

/// A European vanilla call at the default forward without a barrier; with a
/// barrier the instance prices a down-and-out call and reports `NaN`, case
/// 2 of the failure convention.
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
