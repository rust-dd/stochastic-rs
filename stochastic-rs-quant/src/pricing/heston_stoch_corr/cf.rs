//! Characteristic function for the Heston model with stochastic correlation.
//!
//! ODE system (Teng et al. 2016, Lemma 3.1) integrated by RK4.

use num_complex::Complex64;

use super::model::HestonStochCorrPricer;

impl HestonStochCorrPricer {
  /// Evaluate the characteristic function φ(u) via RK4 integration
  /// of the ODE system for (A, C, D).
  ///
  /// ODE system (Lemma 3.1):
  /// ```text
  /// dD/dτ = −½u² + ½σ_v²·D² − ½iu − κ_v·D
  /// dC/dτ = σ_v·v₀·iu·D − κ_r·C
  /// dA/dτ = iu·(r−q) + κ_v·θ_v·D + κ_r·μ_r·C + ½σ_r²·C² + σ_r·ρ₂·m·iu·C
  /// ```
  /// where m = √(θ_v − σ_v²/(8κ_v)). The paper writes the `dA` drift as
  /// `iu·r` because it carries no dividend yield; here it is the full
  /// risk-neutral log-drift `iu·(r−q)`, which is what makes the returned
  /// exponential the characteristic function of `ln S_τ` and nothing else.
  pub fn char_func(&self, u: f64, s: f64, r: f64, q: f64, tau: f64) -> Complex64 {
    self.char_func_complex(Complex64::new(u, 0.0), s, r, q, tau)
  }

  /// Characteristic function accepting complex u (needed for Carr-Madan
  /// where we evaluate at u − (α+1)i).
  pub fn char_func_complex(&self, u: Complex64, s: f64, r: f64, q: f64, tau: f64) -> Complex64 {
    let x0 = s.ln();
    let iu = Complex64::i() * u;
    // Dividend yield: enters the log-stock drift as (r - q), not the discount.
    let drift = r - q;
    let kv = self.kappa_v;
    let mv = self.theta_v;
    let sv = self.sigma_v;
    let v0 = self.v0;
    let kr = self.kappa_r;
    let mr = self.mu_r;
    let sr = self.sigma_r;
    let rho2 = self.rho2;

    // Auxiliary: m = √(θ_v − σ_v²/(8κ_v))
    let m_aux = Complex64::new(mv - sv * sv / (8.0 * kv), 0.0).sqrt();

    // Adaptive step count
    let u_mag = u.norm();
    let n_steps = (200.0 * tau * (1.0 + u_mag * 0.1)).ceil().max(100.0) as usize;
    let dt = tau / n_steps as f64;

    let rhs = |_a: Complex64, c: Complex64, d: Complex64| -> (Complex64, Complex64, Complex64) {
      let da =
        iu * drift + kv * mv * d + kr * mr * c + 0.5 * sr * sr * c * c + sr * rho2 * m_aux * iu * c;
      let dc = sv * v0 * iu * d - kr * c;
      let dd = -0.5 * u * u + 0.5 * sv * sv * d * d - 0.5 * iu - kv * d;
      (da, dc, dd)
    };

    let mut a = Complex64::new(0.0, 0.0);
    let mut c = Complex64::new(0.0, 0.0);
    let mut d = Complex64::new(0.0, 0.0);

    for _ in 0..n_steps {
      let (k1a, k1c, k1d) = rhs(a, c, d);
      let (k2a, k2c, k2d) = rhs(a + 0.5 * dt * k1a, c + 0.5 * dt * k1c, d + 0.5 * dt * k1d);
      let (k3a, k3c, k3d) = rhs(a + 0.5 * dt * k2a, c + 0.5 * dt * k2c, d + 0.5 * dt * k2d);
      let (k4a, k4c, k4d) = rhs(a + dt * k3a, c + dt * k3c, d + dt * k3d);

      a += dt / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
      c += dt / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c);
      d += dt / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d);
    }

    // No discount factor here: `da` already carries the full risk-neutral
    // drift `iu(r - q)`, so this exponential is φ(u) = E[exp(iu·ln S_τ)]
    // itself. Discounting is the caller's job — `price_call_carr_madan`
    // applies the transform's single `e^{-rτ}`.
    (a + iu * x0 + c * self.rho0 + d * v0).exp()
  }
}
