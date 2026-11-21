//! Lévy Fourier models (Variance Gamma, CGMY, Merton, Kou, Normal Inverse Gaussian); the jump
//! part's cumulants are `κₙ = λtE[Yⁿ]`.
//! Reference: Cont & Tankov (2004), Chapman & Hall/CRC, Prop. 3.13, DOI 10.1201/9780203485217

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;
use super::gamma_neg_y;

/// Variance Gamma model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct VarianceGammaFourier {
  pub sigma: f64,
  pub theta: f64,
  pub nu: f64,
}

impl VarianceGammaFourier {
  /// Martingale drift `ω = ln(1 − θν − σ²ν/2)/ν`, which sets `E[e^{X_t}] = 1`.
  fn compensator(&self) -> f64 {
    (1.0 / self.nu) * (1.0 - self.nu * self.theta - 0.5 * self.nu * self.sigma.powi(2)).ln()
  }
}

impl FourierModelExt for VarianceGammaFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let omega = self.compensator();
    let inner = Complex64::new(1.0, 0.0) - i * self.theta * self.nu * xi
      + Complex64::new(0.5 * self.sigma.powi(2) * self.nu, 0.0) * xi * xi;
    let psi = -inner.ln() / self.nu;
    (i * xi * (omega * t) + psi * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    Cumulants {
      c1: (self.theta + self.compensator()) * t,
      c2: (self.sigma.powi(2) + self.nu * self.theta.powi(2)) * t,
      c4: Some(
        3.0
          * (self.sigma.powi(4) * self.nu
            + 2.0 * self.theta.powi(4) * self.nu.powi(3)
            + 4.0 * self.sigma.powi(2) * self.theta.powi(2) * self.nu.powi(2))
          * t,
      ),
    }
  }
}

/// Cgmy model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct CgmyFourier {
  pub c: f64,
  pub g: f64,
  pub m: f64,
  pub y: f64,
}

impl CgmyFourier {
  /// Martingale drift `ω = −CΓ(−Y)[(M − 1)^Y − M^Y + (G + 1)^Y − G^Y]`, given `Γ(−Y)`.
  fn compensator(&self, gamma_neg_y: f64) -> f64 {
    -self.c
      * gamma_neg_y
      * ((self.m - 1.0).powf(self.y) - self.m.powf(self.y) + (self.g + 1.0).powf(self.y)
        - self.g.powf(self.y))
  }
}

impl FourierModelExt for CgmyFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let gamma_neg_y_val = gamma_neg_y(self.y);

    let psi = self.c
      * gamma_neg_y_val
      * ((Complex64::new(self.m, 0.0) - i * xi).powf(self.y) - self.m.powf(self.y)
        + (Complex64::new(self.g, 0.0) + i * xi).powf(self.y)
        - self.g.powf(self.y));

    let omega = self.compensator(gamma_neg_y_val);

    (i * xi * omega * t + psi * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let gamma_fn = |z: f64| stochastic_rs_distributions::special::gamma(z);
    let mean =
      self.c * gamma_fn(1.0 - self.y) * (self.m.powf(self.y - 1.0) - self.g.powf(self.y - 1.0));
    Cumulants {
      c1: (mean + self.compensator(gamma_neg_y(self.y))) * t,
      c2: self.c
        * gamma_fn(2.0 - self.y)
        * (self.m.powf(self.y - 2.0) + self.g.powf(self.y - 2.0))
        * t,
      c4: Some(
        self.c
          * gamma_fn(4.0 - self.y)
          * (self.m.powf(self.y - 4.0) + self.g.powf(self.y - 4.0))
          * t,
      ),
    }
  }
}

/// Merton jump-diffusion model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct MertonJdFourier {
  pub sigma: f64,
  pub lambda: f64,
  pub mu_j: f64,
  pub sigma_j: f64,
}

impl MertonJdFourier {
  /// Martingale drift `ω = −σ²/2 − λ(e^{μ_J + σ_J²/2} − 1)`, which sets `E[e^{X_t}] = 1`.
  pub(super) fn compensator(&self) -> f64 {
    let m = (self.mu_j + 0.5 * self.sigma_j.powi(2)).exp() - 1.0;
    -0.5 * self.sigma.powi(2) - self.lambda * m
  }

  /// Lévy exponent of the jumps alone, `λ(E[e^{iξJ}] − 1)`.
  pub(super) fn jump_exponent(&self, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let jump_exp =
      (i * self.mu_j * xi - Complex64::new(0.5 * self.sigma_j.powi(2), 0.0) * xi * xi).exp();
    self.lambda * (jump_exp - 1.0)
  }
}

impl FourierModelExt for MertonJdFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let omega = self.compensator();
    let diffusion = Complex64::new(-0.5 * self.sigma.powi(2), 0.0) * xi * xi;
    (i * xi * omega * t + (diffusion + self.jump_exponent(xi)) * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    Cumulants {
      c1: (self.compensator() + self.lambda * self.mu_j) * t,
      c2: (self.sigma.powi(2) + self.lambda * (self.mu_j.powi(2) + self.sigma_j.powi(2))) * t,
      c4: Some(
        self.lambda
          * t
          * (self.mu_j.powi(4)
            + 6.0 * self.mu_j.powi(2) * self.sigma_j.powi(2)
            + 3.0 * self.sigma_j.powi(4)),
      ),
    }
  }
}

/// Kou double-exponential jump-diffusion model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct KouFourier {
  pub sigma: f64,
  pub lambda: f64,
  pub p_up: f64,
  pub eta1: f64,
  pub eta2: f64,
}

impl KouFourier {
  /// Martingale drift `ω = −σ²/2 − λ(pη₁/(η₁ − 1) + (1 − p)η₂/(η₂ + 1) − 1)`.
  pub(super) fn compensator(&self) -> f64 {
    let p = self.p_up;
    let up0 = p * self.eta1 / (self.eta1 - 1.0);
    let dn0 = (1.0 - p) * self.eta2 / (self.eta2 + 1.0);
    let m = self.lambda * (up0 + dn0 - 1.0);
    -0.5 * self.sigma.powi(2) - m
  }

  /// Lévy exponent of the jumps alone, `λ(E[e^{iξJ}] − 1)`.
  pub(super) fn jump_exponent(&self, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let p = self.p_up;
    let up = p * self.eta1 / (Complex64::new(self.eta1, 0.0) - i * xi);
    let dn = (1.0 - p) * self.eta2 / (Complex64::new(self.eta2, 0.0) + i * xi);
    self.lambda * (up + dn - 1.0)
  }
}

impl FourierModelExt for KouFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let omega = self.compensator();

    let diffusion = Complex64::new(-0.5 * self.sigma.powi(2), 0.0) * xi * xi;
    (i * xi * omega * t + (diffusion + self.jump_exponent(xi)) * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let p = self.p_up;
    let c1_jump = self.lambda * (p / self.eta1 - (1.0 - p) / self.eta2);
    let c2_jump = self.lambda * (2.0 * p / self.eta1.powi(2) + 2.0 * (1.0 - p) / self.eta2.powi(2));
    Cumulants {
      c1: (self.compensator() + c1_jump) * t,
      c2: (self.sigma.powi(2) + c2_jump) * t,
      c4: Some(
        24.0
          * self.lambda
          * t
          * (self.p_up / self.eta1.powi(4) + (1.0 - self.p_up) / self.eta2.powi(4)),
      ),
    }
  }
}

/// Normal Inverse Gaussian Lévy model, martingale-corrected to `E[S_T] = S₀e^{(r−q)T}`; `α > 0`
/// tail heaviness, `β ∈ (−α, α)` skew, `δ > 0` scale.
///
/// $$
/// \psi(\xi) = \delta\bigl(\sqrt{\alpha^2 - \beta^2}
///                          - \sqrt{\alpha^2 - (\beta + i\xi)^2}\bigr).
/// $$
///
/// Reference: Barndorff-Nielsen (1997), Scand. J. Stat. 24, 1–13
/// Reference: Schoutens (2003), Lévy Processes in Finance, §5.3
#[derive(Debug, Clone)]
pub struct NigFourier {
  /// Tail heaviness ($\alpha > 0$).
  pub alpha: f64,
  /// Skewness ($-\alpha < \beta < \alpha$).
  pub beta: f64,
  /// Scale ($\delta > 0$).
  pub delta: f64,
}

impl NigFourier {
  /// Martingale drift `ω = −δ(√(α² − β²) − √(α² − (β + 1)²))`, which sets `E[e^{X_t}] = 1`.
  fn compensator(&self) -> f64 {
    let a2 = self.alpha * self.alpha;
    let base = (a2 - self.beta * self.beta).sqrt();
    -self.delta * (base - (a2 - (self.beta + 1.0).powi(2)).sqrt())
  }
}

impl FourierModelExt for NigFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let alpha = self.alpha;
    let beta = self.beta;
    let delta = self.delta;
    let a2 = alpha * alpha;

    let base = (a2 - beta * beta).sqrt();
    let shifted = Complex64::new(beta, 0.0) + i * xi;
    let branch = (Complex64::new(a2, 0.0) - shifted * shifted).sqrt();
    let psi = delta * (Complex64::new(base, 0.0) - branch);

    (i * xi * self.compensator() * t + psi * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let alpha = self.alpha;
    let beta = self.beta;
    let delta = self.delta;
    let a2 = alpha * alpha;
    let b2 = beta * beta;
    let denom = (a2 - b2).sqrt();
    let denom3 = denom.powi(3);
    let c1 = (delta * beta / denom + self.compensator()) * t;
    let c2 = delta * a2 / denom3 * t;
    let c4 = 3.0 * delta * a2 * (a2 + 4.0 * b2) / (a2 - b2).powf(3.5) * t;
    Cumulants {
      c1,
      c2,
      c4: Some(c4),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// κ₄ = λ t E[Y⁴] (Cont & Tankov 2004, Prop. 3.13), pinned to mpmath's 4th derivative at 0 of
  /// each model's own log-chf.
  #[test]
  fn merton_and_kou_fourth_cumulants_are_the_jump_moments() {
    let merton = |sigma, lambda, mu_j, sigma_j, t| {
      let model = MertonJdFourier {
        sigma,
        lambda,
        mu_j,
        sigma_j,
      };
      model.cumulants(t).c4.unwrap()
    };
    let kou = |sigma, lambda, p_up, eta1, eta2, t| {
      let model = KouFourier {
        sigma,
        lambda,
        p_up,
        eta1,
        eta2,
      };
      model.cumulants(t).c4.unwrap()
    };
    for (got, want) in [
      (merton(0.2, 0.5, -0.1, 0.15, 2.0), 0.00296875),
      (merton(0.12, 0.4, -0.12, 0.18, 1.0), 0.0024624),
      (kou(0.2, 0.8, 0.4, 10.0, 5.0, 1.5), 0.0288),
      (kou(0.15, 3.0, 0.2, 25.0, 10.0, 1.0), 0.005796864),
    ] {
      assert!(
        ((got - want) / want).abs() < 1e-14,
        "c4 = {got}, mpmath {want}"
      );
    }
  }
}
