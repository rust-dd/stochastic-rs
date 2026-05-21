//! Pure-jump Lévy and jump-diffusion Fourier models.
//!
//! Variance Gamma, CGMY, Merton jump-diffusion, Kou double-exponential, and
//! Normal Inverse Gaussian.

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
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for VarianceGammaFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let omega =
      (1.0 / self.nu) * (1.0 - self.nu * self.theta - 0.5 * self.nu * self.sigma.powi(2)).ln();
    let inner = Complex64::new(1.0, 0.0) - i * self.theta * self.nu * xi
      + Complex64::new(0.5 * self.sigma.powi(2) * self.nu, 0.0) * xi * xi;
    let psi = -inner.ln() / self.nu;
    (i * xi * ((self.r - self.q + omega) * t) + psi * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    Cumulants {
      c1: (self.r - self.q + self.theta) * t,
      c2: (self.sigma.powi(2) + self.nu * self.theta.powi(2)) * t,
      c4: 3.0
        * (self.sigma.powi(4) * self.nu
          + 2.0 * self.theta.powi(4) * self.nu.powi(3)
          + 4.0 * self.sigma.powi(2) * self.theta.powi(2) * self.nu.powi(2))
        * t,
    }
  }
}

/// Cgmy model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct CGMYFourier {
  pub c: f64,
  pub g: f64,
  pub m: f64,
  pub y: f64,
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for CGMYFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let gamma_neg_y_val = gamma_neg_y(self.y);

    let psi = self.c
      * gamma_neg_y_val
      * ((Complex64::new(self.m, 0.0) - i * xi).powf(self.y) - self.m.powf(self.y)
        + (Complex64::new(self.g, 0.0) + i * xi).powf(self.y)
        - self.g.powf(self.y));

    let omega = -self.c
      * gamma_neg_y_val
      * ((self.m - 1.0).powf(self.y) - self.m.powf(self.y) + (self.g + 1.0).powf(self.y)
        - self.g.powf(self.y));

    (i * xi * (self.r - self.q + omega) * t + psi * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let gamma_fn = |z: f64| stochastic_rs_distributions::special::gamma(z);
    Cumulants {
      c1: self.c
        * gamma_fn(1.0 - self.y)
        * (self.m.powf(self.y - 1.0) - self.g.powf(self.y - 1.0))
        * t,
      c2: self.c
        * gamma_fn(2.0 - self.y)
        * (self.m.powf(self.y - 2.0) + self.g.powf(self.y - 2.0))
        * t,
      c4: self.c
        * gamma_fn(4.0 - self.y)
        * (self.m.powf(self.y - 4.0) + self.g.powf(self.y - 4.0))
        * t,
    }
  }
}

/// Merton jump-diffusion model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct MertonJDFourier {
  pub sigma: f64,
  pub lambda: f64,
  pub mu_j: f64,
  pub sigma_j: f64,
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for MertonJDFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let m = (self.mu_j + 0.5 * self.sigma_j.powi(2)).exp() - 1.0;
    let omega = self.r - self.q - 0.5 * self.sigma.powi(2) - self.lambda * m;
    let diffusion = Complex64::new(-0.5 * self.sigma.powi(2), 0.0) * xi * xi;
    let jump_exp =
      (i * self.mu_j * xi - Complex64::new(0.5 * self.sigma_j.powi(2), 0.0) * xi * xi).exp();
    let jump = self.lambda * (jump_exp - 1.0);
    (i * xi * omega * t + (diffusion + jump) * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    Cumulants {
      c1: (self.r - self.q - 0.5 * self.sigma.powi(2)) * t + self.lambda * self.mu_j * t,
      c2: (self.sigma.powi(2) + self.lambda * (self.mu_j.powi(2) + self.sigma_j.powi(2))) * t,
      c4: 0.0,
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
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for KouFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let p = self.p_up;
    let up = p * self.eta1 / (Complex64::new(self.eta1, 0.0) - i * xi);
    let dn = (1.0 - p) * self.eta2 / (Complex64::new(self.eta2, 0.0) + i * xi);
    let jump = self.lambda * (up + dn - 1.0);

    let up0 = p * self.eta1 / (self.eta1 - 1.0);
    let dn0 = (1.0 - p) * self.eta2 / (self.eta2 + 1.0);
    let m = self.lambda * (up0 + dn0 - 1.0);
    let omega = self.r - self.q - 0.5 * self.sigma.powi(2) - m;

    let diffusion = Complex64::new(-0.5 * self.sigma.powi(2), 0.0) * xi * xi;
    (i * xi * omega * t + (diffusion + jump) * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let p = self.p_up;
    let c1_jump = self.lambda * (p / self.eta1 - (1.0 - p) / self.eta2);
    let c2_jump = self.lambda * (2.0 * p / self.eta1.powi(2) + 2.0 * (1.0 - p) / self.eta2.powi(2));
    Cumulants {
      c1: (self.r - self.q - 0.5 * self.sigma.powi(2)) * t + c1_jump * t,
      c2: (self.sigma.powi(2) + c2_jump) * t,
      c4: 0.0,
    }
  }
}

/// Normal Inverse Gaussian (NIG) Lévy model for Fourier pricing.
///
/// Characteristic exponent (Barndorff-Nielsen 1997, with the standard
/// martingale correction so $E[S_T] = S_0 e^{(r-q)T}$):
///
/// $$
/// \psi(\xi) = \delta\bigl(\sqrt{\alpha^2 - \beta^2}
///                          - \sqrt{\alpha^2 - (\beta + i\xi)^2}\bigr).
/// $$
///
/// Parameters:
/// - $\alpha > 0$: tail heaviness.
/// - $\beta \in (-\alpha, \alpha)$: skewness.
/// - $\delta > 0$: scale.
///
/// References:
/// - Barndorff-Nielsen (1997), "Normal inverse Gaussian distributions and
///   stochastic volatility modelling", Scand. J. Statist. 24, 1-13.
/// - Schoutens (2003), *Lévy Processes in Finance*, §5.3.
#[derive(Debug, Clone)]
pub struct NigFourier {
  /// Tail heaviness ($\alpha > 0$).
  pub alpha: f64,
  /// Skewness ($-\alpha < \beta < \alpha$).
  pub beta: f64,
  /// Scale ($\delta > 0$).
  pub delta: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
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

    let omega = -delta * (base - (a2 - (beta + 1.0).powi(2)).sqrt());

    (i * xi * (self.r - self.q + omega) * t + psi * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let alpha = self.alpha;
    let beta = self.beta;
    let delta = self.delta;
    let a2 = alpha * alpha;
    let b2 = beta * beta;
    let denom = (a2 - b2).sqrt();
    let denom3 = denom.powi(3);
    let c1 = (self.r - self.q) * t + delta * beta / denom * t;
    let c2 = delta * a2 / denom3 * t;
    let c4 = 3.0 * delta * a2 * (a2 + 4.0 * b2) / (a2 - b2).powf(3.5) * t;
    Cumulants { c1, c2, c4 }
  }
}
