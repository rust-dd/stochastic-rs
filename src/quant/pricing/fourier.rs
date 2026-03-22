//! # Fourier
//!
//! $$
//! C(K)=\frac{e^{-\alpha k}}{\pi}\int_0^\infty e^{-ivk}\,\psi_T(v)\,dv,\quad
//! \psi_T(v)=\frac{e^{-rT}\,\phi_T(v-(1+\alpha)i)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}
//! $$
//!
//! Source:
//! - Carr, P. & Madan, D. (1999), "Option valuation using the fast Fourier transform"
//! - Lewis, A. (2001), "A Simple Option Formula for General Jump-Diffusion and Other
//!   Exponential Lévy Processes"
//! - Gil-Pelaez, J. (1951), "Note on the inversion theorem"
//!
use std::f64::consts::FRAC_1_PI;
use std::f64::consts::PI;

use ndarray::Array1;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex64;
use quadrature::double_exponential;

/// Cumulants of the log-price distribution.
pub struct Cumulants {
  /// First cumulant (mean of log-return).
  pub c1: f64,
  /// Second cumulant (variance of log-return).
  pub c2: f64,
  /// Fourth cumulant.
  pub c4: f64,
}

/// Trait for models that expose a characteristic function of the log-price,
/// enabling pricing via Fourier inversion.
pub trait FourierModelExt {
  /// Characteristic function of the log-price $\ln(S_T/S_0)$:
  /// $\phi_T(\xi) = E[e^{i\xi \ln(S_T/S_0)}]$.
  ///
  /// Must already include the risk-neutral drift, i.e.\ the expectation is
  /// under the risk-neutral measure $\mathbb{Q}$.
  fn chf(&self, t: f64, xi: Complex64) -> Complex64;

  /// Cumulants of the log-return for maturity `t`.
  fn cumulants(&self, t: f64) -> Cumulants;
}

/// Carr–Madan FFT pricer.
///
/// Evaluates call prices across a grid of strikes with a single FFT pass.
pub struct CarrMadanPricer {
  /// Number of FFT points (must be a power of 2).
  pub n: usize,
  /// Dampening parameter (typically 0.75 for calls).
  pub alpha: f64,
  /// Integration step size (default 0.25).
  pub eta: f64,
}

impl Default for CarrMadanPricer {
  fn default() -> Self {
    Self {
      n: 4096,
      alpha: 0.75,
      eta: 0.25,
    }
  }
}

impl CarrMadanPricer {
  pub fn new(n: usize, alpha: f64) -> Self {
    assert!(n.is_power_of_two());
    Self {
      n,
      alpha,
      eta: 0.25,
    }
  }

  /// Compute call prices on the FFT strike grid.
  ///
  /// Returns `(log_strikes, call_prices)` where `log_strikes` are
  /// $k_u = b + \lambda u$.
  pub fn price_call_surface(
    &self,
    model: &dyn FourierModelExt,
    s: f64,
    r: f64,
    t: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    let n = self.n;
    let alpha = self.alpha;
    let eta = self.eta;
    let lambda = 2.0 * PI / (n as f64 * eta);
    let b = -0.5 * n as f64 * lambda;

    let i_unit = Complex64::i();
    let ln_s = s.ln();
    let disc = (-r * t).exp();

    let mut input = Array1::<Complex64>::zeros(n);

    for j in 0..n {
      let v_j = eta * j as f64;

      let simpson = if j == 0 {
        eta / 3.0
      } else if j % 2 == 1 {
        eta * 4.0 / 3.0
      } else {
        eta * 2.0 / 3.0
      };

      let xi = Complex64::new(v_j, -(alpha + 1.0));
      let phi = model.chf(t, xi) * (i_unit * xi * ln_s).exp();

      let denom = Complex64::new(alpha * alpha + alpha - v_j * v_j, (2.0 * alpha + 1.0) * v_j);

      let psi = disc * phi / denom;

      input[j] = (i_unit * v_j * b).exp() * psi * simpson;
    }

    let handler = FftHandler::<f64>::new(n);
    let mut output = Array1::<Complex64>::zeros(n);
    ndfft(&input, &mut output, &handler, 0);

    let mut log_strikes = Array1::<f64>::zeros(n);
    let mut prices = Array1::<f64>::zeros(n);
    for u in 0..n {
      let k_u = b + lambda * u as f64;
      log_strikes[u] = k_u;
      prices[u] = ((-alpha * k_u).exp() * output[u].re / PI).max(0.0);
    }

    (log_strikes, prices)
  }

  /// Price a single call option by interpolating the FFT surface.
  pub fn price_call(&self, model: &dyn FourierModelExt, s: f64, k: f64, r: f64, t: f64) -> f64 {
    let (log_strikes, prices) = self.price_call_surface(model, s, r, t);
    let target = k.ln();

    // Linear interpolation
    for i in 0..log_strikes.len() - 1 {
      if log_strikes[i] <= target && target <= log_strikes[i + 1] {
        let w = (target - log_strikes[i]) / (log_strikes[i + 1] - log_strikes[i]);
        return prices[i] * (1.0 - w) + prices[i + 1] * w;
      }
    }
    0.0
  }

  /// Price a single put option via put-call parity.
  pub fn price_put(
    &self,
    model: &dyn FourierModelExt,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
  ) -> f64 {
    let call = self.price_call(model, s, k, r, t);
    call - s * (-q * t).exp() + k * (-r * t).exp()
  }
}

/// Gil-Pelaez (1951) quadrature pricer.
///
/// Uses numerical quadrature on the two half-integrals $P_1,P_2$.
pub struct GilPelaezPricer;

impl GilPelaezPricer {
  /// Price a European call using double-exponential quadrature.
  pub fn price_call(model: &dyn FourierModelExt, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    let ln_ks = (k / s).ln();

    let i_unit = Complex64::i();

    // P1 integral: using the measure shift φ(u-i) / φ(-i)
    let integrand_p1 = |u: f64| -> f64 {
      if u.abs() < 1e-12 {
        return 0.0;
      }
      let xi = Complex64::new(u, -1.0);
      let phi = model.chf(t, xi);
      let phi_neg_i = model.chf(t, Complex64::new(0.0, -1.0));
      if phi_neg_i.norm() < 1e-30 {
        return 0.0;
      }
      let kernel = (-i_unit * u * ln_ks).exp() / (i_unit * u);
      (kernel * phi / phi_neg_i).re
    };

    let integrand_p2 = |u: f64| -> f64 {
      if u.abs() < 1e-12 {
        return 0.0;
      }
      let xi = Complex64::new(u, 0.0);
      let phi = model.chf(t, xi);
      let kernel = (-i_unit * u * ln_ks).exp() / (i_unit * u);
      (kernel * phi).re
    };

    let p1 =
      0.5 + FRAC_1_PI * double_exponential::integrate(integrand_p1, 1e-8, 100.0, 1e-8).integral;
    let p2 =
      0.5 + FRAC_1_PI * double_exponential::integrate(integrand_p2, 1e-8, 100.0, 1e-8).integral;

    let call = s * (-q * t).exp() * p1 - k * (-r * t).exp() * p2;
    call.max(0.0)
  }
}

/// Lewis (2001) single-strike quadrature pricer.
///
/// $$
/// C=S F e^{-rT}-\frac{\sqrt{SK}\,e^{-rT}}{\pi}
/// \int_0^\infty\!\operatorname{Re}\!\left[\frac{\phi_T(u-\tfrac i2)\,e^{-iu\ln(K/S)}}{u^2+\tfrac14}\right]du
/// $$
pub struct LewisPricer;

impl LewisPricer {
  pub fn price_call(model: &dyn FourierModelExt, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    let ln_ks = (k / s).ln();
    let disc = (-r * t).exp();
    let fwd_factor = ((r - q) * t).exp();
    let sqrt_sk = (s * k).sqrt();

    let i_unit = Complex64::i();

    let integrand = |u: f64| -> f64 {
      let xi = Complex64::new(u, -0.5);
      // chf gives CF of log-return ln(S_T/S_0), Lewis uses it directly
      let phi = model.chf(t, xi);
      let kernel = (-i_unit * u * ln_ks).exp() / (u * u + 0.25);
      (kernel * phi).re
    };

    let integral = double_exponential::integrate(integrand, 1e-8, 100.0, 1e-8).integral;

    let call = s * disc * fwd_factor - sqrt_sk * disc * FRAC_1_PI * integral;
    call.max(0.0)
  }
}

/// Black–Scholes–Merton model for Fourier pricing.
pub struct BSMFourier {
  pub sigma: f64,
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for BSMFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let drift = (self.r - self.q - 0.5 * self.sigma.powi(2)) * t;
    (i * xi * drift - 0.5 * self.sigma.powi(2) * t * xi * xi).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    Cumulants {
      c1: (self.r - self.q - 0.5 * self.sigma.powi(2)) * t,
      c2: self.sigma.powi(2) * t,
      c4: 0.0,
    }
  }
}

/// Heston stochastic volatility model for Fourier pricing.
pub struct HestonFourier {
  pub v0: f64,
  pub kappa: f64,
  pub theta: f64,
  pub sigma: f64,
  pub rho: f64,
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for HestonFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let sigma2 = self.sigma * self.sigma;
    let rsi = self.rho * self.sigma * i;

    let d = ((self.kappa - rsi * xi).powi(2) + sigma2 * (i * xi + xi * xi)).sqrt();
    let g = (self.kappa - rsi * xi - d) / (self.kappa - rsi * xi + d);

    let exp_dt = (-d * t).exp();
    let c_val = (self.kappa * self.theta / sigma2)
      * ((self.kappa - rsi * xi - d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
    let d_val = ((self.kappa - rsi * xi - d) / sigma2) * (1.0 - exp_dt) / (1.0 - g * exp_dt);

    (c_val + d_val * self.v0 + i * xi * (self.r - self.q) * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let ekt = (-self.kappa * t).exp();
    let c1 = (self.r - self.q) * t + (1.0 - ekt) * (self.theta - self.v0) / (2.0 * self.kappa)
      - 0.5 * self.theta * t;
    let c2 = self.sigma.powi(2) * t * self.theta / (2.0 * self.kappa);
    Cumulants { c1, c2, c4: 0.0 }
  }
}

/// Variance Gamma model for Fourier pricing.
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

/// CGMY model for Fourier pricing.
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
    let gamma_neg_y = gamma_neg_y(self.y);

    let psi = self.c
      * gamma_neg_y
      * ((Complex64::new(self.m, 0.0) - i * xi).powf(self.y) - self.m.powf(self.y)
        + (Complex64::new(self.g, 0.0) + i * xi).powf(self.y)
        - self.g.powf(self.y));

    let omega = -self.c
      * gamma_neg_y
      * ((self.m - 1.0).powf(self.y) - self.m.powf(self.y) + (self.g + 1.0).powf(self.y)
        - self.g.powf(self.y));

    (i * xi * (self.r - self.q + omega) * t + psi * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let gamma_fn = |z: f64| statrs::function::gamma::gamma(z);
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

    // Omega for martingale correction
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

/// Heston + Kou Double-Exponential jump model (HKDE) for Fourier pricing.
///
/// $$
/// dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t
/// $$
/// $$
/// dv_t = \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v
/// $$
///
/// where $J_t$ follows a Kou double-exponential distribution with upward jump rate $\eta_1$,
/// downward jump rate $\eta_2$, and probability of upward jump $p$.
///
/// Source:
/// - Kirkby, J.L. (PROJ\_Option\_Pricing\_Matlab)
pub struct HKDEFourier {
  /// Initial variance.
  pub v0: f64,
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance.
  pub sigma_v: f64,
  /// Correlation.
  pub rho: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Jump intensity (Poisson rate).
  pub lam: f64,
  /// Probability of upward jump.
  pub p_up: f64,
  /// Upward jump rate parameter (eta1 > 1 required for finite expectation).
  pub eta1: f64,
  /// Downward jump rate parameter (eta2 > 0).
  pub eta2: f64,
}

impl FourierModelExt for HKDEFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let sigma_v2 = self.sigma_v * self.sigma_v;

    // Heston part (same as HestonFourier)
    let rsi = self.rho * self.sigma_v * i;
    let d = ((self.kappa - rsi * xi).powi(2) + sigma_v2 * (i * xi + xi * xi)).sqrt();
    let g = (self.kappa - rsi * xi - d) / (self.kappa - rsi * xi + d);
    let exp_dt = (-d * t).exp();

    let c_heston = (self.kappa * self.theta / sigma_v2)
      * ((self.kappa - rsi * xi - d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
    let d_heston = ((self.kappa - rsi * xi - d) / sigma_v2) * (1.0 - exp_dt) / (1.0 - g * exp_dt);

    // Kou jump compensator
    // k_bar = E[e^J - 1] = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1
    let k_bar = self.p_up * self.eta1 / (self.eta1 - 1.0)
      + (1.0 - self.p_up) * self.eta2 / (self.eta2 + 1.0)
      - 1.0;

    // Jump characteristic function
    let jump_cf = self.lam
      * ((1.0 - self.p_up) * self.eta2 / (Complex64::new(self.eta2, 0.0) + i * xi)
        + self.p_up * self.eta1 / (Complex64::new(self.eta1, 0.0) - i * xi)
        - 1.0);

    // Convexity correction (martingale condition)
    let drift_correction = -self.lam * k_bar;

    (c_heston
      + d_heston * self.v0
      + i * xi * (self.r - self.q + drift_correction) * t
      + jump_cf * t)
      .exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    // Heston cumulants
    let ekt = (-self.kappa * t).exp();
    let c1_h = (self.r - self.q) * t + (1.0 - ekt) * (self.theta - self.v0) / (2.0 * self.kappa)
      - 0.5 * self.theta * t;
    let c2_h = self.sigma_v.powi(2) * t * self.theta / (2.0 * self.kappa);

    // Kou jump cumulants
    let c1_j = self.lam * t * (self.p_up / self.eta1 - (1.0 - self.p_up) / self.eta2);
    let c2_j = 2.0
      * self.lam
      * t
      * (self.p_up / (self.eta1 * self.eta1) + (1.0 - self.p_up) / (self.eta2 * self.eta2));
    let c4_j =
      24.0 * self.lam * t * (self.p_up / self.eta1.powi(4) + (1.0 - self.p_up) / self.eta2.powi(4));

    Cumulants {
      c1: c1_h + c1_j,
      c2: c2_h + c2_j,
      c4: c4_j,
    }
  }
}

/// Bates / Stochastic Volatility with Jumps (SVJ) model for Fourier pricing.
///
/// Heston + Merton log-normal jumps:
///
/// $$
/// dS = (r-q)S\,dt + \sqrt{v}\,S\,dW^S + (e^J - 1)S\,dN_t,
/// \quad dv = \kappa(\theta - v)\,dt + \sigma_v\sqrt{v}\,dW^v
/// $$
///
/// Reference: Bates (1996), "Jumps and Stochastic Volatility"
pub struct BatesFourier {
  pub v0: f64,
  pub kappa: f64,
  pub theta: f64,
  pub sigma_v: f64,
  pub rho: f64,
  pub lambda: f64,
  pub mu_j: f64,
  pub sigma_j: f64,
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for BatesFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let sigma_v2 = self.sigma_v * self.sigma_v;

    // Heston part
    let rsi = self.rho * self.sigma_v * i;
    let d = ((self.kappa - rsi * xi).powi(2) + sigma_v2 * (i * xi + xi * xi)).sqrt();
    let g = (self.kappa - rsi * xi - d) / (self.kappa - rsi * xi + d);
    let exp_dt = (-d * t).exp();

    let c_heston = (self.kappa * self.theta / sigma_v2)
      * ((self.kappa - rsi * xi - d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
    let d_heston =
      ((self.kappa - rsi * xi - d) / sigma_v2) * (1.0 - exp_dt) / (1.0 - g * exp_dt);

    // Merton jump compensator
    let k_bar = (self.mu_j + 0.5 * self.sigma_j * self.sigma_j).exp() - 1.0;
    let jump_cf =
      self.lambda * ((i * self.mu_j * xi - 0.5 * self.sigma_j * self.sigma_j * xi * xi).exp() - 1.0);
    let drift_correction = -self.lambda * k_bar;

    (c_heston
      + d_heston * self.v0
      + i * xi * (self.r - self.q + drift_correction) * t
      + jump_cf * t)
      .exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let ekt = (-self.kappa * t).exp();
    let c1_h = (self.r - self.q) * t + (1.0 - ekt) * (self.theta - self.v0) / (2.0 * self.kappa)
      - 0.5 * self.theta * t;
    let c2_h = self.sigma_v.powi(2) * t * self.theta / (2.0 * self.kappa);
    let c1_j = self.lambda * self.mu_j * t;
    let c2_j = self.lambda * (self.mu_j.powi(2) + self.sigma_j.powi(2)) * t;

    Cumulants {
      c1: c1_h + c1_j,
      c2: c2_h + c2_j,
      c4: 0.0,
    }
  }
}

fn gamma_neg_y(y: f64) -> f64 {
  if y.abs() < 1e-8 {
    return 1e15;
  }
  if y < 0.0 {
    return statrs::function::gamma::gamma(-y);
  }
  if (y - 1.0).abs() < 1e-8 {
    return statrs::function::gamma::gamma(-0.999);
  }
  let g = statrs::function::gamma::gamma(y);
  let sin_val = (std::f64::consts::PI * y).sin();
  if sin_val.abs() < 1e-8 || g.abs() < 1e-8 {
    return 1e15;
  }
  -std::f64::consts::PI / (y * sin_val * g)
}

#[cfg(test)]
mod tests {
  use super::*;

  // Common market: S=100, r=0.05, q=0.01, T=1, is_call=true
  //
  // BSM   sigma=0.15                  → 7.94871378854164
  // VG    sigma=0.2, theta=0.1, nu=0.85 → 10.13935062748614
  // CGMY  C=0.02, G=5, M=15, Y=1.2   → 5.80222163947386
  // MJD   sigma=0.12, lam=0.4, muj=-0.12, sigj=0.18 → 8.675684635426279
  // NIG   alpha=15, beta=-5, delta=0.5 → 9.63000693130414
  // Kou   sigma=0.15, lam=3, p=0.2, eta1=25, eta2=10 → 11.92430307601936
  //
  // BSM (r=0.05, q=0, sigma=0.2, T=1, K=100) → 10.45058357218556 (analytical)
  // Barrier DO call (S=100,K=100,H=90,r=0.05,σ=0.2,T=1) → 8.66547165824566

  const TOL_FOURIER: f64 = 0.15; // Fourier pricer tolerance
  const TOL_TIGHT: f64 = 0.05;

  #[test]
  fn carr_madan_bsm_reference() {
    // BSM sigma=0.15, r=0.05, q=0.01, T=1, K=100 → 7.94871378854164
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 7.94871378854164;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan BSM: got={price}, expected={expected}"
    );
  }

  #[test]
  fn lewis_bsm_reference() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.01,
    };
    let price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let expected = 7.94871378854164;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Lewis BSM: got={price}, expected={expected}"
    );
  }

  #[test]
  fn gil_pelaez_bsm_reference() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.01,
    };
    let price = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let expected = 7.94871378854164;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Gil-Pelaez BSM: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_vg_reference() {
    // VG sigma=0.2, theta=0.1, nu=0.85, r=0.05, q=0.01, T=1, K=100 → 10.13935
    let model = VarianceGammaFourier {
      sigma: 0.2,
      theta: 0.1,
      nu: 0.85,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 10.13935062748614;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan VG: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_cgmy_reference() {
    // CGMY C=0.02, G=5, M=15, Y=1.2, r=0.05, q=0.01, T=1, K=100 → 5.80222
    let model = CGMYFourier {
      c: 0.02,
      g: 5.0,
      m: 15.0,
      y: 1.2,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 5.80222163947386;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan CGMY: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_merton_reference() {
    // MJD sigma=0.12, lam=0.4, mu_j=-0.12, sigma_j=0.18, r=0.05, q=0.01, T=1 → 8.67568
    let model = MertonJDFourier {
      sigma: 0.12,
      lambda: 0.4,
      mu_j: -0.12,
      sigma_j: 0.18,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 8.675684635426279;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan MJD: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_kou_reference() {
    // Kou sigma=0.15, lam=3, p_up=0.2, eta1=25, eta2=10, r=0.05, q=0.01, T=1 → 11.92430
    let model = KouFourier {
      sigma: 0.15,
      lambda: 3.0,
      p_up: 0.2,
      eta1: 25.0,
      eta2: 10.0,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 11.92430307601936;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan Kou: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_heston_vs_analytical() {
    // Heston: v0=0.04, kappa=2, theta=0.04, sigma=0.1, rho=-0.6, r=0.05, q=0
    // K=100 → ~10.474
    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.1,
      rho: -0.6,
      r: 0.05,
      q: 0.0,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    assert!(
      (price - 10.474).abs() < 0.2,
      "Carr-Madan Heston: got={price}, expected≈10.474"
    );
  }

  #[test]
  fn barrier_down_and_out_call_vs_haug() {
    // Haug: S=100, K=100, H=90, r=0.05, q=0, σ=0.2, T=1 → 8.66547165824566
    use crate::quant::OptionType;
    use crate::quant::pricing::barrier::BarrierPricer;
    use crate::quant::pricing::barrier::BarrierType;
    let p = BarrierPricer {
      s: 100.0,
      k: 100.0,
      h: 90.0,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      rebate: 0.0,
      barrier_type: BarrierType::DownAndOut,
      option_type: OptionType::Call,
    };
    let price = p.price();
    let expected = 8.66547165824566;
    assert!(
      (price - expected).abs() < TOL_TIGHT,
      "Barrier DO call: got={price}, expected={expected}"
    );
  }

  #[test]
  fn hkde_zero_jumps_equals_heston() {
    // HKDE with lam=0 should equal Heston
    let heston = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.1,
      rho: -0.6,
      r: 0.05,
      q: 0.0,
    };
    let hkde = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.1,
      rho: -0.6,
      r: 0.05,
      q: 0.0,
      lam: 0.0,
      p_up: 0.5,
      eta1: 10.0,
      eta2: 10.0,
    };
    let pricer = CarrMadanPricer::default();
    let heston_price = pricer.price_call(&heston, 100.0, 100.0, 0.05, 1.0);
    let hkde_price = pricer.price_call(&hkde, 100.0, 100.0, 0.05, 1.0);
    assert!(
      (heston_price - hkde_price).abs() < TOL_TIGHT,
      "HKDE(lam=0) vs Heston: hkde={hkde_price}, heston={heston_price}"
    );
  }

  #[test]
  fn hkde_prices_positive() {
    let model = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.01,
      lam: 3.0,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    assert!(
      price > 0.0,
      "HKDE call price should be positive, got={price}"
    );
  }

  #[test]
  fn hkde_call_decreases_with_strike() {
    let model = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.01,
      lam: 3.0,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    };
    let pricer = CarrMadanPricer::default();
    let p1 = pricer.price_call(&model, 100.0, 90.0, 0.05, 1.0);
    let p2 = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let p3 = pricer.price_call(&model, 100.0, 110.0, 0.05, 1.0);
    assert!(
      p1 > p2 && p2 > p3,
      "HKDE call should decrease with strike: p(90)={p1}, p(100)={p2}, p(110)={p3}"
    );
  }

  #[test]
  fn hkde_lewis_vs_carr_madan() {
    let model = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.01,
      lam: 1.0,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    };
    let cm_price = CarrMadanPricer::default().price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let lw_price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    assert!(
      (cm_price - lw_price).abs() < TOL_FOURIER,
      "HKDE Lewis vs Carr-Madan: lewis={lw_price}, cm={cm_price}"
    );
  }
}
