//! Heston and Double-Heston Fourier models.

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;

/// Heston stochastic volatility model for Fourier pricing.
#[derive(Debug, Clone)]
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
  /// Albrecher-Mayer-Schoutens-Tistaert (2007) "Little Heston Trap" form:
  /// `g̃ = 1/g_original` with `exp(-d·t)` keeps the principal-branch logarithm
  /// stable for large τ and `|ρ| → 1`. Reverting to the original Heston (1993)
  /// numerator `(κ - ρσiξ + d)` / `exp(+d·t)` triggers branch-cut jumps —
  /// see the `heston_fourier_little_trap_long_maturity_high_rho` regression.
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

/// Double Heston stochastic volatility model for Fourier pricing.
///
/// Two independent Cox-Ingersoll-Ross variance factors driving the spot:
/// $$
/// \begin{aligned}
/// dS_t &= (r-q)\,S_t\,dt + \sqrt{v_{1,t}}\,S_t\,dW_{1,t}^S + \sqrt{v_{2,t}}\,S_t\,dW_{2,t}^S \\
/// dv_{1,t} &= \kappa_1(\theta_1 - v_{1,t})\,dt + \sigma_1\sqrt{v_{1,t}}\,dW_{1,t}^v \\
/// dv_{2,t} &= \kappa_2(\theta_2 - v_{2,t})\,dt + \sigma_2\sqrt{v_{2,t}}\,dW_{2,t}^v
/// \end{aligned}
/// $$
/// with $d\langle W_1^S, W_1^v\rangle_t = \rho_1\,dt$ and
/// $d\langle W_2^S, W_2^v\rangle_t = \rho_2\,dt$. All other Brownian motion
/// pairs are independent. Because the factors are independent, the
/// characteristic function of $\ln(S_T/S_0)$ factorises into a sum of two
/// Heston-type contributions plus a single risk-neutral drift:
/// $$
/// \phi_T(u) = \exp\!\left(iu(r-q)T + \sum_{j=1}^{2}\bigl[C_j(u,T) + D_j(u,T)\,v_{j,0}\bigr]\right)
/// $$
/// with, for $j=1,2$,
/// $$
/// \begin{aligned}
/// d_j &= \sqrt{(\kappa_j - i\rho_j\sigma_j u)^2 + \sigma_j^2(u^2 + iu)} \\
/// g_j &= \frac{\kappa_j - i\rho_j\sigma_j u - d_j}{\kappa_j - i\rho_j\sigma_j u + d_j} \\
/// D_j &= \frac{\kappa_j - i\rho_j\sigma_j u - d_j}{\sigma_j^2}\cdot\frac{1 - e^{-d_j T}}{1 - g_j e^{-d_j T}} \\
/// C_j &= \frac{\kappa_j\theta_j}{\sigma_j^2}\left[(\kappa_j - i\rho_j\sigma_j u - d_j)T - 2\ln\!\left(\frac{1 - g_j e^{-d_j T}}{1 - g_j}\right)\right]
/// \end{aligned}
/// $$
///
/// Source:
/// - Christoffersen, Heston & Jacobs (2009), "The Shape and Term Structure of
///   the Index Option Smirk: Why Multifactor Stochastic Volatility Models Work
///   So Well", <https://doi.org/10.1287/mnsc.1090.1065>
/// - Mehrdoust, Noorani & Hamdi (2021), "Calibration of the double Heston
///   model and an analytical formula in pricing American put option",
///   J. Comput. Appl. Math. 392, 113422,
///   <https://doi.org/10.1016/j.cam.2021.113422>
#[derive(Debug, Clone)]
pub struct DoubleHestonFourier {
  /// Initial variance of factor 1.
  pub v1_0: f64,
  /// Mean-reversion speed of factor 1.
  pub kappa1: f64,
  /// Long-run variance of factor 1.
  pub theta1: f64,
  /// Volatility-of-variance of factor 1.
  pub sigma1: f64,
  /// Spot-variance correlation for factor 1.
  pub rho1: f64,
  /// Initial variance of factor 2.
  pub v2_0: f64,
  /// Mean-reversion speed of factor 2.
  pub kappa2: f64,
  /// Long-run variance of factor 2.
  pub theta2: f64,
  /// Volatility-of-variance of factor 2.
  pub sigma2: f64,
  /// Spot-variance correlation for factor 2.
  pub rho2: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
}

impl DoubleHestonFourier {
  /// Compute a single Heston factor contribution $(C_j, D_j)$ evaluated at `xi`.
  ///
  /// Uses the Albrecher-Mayer-Schoutens-Tistaert (2007) "Little Heston Trap"
  /// form (`g̃ = 1/g_original`, `exp(-d·t)`) so each factor stays on the
  /// principal log-branch for large τ and `|ρ_j| → 1`.
  #[inline]
  fn factor_cd(
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    t: f64,
    xi: Complex64,
  ) -> (Complex64, Complex64) {
    let i = Complex64::i();
    let sigma2 = sigma * sigma;
    let rsi = rho * sigma * i;

    let d = ((kappa - rsi * xi).powi(2) + sigma2 * (i * xi + xi * xi)).sqrt();
    let g = (kappa - rsi * xi - d) / (kappa - rsi * xi + d);
    let exp_dt = (-d * t).exp();

    let c_val = (kappa * theta / sigma2)
      * ((kappa - rsi * xi - d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
    let d_val = ((kappa - rsi * xi - d) / sigma2) * (1.0 - exp_dt) / (1.0 - g * exp_dt);

    (c_val, d_val)
  }
}

impl FourierModelExt for DoubleHestonFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();

    let (c1, d1) = Self::factor_cd(self.kappa1, self.theta1, self.sigma1, self.rho1, t, xi);
    let (c2, d2) = Self::factor_cd(self.kappa2, self.theta2, self.sigma2, self.rho2, t, xi);

    (c1 + c2 + d1 * self.v1_0 + d2 * self.v2_0 + i * xi * (self.r - self.q) * t).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let ekt1 = (-self.kappa1 * t).exp();
    let ekt2 = (-self.kappa2 * t).exp();

    let int_v1 = self.theta1 * t + (self.v1_0 - self.theta1) * (1.0 - ekt1) / self.kappa1;
    let int_v2 = self.theta2 * t + (self.v2_0 - self.theta2) * (1.0 - ekt2) / self.kappa2;

    let c1 = (self.r - self.q) * t - 0.5 * (int_v1 + int_v2);
    let c2 = int_v1 + int_v2;
    Cumulants { c1, c2, c4: 0.0 }
  }
}
