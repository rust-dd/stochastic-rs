//! Heston and Double-Heston Fourier models.

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;
use super::heston_cumulants;
use crate::pricing::heston::HestonParams;

/// One Heston factor's `(C, D)` at `xi`, `chf = exp(C + D·v0)`: the Little-Trap form with `g̃`
/// multiplied out by its denominator `κ − iρσξ + d`, which is 0 at `ξ = −i` when `ρσ > κ`.
/// Albrecher, Mayer, Schoutens & Tistaert (2007), "The Little Heston Trap", Wilmott, 83–92.
pub(crate) fn factor_cd(p: &HestonParams, t: f64, xi: Complex64) -> (Complex64, Complex64) {
  let HestonParams {
    kappa,
    theta,
    sigma,
    rho,
    ..
  } = *p;
  let i = Complex64::i();
  let sigma2 = sigma * sigma;
  let beta = kappa - rho * sigma * i * xi;
  let moment = i * xi + xi * xi;

  let d = (beta * beta + sigma2 * moment).sqrt();
  let exp_dt = (-d * t).exp();
  let decay = if d == Complex64::ZERO {
    Complex64::from(t)
  } else {
    (1.0 - exp_dt) / d
  };
  let denom = beta * decay + 1.0 + exp_dt;

  let c_val = (kappa * theta / sigma2) * ((beta - d) * t - 2.0 * (0.5 * denom).ln());
  let d_val = -moment * decay / denom;

  (c_val, d_val)
}

/// Heston stochastic volatility model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct HestonFourier {
  params: HestonParams,
}

impl HestonFourier {
  /// Panics unless `params` passes [`HestonParams::new`].
  pub fn new(params: HestonParams) -> Self {
    Self {
      params: params.checked(),
    }
  }

  pub fn params(&self) -> HestonParams {
    self.params
  }
}

impl FourierModelExt for HestonFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let (c_val, d_val) = factor_cd(&self.params, t, xi);
    (c_val + d_val * self.params.v0).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    heston_cumulants(&self.params, t)
  }
}

/// Double Heston: two independent CIR variance factors, each correlated only with its own spot
/// Brownian motion ($\rho_j$), so the chf of $\ln(S_T/F_T)$ is a sum of two Heston exponents.
/// $$
/// \begin{aligned}
/// dS_t &= (r-q)\,S_t\,dt + \sqrt{v_{1,t}}\,S_t\,dW_{1,t}^S + \sqrt{v_{2,t}}\,S_t\,dW_{2,t}^S \\
/// dv_{1,t} &= \kappa_1(\theta_1 - v_{1,t})\,dt + \sigma_1\sqrt{v_{1,t}}\,dW_{1,t}^v \\
/// dv_{2,t} &= \kappa_2(\theta_2 - v_{2,t})\,dt + \sigma_2\sqrt{v_{2,t}}\,dW_{2,t}^v
/// \end{aligned}
/// $$
/// $$
/// \phi_T(u) = \exp\!\left(\sum_{j=1}^{2}\bigl[C_j(u,T) + D_j(u,T)\,v_{j,0}\bigr]\right)
/// $$
/// $$
/// \begin{aligned}
/// d_j &= \sqrt{(\kappa_j - i\rho_j\sigma_j u)^2 + \sigma_j^2(u^2 + iu)} \\
/// g_j &= \frac{\kappa_j - i\rho_j\sigma_j u - d_j}{\kappa_j - i\rho_j\sigma_j u + d_j} \\
/// D_j &= \frac{\kappa_j - i\rho_j\sigma_j u - d_j}{\sigma_j^2}\cdot\frac{1 - e^{-d_j T}}{1 - g_j e^{-d_j T}} \\
/// C_j &= \frac{\kappa_j\theta_j}{\sigma_j^2}\left[(\kappa_j - i\rho_j\sigma_j u - d_j)T - 2\ln\!\left(\frac{1 - g_j e^{-d_j T}}{1 - g_j}\right)\right]
/// \end{aligned}
/// $$
///
/// Reference: Christoffersen, P., Heston, S. & Jacobs, K. (2009), "The Shape and Term Structure of the Index Option Smirk: Why Multifactor Stochastic Volatility Models Work So Well", Management Science 55(12), 1914–1932, DOI 10.1287/mnsc.1090.1065
/// Reference: Mehrdoust, F., Noorani, I. & Hamdi, A. (2021), "Calibration of the double Heston model and an analytical formula in pricing American put option", J. Comput. Appl. Math. 392, 113422, DOI 10.1016/j.cam.2021.113422
#[derive(Debug, Clone)]
pub struct DoubleHestonFourier {
  first: HestonParams,
  second: HestonParams,
}

impl DoubleHestonFourier {
  /// Panics unless both factors pass [`HestonParams::new`].
  pub fn new(first: HestonParams, second: HestonParams) -> Self {
    Self {
      first: first.checked(),
      second: second.checked(),
    }
  }
}

impl FourierModelExt for DoubleHestonFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let (c1, d1) = factor_cd(&self.first, t, xi);
    let (c2, d2) = factor_cd(&self.second, t, xi);

    (c1 + c2 + d1 * self.first.v0 + d2 * self.second.v0).exp()
  }

  /// The factors are independent, so their cumulants add.
  fn cumulants(&self, t: f64) -> Cumulants {
    heston_cumulants(&self.first, t).plus(heston_cumulants(&self.second, t))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn task1_params() -> HestonFourier {
    HestonFourier::new(HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    })
  }

  fn ln_chf(model: &HestonFourier, t: f64, u: f64) -> Complex64 {
    model.chf(t, Complex64::new(u, 0.0)).ln()
  }

  /// `c1 = -i \cdot f'(0)`, `c2 = -f''(0)` for `f(u) = \ln\varphi(u)`, the
  /// standard relation between a characteristic function's log and the
  /// cumulants of the distribution it belongs to (`\varphi(u) = E[e^{iuX}]`
  /// is the moment generating function evaluated at `iu`, so
  /// `d^n/du^n \ln\varphi(u)|_0 = i^n \kappa_n`). Central differences at
  /// `h=1e-3`: small enough that `O(h^2)` truncation error is negligible
  /// against the `1e-6` relative tolerance these tests require, large
  /// enough that the second-difference's `O(\epsilon_{mach}/h^2)`
  /// cancellation noise stays far below it too.
  fn fd_c1_c2(model: &HestonFourier, t: f64) -> (f64, f64) {
    let h = 1e-3;
    let f0 = ln_chf(model, t, 0.0);
    let fp = ln_chf(model, t, h);
    let fm = ln_chf(model, t, -h);
    let i_unit = Complex64::i();
    let c1 = (-i_unit * (fp - fm) / (2.0 * h)).re;
    let c2 = (-(fp - 2.0 * f0 + fm) / (h * h)).re;
    (c1, c2)
  }

  /// `c4 = f''''(0)` by the 5-point stencil at `h = 1e-2`, whose `O(ε/h⁴)` rounding, near `1e-7`,
  /// and `O(h²)` truncation limit it to a `c4` far above `1e-7`.
  fn fd_c4(model: &HestonFourier, t: f64) -> f64 {
    let h = 1e-2;
    let f0 = ln_chf(model, t, 0.0);
    let fp1 = ln_chf(model, t, h);
    let fm1 = ln_chf(model, t, -h);
    let fp2 = ln_chf(model, t, 2.0 * h);
    let fm2 = ln_chf(model, t, -2.0 * h);
    ((fp2 - 4.0 * fp1 + 6.0 * f0 - 4.0 * fm1 + fm2) / h.powi(4)).re
  }

  /// `c2` against a central finite difference of `ln chf` at `τ = 1` and at `τ = 0.1`.
  #[test]
  fn heston_c2_matches_fd_of_log_chf() {
    let model = task1_params();
    for t in [1.0, 0.1] {
      let (_, c2_fd) = fd_c1_c2(&model, t);
      let c2 = model.cumulants(t).c2;
      let rel = (c2 - c2_fd).abs() / c2_fd.abs();
      assert!(
        rel < 1e-6,
        "t={t}: cumulants().c2={c2}, fd={c2_fd}, rel={rel}"
      );
    }
  }

  /// `c1` against the same finite-difference criterion as `c2`.
  #[test]
  fn heston_c1_matches_fd_of_log_chf() {
    let model = task1_params();
    for t in [1.0, 0.1] {
      let (c1_fd, _) = fd_c1_c2(&model, t);
      let c1 = model.cumulants(t).c1;
      let rel = (c1 - c1_fd).abs() / c1_fd.abs().max(1e-12);
      assert!(
        rel < 1e-6,
        "t={t}: cumulants().c1={c1}, fd={c1_fd}, rel={rel}"
      );
    }
  }

  /// `c4` against a finite difference of `ln chf` at `τ = 1` and at `τ = 5`.
  #[test]
  fn heston_c4_matches_fd_of_log_chf() {
    let model = task1_params();
    for t in [1.0, 5.0] {
      let c4 = model.cumulants(t).c4.unwrap();
      let c4_fd = fd_c4(&model, t);
      let rel = ((c4 - c4_fd) / c4_fd).abs();
      assert!(
        rel < 1e-3,
        "t={t}: cumulants().c4={c4}, fd={c4_fd}, rel={rel}"
      );
    }
  }
}
