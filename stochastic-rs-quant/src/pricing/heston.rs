//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::f64::consts::FRAC_1_PI;

use num_complex::Complex64;
use stochastic_rs_distributions::special::norm_pdf;

use super::cf_quadrature::CONTROL_VARIANCE_CAP;
use super::cf_quadrature::Inversion;
use super::cf_quadrature::Walk;
use super::cf_quadrature::gaussian_share;
use super::cf_quadrature::head_floor;
use super::cf_quadrature::reach_and_scale;
use super::fourier::mean_integrated_variance;
use super::fourier::riccati_cd;
use crate::CallPut;
use crate::OptionType;
use crate::pricing::digital::vanilla_payoff;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

mod greeks;
mod params;

pub use params::HestonParams;

#[cfg(test)]
thread_local! {
  /// This thread's `f_and_d` calls, the cost the evaluation-count tests hold.
  static EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Heston (1993) semi-closed-form pricer; the query `(s, k, r, q, τ)` travels as arguments, so one
/// instance prices a whole strike/maturity grid.
/// Heston, S. L. (1993), Rev. Financ. Stud. 6(2), 327–343, DOI 10.1093/rfs/6.2.327.
///
/// ```
/// use stochastic_rs_quant::pricing::heston::HestonParams;
/// use stochastic_rs_quant::pricing::heston::HestonPricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let params = HestonParams { v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7 };
/// let model = HestonPricer::new(params);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.02, 0.75);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.02, 0.75);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HestonPricer {
  params: HestonParams,
  /// Heston's market price of volatility risk `λ`.
  lambda: f64,
}

impl HestonPricer {
  /// Panics unless `params` passes [`HestonParams::new`].
  pub fn new(params: HestonParams) -> Self {
    Self {
      params: params.checked(),
      lambda: 0.0,
    }
  }

  /// Sets Heston's `λ` (default `0`); panics unless `lambda` is finite.
  pub fn with_vol_risk_premium(mut self, lambda: f64) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    self.lambda = lambda;
    self
  }

  pub fn params(&self) -> HestonParams {
    self.params
  }

  pub fn vol_risk_premium(&self) -> f64 {
    self.lambda
  }

  /// Call and put from one pair `P_1`, `P_2`; the payoff at `tau = 0`. NaN for a negative or NaN
  /// `tau` and from an `E_j[∫₀^τ v]` near `1e149` on (`b_jτ = −353` at `λ = −355`), finite or not.
  ///
  /// $$
  /// C=Se^{-q\tau}P_1-Ke^{-r\tau}P_2,\qquad P=C+Ke^{-r\tau}-Se^{-q\tau}
  /// $$
  pub fn price_call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> CallPut {
    if tau.is_nan() || tau < 0.0 {
      return CallPut {
        call: f64::NAN,
        put: f64::NAN,
      };
    }
    if tau == 0.0 {
      let payoff =
        |option_type| vanilla_payoff(option_type, s * (-q * tau).exp(), k * (-r * tau).exp());
      return CallPut {
        call: payoff(OptionType::Call),
        put: payoff(OptionType::Put),
      };
    }
    let reach = self.reach(tau, (s, k, r, q));
    let call = s * (-q * tau).exp() * self.p(1, tau, (s, k, r, q), reach)
      - k * (-r * tau).exp() * self.p(2, tau, (s, k, r, q), reach);
    let put = call + k * (-r * tau).exp() - s * (-q * tau).exp();

    CallPut { call, put }
  }

  /// Black volatility implied by `c_price` under the carry `b = r - q`, whatever the model's
  /// parameters; NaN for a price outside the no-arbitrage bounds.
  pub fn implied_volatility(
    &self,
    c_price: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    let forward = s * ((r - q) * tau).exp();
    let undiscounted_price = c_price * (r * tau).exp();
    crate::vol_surface::implied::black_implied_vol(undiscounted_price, forward, k, tau, option_type)
  }
}

impl ModelPricer for HestonPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).call
  }

  /// The pair's put, not the trait's parity default, which sums the same three terms in another
  /// order and can land an ulp away.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).put
  }
}

/// European vanilla call on the forward $Se^{(r-q)\tau}$ — this model's
/// cost of carry is $b=r-q$, so the default forward applies unchanged.
impl VanillaEuropeanCall for HestonPricer {}

impl HestonPricer {
  /// `∂C/∂v₀ = ∂P/∂v₀`, the price's own split differentiated in `v₀`, no finite difference. NaN for
  /// `tau ≤ 0` or a NaN `tau`, and from an `E_j[∫₀^τ v]` near `1e147` on (`b_jτ = −348`).
  pub fn initial_variance_vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau.is_nan() || tau <= 0.0 {
      return f64::NAN;
    }
    let reach = self.reach(tau, (s, k, r, q));
    s * (-q * tau).exp() * self.p_initial_variance_derivative(1, tau, (s, k, r, q), reach)
      - k * (-r * tau).exp() * self.p_initial_variance_derivative(2, tau, (s, k, r, q), reach)
  }

  pub(self) fn u(&self, j: u8) -> f64 {
    match j {
      1 => 0.5,
      2 => -0.5,
      _ => unreachable!("Heston P_j index must be 1 or 2"),
    }
  }

  pub(self) fn b(&self, j: u8) -> f64 {
    match j {
      1 => self.params.kappa + self.lambda - self.params.rho * self.params.sigma,
      2 => self.params.kappa + self.lambda,
      _ => unreachable!("Heston P_j index must be 1 or 2"),
    }
  }

  /// `(f_j, D_j)` at `φ`, `f_j` of `ln(S_T/F)`: the chf's own exponent at `β = b_j − iρσφ`,
  /// `m = φ² − 2iu_jφ`, which for `j = 1` is the chf at `φ − i`, the share measure's.
  fn f_and_d(&self, j: u8, phi: f64, tau: f64) -> (Complex64, Complex64) {
    let HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    } = self.params;
    #[cfg(test)]
    EVALUATIONS.with(|n| n.set(n.get() + 1));
    let i = Complex64::i();
    let beta = self.b(j) - rho * sigma * i * phi;
    let moment = Complex64::new(phi * phi, -2.0 * self.u(j) * phi);
    let (big_c, big_d) = riccati_cd(beta, moment, kappa * theta, sigma * sigma, tau);
    ((big_c + big_d * v0).exp(), big_d)
  }

  /// `E_j[∫₀^τ v]` under `P_j`'s mean reversion `b_j`, its Gaussian control variate's variance: the
  /// model's own `−2c₁` at `b_j`.
  fn mean_integrated_variance(&self, j: u8, tau: f64) -> f64 {
    let HestonParams {
      v0, kappa, theta, ..
    } = self.params;
    mean_integrated_variance(v0, self.b(j), kappa * theta, tau)
  }

  /// The walk's `(reach, scale)`, from the chf's decay and `κ = ln(K/F)`; one for both `P_j`.
  fn reach(&self, tau: f64, (s, k, r, q): (f64, f64, f64, f64)) -> (f64, f64) {
    let kappa = (k / s).ln() - (r - q) * tau;
    reach_and_scale(|phi| self.f_and_d(2, phi, tau).0.norm(), kappa)
  }

  /// `(w, ∂w/∂v₀)`: `E_j[∫v]`, the Gaussian control's variance, held at the cap past it.
  fn control(&self, j: u8, tau: f64) -> (f64, f64) {
    let expected = self.mean_integrated_variance(j, tau);
    if expected > CONTROL_VARIANCE_CAP {
      (CONTROL_VARIANCE_CAP, 0.0)
    } else {
      (expected, mean_integrated_variance(1.0, self.b(j), 0.0, tau))
    }
  }

  /// Heston (1993)'s `P_j` less a Gaussian of variance `w = E_j[∫v]`, whose share is `N(d_j)`.
  pub(self) fn p(
    &self,
    j: u8,
    tau: f64,
    (s, k, r, q): (f64, f64, f64, f64),
    (reach, scale): (f64, f64),
  ) -> f64 {
    if !self.mean_integrated_variance(j, tau).is_finite() {
      return f64::NAN;
    }
    let (w, _) = self.control(j, tau);
    let mean = self.u(j) * w;
    let kappa = (k / s).ln() - (r - q) * tau;
    // The phase is `φκ`, `κ = ln(K/F)`: `φ ln F` and `φ ln K` apart round to 1e-8 rad at `φ = 1e7`.
    let kernel = |phi: f64| Complex64::new(0.0, -phi * kappa).exp() / Complex64::new(0.0, phi);
    let inversion = Inversion {
      chf_part: |phi: f64| kernel(phi) * self.f_and_d(j, phi, tau).0,
      control_part: |phi: f64| kernel(phi) * gaussian(phi, mean, w),
      control_frequency: mean - kappa,
      head_floor: Some(self.head_floor(j, tau)),
      reach,
      scale,
    };
    gaussian_share(kappa, mean, w) + FRAC_1_PI * inversion.integrate(&mut Walk::Adaptive)
  }

  /// `∂P_j/∂v₀`: the share's `n(d_j)∂d_j/∂v₀`, and under the integral `D_j f_j` less the control's.
  fn p_initial_variance_derivative(
    &self,
    j: u8,
    tau: f64,
    (s, k, r, q): (f64, f64, f64, f64),
    (reach, scale): (f64, f64),
  ) -> f64 {
    if !self.mean_integrated_variance(j, tau).is_finite() {
      return f64::NAN;
    }
    let (w, slope) = self.control(j, tau);
    let u = self.u(j);
    let kappa = (k / s).ln() - (r - q) * tau;
    let kernel = |phi: f64| Complex64::new(0.0, -phi * kappa).exp() / Complex64::new(0.0, phi);
    let inversion = Inversion {
      chf_part: |phi: f64| {
        let (f, big_d) = self.f_and_d(j, phi, tau);
        kernel(phi) * big_d * f
      },
      control_part: |phi: f64| {
        let exponent_slope = Complex64::new(-0.5 * phi * phi, u * phi) * slope;
        kernel(phi) * gaussian(phi, u * w, w) * exponent_slope
      },
      control_frequency: u * w - kappa,
      head_floor: Some(self.head_floor(j, tau)),
      reach,
      scale,
    };
    let share = if w > 0.0 {
      norm_pdf((u * w - kappa) / w.sqrt()) * (u * w + kappa) / (2.0 * w * w.sqrt()) * slope
    } else {
      0.0
    };
    share + FRAC_1_PI * inversion.integrate(&mut Walk::Adaptive)
  }

  /// Where `f_j` is within `1e-6` of its `φ → 0` limit 1.
  fn head_floor(&self, j: u8, tau: f64) -> f64 {
    head_floor(|phi| (self.f_and_d(j, phi, tau).0 - 1.0).norm())
  }
}

/// The control's characteristic function of `ln(S_T/F) ~ N(mean, w)` at `φ`.
fn gaussian(phi: f64, mean: f64, w: f64) -> Complex64 {
  Complex64::new(-0.5 * phi * phi * w, phi * mean).exp()
}

#[cfg(test)]
mod tests;
