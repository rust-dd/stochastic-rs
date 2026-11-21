use std::f64::consts::FRAC_1_PI;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use num_complex::Complex64;

use super::SvjParams;
use super::calibrator::SvjCalibrator;
use super::params::BOX;
use super::params::BOXED;
use crate::OptionType;
use crate::calibration::clip_negative;
use crate::calibration::feller_share::natural_jacobian;
use crate::calibration::integrate_gl_to_convergence;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::periodic_slopes;
use crate::calibration::scale_columns;
use crate::calibration::stencil::derivative;
use crate::calibration::stencil::step;

/// Bates/SVJ characteristic function $\phi_T(\xi)$.
///
/// $$
/// \phi_T(\xi) = \exp\!\bigl[C(\xi,T)+D(\xi,T)\,v_0+i\xi\ln S\bigr]
/// $$
///
/// where $C$ and $D$ are the Heston terms augmented by the jump compensator:
/// $$
/// C_{\mathrm{Bates}} = C_{\mathrm{Heston}} +
///   T\,\lambda\bigl(e^{i\mu_J\xi-\frac12\sigma_J^2\xi^2}-1\bigr) -
///   i\xi\,T\,\lambda\bigl(e^{\mu_J+\frac12\sigma_J^2}-1\bigr)
/// $$
///
/// The Heston diffusion part uses the Albrecher-Mayer-Schoutens-Tistaert (2007)
/// "Little Heston Trap" form (`g̃ = 1/g_original`, `exp(-d·τ)`) so the
/// log-argument stays on the principal branch at long τ / high `|ρ|`.
pub(super) fn bates_cf(p: &SvjParams, s: f64, r: f64, q: f64, tau: f64, u: Complex64) -> Complex64 {
  let i = Complex64::i();
  let iu = i * u;
  let rq = r - q;

  let xi_h = Complex64::new(p.kappa, 0.0) - p.sigma * p.rho * iu;
  let d = (xi_h * xi_h + p.sigma * p.sigma * (u * u + iu)).sqrt();

  let exp_neg_dt = (-d * tau).exp();
  let g = (xi_h - d) / (xi_h + d);

  let sigma_v2 = p.sigma * p.sigma;

  let c_heston = iu * (s.ln() + rq * tau)
    + (p.kappa * p.theta / sigma_v2)
      * ((xi_h - d) * tau - 2.0 * ((1.0 - g * exp_neg_dt) / (1.0 - g)).ln());

  let d_heston = ((xi_h - d) / sigma_v2) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt));

  let k_bar = (p.mu_j + 0.5 * p.sigma_j * p.sigma_j).exp() - 1.0;
  let jump_cf = p.lambda * ((i * p.mu_j * u - 0.5 * p.sigma_j * p.sigma_j * u * u).exp() - 1.0);
  let jump_drift = -p.lambda * k_bar * iu;

  let log_phi = c_heston + d_heston * p.v0 + (jump_cf + jump_drift) * tau;
  log_phi.exp()
}

/// Price a European call option under the Bates/SVJ model using a
/// convergence-controlled Gauss-Legendre quadrature.
pub(super) fn bates_call_price(p: &SvjParams, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
  let Some([integral]) = integrate_gl_to_convergence(
    |u_real| {
      let xi = Complex64::new(u_real, 0.0);
      let xi_shift = Complex64::new(u_real, -1.0);
      let phi = bates_cf(p, s, r, q, tau, xi);
      let phi_shift = bates_cf(p, s, r, q, tau, xi_shift);
      let kernel = (Complex64::new(0.0, -u_real * k.ln())).exp()
        / (Complex64::i() * Complex64::new(u_real, 0.0));
      Some([(kernel * (phi_shift - k * phi)).re])
    },
    1e-8,
  ) else {
    return f64::NAN;
  };

  let disc_r = (-r * tau).exp();
  let disc_q = (-q * tau).exp();
  let call = 0.5 * (s * disc_q - k * disc_r) + disc_r * FRAC_1_PI * integral;
  clip_negative(call)
}

pub(super) struct SvjProblem<'a> {
  calibrator: &'a SvjCalibrator,
  /// The folded point: `sigma` holds σ's share of the way to Feller's cap.
  params: SvjParams,
  /// `dθⱼ/dxⱼ` of the box's reflection at the optimiser's raw coordinates.
  reflections: Array1<f64>,
  pub(super) market: Vec<f64>,
  pub(super) trail: Trail<SvjParams>,
}

impl<'a> SvjProblem<'a> {
  pub(super) fn new(calibrator: &'a SvjCalibrator, start: SvjParams) -> Self {
    Self {
      calibrator,
      params: start.to_share(),
      reflections: Array1::ones(8),
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::new(),
    }
  }

  pub(super) fn effective_params(&self) -> SvjParams {
    self.params.at_share()
  }

  pub(super) fn compute_model_prices_for(&self, p: &SvjParams) -> Vec<f64> {
    let SvjCalibrator { s, r, q, .. } = *self.calibrator;
    self
      .calibrator
      .quotes
      .iter()
      .map(|quote| {
        let (k, tau) = (quote.strike, quote.tau);
        let call = bates_call_price(p, s, k, r, q, tau);
        match quote.option_type {
          OptionType::Call => call,
          OptionType::Put => clip_negative(call - s * (-q * tau).exp() + k * (-r * tau).exp()),
        }
      })
      .collect()
  }

  fn residuals_for(&self, p: &SvjParams) -> Array1<f64> {
    self.market_minus(&self.compute_model_prices_for(p))
  }

  fn market_minus(&self, model: &[f64]) -> Array1<f64> {
    Array1::from_iter(
      self
        .market
        .iter()
        .zip(model)
        .map(|(market, model)| market - model),
    )
  }

  /// `∂r/∂x` at the folded point, each stencil point through σ's share; `scale` times the shared
  /// step.
  pub(super) fn difference_jacobian(&self, scale: f64) -> Array2<f64> {
    let point = Array1::from(self.params);
    let mut jacobian = Array2::zeros((self.market.len(), point.len()));
    for (j, mut column) in jacobian.columns_mut().into_iter().enumerate() {
      let x = point[j];
      column.assign(&derivative(x, scale * step(x), BOXED[j], |value| {
        let mut moved = point.clone();
        moved[j] = value;
        self.residuals_for(&SvjParams::from(moved).at_share())
      }));
    }
    jacobian
  }
}

impl LeastSquaresProblem for SvjProblem<'_> {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    self.reflections = periodic_slopes(params, &BOX);
    self.params = SvjParams::from(params.to_owned()).folded();
  }

  fn params(&self) -> Array1<f64> {
    self.params.into()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let params_eff = self.effective_params();
    let model = self.compute_model_prices_for(&params_eff);
    let residuals = self.market_minus(&model);
    let options = &self.calibrator.options;
    self
      .trail
      .stage(options, &self.market, &model, &params_eff, residuals.view());
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let jacobian = self.difference_jacobian(1.0);
    Some(scale_columns(jacobian, self.reflections.view()))
  }

  fn accept(&mut self) {
    self.trail.accept();
  }

  fn natural_params(&self) -> Vec<f64> {
    Array1::from(self.effective_params()).to_vec()
  }

  fn natural_jacobian(&self) -> Array2<f64> {
    natural_jacobian(self.reflections.view(), &[(1, self.params.share_slopes())])
  }
}
