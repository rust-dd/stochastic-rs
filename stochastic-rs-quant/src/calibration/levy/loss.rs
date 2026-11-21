use super::types::LevyModel;
use super::types::LevyModelType;
use crate::OptionType;
use crate::calibration::clip_negative;
use crate::pricing::cf_quadrature::Rule;
use crate::pricing::cf_quadrature::Walk;
use crate::pricing::fourier::LewisPricer;

/// The call of the fit's own model by Lewis, as its `ModelPricer` prices it, run as `walk` says.
pub(super) fn fourier_call_price(
  model_type: LevyModelType,
  params: &[f64],
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
  walk: &mut Walk<'_, Rule>,
) -> f64 {
  match LevyModel::of(model_type, params) {
    LevyModel::VarianceGamma(m) => LewisPricer::price_call_walk(&m, s, k, r, q, t, walk),
    LevyModel::Nig(m) => LewisPricer::price_call_walk(&m, s, k, r, q, t, walk),
    LevyModel::Cgmy(m) => LewisPricer::price_call_walk(&m, s, k, r, q, t, walk),
    LevyModel::MertonJd(m) => LewisPricer::price_call_walk(&m, s, k, r, q, t, walk),
    LevyModel::Kou(m) => LewisPricer::price_call_walk(&m, s, k, r, q, t, walk),
  }
}

/// Price a European option (call or put) via put-call parity from the Fourier call price.
pub(super) fn fourier_option_price(
  model_type: LevyModelType,
  params: &[f64],
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
  option_type: OptionType,
  walk: &mut Walk<'_, Rule>,
) -> f64 {
  let call = fourier_call_price(model_type, params, s, k, r, q, t, walk);
  match option_type {
    OptionType::Call => call,
    OptionType::Put => clip_negative(call - s * (-q * t).exp() + k * (-r * t).exp()),
  }
}

pub(super) fn param_count(model_type: LevyModelType) -> usize {
  match model_type {
    LevyModelType::VarianceGamma => 3,
    LevyModelType::Nig => 3,
    LevyModelType::Cgmy => 4,
    LevyModelType::MertonJd => 4,
    LevyModelType::Kou => 5,
  }
}

pub(super) fn param_bounds(model_type: LevyModelType) -> Vec<(f64, f64)> {
  match model_type {
    LevyModelType::VarianceGamma => {
      vec![(0.01, 2.0), (-1.0, 1.0), (0.01, 5.0)]
    }
    LevyModelType::Nig => {
      vec![(0.01, 50.0), (-50.0, 50.0), (0.001, 5.0)]
    }
    LevyModelType::Cgmy => {
      vec![(0.001, 10.0), (0.01, 50.0), (0.01, 50.0), (-1.0, 1.999)]
    }
    LevyModelType::MertonJd => {
      vec![(0.01, 2.0), (0.01, 20.0), (-1.0, 1.0), (0.01, 2.0)]
    }
    LevyModelType::Kou => {
      vec![
        (0.01, 2.0),
        (0.01, 20.0),
        (0.01, 0.99),
        (1.0, 100.0),
        (1.0, 100.0),
      ]
    }
  }
}

pub(super) fn default_params(model_type: LevyModelType) -> Vec<f64> {
  match model_type {
    LevyModelType::VarianceGamma => vec![0.2, -0.1, 0.5],
    LevyModelType::Nig => vec![15.0, -5.0, 0.5],
    LevyModelType::Cgmy => vec![1.0, 10.0, 15.0, 0.5],
    LevyModelType::MertonJd => vec![0.15, 1.0, -0.05, 0.1],
    LevyModelType::Kou => vec![0.15, 3.0, 0.5, 10.0, 10.0],
  }
}

/// Project parameters into their admissible bounds.
pub(super) fn project_params(model_type: LevyModelType, params: &mut [f64]) {
  clamp_params(model_type, params);
  // The NIG exponent is defined only when alpha > |beta|.
  if model_type == LevyModelType::Nig {
    let alpha = params[0];
    let beta = params[1];
    if alpha <= beta.abs() {
      params[0] = beta.abs() + 0.01;
    }
  }
}

/// Each parameter clamped into its bounds, NIG's `α > |β|` aside.
pub(super) fn clamp_params(model_type: LevyModelType, params: &mut [f64]) {
  for (p, (lo, hi)) in params.iter_mut().zip(param_bounds(model_type)) {
    *p = p.clamp(lo, hi);
  }
}
