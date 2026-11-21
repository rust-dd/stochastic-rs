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

/// How far inside its model's `E[e^X] < ∞` domain a fit's point stays.
const DOMAIN_MARGIN: f64 = 0.01;

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
      // `E[e^X]` is finite for `M > 1`, the compensator's `(M − 1)^Y`.
      // Reference: Carr, Geman, Madan & Yor (2002), DOI 10.1086/338705.
      let m = (1.0 + DOMAIN_MARGIN, 50.0);
      vec![(0.001, 10.0), (0.01, 50.0), m, (-1.0, 1.999)]
    }
    LevyModelType::MertonJd => {
      vec![(0.01, 2.0), (0.01, 20.0), (-1.0, 1.0), (0.01, 2.0)]
    }
    LevyModelType::Kou => {
      vec![
        (0.01, 2.0),
        (0.01, 20.0),
        (0.01, 0.99),
        // `E[e^J]` is finite for `η₁ > 1`, the compensator's `pη₁/(η₁ − 1)`.
        // Reference: Kou (2002), eq. (4), DOI 10.1287/mnsc.48.8.1086.166.
        (1.0 + DOMAIN_MARGIN, 100.0),
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

/// Each parameter clamped into its bounds, then the point moved `DOMAIN_MARGIN` inside its
/// model's `E[e^X] < ∞`, where its compensator is real: Variance Gamma's `ν`, NIG's `α`.
pub(super) fn project_params(model_type: LevyModelType, params: &mut [f64]) {
  clamp_params(model_type, params);
  match model_type {
    // Reference: Madan, Carr & Chang (1998), eq. (22), DOI 10.1023/A:1009703431535.
    LevyModelType::VarianceGamma => {
      let rate = params[1] + 0.5 * params[0] * params[0];
      if rate * params[2] > 1.0 - DOMAIN_MARGIN {
        params[2] = (1.0 - DOMAIN_MARGIN) / rate;
      }
    }
    // Reference: Barndorff-Nielsen (1997), Scand. J. Statist. 24(1), DOI 10.1111/1467-9469.00045.
    LevyModelType::Nig => {
      // `β` leaves `α` room under its own ceiling.
      let top = param_bounds(model_type)[0].1 - DOMAIN_MARGIN;
      let beta = params[1].clamp(-top, top - 1.0);
      params[1] = beta;
      params[0] = params[0].max(beta.abs().max((beta + 1.0).abs()) + DOMAIN_MARGIN);
    }
    _ => {}
  }
}

/// The condition of `E[e^X] < ∞` that `p` breaks, the ones [`project_params`] keeps a fit inside,
/// in a constructor panic's words.
pub(super) fn domain_violation(model_type: LevyModelType, p: &[f64]) -> Option<String> {
  let above_one = |i: usize, name: &str| {
    let got = format!("got start.values[{i}] = {:?}", p[i]);
    (p[i] <= 1.0).then(|| format!("start.values[{i}] must satisfy `{name} > 1.0`, {got}"))
  };
  match model_type {
    LevyModelType::VarianceGamma => {
      let margin = 1.0 - p[1] * p[2] - 0.5 * p[0] * p[0] * p[2];
      let rule = "`1.0 - theta * nu - 0.5 * sigma * sigma * nu > 0.0`";
      let got = format!("got {margin:?} at (sigma, theta, nu) = {p:?}");
      (margin <= 0.0).then(|| format!("start.values must satisfy {rule}, {got}"))
    }
    LevyModelType::Nig => {
      let (alpha, beta) = (p[0], p[1]);
      let rule = "`alpha > beta.abs() && alpha >= (beta + 1.0).abs()`";
      let got = format!("got (alpha, beta) = ({alpha:?}, {beta:?})");
      let inside = alpha > beta.abs() && alpha >= (beta + 1.0).abs();
      (!inside).then(|| format!("start.values must satisfy {rule}, {got}"))
    }
    LevyModelType::Cgmy => above_one(2, "m"),
    LevyModelType::Kou => above_one(3, "eta1"),
    LevyModelType::MertonJd => None,
  }
}

/// Each parameter clamped into its bounds, the domain's coupled conditions aside.
pub(super) fn clamp_params(model_type: LevyModelType, params: &mut [f64]) {
  for (p, (lo, hi)) in params.iter_mut().zip(param_bounds(model_type)) {
    *p = p.clamp(lo, hi);
  }
}
