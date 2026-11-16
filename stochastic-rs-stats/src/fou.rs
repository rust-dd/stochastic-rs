//! fOU estimates `(H, σ, θ)` by method: `H` from the ratio of a filtered path's variations at two
//! scales, `σ` from their level, `θ` from the ergodic second moment.
//!
//! $$
//! \hat H=\tfrac12\log_2\frac{V_{\mathbf a^2}}{V_{\mathbf a}},\qquad
//! \hat\sigma^2=\frac{-2\,V_{\mathbf a}}{\Delta^{2\hat H}\sum_{k,\ell}a_ka_\ell|k-\ell|^{2\hat H}},\qquad
//! \hat\theta=\Bigl(\frac{2\hat\mu_2}{\hat\sigma^2\,\Gamma(2\hat H+1)}\Bigr)^{-1/(2\hat H)}
//! $$
//!
//! Reference: Brouste, A. & Iacus, S. M. (2013), "Parameter estimation for the discretely observed fractional Ornstein–Uhlenbeck process and the Yuima R package", Computational Statistics 28(4), 1529–1547, DOI 10.1007/s00180-012-0365-6, arXiv:1112.3777.
//!
//! Reference: Hu, Y., Nualart, D. & Zhou, H. (2019), "Parameter estimation for fractional Ornstein–Uhlenbeck processes of general Hurst parameter", Statistical Inference for Stochastic Processes 22(1), 111–142, DOI 10.1007/s11203-017-9168-2, arXiv:1703.09372.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::s;
use stochastic_rs_distributions::special::gamma;

use crate::StatsError;
use crate::hurst::variations::a2_coefficients;
use crate::hurst::variations::daubechies_coeffs;
use crate::hurst::variations::lfilter_f64;
use crate::hurst::variations::power_variation;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;

/// Structured result of an fOU parameter estimator.
///
/// Build with [`From<(f64, f64, f64, f64)>`] — the estimator output tuple
/// is converted positionally as `(hurst, sigma, mu, theta)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FouFit {
  pub hurst: f64,
  pub sigma: f64,
  pub mu: f64,
  pub theta: f64,
}

impl From<(f64, f64, f64, f64)> for FouFit {
  fn from((hurst, sigma, mu, theta): (f64, f64, f64, f64)) -> Self {
    Self {
      hurst,
      sigma,
      mu,
      theta,
    }
  }
}

impl From<FouFit> for (f64, f64, f64, f64) {
  fn from(r: FouFit) -> Self {
    (r.hurst, r.sigma, r.mu, r.theta)
  }
}

/// Linear filter selection for [`fit_fou_filtered_variations`].
#[derive(PartialEq, Clone, Copy, Debug)]
#[non_exhaustive]
pub enum FilterType {
  Daubechies,
}

/// Brouste–Iacus fit with filter `a`; `Err` on fewer than its length + 2 points, on non-finite or
/// flat data or no usable `H ∈ (0, 1)`, `σ`, `θ`; panics on a bad `delta` or override.
pub fn fit_fou_filtered_variations(
  path: ArrayView1<f64>,
  filter_type: FilterType,
  delta: Option<f64>,
  hurst_override: Option<f64>,
) -> Result<FouFit, StatsError> {
  assert_step_and_hurst(delta, hurst_override);
  let a = match filter_type {
    FilterType::Daubechies => daubechies_coeffs(),
  };
  let l_filter = a.len();
  checked(path, l_filter + 2)?;
  let n = path.len();

  let v1_path = lfilter_f64::<f64>(&a, &Array1::from_vec(vec![1.0]), path);
  let v1_sum: f64 = v1_path.mapv(|x| x * x).sum();
  if v1_sum == 0.0 {
    return Err(StatsError::DegenerateData("zero filtered variation"));
  }
  let hurst = match hurst_override {
    Some(hurst) => hurst,
    None => {
      let a_2 = a2_coefficients(&a);
      let v2_path = lfilter_f64::<f64>(&a_2, &Array1::from_vec(vec![1.0]), path);
      let v2_sum: f64 = v2_path.mapv(|x| x * x).sum();
      hurst_from_ratio(0.5 * (v2_sum / v1_sum).log2())?
    }
  };
  let dt = delta.unwrap_or(1.0 / n as f64);

  let mut const_filter = 0.0;
  for i in 0..l_filter {
    for j in 0..l_filter {
      const_filter += a[i] * a[j] * ((i as f64 - j as f64).abs()).powf(2.0 * hurst);
    }
  }
  let numerator = -2.0 * v1_sum / ((n - l_filter) as f64);
  let denominator = const_filter * dt.powf(2.0 * hurst);
  let sigma = positive((numerator / denominator).sqrt(), "sigma from the variation")?;

  let mu = path.mean().unwrap();

  let mean_square = path.mapv(|x| x * x).mean().unwrap();
  let theta_num = 2.0 * mean_square;
  let theta_den = sigma.powi(2) * gamma(2.0 * hurst + 1.0);
  let theta = positive(
    (theta_num / theta_den).powf(-1.0 / (2.0 * hurst)),
    "theta from the second moment",
  )?;

  Ok(FouFit {
    hurst,
    sigma,
    mu,
    theta,
  })
}

/// Brouste–Iacus fit with the second difference; `Err` on fewer than five points, on non-finite or
/// flat data or no usable `H ∈ (0, 1)`, `σ`, `θ`; panics on a bad `delta` or override.
pub fn fit_fou_central_differences(
  path: ArrayView1<f64>,
  delta: Option<f64>,
  hurst_override: Option<f64>,
) -> Result<FouFit, StatsError> {
  assert_step_and_hurst(delta, hurst_override);
  checked(path, 5)?;
  let n = path.len();
  let n_f = n as f64;

  let sigma_num: f64 = (0..(n - 2))
    .map(|i| {
      let diff = path[i + 2] - 2.0 * path[i + 1] + path[i];
      diff * diff
    })
    .sum();
  if sigma_num == 0.0 {
    return Err(StatsError::DegenerateData(
      "zero second-difference variation",
    ));
  }
  let hurst = match hurst_override {
    Some(hurst) => hurst,
    None => {
      let sum1: f64 = (0..(n - 4))
        .map(|i| {
          let diff = path[i + 4] - 2.0 * path[i + 2] + path[i];
          diff * diff
        })
        .sum();
      hurst_from_ratio(0.5 * (sum1 / sigma_num).log2())?
    }
  };

  let dt = delta.unwrap_or(1.0 / n_f);
  let sigma_den = n_f * (4.0 - 2.0_f64.powf(2.0 * hurst)) * dt.powf(2.0 * hurst);
  let sigma = positive((sigma_num / sigma_den).sqrt(), "sigma from the variation")?;

  let mu = path.mean().unwrap();

  let sum_x_sq: f64 = path.mapv(|x| x * x).sum();
  let sum_x: f64 = path.sum();
  let theta_num = n_f * sum_x_sq - sum_x.powi(2);
  let theta_den = n_f.powi(2) * sigma.powi(2) * hurst * gamma(2.0 * hurst);
  let theta = positive(
    (theta_num / theta_den).powf(-1.0 / (2.0 * hurst)),
    "theta from the second moment",
  )?;

  Ok(FouFit {
    hurst,
    sigma,
    mu,
    theta,
  })
}

/// Hu–Nualart–Zhou fit; `Err` below `2k + 1` points (`k + 2` with `hurst_override`), on non-finite
/// or flat data or no usable `H ∈ (0, 1)`, `σ`, `θ`; panics on a bad argument.
/// Reference: Brouste & Iacus (2013), arXiv:1112.3777, §3.1, and Coeurjolly (2001), Statistical Inference for Stochastic Processes 4, 199–227, for `H`.
///
/// $$
/// V_{k,p}(s)=\sum_{i=0}^{n-ks}\Bigl|\sum_{j=0}^{k}(-1)^{k-j}\tbinom kj X_{i+js}\Bigr|^p,\qquad
/// \hat H=\frac1p\log_2\frac{V_{k,p}(2)/(n+1-2k)}{V_{k,p}(1)/(n+1-k)}
/// $$
///
/// $$
/// |\hat\sigma|^p=\frac{n^{p\hat H-1}\,V_{k,p}(1)}{c_{k,p}\,T},\qquad
/// \hat\theta=\Bigl(\frac{\sum_i(X_i-\bar X)^2}{n\,\hat\sigma^2\hat H\,\Gamma(2\hat H)}\Bigr)^{-1/(2\hat H)}
/// $$
pub fn fit_fou_power_variation(
  path: ArrayView1<f64>,
  delta: Option<f64>,
  k: usize,
  p: f64,
  hurst_override: Option<f64>,
  sigma_override: Option<f64>,
) -> Result<FouFit, StatsError> {
  assert!(k >= 1, "k must satisfy `k >= 1`, got k = {k:?}");
  assert!(
    p.is_finite(),
    "p must satisfy `p.is_finite()`, got p = {p:?}"
  );
  assert!(p > 0.0, "p must satisfy `p > 0.0`, got p = {p:?}");
  assert_step_and_hurst(delta, hurst_override);
  if let Some(sigma) = sigma_override {
    assert!(
      sigma.is_finite(),
      "sigma_override must satisfy `sigma_override.is_finite()`, got sigma_override = {sigma:?}"
    );
    assert!(
      sigma > 0.0,
      "sigma_override must satisfy `sigma_override > 0.0`, got sigma_override = {sigma:?}"
    );
  }
  let min_n = match hurst_override {
    Some(_) => k.saturating_add(2),
    None => k.saturating_mul(2).saturating_add(1),
  };
  checked(path, min_n)?;

  let n_points = path.len();
  let dt = delta.unwrap_or(1.0 / (n_points - 1) as f64);
  let v1 = power_variation::<f64>(path, k, p, 1);

  let hurst = match hurst_override {
    Some(hurst) => hurst,
    None => {
      let v2 = power_variation::<f64>(path, k, p, 2);
      if v1 == 0.0 || v2 == 0.0 {
        return Err(StatsError::DegenerateData("zero power variation"));
      }
      let mean_ratio = (v2 / (n_points - 2 * k) as f64) / (v1 / (n_points - k) as f64);
      hurst_from_ratio(mean_ratio.log2() / p)?
    }
  };

  let sigma = match sigma_override {
    Some(sigma) => sigma,
    None => {
      let n = (n_points - 1) as f64;
      let t_horizon = n * dt;
      if v1 == 0.0 {
        return Err(StatsError::DegenerateData("zero power variation"));
      }
      let c_kp = c_k_p(k, p, hurst)?;
      let scaled = n.powf(-1.0 + p * hurst) * v1;
      let sigma_abs_p = positive(
        scaled / (c_kp * t_horizon),
        "sigma from the power variation",
      )?;
      sigma_abs_p.powf(1.0 / p)
    }
  };

  let mu = path.mean().unwrap();
  let n = (path.len() - 1) as f64;
  let sum_sq: f64 = path.slice(s![1..]).iter().map(|x| (x - mu).powi(2)).sum();
  let denom = n * sigma.powi(2) * hurst * gamma(2.0 * hurst);
  let base = positive(sum_sq / denom, "theta from the second moment")?;
  let theta = positive(
    base.powf(-1.0 / (2.0 * hurst)),
    "theta from the second moment",
  )?;

  Ok(FouFit {
    hurst,
    sigma,
    mu,
    theta,
  })
}

fn checked(path: ArrayView1<f64>, min_n: usize) -> Result<(), StatsError> {
  let path = as_slice_or_copy(path);
  validate(&path, min_n)?;
  reject_constant(&path, "constant path")
}

#[track_caller]
fn assert_step_and_hurst(delta: Option<f64>, hurst_override: Option<f64>) {
  if let Some(delta) = delta {
    assert!(
      delta.is_finite(),
      "delta must satisfy `delta.is_finite()`, got delta = {delta:?}"
    );
    assert!(
      delta > 0.0,
      "delta must satisfy `delta > 0.0`, got delta = {delta:?}"
    );
  }
  if let Some(hurst) = hurst_override {
    assert!(
      hurst > 0.0 && hurst < 1.0,
      "hurst_override must satisfy `0.0 < hurst_override < 1.0`, got hurst_override = {hurst:?}"
    );
  }
}

fn hurst_from_ratio(hurst: f64) -> Result<f64, StatsError> {
  if hurst > 0.0 && hurst < 1.0 {
    Ok(hurst)
  } else {
    Err(StatsError::Numerical {
      stage: "hurst from the variation ratio",
    })
  }
}

fn positive(value: f64, stage: &'static str) -> Result<f64, StatsError> {
  if value.is_finite() && value > 0.0 {
    Ok(value)
  } else {
    Err(StatsError::Numerical { stage })
  }
}

fn c_k_p(k: usize, p: f64, h: f64) -> Result<f64, StatsError> {
  let rho0 = positive(rho_k_h_zero(k, h), "power-variation constant")?;
  let normal_abs_p = 2.0_f64.powf(p / 2.0) * gamma((p + 1.0) / 2.0) / gamma(0.5);
  Ok(normal_abs_p * rho0.powf(p / 2.0))
}

fn rho_k_h_zero(k: usize, h: f64) -> f64 {
  let mut acc = 0.0;
  for i in -(k as isize)..=(k as isize) {
    let parity = (1_isize - i).rem_euclid(2);
    let sign = if parity == 0 { 1.0 } else { -1.0 };
    let comb = crate::hurst::variations::binomial(2 * k, (k as isize - i) as usize);
    let abs_term = (i.unsigned_abs() as f64).powf(2.0 * h);
    acc += sign * comb * abs_term;
  }
  0.5 * acc
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stochastic::diffusion::fou::Fou;

  use super::*;
  use crate::traits::ProcessExt;

  fn fou_path(hurst: f64, theta: f64, sigma: f64, n: usize, seed: u64) -> Array1<f64> {
    Fou::<f64, _>::new(
      hurst,
      theta,
      0.0,
      sigma,
      n,
      Some(0.0),
      Some(1.0),
      Deterministic::new(seed),
    )
    .sample()
  }

  #[test]
  fn fou_v4_returns_finite_params_with_fixed_hurst() {
    let n = 1024usize;
    let h_true = 0.7;
    let path = fou_path(h_true, 1.5, 0.35, n, 1);
    let res = fit_fou_power_variation(
      path.view(),
      Some(1.0 / (n - 1) as f64),
      2,
      2.0,
      Some(h_true),
      None,
    )
    .unwrap();
    assert!(res.hurst.is_finite() && (0.0..1.0).contains(&res.hurst));
    assert!(res.sigma.is_finite() && res.sigma > 0.0);
    assert!(res.theta.is_finite() && res.theta > 0.0);
  }

  #[test]
  fn fou_v4_estimated_hurst_in_range_when_not_fixed() {
    let n = 768usize;
    let path = fou_path(0.65, 1.2, 0.25, n, 2);
    let res =
      fit_fou_power_variation(path.view(), Some(1.0 / (n - 1) as f64), 2, 2.0, None, None).unwrap();
    assert!((0.0..1.0).contains(&res.hurst));
    assert!(res.sigma > 0.0 && res.theta > 0.0);
  }

  #[test]
  fn fou_v1_runs_on_fbm_path() {
    let path = fou_path(0.7, 1.5, 0.35, 1024, 3);
    let res = fit_fou_filtered_variations(path.view(), FilterType::Daubechies, None, None).unwrap();
    assert!(res.hurst.is_finite() && res.sigma > 0.0);
  }

  #[test]
  fn fou_v2_runs_on_fbm_path() {
    let path = fou_path(0.7, 1.5, 0.35, 1024, 4);
    let res = fit_fou_central_differences(path.view(), None, None).unwrap();
    assert!(res.hurst.is_finite() && res.sigma > 0.0);
  }

  #[test]
  fn fou_v1_with_hurst_override_skips_estimation() {
    let path = fou_path(0.6, 1.0, 0.2, 512, 5);
    let h_pinned = 0.42;
    let res =
      fit_fou_filtered_variations(path.view(), FilterType::Daubechies, None, Some(h_pinned))
        .unwrap();
    assert_eq!(res.hurst, h_pinned);
  }

  #[test]
  fn fou_v2_rejects_short_path() {
    let path = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
      fit_fou_central_differences(path.view(), None, None).unwrap_err(),
      StatsError::InsufficientData { needed: 5, got: 4 }
    );
  }

  #[test]
  fn short_or_non_finite_paths_are_errors() {
    let short = Array1::<f64>::linspace(0.0, 1.0, 5);
    assert_eq!(
      fit_fou_filtered_variations(short.view(), FilterType::Daubechies, None, None).unwrap_err(),
      StatsError::InsufficientData { needed: 6, got: 5 }
    );
    let mut path = fou_path(0.6, 1.0, 0.2, 64, 6);
    path[7] = f64::NAN;
    let nan_at_7 = |r: Result<FouFit, StatsError>| {
      matches!(r, Err(StatsError::InvalidObservation { index: 7, .. }))
    };
    assert!(nan_at_7(fit_fou_filtered_variations(
      path.view(),
      FilterType::Daubechies,
      None,
      None
    )));
    assert!(nan_at_7(fit_fou_central_differences(
      path.view(),
      None,
      None
    )));
    assert!(nan_at_7(fit_fou_power_variation(
      path.view(),
      None,
      2,
      2.0,
      None,
      None
    )));
  }

  #[test]
  fn a_flat_path_is_degenerate() {
    let constant = Array1::<f64>::from_elem(64, 0.3);
    let flat = StatsError::DegenerateData("constant path");
    assert_eq!(
      fit_fou_filtered_variations(constant.view(), FilterType::Daubechies, None, None).unwrap_err(),
      flat
    );
    assert_eq!(
      fit_fou_central_differences(constant.view(), None, None).unwrap_err(),
      flat
    );
    assert_eq!(
      fit_fou_power_variation(constant.view(), None, 2, 2.0, None, None).unwrap_err(),
      flat
    );
    let ramp = Array1::from_iter((0..64).map(|i| i as f64));
    assert_eq!(
      fit_fou_central_differences(ramp.view(), None, None).unwrap_err(),
      StatsError::DegenerateData("zero second-difference variation")
    );
    assert_eq!(
      fit_fou_power_variation(ramp.view(), None, 2, 2.0, None, None).unwrap_err(),
      StatsError::DegenerateData("zero power variation")
    );
    let underflowing = ramp.mapv(|x| 1e-200 * x);
    assert_eq!(
      fit_fou_filtered_variations(underflowing.view(), FilterType::Daubechies, None, None)
        .unwrap_err(),
      StatsError::DegenerateData("zero filtered variation")
    );
  }

  #[test]
  fn a_variation_ratio_outside_the_unit_interval_has_no_hurst() {
    let sawtooth =
      Array1::from_iter((0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } + 1e-3 * i as f64));
    let no_hurst = StatsError::Numerical {
      stage: "hurst from the variation ratio",
    };
    assert_eq!(
      fit_fou_central_differences(sawtooth.view(), None, None).unwrap_err(),
      no_hurst
    );
    assert_eq!(
      fit_fou_power_variation(sawtooth.view(), None, 1, 2.0, None, None).unwrap_err(),
      no_hurst
    );
  }

  #[test]
  #[should_panic(expected = "hurst_override must satisfy `0.0 < hurst_override < 1.0`")]
  fn a_hurst_override_outside_the_unit_interval_is_a_precondition() {
    let path = fou_path(0.6, 1.0, 0.2, 64, 7);
    let _ = fit_fou_central_differences(path.view(), None, Some(1.0));
  }

  #[test]
  #[should_panic(expected = "delta must satisfy `delta > 0.0`")]
  fn a_non_positive_step_is_a_precondition() {
    let path = fou_path(0.6, 1.0, 0.2, 64, 8);
    let _ = fit_fou_filtered_variations(path.view(), FilterType::Daubechies, Some(0.0), None);
  }

  #[test]
  fn fou_estimate_result_round_trips_tuple() {
    let r: FouFit = (0.7, 0.3, 0.05, 1.2).into();
    assert_eq!(r.hurst, 0.7);
    assert_eq!(r.sigma, 0.3);
    let t: (f64, f64, f64, f64) = r.into();
    assert_eq!(t, (0.7, 0.3, 0.05, 1.2));
  }
}
