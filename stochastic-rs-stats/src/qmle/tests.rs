use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::*;

/// Exact OU transition: Gaussian with mean μ+(X−μ)e^{−θΔ} and variance σ²(1−e^{−2θΔ})/(2θ).
fn simulate_ou(
  theta: f64,
  mu: f64,
  sigma: f64,
  x0: f64,
  dt: f64,
  n: usize,
  seed: u64,
) -> Array1<f64> {
  let a = (-theta * dt).exp();
  let sd = (sigma * sigma * (1.0 - a * a) / (2.0 * theta)).sqrt();
  let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
  let mut path = Array1::<f64>::zeros(n);
  path[0] = x0;
  for t in 1..n {
    let z = normal.sample();
    path[t] = mu + (path[t - 1] - mu) * a + sd * z;
  }
  path
}

/// Exact CIR transition via non-central χ² (Poisson mixture of Gamma).
fn simulate_cir(
  theta: f64,
  mu: f64,
  sigma: f64,
  x0: f64,
  dt: f64,
  n: usize,
  seed: u64,
) -> Array1<f64> {
  let a = (-theta * dt).exp();
  let s2 = sigma * sigma;
  let c = s2 * (1.0 - a) / (4.0 * theta);
  let d = 4.0 * theta * mu / s2;
  let mut path = Array1::<f64>::zeros(n);
  path[0] = x0;
  // λ and the χ² shape change every step, so each step seeds its own stream from `seed` and `t`:
  // one seed for every step would restart the same stream and destroy the path.
  for t in 1..n {
    let lambda = path[t - 1] * 4.0 * theta * a / (s2 * (1.0 - a));
    let pois_seed = seed ^ (t as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let gamma_seed = seed ^ (t as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let n_pois: f64 = f64::from(
      SimdPoisson::<u32>::new(lambda / 2.0)
        .seeded(&Deterministic::new(pois_seed))
        .sample(),
    );
    let shape = d / 2.0 + n_pois;
    let chi2 = if shape > 0.0 {
      SimdGamma::<f64>::new(shape, 2.0)
        .seeded(&Deterministic::new(gamma_seed))
        .sample()
    } else {
      0.0
    };
    path[t] = c * chi2;
  }
  path
}

/// `Ou` at speed 5, level −0.5 and `σ = 0.3`, 10 000 points over `t = 100`.
fn negative_level_path() -> Array1<f64> {
  Ou::<f64, _>::new(
    5.0,
    -0.5,
    0.3,
    10_000,
    Some(-0.5),
    Some(100.0),
    Deterministic::new(5),
  )
  .sample()
}

/// A fit of `kind` at fixed parameters, for the conversions' preconditions.
fn literal(kind: DiffusionKind) -> QmleFit {
  QmleFit {
    theta: 1.5,
    mu: 0.04,
    sigma: 0.2,
    kind,
    log_likelihood: 0.0,
    iterations: 0,
    converged: true,
  }
}

/// The OU transition is Gaussian, so the QMLE is the exact MLE and meets the closed form to the
/// optimiser's tolerance.
#[test]
fn qmle_ou_matches_closed_form_mle() {
  let path = simulate_ou(1.8, 0.06, 0.12, 0.06, 1.0 / 252.0, 6000, 3);
  let q = fit_diffusion_qmle(path.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck).unwrap();
  let cf = fit_ou_mle(path.view(), 1.0 / 252.0).unwrap();
  assert!(
    (q.theta - cf.theta).abs() / cf.theta < 1e-3,
    "QMLE θ={} vs closed-form θ={}",
    q.theta,
    cf.theta
  );
  assert!(
    (q.mu - cf.mu).abs() / cf.mu.abs() < 1e-3,
    "QMLE μ={} vs closed-form μ={}",
    q.mu,
    cf.mu
  );
  assert!(
    (q.sigma - cf.sigma).abs() / cf.sigma < 1e-3,
    "QMLE σ={} vs closed-form σ={}",
    q.sigma,
    cf.sigma
  );
}

/// Best of three pinned paths: the level's band sits near one standard error, `σ/(θ√T)` over a
/// span of 32 years, so one path alone is a coin flip.
#[test]
fn qmle_ou_recovers_true_params() {
  let (tht, mt, st) = (1.8, 0.06, 0.12);
  let fits = [5u64, 11, 23].map(|seed| {
    let path = simulate_ou(tht, mt, st, 0.06, 1.0 / 252.0, 8000, seed);
    (
      seed,
      fit_diffusion_qmle(path.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck).unwrap(),
    )
  });
  let ok = fits.iter().any(|(_, q)| {
    q.converged
      && (q.mu - mt).abs() / mt < 0.2
      && (q.theta - tht).abs() / tht < 0.35
      && (q.sigma - st).abs() / st < 0.06
  });
  let report = fits
    .iter()
    .map(|(s, q)| format!("seed {s}: θ={:.4} μ={:.4} σ={:.4}", q.theta, q.mu, q.sigma))
    .collect::<Vec<_>>()
    .join("; ");
  assert!(ok, "no seed recovered (θ={tht}, μ={mt}, σ={st}) — {report}");
}

/// CIR round-trip: QMLE with exact CIR conditional moments recovers the
/// true parameters from an exactly-distributed CIR path.
#[test]
fn qmle_cir_recovers_true_params() {
  let (tht, mt, st) = (2.0, 0.04, 0.25);
  let fits = [7u64, 13, 29].map(|seed| {
    let path = simulate_cir(tht, mt, st, 0.04, 1.0 / 252.0, 8000, seed);
    (
      seed,
      fit_diffusion_qmle(path.view(), 1.0 / 252.0, DiffusionKind::Cir).unwrap(),
    )
  });
  let ok = fits.iter().any(|(_, q)| {
    q.converged
      && (q.mu - mt).abs() / mt < 0.12
      && (q.theta - tht).abs() / tht < 0.5
      && (q.sigma - st).abs() / st < 0.3
  });
  let report = fits
    .iter()
    .map(|(s, q)| format!("seed {s}: θ={:.4} μ={:.4} σ={:.4}", q.theta, q.mu, q.sigma))
    .collect::<Vec<_>>()
    .join("; ");
  assert!(ok, "no seed recovered (θ={tht}, μ={mt}, σ={st}) — {report}");
}

/// The fit's `theta` is the speed and `mu` the level, as in `Ou`, so the conversion hands them
/// over; a level below zero is fitted as any other.
#[test]
fn an_ou_fit_round_trips_into_the_process_it_names() {
  let path = negative_level_path();
  let dt = 100.0 / 9_999.0;
  let fit = fit_diffusion_qmle(path.view(), dt, DiffusionKind::OrnsteinUhlenbeck).unwrap();
  assert!((fit.theta - 5.0).abs() < 1.5, "speed {}", fit.theta);
  assert!((fit.mu + 0.5).abs() < 0.05, "level {}", fit.mu);
  let rebuilt = fit.to_ou(10_000, Some(-0.5), Some(100.0), Deterministic::new(6));
  assert_eq!(
    (rebuilt.theta, rebuilt.mu, rebuilt.sigma),
    (fit.theta, fit.mu, fit.sigma)
  );
  assert_eq!(rebuilt.sample().len(), 10_000);
}

/// The QMLE meets the closed form at a level below zero, and a path scaled by 1e4 scales `mu` and
/// `sigma` alone: the level is carried in the path's standard deviations.
#[test]
fn a_negative_level_is_the_closed_forms_at_every_scale() {
  let path = negative_level_path();
  let dt = 100.0 / 9_999.0;
  let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
  let fit =
    |x: &Array1<f64>| fit_diffusion_qmle(x.view(), dt, DiffusionKind::OrnsteinUhlenbeck).unwrap();
  let unit = fit(&path);
  for c in [1.0, 1e4] {
    let x = path.mapv(|v| c * v);
    let q = fit(&x);
    let cf = fit_ou_mle(x.view(), dt).unwrap();
    for (a, b) in [(q.theta, cf.theta), (q.mu, cf.mu), (q.sigma, cf.sigma)] {
      assert!(
        rel(a, b) < 1e-5,
        "scale {c}: QMLE {a} against the closed form's {b}"
      );
    }
    let scaled = [unit.theta, c * unit.mu, c * unit.sigma];
    for (a, b) in [q.theta, q.mu, q.sigma].into_iter().zip(scaled) {
      assert!(
        rel(a, b) < 1e-5,
        "scale {c}: {a} against the unit fit's {b}"
      );
    }
  }
}

#[test]
fn a_cir_fit_builds_a_cir() {
  let truth = Cir::<f64, _>::new(
    1.5,
    0.04,
    0.2,
    4_000,
    Some(0.04),
    Some(16.0),
    VarianceBoundary::Reflect,
    Deterministic::new(9),
  );
  let path = truth.sample();
  let fit = fit_diffusion_qmle(path.view(), 16.0 / 3_999.0, DiffusionKind::Cir).unwrap();
  assert_eq!(fit.kind, DiffusionKind::Cir);
  let rebuilt = fit.to_cir(
    4_000,
    Some(0.04),
    Some(16.0),
    VarianceBoundary::Reflect,
    Deterministic::new(10),
  );
  assert_eq!(
    (rebuilt.theta, rebuilt.mu, rebuilt.sigma),
    (fit.theta, fit.mu, fit.sigma)
  );
  assert_eq!(rebuilt.boundary, VarianceBoundary::Reflect);
}

#[test]
#[should_panic(expected = "kind must satisfy `kind == DiffusionKind::OrnsteinUhlenbeck`")]
fn to_ou_on_a_cir_fit_is_a_precondition() {
  let _ = literal(DiffusionKind::Cir).to_ou(10, None, None, Unseeded);
}

#[test]
#[should_panic(expected = "kind must satisfy `kind == DiffusionKind::Cir`")]
fn to_cir_on_an_ou_fit_is_a_precondition() {
  let ou = literal(DiffusionKind::OrnsteinUhlenbeck);
  let _ = ou.to_cir(10, None, None, VarianceBoundary::Reflect, Unseeded);
}

#[test]
fn fit_ou_mle_is_an_ou_fit() {
  let path = simulate_ou(1.8, 0.06, 0.12, 0.06, 1.0 / 252.0, 600, 3);
  let fit = fit_ou_mle(path.view(), 1.0 / 252.0).unwrap();
  assert_eq!(fit.kind, DiffusionKind::OrnsteinUhlenbeck);
  assert_eq!(fit.to_ou(10, None, None, Unseeded).theta, fit.theta);
}

/// Three points are two transitions, which the drift's two coefficients fit exactly.
#[test]
fn qmle_rejects_a_short_series() {
  let s = Array1::from(vec![0.05, 0.06, 0.04]);
  for kind in [DiffusionKind::OrnsteinUhlenbeck, DiffusionKind::Cir] {
    assert_eq!(
      fit_diffusion_qmle(s.view(), 1.0 / 252.0, kind).unwrap_err(),
      StatsError::InsufficientData { needed: 4, got: 3 }
    );
  }
}

/// At `θ = 0` the OU variance is `0 / 0`, so the log-likelihood is NaN rather than a number.
#[test]
fn a_conditional_variance_outside_the_model_gives_a_nan_log_likelihood() {
  let x = [0.05, 0.06, 0.04];
  let ou = DiffusionKind::OrnsteinUhlenbeck;
  assert!(quasi_log_likelihood(&x, 1.0 / 252.0, ou, 0.0, 0.05, 0.1).is_nan());
  assert!(quasi_log_likelihood(&x, 1.0 / 252.0, ou, 1.0, 0.05, 0.1).is_finite());
}

#[test]
fn a_negative_cir_observation_is_invalid() {
  let s = Array1::from(vec![0.04, 0.05, 0.045, -0.01, 0.04, 0.05]);
  assert_eq!(
    fit_diffusion_qmle(s.view(), 1.0 / 252.0, DiffusionKind::Cir).unwrap_err(),
    StatsError::InvalidObservation {
      index: 3,
      value: -0.01,
      constraint: "finite and >= 0"
    }
  );
}

/// The sample mean of these finite points overflows, so the regression start is not finite.
#[test]
fn an_overflowing_start_is_a_numerical_error() {
  let s = Array1::from(vec![1e308, 1.5e308, 1e308, 0.5e308, 1.2e308]);
  assert_eq!(
    fit_diffusion_qmle(s.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck).unwrap_err(),
    StatsError::Numerical {
      stage: "starting point"
    }
  );
}

/// The start is finite, `σ₀` near 7e153 clamped to 10, but there a jump of 1e154 among zeros
/// overflows the likelihood at every vertex the simplex tries.
#[test]
fn a_non_finite_optimum_is_a_numerical_error() {
  let mut s = Array1::<f64>::zeros(1000);
  s[500] = 1e154;
  assert_eq!(
    fit_diffusion_qmle(s.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck).unwrap_err(),
    StatsError::Numerical {
      stage: "quasi-log-likelihood"
    }
  );
}

/// The mean, the autocorrelation and the speed are finite, but the residuals near 1.1e154 square
/// past the largest double: `σ₀` is `∞` before its clamp to 10.
#[test]
fn an_overflowing_sigma_start_is_a_numerical_error() {
  let s = Array1::from(vec![6e153, -6e153, 6e153, -6e153]);
  assert_eq!(
    fit_diffusion_qmle(s.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck).unwrap_err(),
    StatsError::Numerical {
      stage: "starting point"
    }
  );
}

/// Deviations near 1e-170 square to zero, so the lag-1 autocorrelation is `0 / 0`.
#[test]
fn an_underflowing_autocorrelation_is_a_numerical_error() {
  let s = Array1::from(vec![1e-170, 2e-170, 1.5e-170, 3e-170]);
  assert_eq!(
    fit_diffusion_qmle(s.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck).unwrap_err(),
    StatsError::Numerical {
      stage: "starting point"
    }
  );
}

/// A negative lag-1 autocorrelation is data, which the start's clamp to `[1e-4, 0.999]` takes.
#[test]
fn a_negative_autocorrelation_starts_at_the_clamp() {
  let s = Array1::from(vec![0.05, 0.03, 0.06, 0.02, 0.07, 0.01]);
  assert!(fit_diffusion_qmle(s.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck).is_ok());
}

/// Three points are two transitions, which the two AR(1) coefficients fit exactly.
#[test]
fn ou_mle_needs_four_points() {
  assert_eq!(
    fit_ou_mle(Array1::from(vec![0.05, 0.06, 0.04]).view(), 1.0 / 252.0).unwrap_err(),
    StatsError::InsufficientData { needed: 4, got: 3 }
  );
  assert!(fit_ou_mle(Array1::from(vec![1.0, 0.8, 0.7, 0.5]).view(), 1.0 / 252.0).is_ok());
}

#[test]
fn an_ar1_slope_outside_the_unit_interval_is_degenerate_data() {
  let explosive = Array1::from_iter((0..12).map(|t| 1.1_f64.powi(t) + 0.01 * (t as f64).sin()));
  assert_eq!(
    fit_ou_mle(explosive.view(), 1.0 / 252.0).unwrap_err(),
    StatsError::DegenerateData("series without mean reversion")
  );
  let alternating =
    Array1::from_iter((0..12).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 } + 0.01 * t as f64));
  assert_eq!(
    fit_ou_mle(alternating.view(), 1.0 / 252.0).unwrap_err(),
    StatsError::DegenerateData("series without positive autocorrelation")
  );
}

/// `X_{t+1} = 0.5 X_t + 1` exactly: the slope is in range and every residual is zero.
#[test]
fn an_exact_ar1_path_has_zero_residual_variance() {
  let x = Array1::from(vec![0.0, 1.0, 1.5, 1.75, 1.875, 1.9375, 1.96875]);
  assert_eq!(
    fit_ou_mle(x.view(), 1.0 / 252.0).unwrap_err(),
    StatsError::DegenerateData("zero residual variance")
  );
}

/// At `θΔ = 5e-7` the slope lies in `(1 − 1e-6, 1)`, and `θ̂` is read off it, not off a bound.
#[test]
fn ou_mle_reads_theta_off_a_slope_near_one() {
  let dt = 1e-6;
  let mut x = simulate_ou(0.5, 0.0, 1e-3, 1000.0, dt, 1000, 5);
  x[0] = 1000.0;
  let fit = fit_ou_mle(x.view(), dt).unwrap();
  assert!(fit.theta < 0.99, "theta = {}", fit.theta);
}

#[test]
fn a_constant_regressor_is_degenerate_data() {
  let s = Array1::from(vec![0.05, 0.05, 0.05, 0.07]);
  assert_eq!(
    fit_ou_mle(s.view(), 1.0 / 252.0).unwrap_err(),
    StatsError::DegenerateData("constant series")
  );
}
