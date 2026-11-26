use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::test_support::simulate_cir;
use super::test_support::simulate_ou;
use super::*;

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

/// `X_{t+1} = 0.37 X_t + 1.1` in doubles from 3, whose residuals are rounding alone, near `1e-15`.
fn rounded_ar1() -> Array1<f64> {
  let mut x = Array1::<f64>::zeros(12);
  x[0] = 3.0;
  for t in 1..12 {
    x[t] = 0.37 * x[t - 1] + 1.1;
  }
  x
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
    message: "converged".into(),
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
  assert!(fits.iter().all(|(_, q)| q.message == "converged"));
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

/// Best of three pinned paths, as the level's band sits near one standard error.
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
  assert!(fits.iter().all(|(_, q)| q.message == "converged"));
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

/// The QMLE meets the closed form at a level below zero and at a `σ` near `2e-4`, and a path scaled
/// by `c` scales `mu` and `sigma` alone: the level and the start are in the path's own units.
#[test]
fn a_negative_level_is_the_closed_forms_at_every_scale() {
  let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
  let paths = [
    (negative_level_path(), 100.0 / 9_999.0),
    (
      simulate_ou(8.0, 0.03, 0.02, 0.03, 1.0 / 252.0, 100, 48),
      1.0 / 252.0,
    ),
  ];
  for (path, dt) in paths {
    let fit =
      |x: &Array1<f64>| fit_diffusion_qmle(x.view(), dt, DiffusionKind::OrnsteinUhlenbeck).unwrap();
    let unit = fit(&path);
    for c in [1e-4, 1e-2, 1.0, 1e4] {
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
}

/// At `2^-532` the start residuals' mean square underflows, so `σ₀` falls back to the `σ` whose
/// stationary variance `v₁ σ²` is the path's: a scale the data set, not a constant.
#[test]
fn a_start_whose_residuals_underflow_takes_its_scale_from_the_path() {
  let dt = 1.0 / 252.0;
  let paths = [
    (
      DiffusionKind::OrnsteinUhlenbeck,
      simulate_ou(2.0, 0.5, 0.1, 0.5, dt, 1000, 11),
    ),
    (
      DiffusionKind::Cir,
      simulate_cir(2.0, 0.5, 0.1, 0.5, dt, 1000, 11),
    ),
  ];
  for (kind, path) in paths {
    let x = path.mapv(|v| v * 2f64.powi(-532));
    let xs = x.as_slice().unwrap();
    let [ln_theta, _, ln_sigma] = ar1_start(xs, dt, kind, Carrier::new(xs, kind).unwrap()).unwrap();
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let ss = xs[..xs.len() - 1]
      .iter()
      .map(|v| (v - mean).powi(2))
      .sum::<f64>();
    let level = if kind == DiffusionKind::Cir {
      mean
    } else {
      1.0
    };
    let v1 = level / (2.0 * ln_theta.exp());
    let want = ss.sqrt() / ((xs.len() - 1) as f64 * v1).sqrt();
    assert!(
      (ln_sigma.exp() / want - 1.0).abs() < 1e-9,
      "{kind:?}: σ₀ = {:e} against {want:e}",
      ln_sigma.exp()
    );
  }
}

/// A CIR path scaled by `c` fits as `(θ, cμ, √c σ)`: the start's level is the path's mean, with no
/// floor in the data's units, down to a mean near `1e-62`.
#[test]
fn a_cir_fit_moves_with_the_data_scale() {
  let dt = 1.0 / 252.0;
  let path = simulate_cir(2.0, 0.04, 0.2, 0.04, dt, 2000, 7);
  let fit = |x: &Array1<f64>| fit_diffusion_qmle(x.view(), dt, DiffusionKind::Cir).unwrap();
  let unit = fit(&path);
  for k in [-30, -200] {
    let c = 2f64.powi(k);
    let q = fit(&path.mapv(|v| c * v));
    let want = [unit.theta, c * unit.mu, c.sqrt() * unit.sigma];
    for (a, b) in [q.theta, q.mu, q.sigma].into_iter().zip(want) {
      assert!((a - b).abs() / b.abs() < 1e-4, "2^{k}: {a} against {b}");
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

/// At `θΔ = 5e-13` a variance from `1 − e^{−θΔ}` formed by subtraction keeps four digits; the
/// series `σ²Δ(1 − θΔ)` (OU) and `σ²Δ(x(1 − 3θΔ/2) + μθΔ/2)` (CIR) keep all sixteen.
#[test]
fn the_variance_keeps_its_digits_at_a_small_speed_times_step() {
  let (theta, mu, sigma, dt) = (50.0, 0.05, 1.0, 1e-14);
  let h = theta * dt;
  let x = [mu, mu + 2e-7];
  let r = x[1] - mu;
  let v_ou = sigma * sigma * dt * (1.0 - h);
  let v_cir = sigma * sigma * dt * (x[0] * (1.0 - 1.5 * h) + mu * 0.5 * h);
  for (kind, v) in [
    (DiffusionKind::OrnsteinUhlenbeck, v_ou),
    (DiffusionKind::Cir, v_cir),
  ] {
    let want = -0.5 * ((2.0 * std::f64::consts::PI * v).ln() + r * r / v);
    let got = quasi_log_likelihood(&x, dt, kind, theta, mu, sigma);
    assert!(
      (got - want).abs() < 1e-12,
      "{kind:?}: {got} against the series' {want}"
    );
  }
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

/// For OU the two fits are one estimator, so a path the closed form refuses gets its `Err` here.
#[test]
fn the_ou_qmle_refuses_what_the_closed_form_refuses() {
  let cases = [
    // `X_{t+1} = 0.5 X_t + 1` exactly: the slope is in range and every residual is zero.
    (
      Array1::from(vec![0.0, 1.0, 1.5, 1.75, 1.875, 1.9375, 1.96875]),
      "zero residual variance",
    ),
    // Rounding alone, which the exact-fit rule refuses as it refuses exact zeros.
    (rounded_ar1(), "zero residual variance"),
    (
      Array1::from_iter((0..12).map(|t| 1.1_f64.powi(t) + 0.01 * (t as f64).sin())),
      "series without mean reversion",
    ),
    (
      Array1::from_iter((0..12).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 } + 0.01 * t as f64)),
      "series without positive autocorrelation",
    ),
    (
      Array1::from(vec![0.05, 0.05, 0.05, 0.07]),
      "constant series",
    ),
    // `X_{t+1} = c` fits with slope 0, which rounding makes `±1e-31`: the slope test a coin flip.
    (Array1::from(vec![1.0, 0.7, 0.7, 0.7]), "constant series"),
    (
      Array1::from(vec![0.11, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]),
      "constant series",
    ),
    // A regressand of neighbouring doubles is a constant's rounding, with a variance of its own.
    (
      Array1::from_iter([0.0, 1.0, -1.0, 1.0, 0.0, -1.0].map(|j| 0.7 + j * f64::EPSILON / 2.0)),
      "constant series",
    ),
    (
      Array1::from(vec![
        1.0,
        0.7,
        0.7 + f64::EPSILON,
        0.7,
        0.7 - f64::EPSILON / 2.0,
        0.7,
      ]),
      "constant series",
    ),
    // Both read the path at unit scale, so its sums and squares stay in the double range.
    (
      Array1::from(vec![6e153, -6e153, 6e153, -6e153]),
      "series without positive autocorrelation",
    ),
    (
      Array1::from(vec![1.3e154, 0.0, 1.3e154, 0.0, 1.3e154]),
      "series without positive autocorrelation",
    ),
    (
      Array1::from(vec![1e308, 1.5e308, 1e308, 0.5e308, 1.2e308]),
      "series without positive autocorrelation",
    ),
    (
      Array1::from(vec![1e-170, 2e-170, 1.5e-170, 3e-170]),
      "series without positive autocorrelation",
    ),
  ];
  let dt = 1.0 / 252.0;
  for (x, why) in cases {
    let fit = fit_diffusion_qmle(x.view(), dt, DiffusionKind::OrnsteinUhlenbeck);
    let refused = Err(StatsError::DegenerateData(why));
    assert_eq!(fit.map(|f| (f.theta, f.sigma, f.converged)), refused);
    assert_eq!(
      fit_ou_mle(x.view(), dt).map(|f| (f.theta, f.sigma, f.converged)),
      refused
    );
  }
}

/// On an exact fit CIR's quasi-likelihood, as OU's, grows without bound as `σ` falls, so its QMLE
/// refuses `X_{t+1} = 0.37 X_t + 1.1` in doubles by the same rule.
#[test]
fn a_cir_exact_fit_has_zero_residual_variance() {
  let fit = fit_diffusion_qmle(rounded_ar1().view(), 1.0 / 252.0, DiffusionKind::Cir);
  assert_eq!(
    fit.map(|f| (f.theta, f.sigma, f.converged)),
    Err(StatsError::DegenerateData("zero residual variance"))
  );
}

/// A negative lag-1 slope puts OU's supremum at the iid limit `θ → ∞`, no interior speed, so OU is
/// refused as the closed form refuses it; CIR's start still takes the autocorrelation at its clamp.
#[test]
fn a_negative_autocorrelation_is_degenerate_for_ou_and_starts_cir_at_the_clamp() {
  let s = Array1::from(vec![0.05, 0.03, 0.06, 0.02, 0.07, 0.01]);
  let dt = 1.0 / 252.0;
  let ou = fit_diffusion_qmle(s.view(), dt, DiffusionKind::OrnsteinUhlenbeck);
  assert_eq!(
    ou.map(|f| (f.theta, f.sigma, f.converged)),
    Err(StatsError::DegenerateData(
      "series without positive autocorrelation"
    ))
  );
  assert!(fit_diffusion_qmle(s.view(), dt, DiffusionKind::Cir).is_ok());
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

/// Shifted far from zero against its spread, the path keeps the closed form's slope: its sums run
/// about the means.
#[test]
fn the_closed_form_keeps_its_slope_at_a_level_far_above_its_spread() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(1.8, 0.06, 0.12, 0.06, dt, 6000, 3);
  let unit = fit_ou_mle(path.view(), dt).unwrap();
  for (c, b) in [(1e-3, 1000.0), (1e-3, -5.0)] {
    let fit = fit_ou_mle(path.mapv(|v| c * v + b).view(), dt).unwrap();
    assert!(
      (fit.theta / unit.theta - 1.0).abs() < 1e-6,
      "{c} x + {b}: θ̂ = {}",
      fit.theta
    );
    assert!(
      (fit.sigma / (c * unit.sigma) - 1.0).abs() < 1e-6,
      "{c} x + {b}: σ̂ = {}",
      fit.sigma
    );
    assert!(
      (fit.mu - (c * unit.mu + b)).abs() < 1e-9,
      "{c} x + {b}: μ̂ = {}",
      fit.mu
    );
  }
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

/// Integers of an AR(1) at slope 0.6 around `spread / 2`, between 0 and `spread`.
fn integer_ar1(n: usize, spread: f64, seed: u64) -> Vec<f64> {
  let mut noise = vec![0.0_f64; n];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut noise);
  let mut state = 0.0;
  let level = |z: f64| {
    (0.5 * spread + 0.16 * spread * z)
      .round()
      .clamp(0.0, spread)
  };
  let step = |z: &f64| {
    state = 0.6 * state + 0.8 * z;
    level(state)
  };
  noise.iter().map(step).collect()
}

/// `c + k ulp(c)` is exact, so a path of neighbouring doubles at `c` is the integers' path moved
/// and scaled: the slope is theirs to `2(m+3)ε(Σ|dᵢeᵢ|/|Σdᵢeᵢ| + 1)`, the sums' own rounding.
#[test]
fn the_ou_closed_form_reads_a_path_of_neighbouring_doubles_as_its_integers() {
  let dt = 1.0;
  for spread in [20.0, 600.0] {
    let integers = integer_ar1(400, spread, 31);
    let m = (integers.len() - 1) as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (lagged, next) = (&integers[..399], &integers[1..]);
    let (lagged_mean, next_mean) = (mean(lagged), mean(next));
    let products = lagged
      .iter()
      .zip(next)
      .map(|(u, v)| (u - lagged_mean) * (v - next_mean));
    let (sum, size) = products.fold((0.0, 0.0), |(s, a), p| (s + p, a + p.abs()));
    let bound = 2.0 * (m + 3.0) * f64::EPSILON * (size / sum.abs() + 1.0);
    let reference = fit_ou_mle(Array1::from_vec(integers.clone()).view(), dt).unwrap();
    for level in [1.0_f64, 3.0, 1e3, 1e8, -7e5] {
      let ulp = level.abs().next_up() - level.abs();
      let path = Array1::from_iter(integers.iter().map(|k| level + k * ulp));
      let fit = fit_ou_mle(path.view(), dt).unwrap();
      let (slope, theirs) = ((-fit.theta * dt).exp(), (-reference.theta * dt).exp());
      assert!(
        (slope - theirs).abs() <= bound * theirs,
        "spread {spread}, level {level}: slope {slope} against the integers' {theirs}"
      );
      let sigma = fit.sigma / ulp;
      assert!(
        (sigma - reference.sigma).abs() <= 4.0 * bound * reference.sigma,
        "spread {spread}, level {level}: sigma {sigma} against {}",
        reference.sigma
      );
      // The level is one rounded mean at `ulp(c)`, and the step over `1 − β` one more.
      let mu = (fit.mu - level) / ulp;
      assert!(
        (mu - reference.mu).abs() <= 1.0 + 1.0 / (1.0 - theirs),
        "spread {spread}, level {level}: mu {mu} lattice steps against {}",
        reference.mu
      );
    }
  }
}

/// A level or a `σ` that is a subnormal in the path's own units holds a few of its digits only: the
/// closed form and the QMLE refuse it as the other fits do, where rc.5 answered it.
#[test]
fn a_fit_that_is_subnormal_in_the_paths_units_is_a_data_scale_error() {
  let path = negative_level_path().mapv(|v| v * 1e-310);
  let dt = 100.0 / 9_999.0;
  let lost = StatsError::Numerical {
    stage: "data scale",
  };
  assert_eq!(fit_ou_mle(path.view(), dt).unwrap_err(), lost);
  let kind = DiffusionKind::OrnsteinUhlenbeck;
  assert_eq!(fit_diffusion_qmle(path.view(), dt, kind).unwrap_err(), lost);
}
