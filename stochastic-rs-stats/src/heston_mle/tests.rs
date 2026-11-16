use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::s;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::HestonMleFit;
use super::fit_heston_nmle;
use super::fit_heston_nmle_with_delta;
use super::fit_heston_pmle;
use super::fit_heston_pmle_with_delta;
use crate::StatsError;

type Fit = fn(ArrayView1<f64>, ArrayView1<f64>) -> Result<HestonMleFit, StatsError>;

fn fits() -> [Fit; 4] {
  [
    |s, v| fit_heston_nmle(s, v, 0.01),
    |s, v| fit_heston_nmle_with_delta(s, v, 0.01, 1.0 / 252.0),
    |s, v| fit_heston_pmle(s, v, 0.01),
    |s, v| fit_heston_pmle_with_delta(s, v, 0.01, 1.0 / 252.0),
  ]
}

fn wiggly_prices(n: usize) -> Array1<f64> {
  Array1::from_iter((0..n).map(|t| 100.0 + (t as f64 * 0.9).sin()))
}

/// Two drift coefficients fit three points exactly, and the closed forms divide by `V` and `ln S`.
#[test]
fn every_fit_needs_four_positive_points() {
  let s = wiggly_prices(5);
  let v = Array1::from(vec![0.04, 0.05, 0.045, 0.042, 0.047]);
  let mut zero_variance = v.clone();
  zero_variance[2] = 0.0;
  let mut negative_price = s.clone();
  negative_price[3] = -1.0;
  for fit in fits() {
    assert_eq!(
      fit(s.slice(s![..3]), v.slice(s![..3])).unwrap_err(),
      StatsError::InsufficientData { needed: 4, got: 3 }
    );
    assert_eq!(
      fit(s.view(), zero_variance.view()).unwrap_err(),
      StatsError::InvalidObservation {
        index: 2,
        value: 0.0,
        constraint: "finite and > 0"
      }
    );
    assert_eq!(
      fit(negative_price.view(), v.view()).unwrap_err(),
      StatsError::InvalidObservation {
        index: 3,
        value: -1.0,
        constraint: "finite and > 0"
      }
    );
  }
}

/// Both closed forms regress on the lagged variances `V_0..V_{n-1}`, equal here.
#[test]
fn a_constant_lagged_variance_is_degenerate_data() {
  let v = Array1::from(vec![0.04, 0.04, 0.04, 0.04, 0.05]);
  for fit in fits() {
    assert_eq!(
      fit(wiggly_prices(5).view(), v.view()).unwrap_err(),
      StatsError::DegenerateData("constant variance series")
    );
  }
}

#[test]
fn a_variance_path_without_mean_reversion_is_degenerate_data() {
  let v = Array1::from_iter(
    (0..40).map(|t| 0.02 * 1.03_f64.powi(t) * (1.0 + 0.02 * (t as f64 * 1.3).sin())),
  );
  for fit in fits() {
    assert_eq!(
      fit(wiggly_prices(40).view(), v.view()).unwrap_err(),
      StatsError::DegenerateData("variance series without mean reversion")
    );
  }
}

/// A path decaying toward `-0.01` fits a mean reversion to a long-run variance below zero.
#[test]
fn a_non_positive_long_run_variance_is_degenerate_data() {
  let v = Array1::from_iter(
    (0..16).map(|t| 0.06 * 0.9_f64.powi(t) - 0.01 + 0.0002 * (t as f64 * 1.7).sin()),
  );
  for fit in fits() {
    assert_eq!(
      fit(wiggly_prices(16).view(), v.view()).unwrap_err(),
      StatsError::DegenerateData("non-positive long-run variance")
    );
  }
}

/// At `κδ = 5e-7` the PMLE's `β̂₁` is in `(1 − 1e-6, 1)`, and `κ̂` is read off it, not off a bound.
#[test]
fn pmle_reads_kappa_off_a_persistence_near_one() {
  let (n, delta) = (1000, 1e-6);
  let a = (-0.5_f64 * delta).exp();
  let mut noise = SimdNormal::<f64>::new(0.0, 1e-7).seeded(&Deterministic::new(3));
  let mut v = Array1::<f64>::zeros(n);
  v[0] = 100.0;
  for t in 1..n {
    v[t] = 0.04 + (v[t - 1] - 0.04) * a + noise.sample();
  }
  let s = Array1::from_elem(n, 100.0);
  let fit = fit_heston_pmle_with_delta(s.view(), v.view(), 0.0, delta).unwrap();
  assert!(fit.kappa < 0.99, "kappa = {}", fit.kappa);
}

/// A variance path on its own drift leaves residuals at rounding level, and `σ̂` is read off them.
#[test]
fn nmle_reads_a_rounding_level_sigma_off_its_residuals() {
  let (n, delta, kappa, theta) = (64, 1.0 / 252.0, 2.0, 0.04);
  let mut root = vec![0.25_f64; n];
  for k in 1..n {
    let lagged = root[k - 1] * root[k - 1];
    root[k] = root[k - 1] + delta / (2.0 * root[k - 1]) * (kappa * theta - kappa * lagged);
  }
  let v = Array1::from_iter(root.iter().map(|r| r * r));
  let s = Array1::from_elem(n, 100.0);
  let fit = fit_heston_nmle_with_delta(s.view(), v.view(), 0.0, delta).unwrap();
  assert!(fit.sigma < 1e-6, "sigma = {}", fit.sigma);
}

#[test]
fn nmle_heston_rejects_mismatched_lengths() {
  let s = Array1::from(vec![100.0, 101.0, 102.0]);
  let v = Array1::from(vec![0.04, 0.05]);
  assert_eq!(
    fit_heston_nmle(s.view(), v.view(), 0.01).unwrap_err(),
    StatsError::DimensionMismatch {
      expected: 3,
      got: 2
    }
  );
}

#[test]
fn nmle_heston_rejects_a_short_series() {
  let s = Array1::from(vec![100.0]);
  let v = Array1::from(vec![0.04]);
  assert_eq!(
    fit_heston_nmle(s.view(), v.view(), 0.01).unwrap_err(),
    StatsError::InsufficientData { needed: 4, got: 1 }
  );
}

#[test]
fn a_non_finite_or_constant_series_is_an_err() {
  let s = Array1::from(vec![100.0, 101.0, 102.0, 101.5]);
  let v = Array1::from(vec![0.04, f64::INFINITY, 0.05, 0.045]);
  for fit in [fit_heston_nmle, fit_heston_pmle] {
    assert_eq!(
      fit(s.view(), v.view(), 0.01).unwrap_err(),
      StatsError::InvalidObservation {
        index: 1,
        value: f64::INFINITY,
        constraint: "finite and > 0"
      }
    );
    assert_eq!(
      fit(s.view(), Array1::from_elem(4, 0.04).view(), 0.01).unwrap_err(),
      StatsError::DegenerateData("constant variance series")
    );
  }
}

#[test]
#[should_panic(expected = "delta must satisfy `delta.is_finite() && delta > 0.0`, got delta = 0.0")]
fn nmle_heston_with_delta_panics_on_non_positive_step() {
  let s = Array1::from(vec![100.0, 101.0]);
  let v = Array1::from(vec![0.04, 0.05]);
  let _ = fit_heston_nmle_with_delta(s.view(), v.view(), 0.01, 0.0);
}

#[test]
#[should_panic(expected = "delta must satisfy `delta.is_finite() && delta > 0.0`, got delta = 0.0")]
fn pmle_heston_with_delta_panics_on_non_positive_step() {
  let s = Array1::from(vec![100.0, 101.0]);
  let v = Array1::from(vec![0.04, 0.05]);
  let _ = fit_heston_pmle_with_delta(s.view(), v.view(), 0.01, 0.0);
}

#[test]
/// Parameter recovery has a sampling distribution of its own, so a single
/// path is a coin flip against these tolerances. Three pinned seeds with an
/// any() over them — the best-of pattern from `integration-test-writing`
/// SKILL section 1.2.
fn nmle_heston_recovers_reasonable_params_from_synthetic_path() {
  // A daily step over ~16 years. The `fit_heston_nmle` default of delta =
  // 1/(n-1) puts the whole path inside one year, where kappa and theta are
  // barely identifiable at all — two of three seeds drove the optimiser to
  // a boundary. Volatility of variance and correlation are estimable
  // either way; mean reversion needs the horizon.
  let n = 4096usize;
  let dt = 1.0_f64 / 252.0;
  let sqrt_dt = dt.sqrt();

  let r = 0.01;
  let kappa_true = 1.2;
  let theta_true = 0.09;
  let sigma_true = 0.20;
  let rho_true = -0.35;
  let beta1_true = (-kappa_true * dt).exp();
  let beta3_true = sigma_true * sigma_true * (1.0 - beta1_true * beta1_true) / (2.0 * kappa_true);

  let run = |seed: u64| {
    let mut s = Array1::<f64>::zeros(n);
    let mut v = Array1::<f64>::zeros(n);
    s[0] = 100.0;
    v[0] = 0.09;

    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));

    for i in 1..n {
      let z1: f64 = normal.sample();
      let z2: f64 = normal.sample();

      let v_prev = v[i - 1].max(1e-10);
      let v_next =
        beta1_true * v_prev + theta_true * (1.0 - beta1_true) + (beta3_true * v_prev).sqrt() * z2;
      v[i] = v_next.max(1e-10);

      let dw2 =
        (v[i] - v_prev - kappa_true * (theta_true - v_prev) * dt) / (sigma_true * v_prev.sqrt());
      let dw1 = rho_true * dw2 + (1.0 - rho_true * rho_true).sqrt() * sqrt_dt * z1;

      let log_s_next = s[i - 1].ln() + (r - 0.5 * v_prev) * dt + v_prev.sqrt() * dw1;
      s[i] = log_s_next.exp();
    }

    fit_heston_nmle_with_delta(s.view(), v.view(), r, dt).unwrap()
  };

  let fits = [7u64, 19, 41].map(|seed| (seed, run(seed)));

  for (seed, est) in fits.iter() {
    assert!(
      est.kappa.is_finite() && est.kappa > 0.0,
      "seed {seed}: kappa must be finite and positive, got {}",
      est.kappa
    );
    assert!(
      est.theta.is_finite() && est.theta > 0.0,
      "seed {seed}: theta must be finite and positive, got {}",
      est.theta
    );
    assert!(
      est.sigma.is_finite() && est.sigma > 0.0,
      "seed {seed}: sigma must be finite and positive, got {}",
      est.sigma
    );
    assert!(est.rho.is_finite(), "seed {seed}: rho must be finite");
  }

  let ok = fits.iter().any(|(_, est)| {
    (est.kappa - kappa_true).abs() < 4.0
      && (est.theta - theta_true).abs() < 0.03
      && (est.sigma - sigma_true).abs() < 0.10
      && (est.rho - rho_true).abs() < 0.12
  });
  let report = fits
    .iter()
    .map(|(s, e)| {
      format!(
        "seed {s}: κ={:.4} θ={:.4} σ={:.4} ρ={:.4}",
        e.kappa, e.theta, e.sigma, e.rho
      )
    })
    .collect::<Vec<_>>()
    .join("; ");
  assert!(ok, "no seed recovered the Heston parameters — {report}");
}

#[test]
fn pmle_heston_returns_finite_params_from_synthetic_path() {
  let n = 2048usize;
  let m = n - 1;
  let dt = 1.0 / m as f64;

  let r = 0.01;
  let kappa_true = 1.2;
  let theta_true = 0.08;
  let sigma_true = 0.2;
  let rho_true = -0.4;

  let mut s = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);
  s[0] = 100.0;
  v[0] = theta_true;

  let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(17));

  for i in 1..n {
    let z1: f64 = normal.sample();
    let z2: f64 = normal.sample();

    let dw1 = dt.sqrt() * z1;
    let dw2 = dt.sqrt() * (rho_true * z1 + (1.0 - rho_true * rho_true).sqrt() * z2);

    let v_prev = v[i - 1].max(1e-10);
    let v_next =
      v_prev + kappa_true * (theta_true - v_prev) * dt + sigma_true * v_prev.sqrt() * dw2;
    v[i] = v_next.max(1e-10);

    let log_s_next = s[i - 1].ln() + (r - 0.5 * v_prev) * dt + v_prev.sqrt() * dw1;
    s[i] = log_s_next.exp();
  }

  let est = fit_heston_pmle(s.view(), v.view(), r).unwrap();
  assert!(est.kappa.is_finite() && est.kappa > 0.0);
  assert!(est.theta.is_finite() && est.theta > 0.0);
  assert!(est.sigma.is_finite() && est.sigma > 0.0);
  assert!(est.rho.is_finite() && est.rho.abs() <= 1.0);
}

#[test]
fn a_non_finite_r_panics_in_the_predicate_form() {
  let s = wiggly_prices(5);
  let v = Array1::from(vec![0.04, 0.05, 0.045, 0.042, 0.047]);
  let fits: [Fit; 4] = [
    |s, v| fit_heston_nmle(s, v, f64::NAN),
    |s, v| fit_heston_nmle_with_delta(s, v, f64::NAN, 1.0 / 252.0),
    |s, v| fit_heston_pmle(s, v, f64::NAN),
    |s, v| fit_heston_pmle_with_delta(s, v, f64::NAN, 1.0 / 252.0),
  ];
  for fit in fits {
    let payload = std::panic::catch_unwind(|| {
      let _ = fit(s.view(), v.view());
    })
    .expect_err("the fit accepted a NaN r");
    let text = payload
      .downcast_ref::<String>()
      .cloned()
      .expect("a text payload");
    assert_eq!(text, "r must satisfy `r.is_finite()`, got r = NaN");
  }
}

/// `κ̂` and `θ̂ / c` do not depend on the units of the variance.
#[test]
fn every_fit_is_invariant_to_the_units_of_the_variance() {
  let (n, dt) = (400, 1.0 / 252.0);
  let mut noise = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(11));
  let mut v = Array1::<f64>::zeros(n);
  v[0] = 0.04;
  for t in 1..n {
    let step = 30.0 * (0.04 - v[t - 1]) * dt + 0.6 * (v[t - 1] * dt).sqrt() * noise.sample();
    v[t] = (v[t - 1] + step).abs();
  }
  let s =
    Array1::from_iter((0..n).map(|t| 100.0 + 2.0 * (t as f64 * 0.37).sin() + 0.01 * t as f64));
  for fit in fits() {
    let base = fit(s.view(), v.view()).unwrap();
    for c in [1e12, 1e14] {
      let scaled = fit(s.view(), (&v * c).view()).unwrap();
      assert!((scaled.kappa / base.kappa - 1.0).abs() < 1e-9, "c = {c}");
      assert!(
        (scaled.theta / c / base.theta - 1.0).abs() < 1e-9,
        "c = {c}"
      );
    }
  }
}
