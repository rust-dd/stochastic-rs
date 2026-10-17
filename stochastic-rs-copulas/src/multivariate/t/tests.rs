use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::bivariate::t_copula::TCopula;
use crate::correlation::tau_matrix_to_corr_matrix;
use crate::multivariate::linalg::condition_number;
use crate::multivariate::linalg::is_spd;
use crate::multivariate::linalg::smallest_eigenvalue;

/// Construct a 3-dim t-copula with mild dependence and verify the
/// sampler produces uniforms on every coordinate.
#[test]
fn t_copula_samples_have_uniform_marginals() {
  let corr = array![[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]];
  let cop = TMultivariate::new(corr, 5.0);
  let u = cop.sample(20_000, &Deterministic::new(7)).expect("sample");
  assert_eq!(u.ncols(), 3);
  assert_eq!(u.nrows(), 20_000);
  for j in 0..3 {
    let col = u.column(j);
    let mean = col.iter().sum::<f64>() / col.len() as f64;
    // Uniform mean = 0.5 with sqrt(1/12N) ≈ 0.002 std on 20k samples;
    // 0.02 tolerance is ~10σ — safe.
    assert!(
      (mean - 0.5).abs() < 0.02,
      "marginal {j} mean = {mean}, expected ~0.5"
    );
    let in_range = col.iter().all(|v| *v > 0.0 && *v < 1.0);
    assert!(in_range, "marginal {j} not strictly in (0,1)");
  }
}

/// In the $\nu \to \infty$ limit the t-copula degenerates to the
/// Gaussian copula. Pick $\nu = 200$ as a practical proxy and verify
/// the multivariate density agrees with the Gaussian copula density on
/// random uniform queries.
#[test]
fn t_copula_large_nu_approaches_gaussian() {
  let corr = array![[1.0, 0.5], [0.5, 1.0]];
  let t_cop = TMultivariate::new(corr.clone(), 200.0);
  let g_cop = super::super::gaussian::GaussianMultivariate::new(corr);
  let queries = array![[0.25, 0.75], [0.5, 0.5], [0.1, 0.9], [0.8, 0.3],];
  let t_pdf = t_cop.pdf(queries.view()).unwrap();
  let g_pdf = g_cop.pdf(queries.view()).unwrap();
  for i in 0..t_pdf.len() {
    assert!(
      (t_pdf[i] - g_pdf[i]).abs() / g_pdf[i].max(1e-10) < 0.02,
      "ν=200 t-pdf[{i}]={} vs Gaussian-pdf[{i}]={}",
      t_pdf[i],
      g_pdf[i]
    );
  }
}

/// At u = (0.5, …, 0.5) the t-copula density reduces to the multivariate
/// kernel at z=0 divided by the marginal kernels at 0. Both reduce to a
/// closed form involving Γ-ratios and |Σ|^{-1/2}.
#[test]
fn t_copula_pdf_at_center() {
  let corr = array![[1.0, 0.3], [0.3, 1.0]];
  let nu = 6.0;
  let cop = TMultivariate::new(corr.clone(), nu);
  let q = array![[0.5, 0.5]];
  let pdf = cop.pdf(q.view()).unwrap()[0];
  // Analytic value: at z=0 the multivariate kernel = 1, marginal kernel = 1.
  //   c(0.5, 0.5) = f_{Σ,ν}(0,0) / [f_ν(0)]^2
  //   f_{Σ,ν}(0,0) = Γ((ν+2)/2) / [Γ(ν/2) · ν · π · √|Σ|]
  //   f_ν(0)       = Γ((ν+1)/2) / [Γ(ν/2) · √(ν · π)]
  let det: f64 = 1.0 - 0.3 * 0.3;
  let f_mv = (ln_gamma(0.5 * (nu + 2.0))
    - ln_gamma(0.5 * nu)
    - (nu * std::f64::consts::PI).ln()
    - 0.5 * det.ln())
  .exp();
  let f_marg =
    (ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu * std::f64::consts::PI).ln())
      .exp();
  let expected = f_mv / (f_marg * f_marg);
  assert!(
    (pdf - expected).abs() / expected < 1e-10,
    "t-pdf at center = {pdf}, expected {expected}"
  );
}

/// Round-trip: sample → recover Σ via Kendall-τ → check entries match
/// the true correlations within MC tolerance. Verifies the fit path.
/// `n = 5000` keeps the test under 1 minute (the ν-profile Brent search
/// scales as n × n_iter × t_quantile_cost); recovery quality is still
/// `|Σ̂ − Σ| < 0.05` on Kendall-τ inversion.
#[test]
fn t_copula_fit_recovers_correlation_from_sample() {
  let true_corr = array![[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]];
  let cop = TMultivariate::new(true_corr.clone(), 5.0);
  let u = cop.sample(5_000, &Deterministic::new(7)).unwrap();
  let mut fitted = TMultivariate::default();
  fitted.fit(u.view()).unwrap();
  let recovered = fitted.correlation().unwrap();
  for i in 0..3 {
    for j in (i + 1)..3 {
      let err = (recovered[[i, j]] - true_corr[[i, j]]).abs();
      assert!(
        err < 0.05,
        "Σ[{i},{j}]: true={}, recovered={} (err={err})",
        true_corr[[i, j]],
        recovered[[i, j]]
      );
    }
  }
  // ν is the harder estimate; profile-likelihood gives ~2-unit error on
  // 5k samples — accept anything in [2.5, 12.0] as recovery of ν=5.
  let nu_hat = fitted.nu();
  assert!(
    (2.5..=12.0).contains(&nu_hat),
    "ν recovered = {nu_hat}, expected ~5"
  );
}

/// ν̂ maximises the fit's own likelihood: scipy's bounded Brent minimiser (`xatol = 1e-10`) on the same objective lands
/// on these values for these two samples.
#[test]
fn t_copula_fit_nu_matches_an_independent_bounded_minimiser() {
  let corr3 = array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]];
  let corr4 = Array2::from_shape_fn((4, 4), |(i, j)| {
    if i == j {
      1.0
    } else {
      0.6_f64.powi((i as i32 - j as i32).abs())
    }
  });
  for (corr, nu, n, seed, want) in [
    (corr3, 4.0, 600, 11, 4.313824466630214),
    (corr4, 7.0, 1000, 5, 6.43192039131383),
  ] {
    let data = TMultivariate::new(corr, nu)
      .sample(n, &Deterministic::new(seed))
      .unwrap();
    let mut fitted = TMultivariate::default();
    fitted.fit(data.view()).unwrap();
    assert!(
      (fitted.nu() - want).abs() < 1e-6,
      "ν̂ = {}, scipy {want}",
      fitted.nu()
    );
  }
}

/// Log-pdf = ln(pdf) for all valid query points (no separate log path
/// divergence).
#[test]
fn t_copula_log_pdf_matches_ln_pdf() {
  let corr = array![[1.0, 0.4], [0.4, 1.0]];
  let cop = TMultivariate::new(corr, 6.0);
  let q = array![[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]];
  let pdf = cop.pdf(q.view()).unwrap();
  let lp = cop.log_pdf(q.view()).unwrap();
  for i in 0..pdf.len() {
    assert!(
      (lp[i] - pdf[i].ln()).abs() < 1e-12,
      "log_pdf[{i}]={} vs ln(pdf[{i}])={}",
      lp[i],
      pdf[i].ln()
    );
  }
}

/// At `d = 2` the multivariate cdf is the bivariate copula's closed form, bit for bit, for an integer and a fractional `ν`.
#[test]
fn t_copula_cdf_matches_bivariate_reference() {
  use crate::traits::BivariateExt;
  let rho = 0.4;
  for nu in [6.0, 6.5] {
    let corr = array![[1.0, rho], [rho, 1.0]];
    let q = array![[0.3, 0.6], [0.5, 0.5], [0.8, 0.2]];
    let cdf_mv = TMultivariate::new(corr, nu).cdf(q.view()).unwrap();
    let cdf_bv = TCopula::new(rho, nu).cdf(q.view()).unwrap();
    assert_eq!(cdf_mv, cdf_bv, "nu={nu}");
  }
}

/// Manual ν override must be respected by sampling and pdf paths.
#[test]
fn t_copula_manual_nu_override() {
  let corr = array![[1.0, 0.3], [0.3, 1.0]];
  let mut cop = TMultivariate::new(corr.clone(), 4.0);
  cop.set_nu(12.0);
  assert_eq!(cop.nu(), 12.0);
  let _ = cop.sample(100, &Deterministic::new(7)).unwrap();
  let _ = cop.pdf(array![[0.5, 0.5]].view()).unwrap();
  let bad = TMultivariate::try_new(corr, 0.0);
  assert!(bad.is_err(), "ν=0 must be rejected");
}

/// `try_new` rejects a `nu` with `TCopula::try_new`'s error, and `set_nu` stores a valid one.
#[test]
fn t_multivariate_exposes_nu() {
  let corr = array![[1.0, 0.3], [0.3, 1.0]];
  let mut cop = TMultivariate::new(corr.clone(), 4.0);
  assert_eq!(cop.nu(), 4.0);
  cop.set_nu(12.0);
  assert_eq!(cop.nu(), 12.0);
  let want = CopulaError::InvalidParameter {
    name: "nu",
    value: 0.0,
    constraint: "0.0 < nu < inf".into(),
  };
  assert_eq!(TMultivariate::try_new(corr, 0.0).unwrap_err(), want);
  assert_eq!(TCopula::try_new(0.3, 0.0).unwrap_err(), want);
}

/// An infinite `nu` is an error from the fallible entry points of either t-copula, so no χ² law is built with `k = ∞`.
#[test]
fn an_infinite_nu_is_rejected_by_name() {
  let want = CopulaError::InvalidParameter {
    name: "nu",
    value: f64::INFINITY,
    constraint: "0.0 < nu < inf".into(),
  };
  let corr = array![[1.0, 0.3], [0.3, 1.0]];
  assert_eq!(
    TMultivariate::try_new(corr, f64::INFINITY).unwrap_err(),
    want
  );
  assert_eq!(TCopula::try_new(0.5, f64::INFINITY).unwrap_err(), want);
}

/// Both setters panic with the same rule.
#[test]
#[should_panic(expected = "nu must satisfy `0.0 < nu < inf`, got nu = inf")]
fn the_bivariate_set_nu_panics_on_an_infinite_nu() {
  TCopula::default().set_nu(f64::INFINITY);
}

#[test]
#[should_panic(expected = "nu must satisfy `0.0 < nu < inf`, got nu = inf")]
fn the_multivariate_set_nu_panics_on_an_infinite_nu() {
  TMultivariate::default().set_nu(f64::INFINITY);
}

/// The documented bisection tolerance and eigenvalue floor of the shrink.
const TOLERANCE: f64 = 1e-6;
const FLOOR: f64 = 1e-3;

/// The `lambda` of `(1 - lambda) R + lambda I` that a fit applied, read off `R[[0, 1]]`, and its analytic minimum
/// `(FLOOR - mu) / (1 - mu)` for a smallest eigenvalue `mu` of `R`.
fn applied_and_minimal_lambda(raw: &Array2<f64>, fitted: &Array2<f64>) -> (f64, f64) {
  let mu = smallest_eigenvalue(raw.view());
  (
    1.0 - fitted[[0, 1]] / raw[[0, 1]],
    (FLOOR - mu) / (1.0 - mu),
  )
}

/// Four shares that sum to a constant give a Kendall estimate that is not positive definite (smallest eigenvalue
/// -4.2e-3); the fit shrinks it by the smallest `lambda` that leaves every eigenvalue at least 1e-3.
#[test]
fn t_copula_fit_shrinks_an_indefinite_estimate_and_keeps_the_unit_diagonal() {
  let ranks = [
    [9, 6, 10, 1],
    [7, 2, 11, 8],
    [8, 10, 6, 3],
    [1, 11, 5, 10],
    [6, 8, 4, 11],
    [10, 4, 8, 6],
    [12, 1, 2, 2],
    [4, 5, 9, 12],
    [5, 12, 1, 7],
    [11, 3, 3, 4],
    [2, 7, 12, 5],
    [3, 9, 7, 9],
  ];
  let u = Array2::from_shape_fn((12, 4), |(i, j)| ranks[i][j] as f64 / 13.0);
  let raw = tau_matrix_to_corr_matrix(kendall_tau(u.view()).unwrap().view());
  assert!(!is_spd(raw.view()));
  let mut cop = TMultivariate::default();
  cop.fit(u.view()).unwrap();
  let fitted = cop.correlation().unwrap();
  let (lambda, want) = applied_and_minimal_lambda(&raw, fitted);
  assert!(
    (want - 1e-12..=want + TOLERANCE + 1e-12).contains(&lambda),
    "{lambda} vs {want}"
  );
  assert!(smallest_eigenvalue(fitted.view()) >= FLOOR - 1e-12);
  assert!(condition_number(fitted.view()) <= 4.0 / FLOOR);
  for i in 0..4 {
    assert_eq!(fitted[[i, i]], 1.0);
    for j in 0..i {
      assert!((fitted[[i, j]] - (1.0 - lambda) * raw[[i, j]]).abs() < 1e-15);
    }
  }
  assert!(TMultivariate::try_new(fitted.clone(), cop.nu()).is_ok());
}

/// Ten observations of six shares give a Kendall estimate with smallest eigenvalue -0.156: the fit shrinks it by
/// `(1e-3 - mu) / (1 - mu)` to the tolerance, so the result keeps its correlations and its eigenvalues stay above 1e-3.
#[test]
fn t_copula_fit_of_a_hopeless_estimate_is_the_minimal_shrink_not_the_identity() {
  let ranks = [
    [3, 10, 10, 2, 4, 2],
    [8, 8, 3, 9, 1, 3],
    [7, 6, 7, 7, 2, 7],
    [9, 5, 2, 5, 8, 1],
    [6, 4, 4, 6, 3, 10],
    [5, 9, 5, 4, 6, 8],
    [2, 2, 9, 8, 5, 6],
    [1, 3, 1, 10, 9, 4],
    [10, 1, 6, 1, 7, 5],
    [4, 7, 8, 3, 10, 9],
  ];
  let u = Array2::from_shape_fn((10, 6), |(i, j)| ranks[i][j] as f64 / 11.0);
  let raw = tau_matrix_to_corr_matrix(kendall_tau(u.view()).unwrap().view());
  assert!((smallest_eigenvalue(raw.view()) + 0.156).abs() < 1e-3);
  let mut cop = TMultivariate::default();
  cop.fit(u.view()).unwrap();
  let fitted = cop.correlation().unwrap();
  let (lambda, want) = applied_and_minimal_lambda(&raw, fitted);
  assert!(
    (want - 1e-12..=want + TOLERANCE + 1e-12).contains(&lambda),
    "{lambda} vs {want}"
  );
  assert!(smallest_eigenvalue(fitted.view()) >= FLOOR - 1e-12);
  assert!(condition_number(fitted.view()) <= 6.0 / FLOOR);
  assert_ne!(fitted, &Array2::<f64>::eye(6));
  for ((i, j), r) in raw.indexed_iter() {
    let expected = if i == j { 1.0 } else { (1.0 - lambda) * r };
    assert!((fitted[[i, j]] - expected).abs() < 1e-15);
  }
  assert!(TMultivariate::try_new(fitted.clone(), cop.nu()).is_ok());
}

/// The fit inverts Kendall's τ-b, which counts ties per margin: `(1,1,2,3)` × `(1,2,2,3)` has τ-b `0.8` where τ-a is `2/3`.
#[test]
fn t_copula_fit_inverts_tau_b_on_tied_data() {
  let u = array![[0.25, 0.25], [0.25, 0.5], [0.5, 0.5], [0.75, 0.75]];
  let mut cop = TMultivariate::default();
  cop.fit(u.view()).unwrap();
  let rho = cop.correlation().unwrap()[[0, 1]];
  assert!(
    (rho - (0.5 * std::f64::consts::PI * 0.8).sin()).abs() < 1e-15,
    "{rho}"
  );
}

/// A constant column leaves its τ-b undefined, which the fit names instead of fitting a zero correlation to it.
#[test]
fn t_copula_fit_rejects_a_constant_column() {
  let u = Array2::from_shape_fn(
    (20, 3),
    |(i, j)| {
      if j == 1 { 0.5 } else { (i + 1) as f64 / 21.0 }
    },
  );
  let got = TMultivariate::default().fit(u.view());
  assert!(
    matches!(&got, Err(CopulaError::InvalidStructure(why)) if why.contains("undefined for column 1:")),
    "{got:?}"
  );
}
