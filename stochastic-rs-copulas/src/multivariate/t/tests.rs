use ndarray::array;

use super::*;

/// Construct a 3-dim t-copula with mild dependence and verify the
/// sampler produces uniforms on every coordinate.
#[test]
fn t_copula_samples_have_uniform_marginals() {
  let corr = array![[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]];
  let cop = TMultivariate::new_with(corr, 5.0).expect("valid corr");
  let u = cop.sample(20_000).expect("sample");
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
  let t_cop = TMultivariate::new_with(corr.clone(), 200.0).unwrap();
  let g_cop = super::super::gaussian::GaussianMultivariate::new_with_corr(corr).unwrap();
  let queries = array![[0.25, 0.75], [0.5, 0.5], [0.1, 0.9], [0.8, 0.3],];
  let t_pdf = t_cop.pdf(&queries).unwrap();
  let g_pdf = g_cop.pdf(&queries).unwrap();
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
  let cop = TMultivariate::new_with(corr.clone(), nu).unwrap();
  let q = array![[0.5, 0.5]];
  let pdf = cop.pdf(&q).unwrap()[0];
  // Analytic value: at z=0 the multivariate kernel = 1, marginal kernel = 1.
  //   c(0.5, 0.5) = f_{Σ,ν}(0,0) / [f_ν(0)]^2
  //   f_{Σ,ν}(0,0) = Γ((ν+2)/2) / [Γ(ν/2) · ν · π · √|Σ|]
  //   f_ν(0)       = Γ((ν+1)/2) / [Γ(ν/2) · √(ν · π)]
  let det: f64 = 1.0 - 0.3 * 0.3;
  let f_mv = (ln_gamma(0.5 * (nu + 2.0))
    - ln_gamma(0.5 * nu)
    - (nu * f64::consts::PI).ln()
    - 0.5 * det.ln())
  .exp();
  let f_marg =
    (ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu * f64::consts::PI).ln()).exp();
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
  let cop = TMultivariate::new_with(true_corr.clone(), 5.0).unwrap();
  let u = cop.sample(5_000).unwrap();
  let mut fitted = TMultivariate::new();
  fitted.fit(u).unwrap();
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

/// Log-pdf = ln(pdf) for all valid query points (no separate log path
/// divergence).
#[test]
fn t_copula_log_pdf_matches_ln_pdf() {
  let corr = array![[1.0, 0.4], [0.4, 1.0]];
  let cop = TMultivariate::new_with(corr, 6.0).unwrap();
  let q = array![[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]];
  let pdf = cop.pdf(&q).unwrap();
  let lp = cop.log_pdf(&q).unwrap();
  for i in 0..pdf.len() {
    assert!(
      (lp[i] - pdf[i].ln()).abs() < 1e-12,
      "log_pdf[{i}]={} vs ln(pdf[{i}])={}",
      lp[i],
      pdf[i].ln()
    );
  }
}

/// CDF in $\mathbb{R}^2$ should approximately match the bivariate
/// Dunnett-Sobel reference implemented in [`crate::bivariate::t_copula`].
/// MC noise on 4000 samples ≈ 1/√4000 ≈ 0.016; allow 0.04 tolerance.
#[test]
fn t_copula_cdf_matches_bivariate_reference() {
  use crate::bivariate::t_copula::TCopula;
  use crate::traits::BivariateExt;
  let rho = 0.4;
  let nu = 6.0;
  let corr = array![[1.0, rho], [rho, 1.0]];
  let mv = TMultivariate::new_with(corr, nu).unwrap();
  let mut bv = TCopula::with_nu(nu);
  bv.set_theta(rho);
  let q = array![[0.3, 0.6], [0.5, 0.5], [0.8, 0.2]];
  let cdf_mv = mv.cdf(&q).unwrap();
  let cdf_bv = bv.cdf(&q).unwrap();
  for i in 0..cdf_mv.len() {
    assert!(
      (cdf_mv[i] - cdf_bv[i]).abs() < 0.04,
      "MV-CDF[{i}]={} vs BV-CDF[{i}]={}",
      cdf_mv[i],
      cdf_bv[i]
    );
  }
}

/// Manual ν override must be respected by sampling and pdf paths.
#[test]
fn t_copula_manual_nu_override() {
  let corr = array![[1.0, 0.3], [0.3, 1.0]];
  let mut cop = TMultivariate::new_with(corr, 4.0).unwrap();
  cop.set_nu(12.0).unwrap();
  assert_eq!(cop.nu(), 12.0);
  let _ = cop.sample(100).unwrap();
  let _ = cop.pdf(&array![[0.5, 0.5]]).unwrap();
  let bad = cop.set_nu(0.0);
  assert!(bad.is_err(), "ν=0 must be rejected");
}

/// `nu()`/`set_nu()` mirror `TCopula`'s naming and validation contract
/// exactly, including the byte-identical error string on invalid input.
#[test]
fn t_multivariate_exposes_nu() {
  let corr = array![[1.0, 0.3], [0.3, 1.0]];
  let mut cop = TMultivariate::new_with(corr, 4.0).unwrap();
  assert_eq!(cop.nu(), 4.0);
  assert!(cop.set_nu(12.0).is_ok());
  assert_eq!(cop.nu(), 12.0);
  let err = cop.set_nu(0.0).unwrap_err();
  assert_eq!(err.to_string(), "Degrees of freedom must be positive");
  assert_eq!(cop.nu(), 12.0, "a failed set_nu must not mutate the field");
}
