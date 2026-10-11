use ndarray::array;
use stochastic_rs_distributions::special::ndtri;

use super::*;
use crate::bivariate::gaussian::GaussianCopula;
use crate::multivariate::t::TMultivariate;

fn approx(a: f64, b: f64, tol: f64) -> bool {
  (a - b).abs() <= tol
}

#[test]
fn t_cdf_marginal_recovers_input() {
  let c = TCopula::new(0.5, 4.0);
  let x = array![[0.4_f64, 1.0], [1.0, 0.7]];
  let cdf = c.cdf(x.view()).unwrap();
  assert!(approx(cdf[0], 0.4, 1e-6));
  assert!(approx(cdf[1], 0.7, 1e-6));
}

#[test]
fn t_cdf_zero_rho_at_origin_is_one_quarter() {
  // For bivariate Student-t with ρ = 0, the components are uncorrelated
  // but **not independent** (they share a common χ²_ν mixing scale).
  // Yet at the marginal medians (0, 0), the sign-decoupling representation
  //   (X, Y) = (Z₁, Z₂) / √(W/ν), Z ⊥ W,
  // gives sign(X) = sign(Z₁), sign(Y) = sign(Z₂), with Z₁ ⊥ Z₂. Hence
  //   P(X ≤ 0, Y ≤ 0) = P(Z₁ ≤ 0) · P(Z₂ ≤ 0) = 1/4 exactly.
  let c = TCopula::new(0.0, 4.0);
  let x = array![[0.5_f64, 0.5]];
  let cdf = c.cdf(x.view()).unwrap();
  assert!(approx(cdf[0], 0.25, 1e-6), "got {}", cdf[0]);
}

#[test]
fn t_theta_from_tau_matches_sin_formula() {
  let c = TCopula::default();
  let expected = (0.5_f64 * std::f64::consts::PI * 0.25).sin();
  assert!(approx(c.theta_from_tau(0.25).unwrap(), expected, 1e-12));
}

#[test]
fn t_pdf_symmetric_in_uv() {
  let c = TCopula::new(0.3, 5.0);
  let x_ab = array![[0.3_f64, 0.7]];
  let x_ba = array![[0.7_f64, 0.3]];
  let pdf_ab = c.pdf(x_ab.view()).unwrap();
  let pdf_ba = c.pdf(x_ba.view()).unwrap();
  assert!(approx(pdf_ab[0], pdf_ba[0], 1e-9));
}

#[test]
fn t_pdf_large_nu_approaches_gaussian_copula() {
  // At ν → ∞ the t-copula collapses to the Gaussian copula.
  // Spot-check density vs Gaussian-copula density at a fixed (u, v, ρ)
  // — use ν = 500.
  let c = TCopula::new(0.5, 500.0);
  let x = array![[0.3_f64, 0.7]];
  let pdf_t = c.pdf(x.view()).unwrap();

  // Gaussian copula density at (0.3, 0.7), ρ=0.5:
  // c(u,v) = (1/√(1-ρ²)) · exp{(2ρ x y - ρ²(x²+y²))/(2(1-ρ²))}
  // with x = Φ⁻¹(u), y = Φ⁻¹(v).
  let rho = 0.5_f64;
  let xx = ndtri(0.3);
  let yy = ndtri(0.7);
  let r2 = rho * rho;
  let factor = (2.0 * rho * xx * yy - r2 * (xx * xx + yy * yy)) / (2.0 * (1.0 - r2));
  let pdf_gauss = factor.exp() / (1.0 - r2).sqrt();
  assert!(
    (pdf_t[0] - pdf_gauss).abs() < 0.01,
    "pdf_t={}, pdf_gauss={}",
    pdf_t[0],
    pdf_gauss
  );
}

#[test]
fn t_cdf_symmetry_in_rho() {
  // C_{ρ,ν}(u, v) - u·v   should be an odd function of ρ at u = v = 1/2.
  // Test: C_{-ρ}(0.5, 0.5) + C_{ρ}(0.5, 0.5) = 2·C_{0}(0.5, 0.5).
  let c_pos = TCopula::new(0.4, 4.0);
  let c_neg = TCopula::new(-0.4, 4.0);
  let c_zero = TCopula::new(0.0, 4.0);
  let pt = array![[0.5_f64, 0.5]];
  let lhs = c_pos.cdf(pt.view()).unwrap()[0] + c_neg.cdf(pt.view()).unwrap()[0];
  let rhs = 2.0 * c_zero.cdf(pt.view()).unwrap()[0];
  assert!(approx(lhs, rhs, 1e-6), "symmetry: lhs = {lhs}, rhs = {rhs}");
}

#[test]
fn t_partial_derivative_matches_finite_diff() {
  let c = TCopula::new(0.4, 4.0);
  let u = 0.3_f64;
  let v = 0.6_f64;
  let h = 1e-4_f64;
  let pd = c.partial_derivative(array![[u, v]].view()).unwrap()[0];
  let cdf_hi = c.cdf(array![[u, v + h]].view()).unwrap()[0];
  let cdf_lo = c.cdf(array![[u, v - h]].view()).unwrap()[0];
  let fd = (cdf_hi - cdf_lo) / (2.0 * h);
  assert!(
    approx(pd, fd, 1e-3),
    "analytic ∂C/∂v = {pd}, finite-diff = {fd}"
  );
}

/// ρ=0.5, ν=3: λ = 2·t₄(−√(4·(1−0.5)/(1+0.5))), checked against the
/// in-tree `t_cdf` (regularised incomplete beta, A&S 26.7.1), plus
/// monotonicity of λ in ρ.
#[test]
fn t_copula_tail_dependence_symmetric_and_monotone() {
  let c = TCopula::new(0.5, 3.0);
  let td = c.tail_dependence().unwrap();
  assert!(approx(td.lower, td.upper, 1e-15), "symmetric: {td:?}");

  let arg = -((4.0_f64 * (1.0 - 0.5) / (1.0 + 0.5)).sqrt());
  let expected = 2.0 * TCopula::t_cdf(arg, 4.0);
  assert!(
    approx(td.lower, expected, 1e-12),
    "λ={}, expected {expected}",
    td.lower
  );

  let mut prev = 0.0_f64;
  for &rho in &[-0.5_f64, -0.1, 0.3, 0.7, 0.9] {
    let cc = TCopula::new(rho, 3.0);
    let lam = cc.tail_dependence().unwrap().lower;
    assert!(
      lam > prev,
      "λ should increase with ρ: ρ={rho}, λ={lam}, prev={prev}"
    );
    prev = lam;
  }
}

/// `partial_derivative` must not jump to a hardcoded constant right at
/// the `v` boundary: its value at `v=1e-12` should be continuous with
/// its value at `v=1e-9` (the F8 regression this guards), and — since
/// the true boundary limit is `u`-independent for this family (see the
/// method's doc) — should also agree across different `u` at the same
/// extreme `v`.
///
/// Tolerance `1e-3`, not `1e-6`: the raw formula's own convergence to its
/// `v\to0^+` asymptote is only `O(1/y)` in `y=t_\nu^{-1}(v)`, and `y`
/// itself grows only polynomially as `v\to0` (`t`-distribution tail), so
/// even a `1000\times` shrink in `v` (`1e-9\to1e-12`) leaves a real,
/// measured `\approx 2.7e-4` gap at `\rho=0.6,\nu=4` — verified directly
/// against this formula in Python, not assumed. `1e-3` still rejects a
/// jump to a hardcoded constant by a wide margin (such a jump is
/// `O(0.1)`-`O(1)`, three-plus orders of magnitude larger).
#[test]
fn t_copula_partial_derivative_no_jump_near_v_boundary() {
  let c = TCopula::new(0.6, 4.0);
  let near_zero_a = c
    .partial_derivative(array![[0.4_f64, 1e-12]].view())
    .unwrap()[0];
  let near_zero_b = c
    .partial_derivative(array![[0.4_f64, 1e-9]].view())
    .unwrap()[0];
  assert!(
    approx(near_zero_a, near_zero_b, 1e-3),
    "expected continuity near v=0: v=1e-12 -> {near_zero_a}, v=1e-9 -> {near_zero_b}"
  );
  let near_one_a = c
    .partial_derivative(array![[0.4_f64, 1.0 - 1e-12]].view())
    .unwrap()[0];
  let near_one_b = c
    .partial_derivative(array![[0.4_f64, 1.0 - 1e-9]].view())
    .unwrap()[0];
  assert!(
    approx(near_one_a, near_one_b, 1e-3),
    "expected continuity near v=1: v=1-1e-12 -> {near_one_a}, v=1-1e-9 -> {near_one_b}"
  );

  // u-independence at the exact boundary (the closed form derived in the
  // method's doc does not depend on u), unlike the pre-fix hardcoded
  // 0.0/1.0 which happened to be u-independent for the wrong reason.
  let other_u = c.partial_derivative(array![[0.8_f64, 0.0]].view()).unwrap()[0];
  let base_u = c.partial_derivative(array![[0.4_f64, 0.0]].view()).unwrap()[0];
  assert!(
    approx(other_u, base_u, 1e-12),
    "boundary limit should not depend on u: u=0.8 -> {other_u}, u=0.4 -> {base_u}"
  );

  // Directional regression check against the exact closed form
  // `T_{nu+1}(rho*sqrt(nu+1)/sqrt(1-rho^2))`, rho=0.6, nu=4: neither 0
  // nor 1 (the pre-fix hardcoded values), and rho>0 pushes it above 0.5.
  assert!(
    base_u > 0.5 && base_u < 1.0,
    "rho>0 boundary limit should sit strictly between 0.5 and 1, got {base_u}"
  );
}

/// `nu` is private: `nu()` reads it and `set_nu` writes a valid value.
#[test]
fn tcopula_nu_validated() {
  let mut c = TCopula::default();
  assert_eq!(c.nu(), 4.0);
  c.set_nu(6.0);
  assert_eq!(c.nu(), 6.0);
}

/// The smallest subnormal passes `nu > 0` but halves to 0, which `SimdStudentT::new` and `SimdChiSquared::new` assert
/// against: both t-copulas reject it up front, so `pdf`, `cdf`, `partial_derivative`, `percent_point` and `sample` cannot panic.
#[test]
fn both_t_copulas_reject_the_nu_their_laws_would_panic_on() {
  let want = Err(CopulaError::InvalidParameter {
    name: "nu",
    value: 5e-324,
    constraint: "nu / 2 > 0".into(),
  });
  let corr = array![[1.0, 0.3], [0.3, 1.0]];
  assert_eq!(TMultivariate::try_new(corr, 5e-324).map(|_| ()), want);
  assert_eq!(TCopula::try_new(0.5, 5e-324).map(|_| ()), want);
  // The next subnormal halves to a positive value.
  let c = TCopula::new(0.5, 1e-323);
  let uv = array![[0.3, 0.7]];
  assert!(
    c.pdf(uv.view()).is_ok() && c.cdf(uv.view()).is_ok() && c.partial_derivative(uv.view()).is_ok()
  );
  assert!(
    c.percent_point(array![0.3].view(), array![0.7].view())
      .is_ok()
  );
}

/// `set_nu` applies the same rule, `nu / 2 > 0` included, as a panic.
#[test]
#[should_panic(expected = "nu must satisfy `nu / 2 > 0`, got nu = 5e-324")]
fn set_nu_panics_on_the_nu_the_laws_would_panic_on() {
  TCopula::default().set_nu(5e-324);
}

#[test]
#[should_panic(expected = "nu must satisfy `nu / 2 > 0`, got nu = 5e-324")]
fn the_multivariate_set_nu_panics_on_the_nu_the_laws_would_panic_on() {
  TMultivariate::default().set_nu(5e-324);
}

/// `generator` has no override in this family, so this exercises
/// `BivariateExt::generator`'s trait-default body directly.
#[test]
fn t_copula_generator_returns_err_not_archimedean() {
  let c = TCopula::default();
  let t = array![0.5_f64, 0.8];
  assert!(c.generator(t.view()).is_err());
}

/// `ρ = ±1` divides by `√(1 − ρ²)`, so it does not construct, with `GaussianCopula`'s error.
#[test]
fn t_copula_rejects_rho_boundary_like_gaussian() {
  for rho in [1.0_f64, -1.0] {
    let got = TCopula::try_new(rho, 4.0).unwrap_err();
    assert_eq!(got, GaussianCopula::try_new(rho).unwrap_err(), "rho={rho}");
    assert!(matches!(
      got,
      CopulaError::InvalidParameter { name: "theta", .. }
    ));
  }
}

/// No coefficient is ever computed at `ρ = 1`: the setter refuses it.
#[test]
#[should_panic(
  expected = "theta must satisfy `-1.0 <= theta <= 1.0, theta not in [-1.0, 1.0]`, got theta = 1.0"
)]
fn t_copula_set_theta_rejects_the_rho_boundary() {
  TCopula::default().set_theta(1.0);
}
