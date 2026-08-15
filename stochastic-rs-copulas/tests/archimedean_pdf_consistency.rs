//! Permanent regression guard: `pdf` must equal the mixed second partial
//! of `cdf` for every Archimedean family, at several θ across its valid
//! range (including negative θ where the family allows it).
//!
//! This is the check that caught the bug this file's sibling fixes exist
//! for: `Frank::pdf`/`Frank::partial_derivative` built a denominator that
//! was wrong for every non-zero θ (up to 268x off at θ=5), and nothing in
//! the existing test suite, an API review, or a documentation pass ever
//! caught it, because nothing ever checked `pdf` against `cdf` directly.
//! A copula density is *defined* as the mixed second partial of its CDF,
//! so a finite difference of `cdf` is an independent probe that trusts
//! neither method's algebra.
//!
//! The finite difference is Richardson-extrapolated (central mixed
//! difference at step `h` and `h/2`, combined to cancel the leading
//! `O(h^2)` truncation term), pushing the residual truncation error to
//! `O(h^4)`. At `h = 1e-4`, an independent reference implementation of
//! every case below puts the worst observed analytic-vs-finite-
//! difference relative error at about `3e-6` — so the `1e-3` relative
//! tolerance used here carries roughly three orders of magnitude of
//! margin above the actual numerical noise floor, while still being two
//! orders of magnitude tighter than the smallest bug this check class
//! was written to catch (36% at Frank's θ=-3).

use ndarray::array;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::traits::BivariateExt;

const STEP: f64 = 1e-4;
const REL_TOL: f64 = 1e-3;
const POINTS: [(f64, f64); 3] = [(0.3, 0.7), (0.5, 0.5), (0.4, 0.6)];

/// Central mixed second difference of `c.cdf` at `(u, v)`, Richardson-
/// extrapolated from step `h` and `h/2` to cancel the leading `O(h^2)`
/// truncation term.
fn fd_mixed_partial(c: &impl BivariateExt, u: f64, v: f64, h: f64) -> f64 {
  let central = |step: f64| {
    let eval = |du: f64, dv: f64| c.cdf(&array![[u + du, v + dv]]).unwrap()[0];
    (eval(step, step) - eval(step, -step) - eval(-step, step) + eval(-step, -step))
      / (4.0 * step * step)
  };
  let d_h = central(h);
  let d_half = central(h / 2.0);
  (4.0 * d_half - d_h) / 3.0
}

fn assert_pdf_matches_fd(c: &impl BivariateExt, u: f64, v: f64, label: &str) {
  let analytic = c.pdf(&array![[u, v]]).unwrap()[0];
  let fd = fd_mixed_partial(c, u, v, STEP);
  let rel_err = (analytic - fd).abs() / fd.abs().max(1e-12);
  assert!(
    rel_err < REL_TOL,
    "{label}: pdf={analytic}, fd(cdf)={fd}, rel_err={rel_err}"
  );
}

/// θ ∈ [0, ∞), including the exact `θ=0` independence boundary: `pdf`
/// and `cdf` now both special-case it directly (`c(u,v)=1`, `C(u,v)=uv`),
/// so it is no longer a disjoint code path that this general sweep has
/// to avoid — see `clayton.rs`'s own `θ=0` exact-value tests for the
/// closed-form assertions this finite-difference check complements.
#[test]
fn clayton_pdf_matches_finite_difference_cdf() {
  for &theta in &[0.0_f64, 0.3, 1.0, 2.5, 5.0] {
    let mut c = Clayton::new();
    c.set_theta(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&c, u, v, &format!("Clayton θ={theta} (u,v)=({u},{v})"));
    }
  }
}

/// θ ∈ {0.5, 2, 5, -3} — the exact values (and the `(0.4, 0.6)` point)
/// from the originally-reported measurement, where `pdf` was 83%-268%
/// off. Frank is the only one of the four families whose in-repo bounds
/// admit negative θ.
#[test]
fn frank_pdf_matches_finite_difference_cdf() {
  for &theta in &[0.5_f64, 2.0, 5.0, -3.0] {
    let f = Frank::new(Some(theta), None);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&f, u, v, &format!("Frank θ={theta} (u,v)=({u},{v})"));
    }
  }
}

/// θ ∈ (1, ∞), avoiding the exact `θ=1` boundary (its own dedicated
/// independence tests live in `gumbel.rs`).
#[test]
fn gumbel_pdf_matches_finite_difference_cdf() {
  for &theta in &[1.3_f64, 2.0, 4.0, 8.0] {
    let g = Gumbel::new(Some(theta), None);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&g, u, v, &format!("Gumbel θ={theta} (u,v)=({u},{v})"));
    }
  }
}

/// θ ∈ (1, ∞). Joe has no reported bug — the "control" family showing
/// this check doesn't just rubber-stamp everything it touches.
#[test]
fn joe_pdf_matches_finite_difference_cdf() {
  let mut j = Joe::new();
  for &theta in &[1.3_f64, 2.0, 4.0, 8.0] {
    j.set_theta(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&j, u, v, &format!("Joe θ={theta} (u,v)=({u},{v})"));
    }
  }
}

/// θ ∈ (-1, 1), including `θ=0`: unlike Clayton's `(u^{-θ}+v^{-θ}-1)^{-1/θ}`,
/// AMH's `D = 1-θ(1-u)(1-v)` never vanishes at `θ=0`, so `pdf`/`cdf` were
/// already correct there without a dedicated branch — confirmed by this
/// same sweep rather than assumed. This test targets `pdf` against `cdf`
/// only; it does not exercise `partial_derivative`, whose separate
/// wrong-argument bug (fixed alongside this test) is instead pinned by
/// `amh.rs`'s own `amh_partial_derivative_at_independence_returns_u` and
/// `amh_sample_matches_closed_form_cdf_at_off_diagonal_points`.
#[test]
fn amh_pdf_matches_finite_difference_cdf() {
  let mut a = Amh::new();
  for &theta in &[-0.9_f64, -0.3, 0.0, 0.5, 0.9] {
    a.set_theta(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&a, u, v, &format!("Amh θ={theta} (u,v)=({u},{v})"));
    }
  }
}
