//! `pdf` equals the mixed second partial of `cdf` for every Archimedean family, across its θ range.
//! The probe is a Richardson-extrapolated finite difference (steps `h`, `h/2`; error `O(h^4)`).

use ndarray::array;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::traits::BivariateExt;

const STEP: f64 = 1e-4;
/// The probe's worst error at `STEP` is ≈ 3e-6, so `1e-3` clears its truncation and still catches a density off by 0.1 %.
const REL_TOL: f64 = 1e-3;
const POINTS: [(f64, f64); 3] = [(0.3, 0.7), (0.5, 0.5), (0.4, 0.6)];

/// Central mixed second difference of `c.cdf` at `(u, v)`, Richardson-extrapolated from steps `h`,
/// `h/2` to cancel the leading `O(h^2)` term.
fn fd_mixed_partial(c: &impl BivariateExt, u: f64, v: f64, h: f64) -> f64 {
  let central = |step: f64| {
    let eval = |du: f64, dv: f64| c.cdf(array![[u + du, v + dv]].view()).unwrap()[0];
    (eval(step, step) - eval(step, -step) - eval(-step, step) + eval(-step, -step))
      / (4.0 * step * step)
  };
  let d_h = central(h);
  let d_half = central(h / 2.0);
  (4.0 * d_half - d_h) / 3.0
}

fn assert_pdf_matches_fd(c: &impl BivariateExt, u: f64, v: f64, label: &str) {
  let analytic = c.pdf(array![[u, v]].view()).unwrap()[0];
  let fd = fd_mixed_partial(c, u, v, STEP);
  let rel_err = (analytic - fd).abs() / fd.abs().max(1e-12);
  assert!(
    rel_err < REL_TOL,
    "{label}: pdf={analytic}, fd(cdf)={fd}, rel_err={rel_err}"
  );
}

/// θ ∈ [0, ∞), the exact `θ=0` independence boundary included, where `pdf` and `cdf` branch to
/// `c(u,v)=1` and `C(u,v)=uv`.
#[test]
fn clayton_pdf_matches_finite_difference_cdf() {
  for &theta in &[0.0_f64, 0.3, 1.0, 2.5, 5.0] {
    let mut c = Clayton::default();
    c.set_theta(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&c, u, v, &format!("Clayton θ={theta} (u,v)=({u},{v})"));
    }
  }
}

/// θ ∈ {0.5, 2, 5, -3}: Frank's bounds admit both signs.
#[test]
fn frank_pdf_matches_finite_difference_cdf() {
  for &theta in &[0.5_f64, 2.0, 5.0, -3.0] {
    let f = Frank::new(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&f, u, v, &format!("Frank θ={theta} (u,v)=({u},{v})"));
    }
  }
}

/// θ ∈ (1, ∞), off the exact `θ=1` boundary, which `gumbel.rs`'s own tests cover.
#[test]
fn gumbel_pdf_matches_finite_difference_cdf() {
  for &theta in &[1.3_f64, 2.0, 4.0, 8.0] {
    let g = Gumbel::new(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&g, u, v, &format!("Gumbel θ={theta} (u,v)=({u},{v})"));
    }
  }
}

#[test]
fn joe_pdf_matches_finite_difference_cdf() {
  let mut j = Joe::default();
  for &theta in &[1.3_f64, 2.0, 4.0, 8.0] {
    j.set_theta(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&j, u, v, &format!("Joe θ={theta} (u,v)=({u},{v})"));
    }
  }
}

/// θ ∈ (-1, 1), `θ=0` included, where AMH's denominator `1-θ(1-u)(1-v)` never vanishes;
/// `partial_derivative` is pinned in `amh.rs`'s own tests.
#[test]
fn amh_pdf_matches_finite_difference_cdf() {
  let mut a = Amh::default();
  for &theta in &[-0.9_f64, -0.3, 0.0, 0.5, 0.9] {
    a.set_theta(theta);
    for &(u, v) in &POINTS {
      assert_pdf_matches_fd(&a, u, v, &format!("Amh θ={theta} (u,v)=({u},{v})"));
    }
  }
}
