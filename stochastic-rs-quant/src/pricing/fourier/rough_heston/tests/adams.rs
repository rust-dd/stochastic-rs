use num_complex::Complex64;
use stochastic_rs_distributions::special::gamma;

use super::super::LogChf;
use super::super::Riccati;
use super::super::RoughHestonParams;
use super::super::adams::Adams;
use super::super::adams::AdamsCf;
use super::super::adams::Corrector;
use super::super::chf;
use super::super::cumulants;
use super::mittag_leffler;
use super::ratios;
use super::z_gr;
use crate::pricing::fourier::riccati_cd;

const GR: Riccati = Riccati {
  kappa: 1.0,
  nu: 0.4,
  rho: -0.65,
};

/// (27) integrates a constant and a line exactly; powf's ulp of `N^{α+1}` in each of `N` weights
/// leaves a relative `N²ε`, taken four times.
#[test]
fn the_weights_integrate_constants_and_lines_exactly() {
  for alpha in [0.5 + 1e-9, 0.55, 0.75, 1.0] {
    let (n, t) = (200_usize, 1.7_f64);
    let scheme = Adams::new(alpha, n);
    let dt = t / n as f64;
    let tol = 4.0 * (n * n) as f64 * f64::EPSILON;
    let c = dt.powf(alpha) / gamma(alpha + 2.0);
    for k in [0, 1, 7, n - 1] {
      let weight = |j: usize| match j {
        0 => scheme.first[k],
        j if j == k + 1 => 1.0,
        j => scheme.second_difference[k - j],
      };
      let end = dt * (k + 1) as f64;
      let ones = c * (0..=k + 1).map(weight).sum::<f64>();
      let line = c * (0..=k + 1).map(|j| weight(j) * dt * j as f64).sum::<f64>();
      let (want_ones, want_line) = (
        end.powf(alpha) / gamma(alpha + 1.0),
        end.powf(alpha + 1.0) / gamma(alpha + 2.0),
      );
      assert!(
        (ones / want_ones - 1.0).abs() < tol,
        "α = {alpha}, k = {k}: {ones}"
      );
      assert!(
        (line / want_line - 1.0).abs() < tol,
        "α = {alpha}, k = {k}: {line}"
      );
    }
    let r = 1.0 - alpha;
    let grid = (0..=n)
      .map(|j| Complex64::from(dt * j as f64))
      .collect::<Vec<_>>();
    let (_, fractional) = scheme.integrals(t, &grid);
    let want = t.powf(r + 1.0) / gamma(r + 2.0);
    assert!(
      (fractional.re / want - 1.0).abs() < tol,
      "α = {alpha}: {fractional}"
    );
  }
}

/// At `ν = 0` the equation is linear, `h = m₀ t^α E_{α,α+1}(−κt^α)`; 1.2 is under both El Euch &
/// Rosenbaum's `2 − α` and the `1 + α` of Diethelm, Ford & Freed (2004).
#[test]
fn the_linear_equation_converges_to_its_mittag_leffler_solution() {
  let (alpha, t, kappa) = (0.6, 1.5_f64, 1.2);
  let linear = Riccati {
    kappa,
    nu: 0.0,
    rho: 0.0,
  };
  let z = z_gr();
  let y = t.powf(alpha);
  let want = Riccati::moment(z) * y * mittag_leffler(alpha, alpha + 1.0, -kappa * y);
  let error = |n: usize| {
    let path = Adams::new(alpha, n).path(t, z, &linear, Corrector::Solved);
    (path[n] - want).norm()
  };
  let errors = [50, 100, 200, 400, 800].map(error);
  assert!(errors[4] < 1e-5 * want.norm(), "{errors:?}");
  for ratio in ratios(&errors) {
    assert!(ratio > 2.0_f64.powf(1.2), "{errors:?}");
  }
}

/// At `α = 1` (26) is the trapezoid on Heston's Riccati equation, second order.
#[test]
fn at_alpha_one_the_scheme_is_the_trapezoid_on_heston_s_riccati() {
  let (kappa, nu, rho, t) = (1.5, 0.3, -0.7, 1.0);
  let heston = Riccati { kappa, nu, rho };
  for z in [Complex64::new(0.5, 0.5), z_gr(), Complex64::new(0.5, 20.0)] {
    let xi = z / Complex64::i();
    let beta = kappa - rho * nu * z;
    let (_, want) = riccati_cd(beta, xi * (xi + Complex64::i()), 0.0, nu * nu, t);
    let error = |n: usize| {
      let path = Adams::new(1.0, n).path(t, z, &heston, Corrector::Solved);
      (path[n] - want).norm()
    };
    let errors = [100, 200, 400].map(error);
    assert!(errors[2] < 1e-4 * want.norm(), "{z}: {errors:?}");
    for ratio in ratios(&errors) {
      assert!((3.6..4.4).contains(&ratio), "{z}: {errors:?}");
    }
  }
}

/// Where the predictor–corrector is stable (`ν|A|Δ^α ≤ 1`) it is (26)'s solution to the scheme's
/// own order.
#[test]
fn the_literal_predictor_corrector_agrees_with_the_solved_corrector() {
  let (alpha, t, z) = (0.55, 1.0, z_gr());
  let gap = |n: usize| {
    let scheme = Adams::new(alpha, n);
    let solved = scheme.path(t, z, &GR, Corrector::Solved);
    let predicted = scheme.path(t, z, &GR, Corrector::Predicted);
    let gaps = solved.iter().zip(&predicted).map(|(a, b)| (a - b).norm());
    gaps.fold(0.0, f64::max)
  };
  let gaps = [200, 400, 800].map(gap);
  assert!(gaps[2] < 1e-4, "{gaps:?}");
  for ratio in ratios(&gaps) {
    assert!(ratio > 2.0_f64.powf(1.2), "{gaps:?}");
  }
}

/// `D^αy = 1 − μy` at a unit step: the explicit scheme holds at `μΔ^α = 1` and diverges at 2.5,
/// the solved corrector holds at 1000.
/// Reference: Garrappa (2010), Int. J. Comput. Math. 87(10), 2281–2290, DOI 10.1080/00207160802624331
#[test]
fn the_explicit_scheme_has_a_stability_bound_and_the_solved_corrector_none() {
  let n = 600;
  let z = Complex64::from(2.0);
  for alpha in [0.5 + 1e-9, 0.55, 0.75, 1.0] {
    let last = |mu: f64, corrector: Corrector| {
      let test = Riccati {
        kappa: mu,
        nu: 0.0,
        rho: 0.0,
      };
      let path = Adams::new(alpha, n).path(n as f64, z, &test, corrector);
      let worst = path.iter().map(|y| (y - 1.0 / mu).norm() * mu);
      worst.fold(
        0.0,
        |a, b| if b.is_nan() { f64::INFINITY } else { a.max(b) },
      )
    };
    assert!(last(1.0, Corrector::Predicted) <= 2.0, "α = {alpha}");
    assert!(last(2.5, Corrector::Predicted) > 1e6, "α = {alpha}");
    assert!(last(1000.0, Corrector::Solved) <= 1.0 + 1e-9, "α = {alpha}");
  }
}

fn gr_cf(steps: usize) -> AdamsCf {
  AdamsCf::new(
    RoughHestonParams {
      v0: 0.04,
      kappa: 1.0,
      theta: 0.06,
      nu: 0.4,
      rho: -0.65,
      hurst: 0.05,
    },
    steps,
  )
}

#[test]
fn the_adams_chf_is_one_at_the_martingale_points_and_nan_off_its_domain() {
  let cf = gr_cf(64);
  let xi = Complex64::new(3.0, -0.5);
  for t in [0.02, 1.0, 5.0] {
    assert_eq!(chf(&cf, t, Complex64::new(0.0, -1.0)), Complex64::ONE);
    assert_eq!(chf(&cf, t, Complex64::ZERO), Complex64::ONE);
    let mirrored = chf(&cf, t, -xi.conj());
    assert!(
      (mirrored - chf(&cf, t, xi).conj()).norm() < 1e-13,
      "t = {t}"
    );
  }
  assert_eq!(chf(&cf, 0.0, xi), Complex64::ONE);
  for t in [-1.0, f64::NAN, f64::INFINITY] {
    assert!(chf(&cf, t, xi).is_nan(), "t = {t}");
    assert!(cumulants(&cf, t).c1.is_nan(), "t = {t}");
  }
  assert!(chf(&cf, 1.0, Complex64::new(f64::NAN, 0.0)).is_nan());
  assert!(chf(&cf, 1.0, Complex64::new(f64::INFINITY, 0.0)).is_nan());
  let at_expiry = cumulants(&cf, 0.0);
  assert_eq!(
    (at_expiry.c1, at_expiry.c2, at_expiry.c4),
    (0.0, 0.0, Some(0.0))
  );
}

/// `m₀[V₀t + κ(θ − V₀)t^{α+1}E_{α,α+2}(−κt^α)]`: `ln φ` at `ν = 0`, and `−2c₁` at any `ν`.
fn integrated_curve(p: &RoughHestonParams, t: f64) -> f64 {
  let alpha = p.alpha();
  let y = t.powf(alpha);
  let series = mittag_leffler(alpha, alpha + 2.0, -p.kappa * y);
  p.v0 * t + p.kappa * (p.theta - p.v0) * t * y * series
}

#[test]
fn without_vol_of_vol_the_chf_is_the_forward_variance_curve_s() {
  let t = 1.5;
  let z = z_gr();
  let error = |steps: usize| {
    let mut cf = gr_cf(steps);
    cf.params.nu = 0.0;
    cf.riccati.nu = 0.0;
    let want = Riccati::moment(z) * integrated_curve(&cf.params, t);
    (cf.log_chf(t, z) - want).norm() / want.norm()
  };
  let errors = [50, 100, 200, 400, 800].map(error);
  assert!(errors[4] < 1e-5, "{errors:?}");
  for ratio in ratios(&errors) {
    assert!(ratio > 2.0_f64.powf(1.2), "{errors:?}");
  }
}

/// `ln chf` along the real line by central stencils: `cumulant_table`'s first two, and a fourth
/// difference whose own error is `O(h²)`.
fn stencil_cumulants(cf: &impl LogChf, t: f64) -> (f64, f64, f64) {
  let f = |u: f64| cf.log_chf(t, Complex64::new(0.0, u));
  let h = 2e-3;
  let (f2p, fp, f0, fm, f2m) = (f(2.0 * h), f(h), f(0.0), f(-h), f(-2.0 * h));
  let c1 = (-f2p + 8.0 * fp - 8.0 * fm + f2m).im / (12.0 * h);
  let c2 = -(-f2p + 16.0 * fp - 30.0 * f0 + 16.0 * fm - f2m).re / (12.0 * h * h);
  let w = 0.05;
  let c4 = (f(2.0 * w) - 4.0 * f(w) + 6.0 * f(0.0) - 4.0 * f(-w) + f(-2.0 * w)).re / w.powi(4);
  (c1, c2, c4)
}

#[test]
fn the_adams_cumulants_are_its_own_chf_s() {
  let cf = gr_cf(100);
  for t in [0.1, 1.0] {
    let got = cumulants(&cf, t);
    let (c1, c2, c4) = stencil_cumulants(&cf, t);
    assert!(
      (got.c1 - c1).abs() < 1e-8 * got.c2.max(1.0),
      "{t}: {} vs {c1}",
      got.c1
    );
    assert!(
      (got.c2 - c2).abs() < 1e-7 * got.c2.max(1.0),
      "{t}: {} vs {c2}",
      got.c2
    );
    let fourth = got.c4.expect("the jet reaches the fourth order");
    assert!(
      (fourth - c4).abs() < 0.02 * c4.abs(),
      "{t}: {fourth} vs {c4}"
    );
    // `c₁ = −½∫ξ` at any `ν`; 100 steps leave a relative 1.0e-4 at `t = 1`, taken 2.5 times.
    let exact = -0.5 * integrated_curve(&cf.params, t);
    assert!(
      (got.c1 / exact - 1.0).abs() < 2.5e-4,
      "{t}: {} vs {exact}",
      got.c1
    );
  }
}

/// The scheme's order on the full equation, against 3200 steps.
#[test]
fn the_adams_chf_converges_at_the_scheme_s_order() {
  let (t, z) = (1.0, z_gr());
  let reference = gr_cf(3200).log_chf(t, z);
  let errors = [100, 200, 400, 800].map(|steps| (gr_cf(steps).log_chf(t, z) - reference).norm());
  assert!(errors[1] < 1e-4 * reference.norm(), "{errors:?}");
  for ratio in ratios(&errors) {
    assert!(ratio > 2.0_f64.powf(1.2), "{errors:?}");
  }
}
