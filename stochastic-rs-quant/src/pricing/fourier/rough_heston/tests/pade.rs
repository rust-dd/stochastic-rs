use num_complex::Complex64;

use super::super::LogChf;
use super::super::Riccati;
use super::super::RoughHestonParams;
use super::super::adams::AdamsCf;
use super::super::cumulants;
use super::super::pade::Pade;
use super::super::pade::PadeCf;
use super::z_gr;

const GR: Riccati = Riccati {
  kappa: 1.0,
  nu: 0.4,
  rho: -0.65,
};

fn params(kappa: f64) -> RoughHestonParams {
  RoughHestonParams {
    v0: 0.04,
    kappa,
    theta: 0.06,
    nu: 0.4,
    rho: -0.65,
    hurst: 0.05,
  }
}

/// `(b₁…b_n, g₀…g_{n−1})`, the two series' coefficients in powers of `t^α`.
fn series(pade: &Pade, z: Complex64) -> (Vec<Complex64>, Vec<Complex64>) {
  let moment = Riccati::moment(z);
  let (short, long) = pade.series(z, &GR);
  let scale = |c: &Complex64| c * moment;
  (
    short.iter().map(scale).collect(),
    long.iter().map(scale).collect(),
  )
}

/// The (2,2) approximant in closed form, as the two matching conditions at each end give it.
#[test]
fn the_solve_reproduces_the_closed_two_two_approximant() {
  let pade = Pade::new(0.55, 2);
  for z in [z_gr(), Complex64::new(0.5, 40.0), Complex64::new(1.0, 0.3)] {
    let moment = Riccati::moment(z);
    let (b, g) = series(&pade, z);
    let (p, q) = pade.coefficients(z, &GR);
    let den = g[0] * g[0] + b[0] * g[1];
    let q1 = (b[0] * g[0] - b[1] * g[1]) / den;
    let q2 = (b[0] * b[0] + b[1] * g[0]) / den;
    let want = [
      (q[1], q1),
      (q[2], q2),
      (p[0] * moment, b[0]),
      (p[1] * moment, b[1] + b[0] * q1),
    ];
    for (got, want) in want {
      assert!(
        (got - want).norm() < 1e-12 * want.norm(),
        "{z}: {got} vs {want}"
      );
    }
  }
}

/// `h^{(n,n)}` leaves the short series at `y^{n+1}` and the long one at `y^{−n}`.
#[test]
fn the_approximant_matches_both_series_to_its_order() {
  let z = z_gr();
  let moment = Riccati::moment(z);
  for order in [3, 4] {
    let pade = Pade::new(0.55, order);
    let (b, g) = series(&pade, z);
    let coefficients = pade.coefficients(z, &GR);
    let h = |y: f64| Pade::h(&coefficients, moment, y);
    let short = |y: f64| {
      let partial = b.iter().rev().fold(Complex64::ZERO, |sum, c| (sum + c) * y);
      (h(y) - partial).norm()
    };
    let long = |y: f64| {
      let partial = g.iter().rev().fold(Complex64::ZERO, |sum, c| sum / y + c);
      (h(y) - partial).norm()
    };
    let (near, far) = (short(0.02) / short(0.002), long(1e2) / long(1e3));
    let (want_near, want_far) = (10.0_f64.powi(order as i32 + 1), 10.0_f64.powi(order as i32));
    assert!(
      (0.3..3.0).contains(&(near / want_near)),
      "order {order}: {near}"
    );
    assert!(
      (0.3..3.0).contains(&(far / want_far)),
      "order {order}: {far}"
    );
  }
}

#[test]
fn the_scaled_system_is_regular_where_the_moment_vanishes() {
  let pade = Pade::new(0.55, 4);
  for origin in [Complex64::ZERO, Complex64::ONE] {
    let (p0, q0) = pade.coefficients(origin, &GR);
    let (p1, q1) = pade.coefficients(origin + Complex64::new(0.0, 1e-8), &GR);
    for (at, near) in p0.iter().zip(&p1).chain(q0.iter().zip(&q1)) {
      assert!(
        at.is_finite() && (at - near).norm() < 1e-6,
        "{origin}: {at} vs {near}"
      );
    }
  }
  let cf = PadeCf::new(params(1.0), 4);
  for xi in [Complex64::new(1e-7, 0.0), Complex64::new(1e-7, -1.0)] {
    assert!((cf.chf(1.0, xi) - 1.0).norm() < 1e-6, "{xi}");
  }
}

#[test]
fn the_pade_chf_is_nan_off_the_strip_and_mirrors_across_the_imaginary_axis() {
  let cf = PadeCf::new(params(1.0), 4);
  for im in [0.1, -1.1, f64::NAN] {
    assert!(cf.chf(1.0, Complex64::new(3.0, im)).is_nan(), "Im ξ = {im}");
  }
  for im in [0.0, -0.5, -1.0] {
    let xi = Complex64::new(3.0, im);
    assert!(cf.chf(1.0, xi).is_finite(), "Im ξ = {im}");
    assert_eq!(cf.chf(1.0, -xi.conj()), cf.chf(1.0, xi).conj());
  }
  assert_eq!(cf.chf(0.0, Complex64::new(3.0, -0.5)), Complex64::ONE);
  assert_eq!(cf.chf(1.0, Complex64::new(0.0, -1.0)), Complex64::ONE);
  assert!(cf.chf(-1.0, Complex64::new(3.0, -0.5)).is_nan());
}

/// The integrand is rational in `y = s^α` against the Jacobi weight `y^{1/α−1}`: 32 nodes against
/// 64 on the chf, whose size bounds what a far frequency's initial layer can cost a price.
#[test]
fn thirty_two_jacobi_nodes_integrate_the_approximant() {
  for (hurst, order) in [(0.05, 4), (0.05, 3), (0.3, 4)] {
    let p = RoughHestonParams {
      hurst,
      ..params(1.0)
    };
    let (coarse, fine) = (
      PadeCf::with_nodes(p, order, 32),
      PadeCf::with_nodes(p, order, 64),
    );
    for t in [1.0 / 52.0, 1.0, 5.0] {
      for u in [0.5, 3.0, 30.0, 300.0] {
        let z = Complex64::new(0.5, u);
        let gap = (coarse.log_chf(t, z).exp() - fine.log_chf(t, z).exp()).norm();
        assert!(gap < 1e-9, "H = {hurst}, t = {t}, u = {u}: {gap}");
      }
    }
  }
}

#[test]
fn the_pade_cumulants_are_its_own_chf_s_and_near_the_adams_ones() {
  for kappa in [1.0, 0.01] {
    let cf = PadeCf::new(params(kappa), 4);
    let reference = cumulants(&AdamsCf::new(params(kappa), 400), 1.0);
    let got = cumulants(&cf, 1.0);
    let f = |u: f64| cf.log_chf(1.0, Complex64::new(0.0, u));
    let h = 2e-3;
    let (f2p, fp, f0, fm, f2m) = (f(2.0 * h), f(h), f(0.0), f(-h), f(-2.0 * h));
    let c1 = (-f2p + 8.0 * fp - 8.0 * fm + f2m).im / (12.0 * h);
    let c2 = -(-f2p + 16.0 * fp - 30.0 * f0 + 16.0 * fm - f2m).re / (12.0 * h * h);
    assert!(
      (got.c1 - c1).abs() < 1e-8,
      "κ = {kappa}: {} vs {c1}",
      got.c1
    );
    assert!(
      (got.c2 - c2).abs() < 1e-7,
      "κ = {kappa}: {} vs {c2}",
      got.c2
    );
    assert!(
      (got.c1 / reference.c1 - 1.0).abs() < 1e-3,
      "κ = {kappa}: {}",
      got.c1
    );
    assert!(
      (got.c2 / reference.c2 - 1.0).abs() < 1e-3,
      "κ = {kappa}: {}",
      got.c2
    );
    let (fourth, want) = (
      got.c4.expect("fourth order"),
      reference.c4.expect("fourth order"),
    );
    assert!(
      (fourth / want - 1.0).abs() < 0.05,
      "κ = {kappa}: {fourth} vs {want}"
    );
  }
}

/// On the strip `|φ| ≤ E[e^{−Im ξ·X}] ≤ 1`; where a pole of the approximant breaks that, the chf
/// is NaN, never a finite value a price could be built on.
#[test]
fn a_value_no_characteristic_function_takes_is_nan() {
  let cases = [
    (
      RoughHestonParams::new(0.04, 1.0, 0.04, 1.0, -0.9, 0.005),
      0.05,
    ),
    (
      RoughHestonParams::new(0.04, 0.01, 0.04, 0.6, -0.9, 0.005),
      1.0,
    ),
  ];
  for (p, tau) in cases {
    let cf = PadeCf::new(p, 4);
    let line = (1..800).map(|i| cf.chf(tau, Complex64::new(0.25 * f64::from(i), -0.5)));
    let values = line.collect::<Vec<_>>();
    assert!(
      values.iter().any(|v| v.is_nan()),
      "{p:?}: the approximant has no pole here"
    );
    for value in values {
      assert!(
        value.is_nan() || value.norm() <= 1.0 + 1e-12,
        "{p:?}: |φ| = {}",
        value.norm()
      );
    }
  }
}
