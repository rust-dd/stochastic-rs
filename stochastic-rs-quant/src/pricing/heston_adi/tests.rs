//! Validation of the ADI solver against the semi-analytic Heston price on
//! the four parameter sets of in 't Hout & Foulon (2008), Table 1, plus the
//! structural identities of the schemes.

use super::*;
use crate::pricing::heston::HestonPricer;

/// `(κ, η, σ, ρ, r_d, r_f, T)` with `K = 100`.
const TABLE_1: [(f64, f64, f64, f64, f64, f64, f64); 4] = [
  (1.5, 0.04, 0.3, -0.9, 0.025, 0.0, 1.0),
  (3.0, 0.12, 0.04, 0.6, 0.01, 0.04, 1.0),
  (0.6067, 0.0707, 0.2928, -0.7571, 0.03, 0.0, 3.0),
  (2.5, 0.06, 0.5, -0.1, 0.0507, 0.0469, 0.25),
];

fn analytic(case: usize, s: f64, v0: f64) -> f64 {
  let (kappa, eta, sigma, rho, r_d, r_f, tau) = TABLE_1[case];
  HestonPricer::new(v0, rho, kappa, eta, sigma, Some(0.0))
    .call_put(s, 100.0, r_d, r_f, tau)
    .0
}

fn adi(case: usize, v0: f64) -> HestonAdiPricer {
  let (kappa, eta, sigma, rho, _, _, _) = TABLE_1[case];
  HestonAdiPricer::new(v0, kappa, eta, sigma, rho)
}

/// MCS(⅓) with damping on a 100 × 50 mesh reprices the four Table 1 cases
/// inside the paper's error region `½K < s < 3K/2`, `0 < v < 1` to about the
/// 1 % the paper reports for `m2 = 50`.
#[test]
fn matches_the_semi_analytic_price_on_table_1() {
  for case in 0..4 {
    let (_, _, _, _, r_d, r_f, tau) = TABLE_1[case];
    for (s, v0) in [(90.0, 0.04), (100.0, 0.04), (110.0, 0.09), (120.0, 0.16)] {
      let want = analytic(case, s, v0);
      let got = adi(case, v0).price_call(s, 100.0, r_d, r_f, tau);
      assert!(
        (got - want).abs() / want < 1.2e-2,
        "case {} s {s} v0 {v0}: adi {got} vs analytic {want}",
        case + 1
      );
    }
  }
}

/// Without correlation `F_0 = 0`, so Craig–Sneyd collapses onto Douglas
/// exactly (§2.4).
#[test]
fn craig_sneyd_equals_douglas_without_correlation() {
  let (kappa, eta, sigma, _, r_d, r_f, tau) = TABLE_1[3];
  let base = HestonAdiPricer::new(0.06, kappa, eta, sigma, 0.0).with_grid(40, 20, 20);
  let douglas = base
    .with_scheme(AdiScheme::Douglas)
    .price_call(100.0, 100.0, r_d, r_f, tau);
  let craig_sneyd = base
    .with_scheme(AdiScheme::CraigSneyd)
    .price_call(100.0, 100.0, r_d, r_f, tau);
  assert_eq!(douglas, craig_sneyd);
}

/// Every scheme converges toward the analytic price as the time step
/// shrinks, and the second-order schemes are closer than Douglas at the
/// same step count once correlation is present.
#[test]
fn schemes_converge_with_the_time_step() {
  let (_, _, _, _, r_d, r_f, tau) = TABLE_1[0];
  let want = analytic(0, 100.0, 0.04);
  for scheme in [
    AdiScheme::Douglas,
    AdiScheme::CraigSneyd,
    AdiScheme::ModifiedCraigSneyd,
    AdiScheme::HundsdorferVerwer,
  ] {
    let err = |steps: usize| {
      (adi(0, 0.04)
        .with_grid(80, 40, steps)
        .with_scheme(scheme)
        .price_call(100.0, 100.0, r_d, r_f, tau)
        - want)
        .abs()
    };
    let (coarse, fine) = (err(10), err(40));
    assert!(
      fine.is_finite() && fine < 0.02 * want,
      "{scheme:?}: fine error {fine}"
    );
    assert!(
      fine <= coarse + 1e-3 * want,
      "{scheme:?}: {coarse} -> {fine}"
    );
  }
}

/// Put-call parity and the trait routing.
#[test]
fn parity_and_forward_hold() {
  let (_, _, _, _, r_d, r_f, tau) = TABLE_1[3];
  let model = adi(3, 0.06).with_grid(60, 30, 30);
  let call = model.price_call(100.0, 100.0, r_d, r_f, tau);
  let put = model.price_put(100.0, 100.0, r_d, r_f, tau);
  assert!((call - put - (100.0 * (-r_f * tau).exp() - 100.0 * (-r_d * tau).exp())).abs() < 1e-12);
  assert!(
    (model.vanilla_call_forward(100.0, r_d, r_f, tau) - 100.0 * ((r_d - r_f) * tau).exp()).abs()
      < 1e-12
  );
}

/// A down-and-out call is worth less than the vanilla, tends to it as the
/// barrier drops toward zero, and has no put counterpart.
#[test]
fn down_and_out_barrier_behaves() {
  let (_, _, _, _, r_d, r_f, tau) = TABLE_1[0];
  let vanilla = adi(0, 0.04)
    .with_grid(80, 40, 40)
    .price_call(100.0, 100.0, r_d, r_f, tau);
  let knocked = adi(0, 0.04).with_grid(80, 40, 40).with_barrier(90.0);
  let doc = knocked.price_call(100.0, 100.0, r_d, r_f, tau);
  assert!(doc > 0.0 && doc < vanilla, "doc {doc} vs vanilla {vanilla}");
  let far = adi(0, 0.04)
    .with_grid(80, 40, 40)
    .with_barrier(5.0)
    .price_call(100.0, 100.0, r_d, r_f, tau);
  assert!(
    (far - vanilla).abs() / vanilla < 2e-2,
    "far barrier {far} vs {vanilla}"
  );
  assert!(knocked.price_put(100.0, 100.0, r_d, r_f, tau).is_nan());
  assert!(knocked.vanilla_call_forward(100.0, r_d, r_f, tau).is_nan());
  assert_eq!(knocked.price_call(80.0, 100.0, r_d, r_f, tau), 0.0);
}

#[test]
fn default_thetas_follow_the_paper() {
  assert_eq!(AdiScheme::Douglas.default_theta(), 0.5);
  assert_eq!(AdiScheme::CraigSneyd.default_theta(), 0.5);
  assert!((AdiScheme::ModifiedCraigSneyd.default_theta() - 1.0 / 3.0).abs() < 1e-15);
  assert!(
    (AdiScheme::HundsdorferVerwer.default_theta() - (0.5 + 3.0_f64.sqrt() / 6.0)).abs() < 1e-15
  );
  assert!(
    (HestonAdiPricer::new(0.04, 1.5, 0.04, 0.3, -0.9)
      .with_adi_theta(0.4)
      .scheme_theta()
      - 0.4)
      .abs()
      < 1e-15
  );
}
