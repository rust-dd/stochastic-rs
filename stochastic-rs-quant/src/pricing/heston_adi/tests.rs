//! The ADI solver against the semi-analytic Heston price on in 't Hout & Foulon's (2008) Table 1
//! sets, and the schemes' structural identities.

use super::*;
use crate::pricing::heston::HestonPricer;

const P: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
};

/// Read back, never written: a write would pass `with_grid`'s or `with_barrier`'s check.
#[test]
fn the_grid_and_the_barrier_are_read_through_getters() {
  let pricer = HestonAdiPricer::new(P)
    .with_grid(4, 6, 1)
    .with_barrier(50.0);
  assert_eq!((pricer.grid(), pricer.barrier()), ((4, 6, 1), Some(50.0)));
  assert_eq!(HestonAdiPricer::new(P).grid(), (100, 50, 50));
  assert_eq!(HestonAdiPricer::new(P).barrier(), None);
}

/// The setters are the only gate, so a barrier of `+∞` is refused as one at zero is.
#[test]
#[should_panic(expected = "barrier must satisfy `barrier.is_finite() && barrier > 0.0`, got")]
fn an_infinite_barrier_is_refused() {
  let _ = HestonAdiPricer::new(P).with_barrier(f64::INFINITY);
}

#[test]
#[should_panic(expected = "m2 must satisfy `m2 >= 4`, got m2 = 3")]
fn a_grid_check_names_its_argument_and_value() {
  let _ = HestonAdiPricer::new(P).with_grid(4, 3, 1);
}

#[test]
fn spot_beyond_eight_strikes_keeps_the_intrinsic_value() {
  let model = HestonAdiPricer::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  })
  .with_grid(40, 20, 20);
  for spot in [80.0, 100.0, 200.0] {
    let call = model.price_call(spot, 10.0, 0.0, 0.0, 1.0);
    let put = model.price_put(spot, 10.0, 0.0, 0.0, 1.0);
    assert!(
      (call - (spot - 10.0)).abs() < 1e-3,
      "spot {spot}: call {call}"
    );
    assert!((-1e-6..1e-3).contains(&put), "spot {spot}: put {put}");
  }
}

#[test]
fn initial_variance_beyond_five_matches_the_analytic_price() {
  let p = HestonParams {
    v0: 6.0,
    kappa: 2.0,
    ..P
  };
  let model = HestonAdiPricer::new(p).with_grid(160, 80, 80);
  let analytic = HestonPricer::new(p);
  let want = analytic.price_call(100.0, 100.0, 0.0, 0.0, 1.0);
  let got = model.price_call(100.0, 100.0, 0.0, 0.0, 1.0);
  assert!(
    (got - want).abs() / want < 0.012,
    "ADI {got}, analytic {want}"
  );
}

/// `(κ, η, σ, ρ, r_d, r_f, T)` with `K = 100`.
const TABLE_1: [(f64, f64, f64, f64, f64, f64, f64); 4] = [
  (1.5, 0.04, 0.3, -0.9, 0.025, 0.0, 1.0),
  (3.0, 0.12, 0.04, 0.6, 0.01, 0.04, 1.0),
  (0.6067, 0.0707, 0.2928, -0.7571, 0.03, 0.0, 3.0),
  (2.5, 0.06, 0.5, -0.1, 0.0507, 0.0469, 0.25),
];

fn analytic(case: usize, s: f64, v0: f64) -> f64 {
  let (kappa, eta, sigma, rho, r_d, r_f, tau) = TABLE_1[case];
  HestonPricer::new(HestonParams {
    v0,
    kappa,
    theta: eta,
    sigma,
    rho,
  })
  .price_call(s, 100.0, r_d, r_f, tau)
}

fn adi(case: usize, v0: f64) -> HestonAdiPricer {
  let (kappa, eta, sigma, rho, _, _, _) = TABLE_1[case];
  HestonAdiPricer::new(HestonParams {
    v0,
    kappa,
    theta: eta,
    sigma,
    rho,
  })
}

/// Damped MCS(⅓) on a 100 × 50 mesh reprices the four Table 1 cases on `½K < s < 3K/2`,
/// `0 < v < 1` to the paper's 1 % at `m2 = 50`.
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
  let base = HestonAdiPricer::new(HestonParams {
    v0: 0.06,
    kappa,
    theta: eta,
    sigma,
    rho: 0.0,
  })
  .with_grid(40, 20, 20);
  let douglas = base
    .with_scheme(AdiScheme::Douglas)
    .price_call(100.0, 100.0, r_d, r_f, tau);
  let craig_sneyd = base
    .with_scheme(AdiScheme::CraigSneyd)
    .price_call(100.0, 100.0, r_d, r_f, tau);
  assert_eq!(douglas, craig_sneyd);
}

/// Every scheme converges as the step shrinks, and with correlation the second-order schemes beat
/// Douglas at equal step counts.
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
    (HestonAdiPricer::new(HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.9
    })
    .with_adi_theta(0.4)
    .scheme_theta()
      - 0.4)
      .abs()
      < 1e-15
  );
}

#[test]
fn each_rejected_parameter_is_named_with_its_value() {
  let cases: [(&str, fn()); 5] = [
    (
      "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = -0.01",
      || {
        let _ = HestonAdiPricer::new(HestonParams { v0: -0.01, ..P });
      },
    ),
    ("kappa must satisfy `kappa > 0.0`, got kappa = 0.0", || {
      let _ = HestonAdiPricer::new(HestonParams { kappa: 0.0, ..P });
    }),
    ("theta must satisfy `theta > 0.0`, got theta = 0.0", || {
      let _ = HestonAdiPricer::new(HestonParams { theta: 0.0, ..P });
    }),
    (
      "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = NaN",
      || {
        let _ = HestonAdiPricer::new(HestonParams {
          sigma: f64::NAN,
          ..P
        });
      },
    ),
    (
      "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = -1.5",
      || {
        let _ = HestonAdiPricer::new(HestonParams { rho: -1.5, ..P });
      },
    ),
  ];
  for (expected, build) in cases {
    let payload = std::panic::catch_unwind(build).expect_err(expected);
    let message = payload
      .downcast_ref::<String>()
      .map(String::as_str)
      .unwrap_or_default();
    assert_eq!(message, expected);
  }
}

/// §2.1 assumes Feller for the `v = 0` outflow boundary. Below it (`2κη/σ²` of 0.48, 0.089 and
/// 0.09 here) the error against the Fourier price still falls threefold or more per mesh halving.
#[test]
fn converges_below_the_feller_condition() {
  for (kappa, theta, sigma, rho, v0) in [
    (1.5, 0.04, 0.5, -0.7, 0.04),
    (0.1, 0.04, 0.3, -0.7, 0.04),
    (0.5, 0.09, 1.0, -0.9, 0.02),
  ] {
    let p = HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    };
    assert!(2.0 * kappa * theta < sigma * sigma);
    let want = HestonPricer::new(p).price_call(100.0, 120.0, 0.03, 0.01, 1.0);
    let error = |m1, m2, steps| {
      let adi = HestonAdiPricer::new(p).with_grid(m1, m2, steps);
      (adi.price_call(100.0, 120.0, 0.03, 0.01, 1.0) - want).abs()
    };
    let (coarse, fine) = (error(100, 50, 50), error(200, 100, 100));
    assert!(
      coarse < 2e-2 && fine < coarse / 3.0,
      "sigma = {sigma}: {coarse} then {fine}"
    );
  }
}
