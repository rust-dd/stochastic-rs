use num_complex::Complex64;

use super::implied_vol;
use crate::pricing::fourier::CarrMadanPricer;
use crate::pricing::fourier::FourierModelExt;
use crate::pricing::fourier::GilPelaezPricer;
use crate::pricing::fourier::LewisPricer;
use crate::pricing::fourier::RoughHeston;
use crate::pricing::fourier::RoughHestonCf;
use crate::pricing::fourier::RoughHestonParams;
use crate::traits::ModelPricer;
use crate::vol_surface::ModelSurface;

/// Gatheral & Radoičić's (4.2) and Abi Jaber's (3.7).
const SETS: [(&str, RoughHestonParams); 2] = [
  (
    "GR",
    RoughHestonParams {
      v0: 0.04,
      kappa: 1.0,
      theta: 0.04,
      nu: 0.4,
      rho: -0.65,
      hurst: 0.05,
    },
  ),
  (
    "AJ",
    RoughHestonParams {
      v0: 0.02,
      kappa: 0.3,
      theta: 0.02,
      nu: 0.3,
      rho: -0.7,
      hurst: 0.1,
    },
  ),
];

const MATURITIES: [f64; 3] = [1.0 / 52.0, 0.25, 1.0];

/// Log-strikes that widen with `√t`, as a smile's range does.
fn strikes(tau: f64) -> [f64; 3] {
  let width = (tau / 0.25).sqrt().clamp(0.3, 1.5);
  [-0.3, 0.0, 0.1].map(|k: f64| (k * width).exp())
}

#[test]
fn the_selector_defaults_to_the_fourth_order_approximant_and_checks_its_knobs() {
  assert_eq!(RoughHestonCf::default(), RoughHestonCf::Pade { order: 4 });
  assert_eq!(RoughHestonCf::ADAMS, RoughHestonCf::Adams { steps: 200 });
  let lifted = RoughHestonCf::Lifted {
    factors: 20,
    ratio: 2.5,
    steps: 300,
  };
  assert_eq!(RoughHestonCf::LIFTED, lifted);
  let bad = [
    (RoughHestonCf::Adams { steps: 0 }, "steps must satisfy"),
    (RoughHestonCf::Pade { order: 5 }, "order must satisfy"),
    (RoughHestonCf::Pade { order: 2 }, "order must satisfy"),
    (
      RoughHestonCf::Lifted {
        factors: 0,
        ratio: 2.5,
        steps: 300,
      },
      "factors must satisfy",
    ),
    (
      RoughHestonCf::Lifted {
        factors: 20,
        ratio: 1.0,
        steps: 300,
      },
      "ratio must satisfy",
    ),
    (
      RoughHestonCf::Lifted {
        factors: 20,
        ratio: f64::NAN,
        steps: 300,
      },
      "ratio must satisfy",
    ),
    (
      RoughHestonCf::Lifted {
        factors: 20,
        ratio: 2.5,
        steps: 0,
      },
      "steps must satisfy",
    ),
  ];
  for (cf, words) in bad {
    let reason = cf.violation();
    assert!(
      reason.as_deref().is_some_and(|r| r.starts_with(words)),
      "{cf:?}: {reason:?}"
    );
  }
  let model = RoughHeston::new(SETS[0].1, RoughHestonCf::LIFTED);
  assert_eq!(
    (model.params(), model.cf()),
    (SETS[0].1, RoughHestonCf::LIFTED)
  );
}

#[test]
#[should_panic(expected = "order must satisfy `order == 3 || order == 4`, got order = 6")]
fn a_selector_off_its_domain_panics_in_new() {
  RoughHeston::new(SETS[0].1, RoughHestonCf::Pade { order: 6 });
}

#[test]
#[should_panic(expected = "hurst must satisfy")]
fn parameters_off_their_domain_panic_in_new() {
  let p = RoughHestonParams {
    hurst: 0.5,
    ..SETS[0].1
  };
  RoughHeston::new(p, RoughHestonCf::default());
}

/// Implied vol against the Adams solution at 100 steps (itself within 3e-5): twice the worst gap
/// measured over these sets, 5.3e-4 for (4,4), 7.1e-4 for (3,3), and for twenty lifted factors
/// 4.1e-3 at a week and 2.0e-3 from three months.
#[test]
fn the_three_characteristic_functions_agree_to_their_stated_accuracy() {
  for (name, p) in SETS {
    let reference = RoughHeston::new(p, RoughHestonCf::Adams { steps: 100 });
    let others = [
      (RoughHestonCf::Pade { order: 4 }, 1.1e-3, 1.1e-3),
      (RoughHestonCf::Pade { order: 3 }, 1.5e-3, 1.5e-3),
      (RoughHestonCf::LIFTED, 8.5e-3, 4.0e-3),
    ];
    let models = others.map(|(cf, week, later)| (RoughHeston::new(p, cf), week, later));
    for tau in MATURITIES {
      for k in strikes(tau) {
        let want = implied_vol(&reference, k, tau);
        for (model, week, later) in &models {
          let got = implied_vol(model, k, tau);
          let bound = if tau < 0.1 { *week } else { *later };
          let cf = model.cf();
          assert!(
            (got - want).abs() < bound,
            "{name} {cf:?} at ({k}, {tau}): {got} vs {want}"
          );
        }
      }
    }
  }
}

/// 100 steps against 200: at the scheme's order `1 + α` the default's own error is under a third of
/// the gap, 1e-5 where the gap is under 4e-5.
#[test]
fn the_adams_price_has_converged_at_its_default_steps() {
  let (_, p) = SETS[0];
  let coarse = RoughHeston::new(p, RoughHestonCf::Adams { steps: 100 });
  let fine = RoughHeston::new(p, RoughHestonCf::ADAMS);
  for k in strikes(0.25) {
    let gap = (implied_vol(&coarse, k, 0.25) - implied_vol(&fine, k, 0.25)).abs();
    assert!(gap < 4e-5, "k = {k}: {gap}");
  }
}

#[test]
fn the_blankets_price_and_invert_every_selector() {
  let (_, p) = SETS[0];
  for cf in [
    RoughHestonCf::Adams { steps: 100 },
    RoughHestonCf::default(),
    RoughHestonCf::LIFTED,
  ] {
    let model = RoughHeston::new(p, cf);
    for k in [90.0, 110.0] {
      let lewis = LewisPricer::price_call(&model, 100.0, k, 0.03, 0.01, 0.5);
      let gil_pelaez = GilPelaezPricer::price_call(&model, 100.0, k, 0.03, 0.01, 0.5);
      assert!(
        (lewis - gil_pelaez).abs() < 1e-6,
        "{cf:?} at {k}: {lewis} vs {gil_pelaez}"
      );
      let put = model.price_put(100.0, k, 0.03, 0.01, 0.5);
      let parity = 100.0 * (-0.01_f64 * 0.5).exp() - k * (-0.03_f64 * 0.5).exp();
      assert!((model.price_call(100.0, k, 0.03, 0.01, 0.5) - put - parity).abs() < 1e-9);
    }
    let surface = model.vol_surface(100.0, 0.03, 0.01, &[90.0, 100.0, 110.0], &[0.25, 1.0]);
    assert!(
      surface.ivs.iter().all(|vol| vol.is_finite() && *vol > 0.05),
      "{cf:?}"
    );
    let chf = model.chf(1.0, Complex64::new(3.0, -0.5));
    assert!(chf.is_finite() && chf.norm() < 1.0, "{cf:?}: {chf}");
    assert_eq!(model.chf(1.0, Complex64::new(0.0, -1.0)), Complex64::ONE);
    assert_eq!(model.cumulants(0.0).c2, 0.0);
  }
}

/// Carr–Madan damps at `Im ξ = −(1 + α)`, off the approximant's strip; the Adams solve goes there.
#[test]
fn off_the_strip_the_approximant_is_nan_and_the_adams_solution_is_not() {
  let (_, p) = SETS[0];
  let carr_madan = CarrMadanPricer::new(256, 1.5);
  let pade = RoughHeston::new(p, RoughHestonCf::default());
  let adams = RoughHeston::new(p, RoughHestonCf::Adams { steps: 50 });
  assert!(pade.chf(1.0, Complex64::new(3.0, -2.5)).is_nan());
  assert!(
    carr_madan
      .price_call(&pade, 1.0, 1.0, 0.0, 0.0, 1.0)
      .is_nan()
  );
  assert!(adams.chf(1.0, Complex64::new(3.0, -2.5)).is_finite());
}
