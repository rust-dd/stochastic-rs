//! Every in-tree `cumulants()` is the cumulant of the log-return its own `chf` describes: mpmath's
//! 40-digit derivatives of each log-chf, and the same identity in f64 by stencils of `ln chf`.

use num_complex::Complex64;
use stochastic_rs_quant::calibration::hkde::HkdeParams;
use stochastic_rs_quant::calibration::svj::SvjParams;
use stochastic_rs_quant::pricing::fourier::BatesFourier;
use stochastic_rs_quant::pricing::fourier::BsmFourier;
use stochastic_rs_quant::pricing::fourier::CgmyFourier;
use stochastic_rs_quant::pricing::fourier::Cumulants;
use stochastic_rs_quant::pricing::fourier::DoubleHestonFourier;
use stochastic_rs_quant::pricing::fourier::FourierModelExt;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::fourier::HkdeFourier;
use stochastic_rs_quant::pricing::fourier::KouFourier;
use stochastic_rs_quant::pricing::fourier::MertonJdFourier;
use stochastic_rs_quant::pricing::fourier::NigFourier;
use stochastic_rs_quant::pricing::fourier::VarianceGammaFourier;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::regime_switching::RegimeSwitchingModel;

/// κ₁ and κ₂ from the log-chf by fourth-order central differences at `h`; the pins carry κ₄.
fn chf_c1_c2(model: &impl FourierModelExt, t: f64, h: f64) -> (f64, f64) {
  let f = |u: f64| model.chf(t, Complex64::new(u, 0.0)).ln();
  let (f2p, fp, f0, fm, f2m) = (f(2.0 * h), f(h), f(0.0), f(-h), f(-2.0 * h));
  let c1 = (-f2p + 8.0 * fp - 8.0 * fm + f2m).im / (12.0 * h);
  let c2 = -(-f2p + 16.0 * fp - 30.0 * f0 + 16.0 * fm - f2m).re / (12.0 * h * h);
  (c1, c2)
}

/// `|a − b| < tol`, false for a NaN.
fn within(a: f64, b: f64, tol: f64) -> bool {
  (a - b).abs() < tol
}

fn check(
  failures: &mut Vec<String>,
  name: &str,
  model: &impl FourierModelExt,
  t: f64,
  c1: f64,
  c2: f64,
  c4: Option<f64>,
) {
  let c = model.cumulants(t);
  if !within(c.c1, c1, 1e-12) {
    failures.push(format!("{name} c1: {} vs {c1}", c.c1));
  }
  if !within(c.c2, c2, 1e-12) {
    failures.push(format!("{name} c2: {} vs {c2}", c.c2));
  }
  match (c.c4, c4) {
    (Some(a), Some(b)) if within(a, b, 1e-12 * b.abs().max(1.0)) => {}
    (None, None) => {}
    (got, want) => failures.push(format!("{name} c4: {got:?} vs {want:?}")),
  }
  let (n1, n2) = chf_c1_c2(model, t, 2e-3);
  if !within(c.c1, n1, 1e-8 * c.c2.max(1.0)) {
    failures.push(format!("{name} c1 vs its chf: {} vs {n1}", c.c1));
  }
  if !within(c.c2, n2, 1e-7 * c.c2.max(1.0)) {
    failures.push(format!("{name} c2 vs its chf: {} vs {n2}", c.c2));
  }
}

#[test]
fn the_cumulant_table() {
  let mut failures = Vec::new();
  let f = &mut failures;
  check(
    f,
    "bsm",
    &BsmFourier { sigma: 0.2 },
    1.0,
    -0.02,
    0.04,
    Some(0.0),
  );
  let h1 = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let h2 = HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.05,
    sigma: 0.4,
    rho: -0.6,
  };
  let h3 = HestonParams {
    v0: 0.02,
    kappa: 0.5,
    theta: 0.09,
    sigma: 1.0,
    rho: -0.9,
  };
  let (c2, c4) = (0.042812050407517586, Some(0.0036778377954532243));
  check(f, "heston 1", &HestonFourier::new(h1), 1.0, -0.02, c2, c4);
  let (c1, c2, c4) = (
    -0.010919698602928607,
    0.02282006837723739,
    Some(0.0009080872879893181),
  );
  check(f, "heston 2", &HestonFourier::new(h2), 0.5, c1, c2, c4);
  let (c1, c2, c4) = (
    -0.1607459499036729,
    0.7818334040390548,
    Some(32.69287447445167),
  );
  check(f, "heston 3", &HestonFourier::new(h3), 5.0, c1, c2, c4);
  let vg = VarianceGammaFourier {
    sigma: 0.2,
    theta: -0.14,
    nu: 0.2,
  };
  check(
    f,
    "vg",
    &vg,
    1.0,
    -0.021417366913419788,
    0.04392,
    Some(0.00135475968),
  );
  let cgmy = CgmyFourier {
    c: 1.0,
    g: 5.0,
    m: 10.0,
    y: 0.5,
  };
  let (c1, c2, c4) = (
    -0.05086520575698068,
    0.10729150203410986,
    Some(0.012940917745892644),
  );
  check(f, "cgmy", &cgmy, 1.0, c1, c2, c4);
  let merton = MertonJdFourier {
    sigma: 0.2,
    lambda: 0.5,
    mu_j: -0.1,
    sigma_j: 0.15,
  };
  check(
    f,
    "merton",
    &merton,
    2.0,
    -0.05507431355915235,
    0.1125,
    Some(0.00296875),
  );
  let kou = KouFourier {
    sigma: 0.2,
    lambda: 0.8,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  };
  check(
    f,
    "kou",
    &kou,
    1.5,
    -0.059333333333333335,
    0.1272,
    Some(0.0288),
  );
  let nig = NigFourier {
    alpha: 15.0,
    beta: -5.0,
    delta: 0.5,
  };
  let (c1, c2, c4) = (
    -0.019428359761631972,
    0.039774756441743296,
    Some(0.0009695096882674929),
  );
  check(f, "nig", &nig, 1.0, c1, c2, c4);
  let bates = BatesFourier::new(SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma_v: 0.3,
    rho: -0.5,
    lambda: 0.5,
    mu_j: -0.1,
    sigma_j: 0.15,
  });
  let (c1, c2, c4) = (
    -0.027537156779576175,
    0.058290711570025924,
    Some(0.004110197131229457),
  );
  check(f, "bates", &bates, 1.0, c1, c2, c4);
  let hkde = HkdeFourier::new(HkdeParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma_v: 0.5,
    rho: -0.5,
    lambda: 0.8,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  });
  let (c1, c2, c4) = (
    -0.03955555555555556,
    0.08787631094153805,
    Some(0.025378548950244037),
  );
  check(f, "hkde", &hkde, 1.0, c1, c2, c4);
  let second = HestonParams {
    v0: 0.03,
    kappa: 2.0,
    theta: 0.05,
    sigma: 0.4,
    rho: -0.6,
  };
  let (c1, c2, c4) = (
    -0.04067667641618306,
    0.0870049599731861,
    Some(0.00795819576959339),
  );
  check(
    f,
    "double heston",
    &DoubleHestonFourier::new(h1, second),
    1.0,
    c1,
    c2,
    c4,
  );
  let regime = |initial_state| RegimeSwitchingModel {
    q_matrix: ndarray::array![[-0.5, 0.5], [0.3, -0.3]],
    vols: ndarray::array![0.2, 0.4],
    initial_state,
  };
  check(
    f,
    "regime t=1",
    &regime(0),
    1.0,
    -0.03168729519299476,
    0.06370181258561475,
    None,
  );
  check(
    f,
    "regime t=0.25",
    &regime(1),
    0.25,
    -0.01947319756968176,
    0.03895130985811067,
    None,
  );
  check(
    f,
    "regime t=5",
    &regime(0),
    5.0,
    -0.2414835455729094,
    0.49040486171133396,
    None,
  );
  assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn the_width_helper_counts_an_absent_or_unusable_c4_as_zero_and_is_nan_without_a_variance() {
  let cumulants = |c2, c4| Cumulants { c1: 0.0, c2, c4 };
  let w = cumulants(0.04, Some(0.0016)).truncation_half_width(10.0);
  assert!((w - 10.0 * (0.04_f64 + 0.04).sqrt()).abs() < 1e-15, "{w}");
  for c4 in [None, Some(f64::NAN), Some(-1.0), Some(f64::INFINITY)] {
    assert_eq!(
      cumulants(0.04, c4).truncation_half_width(10.0),
      2.0,
      "{c4:?}"
    );
  }
  for c2 in [0.0, -0.04, f64::NAN, f64::INFINITY] {
    let w = cumulants(c2, Some(1.0)).truncation_half_width(10.0);
    assert!(w.is_nan(), "c2 = {c2}: {w}");
  }
}
