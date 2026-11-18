//! Every in-tree `cumulants()` is the cumulant of the log-return its own `chf` describes: mpmath's
//! 40-digit derivatives of each log-chf, and the same identity in f64 by stencils of `ln chf`.

use num_complex::Complex64;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::fourier::BatesFourier;
use stochastic_rs_quant::pricing::fourier::BsmFourier;
use stochastic_rs_quant::pricing::fourier::CgmyFourier;
use stochastic_rs_quant::pricing::fourier::CosEngine;
use stochastic_rs_quant::pricing::fourier::Cumulants;
use stochastic_rs_quant::pricing::fourier::DoubleHestonFourier;
use stochastic_rs_quant::pricing::fourier::FourierModelExt;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::fourier::HkdeFourier;
use stochastic_rs_quant::pricing::fourier::HkdeParams;
use stochastic_rs_quant::pricing::fourier::KouFourier;
use stochastic_rs_quant::pricing::fourier::MertonJdFourier;
use stochastic_rs_quant::pricing::fourier::NigFourier;
use stochastic_rs_quant::pricing::fourier::SvjParams;
use stochastic_rs_quant::pricing::fourier::VarianceGammaFourier;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::regime_switching::RegimeSwitchingModel;
use stochastic_rs_quant::traits::ModelPricer;

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
    sigma: 0.3,
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
    sigma: 0.5,
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
    Some(0.0040188323614403845),
  );
  check(
    f,
    "regime t=0.25",
    &regime(1),
    0.25,
    -0.01947319756968176,
    0.03895130985811067,
    Some(5.837728602746485e-5),
  );
  check(
    f,
    "regime t=5",
    &regime(0),
    5.0,
    -0.2414835455729094,
    0.49040486171133396,
    Some(0.08863731099080273),
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

/// `|a − b| < rel·|b|` on each cumulant, false for a NaN or an absent `c4`.
fn check_relative(
  failures: &mut Vec<String>,
  name: &str,
  model: &impl FourierModelExt,
  t: f64,
  want: (f64, f64, f64),
  rel: f64,
) {
  let c = model.cumulants(t);
  let got = [c.c1, c.c2, c.c4.unwrap_or(f64::NAN)];
  for (label, (got, want)) in ["c1", "c2", "c4"]
    .into_iter()
    .zip(got.into_iter().zip([want.0, want.1, want.2]))
  {
    if !within(got, want, rel * want.abs()) {
      failures.push(format!("{name}, t = {t}, {label}: {got} vs {want}"));
    }
  }
}

fn heston(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> HestonParams {
  HestonParams {
    v0,
    kappa,
    theta,
    sigma,
    rho,
  }
}

/// mpmath's 60-digit cumulants at `κt` from 1e-6 to 1.05: the closed forms in `1/κ` cancel below
/// `κt = 1`, the fourth to nothing, and the series in `t` that replaces them meets them there.
#[test]
fn heston_cumulants_keep_their_digits_at_a_small_kappa() {
  let mut failures = Vec::new();
  let f = &mut failures;
  for (kappa, t, c2, c4) in [
    (1e-4, 0.01, 0.00040042029985977505, 7.1449861827034e-9),
    (1e-4, 0.1, 0.00404229985775036, 7.299530715614993e-6),
    (1e-4, 1.0, 0.044499837504549895, 0.009019056120474007),
    (1e-3, 0.01, 0.00040042029859775353, 7.144945874218141e-9),
    (1e-3, 0.1, 0.004042298577536049, 7.299115536368414e-6),
    (1e-3, 1.0, 0.044498375454892526, 0.009013514969836588),
    (0.01, 0.01, 0.00040042028597785106, 7.144542805373049e-9),
    (0.01, 0.1, 0.004042285778604262, 7.294965404866033e-6),
    (0.01, 1.0, 0.044483795392726934, 0.008958341038451643),
    (0.0, 1.0, 0.0445, 0.00901967207142857),
  ] {
    let model = HestonFourier::new(heston(0.04, kappa, 0.04, 0.3, -0.7));
    let name = format!("kappa = {kappa}");
    check_relative(f, &name, &model, t, (-0.02 * t, c2, c4), 1e-13);
  }
  let model = HestonFourier::new(heston(0.04, 1e-4, 0.04, 0.3, 0.3));
  let want = (-0.002, 0.003982300057749861, 4.182840616865885e-6);
  check_relative(f, "rho = 0.3", &model, 0.1, want, 1e-13);
  for (t, c1, c2, c4) in [
    (
      1.9,
      -0.042571871641815084,
      0.14287068545207995,
      1.0812241827866693,
    ),
    (
      1.99,
      -0.04543064111808413,
      0.15494660573663585,
      1.2794204183020048,
    ),
    (
      2.0,
      -0.04575156088200096,
      0.1563169055655099,
      1.3029585145227762,
    ),
    (
      2.1,
      -0.04899564243778087,
      0.17033106610974363,
      1.555914357991067,
    ),
  ] {
    let model = HestonFourier::new(heston(0.02, 0.5, 0.09, 1.0, -0.9));
    check_relative(f, "across the switch", &model, t, (c1, c2, c4), 1e-11);
  }
  let bates = BatesFourier::new(SvjParams {
    v0: 0.04,
    kappa: 1e-3,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.5,
    lambda: 0.5,
    mu_j: -0.1,
    sigma_j: 0.15,
  });
  let want = (
    -0.0027537156779576175,
    0.005655298977526049,
    0.0001539533041376621,
  );
  check_relative(f, "bates", &bates, 0.1, want, 1e-13);
  let hkde = HkdeFourier::new(HkdeParams {
    v0: 0.04,
    kappa: 1e-3,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.5,
    lambda: 0.8,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  });
  let want = (
    -0.003955555555555556,
    0.00853083160421125,
    0.0019355412285453934,
  );
  check_relative(f, "hkde", &hkde, 0.1, want, 1e-13);
  let double = DoubleHestonFourier::new(
    heston(0.04, 1e-3, 0.04, 0.3, -0.7),
    heston(0.03, 2.0, 0.05, 0.4, -0.6),
  );
  let want = (
    -0.0035936537653899095,
    0.007265127257493327,
    1.5030949348073349e-5,
  );
  check_relative(f, "double heston", &double, 0.1, want, 1e-13);
  assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A `c4` of rounding noise (1.3e8 at this κ) widens the range 1,300 times and puts this put 29
/// off at K 80.
#[test]
fn the_default_cos_put_holds_at_a_small_kappa() {
  let model = HestonFourier::new(heston(0.04, 1e-4, 0.04, 0.3, -0.7));
  for k in [80.0, 100.0, 120.0] {
    let put = CosEngine::default().price(&model, 100.0, k, 0.03, 0.01, 0.1, OptionType::Put);
    let want = model.price_put(100.0, k, 0.03, 0.01, 0.1);
    assert!((put - want).abs() < 1e-7, "K = {k}: {put} vs {want}");
  }
}

fn three_states(initial_state: usize) -> RegimeSwitchingModel {
  RegimeSwitchingModel {
    q_matrix: ndarray::array![[-1.0, 0.5, 0.5], [0.3, -0.6, 0.3], [0.4, 0.4, -0.8]],
    vols: ndarray::array![0.1, 0.25, 0.5],
    initial_state,
  }
}

/// mpmath's 60-digit derivatives of the regime log-chf, from a chain that barely switches to one
/// that switches 1,000 times; one state is Black–Scholes, whose `c4` is exactly 0.
#[test]
fn regime_switching_cumulants_match_mpmath() {
  let mut failures = Vec::new();
  let f = &mut failures;
  for (t, c1, c2, c4) in [
    (
      0.01,
      -5.3638687612343844e-5,
      0.00010727986347984686,
      2.988477696453268e-8,
    ),
    (
      0.1,
      -0.0008486105828292529,
      0.0016994853251261662,
      2.738838269815171e-5,
    ),
    (
      1.0,
      -0.02878858083708144,
      0.05856819512957116,
      0.012421687865042367,
    ),
    (
      10.0,
      -0.5123411010586318,
      1.0599735549927356,
      0.4580224554829713,
    ),
  ] {
    check_relative(f, "three states", &three_states(0), t, (c1, c2, c4), 1e-10);
  }
  let want = (
    -0.012089583733568438,
    0.024181961314369407,
    3.32757706889181e-5,
  );
  check_relative(f, "from the high state", &three_states(2), 0.1, want, 1e-10);
  let two = |down: f64, up: f64, vols: [f64; 2]| RegimeSwitchingModel {
    q_matrix: ndarray::array![[-down, down], [up, -up]],
    vols: ndarray::arr1(&vols),
    initial_state: 0,
  };
  for (t, c1, c2, c4) in [
    (
      0.01,
      -0.00021128438519986726,
      0.00042273246425934586,
      1.96632798508574e-6,
    ),
    (
      1.0,
      -0.13940000000007777,
      0.27919200000013966,
      0.00465082264964076,
    ),
    (50.0, -7.2444, 14.5084, 0.2322448319106048),
  ] {
    let fast = two(20.0, 5.0, [0.1, 0.6]);
    check_relative(f, "fast", &fast, t, (c1, c2, c4), 1e-10);
  }
  for (t, c1, c2, c4) in [
    (
      0.01,
      -0.00020000299997000024,
      0.00040000600113997646,
      1.440619204254164e-11,
    ),
    (
      1.0,
      -0.020029970022486507,
      0.04006113764802995,
      1.5019615194837014e-5,
    ),
  ] {
    let slow = two(0.001, 0.002, [0.2, 0.4]);
    check_relative(f, "slow", &slow, t, (c1, c2, c4), 1e-10);
  }
  assert!(failures.is_empty(), "\n{}", failures.join("\n"));
  let one = RegimeSwitchingModel {
    q_matrix: ndarray::array![[0.0]],
    vols: ndarray::array![0.2],
    initial_state: 0,
  };
  assert_eq!(one.cumulants(1.0).c4, Some(0.0));
}

/// A range of `10√c2` alone leaves this put, started in the low state, 2.9e-4 off.
#[test]
fn the_default_cos_put_holds_for_three_regimes_at_a_short_maturity() {
  let model = three_states(0);
  for k in [80.0, 100.0, 120.0] {
    let put = CosEngine::default().price(&model, 100.0, k, 0.03, 0.01, 0.1, OptionType::Put);
    let want = model.price_put(100.0, k, 0.03, 0.01, 0.1);
    assert!((put - want).abs() < 1e-7, "K = {k}: {put} vs {want}");
  }
}
