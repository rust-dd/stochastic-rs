//! A fit whose Kendall's τ the family cannot represent is `InvalidParameter { name: "tau", .. }`, never a stored
//! out-of-range `theta`; a representable τ fits.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::BivariateFamily;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::fgm::Fgm;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::galambos::Galambos;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::husler_reiss::HuslerReiss;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs_copulas::bivariate::plackett::Plackett;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::gof::pseudo_observations;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::TailDependence;
use stochastic_rs_core::simd_rng::Deterministic;

/// Pseudo-observations with Kendall's τ = −1: `v_i = 1 − u_i` on the rank grid.
fn countermonotone(n: usize) -> Array2<f64> {
  Array2::from_shape_fn((n, 2), |(i, j)| {
    let u = (i + 1) as f64 / (n + 1) as f64;
    if j == 0 { u } else { 1.0 - u }
  })
}

/// Pseudo-observations of a Clayton(θ = 3) sample: τ ≈ 0.6, beyond FGM's `2/9` and AMH's `1/3`.
fn strongly_dependent() -> Array2<f64> {
  let truth = Clayton::new(3.0);
  pseudo_observations(truth.sample(600, &Deterministic::new(11)).unwrap().view())
}

/// Pseudo-observations with Kendall's τ = +1: `v_i = u_i`.
fn comonotone(n: usize) -> Array2<f64> {
  Array2::from_shape_fn((n, 2), |(i, _)| (i + 1) as f64 / (n + 1) as f64)
}

/// Kendall's τ-b of a tie-free sample by counting its pairs.
fn kendall(x: &Array2<f64>) -> f64 {
  let n = x.nrows();
  let mut score = 0.0;
  for i in 0..n {
    for j in i + 1..n {
      score += ((x[[i, 0]] - x[[j, 0]]) * (x[[i, 1]] - x[[j, 1]])).signum();
    }
  }
  score / (n * (n - 1) / 2) as f64
}

fn bb1_at(delta: f64) -> Bb1 {
  let mut c = Bb1::default();
  c.set_delta(delta);
  c
}

fn bb7_at(delta: f64) -> Bb7 {
  let mut c = Bb7::default();
  c.set_delta(delta);
  c
}

fn is_tau_error(result: Result<(), CopulaError>) -> bool {
  matches!(
    result,
    Err(CopulaError::InvalidParameter { name: "tau", .. })
  )
}

macro_rules! rejects {
  ($copula:expr, $data:expr) => {{
    let mut c = $copula;
    assert!(is_tau_error(c.fit($data.view())), "{:?}", c.family());
    assert_eq!(c.theta(), None, "{:?} kept a theta", c.family());
  }};
}

#[test]
fn a_fit_outside_the_family_domain_is_an_error_and_stores_nothing() {
  let neg = countermonotone(200);
  rejects!(Clayton::default(), neg);
  rejects!(Gumbel::default(), neg);
  rejects!(Joe::default(), neg);
  rejects!(Galambos::default(), neg);
  rejects!(HuslerReiss::default(), neg);
  rejects!(MarshallOlkin::default(), neg);
  rejects!(Bb1::default(), neg);
  rejects!(Bb7::default(), neg);
  rejects!(Frank::default(), neg);
  rejects!(GaussianCopula::default(), neg);
  rejects!(TCopula::default(), neg);
  rejects!(Plackett::default(), neg);
  let strong = strongly_dependent();
  rejects!(Fgm::default(), strong);
  rejects!(Amh::default(), strong);
}

/// Four pairs with three concordances and three discordances: `τ̂ = 0`, which Marshall–Olkin reaches only at its
/// excluded `θ = 0`.
#[test]
fn marshall_olkin_rejects_a_zero_tau() {
  let ranks = [2.0, 4.0, 1.0, 3.0];
  let data = Array2::from_shape_fn((4, 2), |(i, j)| {
    if j == 0 {
      (i + 1) as f64 / 5.0
    } else {
      ranks[i] / 5.0
    }
  });
  assert_eq!(kendall(&data), 0.0);
  rejects!(MarshallOlkin::default(), data);
}

/// `τ̂ = +1` fits Marshall–Olkin at `θ = 1`, the upper Fréchet bound, and Independence, which ignores τ; BB1 and BB7
/// reject `τ̂ ≥ 1` with the rest.
#[test]
fn a_comonotone_sample_fits_only_the_families_that_reach_tau_one() {
  let co = comonotone(200);
  rejects!(Clayton::default(), co);
  rejects!(Gumbel::default(), co);
  rejects!(Joe::default(), co);
  rejects!(Frank::default(), co);
  rejects!(Amh::default(), co);
  rejects!(Fgm::default(), co);
  rejects!(Plackett::default(), co);
  rejects!(GaussianCopula::default(), co);
  rejects!(TCopula::default(), co);
  rejects!(Galambos::default(), co);
  rejects!(HuslerReiss::default(), co);
  rejects!(Bb1::default(), co);
  rejects!(Bb7::default(), co);
  let mut mo = MarshallOlkin::default();
  mo.fit(co.view()).unwrap();
  assert_eq!(mo.theta(), Some(1.0));
  let mut independence = Independence::new();
  independence.fit(co.view()).unwrap();
  assert_eq!(independence.tau(), Some(0.0));
}

/// A Clayton(0.3) sample, `τ̂ ≈ 0.13`, lies in every family's range: an inversion fit stores the `θ` whose τ is `τ̂` to
/// its τ rule's accuracy, and BB1's and BB7's likelihood fits land near it.
#[test]
fn a_representable_tau_fits_every_family() {
  let mild = pseudo_observations(
    Clayton::new(0.3)
      .sample(600, &Deterministic::new(12))
      .unwrap()
      .view(),
  );
  let tau = kendall(&mild);
  macro_rules! inverts {
    ($copula:expr, $tolerance:expr) => {{
      let mut c = $copula;
      c.fit(mild.view()).unwrap();
      let got = c.tau().unwrap();
      assert!(
        (got - tau).abs() < $tolerance,
        "{:?}: τ {got} vs τ̂ {tau}",
        c.family()
      );
    }};
  }
  inverts!(Clayton::default(), 1e-15);
  inverts!(Gumbel::default(), 1e-15);
  inverts!(Joe::default(), 1e-14);
  inverts!(Frank::default(), 1e-14);
  inverts!(Amh::default(), 1e-14);
  inverts!(Fgm::default(), 1e-15);
  inverts!(GaussianCopula::default(), 1e-15);
  inverts!(TCopula::default(), 1e-15);
  inverts!(MarshallOlkin::default(), 1e-15);
  inverts!(Plackett::default(), 1e-8);
  inverts!(Galambos::default(), 1e-8);
  inverts!(HuslerReiss::default(), 1e-8);
  inverts!(Bb1::default(), 0.05);
  inverts!(Bb7::default(), 0.05);
  let mut independence = Independence::new();
  independence.fit(mild.view()).unwrap();
  assert_eq!(independence.tau(), Some(0.0));
  let strong = strongly_dependent();
  let mut clayton = Clayton::default();
  clayton.fit(strong.view()).unwrap();
  assert!((clayton.tau().unwrap() - 0.6).abs() < 0.08);
}

type Inverse = Box<dyn Fn(f64) -> Result<f64, CopulaError>>;

type Forward = Box<dyn Fn(f64) -> f64>;

#[test]
fn theta_from_tau_round_trips_through_tau_from_theta() {
  let cases: Vec<(&str, Inverse, Forward, Vec<f64>)> = vec![
    (
      "clayton",
      Box::new(|t| Clayton::default().theta_from_tau(t)),
      Box::new(|th| Clayton::default().tau_from_theta(th)),
      vec![0.1, 0.5, 0.9],
    ),
    (
      "frank",
      Box::new(|t| Frank::default().theta_from_tau(t)),
      Box::new(|th| Frank::default().tau_from_theta(th)),
      vec![-0.6, -0.1, 0.3, 0.7],
    ),
    (
      "joe",
      Box::new(|t| Joe::default().theta_from_tau(t)),
      Box::new(|th| Joe::default().tau_from_theta(th)),
      vec![0.2, 0.5, 0.8],
    ),
    (
      "plackett",
      Box::new(|t| Plackett::default().theta_from_tau(t)),
      Box::new(|th| Plackett::default().tau_from_theta(th)),
      vec![-0.3, 0.15, 0.5],
    ),
    (
      "galambos",
      Box::new(|t| Galambos::default().theta_from_tau(t)),
      Box::new(|th| Galambos::default().tau_from_theta(th)),
      vec![0.2, 0.5],
    ),
    (
      "husler_reiss",
      Box::new(|t| HuslerReiss::default().theta_from_tau(t)),
      Box::new(|th| HuslerReiss::default().tau_from_theta(th)),
      vec![0.2, 0.5],
    ),
    (
      "amh",
      Box::new(|t| Amh::default().theta_from_tau(t)),
      Box::new(|th| Amh::default().tau_from_theta(th)),
      vec![-0.1, 0.1, 0.3],
    ),
  ];
  for (name, inverse, forward, taus) in cases {
    for tau in taus {
      let theta = inverse(tau).unwrap();
      assert!(
        (forward(theta) - tau).abs() < 1e-6,
        "{name}: τ {tau} → θ {theta} → τ {}",
        forward(theta)
      );
    }
  }
}

/// One discordant pair in 30 000 leaves `τ̂ = 1 − 4.4e-9`, whose `ρ = sin(πτ̂/2)` rounds onto the excluded `ρ = 1`, and
/// BB1(δ = 1.125) rounds onto `θ = 0` for the four floats above its lower τ: both are the τ error, never a stored θ.
#[test]
fn an_inverse_that_rounds_onto_an_excluded_theta_is_an_error() {
  let n = 30_000;
  let mut data = Array2::from_shape_fn((n, 2), |(i, _)| (i + 1) as f64 / (n + 1) as f64);
  data.swap([0, 1], [1, 1]);
  let mut gaussian = GaussianCopula::default();
  assert!(is_tau_error(gaussian.fit(data.view())));
  assert_eq!(gaussian.theta(), None);
  for tau in [1.0 - 1e-9, -1.0 + 1e-9] {
    let got = GaussianCopula::default().theta_from_tau(tau);
    assert!(
      matches!(got, Err(CopulaError::InvalidParameter { name: "tau", .. })),
      "{got:?}"
    );
    let got = TCopula::default().theta_from_tau(tau);
    assert!(
      matches!(got, Err(CopulaError::InvalidParameter { name: "tau", .. })),
      "{got:?}"
    );
  }
  let bb1 = bb1_at(1.125);
  let mut tau = 1.0_f64 - 1.0 / 1.125;
  for _ in 0..4 {
    tau = tau.next_up();
    assert_eq!(2.0 / (1.125 * (1.0 - tau)) - 2.0, 0.0);
    let got = bb1.theta_from_tau(tau);
    assert!(
      matches!(got, Err(CopulaError::InvalidParameter { name: "tau", .. })),
      "{got:?}"
    );
  }
  assert!(bb1.theta_from_tau(tau.next_up()).unwrap() > 0.0);
}

/// Ten pairs with 15 discordances have `τ̂ = 1/3`, AMH's upper end, reached at `θ = 1`.
#[test]
fn amh_fits_its_upper_tau() {
  let ranks = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.0, 8.0, 9.0, 10.0];
  let data = Array2::from_shape_fn((10, 2), |(i, j)| {
    if j == 0 {
      (i + 1) as f64 / 11.0
    } else {
      ranks[i] / 11.0
    }
  });
  let mut amh = Amh::default();
  amh.fit(data.view()).unwrap();
  assert_eq!(amh.theta(), Some(1.0));
  assert_eq!(amh.tau(), Some(1.0 / 3.0));
}

/// One column, or three, is a `DimensionMismatch` before any column is read, whichever `fit` the family runs.
#[test]
fn fit_takes_exactly_two_columns() {
  let one = Array2::from_shape_fn((20, 1), |(i, _)| (i + 1) as f64 / 21.0);
  let three = Array2::from_shape_fn((20, 3), |(i, j)| ((i + j) % 20 + 1) as f64 / 21.0);
  macro_rules! mismatch {
    ($copula:expr) => {{
      for (data, got) in [(&one, 1), (&three, 3)] {
        let mut c = $copula;
        let before = c.theta();
        let want = Err(CopulaError::DimensionMismatch { expected: 2, got });
        assert_eq!(c.fit(data.view()), want, "{:?}", c.family());
        assert_eq!(c.theta(), before, "{:?} stored a theta", c.family());
      }
    }};
  }
  mismatch!(Clayton::default());
  mismatch!(Independence::new());
  mismatch!(Bb1::default());
  mismatch!(Bb7::default());
}

/// An outside implementor whose `theta_from_tau` ignores its own bounds `[0, 1]`.
struct Careless {
  theta: Option<f64>,
}

impl BivariateExt for Careless {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Clayton
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn set_theta(&mut self, theta: f64) {
    self.theta = Some(theta);
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    Vec::new()
  }

  fn tau_from_theta(&self, theta: f64) -> f64 {
    theta
  }

  fn theta_from_tau(&self, _tau: f64) -> Result<f64, CopulaError> {
    Ok(2.0)
  }

  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    Ok(TailDependence {
      lower: 0.0,
      upper: 0.0,
    })
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(Array1::ones(x.nrows()))
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(&x.column(0) * &x.column(1))
  }
}

/// The default `fit` validates the θ an outside implementor's `theta_from_tau` answers before storing it.
#[test]
fn the_default_fit_validates_an_implementors_inverse() {
  let mut careless = Careless { theta: None };
  let got = careless.fit(countermonotone(20).view());
  assert!(
    matches!(got, Err(CopulaError::InvalidParameter { name: "theta", value, .. }) if value == 2.0),
    "{got:?}"
  );
  assert_eq!(careless.theta(), None);
}

/// Columns, then rows, then range, whichever `fit` the family runs: a 1 × 3 input is a `DimensionMismatch`, a 1 × 2 input
/// of NaN is `InsufficientData`.
#[test]
fn fit_checks_columns_then_rows_then_range() {
  let wide = Array2::from_elem((1, 3), f64::NAN);
  let short = Array2::from_elem((1, 2), f64::NAN);
  macro_rules! order {
    ($copula:expr) => {{
      let mut c = $copula;
      let before = c.theta();
      let want = Err(CopulaError::DimensionMismatch {
        expected: 2,
        got: 3,
      });
      assert_eq!(c.fit(wide.view()), want, "{:?}", c.family());
      let want = Err(CopulaError::InsufficientData { needed: 2, got: 1 });
      assert_eq!(c.fit(short.view()), want, "{:?}", c.family());
      assert_eq!(c.theta(), before, "{:?} stored a theta", c.family());
    }};
  }
  order!(Clayton::default());
  order!(Independence::new());
  order!(Bb1::default());
  order!(Bb7::default());
}

/// A value that is not a finite number in `[0, 1]` is `MarginalOutOfRange`, in either column.
#[test]
fn fit_rejects_a_value_outside_the_unit_interval() {
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.5] {
    for column in 0..2 {
      let mut data = countermonotone(20);
      data[[7, column]] = bad;
      macro_rules! out_of_range {
        ($copula:expr) => {{
          let mut c = $copula;
          let before = c.theta();
          let got = c.fit(data.view());
          assert_eq!(
            got,
            Err(CopulaError::MarginalOutOfRange),
            "{:?} at {bad}",
            c.family()
          );
          assert_eq!(c.theta(), before, "{:?} stored a theta", c.family());
        }};
      }
      out_of_range!(Clayton::default());
      out_of_range!(Independence::new());
      out_of_range!(Bb1::default());
      out_of_range!(Bb7::default());
    }
  }
}

/// Raw draws have uniform margins, yet a Kolmogorov–Smirnov gate at its 1 % bound fails some of these seeds by chance;
/// `fit` tests the domain only, so every seed fits.
#[test]
fn a_raw_uniform_sample_fits_on_every_seed() {
  let truth = Clayton::new(3.0);
  for seed in 0..200 {
    let raw = truth.sample(600, &Deterministic::new(seed)).unwrap();
    let mut fitted = Clayton::default();
    assert_eq!(fitted.fit(raw.view()), Ok(()), "seed {seed}");
  }
}

/// Every `Ok` of `theta_from_tau` passes `validate_theta`, on a grid and at each range end and its float neighbours.
#[test]
fn every_inverse_is_an_admitted_theta() {
  let mut taus: Vec<f64> = (-100..=100).map(|i| f64::from(i) / 100.0).collect();
  for edge in [
    0.0,
    1e-300,
    1e-13,
    1e-12,
    2.0 / 9.0,
    1.0 / 3.0,
    (5.0 - 8.0 * 2.0_f64.ln()) / 3.0,
    1.0 - 1.0 / 1.125,
    1.0 - 6e-9,
    1.0,
  ] {
    for e in [edge, -edge] {
      taus.extend([
        e.next_down().next_down(),
        e.next_down(),
        e,
        e.next_up(),
        e.next_up().next_up(),
      ]);
    }
  }
  macro_rules! admitted {
    ($copula:expr) => {{
      let c = $copula;
      for &tau in &taus {
        if let Ok(theta) = c.theta_from_tau(tau) {
          assert!(
            c.validate_theta(theta).is_ok(),
            "{:?}: τ = {tau:?} → θ = {theta:?}",
            c.family()
          );
        }
      }
    }};
  }
  admitted!(Clayton::default());
  admitted!(Gumbel::default());
  admitted!(Joe::default());
  admitted!(Frank::default());
  admitted!(Amh::default());
  admitted!(Fgm::default());
  admitted!(Plackett::default());
  admitted!(GaussianCopula::default());
  admitted!(TCopula::default());
  admitted!(MarshallOlkin::default());
  admitted!(Independence::new());
  admitted!(bb1_at(1.125));
  admitted!(bb7_at(1.1));
  admitted!(Galambos::default());
  admitted!(HuslerReiss::default());
}

#[test]
fn tail_dependence_shares_the_result_channel() {
  assert_eq!(
    Clayton::default().tail_dependence(),
    Err(CopulaError::NotFitted)
  );
  assert!(matches!(
    Gumbel::try_new(0.5),
    Err(CopulaError::InvalidParameter { name: "theta", .. })
  ));
  let clayton = Clayton::new(2.0);
  let tails = clayton.tail_dependence().unwrap();
  assert!((tails.lower - 2.0_f64.powf(-0.5)).abs() < 1e-12 && tails.upper == 0.0);
  assert_eq!(
    MarshallOlkin::default().tail_dependence(),
    Err(CopulaError::NotFitted)
  );
}
