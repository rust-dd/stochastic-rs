//! A fit whose Kendall's τ the family cannot represent is `InvalidParameter { name: "tau", .. }`, never a stored
//! out-of-range `theta`; a representable τ fits.

use ndarray::Array2;
use stochastic_rs_copulas::CopulaError;
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
use stochastic_rs_copulas::simd_rng::Deterministic;
use stochastic_rs_copulas::traits::BivariateExt;

/// Pseudo-observations with Kendall's τ = −1: `v_i = 1 − u_i` on the rank grid.
fn countermonotone(n: usize) -> Array2<f64> {
  Array2::from_shape_fn((n, 2), |(i, j)| {
    let u = (i + 1) as f64 / (n + 1) as f64;
    if j == 0 { u } else { 1.0 - u }
  })
}

/// Pseudo-observations of a Clayton(θ = 3) sample: τ ≈ 0.6, beyond FGM's `2/9` and AMH's `1/3`.
fn strongly_dependent() -> Array2<f64> {
  let truth = Clayton {
    theta: Some(3.0),
    ..Clayton::new()
  };
  pseudo_observations(truth.sample(600, &Deterministic::new(11)).unwrap().view())
}

fn is_tau_error(result: Result<(), CopulaError>) -> bool {
  matches!(
    result,
    Err(CopulaError::InvalidParameter { name: "tau", .. })
  )
}

#[test]
fn a_fit_outside_the_family_domain_is_an_error_and_stores_nothing() {
  let neg = countermonotone(200);
  macro_rules! rejects {
    ($copula:expr, $data:expr) => {{
      let mut c = $copula;
      assert!(is_tau_error(c.fit($data.view())), "{:?}", c.family());
      assert_eq!(c.theta(), None, "{:?} kept a theta", c.family());
    }};
  }
  rejects!(Clayton::new(), neg);
  rejects!(Gumbel::new(None), neg);
  rejects!(Joe::new(), neg);
  rejects!(Galambos::new(), neg);
  rejects!(HuslerReiss::new(), neg);
  rejects!(MarshallOlkin::new(), neg);
  rejects!(Bb1::new(None, None), neg);
  rejects!(Bb7::new(None, None), neg);
  rejects!(Frank::new(None), neg);
  rejects!(GaussianCopula::new(), neg);
  rejects!(TCopula::new(), neg);
  rejects!(Plackett::new(), neg);
  let strong = strongly_dependent();
  rejects!(Fgm::new(), strong);
  rejects!(Amh::new(), strong);
}

#[test]
fn a_representable_tau_fits_every_family() {
  let strong = strongly_dependent();
  let mut clayton = Clayton::new();
  clayton.fit(strong.view()).unwrap();
  assert!((clayton.tau().unwrap() - 0.6).abs() < 0.08);
  let mut independence = Independence::new();
  independence.fit(strong.view()).unwrap();
  assert_eq!(independence.tau(), Some(0.0));
  let mild = pseudo_observations(
    Clayton {
      theta: Some(0.3),
      ..Clayton::new()
    }
    .sample(600, &Deterministic::new(12))
    .unwrap()
    .view(),
  );
  for (name, result) in [
    ("fgm", Fgm::new().fit(mild.view())),
    ("amh", Amh::new().fit(mild.view())),
    ("frank", Frank::new(None).fit(mild.view())),
    ("plackett", Plackett::new().fit(mild.view())),
  ] {
    assert!(result.is_ok(), "{name}: {result:?}");
  }
}

type Inverse = Box<dyn Fn(f64) -> Result<f64, CopulaError>>;

type Forward = Box<dyn Fn(f64) -> f64>;

#[test]
fn theta_from_tau_round_trips_through_tau_from_theta() {
  let cases: Vec<(&str, Inverse, Forward, Vec<f64>)> = vec![
    (
      "clayton",
      Box::new(|t| Clayton::new().theta_from_tau(t)),
      Box::new(|th| Clayton::new().tau_from_theta(th)),
      vec![0.1, 0.5, 0.9],
    ),
    (
      "frank",
      Box::new(|t| Frank::new(None).theta_from_tau(t)),
      Box::new(|th| Frank::new(None).tau_from_theta(th)),
      vec![-0.6, -0.1, 0.3, 0.7],
    ),
    (
      "joe",
      Box::new(|t| Joe::new().theta_from_tau(t)),
      Box::new(|th| Joe::new().tau_from_theta(th)),
      vec![0.2, 0.5, 0.8],
    ),
    (
      "plackett",
      Box::new(|t| Plackett::new().theta_from_tau(t)),
      Box::new(|th| Plackett::new().tau_from_theta(th)),
      vec![-0.3, 0.15, 0.5],
    ),
    (
      "galambos",
      Box::new(|t| Galambos::new().theta_from_tau(t)),
      Box::new(|th| Galambos::new().tau_from_theta(th)),
      vec![0.2, 0.5],
    ),
    (
      "husler_reiss",
      Box::new(|t| HuslerReiss::new().theta_from_tau(t)),
      Box::new(|th| HuslerReiss::new().tau_from_theta(th)),
      vec![0.2, 0.5],
    ),
    (
      "amh",
      Box::new(|t| Amh::new().theta_from_tau(t)),
      Box::new(|th| Amh::new().tau_from_theta(th)),
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

/// One discordant pair in 30 000 leaves `τ̂ = 1 − 4.4e-9`, whose `ρ = sin(πτ̂/2)` rounds onto the excluded `ρ = 1`: `fit`
/// validates the inverse before storing it.
#[test]
fn an_inverse_that_rounds_onto_an_excluded_theta_is_an_error() {
  let n = 30_000;
  let mut data = Array2::from_shape_fn((n, 2), |(i, _)| (i + 1) as f64 / (n + 1) as f64);
  data.swap([0, 1], [1, 1]);
  let mut gaussian = GaussianCopula::new();
  let got = gaussian.fit(data.view());
  assert!(
    matches!(got, Err(CopulaError::InvalidParameter { name: "theta", value, .. }) if value == 1.0),
    "{got:?}"
  );
  assert_eq!(gaussian.theta(), None);
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
        let want = Err(CopulaError::DimensionMismatch { expected: 2, got });
        assert_eq!(c.fit(data.view()), want, "{:?}", c.family());
        assert_eq!(c.theta(), None, "{:?} kept a theta", c.family());
      }
    }};
  }
  mismatch!(Clayton::new());
  mismatch!(Independence::new());
  mismatch!(Bb1::new(None, None));
  mismatch!(Bb7::new(None, None));
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
          let got = c.fit(data.view());
          assert_eq!(
            got,
            Err(CopulaError::MarginalOutOfRange),
            "{:?} at {bad}",
            c.family()
          );
          assert_eq!(c.theta(), None, "{:?} kept a theta", c.family());
        }};
      }
      out_of_range!(Clayton::new());
      out_of_range!(Independence::new());
      out_of_range!(Bb1::new(None, None));
      out_of_range!(Bb7::new(None, None));
    }
  }
}

/// Raw draws have uniform margins, yet a Kolmogorov–Smirnov gate at its 1 % bound fails some of these seeds by chance;
/// `fit` tests the domain only, so every seed fits.
#[test]
fn a_raw_uniform_sample_fits_on_every_seed() {
  let truth = Clayton {
    theta: Some(3.0),
    ..Clayton::new()
  };
  for seed in 0..200 {
    let raw = truth.sample(600, &Deterministic::new(seed)).unwrap();
    let mut fitted = Clayton::new();
    assert_eq!(fitted.fit(raw.view()), Ok(()), "seed {seed}");
  }
}
