//! At `d = 2` every multivariate copula's density is its bivariate family's, near the edges as inside, now that no
//! 1e-12 clamp moves the coordinates.

use ndarray::Array1;
use ndarray::ArrayView2;
use ndarray::array;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::cvine::CVine;
use stochastic_rs_copulas::multivariate::dvine::DVine;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::multivariate::t::TMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

/// The final review's probe points: a level at `1e-15`, `1e-13` or `1e-300`, one at `1 − 1e-14`, and an interior pair.
const POINTS: [(f64, f64); 5] = [
  (1e-15, 0.5),
  (1e-13, 0.3),
  (0.5, 1.0 - 1e-14),
  (1e-300, 0.7),
  (0.2, 0.3),
];

type Density = Box<dyn Fn(ArrayView2<f64>) -> Result<Array1<f64>, CopulaError>>;

/// A copula's name, its `pdf` and `log_pdf` at `d = 2`, and the bivariate family it must equal.
type Case = (String, (Density, Density), Box<dyn BivariateExt>);

fn densities<C: MultivariateExt + Clone + 'static>(copula: C) -> (Density, Density) {
  let other = copula.clone();
  (
    Box::new(move |x| copula.pdf(x)),
    Box::new(move |x| other.log_pdf(x)),
  )
}

/// The pair's bivariate family.
fn family(pair: PairCopula) -> Box<dyn BivariateExt> {
  match pair {
    PairCopula::Independence => Box::new(Independence::new()),
    PairCopula::Gaussian { rho } => Box::new(GaussianCopula::new(rho)),
    PairCopula::Clayton { theta } => Box::new(Clayton::new(theta)),
    PairCopula::Frank { theta } => Box::new(Frank::new(theta)),
    PairCopula::StudentT { rho, nu } => Box::new(TCopula::new(rho, nu)),
    PairCopula::Bb1 { theta, delta } => Box::new(Bb1::new(theta, delta)),
    PairCopula::Bb7 { theta, delta } => Box::new(Bb7::new(theta, delta)),
    other => panic!("no bivariate family for {other:?}"),
  }
}

#[test]
fn every_two_dimensional_density_is_its_bivariate_familys() {
  let pairs = [
    PairCopula::Independence,
    PairCopula::Gaussian { rho: 0.6 },
    PairCopula::Clayton { theta: 2.0 },
    PairCopula::Frank { theta: 5.0 },
    PairCopula::StudentT { rho: 0.5, nu: 4.0 },
    PairCopula::Bb1 {
      theta: 0.8,
      delta: 1.6,
    },
    PairCopula::Bb7 {
      theta: 1.7,
      delta: 0.9,
    },
  ];
  let mut cases: Vec<Case> = vec![
    (
      "gaussian".into(),
      densities(GaussianMultivariate::new(array![[1.0, 0.6], [0.6, 1.0]])),
      Box::new(GaussianCopula::new(0.6)),
    ),
    (
      "t".into(),
      densities(TMultivariate::new(array![[1.0, 0.5], [0.5, 1.0]], 4.0)),
      Box::new(TCopula::new(0.5, 4.0)),
    ),
  ];
  for pair in pairs {
    let (d, c) = (
      DVine::new(2, vec![vec![pair]]).unwrap(),
      CVine::new(2, vec![vec![pair]]).unwrap(),
    );
    cases.push((format!("dvine {pair:?}"), densities(d), family(pair)));
    cases.push((format!("cvine {pair:?}"), densities(c), family(pair)));
  }
  for (name, (pdf, log_pdf), family) in &cases {
    for (u, v) in POINTS {
      let x = array![[u, v]];
      let (got, want) = (
        log_pdf(x.view()).unwrap()[0],
        family.log_pdf(x.view()).unwrap()[0],
      );
      assert!(
        (got - want).abs() <= 1e-12 * want.abs().max(1.0),
        "{name} ln c({u:e}, {v}): {got} vs {want}"
      );
      let (got, want) = (pdf(x.view()).unwrap()[0], family.pdf(x.view()).unwrap()[0]);
      let close = if want >= f64::MIN_POSITIVE {
        (got - want).abs() <= 1e-12 * want
      } else {
        (got - want).abs() <= f64::MIN_POSITIVE
      };
      assert!(close, "{name} c({u:e}, {v}): {got:e} vs {want:e}");
    }
  }
}
