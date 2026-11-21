//! One input convention for every copula: a batch method answers `MarginalOutOfRange` for a value outside `[0, 1]` or
//! NaN, the densities are NaN on the boundary, where none lives, and the fits and tests on pseudo-observations need them inside `(0, 1)`.

use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::array;
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
use stochastic_rs_copulas::empirical::EmpiricalCopula2D;
use stochastic_rs_copulas::gof::GofConfig;
use stochastic_rs_copulas::gof::gof_cramer_von_mises;
use stochastic_rs_copulas::gof::pseudo_observations;
use stochastic_rs_copulas::multivariate::cvine::CVine;
use stochastic_rs_copulas::multivariate::dvine::DVine;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::multivariate::fit::SelectionCriterion;
use stochastic_rs_copulas::multivariate::fit::VineStructure;
use stochastic_rs_copulas::multivariate::fit::fit_vine;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::multivariate::nac::NacNode;
use stochastic_rs_copulas::multivariate::nac::NestedArchimedean;
use stochastic_rs_copulas::multivariate::rvine::RVine;
use stochastic_rs_copulas::multivariate::t::TMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;
use stochastic_rs_core::simd_rng::Deterministic;

/// NaN, the two sides of `[0, 1]` and the two infinities.
const BAD: [f64; 5] = [f64::NAN, -0.5, 1.5, f64::NEG_INFINITY, f64::INFINITY];

fn with_theta<C: BivariateExt>(mut copula: C, theta: f64) -> C {
  copula.set_theta(theta);
  copula
}

/// Densities NaN on every edge and corner and finite inside; `cdf` exact on the edges and `h` and its inverse in `[0, 1]`
/// there; each batch method `MarginalOutOfRange` at a bad value, and `DimensionMismatch` on one or three columns first.
fn bivariate(copula: &dyn BivariateExt) {
  for (got, fill) in [(1, 0.5), (3, 0.5), (1, f64::NAN), (3, f64::NAN)] {
    let x = Array2::from_elem((2, got), fill);
    let answers = [
      copula.pdf(x.view()),
      copula.log_pdf(x.view()),
      copula.cdf(x.view()),
      copula.partial_derivative(x.view()),
    ];
    for answer in answers {
      assert_eq!(
        answer,
        Err(CopulaError::DimensionMismatch { expected: 2, got })
      );
    }
  }
  let edges = array![
    [0.0, 0.5],
    [1.0, 0.5],
    [0.5, 0.0],
    [0.5, 1.0],
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
  ];
  for values in [copula.pdf(edges.view()), copula.log_pdf(edges.view())] {
    let values = values.unwrap();
    assert!(values.iter().all(|x| x.is_nan()), "{values}");
  }
  let inside = array![[0.3, 0.6], [0.7, 0.2]];
  let pdf = copula.pdf(inside.view()).unwrap();
  let log_pdf = copula.log_pdf(inside.view()).unwrap();
  assert!(pdf.iter().all(|p| p.is_finite() && *p > 0.0), "{pdf}");
  assert!(log_pdf.iter().all(|l| l.is_finite()), "{log_pdf}");
  assert_eq!(
    copula.cdf(edges.view()).unwrap(),
    array![0.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0]
  );
  let h = copula.partial_derivative(edges.view()).unwrap();
  let u = copula
    .percent_point(
      array![0.0, 1.0, 0.3, 0.3].view(),
      array![0.5, 0.5, 0.0, 1.0].view(),
    )
    .unwrap();
  assert!(
    h.iter().chain(&u).all(|x| (0.0..=1.0).contains(x)),
    "{h} {u}"
  );
  for bad in BAD {
    for row in [[bad, 0.5], [0.5, bad]] {
      let x = array![[0.3, 0.6], row];
      let answers = [
        copula.pdf(x.view()),
        copula.log_pdf(x.view()),
        copula.cdf(x.view()),
        copula.partial_derivative(x.view()),
        copula.percent_point(x.column(0), x.column(1)),
        copula.percent_point_numerical(x.column(0), x.column(1)),
        copula.generator(ndarray::aview1(&row)),
      ];
      for answer in answers {
        assert_eq!(answer, Err(CopulaError::MarginalOutOfRange), "{row:?}");
      }
    }
  }
}

macro_rules! bivariate_families {
  ($($name:ident => $copula:expr),* $(,)?) => {
    $(
      #[test]
      fn $name() {
        bivariate(&$copula);
      }
    )*
  };
}

bivariate_families! {
  amh_follows_the_boundary_convention => with_theta(Amh::default(), 0.5),
  bb1_follows_the_boundary_convention => Bb1::new(0.8, 1.6),
  bb7_follows_the_boundary_convention => Bb7::new(1.6, 1.1),
  clayton_follows_the_boundary_convention => Clayton::new(2.0),
  fgm_follows_the_boundary_convention => with_theta(Fgm::default(), 0.5),
  frank_follows_the_boundary_convention => Frank::new(4.0),
  galambos_follows_the_boundary_convention => Galambos::new(1.5),
  gaussian_follows_the_boundary_convention => with_theta(GaussianCopula::default(), 0.5),
  gumbel_follows_the_boundary_convention => Gumbel::new(2.0),
  husler_reiss_follows_the_boundary_convention => HuslerReiss::new(1.5),
  independence_follows_the_boundary_convention => Independence,
  joe_follows_the_boundary_convention => with_theta(Joe::default(), 2.0),
  marshall_olkin_follows_the_boundary_convention => MarshallOlkin::new(0.5, 0.3),
  plackett_follows_the_boundary_convention => Plackett::new(3.0),
  t_follows_the_boundary_convention => TCopula::new(0.5, 4.0),
}

/// Densities NaN on a row with a coordinate at 0 or 1 and finite on an interior row, and `MarginalOutOfRange` from
/// `pdf`, `log_pdf` and `cdf` at a bad value.
fn multivariate(copula: &impl MultivariateExt) {
  let rows = array![
    [0.3, 0.6, 0.45],
    [0.0, 0.6, 0.45],
    [0.3, 1.0, 0.45],
    [0.3, 0.6, 0.0],
    [1.0, 1.0, 1.0]
  ];
  for values in [copula.pdf(rows.view()), copula.log_pdf(rows.view())] {
    let values = values.unwrap();
    assert!(values[0].is_finite(), "{values}");
    assert!(values.iter().skip(1).all(|x| x.is_nan()), "{values}");
  }
  for bad in BAD {
    let x = array![[0.3, 0.6, 0.45], [0.3, bad, 0.45]];
    for answer in [
      copula.pdf(x.view()),
      copula.log_pdf(x.view()),
      copula.cdf(x.view()),
    ] {
      assert_eq!(answer, Err(CopulaError::MarginalOutOfRange), "{bad}");
    }
  }
}

fn correlation() -> Array2<f64> {
  array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
}

fn pairs() -> Vec<Vec<PairCopula>> {
  vec![
    vec![
      PairCopula::Gaussian { rho: 0.4 },
      PairCopula::Clayton { theta: 2.0 },
    ],
    vec![PairCopula::Frank { theta: 3.0 }],
  ]
}

#[test]
fn the_gaussian_multivariate_follows_the_boundary_convention() {
  multivariate(&GaussianMultivariate::new(correlation()));
}

#[test]
fn the_t_multivariate_follows_the_boundary_convention() {
  multivariate(&TMultivariate::new(correlation(), 4.0));
}

#[test]
fn the_vines_follow_the_boundary_convention() {
  multivariate(&DVine::new(3, pairs()).unwrap());
  multivariate(&CVine::new(3, pairs()).unwrap());
  multivariate(&RVine::from_dvine(DVine::new(3, pairs()).unwrap()));
}

#[test]
fn the_nested_archimedean_follows_the_boundary_convention() {
  let root = NacNode {
    theta: 1.0,
    leaves: vec![0],
    children: vec![NacNode::leaf_group(3.0, vec![1, 2])],
  };
  multivariate(&NestedArchimedean::new(BivariateFamily::Clayton, root, 3).unwrap());
}

/// Pseudo-observations of a Clayton(2) sample, with `cell` set to `value`.
fn data(columns: usize, cell: (usize, usize), value: f64) -> Array2<f64> {
  let pairs = Clayton::new(2.0)
    .sample(200, &Deterministic::new(3))
    .unwrap();
  let mut x = pseudo_observations(
    ndarray::concatenate(
      ndarray::Axis(1),
      &[pairs.view(), pairs.column(0).insert_axis(ndarray::Axis(1))],
    )
    .unwrap()
    .slice(ndarray::s![.., ..columns])
    .view(),
  );
  x[cell] = value;
  x
}

/// The rule of every parametric fit and of the test: a 0 or 1 is a `MarginalOnBoundary` at its cell, saying `(0, 1)`, a
/// value off `[0, 1]` stays `MarginalOutOfRange`.
fn assert_fit_rule(fit: impl Fn(ArrayView2<f64>) -> Result<(), CopulaError>, columns: usize) {
  for value in [0.0, 1.0] {
    let got = fit(data(columns, (7, columns - 1), value).view());
    let want = CopulaError::MarginalOnBoundary {
      row: 7,
      column: columns - 1,
      value,
    };
    assert_eq!(got, Err(want.clone()), "{value}");
    assert!(want.to_string().contains("(0, 1)"), "{want}");
  }
  assert_eq!(
    fit(data(columns, (7, 0), 1.5).view()),
    Err(CopulaError::MarginalOutOfRange)
  );
  assert_eq!(fit(data(columns, (7, 0), 0.5).view()), Ok(()));
}

#[test]
fn the_bb_likelihood_fits_reject_a_zero_or_one() {
  assert_fit_rule(|x| Bb1::default().fit(x), 2);
  assert_fit_rule(|x| Bb7::default().fit(x), 2);
}

/// Every bivariate family's `fit`, the Kendall inversions included: the rule holds before the sample's τ, which some
/// families cannot represent, is read.
#[test]
fn every_bivariate_fit_rejects_a_zero_or_one() {
  let mut families: Vec<Box<dyn BivariateExt>> = vec![
    Box::new(Amh::default()),
    Box::new(Bb1::default()),
    Box::new(Bb7::default()),
    Box::new(Clayton::default()),
    Box::new(Fgm::default()),
    Box::new(Frank::default()),
    Box::new(Galambos::default()),
    Box::new(GaussianCopula::default()),
    Box::new(Gumbel::default()),
    Box::new(HuslerReiss::default()),
    Box::new(Independence),
    Box::new(Joe::default()),
    Box::new(MarshallOlkin::default()),
    Box::new(Plackett::default()),
    Box::new(TCopula::default()),
  ];
  for copula in &mut families {
    for value in [0.0, 1.0] {
      let want = CopulaError::MarginalOnBoundary {
        row: 7,
        column: 1,
        value,
      };
      let got = copula.fit(data(2, (7, 1), value).view());
      assert_eq!(got, Err(want), "{:?}", copula.family());
    }
    let got = copula.fit(data(2, (7, 0), 1.5).view());
    assert_eq!(
      got,
      Err(CopulaError::MarginalOutOfRange),
      "{:?}",
      copula.family()
    );
  }
}

#[test]
fn the_nested_archimedean_and_gaussian_fits_reject_a_zero_or_one() {
  assert_fit_rule(
    |x| {
      NestedArchimedean::unfitted(BivariateFamily::Gumbel)
        .unwrap()
        .fit(x)
    },
    3,
  );
  assert_fit_rule(|x| GaussianMultivariate::default().fit(x), 3);
}

#[test]
fn the_vine_fit_rejects_a_zero_or_one() {
  assert_fit_rule(
    |x| {
      fit_vine(
        x,
        VineStructure::DVine,
        BivariateFamily::VINE_CANDIDATES,
        SelectionCriterion::Aic,
      )
      .map(|_| ())
    },
    3,
  );
}

#[test]
fn the_t_multivariate_fit_rejects_a_zero_or_one() {
  assert_fit_rule(|x| TMultivariate::default().fit(x), 3);
}

#[test]
fn the_goodness_of_fit_test_rejects_a_zero_or_one() {
  let config = GofConfig {
    replications: 1,
    alpha: 0.5,
  };
  assert_fit_rule(
    |x| {
      gof_cramer_von_mises(
        &Clayton::new(2.0),
        x,
        config,
        &Deterministic::new(5),
        |c, s| c.fit(s),
      )
      .map(|_| ())
    },
    2,
  );
}

/// The empirical copula's cdf keeps the convention: its edges and corners on the closed square, `MarginalOutOfRange` at a
/// bad value in either coordinate.
#[test]
fn the_empirical_copula_follows_the_boundary_convention() {
  let copula = EmpiricalCopula2D::new(
    array![3.0, 1.0, 2.0, 4.0].view(),
    array![10.0, 30.0, 20.0, 40.0].view(),
  )
  .unwrap();
  assert_eq!(
    copula.cdf(array![[0.0, 0.5], [1.0, 1.0]].view()).unwrap(),
    array![0.0, 1.0]
  );
  for bad in BAD {
    for row in [[bad, 0.5], [0.5, bad]] {
      assert_eq!(
        copula.cdf(array![[0.3, 0.6], row].view()),
        Err(CopulaError::MarginalOutOfRange),
        "{row:?}"
      );
    }
  }
}

/// Marshall–Olkin's degenerate pairs, one shock rate at 1: `h` at `v ∈ {0, 1}` and its inverse at `y, v ∈ {0, 1}` finite.
#[test]
fn marshall_olkin_degenerate_pairs_stay_finite_on_the_edges() {
  for (alpha, beta) in [(1.0, 0.4), (0.7, 1.0)] {
    let copula = MarshallOlkin::new(alpha, beta);
    let h = copula
      .partial_derivative(array![[0.3, 0.0], [0.3, 1.0], [0.0, 0.0], [1.0, 1.0]].view())
      .unwrap();
    let u = copula
      .percent_point(
        array![0.0, 1.0, 0.0, 1.0, 0.4, 0.4].view(),
        array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0].view(),
      )
      .unwrap();
    assert!(
      h.iter().chain(&u).all(|x| x.is_finite()),
      "({alpha}, {beta}): {h} {u}"
    );
  }
}
