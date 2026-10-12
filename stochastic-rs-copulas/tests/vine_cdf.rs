//! The vine cdfs are Genz's separation of variables through the Rosenblatt map their samplers walk: deterministic, about
//! ten times their measured error from the bivariate closed forms and Joe's mixing integrals, exact at independence.
//! Reference: Joe, H. (1996), "Families of m-variate distributions with given margins and m(m−1)/2 bivariate dependence parameters", IMS Lecture Notes – Monograph Series 28, 120–141, DOI 10.1214/lnms/1215452614 (eqs. (2.1), (2.2))

use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use ndarray::s;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::cvine::CVine;
use stochastic_rs_copulas::multivariate::dvine::DVine;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::multivariate::rvine::RVine;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

type Pins3 = &'static [([f64; 3], f64)];

type Pins4 = &'static [([f64; 4], f64)];

/// The largest error over the grids the bands come from, the four-dimensional D-vine's at `(0.95, 0.95, 0.95, 0.25)`.
const ENVELOPE: f64 = 3.1e-4;

fn moderate() -> Vec<Vec<PairCopula>> {
  vec![
    vec![
      PairCopula::Gaussian { rho: 0.4 },
      PairCopula::Clayton { theta: 2.0 },
    ],
    vec![PairCopula::Frank { theta: 3.0 }],
  ]
}

fn mixed() -> Vec<Vec<PairCopula>> {
  vec![
    vec![
      PairCopula::StudentT { rho: 0.5, nu: 4.0 },
      PairCopula::Frank { theta: -3.0 },
    ],
    vec![PairCopula::Gaussian { rho: 0.4 }],
  ]
}

/// Kendall's τ near 0.8 on every edge.
fn strong() -> Vec<Vec<PairCopula>> {
  vec![
    vec![
      PairCopula::Clayton { theta: 8.0 },
      PairCopula::Gaussian { rho: 0.95 },
    ],
    vec![PairCopula::Frank { theta: 18.2 }],
  ]
}

/// The conditioning pair of Joe's double integral (tree 1's middle edge of the D-vine, its first of the C-vine) is Frank.
fn four(structure: char) -> Vec<Vec<PairCopula>> {
  let (gaussian, frank) = (
    PairCopula::Gaussian { rho: 0.6 },
    PairCopula::Frank { theta: 4.0 },
  );
  let first = if structure == 'D' {
    [gaussian, frank]
  } else {
    [frank, gaussian]
  };
  vec![
    vec![first[0], first[1], PairCopula::Clayton { theta: 2.0 }],
    vec![
      PairCopula::Frank { theta: 2.0 },
      PairCopula::Gaussian { rho: 0.3 },
    ],
    vec![PairCopula::Clayton { theta: 1.0 }],
  ]
}

/// `(u, C)` by Joe's eq. (2.1) in mpmath at 36 digits; the three-dimensional C-vine is the D-vine `2–1–3`.
const MODERATE_D: Pins3 = &[
  ([0.3, 0.6, 0.8], 0.22790327199148247),
  ([0.1, 0.5, 0.9], 0.0771401779758453),
  ([0.7, 0.2, 0.4], 0.16318209812436663),
  ([0.95, 0.9, 0.05], 0.04988666428328952),
  ([0.5, 0.5, 0.5], 0.2720335175762624),
  ([0.05, 0.1, 0.07], 0.012744793395126318),
  ([0.9, 0.95, 0.85], 0.7693606783444222),
  ([0.6, 0.3, 0.97], 0.23279390657903098),
  ([0.2, 0.25, 0.22], 0.07669954789180206),
  ([0.75, 0.7, 0.8], 0.5298175983340554),
];

const MODERATE_C: Pins3 = &[
  ([0.3, 0.6, 0.8], 0.22997496903501077),
  ([0.1, 0.5, 0.9], 0.07739257688208538),
  ([0.7, 0.2, 0.4], 0.13893190197551253),
  ([0.95, 0.9, 0.05], 0.04969721760092934),
  ([0.5, 0.5, 0.5], 0.2720335175762624),
  ([0.05, 0.1, 0.07], 0.014603580765187647),
  ([0.9, 0.95, 0.85], 0.766177271921609),
  ([0.6, 0.3, 0.97], 0.23259721619421606),
  ([0.2, 0.25, 0.22], 0.07985471237221675),
  ([0.75, 0.7, 0.8], 0.5296267726797196),
];

const MIXED_D: Pins3 = &[
  ([0.3, 0.6, 0.8], 0.19250842746269262),
  ([0.1, 0.5, 0.9], 0.0739200622477341),
  ([0.7, 0.2, 0.4], 0.030875120772059884),
  ([0.95, 0.9, 0.05], 0.03692335926601171),
  ([0.5, 0.5, 0.5], 0.1288011429060183),
  ([0.05, 0.1, 0.07], 0.0008372892284036277),
  ([0.9, 0.95, 0.85], 0.7450696473373712),
  ([0.6, 0.3, 0.97], 0.2325347293529101),
  ([0.2, 0.25, 0.22], 0.012252266434392102),
  ([0.75, 0.7, 0.8], 0.4552106208535911),
];

const MIXED_C: Pins3 = &[
  ([0.3, 0.6, 0.8], 0.16551566296734493),
  ([0.1, 0.5, 0.9], 0.0648514119845877),
  ([0.7, 0.2, 0.4], 0.06732662980990882),
  ([0.95, 0.9, 0.05], 0.0419355906814844),
  ([0.5, 0.5, 0.5], 0.1288011429060183),
  ([0.05, 0.1, 0.07], 0.0005571096266141769),
  ([0.9, 0.95, 0.85], 0.7366764961359552),
  ([0.6, 0.3, 0.97], 0.23747979058950194),
  ([0.2, 0.25, 0.22], 0.010497125688081078),
  ([0.75, 0.7, 0.8], 0.46324228431827164),
];

const STRONG_D: Pins3 = &[
  ([0.3, 0.6, 0.8], 0.29985628324110786),
  ([0.1, 0.5, 0.9], 0.09999996812504572),
  ([0.7, 0.2, 0.4], 0.1986601865278171),
  ([0.95, 0.9, 0.05], 0.04999999999999969),
  ([0.5, 0.5, 0.5], 0.4447572155075418),
  ([0.05, 0.1, 0.07], 0.04662125512031499),
  ([0.9, 0.95, 0.85], 0.8387784120270381),
  ([0.6, 0.3, 0.97], 0.29985628629976835),
  ([0.2, 0.25, 0.22], 0.18401989562690613),
  ([0.75, 0.7, 0.8], 0.6639238785511651),
];

const STRONG_C: Pins3 = &[
  ([0.3, 0.6, 0.8], 0.299856087880913),
  ([0.1, 0.5, 0.9], 0.09999996812504572),
  ([0.7, 0.2, 0.4], 0.19986649004443527),
  ([0.95, 0.9, 0.05], 0.0499999999999992),
  ([0.5, 0.5, 0.5], 0.4447572155075418),
  ([0.05, 0.1, 0.07], 0.04307042532456269),
  ([0.9, 0.95, 0.85], 0.8387050849810183),
  ([0.6, 0.3, 0.97], 0.2998562862997685),
  ([0.2, 0.25, 0.22], 0.17270915325647154),
  ([0.75, 0.7, 0.8], 0.6642267981711698),
];

/// `(u, C)` by Joe's eq. (2.2), nested mpmath quadrature at 20 digits, matching scipy's `dblquad` to `1e-14`.
const FOUR_D: Pins4 = &[
  ([0.3, 0.6, 0.8, 0.5], 0.2241393760058356),
  ([0.9, 0.2, 0.45, 0.7], 0.1523641814194358),
  ([0.5, 0.5, 0.5, 0.5], 0.24561229891173939),
  ([0.05, 0.95, 0.35, 0.6], 0.03893866910168169),
  ([0.7, 0.8, 0.9, 0.85], 0.5708521926452224),
  ([0.2, 0.3, 0.25, 0.4], 0.08046798720613406),
];

const FOUR_C: Pins4 = &[
  ([0.3, 0.6, 0.8, 0.5], 0.2310833688091219),
  ([0.9, 0.2, 0.45, 0.7], 0.14724208163706112),
  ([0.5, 0.5, 0.5, 0.5], 0.24595018650048933),
  ([0.05, 0.95, 0.35, 0.6], 0.04217811561801062),
  ([0.7, 0.8, 0.9, 0.85], 0.5699114464942632),
  ([0.2, 0.3, 0.25, 0.4], 0.0808184341405294),
];

fn assert_pins<const D: usize>(label: &str, cdf: Array1<f64>, pins: &[([f64; D], f64)], band: f64) {
  for ((u, want), got) in pins.iter().zip(&cdf) {
    assert!(
      (got - want).abs() < band,
      "{label} at {u:?}: {got} vs {want}"
    );
  }
}

fn rows<const D: usize>(pins: &[([f64; D], f64)]) -> Array2<f64> {
  Array2::from_shape_fn((pins.len(), D), |(r, c)| pins[r].0[c])
}

/// `n` points of `u` with coordinate `j` from `from` to `to` and the rest at `base`.
fn sweep(base: &[f64], j: usize, from: f64, to: f64, n: usize) -> Array2<f64> {
  Array2::from_shape_fn((n, base.len()), |(k, i)| {
    if i == j {
      from + (to - from) * k as f64 / (n - 1) as f64
    } else {
      base[i]
    }
  })
}

/// At `d = 2` the rule is one-dimensional, its bias `(g(0) − g(1))/(2N)` reaching `1/(2N) = 1.22e-4` for a `g` that
/// falls from 1 to 0; the band is ten times that bound.
#[test]
fn two_dimensional_vines_match_the_bivariate_closed_forms() {
  let levels = [0.02, 0.25, 0.5, 0.75, 0.98];
  let u = Array2::from_shape_fn((25, 2), |(r, c)| levels[if c == 0 { r / 5 } else { r % 5 }]);
  let band = 10.0 / 8192.0;
  let cases = [
    (
      PairCopula::Clayton { theta: 2.0 },
      Clayton::new(2.0).cdf(u.view()).unwrap(),
    ),
    (
      PairCopula::Gaussian { rho: 0.6 },
      GaussianCopula::new(0.6).cdf(u.view()).unwrap(),
    ),
    (
      PairCopula::Frank { theta: 4.0 },
      Frank::new(4.0).cdf(u.view()).unwrap(),
    ),
    (
      PairCopula::StudentT { rho: 0.5, nu: 4.0 },
      TCopula::new(0.5, 4.0).cdf(u.view()).unwrap(),
    ),
  ];
  for (pair, exact) in cases {
    let qmc = DVine::new(2, vec![vec![pair]])
      .unwrap()
      .cdf(u.view())
      .unwrap();
    for (row, (got, want)) in u.rows().into_iter().zip(qmc.iter().zip(&exact)) {
      assert!(
        (got - want).abs() < band,
        "{pair:?} at {row}: {got} vs {want}"
      );
    }
    let cvine = CVine::new(2, vec![vec![pair]]).unwrap();
    assert_eq!(qmc, cvine.cdf(u.view()).unwrap(), "{pair:?}");
  }
  let dvine = DVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
  assert_eq!(
    dvine.cdf(u.view()).unwrap(),
    dvine.cdf(u.view()).unwrap(),
    "a cdf is a function of its inputs"
  );
}

/// Each band is about ten times the largest error over a 729-point grid, `{0.02, 0.1, …, 0.98}³`, against Joe's
/// integral in double precision: `2.07e-4`, `1.31e-4` and `2.83e-4`.
#[test]
fn three_dimensional_vines_match_joes_integral() {
  for (label, trees, dvine, cvine, band) in [
    ("moderate", moderate(), MODERATE_D, MODERATE_C, 2.1e-3),
    ("mixed", mixed(), MIXED_D, MIXED_C, 1.3e-3),
    ("strong", strong(), STRONG_D, STRONG_C, 2.8e-3),
  ] {
    let d = DVine::new(3, trees.clone()).unwrap();
    let c = CVine::new(3, trees).unwrap();
    assert_pins(label, d.cdf(rows(dvine).view()).unwrap(), dvine, band);
    assert_pins(label, c.cdf(rows(cvine).view()).unwrap(), cvine, band);
  }
}

/// The band is about ten times the largest error over the 625-point grid `{0.05, 0.25, 0.5, 0.75, 0.95}⁴`, `3.08e-4`.
#[test]
fn four_dimensional_vines_match_joes_double_integral() {
  let band = 3.1e-3;
  let d = DVine::new(4, four('D')).unwrap();
  let c = CVine::new(4, four('C')).unwrap();
  assert_pins("D-vine", d.cdf(rows(FOUR_D).view()).unwrap(), FOUR_D, band);
  assert_pins("C-vine", c.cdf(rows(FOUR_C).view()).unwrap(), FOUR_C, band);
  let r = RVine::from_cvine(c.clone());
  assert_eq!(
    r.cdf(rows(FOUR_C).view()).unwrap(),
    c.cdf(rows(FOUR_C).view()).unwrap()
  );
}

/// Every `e_i` is `u_i` at independence and the point count is a power of two, so the mean is the product itself.
#[test]
fn independence_vines_give_the_product_exactly() {
  let u3 = array![[0.2, 0.4, 0.7], [0.5, 0.5, 0.5], [0.9, 0.3, 0.6]];
  let u4 = array![[0.2, 0.4, 0.7, 0.35], [0.95, 0.15, 0.6, 0.8]];
  for u in [u3, u4] {
    let dim = u.ncols();
    for cdf in [
      DVine::independence(dim).unwrap().cdf(u.view()).unwrap(),
      CVine::independence(dim).unwrap().cdf(u.view()).unwrap(),
      RVine::independence(dim).unwrap().cdf(u.view()).unwrap(),
    ] {
      for (row, got) in u.rows().into_iter().zip(cdf.iter()) {
        let want = row.iter().product::<f64>();
        assert!((got - want).abs() <= 1e-15, "{row}: {got} vs {want}");
      }
    }
  }
}

/// A zero coordinate gives 0 and the unit corner 1; a unit last coordinate leaves the vine of the first two bit for bit,
/// and a unit coordinate elsewhere their copula within the envelope; outside `[0, 1]` is `MarginalOutOfRange`.
#[test]
fn zero_and_unit_coordinates() {
  let trees = moderate();
  let d = DVine::new(3, trees.clone()).unwrap();
  let c = CVine::new(3, trees.clone()).unwrap();
  let edge = array![
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]
  ];
  assert_eq!(d.cdf(edge.view()).unwrap().to_vec(), [0.0, 0.0, 0.0, 1.0]);
  assert_eq!(c.cdf(edge.view()).unwrap().to_vec(), [0.0, 0.0, 0.0, 1.0]);
  let last = array![[0.3, 0.6, 1.0], [0.9, 0.2, 1.0]];
  let first_two = last.slice(s![.., ..2]).to_owned();
  let pair = vec![vec![trees[0][0]]];
  assert_eq!(
    d.cdf(last.view()).unwrap(),
    DVine::new(2, pair.clone())
      .unwrap()
      .cdf(first_two.view())
      .unwrap()
  );
  assert_eq!(
    c.cdf(last.view()).unwrap(),
    CVine::new(2, pair).unwrap().cdf(first_two.view()).unwrap()
  );
  let d_23 = d.cdf(array![[1.0, 0.6, 0.8]].view()).unwrap()[0];
  let c_13 = c.cdf(array![[0.6, 1.0, 0.8]].view()).unwrap()[0];
  let clayton = Clayton::new(2.0).cdf(array![[0.6, 0.8]].view()).unwrap()[0];
  assert!((d_23 - clayton).abs() < ENVELOPE, "{d_23} vs {clayton}");
  assert!((c_13 - clayton).abs() < ENVELOPE, "{c_13} vs {clayton}");
  for bad in [
    array![[-0.1, 0.5, 0.5]],
    array![[0.5, 1.1, 0.5]],
    array![[0.5, f64::NAN, 0.5]],
  ] {
    assert!(matches!(
      d.cdf(bad.view()),
      Err(CopulaError::MarginalOutOfRange)
    ));
    assert!(matches!(
      c.cdf(bad.view()),
      Err(CopulaError::MarginalOutOfRange)
    ));
  }
}

/// Swept along every coordinate, and on a grid at `d = 4`, the values never step back or leave the Fréchet bounds by
/// more than the envelope.
#[test]
fn cdfs_are_monotone_and_within_the_frechet_bounds() {
  let mut evaluated = Vec::new();
  for trees in [moderate(), strong()] {
    let d = DVine::new(3, trees.clone()).unwrap();
    let c = CVine::new(3, trees).unwrap();
    for j in 0..3 {
      let u = sweep(&[0.3, 0.6, 0.85], j, 0.02, 0.98, 13);
      for cdf in [d.cdf(u.view()).unwrap(), c.cdf(u.view()).unwrap()] {
        for w in cdf.windows(2) {
          assert!(w[0] - w[1] <= ENVELOPE, "{cdf} along {j} from {}", u.row(0));
        }
        evaluated.push((cdf, u.clone()));
      }
    }
  }
  let levels = [0.1, 0.5, 0.9];
  let grid = Array2::from_shape_fn((81, 4), |(r, c)| levels[(r / 3usize.pow(c as u32)) % 3]);
  for cdf in [
    DVine::new(4, four('D')).unwrap().cdf(grid.view()).unwrap(),
    CVine::new(4, four('C')).unwrap().cdf(grid.view()).unwrap(),
  ] {
    for r in 0..81 {
      for c in (0..4).filter(|&c| (r / 3usize.pow(c)) % 3 < 2) {
        let next = r + 3usize.pow(c);
        let back = cdf[r] - cdf[next];
        assert!(
          back <= ENVELOPE,
          "{} to {}: {back}",
          grid.row(r),
          grid.row(next)
        );
      }
    }
    evaluated.push((cdf, grid.clone()));
  }
  for (cdf, u) in &evaluated {
    for (v, row) in cdf.iter().zip(u.rows()) {
      let upper = row.iter().copied().fold(1.0, f64::min);
      let lower = (row.sum() - (row.len() - 1) as f64).max(0.0);
      assert!(
        (lower - ENVELOPE..=upper + ENVELOPE).contains(v),
        "{v} at {row}"
      );
    }
  }
}
