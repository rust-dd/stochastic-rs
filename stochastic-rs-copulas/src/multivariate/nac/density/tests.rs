use ndarray::Array2;

use crate::bivariate::BivariateFamily;
use crate::bivariate::clayton;
use crate::bivariate::gumbel;
use crate::multivariate::nac::NacNode;
use crate::multivariate::nac::NestedArchimedean;
use crate::traits::BivariateExt;
use crate::traits::MultivariateExt;

const U10: [f64; 10] = [0.3, 0.8, 0.55, 0.15, 0.65, 0.4, 0.9, 0.25, 0.5, 0.7];

fn node(theta: f64, leaves: Vec<usize>, children: Vec<NacNode>) -> NacNode {
  NacNode {
    theta,
    leaves,
    children,
  }
}

fn group(theta: f64, leaves: std::ops::Range<usize>) -> NacNode {
  NacNode::leaf_group(theta, leaves.collect())
}

/// `(…((0, 1)_{θ_k}, 2)_{θ_{k−1}}, …)_{θ_0}`: the deepest binary tree `fit` builds.
fn caterpillar(thetas: &[f64]) -> NacNode {
  let (last, outer) = thetas.split_last().unwrap();
  let mut tree = group(*last, 0..2);
  for (i, &theta) in outer.iter().rev().enumerate() {
    tree = node(theta, vec![i + 2], vec![tree]);
  }
  tree
}

fn density(family: BivariateFamily, root: NacNode, u: &[f64]) -> (f64, f64) {
  let nac = NestedArchimedean::new(family, root, u.len()).unwrap();
  let x = Array2::from_shape_vec((1, u.len()), u.to_vec()).unwrap();
  (
    nac.pdf(x.view()).unwrap()[0],
    nac.log_pdf(x.view()).unwrap()[0],
  )
}

/// The mismatches of `pdf` against `want` past 1e-12 relative, and of `log_pdf` against `ln pdf`.
fn misses(label: String, (pdf, log_pdf): (f64, f64), want: f64) -> Option<String> {
  let rel = (pdf / want - 1.0).abs();
  let log_gap = (log_pdf - pdf.ln()).abs();
  (!(rel <= 1e-12 && log_gap <= 1e-12))
    .then(|| format!("{label}: pdf {pdf:e} vs {want:e} (rel {rel:.1e}), log gap {log_gap:.1e}"))
}

/// Flat densities at `U10[..d]` against mpmath's exact multilinear derivative of `C` at 80 digits, which matches Hofert,
/// Mächler & McNeil (2012), Cor. 3.3, to 1e-79.
#[test]
fn flat_density_matches_the_60_digit_references() {
  let pins = [
    (
      BivariateFamily::Clayton,
      2.0,
      [
        0.46609503448158107,
        0.6384986945323375,
        0.16416427991333837,
        0.13802069806313186,
        0.06679456804551417,
      ],
    ),
    (
      BivariateFamily::Gumbel,
      2.0,
      [
        0.3986413913272548,
        0.5930699121417018,
        0.374855593705163,
        0.5375996071170147,
        0.23919081416084104,
      ],
    ),
    (
      BivariateFamily::Clayton,
      0.5,
      [
        0.8936957053891579,
        0.9886008019996739,
        0.7961504685001781,
        0.9020415244747009,
        1.091953870107605,
      ],
    ),
    (
      BivariateFamily::Gumbel,
      3.5,
      [
        0.05656448671967605,
        0.07996888467526415,
        0.014781332800915991,
        0.007058249296657603,
        3.756227018662004e-5,
      ],
    ),
  ];
  let mut failed = vec![];
  for (family, theta, wants) in pins {
    for (d, want) in [2, 3, 4, 6, 10].into_iter().zip(wants) {
      let got = density(family, group(theta, 0..d), &U10[..d]);
      failed.extend(misses(
        format!("{family:?} θ = {theta}, d = {d}"),
        got,
        want,
      ));
    }
  }
  assert!(failed.is_empty(), "{failed:#?}");
}

/// The same reference on Hofert & Pham's (2013) two- and three-level shapes (Thm 3.3, eq. (33)) and on `fit`'s binary
/// shapes down to nine levels, where their Remark 6.2 recursion carries the density.
#[test]
fn nested_density_matches_the_60_digit_references() {
  let trees = [
    (
      BivariateFamily::Gumbel,
      node(1.5, vec![0], vec![group(3.0, 1..3)]),
      3,
      0.6378154855495975,
    ),
    (
      BivariateFamily::Gumbel,
      node(1.5, vec![], vec![group(3.0, 0..2), group(2.5, 2..4)]),
      4,
      0.07130470884660796,
    ),
    (
      BivariateFamily::Clayton,
      node(1.0, vec![0], vec![group(2.0, 1..3), group(4.0, 3..5)]),
      5,
      0.021473179120931808,
    ),
    (
      BivariateFamily::Gumbel,
      caterpillar(&[1.2, 2.0, 3.5]),
      4,
      0.0779801319038448,
    ),
    (
      BivariateFamily::Clayton,
      node(
        0.8,
        vec![],
        vec![
          node(1.5, vec![0, 1], vec![group(4.0, 2..4)]),
          group(2.5, 4..6),
        ],
      ),
      6,
      0.03230623654384555,
    ),
    (
      BivariateFamily::Clayton,
      caterpillar(&[0.5, 1.0, 1.5, 2.0, 3.0]),
      6,
      0.17464458977331151,
    ),
    (
      BivariateFamily::Gumbel,
      caterpillar(&[1.1, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0, 3.5]),
      10,
      0.022224302927753504,
    ),
    (
      BivariateFamily::Gumbel,
      node(
        1.3,
        vec![],
        vec![
          node(2.0, vec![], vec![group(4.0, 0..2), group(3.0, 2..4)]),
          node(1.8, vec![], vec![group(3.5, 4..6), group(2.5, 6..8)]),
        ],
      ),
      8,
      0.0008303701151460644,
    ),
  ];
  let mut failed = vec![];
  for (family, root, d, want) in trees {
    let label = format!("{family:?} {root:?}");
    failed.extend(misses(label, density(family, root, &U10[..d]), want));
  }
  assert!(failed.is_empty(), "{failed:#?}");
}

/// Corners where `pdf` underflows or a coordinate sits within 1e-12 of 0 or 1: `log_pdf` holds mpmath's log-density.
#[test]
fn log_pdf_stays_finite_where_pdf_underflows() {
  let with = |i: usize, x: f64| {
    let mut u = U10.to_vec();
    u[i] = x;
    u
  };
  let mut deep = with(0, 1e-300);
  deep[9] = 1.0 - 1e-12;
  let three_level = node(
    0.8,
    vec![],
    vec![
      node(1.5, vec![0, 1], vec![group(4.0, 2..4)]),
      group(2.5, 4..6),
    ],
  );
  let cases = [
    (
      BivariateFamily::Clayton,
      group(5.0, 0..10),
      with(0, 1e-30),
      -3041.025386378624,
    ),
    (
      BivariateFamily::Clayton,
      three_level,
      with(2, 1e-150)[..6].to_vec(),
      -2674.3496519957766,
    ),
    (
      BivariateFamily::Gumbel,
      group(3.5, 0..10),
      with(3, 1e-200),
      -147.0215813092495,
    ),
    (
      BivariateFamily::Gumbel,
      caterpillar(&[1.1, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0, 3.5]),
      deep,
      -68.41257722840287,
    ),
    (
      BivariateFamily::Gumbel,
      node(1.5, vec![], vec![group(3.0, 0..2), group(2.5, 2..4)]),
      vec![1.0 - 1e-12, 1.0 - 2e-12, 0.55, 1.0 - 1e-15],
      -36.23061356725766,
    ),
  ];
  for (family, root, u, want) in cases {
    let (pdf, log_pdf) = density(family, root, &u);
    assert!(
      (log_pdf - want).abs() <= 1e-12 * want.abs(),
      "{u:?}: {log_pdf} vs {want}"
    );
    if want < -745.0 {
      assert_eq!(pdf, 0.0, "{u:?}");
    } else {
      assert!(
        (pdf.ln() - log_pdf).abs() <= 1e-12 * want.abs(),
        "{u:?}: pdf {pdf:e}"
      );
    }
  }
}

/// The density lives on the open cube: a coordinate at 0 or 1 is NaN, where a number would be a guess at a limit.
#[test]
fn a_coordinate_on_the_boundary_is_nan() {
  let x = ndarray::array![[0.0, 0.5, 0.6], [0.3, 1.0, 0.6], [0.3, 0.5, 0.6]];
  for family in [BivariateFamily::Clayton, BivariateFamily::Gumbel] {
    let nac =
      NestedArchimedean::new(family, node(1.5, vec![0], vec![group(3.0, 1..3)]), 3).unwrap();
    for values in [nac.pdf(x.view()).unwrap(), nac.log_pdf(x.view()).unwrap()] {
      assert!(
        values[0].is_nan() && values[1].is_nan() && values[2].is_finite(),
        "{family:?}: {values}"
      );
    }
  }
}

/// Equal parameters collapse a nesting to the flat copula, and a Gumbel root at `θ = 1` multiplies its children's densities.
#[test]
fn nesting_identities_hold_to_rounding() {
  let u = &U10[..5];
  for family in [BivariateFamily::Clayton, BivariateFamily::Gumbel] {
    let flat = density(family, group(2.0, 0..5), u).0;
    let nested = density(
      family,
      node(
        2.0,
        vec![4],
        vec![node(2.0, vec![0], vec![group(2.0, 1..4)])],
      ),
      u,
    )
    .0;
    assert!(
      (nested / flat - 1.0).abs() <= 1e-12,
      "{family:?}: {nested} vs {flat}"
    );
  }
  let product = density(
    BivariateFamily::Gumbel,
    node(1.0, vec![], vec![group(2.0, 0..2), group(3.0, 2..5)]),
    u,
  )
  .0;
  let factors = density(BivariateFamily::Gumbel, group(2.0, 0..2), &u[..2]).0
    * density(BivariateFamily::Gumbel, group(3.0, 0..3), &u[2..]).0;
  assert!(
    (product / factors - 1.0).abs() <= 1e-12,
    "{product} vs {factors}"
  );
}

/// At `d = 2` the flat copula is the pair copula, whose closed-form `pdf` the bivariate module carries.
#[test]
fn the_bivariate_case_is_the_pair_copula() {
  let x = ndarray::array![[0.3, 0.8], [0.05, 0.1], [0.9, 0.95], [0.5, 0.02]];
  let pairs = [
    (
      BivariateFamily::Clayton,
      clayton::Clayton::new(2.0).pdf(x.view()).unwrap(),
    ),
    (
      BivariateFamily::Gumbel,
      gumbel::Gumbel::new(2.0).pdf(x.view()).unwrap(),
    ),
  ];
  for (family, want) in pairs {
    let nac = NestedArchimedean::new(family, group(2.0, 0..2), 2).unwrap();
    let got = nac.pdf(x.view()).unwrap();
    for (g, w) in got.iter().zip(&want) {
      assert!((g / w - 1.0).abs() <= 1e-12, "{family:?}: {g} vs {w}");
    }
  }
}
