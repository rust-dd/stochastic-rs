//! The elliptical copula cdfs are deterministic: Genz–Bretz on Sobol points from `d = 3`, pinned against scipy at `1e-9`
//! and the trivariate orthant identity, and closed forms at `d = 2`, pinned against 50-digit mpmath values.

use std::time::Duration;
use std::time::Instant;

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::multivariate::t::TMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

/// `(nu, rho, u, v, C)` by 50-digit mpmath: the quantiles by Newton on the cdf, the density integrated through the
/// conditional `t_{ν+1}` law.
const T_PINS: [(f64, f64, f64, f64, f64); 42] = [
  (1.0, -0.95, 0.3, 0.6, 0.017041316339198472),
  (1.0, 0.95, 0.02, 0.03, 0.01867772588235791),
  (1.0, 0.95, 0.999999, 0.999998, 0.9999979522774426),
  (1.0, -0.5, 1e-06, 0.999, 2.501875931445304e-07),
  (1.0, 0.3, 1e-06, 2e-06, 5.253205655206393e-07),
  (1.0, 0.3, 0.77, 0.21, 0.16192724465788846),
  (2.5, -0.95, 0.3, 0.6, 0.014626527685051498),
  (2.5, 0.95, 0.02, 0.03, 0.018156860992789535),
  (2.5, 0.95, 0.999999, 0.999998, 0.9999979397974755),
  (2.5, -0.5, 1e-06, 0.999, 1.9432646558203377e-07),
  (2.5, 0.3, 1e-06, 2e-06, 3.395663885910207e-07),
  (2.5, 0.3, 0.77, 0.21, 0.1748416836230766),
  (4.0, -0.95, 0.3, 0.6, 0.013943360321661664),
  (4.0, 0.95, 0.02, 0.03, 0.017777626179270294),
  (4.0, 0.95, 0.999999, 0.999998, 0.9999979125886836),
  (4.0, -0.5, 1e-06, 0.999, 1.9717349851673056e-07),
  (4.0, 0.3, 1e-06, 2e-06, 2.2231798065983205e-07),
  (4.0, 0.3, 0.77, 0.21, 0.17873468057968833),
  (6.0, -0.95, 0.3, 0.6, 0.013556461858988978),
  (6.0, 0.95, 0.02, 0.03, 0.017488187369725794),
  (6.0, 0.95, 0.999999, 0.999998, 0.9999978736119465),
  (6.0, -0.5, 1e-06, 0.999, 2.3773494761887218e-07),
  (6.0, 0.3, 1e-06, 2e-06, 1.3207642503251111e-07),
  (6.0, 0.3, 0.77, 0.21, 0.18098163576732001),
  (7.3, -0.95, 0.3, 0.6, 0.013418084515947925),
  (7.3, 0.95, 0.02, 0.03, 0.01736982459862147),
  (7.3, 0.95, 0.999999, 0.999998, 0.9999978502704757),
  (7.3, -0.5, 1e-06, 0.999, 2.7125279713910804e-07),
  (7.3, 0.3, 1e-06, 2e-06, 9.720606143520026e-08),
  (7.3, 0.3, 0.77, 0.21, 0.18179211825848843),
  (30.0, -0.95, 0.3, 0.6, 0.012935372346231412),
  (30.0, 0.95, 0.02, 0.03, 0.01689148573666127),
  (30.0, 0.95, 0.999999, 0.999998, 0.9999976849416946),
  (30.0, -0.5, 1e-06, 0.999, 5.832925817168345e-07),
  (30.0, 0.3, 1e-06, 2e-06, 6.8513819268911725e-09),
  (30.0, 0.3, 0.77, 0.21, 0.18464469498964992),
  (30.5, -0.95, 0.3, 0.6, 0.01293283786813523),
  (30.5, 0.95, 0.02, 0.03, 0.016888698747627856),
  (30.5, 0.95, 0.999999, 0.999998, 0.9999976835662339),
  (30.5, -0.5, 1e-06, 0.999, 5.860080606818363e-07),
  (30.5, 0.3, 1e-06, 2e-06, 6.6580164029822676e-09),
  (30.5, 0.3, 0.77, 0.21, 0.18465976720811114),
];

/// `(rho, u, v, C)` by 50-digit mpmath, the density integrated through the conditional normal law.
const GAUSSIAN_PINS: [(f64, f64, f64, f64); 16] = [
  (-0.95, 0.3, 0.6, 0.012781131020882781),
  (-0.95, 1e-06, 2e-06, 2.1115805479410805e-195),
  (-0.95, 0.999999, 0.999998, 0.999997),
  (-0.95, 1e-06, 0.999, 6.248500918378766e-13),
  (-0.5, 0.3, 0.6, 0.10810931317508182),
  (-0.5, 1e-06, 2e-06, 1.8209233983356634e-22),
  (-0.5, 0.999999, 0.999998, 0.999997),
  (-0.5, 1e-06, 0.999, 7.603514413054294e-07),
  (0.3, 0.3, 0.6, 0.2198643732124294),
  (0.3, 1e-06, 2e-06, 5.349959274871683e-10),
  (0.3, 0.999999, 0.999998, 0.999997000534996),
  (0.3, 1e-06, 0.999, 9.999991594179398e-07),
  (0.95, 0.3, 0.6, 0.29971320707644106),
  (0.95, 1e-06, 2e-06, 5.886049458504109e-07),
  (0.95, 0.999999, 0.999998, 0.9999975886049459),
  (0.95, 1e-06, 0.999, 1e-06),
];

fn r3() -> Array2<f64> {
  array![[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]]
}

fn r5() -> Array2<f64> {
  array![
    [1.0, 0.5, 0.3, 0.2, 0.1],
    [0.5, 1.0, 0.4, 0.3, 0.2],
    [0.3, 0.4, 1.0, 0.5, 0.3],
    [0.2, 0.3, 0.5, 1.0, 0.4],
    [0.1, 0.2, 0.3, 0.4, 1.0]
  ]
}

/// Smallest eigenvalue `1.07e-3`, the floor a fit shrinks its estimate to.
fn near_singular() -> Array2<f64> {
  array![
    [1.0, 0.8692186302101551, 0.9226908554168762],
    [0.8692186302101551, 1.0, 0.9914916282469303],
    [0.9226908554168762, 0.9914916282469303, 1.0]
  ]
}

/// Smallest eigenvalue `1.00004e-3`; sweeping its third coordinate crosses an order switch that steps back `4.7e-4`.
fn near_singular_six() -> Array2<f64> {
  array![
    [
      1.0,
      0.8303107511128212,
      0.8342917329563071,
      0.5687202438299149,
      0.2465263354219746,
      0.8359215785022871
    ],
    [
      0.8303107511128212,
      1.0,
      0.7986542449090619,
      0.700091779667285,
      0.5907087958276408,
      0.7340604783370868
    ],
    [
      0.8342917329563071,
      0.7986542449090619,
      1.0,
      0.2939882394695861,
      0.6009888832668322,
      0.8387937148332812
    ],
    [
      0.5687202438299149,
      0.700091779667285,
      0.2939882394695861,
      1.0,
      0.20214646792765614,
      0.4945448814630901
    ],
    [
      0.2465263354219746,
      0.5907087958276408,
      0.6009888832668322,
      0.20214646792765614,
      1.0,
      0.5834898590157885
    ],
    [
      0.8359215785022871,
      0.7340604783370868,
      0.8387937148332812,
      0.4945448814630901,
      0.5834898590157885,
      1.0
    ]
  ]
}

fn pair(rho: f64) -> Array2<f64> {
  array![[1.0, rho], [rho, 1.0]]
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

/// The rule's largest absolute error against scipy's Genz–Bretz over 240 random correlations, `d` from 3 to 10 with the
/// smallest eigenvalue down to `1e-3`.
const QMC_ENVELOPE: f64 = 2.9e-4;

const U3: [[f64; 3]; 4] = [
  [0.3, 0.5, 0.7],
  [0.5, 0.5, 0.5],
  [0.9, 0.2, 0.6],
  [0.05, 0.95, 0.5],
];

#[test]
fn gaussian_cdf_matches_genz_bretz_references_and_the_orthant_identity() {
  let copula = GaussianMultivariate::try_new(r3()).unwrap();
  let u = Array2::from_shape_fn((4, 3), |(i, j)| U3[i][j]);
  let cdf = copula.cdf(u.view()).unwrap();
  for (got, want) in cdf
    .iter()
    .zip([0.2174712566, 0.2526175054, 0.1709963388, 0.0370124726])
  {
    assert!((got - want).abs() < 5e-4, "{got} vs {want}");
  }
  let exact =
    0.125 + (0.7_f64.asin() + 0.3_f64.asin() + 0.5_f64.asin()) / (4.0 * std::f64::consts::PI);
  assert!((cdf[1] - exact).abs() < 5e-4, "{} vs {exact}", cdf[1]);
  assert_eq!(
    cdf,
    copula.cdf(u.view()).unwrap(),
    "a cdf is a function of its inputs"
  );
  let five = GaussianMultivariate::try_new(r5()).unwrap();
  let u5 = array![
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.3, 0.6, 0.4, 0.7, 0.5],
    [0.9, 0.9, 0.1, 0.5, 0.5]
  ];
  for (got, want) in
    five
      .cdf(u5.view())
      .unwrap()
      .iter()
      .zip([0.1114520628, 0.0857255138, 0.0598675802])
  {
    assert!((got - want).abs() < 1e-3, "{got} vs {want}");
  }
}

/// At `d = 2` both Gaussian copulas evaluate the same bivariate normal integral.
#[test]
fn bivariate_gaussian_cdf_is_the_bivariate_copulas() {
  for (rho, u, v, want) in GAUSSIAN_PINS {
    let q = array![[u, v]];
    let mv = GaussianMultivariate::try_new(pair(rho))
      .unwrap()
      .cdf(q.view())
      .unwrap()[0];
    let bv = GaussianCopula::new(rho).cdf(q.view()).unwrap()[0];
    assert_eq!(mv, bv, "rho={rho} u={u} v={v}");
    assert!(
      (mv - want).abs() <= 1e-12,
      "rho={rho} u={u} v={v}: {mv} vs {want}"
    );
  }
}

#[test]
fn t_cdf_matches_genz_bretz_references() {
  let copula = TMultivariate::try_new(r3(), 5.0).unwrap();
  let u = Array2::from_shape_fn((4, 3), |(i, j)| U3[i][j]);
  let cdf = copula.cdf(u.view()).unwrap();
  for (got, want) in cdf
    .iter()
    .zip([0.213618174, 0.252617512, 0.165936548, 0.035537656])
  {
    assert!((got - want).abs() < 1e-3, "{got} vs {want}");
  }
  assert_eq!(cdf, copula.cdf(u.view()).unwrap());
}

/// At `d = 2` both t copulas evaluate the same closed form: Dunnett–Sobel for an integer `ν`, Genz's generalized
/// Plackett integral otherwise.
#[test]
fn bivariate_t_cdf_matches_50_digit_references() {
  for (nu, rho, u, v, want) in T_PINS {
    let q = array![[u, v]];
    let mv = TMultivariate::try_new(pair(rho), nu)
      .unwrap()
      .cdf(q.view())
      .unwrap()[0];
    let bv = TCopula::new(rho, nu).cdf(q.view()).unwrap()[0];
    assert_eq!(mv, bv, "nu={nu} rho={rho} u={u} v={v}");
    assert!(
      (bv - want).abs() <= 1e-12,
      "nu={nu} rho={rho} u={u} v={v}: {bv} vs {want}"
    );
  }
}

/// `(ρ, ν, u, v, C)` where `H` steps across the march's first panel: `¼ + asin ρ/(2π)` at `(½, ½)`, else 40-digit mpmath.
const STEP_PINS: [(f64, f64, f64, f64, f64); 7] = [
  (-0.999999, 0.3, 0.5, 0.5, 0.0002250790977991068),
  (0.999999, 1.5, 0.5, 0.5, 0.4997749209022009),
  (0.999999, 30.5, 0.5, 0.5, 0.4997749209022009),
  (0.999999, 200.5, 0.5, 0.5, 0.4997749209022009),
  (0.99999999, 2.5, 0.5, 0.5, 0.49997749209202075),
  (
    0.9988962907871344,
    2.5,
    0.5,
    0.49125886598344887,
    0.48719996799382487,
  ),
  (
    0.9999999611070157,
    0.7,
    0.8995983323300545,
    0.9,
    0.8995974422293114,
  ),
];

#[test]
fn the_t_cdf_resolves_a_step_of_h_in_the_first_panel() {
  for (rho, nu, u, v, want) in STEP_PINS {
    let got = TCopula::new(rho, nu).cdf(array![[u, v]].view()).unwrap()[0];
    assert!(
      (got - want).abs() <= 1e-12 * want,
      "nu={nu} rho={rho} u={u} v={v}: {got} vs {want}"
    );
  }
}

/// Two noise-free cells near `ρ = −1` that need fine panels where the integrand vanishes: within `1e-12` of mpmath (Owen's
/// form, the conditional integral), each in under the 500 ms the large-`ν` rows allow.
#[test]
fn the_cells_near_the_countermonotone_limit_return_promptly() {
  let start = Instant::now();
  let gaussian = GaussianCopula::new(-0.9997182188114035)
    .cdf(array![[0.19911836204432967, 0.5143446219176471]].view())
    .unwrap()[0];
  let gaussian_took = start.elapsed();
  let start = Instant::now();
  let t = TCopula::new(-0.9999999999999921, 600.5)
    .cdf(array![[0.999999999999989, 0.25476551016987653]].view())
    .unwrap()[0];
  let t_took = start.elapsed();
  for (got, want, took) in [
    (gaussian, 2.5609941874701457e-258, gaussian_took),
    (t, 0.25476551016986554, t_took),
  ] {
    assert!((got - want).abs() <= 1e-12 * want, "{got:e} vs {want:e}");
    assert!(took < Duration::from_millis(500), "{took:?}");
  }
}

/// A step back is the difference of two errors where a query's order switches, so monotonicity holds to about twice the
/// envelope, swept along every coordinate; the Fréchet bound holds to the envelope itself.
#[test]
fn cdfs_are_monotone_and_bounded() {
  let levels = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95];
  let grid = Array2::from_shape_fn((343, 3), |(i, j)| levels[(i / 7usize.pow(j as u32)) % 7]);
  let switch = [
    0.49844094557537166,
    0.45595156790720204,
    0.0,
    0.8703508794857825,
    0.3642072203540029,
    0.3165082622140987,
  ];
  let mut cases = vec![(r3(), grid.clone()), (near_singular(), grid)];
  for j in 0..3 {
    cases.push((near_singular(), sweep(&[0.3, 0.6, 0.85], j, 0.01, 0.99, 99)));
  }
  cases.push((near_singular_six(), sweep(&switch, 2, 0.55, 0.65, 101)));
  for (k, (r, u)) in cases.into_iter().enumerate() {
    let g = GaussianMultivariate::try_new(r.clone()).unwrap();
    let t = TMultivariate::try_new(r, 4.0).unwrap();
    for c in [g.cdf(u.view()).unwrap(), t.cdf(u.view()).unwrap()] {
      if k >= 2 {
        assert!(
          c.windows(2)
            .into_iter()
            .all(|w| w[0] - w[1] <= 2.0 * QMC_ENVELOPE),
          "case {k}: {c:?}"
        );
      }
      for (v, row) in c.iter().zip(u.rows()) {
        let bound = row.iter().copied().fold(1.0, f64::min);
        assert!((0.0..=bound + QMC_ENVELOPE).contains(v), "{v} at {row}");
      }
    }
  }
}

/// Uncorrelated coordinates leave exact zeros in `L`: an underflowing factor gives 0, not `0 · ∞`, so the value is finite
/// (the product, for the Gaussian), and at `u₃ = ½` both copulas are `C₁₂(u₁, u₂) / 2` exactly.
#[test]
fn a_block_correlation_matrix_gives_the_product_not_nan() {
  for (rho, nu, u1, u2) in [
    (0.999, None, 0.9, 0.2),
    (0.999, None, 0.95, 0.05),
    (0.999, None, 0.999, 0.005),
    (0.999, Some(4.0), 0.9, 0.2),
    (0.999, Some(4.0), 0.9, 0.3),
    (0.99, Some(4.0), 0.999, 0.005),
  ] {
    let r = array![[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let q = array![[u1, u2, 0.5]];
    let pair = array![[u1, u2]];
    let (got, want) = match nu {
      None => (
        GaussianMultivariate::try_new(r)
          .unwrap()
          .cdf(q.view())
          .unwrap()[0],
        GaussianCopula::new(rho).cdf(pair.view()).unwrap()[0] / 2.0,
      ),
      Some(nu) => (
        TMultivariate::try_new(r, nu)
          .unwrap()
          .cdf(q.view())
          .unwrap()[0],
        TCopula::new(rho, nu).cdf(pair.view()).unwrap()[0] / 2.0,
      ),
    };
    assert!(
      (got - want).abs() < 1e-4,
      "rho={rho} nu={nu:?} ({u1}, {u2}): {got} vs {want}"
    );
  }
  let tiny = GaussianMultivariate::try_new(Array2::eye(3))
    .unwrap()
    .cdf(array![[1e-320, 0.5, 0.5]].view())
    .unwrap()[0];
  assert!((0.0..=1e-300).contains(&tiny), "{tiny}");
}

/// A coordinate at 0 gives 0 and one at 1 drops out: what is left is the lower-dimensional copula's own value.
#[test]
fn a_zero_coordinate_gives_zero_and_a_unit_coordinate_drops_out() {
  let q2 = array![
    [0.0, 0.4],
    [0.4, 0.0],
    [1.0, 0.4],
    [0.4, 1.0],
    [1.0, 1.0],
    [0.0, 1.0]
  ];
  let want2 = vec![0.0, 0.0, 0.4, 0.4, 1.0, 0.0];
  let q3 = array![
    [0.3, 0.0, 0.7],
    [0.3, 1.0, 0.7],
    [1.0, 0.4, 0.7],
    [0.3, 0.4, 1.0],
    [1.0, 1.0, 0.7],
    [1.0, 1.0, 1.0]
  ];
  let r = r3();
  let gaussian = |i: usize, j: usize, a: f64, b: f64| {
    GaussianCopula::new(r[[i, j]])
      .cdf(array![[a, b]].view())
      .unwrap()[0]
  };
  let g2 = GaussianMultivariate::try_new(pair(0.5)).unwrap();
  assert_eq!(g2.cdf(q2.view()).unwrap().to_vec(), want2);
  let g3 = GaussianMultivariate::try_new(r3()).unwrap();
  let want3 = vec![
    0.0,
    gaussian(0, 2, 0.3, 0.7),
    gaussian(1, 2, 0.4, 0.7),
    gaussian(0, 1, 0.3, 0.4),
    0.7,
    1.0,
  ];
  assert_eq!(g3.cdf(q3.view()).unwrap().to_vec(), want3);
  for nu in [4.0, 4.5] {
    let t = |i: usize, j: usize, a: f64, b: f64| {
      TCopula::new(r[[i, j]], nu)
        .cdf(array![[a, b]].view())
        .unwrap()[0]
    };
    let t2 = TMultivariate::try_new(pair(0.5), nu).unwrap();
    assert_eq!(t2.cdf(q2.view()).unwrap().to_vec(), want2, "nu={nu}");
    let t3 = TMultivariate::try_new(r3(), nu).unwrap();
    let want3 = vec![
      0.0,
      t(0, 2, 0.3, 0.7),
      t(1, 2, 0.4, 0.7),
      t(0, 1, 0.3, 0.4),
      0.7,
      1.0,
    ];
    assert_eq!(t3.cdf(q3.view()).unwrap().to_vec(), want3, "nu={nu}");
  }
}

/// A unit coordinate sorts last and drops out: the trivariate copula's value bit for bit at any position, whenever every
/// conditional variance clears MVSORT's `1e-10·i` (below it the row falls back to the unordered estimator).
#[test]
fn a_unit_coordinate_drops_out_bit_for_bit_at_any_position() {
  let r = r5();
  let r4 = Array2::from_shape_fn((4, 4), |(i, j)| r[[i, j]]);
  let open = [0.3, 0.6, 0.4];
  for unit in [0, 1, 3] {
    let keep: Vec<usize> = (0..4).filter(|&i| i != unit).collect();
    let sub = Array2::from_shape_fn((3, 3), |(i, j)| r4[[keep[i], keep[j]]]);
    let mut four = vec![0.0; 4];
    four[unit] = 1.0;
    for (k, &i) in keep.iter().enumerate() {
      four[i] = open[k];
    }
    let four = Array2::from_shape_vec((1, 4), four).unwrap();
    let three = Array2::from_shape_vec((1, 3), open.to_vec()).unwrap();
    assert_eq!(
      GaussianMultivariate::try_new(r4.clone())
        .unwrap()
        .cdf(four.view())
        .unwrap(),
      GaussianMultivariate::try_new(sub.clone())
        .unwrap()
        .cdf(three.view())
        .unwrap(),
      "unit coordinate {unit}"
    );
    assert_eq!(
      TMultivariate::try_new(r4.clone(), 5.0)
        .unwrap()
        .cdf(four.view())
        .unwrap(),
      TMultivariate::try_new(sub, 5.0)
        .unwrap()
        .cdf(three.view())
        .unwrap(),
      "unit coordinate {unit}"
    );
  }
}
