//! The conditional inverse behind every bivariate sampler: it converges for every valid parameter, answers NaN outside
//! the unit square, rejects unequal lengths, and its closed forms match 50-digit references and round-trip `h`.

use ndarray::Array1;
use ndarray::Axis;
use ndarray::array;
use ndarray::stack;
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
use stochastic_rs_copulas::traits::BivariateExt;

type Family = (&'static str, Box<dyn BivariateExt + Send>);

fn with_theta<C: BivariateExt>(mut copula: C, theta: f64) -> C {
  copula.set_theta(theta);
  copula
}

/// The fifteen families at the stream fixture's parameters; Independence, which the fixture lacks, at `θ = 0`.
fn families() -> Vec<Family> {
  vec![
    ("amh", Box::new(with_theta(Amh::new(), 0.5))),
    ("bb1", Box::new(Bb1::new(Some(0.8), Some(1.6), None))),
    ("bb7", Box::new(Bb7::new(Some(1.7), Some(0.9), None))),
    ("clayton", Box::new(with_theta(Clayton::new(), 2.0))),
    ("fgm", Box::new(with_theta(Fgm::new(), 0.5))),
    ("frank", Box::new(Frank::new(Some(4.0), None))),
    ("galambos", Box::new(with_theta(Galambos::new(), 1.5))),
    ("gaussian", Box::new(with_theta(GaussianCopula::new(), 0.5))),
    ("gumbel", Box::new(Gumbel::new(Some(2.0), None))),
    (
      "husler_reiss",
      Box::new(with_theta(HuslerReiss::new(), 1.5)),
    ),
    (
      "independence",
      Box::new(with_theta(Independence::new(), 0.0)),
    ),
    ("joe", Box::new(with_theta(Joe::new(), 2.0))),
    (
      "marshall_olkin",
      Box::new(MarshallOlkin::with_alpha_beta(0.3, 0.6)),
    ),
    ("plackett", Box::new(with_theta(Plackett::new(), 3.0))),
    ("t_copula", Box::new(with_theta(TCopula::with_nu(4.0), 0.5))),
  ]
}

#[test]
fn every_family_samples_two_thousand_draws_for_fifty_seeds() {
  std::thread::scope(|scope| {
    for (name, copula) in families() {
      scope.spawn(move || {
        for seed in 0..50 {
          let uv = copula
            .sample_with_seed(2_000, seed)
            .unwrap_or_else(|e| panic!("{name} seed {seed}: {e}"));
          assert!(
            uv.iter().all(|x| (0.0..=1.0).contains(x)),
            "{name} seed {seed} left the unit square"
          );
        }
      });
    }
  });
}

/// Levels and conditioning values a uniform draw can reach, where Joe's h is flat enough to cost Brent 150 iterations.
#[test]
fn the_numerical_inverse_converges_at_extreme_levels() {
  let extremes = [
    1e-15,
    1e-12,
    1e-6,
    0.5,
    1.0 - 1e-6,
    1.0 - 1e-12,
    1.0 - 1e-15,
  ];
  let y = Array1::from_iter(extremes.iter().flat_map(|&y| extremes.map(|_| y)));
  let v = Array1::from_iter(extremes.iter().flat_map(|_| extremes));
  let numerical: [Family; 4] = [
    ("galambos", Box::new(with_theta(Galambos::new(), 1.5))),
    ("gumbel", Box::new(Gumbel::new(Some(2.0), None))),
    (
      "husler_reiss",
      Box::new(with_theta(HuslerReiss::new(), 1.5)),
    ),
    ("joe", Box::new(with_theta(Joe::new(), 2.0))),
  ];
  for (name, copula) in numerical {
    let u = copula
      .percent_point(&y, &v)
      .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert!(u.iter().all(|u| (0.0..=1.0).contains(u)), "{name}: {u}");
  }
}

/// Gumbel's and Joe's old h underflowed to `0·∞` near `u, v = 1` from `θ ≈ 30` on, so their samplers failed at these θ.
#[test]
fn gumbel_and_joe_sample_at_high_dependence() {
  std::thread::scope(|scope| {
    for theta in [50.0, 100.0, 200.0] {
      let pair: [Family; 2] = [
        ("gumbel", Box::new(Gumbel::new(Some(theta), None))),
        ("joe", Box::new(with_theta(Joe::new(), theta))),
      ];
      for (name, copula) in pair {
        scope.spawn(move || {
          for seed in 0..20 {
            let uv = copula
              .sample_with_seed(5_000, seed)
              .unwrap_or_else(|e| panic!("{name} θ = {theta} seed {seed}: {e}"));
            assert!(
              uv.iter().all(|x| (0.0..=1.0).contains(x)),
              "{name} θ = {theta} seed {seed} left the unit square"
            );
          }
        });
      }
    }
  });
}

type Row = (f64, f64, f64, f64);

/// `(θ, u, v, h(u | v))`: 60-digit values of sympy's `∂_v C`, where the old forms answered NaN (near 1) or a staircase.
const GUMBEL_H: [Row; 6] = [
  (2.0, 0.3, 0.7, 0.11559784394154603),
  (100.0, 0.99948, 0.99978, 1.0207814815154873e-37),
  (200.0, 0.977, 0.99, 2.7593906173946483e-73),
  (100.0, 0.9999, 0.99999, 9.954651042431668e-100),
  (2.0, 1e-16, 0.5, 3.7377593626062655e-18),
  (2.0, 1e-17, 0.5, 3.519311518473224e-19),
];

const JOE_H: [Row; 6] = [
  (2.0, 0.3, 0.7, 0.20900157182558335),
  (100.0, 0.99948, 0.99978, 1.036368922966833e-37),
  (200.0, 0.977, 0.99, 1.0379122374284449e-72),
  (100.0, 0.9999, 0.99999, 9.999999995603517e-100),
  (2.0, 1e-16, 0.5, 1e-16),
  (2.0, 1e-17, 0.5, 1e-17),
];

/// `(θ, y, v, h⁻¹(y | v))`: 60-digit bisection roots of that `h`.
const GUMBEL_INVERSE: [Row; 7] = [
  (50.0, 0.5, 1.0 - 1e-6, 0.9999989994379549),
  (100.0, 0.5, 1.0 - 1e-4, 0.9998999860468408),
  (200.0, 0.5, 0.99, 0.9899996575223512),
  (200.0, 1e-6, 0.999, 0.9989281476147631),
  (2.0, 1e-15, 0.5, 2.2837289372229754e-14),
  (2.0, 1e-12, 0.5, 1.802670471158364e-11),
  (50.0, 1e-300, 1.0 - 1e-10, 0.9998674379982581),
];

const JOE_INVERSE: [Row; 6] = [
  (50.0, 0.5, 1.0 - 1e-6, 0.999998999437954),
  (100.0, 0.5, 1.0 - 1e-4, 0.9998999860447332),
  (200.0, 0.5, 0.99, 0.989999652283301),
  (200.0, 1e-6, 0.999, 0.9989281086857948),
  (2.0, 1e-15, 0.5, 9.999999999999999e-16),
  (2.0, 1e-12, 0.5, 9.9999999999975e-13),
];

/// `h` to 1e-12, as `θ ln_1p(−u)` alone carries `θ` rounding units at `θ = 200`; Brent's root to 1e-14.
fn assert_h_and_inverse(
  name: &str,
  family: impl Fn(f64) -> Box<dyn BivariateExt>,
  h_rows: &[Row],
  inverse_rows: &[Row],
) {
  for &(theta, u, v, want) in h_rows {
    let got = family(theta).partial_derivative(&array![[u, v]]).unwrap()[0];
    assert!(
      (got - want).abs() <= 1e-12 * want,
      "{name} θ = {theta}: h({u} | {v}) = {got} vs {want}"
    );
  }
  for &(theta, y, v, want) in inverse_rows {
    assert_reference(name, family(theta).as_ref(), (y, v, want), 1e-14);
  }
}

#[test]
fn gumbel_and_joe_match_the_60_digit_references() {
  let gumbel = |theta| Box::new(Gumbel::new(Some(theta), None)) as Box<dyn BivariateExt>;
  assert_h_and_inverse("gumbel", gumbel, &GUMBEL_H, &GUMBEL_INVERSE);
  let joe = |theta| Box::new(with_theta(Joe::new(), theta)) as Box<dyn BivariateExt>;
  assert_h_and_inverse("joe", joe, &JOE_H, &JOE_INVERSE);
}

#[test]
fn a_query_outside_the_unit_interval_is_nan() {
  for (name, copula) in families() {
    for bad in [f64::NAN, -0.1, 1.5, f64::NEG_INFINITY, f64::INFINITY] {
      let u = copula.ppf(&array![bad, 0.4], &array![0.6, bad]).unwrap();
      assert!(u.iter().all(|u| u.is_nan()), "{name} at {bad}: {u}");
      let h = copula.partial_derivative(&array![[0.4, bad]]).unwrap();
      assert!(h[0].is_nan(), "{name} at v = {bad}: h = {}", h[0]);
    }
  }
}

/// `1 − 2⁻⁵³` is the largest uniform draw, and rounding near `y = 1` must not leave the square either.
#[test]
fn the_edges_of_the_square_answer_inside_it() {
  let (y, v) = (array![0.0, 1.0, 0.3, 0.3], array![0.5, 0.5, 0.0, 1.0]);
  let near_one = Array1::from_iter(
    [1.0, 1.0 - f64::EPSILON / 2.0]
      .into_iter()
      .flat_map(|y| [y; 99]),
  );
  let sweep = Array1::from_iter((0..198).map(|i| (i % 99 + 1) as f64 / 100.0));
  let edges = array![[0.0, 0.5], [1.0, 0.5], [0.3, 0.0], [0.3, 1.0]];
  let mut copulas = families();
  for theta in [800.0, -800.0, 1000.0, -1000.0] {
    copulas.push((
      "frank at |θ| ≥ 800",
      Box::new(Frank::new(Some(theta), None)),
    ));
  }
  copulas.push(("amh 0.99", Box::new(with_theta(Amh::new(), 0.99))));
  copulas.push(("plackett 0.01", Box::new(with_theta(Plackett::new(), 0.01))));
  for (name, copula) in copulas {
    let u = copula.percent_point(&y, &v).unwrap();
    assert!(u.iter().all(|u| (0.0..=1.0).contains(u)), "{name}: {u}");
    let u = copula.percent_point(&near_one, &sweep).unwrap();
    assert!(
      u.iter().all(|u| (0.0..=1.0).contains(u)),
      "{name} near y = 1: {u}"
    );
    let h = copula.partial_derivative(&edges).unwrap();
    assert!(h.iter().all(|h| h.is_finite()), "{name}: {h}");
  }
}

#[test]
fn unequal_lengths_are_a_dimension_mismatch() {
  let pairs = [
    (array![0.3, 0.6], array![0.5]),
    (array![0.3], array![0.5, 0.6]),
    (array![0.1, 0.2, 0.3], array![0.4, 0.5]),
  ];
  for (name, copula) in families() {
    for (y, v) in &pairs {
      let want = CopulaError::DimensionMismatch {
        expected: y.len(),
        got: v.len(),
      };
      assert_eq!(copula.percent_point(y, v).unwrap_err(), want, "{name}");
    }
  }
}

#[test]
fn independence_partial_derivative_is_the_v_derivative() {
  let h = with_theta(Independence::new(), 0.0)
    .partial_derivative(&array![[0.3, 0.8], [0.9, 0.2]])
    .unwrap();
  assert_eq!(h, array![0.3, 0.9]);
}

/// `(θ, y, v, h⁻¹(y | v))`: 50-digit mpmath bisection roots of the crate's own `h`, never a closed form.
const FRANK: [(f64, f64, f64, f64); 16] = [
  (4.0, 0.3, 0.7, 0.49099526153020284),
  (4.0, 1e-06, 0.99, 1.2873803156983498e-05),
  (4.0, 0.99, 1e-06, 0.8927079478110738),
  (4.0, 0.999999, 0.3, 0.9999959641666158),
  (-3.0, 0.3, 0.7, 0.22289828511997253),
  (-3.0, 1e-06, 0.99, 3.2638389105694917e-07),
  (-3.0, 0.99, 1e-06, 0.996817469641072),
  (-3.0, 0.999999, 0.3, 0.9999992209513324),
  (0.0001, 0.3, 0.7, 0.30000420005179945),
  (0.0001, 1e-06, 0.99, 1.0000490015682017e-06),
  (0.0001, 0.99, 1e-06, 0.9899995049848197),
  (0.0001, 0.999999, 0.3, 0.9999989999799994),
  (30.0, 0.3, 0.7, 0.6717549750954881),
  (30.0, 1e-06, 0.99, 0.52948299425107),
  (30.0, 0.99, 1e-06, 0.15350666286610937),
  (30.0, 0.999999, 0.3, 0.7604917196701405),
];

/// The same at `|θ| = 800` and `1000`, where the old closed form overflowed: 1200-digit roots, as `e^{−1000} ≈ 1e-434`.
const FRANK_EXTREME: [Row; 16] = [
  (800.0, 0.3, 0.7, 0.698940877674516),
  (800.0, 1e-06, 0.99, 0.9727306130521259),
  (800.0, 0.99, 1e-06, 0.005757452736444078),
  (800.0, 0.999999, 0.3, 0.31726938694741874),
  (-800.0, 0.3, 0.7, 0.29894087767451605),
  (-800.0, 1e-06, 0.99, 3.720658392370434e-06),
  (-800.0, 0.99, 1e-06, 0.9999874270762221),
  (-800.0, 0.999999, 0.3, 0.7172693869474188),
  (1000.0, 0.3, 0.7, 0.6991527021396128),
  (1000.0, 1e-06, 0.99, 0.9761844904419908),
  (1000.0, 0.99, 1e-06, 0.004606160190936474),
  (1000.0, 0.999999, 0.3, 0.313815509557935),
  (-1000.0, 0.3, 0.7, 0.2991527021396128),
  (-1000.0, 1e-06, 0.99, 2.178740907901743e-05),
  (-1000.0, 0.99, 1e-06, 0.9999899396591949),
  (-1000.0, 0.999999, 0.3, 0.713815509557935),
];

/// `(θ, u, v, h(u | v))` in 1200 digits, where the unfactored `h` divided `0` by `0`.
const FRANK_H: [Row; 2] = [
  (1000.0, 1.0, 0.9, 1.0),
  (-1000.0, 0.05, 0.9, 1.928749847963966e-22),
];

const FGM: [(f64, f64, f64, f64); 12] = [
  (0.5, 0.3, 0.7, 0.34520787991171475),
  (0.5, 1e-06, 0.99, 1.9607806198358564e-06),
  (0.5, 0.99, 1e-06, 0.9803847947383545),
  (0.5, 0.999999, 0.3, 0.9999987500003906),
  (-1.0, 0.3, 0.7, 0.22930936742544508),
  (-1.0, 1e-06, 0.99, 5.050506313003118e-07),
  (-1.0, 0.99, 1e-06, 0.994987432094052),
  (-1.0, 0.999999, 0.3, 0.9999992857141399),
  (1.0, 0.3, 0.7, 0.39564392373895996),
  (1.0, 1e-06, 0.99, 4.987809659850575e-05),
  (1.0, 0.99, 1e-06, 0.90000089999685),
  (1.0, 0.999999, 0.3, 0.9999983333351852),
];

const AMH: [(f64, f64, f64, f64); 12] = [
  (0.5, 0.3, 0.7, 0.360468370229395),
  (0.5, 1e-06, 0.99, 1.9800461188202852e-06),
  (0.5, 0.99, 1e-06, 0.9801980586216278),
  (0.5, 0.999999, 0.3, 0.9999987500003564),
  (-1.0, 0.3, 0.7, 0.2574070595017641),
  (-1.0, 1e-06, 0.99, 5.100501249240588e-07),
  (-1.0, 0.99, 1e-06, 0.9949748693719805),
  (-1.0, 0.999999, 0.3, 0.9999992857141144),
  (0.99, 0.3, 0.7, 0.45662013460060336),
  (0.99, 1e-06, 0.99, 9.709664329243053e-05),
  (0.99, 0.99, 1e-06, 0.49753693108518243),
  (0.99, 0.999999, 0.3, 0.9999983443723467),
];

const PLACKETT: [(f64, f64, f64, f64); 12] = [
  (3.0, 0.3, 0.7, 0.41384445013187743),
  (3.0, 1e-06, 0.99, 2.9601274529415558e-06),
  (3.0, 0.99, 1e-06, 0.970588274117634),
  (3.0, 0.999999, 0.3, 0.9999980800028415),
  (0.2, 0.3, 0.7, 0.22980959937162107),
  (0.2, 1e-06, 0.99, 2.163201558197486e-07),
  (0.2, 0.99, 1e-06, 0.997983854999969),
  (0.2, 0.999999, 0.3, 0.9999990320012699),
  (50.0, 0.3, 0.7, 0.6338544091135255),
  (50.0, 1e-06, 0.99, 4.902240109864848e-05),
  (50.0, 0.99, 1e-06, 0.6644301813418158),
  (50.0, 0.999999, 0.3, 0.999975079298755),
];

/// `(ρ, ν, y, v, h⁻¹(y | v))`: the inversion of `t_{ν+1}` in 50-digit mpmath, `h` re-evaluated at each to 1e-46.
const T_COPULA: [(f64, f64, f64, f64, f64); 8] = [
  (0.5, 4.0, 0.3, 0.7, 0.43803740678313124),
  (0.5, 4.0, 1e-06, 0.99, 1.3078557464437567e-06),
  (0.5, 4.0, 0.99, 1e-06, 0.9999976205764943),
  (0.5, 4.0, 0.999999, 0.3, 0.9999802754607019),
  (-0.8, 2.5, 0.3, 0.7, 0.24884441772983576),
  (-0.8, 2.5, 1e-06, 0.99, 5.2332595819437634e-06),
  (-0.8, 2.5, 0.99, 1e-06, 0.99999984395498),
  (-0.8, 2.5, 0.999999, 0.3, 0.9998900573971097),
];

fn assert_reference(
  name: &str,
  copula: &dyn BivariateExt,
  (y, v, want): (f64, f64, f64),
  rel: f64,
) {
  let got = copula.percent_point(&array![y], &array![v]).unwrap()[0];
  assert!(
    (got - want).abs() <= rel * want,
    "{name} y = {y}, v = {v}: {got} vs {want}"
  );
}

#[test]
fn the_closed_forms_match_the_50_digit_references() {
  for (theta, y, v, want) in FRANK.into_iter().chain(FRANK_EXTREME) {
    assert_reference("frank", &Frank::new(Some(theta), None), (y, v, want), 1e-14);
  }
  for (theta, u, v, want) in FRANK_H {
    let got = Frank::new(Some(theta), None)
      .partial_derivative(&array![[u, v]])
      .unwrap()[0];
    assert!(
      (got - want).abs() <= 1e-14 * want,
      "frank θ = {theta}: h({u} | {v}) = {got} vs {want}"
    );
  }
  for (theta, y, v, want) in FGM {
    assert_reference("fgm", &with_theta(Fgm::new(), theta), (y, v, want), 1e-14);
  }
  for (theta, y, v, want) in AMH {
    assert_reference("amh", &with_theta(Amh::new(), theta), (y, v, want), 1e-14);
  }
  for (theta, y, v, want) in PLACKETT {
    let plackett = with_theta(Plackett::new(), theta);
    assert_reference("plackett", &plackett, (y, v, want), 1e-14);
  }
  for (rho, nu, y, v, want) in T_COPULA {
    let t = with_theta(TCopula::with_nu(nu), rho);
    assert_reference("t_copula", &t, (y, v, want), 1e-12);
  }
}

const GRID: [f64; 7] = [1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6];

fn assert_round_trips(name: &str, copula: &dyn BivariateExt) {
  let y = Array1::from_iter(GRID.iter().flat_map(|&y| GRID.map(|_| y)));
  let v = Array1::from_iter(GRID.iter().flat_map(|_| GRID));
  let u = copula.percent_point(&y, &v).unwrap();
  let back = copula.partial_derivative(&stack![Axis(1), u, v]).unwrap();
  for i in 0..y.len() {
    assert!(
      (back[i] - y[i]).abs() <= 1e-10,
      "{name} y = {}, v = {}: h({}) = {}",
      y[i],
      v[i],
      u[i],
      back[i]
    );
  }
}

#[test]
fn the_closed_forms_round_trip_through_h_on_the_grid() {
  for theta in [4.0, -3.0, 1e-4, 30.0, 800.0, -800.0, 1000.0, -1000.0] {
    assert_round_trips("frank", &Frank::new(Some(theta), None));
  }
  for theta in [0.5, -1.0, 1.0] {
    assert_round_trips("fgm", &with_theta(Fgm::new(), theta));
  }
  for theta in [0.5, -1.0, 0.99] {
    assert_round_trips("amh", &with_theta(Amh::new(), theta));
  }
  for theta in [3.0, 0.2, 50.0] {
    assert_round_trips("plackett", &with_theta(Plackett::new(), theta));
  }
  for (rho, nu) in [(0.5, 4.0), (-0.8, 2.5)] {
    assert_round_trips("t_copula", &with_theta(TCopula::with_nu(nu), rho));
  }
}
