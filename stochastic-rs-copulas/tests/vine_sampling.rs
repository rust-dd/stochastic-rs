//! The C- and D-vine samplers draw the vine they declare (tree-1 pairs, tree-2 pairs given their neighbour and the
//! Rosenblatt transform, each within 4 SE of Kendall's τ̂), and their densities match a conditional-CDF recursion.
//! Reference: Hoeffding, W. (1948), "A class of statistics with asymptotically normal distribution", Annals of Mathematical Statistics 19(3), 293–325, doi:10.1214/aoms/1177730196.

use ndarray::ArrayView1;
use ndarray::array;
use stochastic_rs_copulas::correlation::kendall_tau_b;
use stochastic_rs_copulas::multivariate::cvine::CVine;
use stochastic_rs_copulas::multivariate::dvine::DVine;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::traits::MultivariateExt;
use stochastic_rs_core::simd_rng::Deterministic;

const N: usize = 20_000;

const GAUSSIAN: PairCopula = PairCopula::Gaussian { rho: 0.5 };

const CLAYTON: PairCopula = PairCopula::Clayton { theta: 3.0 };

/// Kendall's τ of Gaussian(0.5), 2 arcsin(ρ)/π, and of Clayton(3), θ/(θ + 2).
const TAU_GAUSSIAN: f64 = 1.0 / 3.0;

const TAU_CLAYTON: f64 = 0.6;

/// ζ₁ = Var(1 − 2U − 2V + 4C(U, V)) by Gauss–Legendre quadrature; the Gaussian value is 1/9 − 4 arcsin²(ρ/2)/π².
const ZETA1_GAUSSIAN: f64 = 0.085235;

const ZETA1_CLAYTON: f64 = 0.053182;

const ZETA1_INDEPENDENCE: f64 = 1.0 / 9.0;

/// Kolmogorov's 0.999 quantile: the √n-scaled distance a uniform sample stays under with probability 0.999.
const KS_999: f64 = 1.9495;

/// Four standard errors of τ̂ at `N` draws, by Hoeffding (1948) Eq. (9.15): Var τ̂ = 2[2(n − 2)ζ₁ + 1 − τ²]/(n(n − 1)).
fn four_se(tau: f64, zeta1: f64) -> f64 {
  let n = N as f64;
  4.0 * (2.0 * (2.0 * (n - 2.0) * zeta1 + 1.0 - tau * tau) / (n * (n - 1.0))).sqrt()
}

fn assert_tau(label: &str, a: &[f64], b: &[f64], tau: f64, zeta1: f64) {
  let got = kendall_tau_b(ArrayView1::from(a), ArrayView1::from(b)).unwrap();
  let tol = four_se(tau, zeta1);
  assert!(
    (got - tau).abs() < tol,
    "{label}: τ̂ {got:.4}, want {tau:.4} ± {tol:.4} (4 SE)"
  );
}

fn columns(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
  (0..rows[0].len())
    .map(|j| rows.iter().map(|r| r[j]).collect())
    .collect()
}

fn draws<C: MultivariateExt>(vine: &C, seed: u64) -> Vec<Vec<f64>> {
  let sample = vine.sample(N, &Deterministic::new(seed)).unwrap();
  sample.rows().into_iter().map(|r| r.to_vec()).collect()
}

/// Every tree carries a different family or parameter per edge, so an edge index that is off moves the draws.
fn mixed_trees() -> Vec<Vec<PairCopula>> {
  vec![
    vec![
      GAUSSIAN,
      PairCopula::Clayton { theta: 2.0 },
      PairCopula::Frank { theta: -3.0 },
      PairCopula::Gaussian { rho: -0.2 },
    ],
    vec![
      CLAYTON,
      PairCopula::Gaussian { rho: 0.6 },
      PairCopula::Frank { theta: 5.0 },
    ],
    vec![
      PairCopula::Frank { theta: 4.0 },
      PairCopula::Clayton { theta: 1.0 },
    ],
    vec![PairCopula::Gaussian { rho: -0.4 }],
  ]
}

/// `(log c(x), Rosenblatt z)` by recursion: entering tree s + 1, `a[e]` is F(x_e | x_{e+1..e+s}) and `b[e]` is
/// F(x_{e+s} | x_{e..e+s−1}); edge e of that tree joins `a[e]` and `b[e + 1]`.
fn dvine_recursion(vine: &DVine, x: &[f64]) -> (f64, Vec<f64>) {
  let (mut a, mut b) = (x.to_vec(), x.to_vec());
  let mut log_c = 0.0;
  let mut z = vec![x[0]];
  for tree in vine.pair_copulas() {
    let (mut next_a, mut next_b) = (Vec::new(), Vec::new());
    for (e, pair) in tree.iter().enumerate() {
      log_c += pair.log_density(a[e], b[e + 1]);
      next_a.push(pair.h(a[e], b[e + 1]));
      next_b.push(pair.h(b[e + 1], a[e]));
    }
    z.push(next_b[0]);
    a = next_a;
    b = next_b;
  }
  (log_c, z)
}

/// The same for a C-vine: entering tree t + 1, `g[k]` is F(x_k | x_0..x_{t−1}) and the tree's root is variable t.
fn cvine_recursion(vine: &CVine, x: &[f64]) -> (f64, Vec<f64>) {
  let mut g = x.to_vec();
  let mut log_c = 0.0;
  let mut z = vec![x[0]];
  for (t, tree) in vine.pair_copulas().iter().enumerate() {
    let root = g[t];
    for (e, pair) in tree.iter().enumerate() {
      let k = t + 1 + e;
      log_c += pair.log_density(root, g[k]);
      g[k] = pair.h(g[k], root);
    }
    z.push(g[t + 1]);
  }
  (log_c, z)
}

fn ks_distance(column: &[f64]) -> f64 {
  let mut sorted = column.to_vec();
  sorted.sort_by(f64::total_cmp);
  let n = sorted.len() as f64;
  sorted
    .iter()
    .enumerate()
    .map(|(i, &x)| ((i + 1) as f64 / n - x).max(x - i as f64 / n))
    .fold(0.0, f64::max)
}

fn assert_independent_uniform(label: &str, z: &[Vec<f64>]) {
  let cols = columns(z);
  for (j, col) in cols.iter().enumerate() {
    let scaled = ks_distance(col) * (N as f64).sqrt();
    assert!(scaled < KS_999, "{label}: z{j} √n·KS {scaled:.3}");
  }
  for i in 0..cols.len() {
    for j in i + 1..cols.len() {
      let pair = format!("{label}: (z{i}, z{j})");
      assert_tau(&pair, &cols[i], &cols[j], 0.0, ZETA1_INDEPENDENCE);
    }
  }
}

fn reviewer_dvine() -> DVine {
  let trees = vec![
    vec![GAUSSIAN; 3],
    vec![CLAYTON; 2],
    vec![PairCopula::Independence],
  ];
  DVine::new(4, trees).unwrap()
}

fn root_cvine() -> CVine {
  let trees = vec![
    vec![GAUSSIAN; 3],
    vec![CLAYTON; 2],
    vec![PairCopula::Independence],
  ];
  CVine::new(4, trees).unwrap()
}

#[test]
fn dvine_adjacent_pairs_follow_the_first_tree() {
  let cols = columns(&draws(&reviewer_dvine(), 7));
  for i in 0..3 {
    let pair = format!("(U{i}, U{})", i + 1);
    assert_tau(&pair, &cols[i], &cols[i + 1], TAU_GAUSSIAN, ZETA1_GAUSSIAN);
  }
}

#[test]
fn dvine_pairs_given_their_neighbour_follow_the_second_tree() {
  let vine = reviewer_dvine();
  let t1 = &vine.pair_copulas()[0];
  let rows = draws(&vine, 7);
  for i in 0..2 {
    let a = rows
      .iter()
      .map(|r| t1[i].h(r[i], r[i + 1]))
      .collect::<Vec<_>>();
    let b = rows
      .iter()
      .map(|r| t1[i + 1].h(r[i + 2], r[i + 1]))
      .collect::<Vec<_>>();
    let pair = format!("(U{i}, U{} | U{})", i + 2, i + 1);
    assert_tau(&pair, &a, &b, TAU_CLAYTON, ZETA1_CLAYTON);
  }
}

#[test]
fn cvine_root_pairs_follow_the_first_tree() {
  let cols = columns(&draws(&root_cvine(), 7));
  for k in 1..4 {
    let pair = format!("(U0, U{k})");
    assert_tau(&pair, &cols[0], &cols[k], TAU_GAUSSIAN, ZETA1_GAUSSIAN);
  }
}

#[test]
fn cvine_pairs_given_the_root_follow_the_second_tree() {
  let vine = root_cvine();
  let t1 = &vine.pair_copulas()[0];
  let rows = draws(&vine, 7);
  let a = rows.iter().map(|r| t1[0].h(r[1], r[0])).collect::<Vec<_>>();
  for k in 2..4 {
    let b = rows
      .iter()
      .map(|r| t1[k - 1].h(r[k], r[0]))
      .collect::<Vec<_>>();
    let pair = format!("(U1, U{k} | U0)");
    assert_tau(&pair, &a, &b, TAU_CLAYTON, ZETA1_CLAYTON);
  }
}

#[test]
fn the_rosenblatt_transform_of_dvine_draws_is_independent_uniform() {
  let vine = DVine::new(5, mixed_trees()).unwrap();
  let z = draws(&vine, 11)
    .iter()
    .map(|x| dvine_recursion(&vine, x).1)
    .collect::<Vec<_>>();
  assert_independent_uniform("D-vine", &z);
}

#[test]
fn the_rosenblatt_transform_of_cvine_draws_is_independent_uniform() {
  let vine = CVine::new(5, mixed_trees()).unwrap();
  let z = draws(&vine, 11)
    .iter()
    .map(|x| cvine_recursion(&vine, x).1)
    .collect::<Vec<_>>();
  assert_independent_uniform("C-vine", &z);
}

#[test]
fn five_dimensional_densities_match_the_recursion() {
  let dvine = DVine::new(5, mixed_trees()).unwrap();
  let cvine = CVine::new(5, mixed_trees()).unwrap();
  let points = array![
    [0.3, 0.6, 0.8, 0.15, 0.55],
    [0.9, 0.2, 0.45, 0.7, 0.05],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.05, 0.95, 0.35, 0.6, 0.25]
  ];
  let d_log = dvine.log_pdf(points.view()).unwrap();
  let c_log = cvine.log_pdf(points.view()).unwrap();
  for (r, row) in points.rows().into_iter().enumerate() {
    let x = row.to_vec();
    let (d_want, c_want) = (dvine_recursion(&dvine, &x).0, cvine_recursion(&cvine, &x).0);
    assert!(
      (d_log[r] - d_want).abs() < 1e-12,
      "D-vine row {r}: {} vs {d_want}",
      d_log[r]
    );
    assert!(
      (c_log[r] - c_want).abs() < 1e-12,
      "C-vine row {r}: {} vs {c_want}",
      c_log[r]
    );
  }
}
