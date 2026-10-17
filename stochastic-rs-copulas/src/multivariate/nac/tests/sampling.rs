use super::*;
use crate::correlation::kendall_tau;

/// `(θ, ζ₁)` of Clayton, ζ₁ = Var(1 − 2U − 2V + 4C(U, V)) by quadrature, for Hoeffding's (1948, Eq. (9.15)) variance
/// of Kendall's τ̂, Var τ̂ = 2[2(n − 2)ζ₁ + 1 − τ²]/(n(n − 1)).
const CLAYTON_ZETA1: [(f64, f64); 8] = [
  (0.25, 0.112392),
  (0.5, 0.109010),
  (1.0, 0.096723),
  (2.0, 0.071489),
  (4.0, 0.040650),
  (6.0, 0.025675),
  (8.0, 0.017582),
  (10.0, 0.012764),
];

const NESTED_N: usize = 20_000;

/// Kolmogorov's 0.999 quantile: the √n-scaled distance a uniform sample stays under with probability 0.999.
const KS_999: f64 = 1.9495;

/// A flat (single-level) Clayton NAC with all leaves under one root must
/// reduce to the standard exchangeable Clayton copula on 3 dims.
#[test]
fn nac_clayton_flat_is_exchangeable_clayton() {
  let root = NacNode::leaf_group(2.0, vec![0, 1, 2]);
  let nac = NestedArchimedean::new(BivariateFamily::Clayton, root, 3).unwrap();
  let u = nac.sample(8_000, &Deterministic::new(7)).unwrap();
  assert_eq!(u.ncols(), 3);
  for j in 0..3 {
    let col = u.column(j);
    let mean = col.iter().sum::<f64>() / col.len() as f64;
    assert!(
      (mean - 0.5).abs() < 0.03,
      "marginal {j} mean = {mean}, expected ~0.5"
    );
  }
}

/// The tilted stable's empirical LST is `exp(−v₀((1 + t)^α − 1))` within 4 SE, `Var e^{−tX} = L(2t) − L(t)²`, for one
/// summand (`v₀ ≤ 1`, and `1 < v₀ < 2 ln 2`) and for several, down to `α = 1/40`.
#[test]
fn nac_clayton_tilted_stable_has_its_laplace_transform() {
  let n = 20_000;
  for (alpha, v0, seed) in [
    (0.25, 0.6, 1),
    (0.5, 1.2, 2),
    (0.1, 3.7, 3),
    (0.9, 9.3, 4),
    (0.025, 2.0, 5),
  ] {
    let mut rng = Deterministic::new(seed).rng();
    let draws = (0..n)
      .map(|_| NestedArchimedean::tilted_stable(&mut rng, alpha, v0))
      .collect::<Vec<_>>();
    let lst = |t: f64| (-v0 * ((1.0 + t).powf(alpha) - 1.0)).exp();
    for t in [0.5, 1.0, 2.0] {
      let got = draws.iter().map(|x| (-t * x).exp()).sum::<f64>() / n as f64;
      let se = ((lst(2.0 * t) - lst(t).powi(2)) / n as f64).sqrt();
      assert!(
        (got - lst(t)).abs() < 4.0 * se,
        "α {alpha} v₀ {v0} t {t}: {got} vs {} ± {}",
        lst(t),
        4.0 * se
      );
    }
  }
}

/// `d = 3` at `α = 1/4` and at `α = 1`, `d = 4` at `α = 1/16`, and three levels at `d = 6` down to `α = 1/40` under a root
/// whose frailty has mean 4, so the tilted stable runs with one summand and with several.
fn nested_clayton_trees() -> Vec<NestedArchimedean> {
  let lg = NacNode::leaf_group;
  let nest = |mut node: NacNode, children| {
    node.children = children;
    node
  };
  let inner = nest(lg(1.0, vec![1]), vec![lg(6.0, vec![2, 3])]);
  [
    (3, nest(lg(1.0, vec![0]), vec![lg(4.0, vec![1, 2])])),
    (3, nest(lg(2.0, vec![0]), vec![lg(2.0, vec![1, 2])])),
    (
      4,
      nest(
        lg(0.5, vec![]),
        vec![lg(8.0, vec![0, 1]), lg(2.0, vec![2, 3])],
      ),
    ),
    (
      6,
      nest(lg(0.25, vec![0]), vec![inner, lg(10.0, vec![4, 5])]),
    ),
  ]
  .into_iter()
  .map(|(d, root)| NestedArchimedean::new(BivariateFamily::Clayton, root, d).unwrap())
  .collect()
}

/// θ of the youngest node whose subtree holds leaves `i` and `j`.
fn ancestor_theta(node: &NacNode, i: usize, j: usize) -> f64 {
  let holds = |c: &&NacNode| {
    let mut leaves = vec![];
    c.collect_leaves(&mut leaves);
    leaves.contains(&i) && leaves.contains(&j)
  };
  node
    .children
    .iter()
    .find(holds)
    .map_or(node.theta, |c| ancestor_theta(c, i, j))
}

/// Every pair of a nested Clayton sample is Clayton on its youngest common ancestor's θ: τ̂ within 4 SE of θ/(θ + 2).
#[test]
fn nac_clayton_nested_pairs_follow_their_youngest_common_ancestor() {
  let n = NESTED_N as f64;
  for nac in nested_clayton_trees() {
    let root = nac.root().unwrap();
    for seed in [7, 11, 13] {
      let u = nac.sample(NESTED_N, &Deterministic::new(seed)).unwrap();
      let tau = kendall_tau(u.view()).unwrap();
      for i in 0..nac.dim() {
        for j in i + 1..nac.dim() {
          let theta = ancestor_theta(root, i, j);
          let want = theta / (theta + 2.0);
          let zeta1 = CLAYTON_ZETA1.iter().find(|(t, _)| *t == theta).unwrap().1;
          let se = (2.0 * (2.0 * (n - 2.0) * zeta1 + 1.0 - want * want) / (n * (n - 1.0))).sqrt();
          assert!(
            (tau[[i, j]] - want).abs() < 4.0 * se,
            "{root:?} seed {seed} ({i}, {j}): τ̂ {} vs {want} ± {}",
            tau[[i, j]],
            4.0 * se
          );
        }
      }
    }
  }
}

/// The share of a nested Clayton sample at or below `q` is the copula's cdf at `q`, within 4 binomial SE.
#[test]
fn nac_clayton_nested_sample_matches_the_cdf() {
  let points = [
    [0.3, 0.5, 0.7, 0.4, 0.6, 0.5],
    [0.6; 6],
    [0.2, 0.9, 0.5, 0.5, 0.8, 0.3],
  ];
  for nac in nested_clayton_trees() {
    let q = ndarray::Array2::from_shape_fn((points.len(), nac.dim()), |(k, j)| points[k][j]);
    let cdf = nac.cdf(q.view()).unwrap();
    for seed in [7, 11, 13] {
      let u = nac.sample(NESTED_N, &Deterministic::new(seed)).unwrap();
      for (k, q) in q.rows().into_iter().enumerate() {
        let below = u
          .rows()
          .into_iter()
          .filter(|r| r.iter().zip(q.iter()).all(|(x, y)| x <= y))
          .count();
        let got = below as f64 / NESTED_N as f64;
        let se = (cdf[k] * (1.0 - cdf[k]) / NESTED_N as f64).sqrt();
        assert!(
          (got - cdf[k]).abs() < 4.0 * se,
          "{:?} seed {seed} q{k}: {got} vs cdf {} ± {}",
          nac.root(),
          cdf[k],
          4.0 * se
        );
      }
    }
  }
}

/// A leaf of each tree's most strongly nested node is uniform: √n·KS below Kolmogorov's 0.999 quantile.
#[test]
fn nac_clayton_nested_margin_is_uniform() {
  let n = NESTED_N as f64;
  for (nac, j) in nested_clayton_trees().into_iter().zip([1, 1, 0, 4]) {
    for seed in [7, 11, 13] {
      let u = nac.sample(NESTED_N, &Deterministic::new(seed)).unwrap();
      let mut column = u.column(j).to_vec();
      column.sort_by(f64::total_cmp);
      let ks = column
        .iter()
        .enumerate()
        .map(|(i, &x)| ((i + 1) as f64 / n - x).max(x - i as f64 / n))
        .fold(0.0, f64::max);
      assert!(
        ks * n.sqrt() < KS_999,
        "{:?} seed {seed} U{j}: √n·KS {}",
        nac.root(),
        ks * n.sqrt()
      );
    }
  }
}

/// 2-level Gumbel NAC: same structural test as Clayton but on the
/// Gumbel family with θ_root=2.0 and inner θ=4.0.
#[test]
fn nac_gumbel_two_level_inner_pair_more_dependent() {
  let inner = NacNode::leaf_group(4.0, vec![1, 2]);
  let root = NacNode {
    theta: 2.0,
    leaves: vec![0],
    children: vec![inner],
  };
  let nac = NestedArchimedean::new(BivariateFamily::Gumbel, root, 3).unwrap();
  let u = nac.sample(8_000, &Deterministic::new(7)).unwrap();
  let tau = kendall_tau(u.view()).unwrap();
  // Gumbel: τ = 1 − 1/θ
  //   Inner θ=4 → τ ≈ 0.75
  //   Outer θ=2 → τ ≈ 0.50
  assert!(
    tau[[1, 2]] > tau[[0, 1]],
    "inner pair τ_(1,2)={} should exceed outer τ_(0,1)={}",
    tau[[1, 2]],
    tau[[0, 1]]
  );
  assert!(
    tau[[1, 2]] > 0.6 && tau[[1, 2]] < 0.85,
    "Gumbel inner τ_(1,2)={} out of expected band [0.6, 0.85]",
    tau[[1, 2]]
  );
}
