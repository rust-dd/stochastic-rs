use stochastic_rs_core::simd_rng::Deterministic;

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

/// In a two-level Clayton NAC's cdf, a pair across levels is exchangeable Clayton on the root generator, a pair inside
/// the sub-tree on the inner one.
#[test]
fn nac_clayton_two_level_cdf_pair_margins() {
  let inner = NacNode::leaf_group(4.0, vec![1, 2]);
  let root = NacNode {
    theta: 1.5,
    leaves: vec![0],
    children: vec![inner],
  };
  let nac = NestedArchimedean::new(BivariateFamily::Clayton, root, 3).unwrap();
  // Outer-inner pair margin: C(u_0, u_1, 1) = (u_0^{-θ_root} + u_1^{-θ_root} - 1)^{-1/θ_root}
  let q_outer_inner = ndarray::array![[0.3, 0.7, 1.0 - 1e-15]];
  let c_oi = nac.cdf(q_outer_inner.view()).unwrap()[0];
  let theta_root: f64 = 1.5;
  let expected_oi =
    (0.3f64.powf(-theta_root) + 0.7f64.powf(-theta_root) - 1.0).powf(-1.0 / theta_root);
  assert!(
    (c_oi - expected_oi).abs() < 5e-3,
    "outer-inner pair CDF={} vs Clayton(θ_root) expected={}",
    c_oi,
    expected_oi
  );
  // Inner-inner pair margin: C(1, u_1, u_2) = (u_1^{-θ_inner} + u_2^{-θ_inner} - 1)^{-1/θ_inner}
  let q_inner_inner = ndarray::array![[1.0 - 1e-15, 0.3, 0.7]];
  let c_ii = nac.cdf(q_inner_inner.view()).unwrap()[0];
  let theta_inner: f64 = 4.0;
  let expected_ii =
    (0.3f64.powf(-theta_inner) + 0.7f64.powf(-theta_inner) - 1.0).powf(-1.0 / theta_inner);
  assert!(
    (c_ii - expected_ii).abs() < 5e-3,
    "inner-inner pair CDF={} vs Clayton(θ_inner) expected={}",
    c_ii,
    expected_ii
  );
  // Inner pair should be MORE dependent: larger CDF at off-diagonal
  // input pair than the outer pair at the same input.
  assert!(
    c_ii > c_oi,
    "inner CDF({c_ii}) must exceed outer CDF({c_oi}) — Clayton inner θ=4 > root θ=1.5"
  );
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

/// SNC violation: child θ less than parent θ must be rejected.
#[test]
fn nac_snc_violation_rejected() {
  let bad_inner = NacNode::leaf_group(0.5, vec![1]); // θ_child=0.5 < θ_parent=2.0
  let bad_root = NacNode {
    theta: 2.0,
    leaves: vec![0],
    children: vec![bad_inner],
  };
  let err = NestedArchimedean::new(BivariateFamily::Clayton, bad_root, 2).unwrap_err();
  let CopulaError::InvalidStructure(why) = &err else {
    panic!("expected InvalidStructure, got {err:?}");
  };
  assert!(why.contains("sufficient nesting condition"), "{why}");
}

/// A θ outside the generator's domain is a named error: Gumbel needs `θ ≥ 1`, Clayton `0 < 1/θ < ∞`.
#[test]
fn nac_below_family_min_rejected() {
  let gumbel = NestedArchimedean::new(
    BivariateFamily::Gumbel,
    NacNode::leaf_group(0.5, vec![0, 1]),
    2,
  );
  assert_eq!(
    gumbel.unwrap_err().to_string(),
    "theta must satisfy `theta >= 1.0 for Gumbel`, got theta = 0.5"
  );
  for theta in [-0.1, 0.0] {
    let clayton = NestedArchimedean::new(
      BivariateFamily::Clayton,
      NacNode::leaf_group(theta, vec![0, 1]),
      2,
    );
    assert_eq!(
      clayton.unwrap_err().to_string(),
      format!("theta must satisfy `0 < 1 / theta < ∞`, got theta = {theta:?}")
    );
  }
}

/// Missing / duplicate leaf indices must be rejected.
#[test]
fn nac_leaf_index_validation() {
  // Duplicate
  let dup_root = NacNode::leaf_group(2.0, vec![0, 0, 1]);
  assert!(NestedArchimedean::new(BivariateFamily::Clayton, dup_root, 3).is_err());
  // Missing (covers 0,2 only, dim=3)
  let miss = NacNode::leaf_group(2.0, vec![0, 2]);
  assert!(matches!(
    NestedArchimedean::new(BivariateFamily::Clayton, miss, 3),
    Err(CopulaError::InvalidStructure(_))
  ));
  // Out of range
  let oor = NacNode::leaf_group(2.0, vec![0, 1, 5]);
  assert!(NestedArchimedean::new(BivariateFamily::Clayton, oor, 3).is_err());
}

/// Next to independence (a small Clayton θ; θ = 0 itself has no frailty) the cdf is close to the product of the margins.
#[test]
fn nac_clayton_independence_cdf() {
  let root = NacNode::leaf_group(0.01, vec![0, 1, 2]);
  let nac = NestedArchimedean::new(BivariateFamily::Clayton, root, 3).unwrap();
  let q = ndarray::array![[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]];
  let c = nac.cdf(q.view()).unwrap();
  let indep_1 = 0.5_f64.powi(3);
  let indep_2 = 0.2_f64 * 0.3 * 0.4;
  assert!(
    (c[0] - indep_1).abs() < 0.05,
    "near-independence CDF[0]={} vs indep={indep_1}",
    c[0]
  );
  assert!(
    (c[1] - indep_2).abs() < 0.05,
    "near-independence CDF[1]={} vs indep={indep_2}",
    c[1]
  );
}

fn leaf_sets(node: &NacNode) -> Vec<Vec<usize>> {
  node
    .children
    .iter()
    .map(|c| {
      let mut leaves = c.leaves.clone();
      leaves.sort();
      leaves
    })
    .collect()
}

/// A nested Gumbel truth `((0,1)_{3.0}, (2,3)_{2.5})_{1.5}` is recovered: two children with those leaves, every θ
/// within sampling error, the root below both children.
#[test]
fn nac_fit_recovers_a_nested_gumbel_structure() {
  let root = NacNode {
    theta: 1.5,
    leaves: vec![],
    children: vec![
      NacNode::leaf_group(3.0, vec![0, 1]),
      NacNode::leaf_group(2.5, vec![2, 3]),
    ],
  };
  let truth = NestedArchimedean::new(BivariateFamily::Gumbel, root, 4).unwrap();
  let u = truth.sample(4000, &Deterministic::new(7)).unwrap();
  let mut fitted = NestedArchimedean::unfitted(BivariateFamily::Gumbel).unwrap();
  fitted.fit(u.view()).unwrap();
  let root = fitted.root().unwrap();
  assert!(
    root.leaves.is_empty() && root.children.len() == 2,
    "{root:?}"
  );
  let mut sets = leaf_sets(root);
  sets.sort();
  assert_eq!(sets, vec![vec![0, 1], vec![2, 3]]);
  for child in &root.children {
    let want = if child.leaves == [0, 1] { 3.0 } else { 2.5 };
    assert!(
      (child.theta - want).abs() < 0.35,
      "child θ {} vs {want}",
      child.theta
    );
    assert!(child.theta >= root.theta);
  }
  assert!((root.theta - 1.5).abs() < 0.3, "root θ {}", root.theta);
  assert_eq!(fitted.dim(), 4);
  assert!(fitted.sample(16, &Deterministic::new(1)).is_ok());
}

/// On exchangeable Clayton data the binary tree's parameters all sit near the one true θ, and the nested fit samples.
#[test]
fn nac_fit_on_a_flat_clayton_sample_keeps_every_theta_near_the_truth() {
  let truth = NestedArchimedean::new(
    BivariateFamily::Clayton,
    NacNode::leaf_group(2.0, vec![0, 1, 2]),
    3,
  )
  .unwrap();
  let u = truth.sample(4000, &Deterministic::new(9)).unwrap();
  let mut fitted = NestedArchimedean::unfitted(BivariateFamily::Clayton).unwrap();
  fitted.fit(u.view()).unwrap();
  fn thetas(node: &NacNode, out: &mut Vec<f64>) {
    out.push(node.theta);
    node.children.iter().for_each(|c| thetas(c, out));
  }
  let mut all = Vec::new();
  thetas(fitted.root().unwrap(), &mut all);
  assert_eq!(all.len(), 2, "{all:?}");
  assert!(all.iter().all(|t| (t - 2.0).abs() < 0.35), "{all:?}");
  assert_eq!(fitted.index_order().len(), 3);
  assert!(
    fitted
      .cdf(u.slice(ndarray::s![..5, ..]))
      .unwrap()
      .iter()
      .all(|c| c.is_finite())
  );
  assert!(fitted.sample(16, &Deterministic::new(1)).is_ok());
}

#[test]
fn nac_fit_rejects_negative_dependence_and_an_unfitted_copula_reports_it() {
  let neg = ndarray::Array2::from_shape_fn((200, 2), |(i, j)| {
    let u = (i + 1) as f64 / 201.0;
    if j == 0 { u } else { 1.0 - u }
  });
  let mut c = NestedArchimedean::unfitted(BivariateFamily::Clayton).unwrap();
  assert!(matches!(
    c.fit(neg.view()),
    Err(CopulaError::InvalidParameter { name: "tau", .. })
  ));
  assert_eq!(
    c.sample(4, &Deterministic::new(1)),
    Err(CopulaError::NotFitted)
  );
  assert_eq!(c.cdf(neg.view()), Err(CopulaError::NotFitted));
  assert_eq!(c.pdf(neg.view()), Err(CopulaError::NotFitted));
  assert_eq!(c.log_pdf(neg.view()), Err(CopulaError::NotFitted));
  assert_eq!(c.check_fit(neg.view()), Err(CopulaError::NotFitted));
  assert!(matches!(
    NestedArchimedean::unfitted(BivariateFamily::Joe),
    Err(CopulaError::Unsupported(_))
  ));
}

/// Average linkage on a hand-made τ̂ matrix: after `(0,1)` at 0.8 it joins `(2,3)` at 0.5, not `{0,1}` with 2 (a mean of
/// 0.4, where single linkage would see 0.7), and the root inverts the mean 0.35 of its four cross pairs.
#[test]
fn nac_fit_joins_the_clusters_with_the_largest_mean_cross_tau() {
  let tau = ndarray::array![
    [1.0, 0.8, 0.7, 0.3],
    [0.8, 1.0, 0.1, 0.3],
    [0.7, 0.1, 1.0, 0.5],
    [0.3, 0.3, 0.5, 1.0]
  ];
  let gumbel = |t: f64| 1.0 / (1.0 - t);
  let root = fit::agglomerate(Generator::Gumbel, tau.view()).unwrap();
  assert!(root.leaves.is_empty(), "{root:?}");
  assert_eq!(leaf_sets(&root), vec![vec![0, 1], vec![2, 3]]);
  assert!((root.children[0].theta - gumbel(0.8)).abs() < 1e-12);
  assert!((root.children[1].theta - gumbel(0.5)).abs() < 1e-12);
  assert!((root.theta - gumbel(0.35)).abs() < 1e-12, "{}", root.theta);
}

/// A nested Gumbel node reaches independence (τ = 0, θ = 1); a Clayton node needs τ > 0, as its frailty is `Γ(1/θ)`.
#[test]
fn nac_kendall_inversion_covers_each_nested_range() {
  assert_eq!(Generator::Gumbel.theta_from_tau(0.0), Ok(1.0));
  assert_eq!(Generator::Clayton.theta_from_tau(0.5), Ok(2.0));
  for tau in [0.0, -0.2, 1.0] {
    assert!(matches!(
      Generator::Clayton.theta_from_tau(tau),
      Err(CopulaError::InvalidParameter { name: "tau", .. })
    ));
  }
  assert!(Generator::Gumbel.theta_from_tau(-0.2).is_err());
}

/// Average linkage never lifts a parent's mean above its children's in exact arithmetic, but five equal τ̂ = 0.2509 make
/// the root's six-pair mean round above them, so the untrimmed tree would fail the sufficient nesting condition.
#[test]
fn nac_fit_trims_a_parent_whose_mean_rounds_above_its_children() {
  let tau = ndarray::Array2::from_shape_fn((5, 5), |(i, j)| if i == j { 1.0 } else { 0.2509 });
  let six = [0.2509_f64; 6].iter().sum::<f64>() / 6.0;
  for generator in [Generator::Clayton, Generator::Gumbel] {
    assert!(generator.theta_from_tau(six).unwrap() > generator.theta_from_tau(0.2509).unwrap());
    let root = fit::agglomerate(generator, tau.view()).unwrap();
    let split = root
      .children
      .iter()
      .map(|c| {
        let mut leaves = vec![];
        c.collect_leaves(&mut leaves);
        leaves.sort();
        leaves
      })
      .collect::<Vec<_>>();
    assert!(root.leaves.is_empty(), "{root:?}");
    assert_eq!(split, vec![vec![2, 3], vec![0, 1, 4]], "{root:?}");
    assert!(
      root.children.iter().all(|c| c.theta == root.theta),
      "{root:?}"
    );
    assert!(NestedArchimedean::new(generator.family(), root, 5).is_ok());
  }
}

/// The fit checks what every multivariate fit checks, in its order: columns, rows, range, then a τ-b for every column.
#[test]
fn nac_fit_checks_columns_rows_range_then_tau() {
  let fit = |x: ndarray::Array2<f64>| {
    NestedArchimedean::unfitted(BivariateFamily::Gumbel)
      .unwrap()
      .fit(x.view())
  };
  assert!(matches!(
    fit(ndarray::array![[2.0]]),
    Err(CopulaError::InvalidStructure(_))
  ));
  assert_eq!(
    fit(ndarray::array![[2.0, 0.5, 0.5]]),
    Err(CopulaError::InsufficientData { needed: 2, got: 1 })
  );
  assert_eq!(
    fit(ndarray::array![[0.2, f64::NAN], [0.4, 0.5], [0.6, 0.7]]),
    Err(CopulaError::MarginalOutOfRange)
  );
  let flat = ndarray::Array2::from_shape_fn((20, 3), |(i, j)| match j {
    0 => (i + 1) as f64 / 21.0,
    1 => ((7 * i) % 20 + 1) as f64 / 21.0,
    _ => 0.5,
  });
  assert_eq!(
    fit(flat),
    Err(CopulaError::InvalidStructure(
      "Kendall's tau-b is undefined for column 2: it is constant or holds a NaN".into()
    ))
  );
}

/// A Clayton `θ` whose frailty shape `1/θ` is not finite is a named error from `new`, at the root and below it.
#[test]
fn nac_clayton_theta_without_a_frailty_is_rejected_by_name() {
  for theta in [0.0, 1e-320] {
    let want = CopulaError::InvalidParameter {
      name: "theta",
      value: theta,
      constraint: "0 < 1 / theta < ∞".into(),
    };
    let flat = NestedArchimedean::new(
      BivariateFamily::Clayton,
      NacNode::leaf_group(theta, vec![0, 1]),
      2,
    );
    assert_eq!(flat.unwrap_err(), want);
    let mut nested = NacNode::leaf_group(2.0, vec![0]);
    nested.children.push(NacNode::leaf_group(theta, vec![1, 2]));
    let nested = NestedArchimedean::new(BivariateFamily::Clayton, nested, 3);
    assert_eq!(nested.unwrap_err(), want);
  }
}

/// A NaN or infinite `theta` anywhere in the tree is a named error from the constructor.
#[test]
fn nac_rejects_a_non_finite_theta_by_name() {
  for theta in [f64::NAN, f64::INFINITY] {
    for family in [BivariateFamily::Clayton, BivariateFamily::Gumbel] {
      let err =
        NestedArchimedean::new(family, NacNode::leaf_group(theta, vec![0, 1]), 2).unwrap_err();
      assert_eq!(
        err.to_string(),
        format!("theta must satisfy `theta.is_finite()`, got theta = {theta:?}")
      );
    }
    let mut root = NacNode::leaf_group(2.0, vec![0]);
    root.children.push(NacNode::leaf_group(theta, vec![1, 2]));
    let err = NestedArchimedean::new(BivariateFamily::Gumbel, root, 3).unwrap_err();
    assert_eq!(
      err.to_string(),
      format!("theta must satisfy `theta.is_finite()`, got theta = {theta:?}")
    );
  }
}
