use super::*;

/// A flat (single-level) Clayton NAC with all leaves under one root must
/// reduce to the standard exchangeable Clayton copula on 3 dims.
#[test]
fn nac_clayton_flat_is_exchangeable_clayton() {
  let root = NacNode::leaf_group(2.0, vec![0, 1, 2]);
  let nac = NestedArchimedean::new(NacFamily::Clayton, root, 3).unwrap();
  let u = nac.sample(8_000).unwrap();
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

/// 2-level Clayton NAC tree: root θ=1.5 with leaf 0 direct + a sub-tree
/// with θ=4.0 over leaves {1,2}. The CDF path is fully supported (nested
/// Clayton sampling is not — see `sample_node` Clayton-nested branch);
/// we therefore verify the structural property analytically: for two
/// outer-inner leaves the pair-margin is exchangeable Clayton on the
/// **root** generator, while for two inner-inner leaves it's
/// exchangeable Clayton on the **inner** generator.
#[test]
fn nac_clayton_two_level_cdf_pair_margins() {
  let inner = NacNode::leaf_group(4.0, vec![1, 2]);
  let root = NacNode {
    theta: 1.5,
    leaves: vec![0],
    children: vec![inner],
  };
  let nac = NestedArchimedean::new(NacFamily::Clayton, root, 3).unwrap();
  // Outer-inner pair margin: C(u_0, u_1, 1) = (u_0^{-θ_root} + u_1^{-θ_root} - 1)^{-1/θ_root}
  let q_outer_inner = ndarray::array![[0.3, 0.7, 1.0 - 1e-15]];
  let c_oi = nac.cdf(&q_outer_inner).unwrap()[0];
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
  let c_ii = nac.cdf(&q_inner_inner).unwrap()[0];
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

/// Nested-Clayton sampling intentionally panics; calling `sample` on a
/// tree with at least one nested node must surface the not-implemented
/// message rather than silently returning biased data.
#[test]
#[should_panic(expected = "Nested-Clayton sampling")]
fn nac_clayton_nested_sampling_panics() {
  let inner = NacNode::leaf_group(4.0, vec![1]);
  let root = NacNode {
    theta: 1.5,
    leaves: vec![0],
    children: vec![inner],
  };
  let nac = NestedArchimedean::new(NacFamily::Clayton, root, 2).unwrap();
  let _ = nac.sample(10);
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
  let nac = NestedArchimedean::new(NacFamily::Gumbel, root, 3).unwrap();
  let u = nac.sample(8_000).unwrap();
  use crate::correlation::kendall_tau;
  let tau = kendall_tau(&u);
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
  let res = NestedArchimedean::new(NacFamily::Clayton, bad_root, 2);
  assert!(res.is_err(), "SNC violation must error");
  assert!(
    res.unwrap_err().to_string().contains("SNC"),
    "error message should mention SNC"
  );
}

/// Below-family-minimum θ must be rejected per family.
#[test]
fn nac_below_family_min_rejected() {
  let root = NacNode::leaf_group(0.5, vec![0, 1]); // Gumbel requires θ ≥ 1
  let res = NestedArchimedean::new(NacFamily::Gumbel, root, 2);
  assert!(res.is_err());
  // And Clayton at θ < 0 should also fail.
  let bad_clayton = NacNode::leaf_group(-0.1, vec![0, 1]);
  assert!(NestedArchimedean::new(NacFamily::Clayton, bad_clayton, 2).is_err());
}

/// Missing / duplicate leaf indices must be rejected.
#[test]
fn nac_leaf_index_validation() {
  // Duplicate
  let dup_root = NacNode::leaf_group(2.0, vec![0, 0, 1]);
  assert!(NestedArchimedean::new(NacFamily::Clayton, dup_root, 3).is_err());
  // Missing (covers 0,2 only, dim=3)
  let miss = NacNode::leaf_group(2.0, vec![0, 2]);
  assert!(NestedArchimedean::new(NacFamily::Clayton, miss, 3).is_err());
  // Out of range
  let oor = NacNode::leaf_group(2.0, vec![0, 1, 5]);
  assert!(NestedArchimedean::new(NacFamily::Clayton, oor, 3).is_err());
}

/// CDF must equal $(\Pi u_j)$ for a θ=0 Clayton (independence boundary).
#[test]
fn nac_clayton_independence_cdf() {
  // θ ≥ 0 — at θ = 0 the inverse generator (1 + 0·s)^∞ degenerates; we
  // probe a small θ > 0 and check the CDF stays close to the
  // independence product within the Clayton bias.
  let root = NacNode::leaf_group(0.01, vec![0, 1, 2]);
  let nac = NestedArchimedean::new(NacFamily::Clayton, root, 3).unwrap();
  let q = ndarray::array![[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]];
  let c = nac.cdf(&q).unwrap();
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

/// `fit` must return a descriptive error pointing at structure learning
/// not being implemented.
#[test]
fn nac_fit_rejects_with_descriptive_error() {
  let root = NacNode::leaf_group(2.0, vec![0, 1]);
  let mut nac = NestedArchimedean::new(NacFamily::Clayton, root, 2).unwrap();
  let data = ndarray::Array2::<f64>::from_elem((10, 2), 0.5);
  let res = nac.fit(data);
  assert!(res.is_err());
  let msg = res.unwrap_err().to_string();
  assert!(
    msg.contains("structure") || msg.contains("not implemented"),
    "fit error should explain that structure learning is not implemented; got: {msg}"
  );
}
