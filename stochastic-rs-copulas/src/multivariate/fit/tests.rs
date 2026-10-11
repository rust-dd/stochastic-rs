use ndarray::Array2;
use ndarray::ArrayView2;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::multivariate::gaussian::GaussianMultivariate;
use crate::traits::MultivariateExt;

fn first_tree(fit: &VineFit) -> &[PairCopula] {
  match &fit.vine {
    RVine::D(d) => &d.pair_copulas()[0],
    RVine::C(c) => &c.pair_copulas()[0],
  }
}

/// Two variables, candidates that do not nest Clayton: the selection picks it and recovers θ by maximum likelihood.
/// BB1 and BB7 contain Clayton at a boundary that AIC cannot reject reliably, so `VINE_CANDIDATES` has its own test.
#[test]
fn two_dimensional_fit_recovers_clayton() {
  let truth = DVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
  let u = truth.sample(3000, &Deterministic::new(7)).unwrap();
  let families = [
    BivariateFamily::Independence,
    BivariateFamily::Gaussian,
    BivariateFamily::TCopula,
    BivariateFamily::Clayton,
    BivariateFamily::Frank,
  ];
  let fit = fit_vine(
    u.view(),
    VineStructure::DVine,
    &families,
    SelectionCriterion::Aic,
  )
  .unwrap();
  assert_eq!(fit.families[0][0], BivariateFamily::Clayton);
  let PairCopula::Clayton { theta } = first_tree(&fit)[0] else {
    panic!("clayton")
  };
  assert!((theta - 2.0).abs() < 0.25, "theta {theta}");
  assert_eq!(fit.parameter_count, 1);
  assert!((fit.aic - (-2.0 * fit.log_likelihood + 2.0)).abs() < 1e-12);
  assert!(fit.bic > fit.aic);
}

/// Over `VINE_CANDIDATES` the pick is Clayton, BB1 or BB7 (the families that contain it) and the fitted vine keeps the truth's τ = 0.5.
#[test]
fn two_dimensional_fit_over_all_families_keeps_the_clayton_dependence() {
  let truth = DVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
  let u = truth.sample(3000, &Deterministic::new(7)).unwrap();
  let fit = fit_vine(
    u.view(),
    VineStructure::DVine,
    BivariateFamily::VINE_CANDIDATES,
    SelectionCriterion::Aic,
  )
  .unwrap();
  assert!(
    matches!(
      fit.families[0][0],
      BivariateFamily::Clayton | BivariateFamily::Bb1 | BivariateFamily::Bb7
    ),
    "{:?}",
    fit.families
  );
  let refit = fit.vine.sample(5000, &Deterministic::new(11)).unwrap();
  let tau = kendall_tau(refit.view()).unwrap()[[0, 1]];
  assert!((tau - 0.5).abs() < 0.05, "fitted tau {tau}");
}

/// A four-dimensional D-vine with a Gaussian, a Clayton and a Frank edge in
/// the first tree and independence above: the fitted model reaches at
/// least the true model's likelihood up to sampling noise, the strong
/// first-tree edges are recovered as neighbours in the fitted order, and
/// the second and third trees select independence.
#[test]
fn dvine_fit_recovers_the_first_tree_families() {
  let trees = vec![
    vec![
      PairCopula::Gaussian { rho: 0.7 },
      PairCopula::Clayton { theta: 3.0 },
      PairCopula::Frank { theta: 6.0 },
    ],
    vec![PairCopula::Independence; 2],
    vec![PairCopula::Independence],
  ];
  let truth = DVine::new(4, trees).unwrap();
  let u = truth.sample(4000, &Deterministic::new(11)).unwrap();
  let fit = fit_vine(
    u.view(),
    VineStructure::DVine,
    BivariateFamily::VINE_CANDIDATES,
    SelectionCriterion::Bic,
  )
  .unwrap();
  let order = fit.vine.order();
  let neighbours = |a: usize, b: usize| {
    let (pa, pb) = (
      order.iter().position(|&x| x == a).unwrap(),
      order.iter().position(|&x| x == b).unwrap(),
    );
    pa.abs_diff(pb) == 1
  };
  assert!(
    neighbours(0, 1) && neighbours(1, 2) && neighbours(2, 3),
    "order {order:?}"
  );
  let mut chosen: Vec<(BivariateFamily, BivariateFamily)> = Vec::new();
  for i in 0..3 {
    let edge = (order[i].min(order[i + 1]), order[i].max(order[i + 1]));
    let expected = match edge {
      (0, 1) => BivariateFamily::Gaussian,
      (1, 2) => BivariateFamily::Clayton,
      (2, 3) => BivariateFamily::Frank,
      other => panic!("unexpected edge {other:?}"),
    };
    chosen.push((expected, fit.families[0][i]));
  }
  for (expected, got) in &chosen {
    assert_eq!(got, expected, "{chosen:?}");
  }
  assert!(
    fit.families[1]
      .iter()
      .chain(&fit.families[2])
      .all(|f| *f == BivariateFamily::Independence),
    "{:?}",
    fit.families
  );
  let true_ll: f64 = truth.pdf(u.view()).unwrap().mapv(f64::ln).sum();
  assert!(
    fit.log_likelihood > true_ll - 0.02 * true_ll.abs(),
    "fit {} truth {true_ll}",
    fit.log_likelihood
  );
  let vine_ll: f64 = fit.vine.pdf(u.view()).unwrap().mapv(f64::ln).sum();
  assert!(
    (vine_ll - fit.log_likelihood).abs() < 1e-6 * vine_ll.abs().max(1.0),
    "{vine_ll} vs {}",
    fit.log_likelihood
  );
}

/// Gaussian data: the C-vine picks the Gaussian family on every edge of
/// the first tree, roots the strongest variable, and beats the
/// independence vine by AIC.
#[test]
fn cvine_fit_on_gaussian_data_selects_gaussian_pairs() {
  let corr = ndarray::array![[1.0, 0.8, 0.6], [0.8, 1.0, 0.5], [0.6, 0.5, 1.0]];
  let truth = GaussianMultivariate::new(corr);
  let u = truth.sample(3000, &Deterministic::new(5)).unwrap();
  let families = [
    BivariateFamily::Independence,
    BivariateFamily::Gaussian,
    BivariateFamily::Clayton,
    BivariateFamily::Frank,
  ];
  let fit = fit_vine(
    u.view(),
    VineStructure::CVine,
    &families,
    SelectionCriterion::Aic,
  )
  .unwrap();
  assert_eq!(fit.vine.order()[0], 0, "root {:?}", fit.vine.order());
  assert!(
    fit.families[0]
      .iter()
      .all(|f| *f == BivariateFamily::Gaussian),
    "{:?}",
    fit.families
  );
  for pair in first_tree(&fit) {
    let PairCopula::Gaussian { rho } = pair else {
      panic!("gaussian")
    };
    assert!(*rho > 0.5);
  }
  let independence = CVine::independence(3).unwrap();
  let ll_independence: f64 = independence.pdf(u.view()).unwrap().mapv(f64::ln).sum();
  assert!(
    fit.aic < -2.0 * ll_independence,
    "aic {} vs independence {}",
    fit.aic,
    -2.0 * ll_independence
  );
  let vine_ll: f64 = fit.vine.pdf(u.view()).unwrap().mapv(f64::ln).sum();
  assert!(
    (vine_ll - fit.log_likelihood).abs() < 1e-6 * vine_ll.abs().max(1.0),
    "{vine_ll} vs {}",
    fit.log_likelihood
  );
}

#[test]
fn dvine_order_is_a_maximal_path_on_tau() {
  let tau = ndarray::array![
    [1.0, 0.1, 0.6, 0.2],
    [0.1, 1.0, 0.05, 0.7],
    [0.6, 0.05, 1.0, 0.3],
    [0.2, 0.7, 0.3, 1.0]
  ];
  let order = dvine_order(tau.view());
  assert_eq!(order.len(), 4);
  let mut sorted = order.clone();
  sorted.sort();
  assert_eq!(sorted, vec![0, 1, 2, 3]);
  let strength: f64 = order.windows(2).map(|w| tau[(w[0], w[1])].abs()).sum();
  assert!(
    strength >= 0.6 + 0.3 + 0.7 - 1e-12,
    "{order:?} strength {strength}"
  );
}

#[test]
fn fit_rejects_degenerate_inputs() {
  let one = Array2::<f64>::zeros((50, 1));
  assert!(
    fit_vine(
      one.view(),
      VineStructure::DVine,
      BivariateFamily::VINE_CANDIDATES,
      SelectionCriterion::Aic
    )
    .is_err()
  );
  let short = Array2::<f64>::zeros((5, 3));
  assert!(
    fit_vine(
      short.view(),
      VineStructure::CVine,
      BivariateFamily::VINE_CANDIDATES,
      SelectionCriterion::Aic
    )
    .is_err()
  );
}

/// A candidate list that cannot represent an edge (Clayton on negative dependence, or no candidate at all) is `Unsupported`.
#[test]
fn an_edge_no_candidate_represents_is_unsupported() {
  let u = Array2::from_shape_fn((40, 2), |(i, j)| {
    if j == 0 {
      (i + 1) as f64 / 41.0
    } else {
      (40 - i) as f64 / 41.0
    }
  });
  for structure in [VineStructure::DVine, VineStructure::CVine] {
    for families in [&[BivariateFamily::Clayton][..], &[]] {
      let err = fit_vine(u.view(), structure, families, SelectionCriterion::Aic).unwrap_err();
      assert!(
        matches!(err, CopulaError::Unsupported(_)),
        "{structure:?} {families:?}: {err:?}"
      );
    }
  }
}

/// `estimate` returns `Some` on positively dependent data for exactly the `VINE_CANDIDATES` and `None` for the rest.
#[test]
fn estimate_covers_exactly_the_vine_candidates() {
  let truth = DVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
  let u = truth.sample(1000, &Deterministic::new(7)).unwrap();
  let (x, y) = (u.column(0).to_vec(), u.column(1).to_vec());
  for &family in BivariateFamily::ALL {
    let candidate = BivariateFamily::VINE_CANDIDATES.contains(&family);
    assert_eq!(estimate(family, &x, &y).is_some(), candidate, "{family:?}");
  }
}

/// A constant column has no τ-b to rank by, so both structures name it instead of repeating a variable in the order.
#[test]
fn a_constant_column_is_rejected_by_both_structures() {
  let u = Array2::from_shape_fn(
    (40, 3),
    |(i, j)| {
      if j == 2 { 0.5 } else { (i + 1) as f64 / 41.0 }
    },
  );
  for structure in [VineStructure::DVine, VineStructure::CVine] {
    let got = fit_vine(
      u.view(),
      structure,
      BivariateFamily::VINE_CANDIDATES,
      SelectionCriterion::Aic,
    );
    assert!(
      matches!(&got, Err(CopulaError::InvalidStructure(why)) if why.contains("undefined for column 2:")),
      "{structure:?}: {got:?}"
    );
  }
}

fn path_weight(tau: ArrayView2<f64>, order: &[usize]) -> f64 {
  order.windows(2).map(|w| tau[(w[0], w[1])].abs()).sum()
}

/// Greedy extension from the strongest pair, |τ| 0.89 at {2, 3}, ends at 2.34; the heaviest path 0-2-3-1 weighs 2.47.
#[test]
fn the_first_dvine_tree_is_the_heaviest_path() {
  let tau = ndarray::array![
    [1.0, 0.53, 0.81, 0.61],
    [0.53, 1.0, 0.84, 0.77],
    [0.81, 0.84, 1.0, 0.89],
    [0.61, 0.77, 0.89, 1.0]
  ];
  assert!((path_weight(tau.view(), &greedy_path(tau.view())) - 2.34).abs() < 1e-12);
  let order = dvine_order(tau.view());
  assert_eq!(order, vec![0, 2, 3, 1]);
  assert!((path_weight(tau.view(), &order) - 2.47).abs() < 1e-12);
}

fn every_order(d: usize, prefix: &mut Vec<usize>, visit: &mut impl FnMut(&[usize])) {
  if prefix.len() == d {
    visit(prefix);
    return;
  }
  for k in 0..d {
    if !prefix.contains(&k) {
      prefix.push(k);
      every_order(d, prefix, visit);
      prefix.pop();
    }
  }
}

/// Held–Karp against all 720 orders of six columns on twenty pseudo-random τ matrices.
#[test]
fn the_heaviest_path_is_the_best_of_every_order() {
  let mut state = 0x2545_f491_4f6c_dd1d_u64;
  let mut uniform = || {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    (state >> 11) as f64 / (1_u64 << 53) as f64
  };
  let d = 6;
  for _ in 0..20 {
    let mut tau = Array2::<f64>::eye(d);
    for i in 0..d {
      for j in (i + 1)..d {
        let t = 2.0 * uniform() - 1.0;
        tau[[i, j]] = t;
        tau[[j, i]] = t;
      }
    }
    let mut best = f64::NEG_INFINITY;
    every_order(d, &mut Vec::with_capacity(d), &mut |order| {
      best = best.max(path_weight(tau.view(), order));
    });
    let order = dvine_order(tau.view());
    assert!(
      (path_weight(tau.view(), &order) - best).abs() < 1e-12,
      "{order:?} weighs {} of {best}",
      path_weight(tau.view(), &order)
    );
  }
}

#[test]
fn dvine_order_extends_greedily_beyond_eight_columns() {
  let tau = Array2::from_shape_fn((9, 9), |(i, j)| {
    if i == j {
      1.0
    } else {
      1.0 / (1.0 + i.abs_diff(j) as f64) + 0.01 * ((i * j) % 5) as f64
    }
  });
  assert_eq!(dvine_order(tau.view()), greedy_path(tau.view()));
}

/// The searches run over the τ each family reaches inside its θ bounds, cut to the Kendall start's `|τ| ≤ 0.99`.
#[test]
fn the_likelihood_searches_follow_the_family_bounds() {
  assert_eq!(tau_range(&Clayton::default()), (0.0, 0.99));
  assert_eq!(tau_range(&GaussianCopula::default()), (-0.99, 0.99));
  assert_eq!(tau_range(&Frank::default()), (-0.99, 0.99));
  assert_eq!(tau_range(&TCopula::default()), (-0.99, 0.99));
}

/// A conditioned column without τ-b stops the C-vine before it picks the next root.
#[test]
fn a_constant_conditioned_column_stops_the_cvine() {
  let rising: Vec<f64> = (1..=20).map(|i| i as f64 / 21.0).collect();
  let falling: Vec<f64> = rising.iter().rev().copied().collect();
  let values = vec![(0, rising), (2, vec![0.5; 20]), (3, falling)];
  let got = next_root(&values, &[1]);
  assert!(
    matches!(&got, Err(CopulaError::InvalidStructure(why)) if why.contains("columns [0, 2, 3] given columns [1]")),
    "{got:?}"
  );
}

/// Each one-parameter estimate maximises the pair likelihood, here on Clayton data, and is no worse than its Kendall start.
#[test]
fn one_parameter_pairs_maximise_the_likelihood() {
  let truth = DVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
  let sample = truth.sample(1000, &Deterministic::new(7)).unwrap();
  let (u, v) = (sample.column(0).to_vec(), sample.column(1).to_vec());
  let tau = tau_of(&u, &v);
  let gaussian = |rho| PairCopula::Gaussian { rho };
  let clayton = |theta| PairCopula::Clayton { theta };
  let frank = |theta| PairCopula::Frank { theta };
  let cases: [(BivariateFamily, &dyn Fn(f64) -> PairCopula, f64); 3] = [
    (
      BivariateFamily::Gaussian,
      &gaussian,
      GaussianCopula::default().theta_from_tau(tau).unwrap(),
    ),
    (
      BivariateFamily::Clayton,
      &clayton,
      Clayton::default().theta_from_tau(tau).unwrap(),
    ),
    (
      BivariateFamily::Frank,
      &frank,
      Frank::default().theta_from_tau(tau).unwrap(),
    ),
  ];
  for (family, pair, start) in cases {
    let theta = match estimate(family, &u, &v).unwrap() {
      PairCopula::Gaussian { rho } => rho,
      PairCopula::Clayton { theta } | PairCopula::Frank { theta } => theta,
      other => panic!("{other:?}"),
    };
    let ll = |t: f64| log_likelihood(&pair(t), &u, &v);
    assert!(
      ll(theta) >= ll(start),
      "{family:?}: {theta} below the start {start}"
    );
    for step in [-1e-4, 1e-4] {
      assert!(
        ll(theta) >= ll(theta * (1.0 + step)),
        "{family:?}: {theta} is not a maximum ({} at {})",
        ll(theta * (1.0 + step)),
        theta * (1.0 + step)
      );
    }
  }
}

/// The Student t maximises the likelihood in (ρ, ν) on t data, and gives way to the Gaussian past ν̂ = 30.
#[test]
fn student_t_pairs_maximise_the_likelihood_and_give_way_past_thirty() {
  let sample = TCopula::new(0.6, 4.0)
    .sample(1000, &Deterministic::new(3))
    .unwrap();
  let (u, v) = (sample.column(0).to_vec(), sample.column(1).to_vec());
  let Some(PairCopula::StudentT { rho, nu }) = estimate(BivariateFamily::TCopula, &u, &v) else {
    panic!("no Student t on t data")
  };
  let ll = |rho: f64, nu: f64| log_likelihood(&PairCopula::StudentT { rho, nu }, &u, &v);
  for (drho, dnu) in [(1e-3, 0.0), (-1e-3, 0.0), (0.0, 0.05), (0.0, -0.05)] {
    assert!(
      ll(rho, nu) >= ll(rho + drho, nu + dnu),
      "(ρ, ν) = ({rho}, {nu}) is not a maximum"
    );
  }
  let gaussian = GaussianCopula::new(0.6)
    .sample(3000, &Deterministic::new(3))
    .unwrap();
  let fitted = estimate(
    BivariateFamily::TCopula,
    &gaussian.column(0).to_vec(),
    &gaussian.column(1).to_vec(),
  );
  assert_eq!(fitted, None, "{fitted:?}");
}
