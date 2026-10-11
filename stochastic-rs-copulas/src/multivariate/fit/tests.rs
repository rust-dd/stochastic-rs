use ndarray::Array2;
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

/// Two variables, candidates that do not nest Clayton: the selection picks it and recovers θ by Kendall inversion.
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
  let order = &fit.order;
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
  let vine_ll: f64 = fit
    .vine
    .pdf(u.select(ndarray::Axis(1), &fit.order).view())
    .unwrap()
    .mapv(f64::ln)
    .sum();
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
  assert_eq!(fit.order[0], 0, "root {:?}", fit.order);
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
  let vine_ll: f64 = fit
    .vine
    .pdf(u.select(ndarray::Axis(1), &fit.order).view())
    .unwrap()
    .mapv(f64::ln)
    .sum();
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
