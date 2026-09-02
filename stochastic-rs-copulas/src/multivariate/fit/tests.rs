use ndarray::Array2;

use super::*;
use crate::multivariate::gaussian::GaussianMultivariate;
use crate::traits::MultivariateExt;

fn first_tree(fit: &VineFit) -> &[PairCopula] {
  match &fit.vine {
    RVine::D(d) => &d.pair_copulas()[0],
    RVine::C(c) => &c.pair_copulas()[0],
  }
}

/// Two variables: the fit is the bivariate family selection and recovers
/// the Clayton parameter by Kendall inversion.
#[test]
fn two_dimensional_fit_recovers_clayton() {
  let truth = DVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).unwrap();
  let u = truth.sample_with_seed(3000, 7).unwrap();
  let fit = fit_vine(
    &u,
    VineStructure::DVine,
    &PairFamily::ALL,
    SelectionCriterion::Aic,
  )
  .unwrap();
  assert_eq!(fit.families[0][0], PairFamily::Clayton);
  let PairCopula::Clayton { theta } = first_tree(&fit)[0] else {
    panic!("clayton")
  };
  assert!((theta - 2.0).abs() < 0.25, "theta {theta}");
  assert_eq!(fit.parameter_count, 1);
  assert!((fit.aic - (-2.0 * fit.log_likelihood + 2.0)).abs() < 1e-12);
  assert!(fit.bic > fit.aic);
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
  let u = truth.sample_with_seed(4000, 11).unwrap();
  let fit = fit_vine(
    &u,
    VineStructure::DVine,
    &PairFamily::ALL,
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
  let mut chosen: Vec<(PairFamily, PairFamily)> = Vec::new();
  for i in 0..3 {
    let edge = (order[i].min(order[i + 1]), order[i].max(order[i + 1]));
    let expected = match edge {
      (0, 1) => PairFamily::Gaussian,
      (1, 2) => PairFamily::Clayton,
      (2, 3) => PairFamily::Frank,
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
      .all(|f| *f == PairFamily::Independence),
    "{:?}",
    fit.families
  );
  let true_ll: f64 = truth.pdf(&u).unwrap().mapv(f64::ln).sum();
  assert!(
    fit.log_likelihood > true_ll - 0.02 * true_ll.abs(),
    "fit {} truth {true_ll}",
    fit.log_likelihood
  );
  let vine_ll: f64 = fit
    .vine
    .pdf(&u.select(ndarray::Axis(1), &fit.order))
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
  let truth = GaussianMultivariate::new_with_corr(corr).unwrap();
  let u = truth.sample_with_seed(3000, 5).unwrap();
  let families = [
    PairFamily::Independence,
    PairFamily::Gaussian,
    PairFamily::Clayton,
    PairFamily::Frank,
  ];
  let fit = fit_vine(&u, VineStructure::CVine, &families, SelectionCriterion::Aic).unwrap();
  assert_eq!(fit.order[0], 0, "root {:?}", fit.order);
  assert!(
    fit.families[0].iter().all(|f| *f == PairFamily::Gaussian),
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
  let ll_independence: f64 = independence.pdf(&u).unwrap().mapv(f64::ln).sum();
  assert!(
    fit.aic < -2.0 * ll_independence,
    "aic {} vs independence {}",
    fit.aic,
    -2.0 * ll_independence
  );
  let vine_ll: f64 = fit
    .vine
    .pdf(&u.select(ndarray::Axis(1), &fit.order))
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
  let order = dvine_order(&tau);
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
      &one,
      VineStructure::DVine,
      &PairFamily::ALL,
      SelectionCriterion::Aic
    )
    .is_err()
  );
  let short = Array2::<f64>::zeros((5, 3));
  assert!(
    fit_vine(
      &short,
      VineStructure::CVine,
      &PairFamily::ALL,
      SelectionCriterion::Aic
    )
    .is_err()
  );
}
