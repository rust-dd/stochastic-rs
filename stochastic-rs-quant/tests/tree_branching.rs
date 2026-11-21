//! The short-rate trees' branching: a probability vector on every branch for any finite input,
//! and each factor's exact Ornstein–Uhlenbeck mean and variance at every level, on any grid.

use stochastic_rs_quant::calibration::BlackKarasinskiParams;
use stochastic_rs_quant::calibration::G2ppParams;
use stochastic_rs_quant::calibration::HullWhiteParams;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::lattice::BlackKarasinskiTree;
use stochastic_rs_quant::lattice::BlackKarasinskiTreeModel;
use stochastic_rs_quant::lattice::CurveFittedHullWhiteTree;
use stochastic_rs_quant::lattice::G2ppTree;
use stochastic_rs_quant::lattice::G2ppTreeModel;
use stochastic_rs_quant::lattice::HullWhiteTree;
use stochastic_rs_quant::lattice::HullWhiteTreeModel;
use stochastic_rs_quant::lattice::TrinomialTree;

fn hw(a: f64, sigma: f64, r0: f64, theta: f64, horizon: f64, steps: usize) -> TrinomialTree<f64> {
  let params = HullWhiteParams {
    mean_reversion: a,
    sigma,
    initial_rate: r0,
    theta,
  };
  HullWhiteTree::new(HullWhiteTreeModel::new(params), horizon, steps).tree
}

fn bk(a: f64, sigma: f64, r0: f64, long: f64, horizon: f64, steps: usize) -> TrinomialTree<f64> {
  let params = BlackKarasinskiParams {
    mean_reversion: a,
    sigma,
    initial_rate: r0,
    long_run_rate: long,
  };
  BlackKarasinskiTree::new(BlackKarasinskiTreeModel::new(params), horizon, steps).tree
}

fn g2(a: f64, b: f64, sx: f64, sy: f64, horizon: f64, steps: usize) -> G2ppTree<f64> {
  let params = G2ppParams {
    mean_reversion_x: a,
    mean_reversion_y: b,
    sigma_x: sx,
    sigma_y: sy,
    rho: -0.7,
    phi: 0.03,
  };
  G2ppTree::new(G2ppTreeModel::new(0.0, 0.0, params), horizon, steps)
}

/// Brigo & Mercurio's nearest node keeps `|η| ≤ Δx/2`, so every probability is at least 1/24.
fn defects(tree: &TrinomialTree<f64>) -> Option<String> {
  let branches = tree.branches.iter().flatten().map(|b| b.probabilities());
  let (mut negative, mut short, mut worst, mut off) = (0, 0, f64::INFINITY, 0.0_f64);
  for p in branches {
    negative += usize::from(p.iter().any(|&q| q < 0.0));
    short += usize::from(!p.iter().all(|&q| q >= 1.0 / 24.0 - 1e-12));
    worst = worst.min(p[0]).min(p[1]).min(p[2]);
    off = off.max((p[0] + p[1] + p[2] - 1.0).abs());
  }
  let total = tree.branches.iter().map(Vec::len).sum::<usize>();
  (short > 0 || off.is_nan() || off >= 1e-15).then(|| {
    format!(
      "{negative} of {total} negative, {short} below 1/24, worst {worst:e}, |sum - 1| {off:e}"
    )
  })
}

/// Every tree the crate's tests, benches and examples build, and the calibrators' trials at their
/// floors and at a reversion past two steps a year.
#[test]
fn every_in_tree_tree_branches_on_probabilities() {
  let rows = [
    ("tree_models HW", hw(0.1, 0.012, 0.03, 0.05, 5.0, 20)),
    ("bench HW", hw(0.4, 0.02, 0.03, 0.05, 5.0, 400)),
    ("bench HW Bermudan", hw(0.1, 0.015, 0.04, 0.04, 10.0, 100)),
    ("deep-OTM HW", hw(0.05, 0.005, 0.02, 0.02, 5.0, 40)),
    ("callable bond HW", hw(0.3, 0.01, 0.04, 0.04, 3.0, 360)),
    ("deterministic HW", hw(0.3, 1e-9, 0.02, 0.02, 3.0, 36)),
    ("tree_models BK", bk(0.6, 0.15, 0.03, 0.05, 5.0, 20)),
    ("low-vol BK", bk(0.7, 1e-6, 0.03, 0.05, 3.0, 240)),
    ("bench BK", bk(0.6, 0.015, 0.03, 0.05, 5.0, 400)),
    (
      "BK at the calibrator's floor",
      bk(0.1, 1e-6, 0.03, 0.03, 6.0, 48),
    ),
  ];
  let mut failures = rows
    .iter()
    .filter_map(|(name, tree)| defects(tree).map(|d| format!("{name}: {d}")))
    .collect::<Vec<_>>();
  let curve = DiscountCurve::from_zero_rates(
    ndarray::array![1.0, 5.0].view(),
    ndarray::array![0.03, 0.035].view(),
    InterpolationMethod::LinearOnZeroRates,
  );
  let fitted = CurveFittedHullWhiteTree::new(0.08, 0.012, &curve, 5.0, 200);
  failures.extend(defects(&fitted.tree).map(|d| format!("curve-fitted HW: {d}")));
  for (name, tree) in [
    ("bench G2++", g2(0.4, 0.8, 0.02, 0.015, 5.0, 240)),
    (
      "vanishing second factor",
      g2(0.1, 50.0, 0.012, 1e-10, 5.0, 20),
    ),
    (
      "G2++ at the sigma floor",
      g2(0.5, 0.05, 1e-6, 0.006, 3.0, 12),
    ),
    ("G2++ with b dt = 2.25", g2(0.5, 9.0, 0.01, 0.006, 3.0, 12)),
    ("G2++ with a dt = 7.5", g2(30.0, 0.05, 0.01, 0.006, 3.0, 12)),
  ] {
    failures.extend(defects(&tree.x_tree).map(|d| format!("{name} x: {d}")));
    failures.extend(defects(&tree.y_tree).map(|d| format!("{name} y: {d}")));
  }
  assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn no_branch_falls_below_a_twenty_fourth_for_any_finite_input() {
  let reversions = [0.0, 1e-300, 1e-12, 0.04, 0.5, 7.264, 8.0, 50.0, 1e6, 1e300];
  let sigmas = [0.0, 1e-300, 1e-160, 1e-10, 0.01, 1.0, 1e100];
  let mut failures = Vec::new();
  for a in reversions {
    for sigma in sigmas {
      for (horizon, steps) in [(1e-9, 3), (0.25, 1), (5.0, 12)] {
        let name = format!("a {a:e}, sigma {sigma:e}, horizon {horizon}, steps {steps}");
        failures.extend(
          defects(&hw(a, sigma, 1e3, -5.0, horizon, steps)).map(|d| format!("HW {name}: {d}")),
        );
        failures.extend(
          defects(&bk(a, sigma, 1e-8, 0.5, horizon, steps)).map(|d| format!("BK {name}: {d}")),
        );
      }
    }
  }
  assert!(
    failures.is_empty(),
    "{} failures: {failures:#?}",
    failures.len()
  );
}

/// Forward mass of a one-factor tree, then the state's mean and variance at the horizon.
fn terminal_moments(tree: &TrinomialTree<f64>) -> (f64, f64) {
  let mut mass = vec![1.0];
  for (level, branches) in tree.branches.iter().enumerate() {
    let mut next = vec![0.0; 2 * level + 3];
    for (node, branch) in branches.iter().enumerate() {
      for (offset, p) in branch.probabilities().iter().enumerate() {
        next[branch.center_index + offset - 1] += mass[node] * p;
      }
    }
    mass = next;
  }
  let states = tree.states.last().unwrap();
  let mean = mass.iter().zip(states).map(|(m, x)| m * x).sum::<f64>();
  let variance = mass
    .iter()
    .zip(states)
    .map(|(m, x)| m * (x - mean).powi(2))
    .sum::<f64>();
  (mean, variance)
}

/// Exact moments per step keep the coarse grid exact in mean and variance, `σ²(1 − e^{−2aT})/(2a)`.
#[test]
fn each_factor_keeps_its_exact_mean_and_variance_on_the_calibrators_coarse_grid() {
  let exact = |a: f64, sigma: f64, t: f64| sigma * sigma * -(-2.0 * a * t).exp_m1() / (2.0 * a);
  let tree = g2(0.5, 9.0, 0.01, 0.006, 3.0, 12);
  for (factor, a, sigma) in [(&tree.x_tree, 0.5, 0.01), (&tree.y_tree, 9.0, 0.006)] {
    let (mean, variance) = terminal_moments(factor);
    assert!(mean.abs() < 1e-15 * variance.sqrt(), "mean {mean}");
    let want = exact(a, sigma, 3.0);
    assert!(
      (variance / want - 1.0).abs() < 1e-12,
      "{variance} vs {want}"
    );
  }
  let (mean, variance) = terminal_moments(&hw(0.4, 0.02, 0.03, 0.05, 2.0, 8));
  let want_mean = 0.05 + (0.03 - 0.05) * (-0.4_f64 * 2.0).exp();
  assert!((mean - want_mean).abs() < 1e-15, "{mean} vs {want_mean}");
  assert!(
    (variance / exact(0.4, 0.02, 2.0) - 1.0).abs() < 1e-12,
    "{variance}"
  );
}
