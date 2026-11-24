use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ou::Ou;

use super::*;
use crate::qmle::fit_ou_mle;
use crate::qmle::test_support::cir;
use crate::qmle::test_support::simulate_cir;
use crate::qmle::test_support::simulate_ou;

fn ou(start: [f64; 3]) -> Ou<f64, Deterministic> {
  let seed = Deterministic::new(0);
  Ou::new(start[0], start[1], start[2], 2, None, None, seed)
}

type Bounds = Vec<(f64, f64)>;

fn fit_ou(
  path: &Array1<f64>,
  dt: f64,
  start: [f64; 3],
  density: DensityApprox,
  b: Bounds,
) -> MleFit {
  fit_mle(&mut ou(start), path.view(), dt, density, Some(b)).unwrap()
}

/// OU paths times `1e9` and `1e12`, from a `σ` 13 to 18 orders below the estimate, a level 17
/// times off, and from the path's own scale: each fit converges at the closed form.
#[test]
fn a_fit_of_data_near_1e9_and_1e12_converges_at_the_closed_form() {
  let laws = [
    ([8.0, 0.03, 0.02], 1.0 / 12.0),
    ([2.0, 1.0, 0.3], 1.0 / 252.0),
  ];
  for (law, dt) in laws {
    for c in [1e9, 1e12] {
      for seed in 101..=104 {
        let path = simulate_ou(law[0], law[1], law[2], law[1], dt, 1000, seed).mapv(|v| c * v);
        let closed = fit_ou_mle(path.view(), dt).unwrap();
        let bounds = vec![(1e-4, 50.0), (-100.0 * c, 100.0 * c), (1e-9, 100.0 * c)];
        let level = law[1] * c;
        for start in [
          [1.0, level, 1e-6],
          [1.0, 17.0 * level, 1e-6],
          [1.0, level, 1e-3 * c],
        ] {
          let fit = fit_ou(&path, dt, start, DensityApprox::Exact, bounds.clone());
          let short = closed.log_likelihood - fit.log_likelihood;
          assert!(
            fit.converged && short.abs() < 1e-3,
            "θ = {} ×{c:e} seed {seed} from {start:?}: converged {}, {short} below at {}",
            law[0],
            fit.converged,
            fit.params
          );
        }
      }
    }
  }
}

/// The fit of `c x` at a step of `k Δ`, from the mapped start in the mapped box, is the mapped fit:
/// the level times `c`, `σ` times `s/√k` and the speed over `k`.
fn assert_mapped(name: &str, fit: impl Fn(f64, f64, f64) -> MleFit, root: bool) {
  let unit = fit(1.0, 1.0, 1.0);
  assert!(unit.converged, "{name}");
  let units = [
    (1e-100, 1.0),
    (1e-12, 1.0),
    (1e12, 1.0),
    (1e100, 1.0),
    (1.0, 252.0),
    (1e12, 252.0),
  ];
  for (c, k) in units {
    let s = if root { f64::sqrt(c) } else { c } / f64::sqrt(k);
    let scaled = fit(c, s, k);
    assert_eq!(scaled.converged, unit.converged, "{name} ×{c:e}, step ×{k}");
    let mapped = [
      scaled.params[0] * k,
      scaled.params[1] / c,
      scaled.params[2] / s,
    ];
    for (got, want) in mapped.iter().zip(&unit.params) {
      assert!(
        (got / want - 1.0).abs() < 1e-4,
        "{name} ×{c:e}, step ×{k}: {mapped:?} for {}",
        unit.params
      );
    }
  }
}

/// The search runs in the units of its start, so the unit of the path and of time is no part of
/// the fit, from `1e-100` to `1e100` and with the step in days.
#[test]
fn a_fit_is_the_same_fit_in_any_unit_of_the_path_and_of_time() {
  let dt = 1.0 / 252.0;
  let x = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, 501);
  for density in [DensityApprox::Exact, DensityApprox::Euler] {
    let fit = |c: f64, s: f64, k: f64| {
      let bounds = vec![
        (1e-4 / k, 50.0 / k),
        (-10.0 * c, 10.0 * c),
        (1e-6 * s, 10.0 * s),
      ];
      let start = [1.0 / k, 0.5 * c, 0.5 * s];
      fit_ou(&x.mapv(|v| c * v), dt * k, start, density, bounds)
    };
    assert_mapped("OU", fit, false);
  }
  let x = simulate_cir(2.0, 0.04, 0.2, 0.04, dt, 1000, 301);
  for density in [DensityApprox::Exact, DensityApprox::Euler] {
    let fit = |c: f64, s: f64, k: f64| {
      let bounds = vec![
        (1e-4 / k, 50.0 / k),
        (1e-6 * c, 10.0 * c),
        (1e-6 * s, 10.0 * s),
      ];
      let mut model = cir(1.0 / k, 0.05 * c, 0.4 * s);
      fit_mle(
        &mut model,
        x.mapv(|v| c * v).view(),
        dt * k,
        density,
        Some(bounds),
      )
      .unwrap()
    };
    assert_mapped("CIR", fit, true);
  }
}

/// A level that starts at zero has no size to be its unit: half the box's width serves, or the
/// path's spread where the box has none, and the fit is the closed form's as from any other start.
#[test]
fn a_parameter_that_starts_at_zero_is_fitted_in_its_boxs_unit() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(8.0, 0.03, 0.02, 0.03, dt, 1000, 2);
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let open = (f64::NEG_INFINITY, f64::INFINITY);
  for level in [(-10.0, 10.0), open] {
    let bounds = vec![(1e-4, 50.0), level, (1e-6, 10.0)];
    let fit = fit_ou(&path, dt, [1.0, 0.0, 0.05], DensityApprox::Exact, bounds);
    let short = closed.log_likelihood - fit.log_likelihood;
    assert!(
      fit.converged && short.abs() < 1e-6,
      "{level:?}: {short} below at {}",
      fit.params
    );
  }
}

/// A level of `c x` from zero in a box open at either end is fitted in the path's unit: of 240
/// fits all converge within `1e-3` of the closed form on macOS, seven of them by a second search.
#[test]
fn a_zero_start_in_an_open_box_is_fitted_in_the_paths_unit() {
  let dt = 1.0 / 252.0;
  let inf = f64::INFINITY;
  let mut converged = 0;
  for c in [1e-9, 1e6, 1e9, 1e12] {
    for seed in 101..=110 {
      let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, seed).mapv(|v| c * v);
      let closed = fit_ou_mle(path.view(), dt).unwrap();
      for level in [(-inf, inf), (-inf, 100.0 * c)] {
        let tiny = c.min(1.0);
        let bounds = vec![(1e-4, 50.0), level, (1e-9 * tiny, 100.0 * c)];
        for start in [
          [2.0, 0.0, 0.3 * c],
          [1.0, 0.0, 1e-3 * c],
          [1.0, 0.0, 1e-6 * tiny],
        ] {
          let fit = fit_ou(&path, dt, start, DensityApprox::Exact, bounds.clone());
          let short = closed.log_likelihood - fit.log_likelihood;
          let said = fit.message == "not stationary" || fit.message.starts_with("stopped at");
          assert!(
            if fit.converged {
              short.abs() < 1e-3
            } else {
              said
            },
            "×{c:e} seed {seed} from {start:?} in {level:?}: {}, {short} below at {}",
            fit.message,
            fit.params
          );
          converged += usize::from(fit.converged);
        }
      }
    }
  }
  // Where a stall lands depends on the platform's `exp` and `ln`: the count is pinned as a range.
  assert!((236..=240).contains(&converged), "{converged} of 240");
}

/// An estimate held by its box is the bound itself, to the bit, in whatever unit the search ran;
/// a parameter whose box is one point stays there, at zero too.
#[test]
fn an_estimate_on_a_bound_is_the_bound_and_a_fixed_parameter_stays() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(8.0, 1.0, 0.02, 1.0, dt, 1000, 13);
  let exact = DensityApprox::Exact;
  let held = vec![(1e-4, 3.3), (-10.0, 10.0), (1e-6, 10.0)];
  for start in [[1.0, 0.5, 0.5], [0.7, 0.9, 0.03]] {
    let fit = fit_ou(&path, dt, start, exact, held.clone());
    assert!(fit.converged, "{start:?}");
    assert_eq!(fit.params[0], 3.3, "{start:?}");
  }
  for level in [1.0, 0.0] {
    let fixed = vec![(1e-4, 50.0), (level, level), (1e-6, 10.0)];
    let fit = fit_ou(&path, dt, [1.0, 0.5, 0.5], exact, fixed);
    assert_eq!(fit.params[1], level);
    assert!(
      fit.params[2] > 1e-6 && fit.log_likelihood.is_finite(),
      "{}",
      fit.params
    );
  }
}

/// A line search from (1, 0.5, 0.5) probes a corner of the box where Ozaki has no density on any
/// transition: the fit still reaches the optimum that the fit from the law reaches.
#[test]
fn an_ozaki_fit_from_a_far_start_passes_a_corner_without_a_likelihood() {
  let dt = 1.0 / 12.0;
  let path = simulate_ou(0.3, 1.0, 0.02, 1.0, dt, 1000, 96);
  let bounds = vec![(1e-4, 50.0), (-10.0, 10.0), (1e-6, 10.0)];
  let near = fit_ou(
    &path,
    dt,
    [0.3, 1.0, 0.02],
    DensityApprox::Ozaki,
    bounds.clone(),
  );
  let far = fit_ou(&path, dt, [1.0, 0.5, 0.5], DensityApprox::Ozaki, bounds);
  assert!(near.converged && far.converged);
  let short = near.log_likelihood - far.log_likelihood;
  assert!(short.abs() < 1.0, "{short} below {}", near.log_likelihood);
}
