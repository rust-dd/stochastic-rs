use super::*;

fn search(objective: impl Fn(&[f64]) -> f64, coordinates: &[Coordinate]) -> Found {
  let start = vec![vec![0.0; coordinates.len()]];
  minimise(objective, by_differences, &[&start], coordinates, 1000)
}

fn pair() -> [Coordinate; 2] {
  [Coordinate::location(0.0, 1.0), Coordinate::log(0.0)]
}

#[test]
fn a_minimum_inside_the_bounds_converges() {
  let bowl = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 50.0 * (p[1] - 0.2).powi(2);
  let found = search(bowl, &pair());
  assert!(
    found.stop.converged(),
    "{}",
    found.stop.message(&["a", "b"])
  );
  // The check's `1e-7` in the cost is `√(1e-7/50)` in a coordinate.
  assert!((found.at[0] - 0.3).abs() < 5e-5 && (found.at[1] - 0.2).abs() < 5e-5);
}

fn shaped() -> [Coordinate; 2] {
  [Coordinate::location(0.0, 1.0), Coordinate::shape(0.0)]
}

/// A valley along `p₀ = p₁` that falls `1e-3` in all to the shape's limit: a step along the
/// shape alone climbs the valley's wall, and the fit pinned at the limit is as low.
#[test]
fn a_limit_that_fits_as_well_is_not_identified() {
  let valley = |p: &[f64]| 10.0 + 50.0 * (p[0] - p[1]).powi(2) + 1e-3 * (-p[1]).exp();
  let found = search(valley, &shaped());
  assert_eq!(found.stop.message(&["mu", "eta"]), "eta is not identified");
}

/// A shelf half a unit above the minimum is no fit as good: wide, and identified.
#[test]
fn a_limit_that_fits_worse_is_identified() {
  let shelf = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 0.5 * (1.0 - (-p[1] * p[1]).exp());
  let found = search(shelf, &shaped());
  assert!(
    found.stop.converged(),
    "{}",
    found.stop.message(&["mu", "eta"])
  );
}

#[test]
fn a_cost_falling_to_the_crate_bound_stops_at_it() {
  let slide = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) - p[1];
  let found = search(slide, &pair());
  assert_eq!(
    found.stop.message(&["mu", "sigma"]),
    "stopped at the bound of sigma"
  );
}

/// Rosenbrock's valley in thirty coordinates outlasts the simplex's cap; wherever the finisher
/// then takes the point, a capped search is no convergence.
#[test]
fn a_capped_search_is_not_converged() {
  let valley = |p: &[f64]| {
    let link = |w: &[f64]| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2);
    p.windows(2).map(link).sum::<f64>()
  };
  let coordinates = [(); 30].map(|()| Coordinate::free());
  let start = vec![vec![-1.2; 30]];
  let found = minimise(valley, by_differences, &[&start], &coordinates, 1000);
  assert_eq!(found.stop.message(&["p"; 30]), "iteration cap reached");
}

fn rosenbrock(p: &[f64]) -> f64 {
  let link = |w: &[f64]| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2);
  p.windows(2).map(link).sum::<f64>()
}

/// A capped search no finisher completes, in thirty coordinates.
fn unfinished(objective: impl Fn(&[f64]) -> f64) -> Found {
  let coordinates = [(); 30].map(|()| Coordinate::free());
  let as_it_is = |_: &dyn Fn(&[f64]) -> f64, p: Vec<f64>| (p, 0);
  minimise(
    objective,
    as_it_is,
    &[&[vec![-1.2; 30]]],
    &coordinates,
    1000,
  )
}

/// The cap is the search's stop, named before what the check finds beside its point.
#[test]
fn a_capped_search_the_check_fails_reads_the_cap() {
  let found = unfinished(rosenbrock);
  assert!(rosenbrock(&found.at) > 1.0, "{}", rosenbrock(&found.at));
  assert_eq!(found.stop.message(&["p"; 30]), "iteration cap reached");
}

/// The last round's gain is not left behind: the answer is the lowest point evaluated.
#[test]
fn the_lowest_point_seen_is_returned() {
  let lowest = std::cell::Cell::new(f64::INFINITY);
  let found = unfinished(|p| {
    let cost = rosenbrock(p);
    lowest.set(lowest.get().min(cost));
    cost
  });
  assert_eq!(rosenbrock(&found.at), lowest.get());
}

/// Along the shape the cost climbs `0.05` three units out and ends `0.01` under the estimate's
/// at the limit: the limit law fits better, whatever lies between.
#[test]
fn a_limit_that_fits_as_well_across_a_rise_reads_its_name() {
  let hill = |p: &[f64]| {
    let v = (p[1] / 3.0).powi(2);
    10.0 + 50.0 * (p[0] - 0.3).powi(2) + 0.05 * v * (1.0 - v).exp() - 0.01 * (1.0 - (-v).exp())
  };
  let found = search(hill, &shaped());
  assert_eq!(found.stop.message(&["mu", "eta"]), "eta is not identified");
}

/// A shelf of height `c` above the minimum, the cost at the limit: a thousandth fits as well,
/// two hundredths do not.
#[test]
fn a_limit_reads_its_name_under_five_thousandths_and_not_above() {
  let shelf =
    |c: f64| move |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + c * (1.0 - (-p[1] * p[1]).exp());
  let words = [1e-3, 2e-2].map(|c| search(shelf(c), &shaped()).stop.message(&["mu", "eta"]));
  assert_eq!(words, ["eta is not identified", "converged"]);
}

/// Past five units the cost has no value: no law to fit as well.
#[test]
fn a_limit_without_a_value_is_no_fit() {
  let walled = |p: &[f64]| {
    let flat = 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 1e-4 * (1.0 - (-p[1] * p[1]).exp());
    if p[1].abs() > 5.0 { NO_VALUE } else { flat }
  };
  assert!(search(walled, &shaped()).stop.converged());
}

/// Thirty capped coordinates and a flat shape beside them: the name comes before the cap.
#[test]
fn a_named_stop_comes_before_the_cap() {
  let mut coordinates = (0..30).map(|_| Coordinate::free()).collect::<Vec<_>>();
  coordinates.push(Coordinate::shape(0.0));
  let as_it_is = |_: &dyn Fn(&[f64]) -> f64, p: Vec<f64>| (p, 0);
  let mut start = vec![-1.2; 30];
  start.push(0.0);
  let found = minimise(
    |p| rosenbrock(&p[..30]),
    as_it_is,
    &[&[start]],
    &coordinates,
    1000,
  );
  let mut names = vec!["p"; 30];
  names.push("eta");
  assert_eq!(found.stop.message(&names), "eta is not identified");
}

/// The model's standard error of the shape is `0.1` in the bowl and `22` on the shelf: the
/// limit is 180 errors from the first, and under one from the second.
#[test]
fn the_screen_skips_a_limit_far_in_standard_errors_and_reads_a_near_one() {
  let bowl = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 50.0 * p[1] * p[1];
  let shelf = |p: &[f64]| 10.0 + 50.0 * (p[0] - 0.3).powi(2) + 1e-3 * (1.0 - (-p[1] * p[1]).exp());
  let (coordinates, at) = (shaped(), [0.3, 0.0]);
  assert_eq!(screened(&bowl, &at, &coordinates), [false, false]);
  assert_eq!(screened(&shelf, &at, &coordinates), [false, true]);
}

/// A stage that ran into its cap hands on no converged point, and says so.
#[test]
fn a_capped_stage_is_reported() {
  let coordinates = [(); 30].map(|()| Coordinate::free());
  let as_it_is = |_: &dyn Fn(&[f64]) -> f64, p: Vec<f64>| (p, 0);
  let (_, _, capped) = descend(rosenbrock, as_it_is, &[vec![-1.2; 30]], &coordinates);
  assert!(capped);
  let (_, _, capped) = descend(|p| p[0] * p[0], as_it_is, &[vec![1.0]], &coordinates[..1]);
  assert!(!capped);
  assert_eq!(
    Stop::Converged.after(true).message(&[]),
    "iteration cap reached"
  );
  assert_eq!(
    Stop::NotStationary.after(true).message(&[]),
    "not stationary"
  );
}

/// Toward the limit the pin crosses a basin `0.95` deep at six units; the limit itself is `0.44`
/// under the first estimate. The search starts again in the basin and ends there.
#[test]
fn a_search_restarts_from_a_fit_on_the_path_below_the_limit() {
  let basin = |p: &[f64]| {
    let near = 0.05 * (1.0 - (-p[1] * p[1]).exp());
    let deep = (-(p[1] - 6.0).powi(2) / 8.0).exp();
    let far = 0.5 / (1.0 + (14.0 - p[1]).exp());
    10.0 + 50.0 * (p[0] - 0.3).powi(2) + near - deep - far
  };
  let found = search(basin, &shaped());
  assert!((found.at[1] - 6.0).abs() < 0.5, "{:?}", found.at);
  assert!(
    found.stop.converged(),
    "{}",
    found.stop.message(&["mu", "eta"])
  );
}

/// Wells at `-17, -15, -11, -3, 13`, each deeper and each on the pin's way from the one before
/// alone: four searches find a lower well, and the fourth has no restart left.
#[test]
fn a_search_out_of_restarts_returns_the_lower_fit_and_is_not_stationary() {
  let wells = [-17.0, -15.0, -11.0, -3.0, 13.0];
  let stairs = |p: &[f64]| {
    let well =
      |(k, at): (usize, &f64)| (0.5 + 0.05 * k as f64) * (-2.0 * (p[1] - at).powi(2)).exp();
    10.0 + 50.0 * (p[0] - 0.3).powi(2) - wells.iter().enumerate().map(well).sum::<f64>()
  };
  let start = [vec![0.3, -17.0]];
  let found = minimise(stairs, by_differences, &[&start], &shaped(), 1000);
  assert_eq!(
    (
      found.stop.message(&["mu", "eta"]).as_str(),
      found.at[1].round()
    ),
    ("not stationary", 13.0)
  );
}

/// Two wells, the shallow one under the better start: the descent from the other group ends
/// lower and is kept; between equal wells the best start's own stays.
#[test]
fn a_descent_from_another_group_is_kept_only_where_it_ends_lower() {
  let wells = |deep: f64| {
    move |p: &[f64]| {
      10.0 - (-4.0 * (p[0] + 2.0).powi(2)).exp() - deep * (-4.0 * (p[0] - 2.0).powi(2)).exp()
    }
  };
  let coordinates = [Coordinate::free()];
  let (near, far) = ([vec![-1.5]], [vec![2.9]]);
  let groups = [&near[..], &far[..]];
  let lower = minimise(wells(2.0), by_differences, &groups, &coordinates, 1000);
  assert!(
    (lower.at[0] - 2.0).abs() < 1e-3 && lower.stop.converged(),
    "{:?}",
    lower.at
  );
  let equal = minimise(wells(1.0), by_differences, &groups, &coordinates, 1000);
  assert!(
    (equal.at[0] + 2.0).abs() < 1e-3 && equal.stop.converged(),
    "{:?}",
    equal.at
  );
}
