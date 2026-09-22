//! A coefficient of time and state tabulated on a rectangular grid:
//! [`Grid2D`] holds one value per `(t, x)` node, evaluates by bilinear
//! interpolation inside the grid and holds the nearest edge flat outside it.
//! It is the third form an [`Fn2D`](super::Fn2D) takes beside a closure and
//! an [`Expr`](super::Expr) — a calibrated leverage surface, a local
//! volatility read off market data, any coefficient that exists as numbers
//! rather than as a formula.

use std::cmp::Ordering;

use ndarray::Array1;
use ndarray::Array2;

use super::float::FloatExt;

/// Values on a `(t, x)` grid, `values[[j, i]] = f(ts[j], xs[i])`, read back
/// by bilinear interpolation.
///
/// Outside the grid the nearest edge is held **flat** — not extrapolated
/// linearly and not `NaN`: a query at `x = 10⁹` on a grid ending at `120`
/// answers what the grid holds at `120`, and a `NaN` coordinate is held to
/// the first node rather than propagating. Both are what a simulation
/// wants from a coefficient whose paths legitimately wander past the range
/// it was tabulated on; a caller who needs to know whether a query carries
/// tabulated information tests [`t_range`](Self::t_range) and
/// [`x_range`](Self::x_range) itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid2D<T: FloatExt> {
  ts: Array1<T>,
  xs: Array1<T>,
  values: Array2<T>,
}

impl<T: FloatExt> Grid2D<T> {
  /// A grid over strictly ascending `ts` and `xs`, with `values` of shape
  /// `(ts.len(), xs.len())`. An axis may hold a single node, along which
  /// the value is then constant.
  ///
  /// # Panics
  ///
  /// When an axis is empty or not strictly ascending, or when the shape of
  /// `values` is not `(ts.len(), xs.len())`.
  pub fn new(ts: Array1<T>, xs: Array1<T>, values: Array2<T>) -> Self {
    assert!(!ts.is_empty(), "Grid2D: the time axis must hold a node");
    assert!(!xs.is_empty(), "Grid2D: the state axis must hold a node");
    assert!(
      ts.windows(2).into_iter().all(|w| w[0] < w[1]),
      "Grid2D: the time axis must be strictly ascending"
    );
    assert!(
      xs.windows(2).into_iter().all(|w| w[0] < w[1]),
      "Grid2D: the state axis must be strictly ascending"
    );
    assert_eq!(
      values.dim(),
      (ts.len(), xs.len()),
      "Grid2D: values must have shape (ts.len(), xs.len())"
    );
    Self { ts, xs, values }
  }

  /// Bilinear interpolation of the value at `(t, x)`, holding the nearest
  /// edge flat past the grid.
  pub fn eval(&self, t: T, x: T) -> T {
    let (j0, j1, wt) = bracket(&self.ts, t);
    let (i0, i1, wx) = bracket(&self.xs, x);
    let one = T::one();
    let v00 = self.values[[j0, i0]];
    let v10 = self.values[[j0, i1]];
    let v01 = self.values[[j1, i0]];
    let v11 = self.values[[j1, i1]];
    (one - wt) * ((one - wx) * v00 + wx * v10) + wt * ((one - wx) * v01 + wx * v11)
  }

  /// The time nodes.
  pub fn ts(&self) -> &Array1<T> {
    &self.ts
  }

  /// The state nodes.
  pub fn xs(&self) -> &Array1<T> {
    &self.xs
  }

  /// The tabulated values, shape `(ts.len(), xs.len())`.
  pub fn values(&self) -> &Array2<T> {
    &self.values
  }

  /// The inclusive time span the grid holds values for.
  pub fn t_range(&self) -> (T, T) {
    (self.ts[0], self.ts[self.ts.len() - 1])
  }

  /// The inclusive state span the grid holds values for.
  pub fn x_range(&self) -> (T, T) {
    (self.xs[0], self.xs[self.xs.len() - 1])
  }

  /// The same grid at another precision, converted through `f64`.
  pub fn cast<U: FloatExt>(&self) -> Grid2D<U> {
    let through = |v: &T| U::from_f64_fast(v.to_f64().unwrap());
    Grid2D {
      ts: self.ts.map(through),
      xs: self.xs.map(through),
      values: self.values.map(through),
    }
  }

  /// The axes and values back, `(ts, xs, values)`.
  pub fn into_parts(self) -> (Array1<T>, Array1<T>, Array2<T>) {
    (self.ts, self.xs, self.values)
  }
}

/// The two nodes bracketing `x` and the weight of the upper one: `(i, i, 0)`
/// at or past an edge — and for a `NaN`, which compares false against every
/// node and so is held to the first — else `(i, i + 1, w)` with
/// `w = (x − g[i]) / (g[i + 1] − g[i])`.
fn bracket<T: FloatExt>(grid: &Array1<T>, x: T) -> (usize, usize, T) {
  let last = grid.len() - 1;
  if !matches!(x.partial_cmp(&grid[0]), Some(Ordering::Greater)) {
    return (0, 0, T::zero());
  }
  if x >= grid[last] {
    return (last, last, T::zero());
  }
  let i1 = grid
    .as_slice()
    .map(|g| g.partition_point(|&node| node <= x))
    .unwrap_or_else(|| grid.iter().position(|&node| node > x).unwrap_or(last));
  let i0 = i1 - 1;
  (i0, i1, (x - grid[i0]) / (grid[i1] - grid[i0]))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `f(t, x) = 1 + 0.1 i + 0.5 j` on `xs = 80..=120` step 10, `ts = [0.25,
  /// 0.5, 1.0]`, so every edge is distinct.
  fn ramp() -> Grid2D<f64> {
    let mut values = Array2::<f64>::zeros((3, 5));
    for j in 0..3 {
      for i in 0..5 {
        values[[j, i]] = 1.0 + 0.1 * i as f64 + 0.5 * j as f64;
      }
    }
    Grid2D::new(
      Array1::from_vec(vec![0.25, 0.5, 1.0]),
      Array1::from_vec(vec![80.0, 90.0, 100.0, 110.0, 120.0]),
      values,
    )
  }

  #[test]
  fn nodes_evaluate_to_their_own_values() {
    let g = ramp();
    for (j, &t) in g.ts().iter().enumerate() {
      for (i, &x) in g.xs().iter().enumerate() {
        assert_eq!(g.eval(t, x), g.values()[[j, i]]);
      }
    }
  }

  #[test]
  fn interior_points_interpolate_bilinearly() {
    let g = ramp();
    let v = g.eval(0.375, 95.0);
    assert!((v - (1.0 + 0.15 + 0.25)).abs() < 1e-12, "got {v}");
    let v = g.eval(0.75, 112.0);
    assert!((v - (1.0 + 0.32 + 0.75)).abs() < 1e-12, "got {v}");
  }

  #[test]
  fn past_an_edge_the_edge_is_held_flat() {
    let g = ramp();
    let low = g.eval(0.5, 80.0);
    let high = g.eval(0.5, 120.0);
    for far in [79.999, 40.0, 1.0, -5.0] {
      assert_eq!(g.eval(0.5, far), low, "x = {far}");
    }
    for far in [120.001, 500.0, 1e9] {
      assert_eq!(g.eval(0.5, far), high, "x = {far}");
    }
    let first_row = g.eval(0.25, 100.0);
    let last_row = g.eval(1.0, 100.0);
    assert_eq!(g.eval(0.0, 100.0), first_row);
    assert_eq!(g.eval(-1.0, 100.0), first_row);
    assert_eq!(g.eval(30.0, 100.0), last_row);
  }

  #[test]
  fn a_nan_coordinate_is_held_to_the_first_node() {
    let g = ramp();
    assert_eq!(g.eval(0.5, f64::NAN), g.eval(0.5, 80.0));
    assert_eq!(g.eval(f64::NAN, 100.0), g.eval(0.25, 100.0));
  }

  #[test]
  fn a_single_node_axis_is_constant_along_it() {
    let g = Grid2D::new(
      Array1::from_vec(vec![1.0]),
      Array1::from_vec(vec![0.0, 1.0]),
      Array2::from_shape_vec((1, 2), vec![2.0, 4.0]).unwrap(),
    );
    for t in [-1.0, 0.0, 1.0, 7.0] {
      assert_eq!(g.eval(t, 0.5), 3.0, "t = {t}");
      assert_eq!(g.eval(t, 9.0), 4.0, "t = {t}");
    }
    assert_eq!(g.t_range(), (1.0, 1.0));
    assert_eq!(g.x_range(), (0.0, 1.0));
  }

  #[test]
  fn cast_round_trips_through_single_precision() {
    let g = ramp();
    let single = g.cast::<f32>();
    assert_eq!(single.values().dim(), (3, 5));
    let back = single.cast::<f64>();
    assert!((back.eval(0.375, 95.0) - g.eval(0.375, 95.0)).abs() < 1e-6);
  }

  #[test]
  #[should_panic(expected = "strictly ascending")]
  fn a_non_ascending_axis_is_rejected() {
    let _ = Grid2D::new(
      Array1::from_vec(vec![0.5, 0.25]),
      Array1::from_vec(vec![1.0]),
      Array2::zeros((2, 1)),
    );
  }

  #[test]
  #[should_panic(expected = "shape")]
  fn a_mismatched_value_shape_is_rejected() {
    let _ = Grid2D::new(
      Array1::from_vec(vec![0.25, 0.5]),
      Array1::from_vec(vec![1.0, 2.0]),
      Array2::zeros((2, 3)),
    );
  }
}
