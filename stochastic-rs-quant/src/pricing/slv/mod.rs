//! Heston Stochastic Local Volatility (SLV) model.
//!
//! $$
//! \frac{dS_t}{S_t} = (r-q)\,dt + L(S_t,t)\,\sqrt{V_t}\,dW_t^S,\qquad
//! dV_t = \kappa(\theta-V_t)\,dt + \eta\sigma\sqrt{V_t}\,dW_t^V
//! $$
//!
//! The leverage function $L(S,t)$ is calibrated so that the model
//! reproduces the Dupire local-volatility surface:
//!
//! $$
//! L^2(K,t) = \frac{\sigma_{\text{LV}}^2(K,t)}{\mathbb{E}[V_t \mid S_t = K]}
//! $$
//!
//! Calibration uses the Guyon–Labordère particle method with
//! Nadaraya–Watson kernel regression for the conditional expectation.
//!
//! Reference: Guyon & Henry-Labordère, "Being particular about
//! calibration", *Risk*, 2012.
//! See also: arXiv 2208.09986 (Djete, McKean–Vlasov existence),
//! arXiv 2406.14074 (Mustapha, strong well-posedness),
//! arXiv 1701.06001 (Cozma et al., control-variate particle method).

pub mod calibration;
pub mod pricer;

pub use calibration::calibrate_from_dupire;
pub use calibration::calibrate_leverage;
use ndarray::Array1;
use ndarray::Array2;
pub use pricer::HestonSlvPricer;

/// Heston model parameters augmented with the SLV mixing factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonSlvParams {
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Vol-of-vol of the base Heston model.
  pub sigma: f64,
  /// Spot–variance correlation.
  pub rho: f64,
  /// Initial variance.
  pub v0: f64,
  /// Mixing factor in $\left[0,1\right]$. $\eta=0$: pure local vol,
  /// $\eta=1$: full stochastic vol.
  pub eta: f64,
}

impl HestonSlvParams {
  /// Effective vol-of-vol after mixing: $\sigma_{\text{mix}} = \eta\,\sigma$.
  pub fn sigma_mixed(&self) -> f64 {
    self.eta * self.sigma
  }
}

/// A 2-D grid storing the calibrated leverage function $L(S,t)$ with
/// bilinear interpolation.
///
/// The grid is indexed by **absolute spot**, not by moneyness, so the span
/// it holds calibrated values for is exactly `spots[0] ..= spots[last]` —
/// [`spot_range`](Self::spot_range) — out to the last calibrated maturity
/// [`horizon`](Self::horizon). [`interpolate`](Self::interpolate) answers
/// every query, in-grid or not, by holding the nearest edge value flat.
/// [`covers`](Self::covers) is the separate statement of which of those
/// answers carry calibrated information.
#[derive(Debug, Clone)]
pub struct LeverageSurface {
  spots: Array1<f64>,
  times: Array1<f64>,
  values: Array2<f64>,
}

impl LeverageSurface {
  /// Build from pre-computed grid values. `values` has shape
  /// `(times.len(), spots.len())`.
  pub fn new(spots: Array1<f64>, times: Array1<f64>, values: Array2<f64>) -> Self {
    Self {
      spots,
      times,
      values,
    }
  }

  /// Bilinear interpolation of $L(S,t)$, clamped at the grid boundary.
  ///
  /// "Clamped" is a **flat hold of the nearest edge**, not a linear
  /// extrapolation and not a `NaN`: on a grid spanning `80 ..= 120` that
  /// carries $L(120,t)=1.9$, this returns `1.9` at `S = 120.001`, at
  /// `S = 500` and at `S = 1e9` alike, and does the same past `times[last]`
  /// on the time axis. Two surfaces that agree on their edge columns are
  /// therefore *bit-identical* everywhere past the edge, however far apart
  /// their interiors are.
  ///
  /// That is the right behaviour here — [`calibrate_leverage`]'s particle
  /// cloud legitimately wanders off its own eval grid and still has to be
  /// given a leverage to move under, and the same hold is what carries the
  /// first calibrated row back to `t = 0`. It is also why a *query* must not
  /// rely on it: see [`covers`](Self::covers).
  ///
  /// A `NaN` coordinate is held to an edge value too rather than propagating,
  /// which is the other reason [`covers`](Self::covers) tests the query
  /// before this runs.
  pub fn interpolate(&self, s: f64, t: f64) -> f64 {
    let si = fractional_index(&self.spots, s);
    let ti = fractional_index(&self.times, t);

    let i0 = (si.floor() as usize).min(self.spots.len() - 2);
    let j0 = (ti.floor() as usize).min(self.times.len() - 2);
    let i1 = i0 + 1;
    let j1 = j0 + 1;

    let ws = si - i0 as f64;
    let wt = ti - j0 as f64;
    let ws = ws.clamp(0.0, 1.0);
    let wt = wt.clamp(0.0, 1.0);

    let v00 = self.values[[j0, i0]];
    let v10 = self.values[[j0, i1]];
    let v01 = self.values[[j1, i0]];
    let v11 = self.values[[j1, i1]];

    (1.0 - wt) * ((1.0 - ws) * v00 + ws * v10) + wt * ((1.0 - ws) * v01 + ws * v11)
  }

  /// Spot grid.
  pub fn spots(&self) -> &Array1<f64> {
    &self.spots
  }

  /// Time grid.
  pub fn times(&self) -> &Array1<f64> {
    &self.times
  }

  /// Raw grid values (shape: times × spots).
  pub fn values(&self) -> &Array2<f64> {
    &self.values
  }

  /// Inclusive spot span the surface holds calibrated values for:
  /// `(spots[0], spots[last])`.
  pub fn spot_range(&self) -> (f64, f64) {
    (self.spots[0], self.spots[self.spots.len() - 1])
  }

  /// Last maturity the surface was calibrated out to: `times[last]`.
  pub fn horizon(&self) -> f64 {
    self.times[self.times.len() - 1]
  }

  /// Whether a `(spot, maturity)` query lands inside the box this surface
  /// holds calibrated values for: [`spot_range`](Self::spot_range) inclusive,
  /// and a maturity in $\left[0, \text{horizon}\right]$, also inclusive.
  ///
  /// The bound is the **grid's own extent**, not a tolerance around the `s0`
  /// [`calibrate_leverage`] was run at. `s0` is not recorded on the surface
  /// and does not need to be: what makes a query answerable is whether the
  /// surface has a value at it, and a strike ladder run around the
  /// calibration spot — the whole point of the type — never leaves the grid.
  ///
  /// The lower maturity bound is `0` rather than `times[0]` on purpose. The
  /// first row *is* the $t \to 0$ slice — [`calibrate_leverage`] evaluates it
  /// against the initial particle cloud — so holding it back to `t = 0` is
  /// the intended reading of the grid. Holding the last row *forward* past
  /// the horizon is not: nothing was calibrated there.
  ///
  /// A `NaN` in either coordinate compares false against both bounds and so
  /// reports `false`, which is what keeps
  /// [`interpolate`](Self::interpolate) from laundering it into an edge value.
  pub fn covers(&self, s: f64, tau: f64) -> bool {
    let (lo, hi) = self.spot_range();
    s >= lo && s <= hi && tau >= 0.0 && tau <= self.horizon()
  }
}

fn fractional_index(grid: &Array1<f64>, x: f64) -> f64 {
  if x <= grid[0] {
    return 0.0;
  }
  let n = grid.len();
  if x >= grid[n - 1] {
    return (n - 1) as f64;
  }
  for i in 0..n - 1 {
    if x >= grid[i] && x < grid[i + 1] {
      return i as f64 + (x - grid[i]) / (grid[i + 1] - grid[i]);
    }
  }
  (n - 1) as f64
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Spot grid `80 ..= 120`, time grid `0.25 ..= 1.0`, with
  /// `L(S_i, t_j) = 1.0 + 0.1 i + 0.5 j` so every edge is distinct.
  fn ramp() -> LeverageSurface {
    let mut values = Array2::<f64>::zeros((3, 5));
    for j in 0..3 {
      for i in 0..5 {
        values[[j, i]] = 1.0 + 0.1 * i as f64 + 0.5 * j as f64;
      }
    }
    LeverageSurface::new(
      Array1::from_vec(vec![80.0, 90.0, 100.0, 110.0, 120.0]),
      Array1::from_vec(vec![0.25, 0.5, 1.0]),
      values,
    )
  }

  #[test]
  fn interpolate_holds_the_edge_value_flat_past_the_grid() {
    let s = ramp();
    let low = s.interpolate(80.0, 0.5);
    let high = s.interpolate(120.0, 0.5);
    for far in [79.999, 40.0, 1.0, 1e-9] {
      assert_eq!(
        s.interpolate(far, 0.5),
        low,
        "below the grid the surface must hold L(80) = {low}, at S = {far}"
      );
    }
    for far in [120.001, 500.0, 1e9] {
      assert_eq!(
        s.interpolate(far, 0.5),
        high,
        "above the grid the surface must hold L(120) = {high}, at S = {far}"
      );
    }
    let last_row = s.interpolate(100.0, 1.0);
    for far in [1.001, 30.0, 1e9] {
      assert_eq!(
        s.interpolate(100.0, far),
        last_row,
        "past the horizon the surface must hold the t = 1 row = {last_row}, at t = {far}"
      );
    }
  }

  #[test]
  fn interpolate_holds_a_nan_coordinate_to_an_edge_instead_of_propagating_it() {
    let s = ramp();
    assert!(
      s.interpolate(f64::NAN, 0.5).is_finite(),
      "a NaN spot does not propagate through interpolate — this is why covers() has to test first"
    );
    assert!(s.interpolate(100.0, f64::NAN).is_finite());
  }

  /// The measurement the pricer's `NaN` gate rests on: past the edge the
  /// answer is a function of the boundary alone, so a surface's calibrated
  /// interior contributes exactly nothing out there.
  #[test]
  fn surfaces_that_agree_on_the_edge_are_bit_identical_past_it() {
    let spots = Array1::from_vec(vec![70.0, 100.0, 130.0]);
    let times = Array1::from_vec(vec![0.25, 0.5]);
    let flat = LeverageSurface::new(spots.clone(), times.clone(), Array2::from_elem((2, 3), 1.0));
    let mut bumped_values = Array2::from_elem((2, 3), 1.0);
    bumped_values[[0, 1]] = 3.0;
    bumped_values[[1, 1]] = 3.0;
    let bumped = LeverageSurface::new(spots, times, bumped_values);

    for far in [131.0, 200.0, 1000.0, 60.0, 5.0] {
      assert_eq!(
        flat.interpolate(far, 0.5).to_bits(),
        bumped.interpolate(far, 0.5).to_bits(),
        "a 3x difference in the interior leaves no trace at S = {far}"
      );
    }
    assert_ne!(
      flat.interpolate(100.0, 0.5).to_bits(),
      bumped.interpolate(100.0, 0.5).to_bits(),
      "inside the grid the interior is what answers"
    );
  }

  #[test]
  fn covers_is_the_grid_extent_and_not_a_tolerance() {
    let s = ramp();
    assert_eq!(s.spot_range(), (80.0, 120.0));
    assert_eq!(s.horizon(), 1.0);

    assert!(s.covers(80.0, 1.0), "both edges inclusive");
    assert!(
      s.covers(120.0, 0.0),
      "t = 0 is inside — the first row is the t -> 0 slice"
    );
    assert!(
      s.covers(100.0, 0.01),
      "a maturity below times[0] is still covered"
    );

    assert!(!s.covers(79.999, 0.5), "just below the spot grid");
    assert!(!s.covers(120.001, 0.5), "just above the spot grid");
    assert!(!s.covers(100.0, 1.001), "just past the horizon");
    assert!(
      !s.covers(100.0, -1e-9),
      "a negative maturity is not a grid edge"
    );
    assert!(!s.covers(f64::NAN, 0.5), "NaN spot");
    assert!(!s.covers(100.0, f64::NAN), "NaN maturity");
  }
}
