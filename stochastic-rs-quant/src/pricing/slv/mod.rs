//! Heston stochastic-local volatility: Heston with a leverage `L(S, t)` on the spot's diffusion.
//!
//! $$
//! \frac{dS_t}{S_t} = (r-q)\,dt + L(S_t,t)\,\sqrt{V_t}\,dW_t^S,\qquad
//! dV_t = \kappa(\theta-V_t)\,dt + \eta\sigma\sqrt{V_t}\,dW_t^V
//! $$
//!
//! `L` is calibrated by Guyon–Henry-Labordère's particle method so the model reproduces Dupire's:
//!
//! $$
//! L^2(K,t) = \frac{\sigma_{\text{LV}}^2(K,t)}{\mathbb{E}[V_t \mid S_t = K]}
//! $$
//!
//! Reference: Guyon & Henry-Labordère (2012), Risk 25(1), 88–93
//! Reference: Djete (2022), arXiv:2208.09986
//! Reference: Mustapha (2024), arXiv:2406.14074
//! Reference: Cozma et al. (2017), arXiv:1701.06001

pub mod calibration;
pub mod fokker_planck;
pub mod pricer;

pub use calibration::ParticleCalibration;
pub use calibration::ParticleMethod;
pub use calibration::calibrate_leverage;
pub use fokker_planck::FokkerPlanckCalibration;
pub use fokker_planck::FokkerPlanckDensity;
pub use fokker_planck::FokkerPlanckMethod;
pub use fokker_planck::calibrate_leverage_fokker_planck;
pub use fokker_planck::heston_slv_density;
use ndarray::Array1;
use ndarray::Array2;
pub use pricer::HestonSlvPricer;
use stochastic_rs_distributions::traits::FloatExt;
use stochastic_rs_distributions::traits::Fn2D;
use stochastic_rs_distributions::traits::Grid2D;

use crate::pricing::heston::HestonParams;

/// Heston model parameters augmented with the SLV mixing factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonSlvParams {
  pub heston: HestonParams,
  /// Mixing factor in $\left[0,1\right]$. $\eta=0$: pure local vol,
  /// $\eta=1$: full stochastic vol.
  pub eta: f64,
}

impl HestonSlvParams {
  /// Effective vol-of-vol after mixing: $\sigma_{\text{mix}} = \eta\,\sigma$.
  pub fn sigma_mixed(&self) -> f64 {
    self.eta * self.heston.sigma
  }
}

/// Calibrated leverage `L(S, t)` on an absolute-spot × time [`Grid2D`], held flat past its edges
/// ([`covers`](Self::covers) bounds the calibrated box); as an [`Fn2D`] it reads `(t, S)`.
#[derive(Debug, Clone, PartialEq)]
pub struct LeverageSurface {
  grid: Grid2D<f64>,
}

impl LeverageSurface {
  /// Panics unless both axes are non-empty and strictly ascending and `values` has the shape
  /// `(times.len(), spots.len())`, [`Grid2D::new`]'s conditions.
  pub fn new(spots: Array1<f64>, times: Array1<f64>, values: Array2<f64>) -> Self {
    Self {
      grid: Grid2D::new(times, spots, values),
    }
  }

  /// Bilinear `L(s, t)`, holding the nearest edge flat past the grid and at a `NaN` coordinate, as
  /// the wandering particle cloud needs; a query checks [`covers`](Self::covers) first.
  pub fn interpolate(&self, s: f64, t: f64) -> f64 {
    self.grid.eval(t, s)
  }

  /// Spot grid.
  pub fn spots(&self) -> &Array1<f64> {
    self.grid.xs()
  }

  /// Time grid.
  pub fn times(&self) -> &Array1<f64> {
    self.grid.ts()
  }

  /// Raw grid values (shape: times × spots).
  pub fn values(&self) -> &Array2<f64> {
    self.grid.values()
  }

  /// The same table as the `(t, x)` grid every [`Fn2D`] speaks.
  pub fn grid(&self) -> &Grid2D<f64> {
    &self.grid
  }

  /// Inclusive spot span the surface holds calibrated values for:
  /// `(spots[0], spots[last])`.
  pub fn spot_range(&self) -> (f64, f64) {
    self.grid.x_range()
  }

  /// Last maturity the surface was calibrated out to: `times[last]`.
  pub fn horizon(&self) -> f64 {
    self.grid.t_range().1
  }

  /// Whether `(s, tau)` lies in [`spot_range`](Self::spot_range) by `[0, horizon]`, inclusive (the
  /// first row is the `t = 0` slice); `false` at a `NaN` coordinate.
  pub fn covers(&self, s: f64, tau: f64) -> bool {
    let (lo, hi) = self.spot_range();
    s >= lo && s <= hi && tau >= 0.0 && tau <= self.horizon()
  }
}

impl From<LeverageSurface> for Grid2D<f64> {
  fn from(surface: LeverageSurface) -> Self {
    surface.grid
  }
}

/// The surface as a process coefficient: `Fn2D::Grid`, evaluated as
/// `L(t, S)`. A grid keeps the process on the host, as a closure does.
impl<T: FloatExt> From<LeverageSurface> for Fn2D<T> {
  fn from(surface: LeverageSurface) -> Self {
    Fn2D::Grid(surface.grid.cast::<T>())
  }
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

  /// Past the edge the answer depends on the boundary alone, which the pricer's `NaN` gate rests
  /// on.
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

  /// Handed to a process the table answers in the `(t, S)` order every
  /// `Fn2D` uses, and the values are the surface's own.
  #[test]
  fn as_a_coefficient_the_surface_evaluates_in_time_state_order() {
    let s = ramp();
    let f = Fn2D::<f64>::from(s.clone());
    assert!(f.program().is_none(), "a grid has no device program");
    for (t, spot) in [(0.5, 95.0), (0.25, 80.0), (1.0, 120.0), (0.7, 200.0)] {
      assert_eq!(f.call(t, spot), s.interpolate(spot, t));
    }
    let single = Fn2D::<f32>::from(s.clone());
    assert!((single.call(0.5, 95.0) as f64 - s.interpolate(95.0, 0.5)).abs() < 1e-6);
  }

  #[test]
  fn covers_is_the_grid_extent_and_not_a_tolerance() {
    let s = ramp();
    assert_eq!(s.spot_range(), (80.0, 120.0));
    assert_eq!(s.horizon(), 1.0);

    assert!(s.covers(80.0, 1.0), "both edges inclusive");
    assert!(
      s.covers(120.0, 0.0),
      "t = 0 is inside — the first row is the t = 0 slice"
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
