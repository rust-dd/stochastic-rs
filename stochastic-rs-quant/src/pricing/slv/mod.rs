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
