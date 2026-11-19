//! Explicit inequality-constrained variance-state projection.

/// Policy for finite posterior variance means outside the positive state set.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum HestonCekfPositiveStatePolicy {
  /// Returns a typed error when the posterior variance reaches the floor.
  #[default]
  Strict,
  /// Projects the posterior mean onto `[floor, +∞)`, `argmin_v (v - raw)² / P` over `v >= floor`;
  /// the CEKF covariance bound stays, as an active inequality is not a zero-uncertainty equality.
  /// Reference: Gupta & Hauser, Eq. 64, <https://arxiv.org/abs/0709.2791>
  Project { floor: f64 },
}

/// One posterior variance-mean projection in variance units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCekfVarianceProjection {
  pub raw_variance: f64,
  pub projected_variance: f64,
}

impl HestonCekfVarianceProjection {
  /// Absolute projection displacement in variance units.
  pub fn absolute_correction(self) -> f64 {
    (self.projected_variance - self.raw_variance).abs()
  }
}

/// Projection aligned with its zero-based observation index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCekfIndexedVarianceProjection {
  pub observation_index: usize,
  pub raw_variance: f64,
  pub projected_variance: f64,
}

/// Aggregate audit trail for a fixed or online filter pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HestonCekfProjectionDiagnostics {
  pub total_steps: usize,
  pub projected_steps: usize,
  pub max_abs_projection_correction: f64,
  pub last_projection: Option<HestonCekfIndexedVarianceProjection>,
}

impl HestonCekfProjectionDiagnostics {
  /// Fraction of filter steps whose posterior state mean was projected.
  pub fn projected_fraction(self) -> f64 {
    if self.total_steps == 0 {
      0.0
    } else {
      self.projected_steps as f64 / self.total_steps as f64
    }
  }

  pub(crate) fn record_step(
    &mut self,
    observation_index: usize,
    projection: Option<HestonCekfVarianceProjection>,
  ) {
    self.total_steps += 1;
    let Some(projection) = projection else {
      return;
    };
    self.projected_steps += 1;
    self.max_abs_projection_correction = self
      .max_abs_projection_correction
      .max(projection.absolute_correction());
    self.last_projection = Some(HestonCekfIndexedVarianceProjection {
      observation_index,
      raw_variance: projection.raw_variance,
      projected_variance: projection.projected_variance,
    });
  }
}
