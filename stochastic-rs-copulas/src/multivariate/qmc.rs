//! The point set every deterministic copula `cdf` integrates over.

use ndarray::Array2;
use stochastic_rs_distributions::qmc::SobolSeq;

/// Unscrambled, so a cdf depends on its inputs alone; the sequence starts past the origin, so no coordinate is 0.
pub(crate) const QMC_POINTS: usize = 4096;

pub(crate) fn sobol_points(dims: usize) -> Array2<f64> {
  SobolSeq::new(dims).sample::<f64>(QMC_POINTS)
}
