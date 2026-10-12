//! The point set every deterministic copula `cdf` integrates over.
//! Reference: Owen, A. B. (2022), "On Dropping the First Sobol' Point", MCQMC 2020, Springer PROMS 71–86, doi:10.1007/978-3-030-98319-2_4, arXiv:2008.08051 (§1, `u_i + 1/(2n)`)

use ndarray::Array2;
use ndarray::s;
use stochastic_rs_distributions::qmc::SobolSeq;

/// A power of two, so the net's coordinates are multiples of `1/QMC_POINTS` and the half-cell shift is exact.
pub(crate) const QMC_POINTS: usize = 4096;

/// The full unscrambled Sobol net, origin included, shifted by half a cell: a cdf depends on its inputs alone, no
/// coordinate is 0 or 1, and every column is the midpoint rule.
pub(crate) fn sobol_points(dims: usize) -> Array2<f64> {
  let mut points = Array2::<f64>::zeros((QMC_POINTS, dims));
  points
    .slice_mut(s![1.., ..])
    .assign(&SobolSeq::new(dims).sample::<f64>(QMC_POINTS - 1));
  points.mapv_inplace(|w| w + 0.5 / QMC_POINTS as f64);
  points
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn every_column_times_8192_is_a_permutation_of_the_odd_integers_below_it() {
    for dims in [1, 2, 3, 10] {
      for column in sobol_points(dims).columns() {
        let mut cells: Vec<f64> = column.iter().map(|&w| w * 8192.0).collect();
        cells.sort_by(f64::total_cmp);
        let odd = (0..QMC_POINTS).map(|k| (2 * k + 1) as f64);
        assert!(cells.into_iter().eq(odd), "dims {dims}");
      }
    }
  }
}
