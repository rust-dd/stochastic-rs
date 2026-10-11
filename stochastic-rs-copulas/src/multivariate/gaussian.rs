//! # Gaussian
//!
//! $$
//! C_\Sigma(u)=\Phi_\Sigma\!\left(\Phi^{-1}(u_1),\dots,\Phi^{-1}(u_d)\right)
//! $$
//!

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use super::MultivariateFamily;
use super::correlation_matrix::CorrelationMatrix;
use super::correlation_matrix::shrink_to_spd;
use crate::error::CopulaError;
use crate::traits::MultivariateExt;

/// The Gaussian copula; `default()` is the unfitted copula that `fit` starts from.
#[derive(Debug, Clone, Default)]
pub struct GaussianMultivariate {
  corr: Option<CorrelationMatrix>,
}

impl GaussianMultivariate {
  /// The copula of `N(0, corr)`; panics unless `corr` is a correlation matrix.
  pub fn new(corr: Array2<f64>) -> Self {
    Self::try_new(corr).unwrap_or_else(|e| panic!("{e}"))
  }

  /// `new` with the structure error returned instead of a panic.
  pub fn try_new(corr: Array2<f64>) -> Result<Self, CopulaError> {
    Ok(Self {
      corr: Some(CorrelationMatrix::try_new(corr)?),
    })
  }

  /// Returns a reference to the internal correlation matrix, if fitted.
  pub fn correlation(&self) -> Option<&Array2<f64>> {
    self.corr.as_ref().map(CorrelationMatrix::corr)
  }

  fn fitted(&self) -> Result<&CorrelationMatrix, CopulaError> {
    self.corr.as_ref().ok_or(CopulaError::NotFitted)
  }

  /// Transform U in (0,1)^{n x d} to Z in R^{n x d} via standard normal inverse CDF.
  fn transform_to_normal(&self, u: ArrayView2<f64>) -> Array2<f64> {
    let eps = 1e-12;
    let mut z = u.to_owned();
    for mut row in z.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        let clamped = val.clamp(eps, 1.0 - eps);
        *val = ndtri(clamped);
      }
    }
    z
  }

  /// Sample correlation of the columns of Z (n x d), off-diagonals clamped to ±0.999999.
  fn estimate_corr_from_normal(z: ArrayView2<f64>) -> Array2<f64> {
    let n = z.nrows() as f64;
    let d = z.ncols();
    // Center columns
    let means = z.mean_axis(Axis(0)).unwrap();
    let mut zc = z.to_owned();
    let nrows = zc.nrows();
    for j in 0..d {
      let m = means[j];
      let mut col = zc.column_mut(j);
      for i in 0..nrows {
        col[i] -= m;
      }
    }
    // Standard deviations
    let mut stds = Array1::<f64>::zeros(d);
    for j in 0..d {
      let col = zc.column(j);
      let var = col.iter().map(|&x| x * x).sum::<f64>() / (n - 1.0);
      stds[j] = var.max(1e-18).sqrt();
    }
    // Correlation matrix
    let mut corr = Array2::<f64>::zeros((d, d));
    for k in 0..d {
      corr[[k, k]] = 1.0;
    }
    for i in 0..d {
      for j in i..d {
        if i == j {
          continue;
        }
        let dot = zc
          .column(i)
          .iter()
          .zip(zc.column(j).iter())
          .map(|(&a, &b)| a * b)
          .sum::<f64>();
        let cov = dot / (n - 1.0);
        let r = (cov / (stds[i] * stds[j])).clamp(-0.999_999, 0.999_999);
        corr[[i, j]] = r;
        corr[[j, i]] = r;
      }
    }
    corr
  }
}

impl MultivariateExt for GaussianMultivariate {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::Gaussian
  }

  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<Array2<f64>, CopulaError> {
    let corr = self.fitted()?;
    let d = corr.dim();
    let l = corr.chol_lower(); // (d x d)
    // Sample standard normals G ~ N(0, I) of shape (n x d)
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(seed);
    let g = Array2::from_shape_fn((n, d), |_| normal.sample());
    // z = g * L^T
    let z = g.dot(&l.t());
    // Transform to uniforms using standard normal CDF
    let mut u = z.clone();
    for mut row in u.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        *val = norm_cdf(*val);
      }
    }
    Ok(u)
  }

  /// Fit the Gaussian copula from U in (0,1)^{n x d}.
  fn fit(&mut self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    if X.ncols() < 2 {
      return Err(CopulaError::InvalidStructure(format!(
        "a Gaussian copula needs dim >= 2, got {}",
        X.ncols()
      )));
    }
    if X.nrows() < 2 {
      return Err(CopulaError::InsufficientData {
        needed: 2,
        got: X.nrows(),
      });
    }
    // Basic range check
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err(CopulaError::MarginalOutOfRange);
    }
    let z = self.transform_to_normal(X);
    let corr = shrink_to_spd(Self::estimate_corr_from_normal(z.view()));
    self.corr = Some(CorrelationMatrix::try_new(corr)?);
    Ok(())
  }

  fn check_fit(&self, X: ArrayView2<f64>) -> Result<(), CopulaError> {
    let dim = self.fitted()?.dim();
    if X.ncols() != dim {
      return Err(CopulaError::DimensionMismatch {
        expected: dim,
        got: X.ncols(),
      });
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err(CopulaError::MarginalOutOfRange);
    }
    Ok(())
  }

  fn pdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    let z = self.transform_to_normal(X); // n x d
    let corr = self.fitted()?;
    let inv = corr.inverse();
    let log_det = corr.log_det();

    // Compute for each row: -1/2 z^T (inv - I) z - 1/2 log det
    let mut out = Array1::<f64>::zeros(z.nrows());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      // q = (inv - I) * z
      let mut q = inv.dot(&row.to_owned());
      for k in 0..q.len() {
        q[k] -= row[k];
      }
      let quad = row.dot(&q);
      let log_c = -0.5 * (log_det + quad);
      out[i] = log_c.exp();
    }
    Ok(out)
  }

  fn cdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    let z = self.transform_to_normal(X); // n x d
    let corr = self.fitted()?;
    let l = corr.chol_lower(); // d x d
    let d = corr.dim();
    let n = z.nrows();
    let m_samples = 4000usize; // Monte Carlo samples per query point
    let mut out = Array1::<f64>::zeros(n);

    // Pre-sample standard normals for efficiency: (m x d)
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Unseeded);
    let g = Array2::from_shape_fn((m_samples, d), |_| normal.sample());
    let y = g.dot(&l.t()); // (m x d) ~ MVN(0, corr)

    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut count = 0usize;
      'outer: for r in 0..m_samples {
        for c in 0..d {
          if y[[r, c]] > row[c] {
            continue 'outer;
          }
        }
        count += 1;
      }
      out[i] = count as f64 / m_samples as f64;
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::bivariate::gaussian::GaussianCopula;
  use crate::traits::BivariateExt;

  /// At `d = 2` the density is the bivariate Gaussian copula's closed form.
  #[test]
  fn the_two_dimensional_density_is_the_bivariate_gaussian_copulas() {
    let rho = 0.6;
    let multivariate = GaussianMultivariate::new(array![[1.0, rho], [rho, 1.0]]);
    let q = array![[0.3, 0.7], [0.5, 0.5], [0.1, 0.9], [0.8, 0.25]];
    let got = multivariate.pdf(q.view()).unwrap();
    let want = GaussianCopula::new(rho).pdf(q.view()).unwrap();
    for i in 0..q.nrows() {
      assert!(
        (got[i] - want[i]).abs() < 1e-12 * want[i],
        "{i}: {} vs {}",
        got[i],
        want[i]
      );
    }
  }

  #[test]
  fn a_fit_recovers_the_correlation_it_was_drawn_from_with_a_unit_diagonal() {
    let truth = array![[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]];
    let u = GaussianMultivariate::new(truth.clone())
      .sample(20_000, &Deterministic::new(11))
      .unwrap();
    let mut fitted = GaussianMultivariate::default();
    fitted.fit(u.view()).unwrap();
    let corr = fitted.correlation().unwrap();
    for i in 0..3 {
      assert_eq!(corr[[i, i]], 1.0);
      for j in 0..i {
        assert!(
          (corr[[i, j]] - truth[[i, j]]).abs() < 0.02,
          "[{i}, {j}]: {}",
          corr[[i, j]]
        );
      }
    }
  }
}
