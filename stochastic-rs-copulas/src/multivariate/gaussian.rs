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
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;

use super::MultivariateFamily;
use super::correlation_matrix::CorrelationMatrix;
use super::correlation_matrix::shrink_to_spd;
use super::genz::mvn_upper;
use super::genz::open_coordinates;
use super::qmc::sobol_points;
use crate::bivariate::gaussian::gaussian_copula_cdf;
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
    let l = corr.chol_lower();
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

  /// Fit the Gaussian copula from U in (0,1)^{n x d}; an indefinite or near-singular estimate is shrunk toward the
  /// identity until every eigenvalue is at least `1e-3`.
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

  /// `Φ_R(Φ⁻¹(u))` on the closed cube: a zero coordinate gives 0 and a unit one drops out; two left are Owen's T closed
  /// form, more are Genz's separation-of-variables integral over the Sobol points.
  fn cdf(&self, X: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    self.check_fit(X)?;
    let corr = self.fitted()?;
    let points = (corr.dim() > 2).then(|| sobol_points(corr.dim() - 1));
    Ok(
      X.rows()
        .into_iter()
        .map(|u| match open_coordinates(u).as_deref() {
          None => 0.0,
          Some([]) => 1.0,
          Some(&[i]) => u[i],
          Some(&[i, j]) => gaussian_copula_cdf(u[i], u[j], corr.corr()[[i, j]]),
          Some(_) => {
            let b: Vec<f64> = u.iter().map(|&x| ndtri(x)).collect();
            let points = points.as_ref().expect("three open coordinates need d > 2");
            mvn_upper(corr.chol_lower(), &b, points.view())
          }
        })
        .collect(),
    )
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::bivariate::gaussian::GaussianCopula;
  use crate::multivariate::linalg::condition_number;
  use crate::multivariate::linalg::smallest_eigenvalue;
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

  /// As many rows as columns leave the sample correlation of rank `d - 1`, which the fit lifts to the eigenvalue floor.
  #[test]
  fn a_fit_with_as_many_rows_as_columns_has_its_rank_deficiency_floored() {
    let u = GaussianMultivariate::new(Array2::<f64>::eye(6))
      .sample(6, &Deterministic::new(4))
      .unwrap();
    let z = GaussianMultivariate::default().transform_to_normal(u.view());
    let raw = GaussianMultivariate::estimate_corr_from_normal(z.view());
    assert!(smallest_eigenvalue(raw.view()) < 1e-10);
    let mut fitted = GaussianMultivariate::default();
    fitted.fit(u.view()).unwrap();
    let corr = fitted.correlation().unwrap();
    assert!(smallest_eigenvalue(corr.view()) >= 1e-3 - 1e-12);
    assert!(condition_number(corr.view()) <= 6.0 / 1e-3);
  }

  /// Perfectly dependent columns estimate `0.999999`, a positive definite matrix below the floor, and fit to `0.999`.
  #[test]
  fn a_fit_of_perfectly_dependent_columns_stops_at_the_floor() {
    let column = Array1::from_shape_fn(50, |i| (i + 1) as f64 / 51.0);
    let u = ndarray::stack![Axis(1), column, column];
    let mut fitted = GaussianMultivariate::default();
    fitted.fit(u.view()).unwrap();
    let rho = fitted.correlation().unwrap()[[0, 1]];
    assert!((0.999 - 1e-6..=0.999).contains(&rho), "{rho}");
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
