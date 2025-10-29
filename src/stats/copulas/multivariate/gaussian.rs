use std::error::Error;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal as RandNormal;
use statrs::distribution::{ContinuousCDF, Normal};

use super::{CopulaType, Multivariate};

#[derive(Debug, Clone)]
pub struct GaussianMultivariate {
  dim: usize,
  /// Correlation matrix (dim x dim)
  corr: Option<Array2<f64>>,
  /// Inverse correlation matrix
  inv_corr: Option<Array2<f64>>,
  /// Lower-triangular Cholesky factor of corr
  chol_lower: Option<Array2<f64>>,
  /// Log determinant of corr
  log_det_corr: Option<f64>,
}

impl GaussianMultivariate {
  pub fn new() -> Self {
    Self {
      dim: 0,
      corr: None,
      inv_corr: None,
      chol_lower: None,
      log_det_corr: None,
    }
  }

  /// Create directly from a correlation matrix.
  pub fn new_with_corr(corr: Array2<f64>) -> Result<Self, Box<dyn Error>> {
    let dim = corr.nrows();
    if dim != corr.ncols() {
      return Err("Correlation matrix must be square".into());
    }
    let mut g = Self::new();
    g.set_corr(corr)?;
    Ok(g)
  }

  /// Returns a reference to the internal correlation matrix, if fitted.
  pub fn correlation(&self) -> Option<&Array2<f64>> {
    self.corr.as_ref()
  }

  fn set_corr(&mut self, corr: Array2<f64>) -> Result<(), Box<dyn Error>> {
    let dim = corr.nrows();
    self.dim = dim;

    // Use nalgebra for Cholesky and inverse robustly
    let corr_na =
      nalgebra::DMatrix::from_row_slice(dim, dim, corr.as_slice().ok_or("Non-contiguous matrix")?);
    let chol = match corr_na.clone().cholesky() {
      Some(c) => c,
      None => return Err("Correlation matrix is not positive definite".into()),
    };
    // Extract lower-triangular L
    let l = chol.l();
    // Compute log det from Cholesky: log det = 2 * sum log diag(L)
    let mut log_det = 0.0;
    for i in 0..dim {
      log_det += l[(i, i)].ln();
    }
    log_det *= 2.0;

    // Inverse via generic inverse (sufficient for small dims)
    let inv_na = match corr_na.try_inverse() {
      Some(m) => m,
      None => return Err("Failed to invert correlation matrix".into()),
    };

    // Convert nalgebra matrices back to ndarray
    let l_arr = Array2::from_shape_vec((dim, dim), l.as_slice().to_vec()).unwrap();
    let inv_arr = Array2::from_shape_vec((dim, dim), inv_na.as_slice().to_vec()).unwrap();

    self.corr = Some(corr);
    self.inv_corr = Some(inv_arr);
    self.chol_lower = Some(l_arr);
    self.log_det_corr = Some(log_det);
    Ok(())
  }

  /// Transform U in (0,1)^{n x d} to Z in R^{n x d} via standard normal inverse CDF.
  fn transform_to_normal(&self, u: &Array2<f64>) -> Array2<f64> {
    let std_norm = Normal::new(0.0, 1.0).unwrap();
    let eps = 1e-12;
    let mut z = u.clone();
    for mut row in z.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        let clamped = val.max(eps).min(1.0 - eps);
        *val = std_norm.inverse_cdf(clamped);
      }
    }
    z
  }

  /// Estimate correlation matrix from Z (n x d). Adds jitter if needed.
  fn estimate_corr_from_normal(&self, z: &Array2<f64>) -> Array2<f64> {
    let n = z.nrows() as f64;
    let d = z.ncols();
    // Center columns
    let means = z.mean_axis(Axis(0)).unwrap();
    let mut zc = z.clone();
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
        let r = (cov / (stds[i] * stds[j])).max(-0.999_999).min(0.999_999);
        corr[[i, j]] = r;
        corr[[j, i]] = r;
      }
    }
    // Ensure positive definiteness by adding jitter if necessary
    let mut jitter = 0usize;
    while !Self::is_spd(&corr) && jitter < 6 {
      let eps = 10f64.powi(-(6 as i32) + jitter as i32); // 1e-6,1e-5,...
      for k in 0..d {
        corr[[k, k]] = 1.0 + eps;
      }
      jitter += 1;
    }
    corr
  }

  fn is_spd(a: &Array2<f64>) -> bool {
    let dim = a.nrows();
    if dim != a.ncols() {
      return false;
    }
    if let Some(slice) = a.as_slice() {
      let m = nalgebra::DMatrix::from_row_slice(dim, dim, slice);
      m.cholesky().is_some()
    } else {
      false
    }
  }

  fn require_fitted(&self) -> Result<(), Box<dyn Error>> {
    if self.corr.is_none()
      || self.inv_corr.is_none()
      || self.chol_lower.is_none()
      || self.log_det_corr.is_none()
    {
      return Err("Fit the copula or provide a correlation matrix first".into());
    }
    Ok(())
  }
}

impl Multivariate for GaussianMultivariate {
  fn r#type(&self) -> CopulaType {
    CopulaType::Gaussian
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    self.require_fitted()?;
    let d = self.dim;
    let l = self.chol_lower.as_ref().unwrap(); // (d x d)
                                               // Sample standard normals G ~ N(0, I) of shape (n x d)
    let g = Array2::<f64>::random((n, d), RandNormal::new(0.0, 1.0).unwrap());
    // z = g * L^T
    let z = g.dot(&l.t());
    // Transform to uniforms using standard normal CDF
    let std_norm = Normal::new(0.0, 1.0).unwrap();
    let mut u = z.clone();
    for mut row in u.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        *val = std_norm.cdf(*val);
      }
    }
    Ok(u)
  }

  /// Fit the Gaussian copula from U in (0,1)^{n x d}.
  fn fit(&mut self, X: Array2<f64>) -> Result<(), Box<dyn Error>> {
    if X.nrows() < 2 || X.ncols() < 2 {
      return Err("Need at least 2 samples and 2 dimensions".into());
    }
    // Basic range check
    if X.iter().any(|&v| !(v >= 0.0 && v <= 1.0)) {
      return Err("Input data must be in [0,1] for Gaussian copula fit".into());
    }
    self.dim = X.ncols();
    let z = self.transform_to_normal(&X);
    let corr = self.estimate_corr_from_normal(&z);
    self.set_corr(corr)?;
    Ok(())
  }

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    self.require_fitted()?;
    if X.ncols() != self.dim {
      return Err("Dimension mismatch".into());
    }
    if X.iter().any(|&v| !(v >= 0.0 && v <= 1.0)) {
      return Err("Input X must be in [0,1] for Gaussian copula".into());
    }
    Ok(())
  }

  fn pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let z = self.transform_to_normal(&X); // n x d
    let inv = self.inv_corr.as_ref().unwrap();
    let log_det = self.log_det_corr.unwrap();

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

  fn cdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let z = self.transform_to_normal(&X); // n x d
    let l = self.chol_lower.as_ref().unwrap(); // d x d
    let n = z.nrows();
    let m_samples = 4000usize; // Monte Carlo samples per query point
    let mut out = Array1::<f64>::zeros(n);

    // Pre-sample standard normals for efficiency: (m x d)
    let g = Array2::<f64>::random((m_samples, self.dim), RandNormal::new(0.0, 1.0).unwrap());
    let y = g.dot(&l.t()); // (m x d) ~ MVN(0, corr)

    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut count = 0usize;
      'outer: for r in 0..m_samples {
        for c in 0..self.dim {
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
  use ndarray::{arr2, Array2};

  use super::*;

  #[test]
  fn gaussian_copula_sample_in_unit_cube() {
    let corr = arr2(&[[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]);
    let gm = GaussianMultivariate::new_with_corr(corr).unwrap();
    let u = gm.sample(1000).unwrap();
    assert_eq!(u.ncols(), 3);
    assert!(u.iter().all(|&v| v >= 0.0 && v <= 1.0));
  }

  #[test]
  fn gaussian_copula_pdf_positive() {
    let corr = arr2(&[[1.0, 0.6], [0.6, 1.0]]);
    let gm = GaussianMultivariate::new_with_corr(corr).unwrap();
    let u = gm.sample(256).unwrap();
    let d = gm.pdf(u).unwrap();
    assert!(d.iter().all(|&v| v.is_finite() && v > 0.0));
  }

  #[test]
  fn gaussian_copula_cdf_bounds_and_reasonable() {
    let corr = arr2(&[[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]]);
    let gm = GaussianMultivariate::new_with_corr(corr).unwrap();
    // Evaluate CDF at (0.5,0.5,0.5)
    let u = Array2::from_shape_vec((1, 3), vec![0.5, 0.5, 0.5]).unwrap();
    let c = gm.cdf(u).unwrap()[0];
    assert!(c >= 0.0 && c <= 1.0);
    // With positive correlation, expect > 0.125 (independent case)
    assert!(c > 0.10);
  }

  #[test]
  fn gaussian_copula_fit_recovers_corr() {
    let corr = arr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.0]]);
    let gm_true = GaussianMultivariate::new_with_corr(corr.clone()).unwrap();
    let u = gm_true.sample(4000).unwrap();

    let mut gm_fit = GaussianMultivariate::new();
    gm_fit.fit(u).unwrap();
    let est = gm_fit.correlation().unwrap().clone();

    for i in 0..3 {
      for j in 0..3 {
        let diff = (est[[i, j]] - corr[[i, j]]).abs();
        if i == j {
          assert!((est[[i, j]] - 1.0).abs() < 1e-6);
        } else {
          assert!(
            diff < 0.15,
            "diff {} at ({},{}): est={}, true={}",
            diff,
            i,
            j,
            est[[i, j]],
            corr[[i, j]]
          );
        }
      }
    }
  }
}
