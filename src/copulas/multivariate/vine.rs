use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use super::CopulaType;
use crate::copulas::correlation::kendall_tau;
use crate::traits::MultivariateExt;

#[derive(Debug, Clone, Default)]
pub struct VineMultivariate {
  dim: usize,
  corr: Option<Array2<f64>>, // correlation consistent with a simple C-vine (star)
  inv_corr: Option<Array2<f64>>,
  chol_lower: Option<Array2<f64>>,
  log_det_corr: Option<f64>,
}

impl VineMultivariate {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn new_with_corr(corr: Array2<f64>) -> Result<Self, Box<dyn Error>> {
    let mut s = Self::new();
    s.set_corr(corr)?;
    Ok(s)
  }

  pub fn correlation(&self) -> Option<&Array2<f64>> {
    self.corr.as_ref()
  }

  fn set_corr(&mut self, corr: Array2<f64>) -> Result<(), Box<dyn Error>> {
    let d = corr.nrows();
    if d != corr.ncols() {
      return Err("Correlation matrix must be square".into());
    }
    self.dim = d;
    let corr_na =
      nalgebra::DMatrix::from_row_slice(d, d, corr.as_slice().ok_or("Non-contiguous matrix")?);
    let chol = corr_na.clone().cholesky().ok_or("Correlation not PD")?;
    let l = chol.l();
    let mut log_det = 0.0;
    for i in 0..d {
      log_det += l[(i, i)].ln();
    }
    let log_det = 2.0 * log_det;
    let inv = corr_na.try_inverse().ok_or("Failed to invert corr")?;
    self.chol_lower = Some(Array2::from_shape_vec((d, d), l.as_slice().to_vec()).unwrap());
    self.inv_corr = Some(Array2::from_shape_vec((d, d), inv.as_slice().to_vec()).unwrap());
    self.corr = Some(corr);
    self.log_det_corr = Some(log_det);
    Ok(())
  }

  fn require_fitted(&self) -> Result<(), Box<dyn Error>> {
    if self.corr.is_none()
      || self.inv_corr.is_none()
      || self.chol_lower.is_none()
      || self.log_det_corr.is_none()
    {
      return Err("Fit the copula first".into());
    }
    Ok(())
  }

  fn transform_to_normal(&self, u: &Array2<f64>) -> Array2<f64> {
    let std_norm = Normal::new(0.0, 1.0).unwrap();
    let eps = 1e-12;
    let mut z = u.clone();
    for mut row in z.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        *val = std_norm.inverse_cdf(val.max(eps).min(1.0 - eps));
      }
    }
    z
  }

  fn tau_to_rho_gaussian(t: f64) -> f64 {
    (std::f64::consts::PI * 0.5 * t).sin()
  }
}

impl MultivariateExt for VineMultivariate {
  fn r#type(&self) -> CopulaType {
    CopulaType::Vine
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    self.require_fitted()?;
    let l = self.chol_lower.as_ref().unwrap();
    let d = self.dim;
    let std_norm = Normal::new(0.0, 1.0).unwrap();
    let mut g = Array2::<f64>::zeros((n, d));
    for i in 0..n {
      for j in 0..d {
        g[[i, j]] = std_norm.inverse_cdf(rand::random::<f64>().clamp(1e-12, 1.0 - 1e-12));
      }
    }
    let z = g.dot(&l.t());
    let mut u = z.clone();
    for mut row in u.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        *val = std_norm.cdf(*val);
      }
    }
    Ok(u)
  }

  fn fit(&mut self, X: Array2<f64>) -> Result<(), Box<dyn Error>> {
    if X.nrows() < 2 || X.ncols() < 2 {
      return Err("Need at least 2 samples and 2 dimensions".into());
    }
    let tau = kendall_tau(&X);
    let d = tau.nrows();
    let mut rho = Array2::<f64>::zeros((d, d));
    for i in 0..d {
      for j in 0..d {
        if i == j {
          rho[[i, i]] = 1.0;
        } else {
          rho[[i, j]] = Self::tau_to_rho_gaussian(tau[[i, j]]);
        }
      }
    }

    // Choose root maximizing sum of |rho|
    let mut best_root = 0usize;
    let mut best_sum = f64::NEG_INFINITY;
    for i in 0..d {
      let s: f64 = (0..d).filter(|&j| j != i).map(|j| rho[[i, j]].abs()).sum();
      if s > best_sum {
        best_sum = s;
        best_root = i;
      }
    }

    // Build star C-vine correlation
    let mut corr = Array2::<f64>::zeros((d, d));
    for i in 0..d {
      corr[[i, i]] = 1.0;
    }
    let r = best_root;
    for j in 0..d {
      if j != r {
        corr[[r, j]] = rho[[r, j]].clamp(-0.999_999, 0.999_999);
        corr[[j, r]] = corr[[r, j]];
      }
    }
    for i in 0..d {
      if i == r {
        continue;
      }
      for j in (i + 1)..d {
        if j == r {
          continue;
        }
        let v = (corr[[i, r]] * corr[[r, j]]).clamp(-0.999_999, 0.999_999);
        corr[[i, j]] = v;
        corr[[j, i]] = v;
      }
    }

    // ensure SPD via jitter
    let mut corr_try = corr.clone();
    let mut tries = 0;
    loop {
      if nalgebra::DMatrix::from_row_slice(d, d, corr_try.as_slice().unwrap())
        .cholesky()
        .is_some()
      {
        break;
      }
      for k in 0..d {
        corr_try[[k, k]] += 1e-6;
      }
      tries += 1;
      if tries > 6 {
        break;
      }
    }
    self.set_corr(corr_try)
  }

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    self.require_fitted()?;
    if X.ncols() != self.dim {
      return Err("Dimension mismatch".into());
    }
    if X.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
      return Err("Input must be in [0,1]".into());
    }
    Ok(())
  }

  fn pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let z = self.transform_to_normal(&X);
    let inv = self.inv_corr.as_ref().unwrap();
    let log_det = self.log_det_corr.unwrap();
    let mut out = Array1::<f64>::zeros(z.nrows());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut q = inv.dot(&row.to_owned());
      for k in 0..q.len() {
        q[k] -= row[k];
      }
      let quad = row.dot(&q);
      out[i] = (-0.5 * (log_det + quad)).exp();
    }
    Ok(out)
  }

  fn cdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit(&X)?;
    let z = self.transform_to_normal(&X);
    let l = self.chol_lower.as_ref().unwrap();
    let m = 4000usize;
    let std_norm = Normal::new(0.0, 1.0).unwrap();
    let mut g = Array2::<f64>::zeros((m, self.dim));
    for i in 0..m {
      for j in 0..self.dim {
        g[[i, j]] = std_norm.inverse_cdf(rand::random::<f64>().clamp(1e-12, 1.0 - 1e-12));
      }
    }
    let y = g.dot(&l.t());
    let mut out = Array1::<f64>::zeros(z.nrows());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
      let mut count = 0usize;
      'outer: for r in 0..m {
        for c in 0..self.dim {
          if y[[r, c]] > row[c] {
            continue 'outer;
          }
        }
        count += 1;
      }
      out[i] = count as f64 / m as f64;
    }
    Ok(out)
  }
}
