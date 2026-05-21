use anyhow::Result;
use anyhow::bail;
use ndarray::Array2;

#[derive(Clone, Debug)]
pub(super) struct BoundedScaler {
  lb: Vec<f32>,
  ub: Vec<f32>,
}

impl BoundedScaler {
  pub(super) fn new(lb: Vec<f32>, ub: Vec<f32>) -> Self {
    Self { lb, ub }
  }

  pub(super) fn scale_array2(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
    if x.ncols() != self.lb.len() {
      bail!("input width mismatch for bounded scaling");
    }
    let mut out = x.clone();
    for i in 0..out.nrows() {
      for j in 0..out.ncols() {
        let center = 0.5 * (self.lb[j] + self.ub[j]);
        let half_range = 0.5 * (self.ub[j] - self.lb[j]);
        out[[i, j]] = (x[[i, j]] - center) / half_range;
      }
    }
    Ok(out)
  }

  pub(super) fn scale_vector(&self, x: &[f32]) -> Result<Vec<f32>> {
    if x.len() != self.lb.len() {
      bail!("input length mismatch for bounded scaling");
    }
    let mut out = vec![0.0_f32; x.len()];
    for i in 0..x.len() {
      let center = 0.5 * (self.lb[i] + self.ub[i]);
      let half_range = 0.5 * (self.ub[i] - self.lb[i]);
      out[i] = (x[i] - center) / half_range;
    }
    Ok(out)
  }
}

#[derive(Clone, Debug)]
pub(super) struct StandardScaler {
  pub(super) mean: Vec<f32>,
  pub(super) std: Vec<f32>,
}

impl StandardScaler {
  pub(super) fn fit(data: &Array2<f32>) -> Result<Self> {
    if data.nrows() == 0 || data.ncols() == 0 {
      bail!("cannot fit StandardScaler on empty matrix");
    }
    let rows = data.nrows();
    let cols = data.ncols();
    let mut mean = vec![0.0_f32; cols];
    let mut std = vec![0.0_f32; cols];

    for j in 0..cols {
      let mut s = 0.0_f32;
      for i in 0..rows {
        s += data[[i, j]];
      }
      mean[j] = s / rows as f32;
    }

    for j in 0..cols {
      let mut s2 = 0.0_f32;
      for i in 0..rows {
        let d = data[[i, j]] - mean[j];
        s2 += d * d;
      }
      std[j] = (s2 / rows as f32).sqrt().max(1e-6);
    }

    Ok(Self { mean, std })
  }

  pub(super) fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
    if data.ncols() != self.mean.len() {
      bail!("input width mismatch for StandardScaler::transform");
    }
    let mut out = data.clone();
    for i in 0..out.nrows() {
      for j in 0..out.ncols() {
        out[[i, j]] = (data[[i, j]] - self.mean[j]) / self.std[j];
      }
    }
    Ok(out)
  }

  pub(super) fn inverse_transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
    if data.ncols() != self.mean.len() {
      bail!("input width mismatch for StandardScaler::inverse_transform");
    }
    let mut out = data.clone();
    for i in 0..out.nrows() {
      for j in 0..out.ncols() {
        out[[i, j]] = data[[i, j]] * self.std[j] + self.mean[j];
      }
    }
    Ok(out)
  }
}
