//! # Tree
//!
//! $$
//! \text{Tree copula factorization over pair-copulas on spanning levels}
//! $$
//!
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
pub struct TreeMultivariate {
  dim: usize,
  corr: Option<Array2<f64>>,       // full correlation implied by the tree
  inv_corr: Option<Array2<f64>>,   // inverse correlation
  chol_lower: Option<Array2<f64>>, // Cholesky factor (lower)
  log_det_corr: Option<f64>,
}

impl TreeMultivariate {
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
    let dim = corr.nrows();
    if dim != corr.ncols() {
      return Err("Correlation matrix must be square".into());
    }
    self.dim = dim;

    let corr_na =
      nalgebra::DMatrix::from_row_slice(dim, dim, corr.as_slice().ok_or("Non-contiguous matrix")?);
    let chol = match corr_na.clone().cholesky() {
      Some(c) => c,
      None => return Err("Correlation matrix is not positive definite".into()),
    };
    let l = chol.l();
    let mut log_det = 0.0;
    for i in 0..dim {
      log_det += l[(i, i)].ln();
    }
    log_det *= 2.0;

    let inv_na = match corr_na.try_inverse() {
      Some(m) => m,
      None => return Err("Failed to invert correlation matrix".into()),
    };

    let l_arr = Array2::from_shape_vec((dim, dim), l.as_slice().to_vec()).unwrap();
    let inv_arr = Array2::from_shape_vec((dim, dim), inv_na.as_slice().to_vec()).unwrap();

    self.corr = Some(corr);
    self.inv_corr = Some(inv_arr);
    self.chol_lower = Some(l_arr);
    self.log_det_corr = Some(log_det);
    Ok(())
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

  fn transform_to_normal(&self, u: &Array2<f64>) -> Array2<f64> {
    let std_norm = Normal::new(0.0, 1.0).unwrap();
    let eps = 1e-12;
    let mut z = u.clone();
    for mut row in z.axis_iter_mut(Axis(0)) {
      for val in row.iter_mut() {
        let clamped = (*val).max(eps).min(1.0 - eps);
        *val = std_norm.inverse_cdf(clamped);
      }
    }
    z
  }

  fn tau_to_rho_gaussian(t: f64) -> f64 {
    (std::f64::consts::PI * 0.5 * t).sin()
  }

  fn build_mst_rho(&self, rho: &Array2<f64>) -> (Vec<Vec<usize>>, Array2<f64>) {
    let d = rho.nrows();
    let mut in_tree = vec![false; d];
    let mut best_w = vec![-1.0; d];
    let mut parent: Vec<Option<usize>> = vec![None; d];

    in_tree[0] = true;
    for j in 1..d {
      best_w[j] = rho[[0, j]].abs();
      parent[j] = Some(0);
    }

    let mut adj: Vec<Vec<usize>> = vec![vec![]; d];
    let mut edge_r = Array2::<f64>::zeros((d, d));

    for _ in 0..(d - 1) {
      // pick max weight outside tree
      let mut best = -1.0;
      let mut v = None;
      for j in 0..d {
        if !in_tree[j] && best_w[j] > best {
          best = best_w[j];
          v = Some(j);
        }
      }
      let v = v.expect("graph must be connected over rho");
      let p = parent[v].unwrap();
      in_tree[v] = true;

      // add edge both ways
      adj[p].push(v);
      adj[v].push(p);
      edge_r[[p, v]] = rho[[p, v]];
      edge_r[[v, p]] = rho[[v, p]];

      // update best_w
      for k in 0..d {
        if !in_tree[k] && rho[[v, k]].abs() > best_w[k] {
          best_w[k] = rho[[v, k]].abs();
          parent[k] = Some(v);
        }
      }
    }

    (adj, edge_r)
  }

  fn corr_from_tree_edges(&self, adj: &[Vec<usize>], edge_r: &Array2<f64>) -> Array2<f64> {
    let d = edge_r.nrows();
    let mut corr = Array2::<f64>::zeros((d, d));
    for i in 0..d {
      corr[[i, i]] = 1.0;
    }

    for s in 0..d {
      let mut visited = vec![false; d];
      let mut stack: Vec<(usize, usize, f64)> = vec![]; // (node, parent, corr_prod)
      visited[s] = true;
      for &nbr in &adj[s] {
        stack.push((nbr, s, edge_r[[s, nbr]]));
        visited[nbr] = true;
      }

      while let Some((node, parent, prod)) = stack.pop() {
        corr[[s, node]] = prod;
        corr[[node, s]] = prod;
        for &nbr in &adj[node] {
          if nbr == parent {
            continue;
          }
          if !visited[nbr] {
            visited[nbr] = true;
            stack.push((nbr, node, prod * edge_r[[node, nbr]]));
          }
        }
      }
    }

    corr
  }
}

impl MultivariateExt for TreeMultivariate {
  fn r#type(&self) -> CopulaType {
    CopulaType::Tree
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    self.require_fitted()?;
    let l = self.chol_lower.as_ref().unwrap();
    let d = self.dim;
    // standard normals
    let mut z = Array2::<f64>::zeros((n, d));
    let std_norm = Normal::new(0.0, 1.0).unwrap();
    for i in 0..n {
      for j in 0..d {
        z[[i, j]] = std_norm.inverse_cdf(rand::random::<f64>().clamp(1e-12, 1.0 - 1e-12));
      }
    }
    let z = z.dot(&l.t());
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
    // Kendall's tau on uniforms, map to Gaussian rho
    let tau = kendall_tau(&X);
    let d = tau.nrows();
    let mut rho = Array2::<f64>::zeros((d, d));
    for i in 0..d {
      for j in 0..d {
        if i == j {
          rho[[i, j]] = 1.0;
        } else {
          rho[[i, j]] = Self::tau_to_rho_gaussian(tau[[i, j]]);
        }
      }
    }

    let (adj, edge_r) = self.build_mst_rho(&rho);
    let corr = self.corr_from_tree_edges(&adj, &edge_r);
    // ensure SPD: slight jitter if needed
    let mut tries = 0;
    let mut corr_try = corr.clone();
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
      return Err("Input X must be in [0,1]".into());
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
    // Monte Carlo like Gaussian
    let z = self.transform_to_normal(&X);
    let l = self.chol_lower.as_ref().unwrap();
    let m = 4000usize;
    let g = {
      let std_norm = Normal::new(0.0, 1.0).unwrap();
      let mut g = Array2::<f64>::zeros((m, self.dim));
      for i in 0..m {
        for j in 0..self.dim {
          g[[i, j]] = std_norm.inverse_cdf(rand::random::<f64>().clamp(1e-12, 1.0 - 1e-12));
        }
      }
      g
    };
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