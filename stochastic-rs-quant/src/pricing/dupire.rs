//! Dupire local volatility from a call-price surface on a strike–maturity grid:
//!
//! $$
//! \sigma_{loc}^2(K,T)=\frac{\partial_T C + (r-q)K\partial_K C + qC}{\tfrac12 K^2\partial_{KK}C}
//! $$
//!
use ndarray::Array2;
use ndarray::Axis;

use crate::QuantError;
use crate::error::check_grid_shape;

#[derive(Clone, Debug)]
pub struct Dupire {
  /// Strikes (ascending), length N_K
  pub ks: Vec<f64>,
  /// Maturities in years (ascending), length N_T
  pub ts: Vec<f64>,
  /// Call price surface with shape (N_T, N_K), row = fixed T_j, col = K_i, values are present call
  /// prices C(K_i, T_j)
  pub calls: Array2<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield
  pub q: f64,
  /// Small number to stabilize division when ∂²C/∂K² is near zero
  pub eps: f64,
  /// Optional pre-calculated ∂C/∂K with shape (N_T, N_K) (overrides finite-difference computation)
  pub dc_dk: Option<Array2<f64>>,
  /// Optional pre-calculated ∂²C/∂K² with shape (N_T, N_K) (overrides finite-difference
  /// computation)
  pub d2c_dk2: Option<Array2<f64>>,
  /// Optional pre-calculated ∂C/∂T with shape (N_T, N_K) (overrides finite-difference computation)
  pub dc_dt: Option<Array2<f64>>,
}

impl Dupire {
  pub fn new(
    ks: Vec<f64>,
    ts: Vec<f64>,
    calls: Array2<f64>,
    r: f64,
    q: f64,
    eps: f64,
    dc_dk: Option<Array2<f64>>,
    d2c_dk2: Option<Array2<f64>>,
    dc_dt: Option<Array2<f64>>,
  ) -> Self {
    Self {
      ks,
      ts,
      calls,
      r,
      q,
      eps,
      dc_dk,
      d2c_dk2,
      dc_dt,
    }
  }

  /// Builder for fluent construction with sensible defaults.
  pub fn builder(ks: Vec<f64>, ts: Vec<f64>, calls: Array2<f64>) -> DupireBuilder {
    DupireBuilder {
      ks,
      ts,
      calls,
      r: 0.0,
      q: 0.0,
      eps: 1e-6,
      dc_dk: None,
      d2c_dk2: None,
      dc_dt: None,
    }
  }
}

#[derive(Debug, Clone)]
pub struct DupireBuilder {
  ks: Vec<f64>,
  ts: Vec<f64>,
  calls: Array2<f64>,
  r: f64,
  q: f64,
  eps: f64,
  dc_dk: Option<Array2<f64>>,
  d2c_dk2: Option<Array2<f64>>,
  dc_dt: Option<Array2<f64>>,
}

impl DupireBuilder {
  pub fn r(mut self, r: f64) -> Self {
    self.r = r;
    self
  }

  pub fn q(mut self, q: f64) -> Self {
    self.q = q;
    self
  }

  pub fn eps(mut self, eps: f64) -> Self {
    self.eps = eps;
    self
  }

  pub fn dc_dk(mut self, dc_dk: Array2<f64>) -> Self {
    self.dc_dk = Some(dc_dk);
    self
  }

  pub fn d2c_dk2(mut self, d2c_dk2: Array2<f64>) -> Self {
    self.d2c_dk2 = Some(d2c_dk2);
    self
  }

  pub fn dc_dt(mut self, dc_dt: Array2<f64>) -> Self {
    self.dc_dt = Some(dc_dt);
    self
  }

  pub fn build(self) -> Dupire {
    Dupire {
      ks: self.ks,
      ts: self.ts,
      calls: self.calls,
      r: self.r,
      q: self.q,
      eps: self.eps,
      dc_dk: self.dc_dk,
      d2c_dk2: self.d2c_dk2,
      dc_dt: self.dc_dt,
    }
  }
}

impl Dupire {
  /// `σ_loc` on the input `(T, K)` grid, `(N_T, N_K)`; `NaN`, not a usable-looking `0`, at the `K`
  /// edges, a zero-`dt` stencil, or a `½K²∂²C/∂K²` below `eps`, non-finite or giving `σ² < 0`.
  #[must_use]
  pub fn local_vol_surface(&self) -> Array2<f64> {
    assert_eq!(
      self.calls.dim().0,
      self.ts.len(),
      "calls rows must match ts length"
    );
    assert_eq!(
      self.calls.dim().1,
      self.ks.len(),
      "calls cols must match ks length"
    );

    let nt = self.ts.len();
    let nk = self.ks.len();

    let mut sigma = Array2::<f64>::from_elem((nt, nk), f64::NAN);

    for j in 0..nt {
      for i in 1..nk - 1 {
        let k_im1 = self.ks[i - 1];
        let k_i = self.ks[i];
        let k_ip1 = self.ks[i + 1];

        let c_im1 = self.calls[[j, i - 1]];
        let c_i = self.calls[[j, i]];
        let c_ip1 = self.calls[[j, i + 1]];

        // ∂C/∂K at (j,i) using non-uniform 3-pt central stencil
        let h_i = k_i - k_im1;
        let h_ip1 = k_ip1 - k_i;
        let dcdk = (-h_ip1 / (h_i * (h_i + h_ip1))) * c_im1
          + ((h_ip1 - h_i) / (h_i * h_ip1)) * c_i
          + (h_i / (h_ip1 * (h_i + h_ip1))) * c_ip1;

        // ∂²C/∂K² at (j,i) using non-uniform 3-pt stencil
        let denom_left = h_i * (h_i + h_ip1);
        let denom_mid = h_i * h_ip1;
        let denom_right = h_ip1 * (h_i + h_ip1);
        let d2cdk2 = 2.0 * (c_im1 / denom_left - c_i / denom_mid + c_ip1 / denom_right);

        // ∂C/∂T at (j,i) using central difference in T (non-uniform aware)
        let dcdt = if j == 0 {
          let dt = self.ts[1] - self.ts[0];
          if dt.abs() < f64::EPSILON {
            f64::NAN
          } else {
            (self.calls[[1, i]] - self.calls[[0, i]]) / dt
          }
        } else if j == nt - 1 {
          let dt = self.ts[nt - 1] - self.ts[nt - 2];
          if dt.abs() < f64::EPSILON {
            f64::NAN
          } else {
            (self.calls[[nt - 1, i]] - self.calls[[nt - 2, i]]) / dt
          }
        } else {
          let dt = self.ts[j + 1] - self.ts[j - 1];
          if dt.abs() < f64::EPSILON {
            f64::NAN
          } else {
            (self.calls[[j + 1, i]] - self.calls[[j - 1, i]]) / dt
          }
        };

        let denom = 0.5 * k_i * k_i * d2cdk2;
        let numer = dcdt + (self.r - self.q) * k_i * dcdk + self.q * c_i;

        if denom.abs() > self.eps && denom.is_finite() && numer.is_finite() {
          let s2 = numer / denom;
          if s2.is_sign_positive() {
            sigma[[j, i]] = s2.max(0.0).sqrt();
          }
        }
      }
    }

    sigma
  }

  /// [`local_vol_surface`](Self::local_vol_surface) from the supplied `dc_dk`, `d2c_dk2`, `dc_dt`;
  /// panics if one is `None` (see [`Self::try_local_vol_surface_from_custom_derivatives`]).
  #[must_use]
  pub fn local_vol_surface_from_custom_derivatives(&self) -> Array2<f64> {
    self
      .try_local_vol_surface_from_custom_derivatives()
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible [`Self::local_vol_surface_from_custom_derivatives`]: `Err` for a `None` or misshapen
  /// derivative; cells follow the `NaN` contract but for the stencil edges.
  pub fn try_local_vol_surface_from_custom_derivatives(&self) -> Result<Array2<f64>, QuantError> {
    let (dc_dk, d2c_dk2, dc_dt) = self.custom_derivatives()?;
    let nt = self.ts.len();
    let nk = self.ks.len();

    let mut sigma = Array2::<f64>::from_elem((nt, nk), f64::NAN);

    for j in 0..nt {
      for i in 0..nk {
        let k_i = self.ks[i];
        let c_i = self.calls[[j, i]];

        let dcdk = dc_dk[[j, i]];
        let d2cdk2 = d2c_dk2[[j, i]];
        let dcdt = dc_dt[[j, i]];

        let denom = 0.5 * k_i * k_i * d2cdk2;
        let numer = dcdt + (self.r - self.q) * k_i * dcdk + self.q * c_i;

        if denom.abs() > self.eps && denom.is_finite() && numer.is_finite() {
          let s2 = numer / denom;
          if s2.is_sign_positive() {
            sigma[[j, i]] = s2.max(0.0).sqrt();
          }
        }
      }
    }

    Ok(sigma)
  }

  /// Maturity slice `j` under [`local_vol_surface`](Self::local_vol_surface)'s `NaN` contract; all
  /// `NaN` below 3 strikes.
  #[must_use]
  pub fn local_vol_slice(&self, j: usize) -> Vec<f64> {
    assert!(
      j < self.ts.len(),
      "j must satisfy `j < self.ts.len()`, got j = {j:?}"
    );
    let row = self.calls.index_axis(Axis(0), j);
    let mut out = vec![f64::NAN; self.ks.len()];

    if self.ks.len() < 3 {
      return out;
    }

    for i in 1..self.ks.len() - 1 {
      let k_im1 = self.ks[i - 1];
      let k_i = self.ks[i];
      let k_ip1 = self.ks[i + 1];

      let c_im1 = row[i - 1];
      let c_i = row[i];
      let c_ip1 = row[i + 1];

      let h_i = k_i - k_im1;
      let h_ip1 = k_ip1 - k_i;

      let dcdk = (-h_ip1 / (h_i * (h_i + h_ip1))) * c_im1
        + ((h_ip1 - h_i) / (h_i * h_ip1)) * c_i
        + (h_i / (h_ip1 * (h_i + h_ip1))) * c_ip1;

      let denom_left = h_i * (h_i + h_ip1);
      let denom_mid = h_i * h_ip1;
      let denom_right = h_ip1 * (h_i + h_ip1);
      let d2cdk2 = 2.0 * (c_im1 / denom_left - c_i / denom_mid + c_ip1 / denom_right);

      let dcdt = if j == 0 {
        let dt = self.ts[1] - self.ts[0];
        if dt.abs() < f64::EPSILON {
          f64::NAN
        } else {
          (self.calls[[1, i]] - self.calls[[0, i]]) / dt
        }
      } else if j == self.ts.len() - 1 {
        let dt = self.ts[j] - self.ts[j - 1];
        if dt.abs() < f64::EPSILON {
          f64::NAN
        } else {
          (self.calls[[j, i]] - self.calls[[j - 1, i]]) / dt
        }
      } else {
        let dt = self.ts[j + 1] - self.ts[j - 1];
        if dt.abs() < f64::EPSILON {
          f64::NAN
        } else {
          (self.calls[[j + 1, i]] - self.calls[[j - 1, i]]) / dt
        }
      };

      let denom = 0.5 * k_i * k_i * d2cdk2;
      let numer = dcdt + (self.r - self.q) * k_i * dcdk + self.q * c_i;

      if denom.abs() > self.eps && denom.is_finite() && numer.is_finite() {
        let s2 = numer / denom;
        if s2.is_sign_positive() {
          out[i] = s2.max(0.0).sqrt();
        }
      }
    }

    out
  }

  /// One slice from the supplied derivatives; panics if one is `None` (see
  /// [`Self::try_local_vol_slice_from_custom_derivatives`]).
  #[must_use]
  pub fn local_vol_slice_from_custom_derivatives(&self, j: usize) -> Vec<f64> {
    self
      .try_local_vol_slice_from_custom_derivatives(j)
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Fallible [`Self::local_vol_slice_from_custom_derivatives`]: `Err` for a `None` or misshapen
  /// derivative or an out-of-range `j`; cells follow the `NaN` contract.
  pub fn try_local_vol_slice_from_custom_derivatives(
    &self,
    j: usize,
  ) -> Result<Vec<f64>, QuantError> {
    if j >= self.ts.len() {
      return Err(QuantError::invalid(
        "j",
        format!("must be below ts.len() = {}, got {j}", self.ts.len()),
      ));
    }
    let (dc_dk, d2c_dk2, dc_dt) = self.custom_derivatives()?;

    let row = self.calls.index_axis(Axis(0), j);
    let mut out = vec![f64::NAN; self.ks.len()];

    for i in 0..self.ks.len() {
      let k_i = self.ks[i];
      let c_i = row[i];

      let dcdk = dc_dk[[j, i]];
      let d2cdk2 = d2c_dk2[[j, i]];
      let dcdt = dc_dt[[j, i]];

      let denom = 0.5 * k_i * k_i * d2cdk2;
      let numer = dcdt + (self.r - self.q) * k_i * dcdk + self.q * c_i;

      if denom.abs() > self.eps && denom.is_finite() && numer.is_finite() {
        let s2 = numer / denom;
        if s2.is_sign_positive() {
          out[i] = s2.max(0.0).sqrt();
        }
      }
    }

    Ok(out)
  }

  fn custom_derivatives(&self) -> Result<(&Array2<f64>, &Array2<f64>, &Array2<f64>), QuantError> {
    let missing = |what| QuantError::MissingInput {
      what,
      detail: "the custom-derivative local volatility reads it".to_string(),
    };
    let dc_dk = self.dc_dk.as_ref().ok_or_else(|| missing("dc_dk"))?;
    let d2c_dk2 = self.d2c_dk2.as_ref().ok_or_else(|| missing("d2c_dk2"))?;
    let dc_dt = self.dc_dt.as_ref().ok_or_else(|| missing("dc_dt"))?;
    let shape = (self.ts.len(), self.ks.len());
    check_grid_shape(
      [
        "dc_dk rows, one per maturity",
        "dc_dk columns, one per strike",
      ],
      shape,
      dc_dk.dim(),
    )?;
    check_grid_shape(
      [
        "d2c_dk2 rows, one per maturity",
        "d2c_dk2 columns, one per strike",
      ],
      shape,
      d2c_dk2.dim(),
    )?;
    check_grid_shape(
      [
        "dc_dt rows, one per maturity",
        "dc_dt columns, one per strike",
      ],
      shape,
      dc_dt.dim(),
    )?;
    Ok((dc_dk, d2c_dk2, dc_dt))
  }
}
