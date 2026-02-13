//! Dupire local volatility from call price surface
//! σ_loc^2(K,T) = [ ∂C/∂T + (r - q) K ∂C/∂K + q C ] / [ 0.5 K^2 ∂²C/∂K² ]

use ndarray::Array2;
use ndarray::Axis;

#[derive(Clone, Debug)]
pub struct Dupire {
  /// Strikes (ascending), length N_K
  pub ks: Vec<f64>,
  /// Maturities in years (ascending), length N_T
  pub ts: Vec<f64>,
  /// Call price surface with shape (N_T, N_K), row = fixed T_j, col = K_i, values are present call prices C(K_i, T_j)
  pub calls: Array2<f64>,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: f64,
  /// Small number to stabilize division when ∂²C/∂K² is near zero
  pub eps: f64,
  /// Optional pre-calculated ∂C/∂K with shape (N_T, N_K) (overrides finite-difference computation)
  pub dc_dk: Option<Array2<f64>>,
  /// Optional pre-calculated ∂²C/∂K² with shape (N_T, N_K) (overrides finite-difference computation)
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
}

impl Dupire {
  /// Compute local volatility surface on the same (T, K) grid as the input call surface.
  ///
  /// Returns Array2 (N_T, N_K) with σ_loc(K_i, T_j); boundaries in K use NaN where the second derivative is ill-defined.
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

  /// Compute local volatility surface using pre-calculated partial derivatives ∂C/∂K, ∂²C/∂K², ∂C/∂T.
  /// Requires `dc_dk`, `d2c_dk2`, and `dc_dt` fields to be populated.
  ///
  /// Returns Array2 (N_T, N_K) with σ_loc(K_i, T_j).
  #[must_use]
  pub fn local_vol_surface_from_custom_derivatives(&self) -> Array2<f64> {
    let dc_dk = self
      .dc_dk
      .as_ref()
      .expect("dc_dk must be provided for custom derivatives");
    let d2c_dk2 = self
      .d2c_dk2
      .as_ref()
      .expect("d2c_dk2 must be provided for custom derivatives");
    let dc_dt = self
      .dc_dt
      .as_ref()
      .expect("dc_dt must be provided for custom derivatives");

    let nt = self.ts.len();
    let nk = self.ks.len();

    assert_eq!(dc_dk.dim(), (nt, nk), "dc_dk must have shape (N_T, N_K)");
    assert_eq!(
      d2c_dk2.dim(),
      (nt, nk),
      "d2c_dk2 must have shape (N_T, N_K)"
    );
    assert_eq!(dc_dt.dim(), (nt, nk), "dc_dt must have shape (N_T, N_K)");

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

    sigma
  }

  /// Convenience: compute local volatility for a single maturity slice at time index j.
  #[must_use]
  pub fn local_vol_slice(&self, j: usize) -> Vec<f64> {
    assert!(j < self.ts.len());
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

  /// Compute local volatility for a single maturity slice using pre-calculated partial derivatives.
  /// Requires `dc_dk`, `d2c_dk2`, and `dc_dt` fields to be populated.
  #[must_use]
  pub fn local_vol_slice_from_custom_derivatives(&self, j: usize) -> Vec<f64> {
    assert!(j < self.ts.len());

    let dc_dk = self
      .dc_dk
      .as_ref()
      .expect("dc_dk must be provided for custom derivatives");
    let d2c_dk2 = self
      .d2c_dk2
      .as_ref()
      .expect("d2c_dk2 must be provided for custom derivatives");
    let dc_dt = self
      .dc_dt
      .as_ref()
      .expect("dc_dt must be provided for custom derivatives");

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

    out
  }
}
