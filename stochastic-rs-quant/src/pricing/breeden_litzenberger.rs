//! Breeden–Litzenberger risk-neutral density from the second strike derivative of calls:
//!
//! $$
//! f_{\mathbb Q}(K)=e^{rT}\frac{\partial^2 C(K,T)}{\partial K^2}
//! $$
//!

use crate::QuantError;

#[derive(Clone, Debug)]
pub struct BreedenLitzenberger {
  /// Strikes (strictly increasing)
  pub strikes: Vec<f64>,
  /// Option prices `C(K_i, T)` or `P(K_i, T)` at the same maturity `T` (present values).
  pub prices: Vec<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Time to maturity in years.
  pub tau: f64,
  /// Optional pre-calculated second derivative ∂²C/∂K² at each strike (overrides finite-difference
  /// computation)
  pub d2c_dk2: Option<Vec<f64>>,
}

impl BreedenLitzenberger {
  pub fn new(
    strikes: Vec<f64>,
    prices: Vec<f64>,
    r: f64,
    tau: f64,
    d2c_dk2: Option<Vec<f64>>,
  ) -> Self {
    Self {
      strikes,
      prices,
      r,
      tau,
      d2c_dk2,
    }
  }
}

impl BreedenLitzenberger {
  /// Risk-neutral density at each strike by finite differences on a possibly non-uniform grid;
  /// `NaN` at both endpoints (no 3-point stencil fits) and everywhere below 3 strikes.
  #[must_use]
  pub fn density(&self) -> Vec<f64> {
    assert!(
      self.strikes.len() == self.prices.len(),
      "strikes and prices must have same length"
    );
    let n = self.strikes.len();
    let mut dens = vec![f64::NAN; n];
    if n < 3 {
      return dens;
    }

    for (dens_val, (k_win, c_win)) in dens[1..n - 1]
      .iter_mut()
      .zip(self.strikes.windows(3).zip(self.prices.windows(3)))
    {
      let h_i = k_win[1] - k_win[0];
      let h_ip1 = k_win[2] - k_win[1];

      let denom_left = h_i * (h_i + h_ip1);
      let denom_mid = h_i * h_ip1;
      let denom_right = h_ip1 * (h_i + h_ip1);

      let cdd = 2.0 * (c_win[0] / denom_left - c_win[1] / denom_mid + c_win[2] / denom_right);
      *dens_val = (self.r * self.tau).exp() * cdd;
    }

    dens
  }

  /// The density from the supplied `∂²C/∂K²`; panics if `d2c_dk2` is `None` or of the wrong length
  /// (see [`Self::try_density_from_custom_derivatives`]).
  #[must_use]
  pub fn density_from_custom_derivatives(&self) -> Vec<f64> {
    self
      .try_density_from_custom_derivatives()
      .unwrap_or_else(|e| panic!("{e}"))
  }

  /// Falliable variant of [`Self::density_from_custom_derivatives`].
  /// Returns an error if `d2c_dk2` is `None` or its length does not match `strikes`.
  pub fn try_density_from_custom_derivatives(&self) -> Result<Vec<f64>, QuantError> {
    let d2c = self
      .d2c_dk2
      .as_ref()
      .ok_or_else(|| QuantError::MissingInput {
        what: "d2c_dk2",
        detail: "the custom-derivative density reads it".to_string(),
      })?;
    let n = self.strikes.len();
    if d2c.len() != n {
      return Err(QuantError::DimensionMismatch {
        what: "d2c_dk2, one per strike",
        expected: n,
        got: d2c.len(),
      });
    }

    Ok(d2c.iter().map(|d| (self.r * self.tau).exp() * d).collect())
  }
}
