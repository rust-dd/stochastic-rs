//! Breeden–Litzenberger formula utilities
//! f_{RN}(K, T) = e^{r T} * ∂²C(K,T)/∂K² (same for P)

#[derive(Clone, Debug)]
pub struct BreedenLitzenberger {
  /// Strikes (strictly increasing)
  pub strikes: Vec<f64>,
  /// Option prices C(K[i], T) or P(K[i], T) at the same maturity T (present values)
  pub prices: Vec<f64>,
  /// Risk-free rate
  pub r: f64,
  /// Time to maturity in years
  pub tau: f64,
  /// Optional pre-calculated second derivative ∂²C/∂K² at each strike (overrides finite-difference computation)
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
  /// Compute risk–neutral density across strikes using finite differences on a (possibly non-uniform) grid.
  ///
  /// Returns a Vec of length strikes.len() with the estimated density at each strike.
  /// Endpoints (i = 0 and i = n-1) are set to NaN as second derivatives are ill-defined there with 3-point stencils.
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

  /// Compute risk–neutral density using pre-calculated second derivatives ∂²C/∂K².
  /// Requires `d2c_dk2` field to be populated.
  ///
  /// Returns a Vec of length strikes.len() with the estimated density at each strike.
  #[must_use]
  pub fn density_from_custom_derivatives(&self) -> Vec<f64> {
    let d2c = self
      .d2c_dk2
      .as_ref()
      .expect("d2c_dk2 must be provided for custom derivatives");
    let n = self.strikes.len();
    assert_eq!(d2c.len(), n, "d2c_dk2 must have same length as strikes");

    d2c.iter().map(|d| (self.r * self.tau).exp() * d).collect()
  }
}
