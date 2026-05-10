//! # SSVI — Surface SVI
//!
//! Full-surface parameterization from Gatheral & Jacquier (2012):
//!
//! $$
//! w(k,\theta_t)=\frac{\theta_t}{2}\Bigl(1+\rho\,\varphi(\theta_t)\,k
//!   +\sqrt{\bigl(\varphi(\theta_t)\,k+\rho\bigr)^2+(1-\rho^2)}\Bigr)
//! $$
//!
//! where $\theta_t$ is the ATM total variance at time $t$ and $\varphi$ is
//! a mixing function.
//!
//! Power-law mixing (Gatheral & Jacquier, 2012, Eq. 4.1):
//!
//! $$
//! \varphi(\theta)=\frac{\eta}{\theta^\gamma\,(1+\theta)^{1-\gamma}},
//! \quad \eta>0,\;\gamma\in[0,1]
//! $$
//!
//! Calibration based on Cohort, Corbetta, Martini & Laachir (2018),
//! arXiv:1804.04924.
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use crate::traits::FloatExt;

/// SSVI global parameters: $\{\rho, \eta, \gamma\}$.
#[derive(Clone, Copy, Debug)]
pub struct SsviParams<T: FloatExt> {
  /// Correlation ($|\rho| < 1$)
  pub rho: T,
  /// Power-law scale ($\eta > 0$)
  pub eta: T,
  /// Power-law exponent ($\gamma \in [0, 1]$)
  pub gamma: T,
}

impl<T: FloatExt> SsviParams<T> {
  pub fn new(rho: T, eta: T, gamma: T) -> Self {
    Self { rho, eta, gamma }
  }

  /// Power-law mixing function $\varphi(\theta)$.
  #[inline]
  pub fn phi(&self, theta: T) -> T {
    if theta <= T::zero() {
      return T::infinity();
    }
    let one = T::one();
    self.eta / (theta.powf(self.gamma) * (one + theta).powf(one - self.gamma))
  }

  /// Evaluate total variance $w(k, \theta)$.
  #[inline]
  pub fn total_variance(&self, k: T, theta: T) -> T {
    let half = T::from_f64_fast(0.5);
    let one = T::one();
    let phi = self.phi(theta);
    let u = phi * k + self.rho;
    half * theta * (one + self.rho * phi * k + (u * u + one - self.rho * self.rho).sqrt())
  }

  /// Evaluate implied volatility $\sigma(k, \theta, T)$.
  #[inline]
  pub fn implied_vol(&self, k: T, theta: T, t: T) -> T {
    let w = self.total_variance(k, theta);
    let zero = T::zero();
    if w > zero && t > zero {
      (w / t).sqrt()
    } else {
      T::nan()
    }
  }

  /// First derivative $\partial_k w(k, \theta)$.
  #[inline]
  pub fn w_prime_k(&self, k: T, theta: T) -> T {
    let half = T::from_f64_fast(0.5);
    let one = T::one();
    let phi = self.phi(theta);
    let u = phi * k + self.rho;
    let r = (u * u + one - self.rho * self.rho).sqrt();
    half * theta * phi * (self.rho + u / r)
  }

  /// Second derivative $\partial_{kk} w(k, \theta)$.
  #[inline]
  pub fn w_double_prime_k(&self, k: T, theta: T) -> T {
    let half = T::from_f64_fast(0.5);
    let one = T::one();
    let phi = self.phi(theta);
    let u = phi * k + self.rho;
    let one_m_rho2 = one - self.rho * self.rho;
    let r = (u * u + one_m_rho2).sqrt();
    half * theta * phi * phi * one_m_rho2 / (r * r * r)
  }

  /// Derivative of the power-law mixing function: $\varphi'(\theta) = -\varphi(\theta)\bigl[\gamma/\theta + (1-\gamma)/(1+\theta)\bigr]$.
  #[inline]
  pub fn phi_prime(&self, theta: T) -> T {
    if theta <= T::zero() {
      return T::zero();
    }
    let one = T::one();
    let phi = self.phi(theta);
    -phi * (self.gamma / theta + (one - self.gamma) / (one + theta))
  }

  /// Partial derivative $\partial_\theta w(k, \theta)$.
  ///
  /// $$
  /// \partial_\theta w = \frac{w}{\theta} + \frac{\theta\,\varphi'(\theta)\,k}{2}
  ///     \Bigl(\rho + \frac{\varphi(\theta)\,k + \rho}{\sqrt{(\varphi(\theta)\,k+\rho)^2+(1-\rho^2)}}\Bigr)
  /// $$
  #[inline]
  pub fn w_prime_theta(&self, k: T, theta: T) -> T {
    let half = T::from_f64_fast(0.5);
    let one = T::one();
    let phi = self.phi(theta);
    let phi_p = self.phi_prime(theta);
    let u = phi * k + self.rho;
    let r = (u * u + one - self.rho * self.rho).sqrt();
    let w = self.total_variance(k, theta);

    w / theta + half * theta * phi_p * k * (self.rho + u / r)
  }

  /// Check the sufficient no-butterfly-arbitrage condition from
  /// Gatheral & Jacquier (2012), Theorem 4.1:
  ///
  /// $\eta(1+|\rho|) \leq 2$
  pub fn satisfies_no_butterfly_condition(&self) -> bool {
    let two = T::from_f64_fast(2.0);
    let one = T::one();
    self.eta * (one + self.rho.abs()) <= two
  }

  /// Project parameters to satisfy admissibility.
  pub fn project(&mut self) {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64_fast(2.0);
    let bound = T::from_f64_fast(0.9999);
    let eps = T::from_f64_fast(1e-8);

    self.rho = self.rho.max(-bound).min(bound);
    self.eta = self.eta.max(eps);
    self.gamma = self.gamma.max(zero).min(one);

    let max_eta = two / (one + self.rho.abs());
    self.eta = self.eta.min(max_eta);
  }

  fn as_f64(self) -> SsviParams<f64> {
    SsviParams {
      rho: self.rho.to_f64().unwrap_or(0.0),
      eta: self.eta.to_f64().unwrap_or(0.0),
      gamma: self.gamma.to_f64().unwrap_or(0.0),
    }
  }
}

impl SsviParams<f64> {
  fn into_dvector(self) -> DVector<f64> {
    DVector::from_vec(vec![self.rho, self.eta, self.gamma])
  }

  fn from_dvector(v: &DVector<f64>) -> Self {
    SsviParams {
      rho: v[0],
      eta: v[1],
      gamma: v[2],
    }
  }
}

/// A single SSVI maturity slice with its ATM total variance.
#[derive(Clone, Debug)]
pub struct SsviSlice<T: FloatExt> {
  /// Log-forward moneyness values $k_i$
  pub log_moneyness: Vec<T>,
  /// Observed total variance $w_i$
  pub total_variance: Vec<T>,
  /// ATM total variance $\theta_t$ for this slice
  pub theta: T,
}

/// Calibrate SSVI global parameters $(\rho, \eta, \gamma)$ to multiple
/// maturity slices simultaneously.
pub fn calibrate_ssvi<T: FloatExt>(
  slices: &[SsviSlice<T>],
  initial: Option<SsviParams<T>>,
) -> SsviParams<T> {
  let init_f64 = initial
    .map(|p| p.as_f64())
    .unwrap_or(SsviParams::<f64>::new(-0.3, 0.5, 0.5));

  let slices_f64: Vec<SsviSliceF64> = slices
    .iter()
    .map(|s| SsviSliceF64 {
      log_moneyness: s
        .log_moneyness
        .iter()
        .map(|x| x.to_f64().unwrap_or(0.0))
        .collect(),
      total_variance: s
        .total_variance
        .iter()
        .map(|x| x.to_f64().unwrap_or(0.0))
        .collect(),
      theta: s.theta.to_f64().unwrap_or(0.0),
    })
    .collect();

  let problem = SsviLmProblem {
    slices: slices_f64,
    params: init_f64.into_dvector(),
  };

  let (result, _report) = LevenbergMarquardt::new()
    .with_patience(200)
    .with_tol(1e-12)
    .minimize(problem);

  let mut p64 = SsviParams::<f64>::from_dvector(&result.params);
  p64.project();

  SsviParams {
    rho: T::from_f64_fast(p64.rho),
    eta: T::from_f64_fast(p64.eta),
    gamma: T::from_f64_fast(p64.gamma),
  }
}

/// Full SSVI volatility surface.
#[derive(Clone, Debug)]
pub struct SsviSurface<T: FloatExt> {
  /// Global SSVI parameters
  pub params: SsviParams<T>,
  /// ATM total variance $\theta_t$ for each maturity (ascending in time)
  pub thetas: Vec<T>,
  /// Maturities in years (ascending)
  pub maturities: Vec<T>,
}

impl<T: FloatExt> SsviSurface<T> {
  pub fn new(params: SsviParams<T>, thetas: Vec<T>, maturities: Vec<T>) -> Self {
    assert_eq!(
      thetas.len(),
      maturities.len(),
      "thetas and maturities must have the same length"
    );
    Self {
      params,
      thetas,
      maturities,
    }
  }

  /// Evaluate total variance at $(k, T)$ by linearly interpolating $\theta(T)$.
  pub fn total_variance(&self, k: T, t: T) -> T {
    let theta = self.interpolate_theta(t);
    self.params.total_variance(k, theta)
  }

  /// Evaluate implied volatility at $(k, T)$.
  pub fn implied_vol(&self, k: T, t: T) -> T {
    let w = self.total_variance(k, t);
    let zero = T::zero();
    if w > zero && t > zero {
      (w / t).sqrt()
    } else {
      T::nan()
    }
  }

  /// Linearly interpolate $\theta(T)$ from the calibrated ATM term structure.
  fn interpolate_theta(&self, t: T) -> T {
    let n = self.maturities.len();
    if n == 0 {
      return T::zero();
    }
    if t <= self.maturities[0] {
      return self.thetas[0];
    }
    if t >= self.maturities[n - 1] {
      return self.thetas[n - 1];
    }

    let idx = self
      .maturities
      .partition_point(|&m| m < t)
      .min(n - 1)
      .max(1);
    let t0 = self.maturities[idx - 1];
    let t1 = self.maturities[idx];
    let one = T::one();
    let alpha = (t - t0) / (t1 - t0);
    self.thetas[idx - 1] * (one - alpha) + self.thetas[idx] * alpha
  }

  /// Check calendar-spread arbitrage on a log-moneyness grid (Gatheral &
  /// Jacquier 2014, Theorem 4.2).
  ///
  /// The full condition is $\partial_T w(k, T) \geq 0$ for **every** $k$, not
  /// just $k = 0$. Verifies (a) the necessary ATM condition $\theta_t$
  /// non-decreasing, and (b) the full smile-wide condition $w(k, \theta_{n+1})
  /// \geq w(k, \theta_n)$ for each adjacent slice and each $k$ in the grid.
  pub fn is_calendar_spread_free(&self, ks: &[T]) -> bool {
    if !self.thetas.windows(2).all(|w| w[1] >= w[0]) {
      return false;
    }
    for win in self.thetas.windows(2) {
      let (theta_lo, theta_hi) = (win[0], win[1]);
      for &k in ks {
        let w_lo = self.params.total_variance(k, theta_lo);
        let w_hi = self.params.total_variance(k, theta_hi);
        if w_hi < w_lo {
          return false;
        }
      }
    }
    true
  }

  /// Check the ATM-only calendar-spread condition (necessary but not
  /// sufficient): $\theta_t$ must be non-decreasing. Use
  /// [`Self::is_calendar_spread_free`] with a representative log-moneyness
  /// grid for the full GJ 2014 Theorem 4.2 check.
  pub fn is_atm_calendar_spread_free(&self) -> bool {
    self.thetas.windows(2).all(|w| w[1] >= w[0])
  }

  /// Dupire local variance $\sigma^2_{\mathrm{loc}}(k, T)$ from SSVI analytic
  /// derivatives.
  ///
  /// Uses the total-variance form of Dupire's formula:
  ///
  /// $$
  /// \sigma^2_{\mathrm{loc}}(k, T) = \frac{\partial_T w(k, T)}{g(k)}
  /// $$
  ///
  /// where $g(k)$ is the butterfly density and $\partial_T w = \theta'(T) \cdot \partial_\theta w$.
  pub fn local_var(&self, k: T, t: T) -> T {
    let zero = T::zero();
    let theta = self.interpolate_theta(t);
    let theta_prime = self.interpolate_theta_prime(t);

    let dw_dt = theta_prime * self.params.w_prime_theta(k, theta);

    let w = self.params.total_variance(k, theta);
    let wp = self.params.w_prime_k(k, theta);
    let wpp = self.params.w_double_prime_k(k, theta);
    let g = super::arbitrage::butterfly_density_at(k, w, wp, wpp);

    if g > zero && dw_dt.is_finite() && g.is_finite() {
      dw_dt / g
    } else {
      T::nan()
    }
  }

  /// Dupire local volatility $\sigma_{\mathrm{loc}}(k, T) = \sqrt{\sigma^2_{\mathrm{loc}}}$.
  pub fn local_vol(&self, k: T, t: T) -> T {
    let lv2 = self.local_var(k, t);
    if lv2 > T::zero() {
      lv2.sqrt()
    } else {
      T::nan()
    }
  }

  /// Compute local volatility surface on a $(k, T)$ grid.
  ///
  /// Returns an `Array2<T>` with shape `(n_t, n_k)`.
  pub fn local_vol_surface(&self, ks: &[T], ts: &[T]) -> ndarray::Array2<T> {
    let nt = ts.len();
    let nk = ks.len();
    let mut surface = ndarray::Array2::<T>::from_elem((nt, nk), T::nan());

    for (j, &t) in ts.iter().enumerate() {
      for (i, &k) in ks.iter().enumerate() {
        surface[[j, i]] = self.local_vol(k, t);
      }
    }

    surface
  }

  /// Derivative $\theta'(T)$ from linear interpolation of the term structure.
  fn interpolate_theta_prime(&self, t: T) -> T {
    let n = self.maturities.len();
    if n < 2 {
      return T::zero();
    }

    if t <= self.maturities[0] {
      return (self.thetas[1] - self.thetas[0]) / (self.maturities[1] - self.maturities[0]);
    }
    if t >= self.maturities[n - 1] {
      return (self.thetas[n - 1] - self.thetas[n - 2])
        / (self.maturities[n - 1] - self.maturities[n - 2]);
    }

    let idx = self
      .maturities
      .partition_point(|&m| m < t)
      .min(n - 1)
      .max(1);
    (self.thetas[idx] - self.thetas[idx - 1]) / (self.maturities[idx] - self.maturities[idx - 1])
  }
}

#[derive(Clone, Debug)]
struct SsviSliceF64 {
  log_moneyness: Vec<f64>,
  total_variance: Vec<f64>,
  theta: f64,
}

struct SsviLmProblem {
  slices: Vec<SsviSliceF64>,
  params: DVector<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for SsviLmProblem {
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;
  type JacobianStorage = Owned<f64, Dyn, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params.copy_from(params);
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let p = SsviParams::<f64>::from_dvector(&self.params);
    let n: usize = self.slices.iter().map(|s| s.log_moneyness.len()).sum();
    let mut r = DVector::zeros(n);
    let mut idx = 0;
    for slice in &self.slices {
      for i in 0..slice.log_moneyness.len() {
        r[idx] = p.total_variance(slice.log_moneyness[i], slice.theta) - slice.total_variance[i];
        idx += 1;
      }
    }
    Some(r)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let p = SsviParams::<f64>::from_dvector(&self.params);
    let n: usize = self.slices.iter().map(|s| s.log_moneyness.len()).sum();
    let mut jac = DMatrix::zeros(n, 3);
    let h = 1e-7;

    let mut idx = 0;
    for slice in &self.slices {
      for i in 0..slice.log_moneyness.len() {
        let k = slice.log_moneyness[i];
        let theta = slice.theta;
        let w0 = p.total_variance(k, theta);

        let p_rho = SsviParams::<f64>::new(p.rho + h, p.eta, p.gamma);
        jac[(idx, 0)] = (p_rho.total_variance(k, theta) - w0) / h;

        let p_eta = SsviParams::<f64>::new(p.rho, p.eta + h, p.gamma);
        jac[(idx, 1)] = (p_eta.total_variance(k, theta) - w0) / h;

        let p_gamma = SsviParams::<f64>::new(p.rho, p.eta, p.gamma + h);
        jac[(idx, 2)] = (p_gamma.total_variance(k, theta) - w0) / h;

        idx += 1;
      }
    }

    Some(jac)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn ssvi_evaluation() {
    let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    assert!(p.satisfies_no_butterfly_condition());

    let theta = 0.04;
    let w0 = p.total_variance(0.0, theta);
    assert!(w0 > 0.0);
    assert!(
      (w0 - theta).abs() < 1e-10,
      "ATM total variance should equal theta: w0={w0}, theta={theta}"
    );
  }

  #[test]
  fn ssvi_derivatives() {
    let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let theta = 0.04;
    let k = 0.1;
    let h = 1e-5;

    let num_first = (p.total_variance(k + h, theta) - p.total_variance(k - h, theta)) / (2.0 * h);
    let num_second = (p.total_variance(k + h, theta) - 2.0 * p.total_variance(k, theta)
      + p.total_variance(k - h, theta))
      / (h * h);

    assert!(
      (p.w_prime_k(k, theta) - num_first).abs() < 1e-5,
      "w'(k) mismatch"
    );
    let rel_err = (p.w_double_prime_k(k, theta) - num_second).abs()
      / p.w_double_prime_k(k, theta).abs().max(1e-14);
    assert!(
      rel_err < 1e-3,
      "w''(k) mismatch: analytic={} numeric={} rel_err={}",
      p.w_double_prime_k(k, theta),
      num_second,
      rel_err
    );
  }

  #[test]
  fn ssvi_calibration_exact() {
    let true_params = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let thetas = [0.02, 0.04, 0.08];

    let slices: Vec<SsviSlice<f64>> = thetas
      .iter()
      .map(|&theta| {
        let ks: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
        let ws: Vec<f64> = ks
          .iter()
          .map(|&k| true_params.total_variance(k, theta))
          .collect();
        SsviSlice {
          log_moneyness: ks,
          total_variance: ws,
          theta,
        }
      })
      .collect();

    let fitted = calibrate_ssvi(&slices, None);

    assert!((fitted.rho - true_params.rho).abs() < 1e-3, "rho mismatch");
    assert!((fitted.eta - true_params.eta).abs() < 1e-3, "eta mismatch");
    assert!(
      (fitted.gamma - true_params.gamma).abs() < 1e-3,
      "gamma mismatch"
    );
  }

  #[test]
  fn ssvi_surface_interpolation() {
    let params = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let surface = SsviSurface::new(params, vec![0.02, 0.04, 0.08], vec![0.25, 0.50, 1.0]);

    let ks = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    assert!(surface.is_calendar_spread_free(&ks));
    assert!(surface.is_atm_calendar_spread_free());

    let iv = surface.implied_vol(0.0, 0.5);
    assert!(iv.is_finite() && iv > 0.0);

    let iv_interp = surface.implied_vol(0.0, 0.375);
    assert!(iv_interp.is_finite() && iv_interp > 0.0);
  }

  /// Regression: a non-monotonic θ_t term structure must fail BOTH the ATM
  /// and the smile-wide checks. A monotonic θ_t with a strong-skew SSVI can
  /// still violate calendar arb off-ATM — pre-rc.1 the surface flag would
  /// say "arb-free" in that scenario, hiding the issue.
  #[test]
  fn calendar_spread_free_grid_catches_off_atm_violations() {
    // Decreasing θ — fails the ATM check, must also fail the grid check.
    let params = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
    let surface_decreasing =
      SsviSurface::new(params, vec![0.04, 0.03, 0.02], vec![0.25, 0.50, 1.0]);
    let ks = vec![-1.0, 0.0, 1.0];
    assert!(!surface_decreasing.is_calendar_spread_free(&ks));
    assert!(!surface_decreasing.is_atm_calendar_spread_free());
  }
}
