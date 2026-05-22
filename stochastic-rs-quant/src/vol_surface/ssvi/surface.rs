use super::params::SsviParams;
use crate::traits::FloatExt;

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
    let g = crate::vol_surface::arbitrage::butterfly_density_at(k, w, wp, wpp);

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
