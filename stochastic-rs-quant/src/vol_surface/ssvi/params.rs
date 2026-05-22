use nalgebra::DVector;

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
  ///
  /// Enforces the **butterfly bound** $\eta (1 + |\rho|) \leq 2$ (Gatheral &
  /// Jacquier 2012, Theorem 4.1). This is necessary but **not sufficient**
  /// for full GJ 2014 Theorem 4.2 calendar-spread-free admissibility, which
  /// requires $\partial_T w(k, T) \geq 0$ for every $k$ — the cross-slice
  /// constraint depends on the realised $\theta_t$ schedule and cannot be
  /// expressed in $(\rho, \eta, \gamma)$ alone. Use
  /// [`Self::project_with_theta_range`] when a $\theta$ range is known.
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

  /// Project parameters to satisfy admissibility **plus** the GJ 2014 power-law
  /// SSVI calendar-spread-free condition over a known $\theta$ range.
  ///
  /// In addition to butterfly admissibility, enforces the sufficient condition
  /// (Gatheral & Jacquier 2014, equation 4.4):
  ///
  /// $$
  /// \frac{1}{4} \, \eta\, \theta_{max}^{-\gamma} \, (1 + |\rho|) \leq 1
  /// $$
  ///
  /// which guarantees $\partial_\theta w(k, \theta) \geq 0$ for all $k$ and
  /// all $\theta \in [\theta_{min}, \theta_{max}]$. The constraint is dominated
  /// by the largest $\theta$ in the surface.
  ///
  /// `theta_max` is the largest ATM total variance across the surface. Pass
  /// the maximum of `SsviSurface::thetas` (or a conservative upper bound).
  pub fn project_with_theta_range(&mut self, theta_max: T) {
    self.project();
    if theta_max <= T::zero() || !theta_max.is_finite() {
      return;
    }
    let one = T::one();
    let four = T::from_f64_fast(4.0);
    let calendar_bound =
      four * <T as num_traits::Float>::powf(theta_max, self.gamma) / (one + self.rho.abs());
    if self.eta > calendar_bound {
      self.eta = calendar_bound;
    }
  }

  pub(super) fn as_f64(self) -> SsviParams<f64> {
    SsviParams {
      rho: self.rho.to_f64().unwrap_or(0.0),
      eta: self.eta.to_f64().unwrap_or(0.0),
      gamma: self.gamma.to_f64().unwrap_or(0.0),
    }
  }
}

impl SsviParams<f64> {
  pub(super) fn into_dvector(self) -> DVector<f64> {
    DVector::from_vec(vec![self.rho, self.eta, self.gamma])
  }

  pub(super) fn from_dvector(v: &DVector<f64>) -> Self {
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
