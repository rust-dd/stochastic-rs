//! # Double Heston
//!
//! Two-factor stochastic volatility model with independent Cox-Ingersoll-Ross
//! variance factors.
//!
//! $$
//! \begin{aligned}
//! dS_t &= \mu S_t\,dt + \sqrt{v_{1,t}}\,S_t\,dW_{1,t}^S + \sqrt{v_{2,t}}\,S_t\,dW_{2,t}^S \\
//! dv_{1,t} &= \kappa_1(\theta_1 - v_{1,t})\,dt + \sigma_1\sqrt{v_{1,t}}\,dW_{1,t}^v \\
//! dv_{2,t} &= \kappa_2(\theta_2 - v_{2,t})\,dt + \sigma_2\sqrt{v_{2,t}}\,dW_{2,t}^v
//! \end{aligned}
//! $$
//! with $d\langle W_1^S,W_1^v\rangle_t=\rho_1\,dt$,
//! $d\langle W_2^S,W_2^v\rangle_t=\rho_2\,dt$, and every other Brownian
//! motion pair independent.
//!
//! Source:
//! - Christoffersen, Heston & Jacobs (2009), "The Shape and Term Structure of
//!   the Index Option Smirk", <https://doi.org/10.1287/mnsc.1090.1065>
//! - Mehrdoust, Noorani & Hamdi (2021), "Calibration of the double Heston
//!   model and an analytical formula in pricing American put option",
//!   <https://doi.org/10.1016/j.cam.2021.113422>

use ndarray::Array1;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use crate::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Double Heston stochastic volatility process.
///
/// The two variance factors are assumed independent of each other; only
/// within a factor is there a correlation between the stock shock and the
/// variance shock ($\rho_1$ and $\rho_2$).
pub struct DoubleHeston<T: FloatExt, S: SeedExt = Unseeded> {
  /// Initial stock price.
  pub s0: Option<T>,
  /// Initial variance of factor 1.
  pub v1_0: Option<T>,
  /// Initial variance of factor 2.
  pub v2_0: Option<T>,
  /// Mean-reversion speed of factor 1.
  pub kappa1: T,
  /// Long-run variance of factor 1.
  pub theta1: T,
  /// Volatility-of-variance of factor 1.
  pub sigma1: T,
  /// Spot-variance correlation for factor 1.
  pub rho1: T,
  /// Mean-reversion speed of factor 2.
  pub kappa2: T,
  /// Long-run variance of factor 2.
  pub theta2: T,
  /// Volatility-of-variance of factor 2.
  pub sigma2: T,
  /// Spot-variance correlation for factor 2.
  pub rho2: T,
  /// Drift of the stock price (risk-neutral drift = r - q).
  pub mu: T,
  /// Number of time steps.
  pub n: usize,
  /// Time to maturity.
  pub t: Option<T>,
  /// Use the reflection method for the variance to avoid negative values.
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  /// First factor's correlated Gaussian noise source: $(W_1^S, W_1^v)$.
  cgns1: CGNS<T>,
  /// Second factor's correlated Gaussian noise source: $(W_2^S, W_2^v)$.
  cgns2: CGNS<T>,
}

impl<T: FloatExt> DoubleHeston<T> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    s0: Option<T>,
    v1_0: Option<T>,
    v2_0: Option<T>,
    kappa1: T,
    theta1: T,
    sigma1: T,
    rho1: T,
    kappa2: T,
    theta2: T,
    sigma2: T,
    rho2: T,
    mu: T,
    n: usize,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    assert!(kappa1 >= T::zero(), "kappa1 must be non-negative");
    assert!(theta1 >= T::zero(), "theta1 must be non-negative");
    assert!(sigma1 >= T::zero(), "sigma1 must be non-negative");
    assert!(kappa2 >= T::zero(), "kappa2 must be non-negative");
    assert!(theta2 >= T::zero(), "theta2 must be non-negative");
    assert!(sigma2 >= T::zero(), "sigma2 must be non-negative");
    if let Some(v) = v1_0 {
      assert!(v >= T::zero(), "v1_0 must be non-negative");
    }
    if let Some(v) = v2_0 {
      assert!(v >= T::zero(), "v2_0 must be non-negative");
    }

    Self {
      s0,
      v1_0,
      v2_0,
      kappa1,
      theta1,
      sigma1,
      rho1,
      kappa2,
      theta2,
      sigma2,
      rho2,
      mu,
      n,
      t,
      use_sym,
      seed: Unseeded,
      cgns1: CGNS::new(rho1, n - 1, t),
      cgns2: CGNS::new(rho2, n - 1, t),
    }
  }
}

impl<T: FloatExt> DoubleHeston<T, Deterministic> {
  #[allow(clippy::too_many_arguments)]
  pub fn seeded(
    s0: Option<T>,
    v1_0: Option<T>,
    v2_0: Option<T>,
    kappa1: T,
    theta1: T,
    sigma1: T,
    rho1: T,
    kappa2: T,
    theta2: T,
    sigma2: T,
    rho2: T,
    mu: T,
    n: usize,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: u64,
  ) -> Self {
    assert!(kappa1 >= T::zero(), "kappa1 must be non-negative");
    assert!(theta1 >= T::zero(), "theta1 must be non-negative");
    assert!(sigma1 >= T::zero(), "sigma1 must be non-negative");
    assert!(kappa2 >= T::zero(), "kappa2 must be non-negative");
    assert!(theta2 >= T::zero(), "theta2 must be non-negative");
    assert!(sigma2 >= T::zero(), "sigma2 must be non-negative");
    if let Some(v) = v1_0 {
      assert!(v >= T::zero(), "v1_0 must be non-negative");
    }
    if let Some(v) = v2_0 {
      assert!(v >= T::zero(), "v2_0 must be non-negative");
    }

    Self {
      s0,
      v1_0,
      v2_0,
      kappa1,
      theta1,
      sigma1,
      rho1,
      kappa2,
      theta2,
      sigma2,
      rho2,
      mu,
      n,
      t,
      use_sym,
      seed: Deterministic(seed),
      cgns1: CGNS::new(rho1, n - 1, t),
      cgns2: CGNS::new(rho2, n - 1, t),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for DoubleHeston<T, S> {
  /// Output tuple: `[S, v1, v2]`.
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns1.dt();
    let mut seed = self.seed;
    let [ds1, dv1n] = &self.cgns1.sample_impl(seed.derive());
    let [ds2, dv2n] = &self.cgns2.sample_impl(seed.derive());

    let mut s = Array1::<T>::zeros(self.n);
    let mut v1 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::zero());
    v1[0] = self.v1_0.unwrap_or(T::zero()).max(T::zero());
    v2[0] = self.v2_0.unwrap_or(T::zero()).max(T::zero());

    let use_sym = self.use_sym.unwrap_or(false);

    for i in 1..self.n {
      let v1_prev = v1[i - 1].max(T::zero());
      let v2_prev = v2[i - 1].max(T::zero());

      // Stock increment receives two independent (across factors) variance shocks.
      let ds = self.mu * s[i - 1] * dt
        + s[i - 1] * v1_prev.sqrt() * ds1[i - 1]
        + s[i - 1] * v2_prev.sqrt() * ds2[i - 1];
      s[i] = s[i - 1] + ds;

      let dv1 =
        self.kappa1 * (self.theta1 - v1_prev) * dt + self.sigma1 * v1_prev.sqrt() * dv1n[i - 1];
      let dv2 =
        self.kappa2 * (self.theta2 - v2_prev) * dt + self.sigma2 * v2_prev.sqrt() * dv2n[i - 1];

      let new_v1 = v1[i - 1] + dv1;
      let new_v2 = v2[i - 1] + dv2;

      v1[i] = if use_sym {
        new_v1.abs()
      } else {
        new_v1.max(T::zero())
      };
      v2[i] = if use_sym {
        new_v2.abs()
      } else {
        new_v2.max(T::zero())
      };
    }

    [s, v1, v2]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[should_panic(expected = "v1_0 must be non-negative")]
  fn negative_initial_variance_panics() {
    let _ = DoubleHeston::new(
      Some(100.0_f64),
      Some(-0.01),
      Some(0.02),
      1.0,
      0.04,
      0.3,
      -0.5,
      0.5,
      0.04,
      0.2,
      -0.3,
      0.0,
      8,
      Some(1.0),
      Some(false),
    );
  }

  #[test]
  fn variance_paths_stay_non_negative() {
    let p = DoubleHeston::new(
      Some(100.0_f64),
      Some(0.02),
      Some(0.02),
      3.0,
      0.02,
      0.4,
      -0.6,
      0.5,
      0.02,
      0.2,
      -0.3,
      0.05,
      128,
      Some(1.0),
      Some(true),
    );
    let [_s, v1, v2] = p.sample();
    assert!(v1.iter().all(|x| *x >= 0.0));
    assert!(v2.iter().all(|x| *x >= 0.0));
  }

  #[test]
  fn stock_path_is_finite() {
    let p = DoubleHeston::seeded(
      Some(100.0_f64),
      Some(0.02),
      Some(0.02),
      3.0,
      0.02,
      0.4,
      -0.6,
      0.5,
      0.02,
      0.2,
      -0.3,
      0.05,
      64,
      Some(0.5),
      Some(true),
      42,
    );
    let [s, _, _] = p.sample();
    assert!(s.iter().all(|x| x.is_finite()));
  }
}
