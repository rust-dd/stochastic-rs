//! # Rough Heston
//!
//! $$
//! \begin{aligned}
//! dS_t &= \mu S_t\,dt + \sqrt{V_t}\,S_t\,dW^s_t,\qquad d\langle W^s, W^v\rangle_t = \rho\,dt \\[3pt]
//! V_t &= V_0 + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} \kappa(\theta - V_s)\,ds
//!           + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} \nu\sqrt{V_s}\,dW^v_s
//! \end{aligned}
//! $$
//!
//! The variance process is a Volterra-CIR with Riemann–Liouville kernel
//! (El Euch–Rosenbaum 2019). It is simulated by the Bilokon–Wong modified
//! fast algorithm — a single Markov-lift with $f(V)=\kappa(\theta-V)$ and
//! $g(V)=\nu\sqrt{V^+}$ — while the asset price is integrated with a
//! standard Euler step driven by $W^s$ correlated to $W^v$.
//!
//! Reference: Bilokon & Wong (2026) §5.5; El Euch O., Rosenbaum M. *The
//! characteristic function of rough Heston models*, Math. Finance 29 (2019),
//! 3–38.
use ndarray::Array1;

use super::kernel::RlKernel;
use super::markov_lift::MarkovLift;
use super::markov_lift::RoughSimd;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Rough Heston model with Volterra-CIR variance and correlated GBM asset.
#[derive(Clone)]
pub struct RlHeston<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent $H \in (0, 1/2)$ of the variance kernel.
  pub hurst: T,
  /// Initial spot $S_0$.
  pub s0: Option<T>,
  /// Initial variance $V_0$.
  pub v0: Option<T>,
  /// Variance mean-reversion speed $\kappa$.
  pub kappa: T,
  /// Long-run variance $\theta$.
  pub theta: T,
  /// Volatility of variance $\nu$.
  pub sigma: T,
  /// Correlation between $W^s$ and $W^v$.
  pub rho: T,
  /// Asset drift $\mu$ (risk-free rate under $\mathbb{Q}$).
  pub mu: T,
  /// Number of simulation points.
  pub n: usize,
  /// Simulation horizon.
  pub t: Option<T>,
  /// Quadrature degree passed through to the kernel.
  pub degree: Option<usize>,
  /// Seed strategy.
  pub seed: S,
  cgns: CGNS<T>,
}

impl<T: FloatExt> RlHeston<T> {
  #[must_use]
  pub fn new(
    hurst: T,
    s0: Option<T>,
    v0: Option<T>,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    mu: T,
    n: usize,
    t: Option<T>,
    degree: Option<usize>,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(kappa >= T::zero(), "kappa must be non-negative");
    assert!(theta >= T::zero(), "theta must be non-negative");
    assert!(sigma >= T::zero(), "sigma must be non-negative");
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }
    Self {
      hurst,
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      t,
      degree,
      seed: Unseeded,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> RlHeston<T, Deterministic> {
  #[must_use]
  pub fn seeded(
    hurst: T,
    s0: Option<T>,
    v0: Option<T>,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    mu: T,
    n: usize,
    t: Option<T>,
    degree: Option<usize>,
    seed: u64,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    Self {
      hurst,
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      t,
      degree,
      seed: Deterministic(seed),
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> ProcessExt<T> for RlHeston<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let mut seed = self.seed;
    let [dw_s, dw_v] = self.cgns.sample_impl(seed.derive());

    let degree = self.degree.unwrap_or_else(|| RlKernel::<T>::default_degree(self.n));
    let kernel = RlKernel::<T>::new(self.hurst, degree);
    let step = MarkovLift::new(kernel, dt);

    let kappa = self.kappa;
    let theta = self.theta;
    let sigma = self.sigma;
    let v0 = self.v0.unwrap_or(T::zero()).max(T::zero());
    let v = step.simulate(
      v0,
      |vv| kappa * (theta - vv.max(T::zero())),
      |vv| sigma * vv.max(T::zero()).sqrt(),
      dw_v.as_slice().expect("dw_v must be contiguous"),
    );

    let mut s = Array1::<T>::zeros(self.n);
    s[0] = self.s0.unwrap_or(T::zero());
    for i in 1..self.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v_prev.sqrt() * dw_s[i - 1];
    }

    let v_clipped = v.map(|x| (*x).max(T::zero()));
    [s, v_clipped]
  }
}

#[cfg(test)]
mod tests {
  use super::RlHeston;
  use crate::traits::ProcessExt;

  #[test]
  #[should_panic(expected = "n must be at least 2")]
  fn rejects_too_short_grid() {
    let _ = RlHeston::<f64>::new(0.3, Some(100.0), Some(0.04), 1.0, 0.04, 0.5, -0.5, 0.0, 1, Some(1.0), None);
  }

  #[test]
  fn variance_stays_non_negative() {
    let p = RlHeston::seeded(
      0.12_f64,
      Some(100.0),
      Some(0.04),
      0.1,
      0.3156,
      0.0331,
      -0.681,
      0.0,
      256,
      Some(1.0),
      None,
      11,
    );
    let [s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
    assert!(s.iter().all(|x| x.is_finite()));
  }

  #[test]
  fn heston_limit_h_half_asymptotic_variance_is_finite() {
    // We can't use H = 1/2 directly (kernel requires H < 1/2), but at H close
    // to 1/2 the rough Heston variance should behave like classical Heston:
    // positive, bounded, mean-reverting around theta.
    let p = RlHeston::seeded(
      0.49_f64,
      Some(100.0),
      Some(0.04),
      1.0,
      0.04,
      0.3,
      -0.5,
      0.0,
      512,
      Some(1.0),
      None,
      3,
    );
    let [_s, v] = p.sample();
    let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
    assert!(mean > 0.0 && mean < 1.0, "mean variance out of sanity range: {mean}");
  }
}
