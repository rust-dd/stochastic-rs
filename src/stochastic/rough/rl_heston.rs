//! # Rough Heston
//!
//! $$
//! \begin{aligned} dS_t &= \mu S_t\,dt + \sqrt{V_t}\,S_t\,dW^s_t,\qquad d\langle W^s, W^v\rangle_t = \rho\,dt \\ V_t &= V_0 + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} \kappa(\theta - V_s)\,ds + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} \nu\sqrt{V_s}\,dW^v_s \end{aligned}
//! $$
//!
//! The variance process is a Volterra-CIR with Riemann–Liouville kernel
//! (El Euch–Rosenbaum 2019), simulated via the Bilokon–Wong modified fast
//! algorithm — a single Markov-lift with $f(V)=\kappa(\theta-V)$ and
//! $g(V)=\nu\sqrt{V^+}$ — while the asset price is integrated with a
//! standard Euler step driven by $W^s$ correlated to $W^v$.
//!
//! The underlying kernel + Markov-lift is built once at struct construction
//! and reused across every [`sample`] / [`sample_batch`] call.
//!
//! Reference: Bilokon & Wong (2026) §5.5; El Euch O., Rosenbaum M. *The
//! characteristic function of rough Heston models*, Math. Finance 29 (2019),
//! 3–38.
use ndarray::Array1;
use ndarray::Array2;

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
  markov: MarkovLift<T>,
}

fn build_markov<T: FloatExt>(
  hurst: T,
  n: usize,
  t: Option<T>,
  degree: Option<usize>,
) -> MarkovLift<T> {
  let dt = t.unwrap_or(T::one()) / T::from_usize_(n - 1);
  let deg = degree.unwrap_or_else(|| RlKernel::<T>::default_degree(n));
  let kernel = RlKernel::<T>::new(hurst, deg);
  MarkovLift::new(kernel, dt)
}

impl<T: FloatExt> RlHeston<T> {
  #[must_use]
  #[allow(clippy::too_many_arguments)]
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
      markov: build_markov(hurst, n, t, degree),
    }
  }
}

impl<T: FloatExt> RlHeston<T, Deterministic> {
  #[must_use]
  #[allow(clippy::too_many_arguments)]
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
      markov: build_markov(hurst, n, t, degree),
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlHeston<T, S> {
  /// Generate $m$ independent rough Heston paths.
  /// Returns `[spot_paths, variance_paths]` where each is an $(m, n)$ array.
  pub fn sample_batch(&self, m: usize) -> [Array2<T>; 2] {
    let mut seed = self.seed;
    let n_minus_1 = self.n - 1;

    let mut dw_s = Array2::<T>::zeros((m, n_minus_1));
    let mut dw_v = Array2::<T>::zeros((m, n_minus_1));
    for p in 0..m {
      let [s_row, v_row] = self.cgns.sample_impl(seed.derive());
      dw_s.row_mut(p).assign(&s_row);
      dw_v.row_mut(p).assign(&v_row);
    }

    let kappa = self.kappa;
    let theta = self.theta;
    let sigma = self.sigma;
    let v0 = self.v0.unwrap_or(T::zero()).max(T::zero());
    let variance = self.markov.simulate_batch(
      v0,
      |vv| kappa * (theta - vv.max(T::zero())),
      |vv| sigma * vv.max(T::zero()).sqrt(),
      dw_v.view(),
    );

    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let s0 = self.s0.unwrap_or(T::zero());
    let mut spots = Array2::<T>::zeros((m, self.n));
    for p in 0..m {
      spots[[p, 0]] = s0;
      for i in 1..self.n {
        let v_prev = variance[[p, i - 1]].max(T::zero());
        spots[[p, i]] = spots[[p, i - 1]]
          + self.mu * spots[[p, i - 1]] * dt
          + spots[[p, i - 1]] * v_prev.sqrt() * dw_s[[p, i - 1]];
      }
    }

    let variance_clipped = variance.map(|v| (*v).max(T::zero()));
    [spots, variance_clipped]
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> ProcessExt<T> for RlHeston<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let mut seed = self.seed;
    let [dw_s, dw_v] = self.cgns.sample_impl(seed.derive());

    let kappa = self.kappa;
    let theta = self.theta;
    let sigma = self.sigma;
    let v0 = self.v0.unwrap_or(T::zero()).max(T::zero());
    let v = self.markov.simulate(
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
    let _ = RlHeston::<f64>::new(
      0.3,
      Some(100.0),
      Some(0.04),
      1.0,
      0.04,
      0.5,
      -0.5,
      0.0,
      1,
      Some(1.0),
      None,
    );
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
  fn batch_shape_and_nonneg_variance() {
    let p = RlHeston::seeded(
      0.12_f64,
      Some(100.0),
      Some(0.04),
      0.1,
      0.3156,
      0.0331,
      -0.681,
      0.0,
      64,
      Some(1.0),
      Some(25),
      17,
    );
    let [spots, variances] = p.sample_batch(13);
    assert_eq!(spots.dim(), (13, 64));
    assert_eq!(variances.dim(), (13, 64));
    assert!(variances.iter().all(|x| *x >= 0.0));
    assert!(spots.iter().all(|x| x.is_finite()));
  }

  #[test]
  fn heston_limit_h_half_asymptotic_variance_is_finite() {
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
    assert!(
      mean > 0.0 && mean < 1.0,
      "mean variance out of sanity range: {mean}"
    );
  }
}
