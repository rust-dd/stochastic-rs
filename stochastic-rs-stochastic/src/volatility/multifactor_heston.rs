//! # Multifactor Heston (Christoffersen-Heston-Jacobs 2009)
//!
//! $$
//! \begin{aligned}
//! \frac{dS_t}{S_t} &= \mu\,dt + \sum_{k=1}^{K} \sqrt{V_{k,t}}\,dW^{S}_{k,t}, \\
//! dV_{k,t} &= \kappa_k(\theta_k - V_{k,t})\,dt + \sigma_k\,\sqrt{V_{k,t}}\,dW^{V}_{k,t},\\
//! d\langle W^S_k, W^V_k\rangle_t &= \rho_k\,dt,\\
//! d\langle W^S_j, W^S_k\rangle_t &= 0,\ j \ne k,\\
//! d\langle W^V_j, W^V_k\rangle_t &= 0,\ j \ne k.
//! \end{aligned}
//! $$
//!
//! Each variance factor $V_{k,t}$ is an independent CIR process with its
//! own mean-reversion speed $\kappa_k$, long-run level $\theta_k$,
//! vol-of-vol $\sigma_k$ and **asset-correlation** $\rho_k$. The asset
//! diffusion sums the contributions of the $K$ factors.
//!
//! **Why multi-factor.** Christoffersen-Heston-Jacobs (CHJ, 2009) show
//! that a single CIR factor cannot simultaneously fit (a) the short-end
//! skew, (b) the long-end skew, and (c) the term structure of ATM
//! variance — separate factors with different $\kappa_k$ and $\rho_k$
//! disentangle these regimes.
//!
//! **Discretisation.** Plain explicit Euler on the asset and a
//! full-truncation Euler on each variance factor (Lord-Koekkoek-van Dijk
//! 2010). The asset draws $K$ correlated $(\Delta W^S_k, \Delta W^V_k)$
//! pairs per time step through the existing
//! [`Cgns`](crate::noise::cgns::Cgns) generator, independent across $k$.
//!
//! Reference: Christoffersen, P., Heston, S., Jacobs, K. (2009),
//! "The shape and term structure of the index option smirk: why
//! multifactor stochastic volatility models work so well",
//! *Management Science* 55(12), 1914-1932.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Multifactor Heston model with `K` independent variance factors driving
/// a single stock. See module docs for the SDE form and reference.
pub struct MultifactorHeston<T: FloatExt, const K: usize, S: SeedExt = Unseeded> {
  /// Initial stock price.
  pub s0: Option<T>,
  /// Initial variance per factor.
  pub v0: [T; K],
  /// Mean-reversion rate per factor.
  pub kappa: [T; K],
  /// Long-run variance per factor.
  pub theta: [T; K],
  /// Vol-of-vol per factor.
  pub sigma: [T; K],
  /// Asset-variance correlation per factor, $\rho_k \in (-1, 1)$.
  pub rho: [T; K],
  /// Drift of the stock (risk-neutral $r - q$ when pricing).
  pub mu: T,
  /// Number of time steps.
  pub n: usize,
  /// Time to maturity.
  pub t: Option<T>,
  /// Seed source.
  pub seed: S,
  /// One `Cgns` per factor for $(\Delta W^S_k, \Delta W^V_k)$ with
  /// correlation $\rho_k$.
  cgns: Vec<Cgns<T>>,
}

impl<T: FloatExt, const K: usize, S: SeedExt> MultifactorHeston<T, K, S> {
  pub fn new(
    s0: Option<T>,
    v0: [T; K],
    kappa: [T; K],
    theta: [T; K],
    sigma: [T; K],
    rho: [T; K],
    mu: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(K >= 1, "K must be ≥ 1");
    for k in 0..K {
      assert!(kappa[k] >= T::zero(), "κ_k must be non-negative");
      assert!(theta[k] >= T::zero(), "θ_k must be non-negative");
      assert!(sigma[k] >= T::zero(), "σ_k must be non-negative");
      assert!(v0[k] >= T::zero(), "v0_k must be non-negative");
      let r = rho[k].to_f64().unwrap();
      assert!(
        r.abs() < 1.0,
        "ρ_k must lie strictly in (-1, 1); got {r} at factor {k}"
      );
    }
    let cgns = (0..K)
      .map(|k| Cgns::new(rho[k], n - 1, t, Unseeded))
      .collect::<Vec<_>>();
    Self {
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      t,
      seed,
      cgns,
    }
  }
}

impl<T: FloatExt, const K: usize, S: SeedExt> ProcessExt<T> for MultifactorHeston<T, K, S> {
  /// `(stock_path, [variance_path; K])`.
  type Output = (Array1<T>, [Array1<T>; K]);
  type Sampler<'s>
    = MultifactorHestonSampler<T, K, S>
  where
    Self: 's;

  fn sampler(&self) -> MultifactorHestonSampler<T, K, S> {
    MultifactorHestonSampler {
      n: self.n,
      s0: self.s0.unwrap_or(T::zero()),
      v0: std::array::from_fn(|k| self.v0[k].max(T::zero())),
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma,
      mu: self.mu,
      dt: self.cgns[0].dt(),
      cgns: self.cgns.clone(),
      seed: self.seed.clone(),
    }
  }
}

/// Reusable [`MultifactorHeston`] sampling state: owns one correlated-Gaussian
/// generator per factor plus the seed source so a Monte-Carlo loop reuses the
/// stock buffer and all `K` variance buffers. Per-factor noise is re-derived in
/// factor order, so the first call reproduces the original stream bit-for-bit.
#[doc(hidden)]
pub struct MultifactorHestonSampler<T: FloatExt, const K: usize, S: SeedExt> {
  n: usize,
  s0: T,
  v0: [T; K],
  kappa: [T; K],
  theta: [T; K],
  sigma: [T; K],
  mu: T,
  dt: T,
  cgns: Vec<Cgns<T>>,
  seed: S,
}

impl<T: FloatExt, const K: usize, S: SeedExt> MultifactorHestonSampler<T, K, S> {
  fn fill_paths(&mut self, s: &mut [T], v: &mut [&mut [T]; K]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;

    s[0] = self.s0;
    for k in 0..K {
      v[k][0] = self.v0[k];
    }

    // Pre-sample all factor noises so the inner loop is a tight scalar pass.
    let factor_noises = (0..K)
      .map(|k| self.cgns[k].sample_impl(&self.seed.derive()))
      .collect::<Vec<_>>();

    for i in 1..self.n {
      // 1) Compute the total asset increment from all factor draws.
      let mut sum_sqrt_v_dw = T::zero();
      for k in 0..K {
        let v_prev = v[k][i - 1].max(T::zero());
        sum_sqrt_v_dw += v_prev.sqrt() * factor_noises[k][0][i - 1];
      }
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * sum_sqrt_v_dw;

      // 2) Full-truncation Euler step for each variance factor independently.
      for k in 0..K {
        let v_prev = v[k][i - 1].max(T::zero());
        let dv = self.kappa[k] * (self.theta[k] - v_prev) * dt
          + self.sigma[k] * v_prev.sqrt() * factor_noises[k][1][i - 1];
        v[k][i] = (v[k][i - 1] + dv).max(T::zero());
      }
    }
  }
}

impl<T: FloatExt, const K: usize, S: SeedExt> PathSampler<T> for MultifactorHestonSampler<T, K, S> {
  type Output = (Array1<T>, [Array1<T>; K]);

  fn sample_into(&mut self, out: &mut (Array1<T>, [Array1<T>; K])) {
    let (s_arr, v_arr) = out;
    let s = s_arr
      .as_slice_mut()
      .expect("MultifactorHeston stock output must be contiguous");
    let mut v: [&mut [T]; K] = v_arr.each_mut().map(|arr| {
      arr
        .as_slice_mut()
        .expect("MultifactorHeston variance output must be contiguous")
    });
    self.fill_paths(s, &mut v);
  }

  fn sample(&mut self) -> (Array1<T>, [Array1<T>; K]) {
    let mut s_arr = Array1::<T>::zeros(self.n);
    let mut v_arr: [Array1<T>; K] = std::array::from_fn(|_| Array1::<T>::zeros(self.n));
    {
      let s = s_arr.as_slice_mut().expect("contiguous");
      let mut v: [&mut [T]; K] = v_arr
        .each_mut()
        .map(|arr| arr.as_slice_mut().expect("contiguous"));
      self.fill_paths(s, &mut v);
    }
    (s_arr, v_arr)
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// K = 2: 2-factor Heston produces non-negative variances and a
  /// log-normal-like stock path (no negative S).
  #[test]
  fn two_factor_paths_non_negative() {
    let p = MultifactorHeston::<f64, 2, _>::new(
      Some(100.0),
      [0.04, 0.04],
      [1.5, 0.5],
      [0.04, 0.04],
      [0.3, 0.6],
      [-0.5, -0.2],
      0.0,
      256,
      Some(1.0),
      Unseeded,
    );
    let (s, [v1, v2]) = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
    assert!(v1.iter().all(|x| *x >= 0.0));
    assert!(v2.iter().all(|x| *x >= 0.0));
    assert_eq!(s.len(), 256);
    assert_eq!(v1.len(), 256);
    assert_eq!(v2.len(), 256);
  }

  /// Long-run variance E[V_k,∞] = θ_k, so the time-averaged path of each
  /// factor should approach θ_k on a long horizon.
  #[test]
  fn two_factor_long_run_variance_to_theta() {
    let theta = [0.04, 0.10];
    let p = MultifactorHeston::<f64, 2, _>::new(
      Some(100.0),
      [0.04, 0.10],
      [3.0, 3.0],
      theta,
      [0.2, 0.2],
      [-0.5, -0.5],
      0.0,
      4096,
      Some(10.0), // long horizon
      Deterministic::new(42),
    );
    let (_s, [v1, v2]) = p.sample();
    let mean1: f64 = v1.iter().skip(1024).copied().sum::<f64>() / (v1.len() - 1024) as f64;
    let mean2: f64 = v2.iter().skip(1024).copied().sum::<f64>() / (v2.len() - 1024) as f64;
    assert!(
      (mean1 - theta[0]).abs() < 0.02,
      "V_1 long-run mean = {mean1}, expected ≈ {}",
      theta[0]
    );
    assert!(
      (mean2 - theta[1]).abs() < 0.05,
      "V_2 long-run mean = {mean2}, expected ≈ {}",
      theta[1]
    );
  }

  /// K = 3: variant works without changes.
  #[test]
  fn three_factor_paths_finite() {
    let p = MultifactorHeston::<f64, 3, _>::new(
      Some(100.0),
      [0.02, 0.04, 0.06],
      [2.0, 1.0, 0.5],
      [0.02, 0.04, 0.06],
      [0.2, 0.3, 0.4],
      [-0.5, -0.3, -0.1],
      0.0,
      128,
      Some(1.0),
      Unseeded,
    );
    let (s, [v1, v2, v3]) = p.sample();
    for arr in [&s, &v1, &v2, &v3] {
      assert!(arr.iter().all(|x| x.is_finite()));
    }
  }

  /// Rejecting invalid ρ_k.
  #[test]
  #[should_panic(expected = "ρ_k must lie strictly in (-1, 1)")]
  fn rho_out_of_range_panics() {
    let _ = MultifactorHeston::<f64, 2, _>::new(
      Some(100.0),
      [0.04, 0.04],
      [1.0, 1.0],
      [0.04, 0.04],
      [0.2, 0.2],
      [-1.5, -0.5], // ρ_1 < -1 → invalid
      0.0,
      64,
      Some(1.0),
      Unseeded,
    );
  }
}
