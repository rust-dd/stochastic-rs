//! # Multi-factor (dynamic) SABR with a term structure of skew
//!
//! $$
//! \begin{aligned}
//! dF_t &= \alpha_t\,F_t^{\beta(t)}\,dW^1_t,\\
//! d\alpha_t &= \nu(t)\,\alpha_t\,dW^2_t,\\
//! d\langle W^1, W^2\rangle_t &= \rho(t)\,dt,
//! \end{aligned}
//! $$
//!
//! where the SABR coefficients $\beta(t)$, $\nu(t)$, $\rho(t)$ are
//! **piecewise-constant functions of time** instead of the scalar
//! constants of the classic Hagan-Kumar-Lesniewski-Woodward (2002) SABR.
//! Letting $\beta$ and $\rho$ vary across maturity buckets reproduces a
//! **term structure of skew** — the smile slope at short maturities can
//! differ from the long end, which the static SABR cannot match
//! simultaneously.
//!
//! The term structure is given by `knots` — sorted breakpoints in
//! $(0, T)$ that split $[0, T]$ into $K + 1$ buckets — and three
//! per-bucket coefficient vectors of length $K + 1$. Bucket $k$ covers
//! $[\text{knots}[k-1], \text{knots}[k])$ (with the obvious open ends),
//! and at simulation time $t_i$ the active bucket is the number of knots
//! $\le t_i$.
//!
//! ## Discretisation
//!
//! Because $\rho(t)$ is time-varying we cannot bake a single correlation
//! into the [`Cgns`](crate::noise::cgns::Cgns) generator. Instead we draw
//! two independent Gaussian increment streams $z^1, z^2$ via [`Gn`] and
//! correlate them on the fly per step:
//!
//! $$
//! \Delta W^1_i = z^1_i, \qquad
//! \Delta W^2_i = \rho(t_i)\,z^1_i + \sqrt{1 - \rho(t_i)^2}\,z^2_i.
//! $$
//!
//! The forward uses an explicit Euler step with absorption at 0
//! (the natural SABR boundary for $\beta < 1$); the volatility uses the
//! exact log-Euler step $\alpha_{i} = \alpha_{i-1}\exp(\nu\,\Delta W^2 -
//! \tfrac12\nu^2\Delta t)$, which preserves non-negativity for any step
//! size.
//!
//! References:
//! - Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002),
//!   "Managing smile risk", *Wilmott Magazine*, 84-108.
//! - Fernández, J.L., Ferreiro, A.M., García, J.A. et al. (2024),
//!   "Static and dynamic SABR stochastic volatility models: calibration
//!   and option pricing using GPUs", arXiv:2407.20713 — dynamic
//!   (time-dependent-parameter) SABR formulation.
//! - Osajima, Y. (2007), "The asymptotic expansion formula of implied
//!   volatility for dynamic SABR model", SSRN 965265.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Dynamic / multi-factor SABR with piecewise-constant `beta`, `rho` and
/// `nu` term structures. See module docs for the SDE and discretisation.
pub struct MultifactorSabr<T: FloatExt, S: SeedExt = Unseeded> {
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Initial volatility level $\alpha_0$.
  pub v0: Option<T>,
  /// Sorted time breakpoints in $(0, T)$. Length `K`; produces `K + 1`
  /// buckets. May be empty (then the model is the static SABR with the
  /// single bucket's coefficients).
  pub knots: Vec<T>,
  /// Backbone exponent $\beta(t) \in [0, 1]$ per bucket. Length `K + 1`.
  pub beta: Vec<T>,
  /// Correlation $\rho(t) \in (-1, 1)$ per bucket. Length `K + 1`.
  pub rho: Vec<T>,
  /// Vol-of-vol $\nu(t) \ge 0$ per bucket. Length `K + 1`.
  pub nu: Vec<T>,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> MultifactorSabr<T, S> {
  pub fn new(
    f0: Option<T>,
    v0: Option<T>,
    knots: Vec<T>,
    beta: Vec<T>,
    rho: Vec<T>,
    nu: Vec<T>,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    let buckets = knots.len() + 1;
    assert_eq!(
      beta.len(),
      buckets,
      "beta must have knots.len() + 1 = {buckets} entries"
    );
    assert_eq!(rho.len(), buckets, "rho must have {buckets} entries");
    assert_eq!(nu.len(), buckets, "nu must have {buckets} entries");
    for k in 0..buckets {
      assert!(
        beta[k] >= T::zero() && beta[k] <= T::one(),
        "beta[{k}] must be in [0, 1]"
      );
      assert!(nu[k] >= T::zero(), "nu[{k}] must be non-negative");
      let r = rho[k].to_f64().unwrap();
      assert!(r.abs() < 1.0, "rho[{k}] must lie strictly in (-1, 1)");
    }
    for w in knots.windows(2) {
      assert!(w[0] < w[1], "knots must be strictly increasing");
    }
    if let Some(v0) = v0 {
      assert!(v0 >= T::zero(), "v0 must be non-negative");
    }
    Self {
      f0,
      v0,
      knots,
      beta,
      rho,
      nu,
      n,
      t,
      seed,
    }
  }

  /// Index of the active bucket at time `time`: the number of knots
  /// less than or equal to `time`, clamped to the last bucket.
  #[inline]
  fn bucket_at(&self, time: T) -> usize {
    let mut idx = 0usize;
    for &knot in &self.knots {
      if time >= knot {
        idx += 1;
      } else {
        break;
      }
    }
    idx.min(self.beta.len() - 1)
  }

  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for MultifactorSabr<T, S> {
  /// `(forward path, volatility path)`.
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.dt();

    // Two independent standard-normal increment streams (std-dev √dt),
    // combined per step under the time-varying correlation ρ(t_i).
    let gn1 = Gn::<T, _>::new(self.n - 1, self.t, self.seed.derive());
    let gn2 = Gn::<T, _>::new(self.n - 1, self.t, self.seed.derive());
    let z1 = gn1.sample();
    let z2 = gn2.sample();

    let mut f_ = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    f_[0] = self.f0.unwrap_or(T::zero());
    v[0] = self.v0.unwrap_or(T::zero()).max(T::zero());

    for i in 1..self.n {
      let time = T::from_usize_(i - 1) * dt;
      let bucket = self.bucket_at(time);
      let beta = self.beta[bucket];
      let rho = self.rho[bucket];
      let nu = self.nu[bucket];

      let dw1 = z1[i - 1];
      // Correlate the second stream against the first at ρ(t_i). z1, z2
      // already carry the √dt scaling from `Gn`.
      let dw2 = rho * dw1 + (T::one() - rho * rho).sqrt() * z2[i - 1];

      let f_prev = f_[i - 1].max(T::zero());
      let v_prev = v[i - 1].max(T::zero());
      f_[i] = (f_[i - 1] + v_prev * f_prev.powf(beta) * dw1).max(T::zero());
      // Exact step for dα = ν α dW: α_i = α_{i-1} exp(ν ΔW₂ − ½ν²Δt).
      v[i] = v_prev * (nu * dw2 - T::from_f64_fast(0.5) * nu * nu * dt).exp();
    }

    [f_, v]
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Volatility stays non-negative across a 2-bucket term structure.
  #[test]
  fn dynamic_sabr_volatility_non_negative() {
    let p = MultifactorSabr::<f64, _>::new(
      Some(1.0),
      Some(0.2),
      vec![0.5],        // single knot → 2 buckets
      vec![0.3, 0.7],   // β: 0.3 short end, 0.7 long end
      vec![-0.5, -0.2], // ρ term structure
      vec![0.4, 0.3],   // ν term structure
      512,
      Some(1.0),
      Unseeded,
    );
    let [_f, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }

  /// Empty-knot dynamic SABR with a single bucket must reproduce the
  /// classic static SABR dynamics (forward stays non-negative, vol path
  /// non-negative).
  #[test]
  fn dynamic_sabr_single_bucket_reduces_to_static() {
    let p = MultifactorSabr::<f64, _>::new(
      Some(1.0),
      Some(0.2),
      vec![], // no knots → 1 bucket
      vec![0.5],
      vec![-0.3],
      vec![0.4],
      256,
      Some(1.0),
      Unseeded,
    );
    let [f, v] = p.sample();
    assert!(f.iter().all(|x| *x >= 0.0));
    assert!(v.iter().all(|x| *x >= 0.0));
    assert_eq!(f.len(), 256);
  }

  /// The bucket lookup picks the right segment across the knots.
  #[test]
  fn dynamic_sabr_bucket_lookup() {
    let p = MultifactorSabr::<f64, _>::new(
      Some(1.0),
      Some(0.2),
      vec![0.25, 0.5, 0.75],
      vec![0.1, 0.2, 0.3, 0.4],
      vec![-0.1, -0.2, -0.3, -0.4],
      vec![0.1, 0.2, 0.3, 0.4],
      128,
      Some(1.0),
      Unseeded,
    );
    assert_eq!(p.bucket_at(0.0), 0);
    assert_eq!(p.bucket_at(0.1), 0);
    assert_eq!(p.bucket_at(0.25), 1);
    assert_eq!(p.bucket_at(0.3), 1);
    assert_eq!(p.bucket_at(0.5), 2);
    assert_eq!(p.bucket_at(0.75), 3);
    assert_eq!(p.bucket_at(0.99), 3);
    // Beyond the last knot stays in the last bucket.
    assert_eq!(p.bucket_at(5.0), 3);
  }

  /// Different β across buckets must produce a forward path whose
  /// short-end and long-end realised local-vol scale differently. We
  /// check determinism reproduces under a fixed seed.
  #[test]
  fn dynamic_sabr_seed_determinism() {
    let make = || {
      MultifactorSabr::<f64, _>::new(
        Some(1.0),
        Some(0.25),
        vec![0.5],
        vec![0.2, 0.8],
        vec![-0.4, -0.1],
        vec![0.5, 0.2],
        256,
        Some(1.0),
        Deterministic::new(7),
      )
    };
    let [f1, v1] = make().sample();
    let [f2, v2] = make().sample();
    assert_eq!(f1, f2, "forward path must be seed-deterministic");
    assert_eq!(v1, v2, "vol path must be seed-deterministic");
  }

  /// Coefficient-vector length mismatch is rejected.
  #[test]
  #[should_panic(expected = "beta must have")]
  fn dynamic_sabr_length_mismatch_panics() {
    let _ = MultifactorSabr::<f64, _>::new(
      Some(1.0),
      Some(0.2),
      vec![0.5], // 2 buckets
      vec![0.5], // β has only 1 entry → mismatch
      vec![-0.3, -0.2],
      vec![0.4, 0.3],
      64,
      Some(1.0),
      Unseeded,
    );
  }
}
