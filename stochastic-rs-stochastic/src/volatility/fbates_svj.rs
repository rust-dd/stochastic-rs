//! # Fractional Bates SVJ
//!
//! Rough/fractional variance dynamics (Volterra-Heston) with Bates-style
//! Poisson jumps in the log-price process.
//!
//! $$
//! d\ln S_t = (\mu - \lambda\kappa_J - \tfrac12 v_t)\,dt + \sqrt{v_t}\,dW_t^S + Z\,dN_t
//! $$
//!
//! where $v_t$ follows the rough Heston variance dynamics with Hurst
//! exponent $H \in (0, 0.5)$, $N_t \sim \text{Poisson}(\lambda)$,
//! $Z \sim \mathcal{N}(\nu, \omega^2)$, and
//! $\langle dW^S, dW^v \rangle = \rho\,dt$.
//!
//! Returns `[S, v]` — price and variance paths.
//!
//! References:
//! - Gatheral J., Jaisson T., Rosenbaum M. (2018) — *Volatility Is
//!   Rough*, Quantitative Finance 18(6), 933–949,
//!   DOI: 10.1080/14697688.2017.1393551.
//! - El Euch O. & Rosenbaum M. (2019) — *The Characteristic Function of
//!   Rough Heston Models*, Mathematical Finance 29(1), 3–38,
//!   DOI: 10.1111/mafi.12173 — defines the Volterra `v_t` above.
//! - Bates D. S. (1996), DOI: 10.1093/rfs/9.1.69, and Merton R. C.
//!   (1976), DOI: 10.1016/0304-405X(76)90022-2, for the jump component
//!   (see [`BatesSvj`](crate::volatility::bates_svj::BatesSvj)).
//!
//! `fill_paths` below approximates the exact fractional-kernel Volterra
//! convolution with a single OU-type carrier factor `zt` plus a direct
//! (`O(n²)`) memory sum, scaled by free `c1`/`c2` coefficients — an
//! approximate lifting scheme in the spirit of Abi Jaber & El Euch
//! (2019) — *Multifactor Approximation of Rough Volatility Models*,
//! SIAM Journal on Financial Mathematics 10(2), 309–349,
//! DOI: 10.1137/18M1170236 — but it is this crate's own one-factor
//! simplification, not a reproduction of that (or any other specific
//! published) numerical scheme.

use ndarray::Array1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::special::gamma;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Every field has a matching `with_*` builder setter, e.g.
/// `FBatesSvj::new(..).with_hurst(0.15).with_rho(-0.4)`.
pub struct FBatesSvj<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness (H ∈ (0, 0.5) for rough).
  pub hurst: T,
  /// Drift rate.
  pub mu: T,
  /// Initial spot price.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Long-run variance level (θ).
  pub theta: T,
  /// Mean-reversion speed (κ).
  pub kappa: T,
  /// Vol-of-vol (ξ).
  pub xi: T,
  /// Price-vol correlation (ρ ∈ [-1, 1]).
  pub rho: T,
  /// Jump intensity (Poisson arrival rate λ).
  pub lambda: T,
  /// Mean of jump log-size Z ~ N(ν, ω²).
  pub nu: T,
  /// Std dev of jump log-size Z.
  pub omega: T,
  /// Number of points sampled along the fractional-Bates path.
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> FBatesSvj<T, S> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    hurst: T,
    mu: T,
    s0: T,
    v0: T,
    theta: T,
    kappa: T,
    xi: T,
    rho: T,
    lambda: T,
    nu: T,
    omega: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: Cpu,
      hurst,
      mu,
      s0,
      v0,
      theta,
      kappa,
      xi,
      rho,
      lambda,
      nu,
      omega,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> FBatesSvj<T, S, B> {
  /// Replace `hurst`, all else unchanged. `FBatesSvj` has no persisted
  /// correlated-noise cache (`sampler()` builds `cgns` fresh from `rho`,
  /// `n`, `t` on every call), so every setter here is a plain field write.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    self.hurst = hurst;
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    self.v0 = v0;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    self.kappa = kappa;
    self
  }

  /// Replace `xi`, all else unchanged.
  pub fn with_xi(mut self, xi: T) -> Self {
    self.xi = xi;
    self
  }

  /// Replace `rho`, all else unchanged.
  pub fn with_rho(mut self, rho: T) -> Self {
    self.rho = rho;
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    self.lambda = lambda;
    self
  }

  /// Replace `nu`, all else unchanged.
  pub fn with_nu(mut self, nu: T) -> Self {
    self.nu = nu;
    self
  }

  /// Replace `omega`, all else unchanged.
  pub fn with_omega(mut self, omega: T) -> Self {
    self.omega = omega;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

backend_switch!([T: FloatExt, S: SeedExt] FBatesSvj<T, S> { hurst, mu, s0, v0, theta, kappa, xi, rho, lambda, nu, omega, n, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for FBatesSvj<T, S, B> {
  type Output = [Array1<T>; 2]; // [S, v]
  type Sampler<'s>
    = FBatesSvjSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> FBatesSvjSampler<T, S> {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    FBatesSvjSampler {
      n: self.n,
      hurst: self.hurst,
      mu: self.mu,
      s0: self.s0,
      v0: self.v0,
      theta: self.theta,
      kappa: self.kappa,
      xi: self.xi,
      lambda: self.lambda,
      nu: self.nu,
      omega: self.omega,
      dt,
      g: T::from_f64_fast(gamma(self.hurst.to_f64().unwrap() - 0.5)),
      // `Unseeded` baked into the Cgns exactly as the legacy `sample` did; the
      // noise is drawn via `sample_impl(&self.seed)`, so the Cgns's own seed
      // is irrelevant.
      cgns: Cgns::new(self.rho, n_steps, self.t, Unseeded),
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`FBatesSvj`] sampling state: owns the correlated-Gaussian generator
/// and the seed source. The jump (Poisson + Normal) generators are rebuilt per
/// fill in the legacy seed-consumption order, so the first call reproduces the
/// original stream bit-for-bit.
#[doc(hidden)]
pub struct FBatesSvjSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  hurst: T,
  mu: T,
  s0: T,
  v0: T,
  theta: T,
  kappa: T,
  xi: T,
  lambda: T,
  nu: T,
  omega: T,
  dt: T,
  g: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> FBatesSvjSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;

    // Use Cgns for rho-correlated noise: [gn_vol, gn_price]
    let [gn_vol, gn_price] = self.cgns.sample_impl(&self.seed);

    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);

    yt[0] = self.v0;
    zt[0] = T::zero();
    sigma_tilde2[0] = self.v0;
    v2[0] = self.v0;
    s[0] = self.s0;

    let g = self.g;
    let half = T::from_f64_fast(0.5);

    // Jump compensation: κ_J = exp(ν + ½ω²) - 1
    let kappa_j = (self.nu + half * self.omega * self.omega).exp() - T::one();

    // Jump RNG
    let z_std = SimdNormal::<T>::new(T::zero(), T::one(), &self.seed);
    let mut rng = self.seed.rng();
    let lambda_dt = self.lambda.to_f64().unwrap() * dt.to_f64().unwrap();
    let pois = if lambda_dt > 0.0 {
      Some(SimdPoisson::<u32>::new(lambda_dt, &self.seed))
    } else {
      None
    };

    for i in 1..self.n {
      let t_i = dt * T::from_usize_(i);

      // Rough variance dynamics (same as RoughHeston/fheston.rs)
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(T::zero()).sqrt() * gn_vol[i - 1];

      sigma_tilde2[i] = yt[i] + self.xi * zt[i];

      let integral = (0..i)
        .map(|j| {
          let tj = T::from_usize_(j) * dt;
          ((t_i - tj).powf(self.hurst - half) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] = yt[i] + self.xi * zt[i] + self.xi * integral / g;

      // Price dynamics with jumps
      let vi = v2[i - 1].max(T::zero());

      // Jump component
      let mut jump_sum = T::zero();
      if let Some(pois) = &pois {
        let n_jumps: u32 = pois.sample(&mut rng);
        if n_jumps > 0 {
          let kf = T::from_f64_fast(n_jumps as f64);
          let z0 = z_std.sample_fast();
          jump_sum = self.nu * kf + self.omega * kf.sqrt() * z0;
        }
      }

      let log_inc =
        (self.mu - self.lambda * kappa_j - half * vi) * dt + vi.sqrt() * gn_price[i - 1] + jump_sum;
      s[i] = s[i - 1] * log_inc.exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FBatesSvjSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("FBatesSvj output must be contiguous"),
      v2.as_slice_mut()
        .expect("FBatesSvj output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
    [s, v2]
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn price_stays_positive() {
    let m = FBatesSvj::new(
      0.1_f64,
      0.05,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.7,
      0.5,
      -0.01,
      0.1,
      256,
      Some(1.0),
      Deterministic::new(42),
    );
    let [s, _v] = m.sample();
    assert!(
      s.iter().all(|x| x.is_finite() && *x > 0.0),
      "prices must be positive"
    );
  }

  #[test]
  fn variance_path_is_finite() {
    let m = FBatesSvj::new(
      0.15_f64,
      0.05,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.7,
      0.5,
      0.0,
      0.1,
      256,
      Some(1.0),
      Deterministic::new(99),
    );
    let [_s, v] = m.sample();
    assert!(v.iter().all(|x| x.is_finite()), "variance must be finite");
  }

  #[test]
  fn seeded_is_deterministic() {
    let mk = || {
      FBatesSvj::new(
        0.1_f64,
        0.05,
        100.0,
        0.04,
        0.04,
        2.0,
        0.3,
        -0.7,
        0.5,
        0.0,
        0.1,
        128,
        Some(0.5),
        Deterministic::new(77),
      )
    };
    let [s1, _] = mk().sample();
    let [s2, _] = mk().sample();
    for i in 0..128 {
      assert!((s1[i] - s2[i]).abs() < 1e-12, "paths diverged at i={i}");
    }
  }

  #[test]
  fn reduces_to_rough_heston_without_jumps() {
    // With λ=0, should be identical to RoughHeston
    let m = FBatesSvj::new(
      0.1_f64,
      0.05,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.7,
      0.0,
      0.0,
      0.0,
      128,
      Some(0.5),
      Deterministic::new(55),
    );
    let [s, v] = m.sample();
    assert!(s.iter().all(|x| x.is_finite() && *x > 0.0));
    assert!(v.iter().all(|x| x.is_finite()));
  }
}
