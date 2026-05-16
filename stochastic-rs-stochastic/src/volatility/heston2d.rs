//! # Heston 2D (bivariate Heston)
//!
//! $$
//! \begin{aligned}
//! dx^i_t &= \left(\mu_i - \tfrac{1}{2} v^i_t\right) dt + \sqrt{v^i_t}\, dW^i_t \\
//! dv^i_t &= \kappa_i(\theta_i - v^i_t)\, dt + \sigma_i \sqrt{v^i_t}\, dZ^i_t, \quad i=1,2
//! \end{aligned}
//! $$
//!
//! Two correlated Heston stochastic-volatility processes (one per asset) with
//! a full 4-dimensional correlation structure on `(Z_1, Z_2, W_1, W_2)` where
//! `W_i` drives the log-price `x^i` and `Z_i` drives the variance `v^i`.
//!
//! Direct port of `Heston2D.m` from the MATLAB FSDA toolbox (Sanfelici &
//! Toscano 2024, arXiv:2402.00172). The 6-element correlation vector matches
//! the MATLAB convention element-for-element so the same parameters reproduce
//! the same bivariate trajectories (up to RNG choice).
//!
//! Output is **log-prices** `x_i`, not levels `S_i = exp(x_i)`, matching MATLAB
//! and the convention expected by `FMVol`.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Bivariate Heston stochastic-volatility process.
///
/// Two correlated Heston SDEs with a 4×4 instantaneous-correlation structure
/// on `(Z_1, Z_2, W_1, W_2)`, parameterised by the 6-element MATLAB-style
/// correlation vector `rho`. Sampling produces four arrays `[x_1, v_1, x_2,
/// v_2]` of length `n`, where `x_i` is the **log-price** and `v_i` is the
/// instantaneous variance.
pub struct Heston2D<T: FloatExt, S: SeedExt = Unseeded> {
  /// Initial log-prices `[x_1(0), x_2(0)]`.
  pub x0: [Option<T>; 2],
  /// Initial variances `[v_1(0), v_2(0)]`.
  pub v0: [Option<T>; 2],
  /// Drifts `[μ_1, μ_2]` of the log-price processes.
  pub mu: [T; 2],
  /// Long-run variances `[θ_1, θ_2]`.
  pub theta: [T; 2],
  /// Mean-reversion speeds `[κ_1, κ_2]`.
  pub kappa: [T; 2],
  /// Volatility-of-volatilities `[σ_1, σ_2]`.
  pub sigma: [T; 2],
  /// 6-element correlation vector matching the MATLAB `Heston2D.m`
  /// convention, ordered `[ρ(Z_1,Z_2), ρ(Z_1,W_1), ρ(Z_1,W_2),
  /// ρ(Z_2,W_1), ρ(Z_2,W_2), ρ(W_1,W_2)]`.
  pub rho: [T; 6],
  /// Number of points (so `n - 1` time steps).
  pub n: usize,
  /// Time horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Reflect negative variance to its absolute value (true) or floor at zero (false / None).
  pub use_sym: Option<bool>,
  /// Seed strategy.
  pub seed: S,
  /// Cholesky factor `L` of the 4×4 correlation matrix, stored as
  /// `[L_11, L_21, L_22, L_31, L_32, L_33, L_41, L_42, L_43, L_44]`.
  chol: [T; 10],
}

/// Lower-triangular Cholesky factor of the 4×4 correlation matrix induced by
/// the MATLAB-ordered `rho` vector `[ρ(Z1,Z2), ρ(Z1,W1), ρ(Z1,W2), ρ(Z2,W1),
/// ρ(Z2,W2), ρ(W1,W2)]` over the basis `(Z_1, Z_2, W_1, W_2)`. Panics if the
/// correlation matrix is not positive semidefinite.
fn cholesky_4x4<T: FloatExt>(rho: [T; 6]) -> [T; 10] {
  // Numerical tolerance for clamping diagonal pivots that should be zero
  // (rank-deficient but valid correlation matrices) but appear slightly
  // negative due to floating-point error.
  let tol = T::from_f64_fast(1e-10);
  let nonneg = |x: T, pivot: usize| {
    assert!(x >= -tol, "correlation matrix not PSD at pivot {}", pivot);
    x.max(T::zero())
  };

  let l11 = T::one();
  let l21 = rho[0];
  let l22 = nonneg(T::one() - l21 * l21, 2).sqrt();
  let l31 = rho[1];
  let l32 = if l22 > T::zero() {
    (rho[3] - l31 * l21) / l22
  } else {
    T::zero()
  };
  let l33 = nonneg(T::one() - l31 * l31 - l32 * l32, 3).sqrt();
  let l41 = rho[2];
  let l42 = if l22 > T::zero() {
    (rho[4] - l41 * l21) / l22
  } else {
    T::zero()
  };
  let l43 = if l33 > T::zero() {
    (rho[5] - l41 * l31 - l42 * l32) / l33
  } else {
    T::zero()
  };
  let l44 = nonneg(T::one() - l41 * l41 - l42 * l42 - l43 * l43, 4).sqrt();
  [l11, l21, l22, l31, l32, l33, l41, l42, l43, l44]
}

fn validate_params<T: FloatExt>(
  v0: &[Option<T>; 2],
  theta: &[T; 2],
  kappa: &[T; 2],
  sigma: &[T; 2],
  rho: &[T; 6],
  n: usize,
) {
  assert!(n >= 2, "n must be >= 2");
  for i in 0..2 {
    assert!(kappa[i] >= T::zero(), "kappa[{}] must be non-negative", i);
    assert!(theta[i] >= T::zero(), "theta[{}] must be non-negative", i);
    assert!(sigma[i] >= T::zero(), "sigma[{}] must be non-negative", i);
    if let Some(v) = v0[i] {
      assert!(v >= T::zero(), "v0[{}] must be non-negative", i);
    }
  }
  for (idx, r) in rho.iter().enumerate() {
    assert!(
      *r >= -T::one() && *r <= T::one(),
      "rho[{}] out of [-1, 1]",
      idx
    );
  }
}

impl<T: FloatExt, S: SeedExt> Heston2D<T, S> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    x0: [Option<T>; 2],
    v0: [Option<T>; 2],
    mu: [T; 2],
    theta: [T; 2],
    kappa: [T; 2],
    sigma: [T; 2],
    rho: [T; 6],
    n: usize,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    validate_params(&v0, &theta, &kappa, &sigma, &rho, n);
    let chol = cholesky_4x4::<T>(rho);
    Self {
      x0,
      v0,
      mu,
      theta,
      kappa,
      sigma,
      rho,
      n,
      t,
      use_sym,
      seed,
      chol,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Heston2D<T, S> {
  /// Output: `[x_1, v_1, x_2, v_2]` — log-prices and instantaneous variances
  /// for both assets, each of length `n`.
  type Output = [Array1<T>; 4];

  fn sample(&self) -> Self::Output {
    let t_total = self.t.unwrap_or(T::one());
    let n_steps = self.n - 1;
    let dt = t_total / T::from_usize_(n_steps);
    let sqrt_dt = dt.sqrt();

    // Four independent N(0, dt) streams driven by derived seeds.
    let mut e1 = Array1::<T>::zeros(n_steps);
    let mut e2 = Array1::<T>::zeros(n_steps);
    let mut e3 = Array1::<T>::zeros(n_steps);
    let mut e4 = Array1::<T>::zeros(n_steps);
    for arr in [&mut e1, &mut e2, &mut e3, &mut e4] {
      let n_norm = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed.derive());
      n_norm.fill_slice_fast(arr.as_slice_mut().expect("noise slice contiguous"));
    }

    let [l11, l21, l22, l31, l32, l33, l41, l42, l43, l44] = self.chol;

    let mut x1 = Array1::<T>::zeros(self.n);
    let mut v1 = Array1::<T>::zeros(self.n);
    let mut x2 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);

    x1[0] = self.x0[0].unwrap_or(T::zero());
    v1[0] = self.v0[0].unwrap_or(T::zero()).max(T::zero());
    x2[0] = self.x0[1].unwrap_or(T::zero());
    v2[0] = self.v0[1].unwrap_or(T::zero()).max(T::zero());

    let half = T::from_f64_fast(0.5);
    let use_sym = self.use_sym.unwrap_or(false);
    for i in 1..self.n {
      let dz1 = l11 * e1[i - 1];
      let dz2 = l21 * e1[i - 1] + l22 * e2[i - 1];
      let dw1 = l31 * e1[i - 1] + l32 * e2[i - 1] + l33 * e3[i - 1];
      let dw2 = l41 * e1[i - 1] + l42 * e2[i - 1] + l43 * e3[i - 1] + l44 * e4[i - 1];

      let v1_prev = v1[i - 1].max(T::zero());
      let v2_prev = v2[i - 1].max(T::zero());

      let dv1 =
        self.kappa[0] * (self.theta[0] - v1_prev) * dt + self.sigma[0] * v1_prev.sqrt() * dz1;
      let dv2 =
        self.kappa[1] * (self.theta[1] - v2_prev) * dt + self.sigma[1] * v2_prev.sqrt() * dz2;

      v1[i] = if use_sym {
        (v1[i - 1] + dv1).abs()
      } else {
        (v1[i - 1] + dv1).max(T::zero())
      };
      v2[i] = if use_sym {
        (v2[i - 1] + dv2).abs()
      } else {
        (v2[i - 1] + dv2).max(T::zero())
      };

      x1[i] = x1[i - 1] + (self.mu[0] - half * v1_prev) * dt + v1_prev.sqrt() * dw1;
      x2[i] = x2[i - 1] + (self.mu[1] - half * v2_prev) * dt + v2_prev.sqrt() * dw2;
    }

    [x1, v1, x2, v2]
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn rho_default<T: FloatExt>() -> [T; 6] {
    // MATLAB Heston2D.m example: Rho=[0.5,-0.5,0,0,-0.5,0.5]
    //   ρ(Z1,Z2)=0.5, ρ(Z1,W1)=-0.5, ρ(Z1,W2)=0, ρ(Z2,W1)=0, ρ(Z2,W2)=-0.5, ρ(W1,W2)=0.5
    [
      T::from_f64_fast(0.5),
      T::from_f64_fast(-0.5),
      T::zero(),
      T::zero(),
      T::from_f64_fast(-0.5),
      T::from_f64_fast(0.5),
    ]
  }

  #[test]
  fn shapes_match_n() {
    let h = Heston2D::<f64, _>::new(
      [Some(0.0), Some(0.0)],
      [Some(0.4), Some(0.4)],
      [0.0, 0.0],
      [0.4, 0.4],
      [2.0, 2.0],
      [1.0, 1.0],
      rho_default(),
      512,
      Some(1.0),
      Some(false),
      Unseeded,
    );
    let [x1, v1, x2, v2] = h.sample();
    assert_eq!(x1.len(), 512);
    assert_eq!(v1.len(), 512);
    assert_eq!(x2.len(), 512);
    assert_eq!(v2.len(), 512);
    assert!(v1.iter().all(|x| *x >= 0.0));
    assert!(v2.iter().all(|x| *x >= 0.0));
  }

  #[test]
  fn seeded_is_deterministic() {
    let mk = || {
      Heston2D::<f64, Deterministic>::new(
        [Some(0.0), Some(0.0)],
        [Some(0.4), Some(0.4)],
        [0.0, 0.0],
        [0.4, 0.4],
        [2.0, 2.0],
        [1.0, 1.0],
        rho_default(),
        128,
        Some(1.0),
        Some(false),
        Deterministic::new(42),
      )
    };
    let [a, _, b, _] = mk().sample();
    let [c, _, d, _] = mk().sample();
    for i in 0..a.len() {
      assert!((a[i] - c[i]).abs() < 1e-12);
      assert!((b[i] - d[i]).abs() < 1e-12);
    }
  }

  #[test]
  fn cross_correlation_matches_rho() {
    // With a long path and ρ(W1,W2) = 0.8, the sample correlation of the
    // log-price increments should be close to ρ_W1W2 · sqrt(v_1 v_2) /
    // sqrt(v_1)/sqrt(v_2) = ρ_W1W2 (when variances are constant on average).
    let rho_w1w2 = 0.8_f64;
    let rho: [f64; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, rho_w1w2];
    let h = Heston2D::<f64, Deterministic>::new(
      [Some(0.0), Some(0.0)],
      [Some(0.4), Some(0.4)],
      [0.0, 0.0],
      [0.4, 0.4],
      [2.0, 2.0],
      [0.5, 0.5],
      rho,
      20_000,
      Some(1.0),
      Some(false),
      Deterministic::new(7),
    );
    let [x1, _v1, x2, _v2] = h.sample();
    let r1: Vec<f64> = (1..x1.len()).map(|i| x1[i] - x1[i - 1]).collect();
    let r2: Vec<f64> = (1..x2.len()).map(|i| x2[i] - x2[i - 1]).collect();
    let n = r1.len() as f64;
    let mean1 = r1.iter().sum::<f64>() / n;
    let mean2 = r2.iter().sum::<f64>() / n;
    let cov: f64 = r1
      .iter()
      .zip(r2.iter())
      .map(|(a, b)| (a - mean1) * (b - mean2))
      .sum::<f64>()
      / n;
    let var1: f64 = r1.iter().map(|a| (a - mean1).powi(2)).sum::<f64>() / n;
    let var2: f64 = r2.iter().map(|b| (b - mean2).powi(2)).sum::<f64>() / n;
    let corr = cov / (var1.sqrt() * var2.sqrt());
    assert!(
      (corr - rho_w1w2).abs() < 0.05,
      "sample corr {corr:.4} far from target {rho_w1w2}"
    );
  }

  #[test]
  #[should_panic(expected = "not PSD")]
  fn rejects_non_psd_correlation() {
    let bad: [f64; 6] = [0.99, 0.99, 0.99, 0.99, 0.99, -0.99];
    let _ = Heston2D::<f64, _>::new(
      [Some(0.0), Some(0.0)],
      [Some(0.4), Some(0.4)],
      [0.0, 0.0],
      [0.4, 0.4],
      [2.0, 2.0],
      [1.0, 1.0],
      bad,
      16,
      Some(1.0),
      Some(false),
      Unseeded,
    );
  }
}
