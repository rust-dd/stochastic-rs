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
//! Based on `Heston2D.m` from the MATLAB FSDA toolbox (Sanfelici & Toscano,
//! arXiv:2402.00172). The correlation-vector ordering and Euler drift/diffusion
//! terms match MATLAB. Negative Euler variance proposals are stabilised by
//! reflection or full truncation, so paths can differ from the raw MATLAB Euler
//! scheme even when the random draws agree.
//!
//! Output is **log-prices** `x_i`, not levels `S_i = exp(x_i)`, matching MATLAB
//! and the convention expected by `FMVol`.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::PathSampler;
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
  /// Initial variances `[v_1(0), v_2(0)]`; both entries must be present,
  /// finite and strictly positive, matching the MATLAB source contract.
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
  /// Positive finite time horizon (defaults to 1 when omitted).
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
  // A valid rank-deficient correlation matrix can produce a tiny negative pivot after rounding.
  let tol = T::from_f64_fast(1e-10).max(T::from_f64_fast(100.0) * T::epsilon());
  let nonneg = |x: T, pivot: usize| {
    assert!(x >= -tol, "correlation matrix not PSD at pivot {}", pivot);
    x.max(T::zero())
  };
  let divide_or_zero = |residual: T, diagonal: T, pivot: usize| {
    if diagonal > T::zero() {
      residual / diagonal
    } else {
      assert!(
        residual.abs() <= tol,
        "correlation matrix not PSD at pivot {}",
        pivot
      );
      T::zero()
    }
  };

  let l11 = T::one();
  let l21 = rho[0];
  let l22 = nonneg(T::one() - l21 * l21, 2).sqrt();
  let l31 = rho[1];
  let l32 = divide_or_zero(rho[3] - l31 * l21, l22, 2);
  let l33 = nonneg(T::one() - l31 * l31 - l32 * l32, 3).sqrt();
  let l41 = rho[2];
  let l42 = divide_or_zero(rho[4] - l41 * l21, l22, 2);
  let l43 = divide_or_zero(rho[5] - l41 * l31 - l42 * l32, l33, 3);
  let l44 = nonneg(T::one() - l41 * l41 - l42 * l42 - l43 * l43, 4).sqrt();
  [l11, l21, l22, l31, l32, l33, l41, l42, l43, l44]
}

fn validate_params<T: FloatExt>(
  x0: &[Option<T>; 2],
  v0: &[Option<T>; 2],
  mu: &[T; 2],
  theta: &[T; 2],
  kappa: &[T; 2],
  sigma: &[T; 2],
  rho: &[T; 6],
  n: usize,
  t: Option<T>,
) {
  assert!(n >= 2, "n must be >= 2");
  if let Some(value) = t {
    assert!(value.is_finite(), "t must be finite");
    assert!(value > T::zero(), "t must be positive");
  }
  for i in 0..2 {
    if let Some(value) = x0[i] {
      assert!(value.is_finite(), "x0[{}] must be finite", i);
    }
    let value = v0[i].expect("both initial variances v0 must be specified");
    assert!(value.is_finite(), "v0[{}] must be finite", i);
    assert!(value > T::zero(), "v0[{}] must be positive", i);
    assert!(mu[i].is_finite(), "mu[{}] must be finite", i);
    assert!(theta[i].is_finite(), "theta[{}] must be finite", i);
    assert!(kappa[i].is_finite(), "kappa[{}] must be finite", i);
    assert!(sigma[i].is_finite(), "sigma[{}] must be finite", i);
    assert!(kappa[i] >= T::zero(), "kappa[{}] must be non-negative", i);
    assert!(theta[i] >= T::zero(), "theta[{}] must be non-negative", i);
    assert!(sigma[i] >= T::zero(), "sigma[{}] must be non-negative", i);
    let feller_lhs = T::from_f64_fast(2.0) * kappa[i] * theta[i];
    assert!(
      feller_lhs >= sigma[i] * sigma[i],
      "asset {i} does not satisfy the Feller condition"
    );
  }
  for (idx, r) in rho.iter().enumerate() {
    assert!(r.is_finite(), "rho[{}] must be finite", idx);
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
    validate_params(&x0, &v0, &mu, &theta, &kappa, &sigma, &rho, n, t);
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
  type Sampler<'s>
    = Heston2DSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> Heston2DSampler<T> {
    let t_total = self.t.unwrap_or(T::one());
    let n_steps = self.n - 1;
    let dt = t_total / T::from_usize_(n_steps);
    let sqrt_dt = dt.sqrt();
    // Derived streams preserve the historical e1..e4 draw order for seeded reproducibility.
    let normals =
      std::array::from_fn(|_| SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed.derive()));
    Heston2DSampler {
      n: self.n,
      x0: [
        self.x0[0].unwrap_or(T::zero()),
        self.x0[1].unwrap_or(T::zero()),
      ],
      v0: [
        self.v0[0].expect("validated initial variance must be present"),
        self.v0[1].expect("validated initial variance must be present"),
      ],
      mu: self.mu,
      theta: self.theta,
      kappa: self.kappa,
      sigma: self.sigma,
      chol: self.chol,
      dt,
      use_sym: self.use_sym.unwrap_or(false),
      normals,
    }
  }
}

/// Reusable [`Heston2D`] sampling state: owns the four independent Gaussian
/// streams plus the precomputed Cholesky factor and step size so a Monte-Carlo
/// loop reuses all four output buffers and the RNG setup.
#[doc(hidden)]
pub struct Heston2DSampler<T: FloatExt> {
  n: usize,
  x0: [T; 2],
  v0: [T; 2],
  mu: [T; 2],
  theta: [T; 2],
  kappa: [T; 2],
  sigma: [T; 2],
  chol: [T; 10],
  dt: T,
  use_sym: bool,
  normals: [SimdNormal<T>; 4],
}

impl<T: FloatExt> Heston2DSampler<T> {
  fn fill_paths(&mut self, x1: &mut [T], v1: &mut [T], x2: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let n_steps = self.n - 1;
    let mut e1 = Array1::<T>::zeros(n_steps);
    let mut e2 = Array1::<T>::zeros(n_steps);
    let mut e3 = Array1::<T>::zeros(n_steps);
    let mut e4 = Array1::<T>::zeros(n_steps);
    let [n1, n2, n3, n4] = &self.normals;
    n1.fill_slice_fast(e1.as_slice_mut().expect("noise slice contiguous"));
    n2.fill_slice_fast(e2.as_slice_mut().expect("noise slice contiguous"));
    n3.fill_slice_fast(e3.as_slice_mut().expect("noise slice contiguous"));
    n4.fill_slice_fast(e4.as_slice_mut().expect("noise slice contiguous"));

    let [l11, l21, l22, l31, l32, l33, l41, l42, l43, l44] = self.chol;

    x1[0] = self.x0[0];
    v1[0] = self.v0[0];
    x2[0] = self.x0[1];
    v2[0] = self.v0[1];

    let dt = self.dt;
    let half = T::from_f64_fast(0.5);
    let use_sym = self.use_sym;
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
  }
}

impl<T: FloatExt> PathSampler<T> for Heston2DSampler<T> {
  type Output = [Array1<T>; 4];

  fn sample_into(&mut self, out: &mut [Array1<T>; 4]) {
    let [x1, v1, x2, v2] = out;
    self.fill_paths(
      x1.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
      v1.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
      x2.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
      v2.as_slice_mut()
        .expect("Heston2D output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 4] {
    let mut x1 = Array1::<T>::zeros(self.n);
    let mut v1 = Array1::<T>::zeros(self.n);
    let mut x2 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      x1.as_slice_mut().expect("contiguous"),
      v1.as_slice_mut().expect("contiguous"),
      x2.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
    [x1, v1, x2, v2]
  }
}

#[cfg(test)]
mod tests;
